"""工单分诊：Agent 版（v5）。

和 v4 的区别：检索从"代码写死的一步"变成"模型可以主动调用的工具"。
模型自己决定：查不查、查什么、查几次，查完再判断"够不够答 / 转人工"。

为什么这么做：v4 的固定流水线有两个死板的地方——
1. 不管用户问什么，都先检索一遍。寒暄（"你好"）也会被拿去检索，然后转人工。
2. 只用原话检索一次。口语问句对不上文档术语时，一次查不到就转人工，没有第二次机会。

改成 tool calling 后，这两个都交给模型自己决定，符合"得先看情况才知道该查什么"的 Agent 场景。

核心约束不变：模型只能用检索到的片段回答，片段里没有就说不知道。
"""

import json
import os

import kb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CLIENT = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)
MODEL = os.getenv("MODEL", "deepseek-flash")

# 会话表：thread_id -> [{"user": 用户说过的话, "assistant": 模型上一轮的原始输出}]
# 用模块级 dict 而不是上数据库：现在重启丢会话无所谓，等出现"丢了就真丢"的需求再换。
# 这和"几百条知识只读 json 就够"是同一个判断标准。
SESSIONS = {}

# 只留最近 2 轮。上限是给 v4 留的：v4 要拿 prompt token 当主指标，
# 这里无限累积的话，token 涨多少就和检索优化无关了，指标归因不回去。
MAX_TURNS = 2

# Agent 最多检索几次就强制收尾。没有这个上限，模型可能反复查、烧 token、
# 或者查不到就卡在"再查一次"里出不来。上限给 3：一次原始问句、一次换说法，
# 最多再来一次，足够，再多就是浪费。
MAX_RETRIEVE = 3

SYSTEM = """你是 FastGPT 的客服助理。用户会提交一条问题反馈，你要判断它是什么类型，并决定怎么处理。

你可以调用 retrieve 工具去检索知识库。规则：
- 需要知识库才能回答的问题，先调用 retrieve 检索（可以用你自己改写的话检索，不必照抄用户原话）
- 寒暄、自我介绍这类不需要知识库的，直接回答，不要检索
- 检索一次后如果结果不够，可以换个说法再检索，最多再查一两次

处理方式只有三种，最终必须选一种输出：
- answer：知识片段能直接回答问题，你写一段回复交给用户
- ask_more：片段可能相关，但缺关键信息（版本号、报错原文、部署方式），先反问用户
- escalate：片段里找不到依据，转人工处理，不要猜

硬规则：
1. 只能使用检索到的知识片段，不许用你自己的知识补充。片段没写的内容，就是不知道。
2. 选 answer 时，回复里必须用方括号标出依据的片段编号，例如 [chat#浏览器不支持语音输入]。
3. 选 ask_more 时，问题要具体到能行动（"你的 FastGPT 是哪个版本"），不许问"请提供更多信息"。
4. 分类只能是这四个之一：使用问题、报错、功能需求、其他。
5. 用户写得很短、看不出具体故障时，一律走 ask_more 或 escalate，不要硬猜。

最终只输出 JSON，不要任何解释文字，字段固定为：
{"category": "使用问题|报错|功能需求|其他",
 "decision": "answer|ask_more|escalate",
 "reply": "要发给用户的话",
 "reason": "为什么这么判，必须提到用到的片段编号，或说明为什么找不到依据"}"""


# retrieve 工具的声明。模型看到这个，就知道它能调 retrieve(query) 去查知识库。
TOOLS = [{
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "在 FastGPT 知识库中检索相关文档片段。参数 query 是检索语句，可以用改写后的关键词，不必照抄用户原话。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]


FOLLOWUP_RULE = """
6. 这是对话的后续轮次：用户这一句可能只是在补充你上一轮问的信息（比如只回了一个版本号）。
   把它和前面的描述合起来判断，不要因为这一句很短就再反问同样的问题。"""


def clean_body(body):
    """去掉 issue 模板自带的 checkbox。

    每个 issue 正文开头都是一堆「- [x] 我已确认...」，对判断没帮助，
    但会占掉半屏，让人看不到真正的问题描述。
    """
    lines = []
    for line in (body or "").splitlines():
        if line.strip().startswith("- ["):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def build_system(has_history):
    """单轮不加第 6 条规则。

    这一点不能省：要证明"多轮改造没损伤单轮效果"，前提是单轮路径的 prompt 和 v2 一字不差。
    多一条规则，30 条基线就不能直接对比了，你也说不清差异是改动带来的还是 prompt 带来的。
    """
    if not has_history:
        return SYSTEM
    return SYSTEM + FOLLOWUP_RULE


def build_user_message(ticket_text, hits):
    """把工单和检索结果拼成一条消息。

    片段放在 user 消息里而不是 system 里，因为片段是每次不同的输入，
    system 是固定规则，两者混在一起以后就没法单独换检索结果了。
    """
    if not hits:
        knowledge = "（知识库没有检索到任何相关内容）"
    else:
        lines = []
        for hit in hits:
            lines.append(f"[{hit['id']}]（相似度 {hit['score']}）\n{hit['text']}")
        knowledge = "\n\n".join(lines)

    return f"""用户反馈：
{ticket_text}

检索到的知识片段：
{knowledge}"""


def build_query(text, history):
    """多轮时把上一轮用户说的话拼进检索 query。

    用户第二轮往往只回一句「4.17.3」或者「报的是 413」，
    单拿这一句去 BM25 检索，一条都命中不了 —— 关键词全在上一轮里。
    只取最后一轮：更早的内容已经被这一轮继承过了，全拼进去只会稀释关键词。

    注意拼的是用户说的话，不含助手回复：助手回复里是反问话术，
    「请提供版本号」这种模板句会把检索带偏。
    """
    if not history:
        return text
    return history[-1]["user"] + "\n" + text


def remember(thread_id, user_text, assistant_text):
    """把这一轮存下来，只留最近 MAX_TURNS 轮。

    存的是模型原始输出而不是解析后的 reply：历史里放"它上一轮真说过的话"，
    不做改写，行为最可解释。反正下一轮它还是被要求输出 JSON。
    """
    history = SESSIONS.setdefault(thread_id, [])
    history.append({"user": user_text, "assistant": assistant_text})
    del history[:-MAX_TURNS]


def parse_result(content):
    """把模型返回的文本转成 dict。

    JSON 模式偶尔还是会带 ```json 包裹，所以先剥一次再解析。
    解析失败不抛异常，返回一个 escalate —— 判不出来的工单，转人工永远不会错。
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "category": "其他",
            "decision": "escalate",
            "reply": "",
            "reason": f"模型返回的不是合法 JSON，已转人工。原始内容前 100 字：{text[:100]}",
        }


def format_hits(hits):
    """把检索结果拼成给模型看的文本。

    一次 Agent 循环里可能检索多次，每次检索结果都作为一条 tool 消息喂回去，
    所以这里只负责"把 hits 列表变成文字"，不再拼用户反馈（那在循环开头拼一次）。
    """
    if not hits:
        return "（没有检索到相关内容）"
    lines = []
    for hit in hits:
        lines.append(f"[{hit['id']}]（相关度 {hit['score']}）\n{hit['text']}")
    return "\n\n".join(lines)


def triage(ticket_text, thread_id=None, top_k=5):
    """处理一条工单（Agent 循环版）。

    流程：先把工单交给模型，模型决定「调 retrieve 检索」还是「直接给最终判断」。
    - 调 retrieve：执行检索，把结果作为 tool 消息喂回去，让模型继续决定。
    - 给最终判断：解析 JSON，返回 answer/ask_more/escalate。

    最多检索 MAX_RETRIEVE 次，到上限强制让模型收尾，防止死循环和 token 失控。
    thread_id 语义和 v4 一致：None 是单轮，不读写会话表。
    """
    history = SESSIONS.get(thread_id, []) if thread_id else []

    # 首条 user 消息：把工单（带上多轮历史）交给模型，模型自己决定要不要检索
    query_text = build_query(ticket_text, history)
    messages = [{"role": "system", "content": build_system(bool(history))}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": f"用户反馈：\n{query_text}"})

    all_hits = []          # 累积所有检索到的片段，最后返回
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    result = None

    for _ in range(MAX_RETRIEVE + 1):  # 最多 MAX_RETRIEVE 次检索 + 1 次最终判断
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        total_usage["prompt_tokens"] += response.usage.prompt_tokens
        total_usage["completion_tokens"] += response.usage.completion_tokens

        msg = response.choices[0].message

        # 模型要求调工具：执行检索，把结果作为 tool 消息追加回去
        if msg.tool_calls:
            messages.append({"role": "assistant", "tool_calls": msg.tool_calls,
                             "content": msg.content})
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                hits = kb.search(args.get("query", ""), top_k=top_k)
                all_hits.extend(hits)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": format_hits(hits),
                })
            continue

        # 模型给了最终判断：解析 JSON
        content = msg.content or ""
        result = parse_result(content)
        break

    # 循环结束仍没结果（例如一直调工具到上限），兜底转人工
    if result is None:
        result = {"category": "其他", "decision": "escalate",
                  "reply": "", "reason": "达到检索次数上限仍未给出判断，转人工。"}

    result["hits"] = all_hits
    result["usage"] = total_usage

    # 控制会话生命周期（和 v4 一致）
    if thread_id:
        if result.get("decision") == "ask_more":
            remember(thread_id, ticket_text, json.dumps(result, ensure_ascii=False))
        else:
            SESSIONS.pop(thread_id, None)

    return result


if __name__ == "__main__":
    import pathlib

    tickets = json.loads(
        (pathlib.Path(__file__).parent / "data" / "tickets.json").read_text(encoding="utf-8"))

    # 挑一条正文较长的来试，太短的会直接走 ask_more，看不出回答质量
    sample = max(tickets, key=lambda t: len(t["body"]))
    print(f"工单 #{sample['number']}：{sample['title']}\n")
    print(f"正文：{sample['body'][:300]}\n")

    outcome = triage(sample["title"] + "\n" + sample["body"])
    print(f"分类：{outcome['category']}")
    print(f"决策：{outcome['decision']}")
    print(f"理由：{outcome['reason']}")
    print(f"回复：{outcome.get('reply', '')[:400]}")
    print(f"\ntoken：{outcome['usage']}")
