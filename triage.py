"""工单分诊：先检索知识库，再让模型判断"能不能答、该怎么答"。

这一版是固定流水线（检索 → 一次判断），还不是 Agent。
区别在于"要不要查知识库"是代码写死的，不是模型决定的。等流程里出现
"得先看情况才知道该查什么"的时候，才需要换成 LangGraph + tool calling。

核心约束只有一条：模型只能用检索到的片段回答，片段里没有就说不知道。
这比在提示词里写"不要产生幻觉"有用得多——因为它把"有没有依据"变成了可检查的东西。
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

SYSTEM = """你是 FastGPT 的客服助理。用户会提交一条问题反馈，你要判断它是什么类型，并决定怎么处理。

处理方式只有三种，必须选一种：
- answer：给出的知识片段能直接回答问题，你写一段回复交给用户
- ask_more：片段可能相关，但缺关键信息（版本号、报错原文、部署方式），先反问用户
- escalate：片段里找不到依据，转人工处理，不要猜

硬规则：
1. 只能使用给出的知识片段，不许用你自己的知识补充。片段没写的内容，就是不知道。
2. 选 answer 时，回复里必须用方括号标出依据的片段编号，例如 [chat#浏览器不支持语音输入]。
3. 选 ask_more 时，问题要具体到能行动（"你的 FastGPT 是哪个版本"），不许问"请提供更多信息"。
4. 分类只能是这四个之一：使用问题、报错、功能需求、其他。
5. 用户写得很短、看不出具体故障时，一律走 ask_more 或 escalate，不要硬猜。

只输出 JSON，不要任何解释文字，字段固定为：
{"category": "使用问题|报错|功能需求|其他",
 "decision": "answer|ask_more|escalate",
 "reply": "要发给用户的话",
 "reason": "为什么这么判，必须提到用到的片段编号，或说明为什么找不到依据"}"""


FOLLOWUP_RULE = """
6. 这是对话的后续轮次：用户这一句可能只是在补充你上一轮问的信息（比如只回了一个版本号）。
   把它和前面的描述合起来判断，不要因为这一句很短就再反问同样的问题。"""


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


def triage(ticket_text, thread_id=None, top_k=5):
    """处理一条工单，返回分类、决策、回复和用到的知识片段。

    thread_id 为 None 就是单轮，不读也不写会话表 —— 这条路径的行为必须和 v2 一字不差。
    评测跑 30 条走的正是这条路，所以"多轮没把原来能答的搞坏"是可以直接比出来的：
    同一批工单、同一个 prompt、同一个 top_k，两次结果不一样就是改动引入的。

    top_k 取 5 不是拍脑袋：实测 BM25 在 741 条知识上 top1 只中 6/10，
    但 top5 能中 9/10。既然正确答案九成都在候选里，就多给模型几条让它自己挑，
    剩下的由"片段里没有就不许答"这条硬规则兜住。
    """
    history = SESSIONS.get(thread_id, []) if thread_id else []

    hits = kb.search(build_query(ticket_text, history), top_k=top_k)

    messages = [{"role": "system", "content": build_system(bool(history))}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": build_user_message(ticket_text, hits)})

    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        # 温度必须设 0：实测同一句"报错了怎么办"，默认温度下分类会在"报错"和"其他"之间跳。
        # 分类不稳定，后面的评测就不可复现，改 prompt 也看不出是变好还是随机波动。
        temperature=0,
        # 关掉思考模式：这一步是判断"片段够不够答"，不需要长推理，开着只会变慢变贵
        extra_body={"thinking": {"type": "disabled"}},
    )

    content = response.choices[0].message.content
    result = parse_result(content)
    result["hits"] = hits
    result["usage"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }

    # 控制会话生命周期
    if thread_id:
        if result.get("decision") == "ask_more":
            remember(thread_id, ticket_text, content)
        else:
            # 答完了或者转人工了，这个会话就结束了。顺手清掉，
            # 会话表就只会存"正在等用户回话"的那几个，不需要额外做淘汰。
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
