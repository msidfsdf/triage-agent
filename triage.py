"""工单分诊：Agent 版（v6，LangGraph 状态图）。

和 v5 的区别：v5 用 for 循环 + if/break 手写 ReAct 循环，控制流藏在代码里；
v6 用 LangGraph 把「模型判断 → 要不要检索 → 检索 → 再判断」画成图上的节点和边，
循环是图上的一条环，控制流一眼可见。

为什么把 20 行循环搬成一张图（不是为了提指标，指标一个都不会变）：
1. 状态机显式化：Agent 每一步读什么状态、往哪走，从注释变成图的结构。
2. 好扩展：以后加「自动建单」「查订单」这类工具，就是往图里加节点+边，
   不用再改手写循环的控制流。
3. 天然支持中断/恢复/回溯，手写 while 循环里做 checkpoint 很难。

LangGraph 只负责「编排」，调模型/检索/解析这些业务逻辑还是 v5 的函数。
这就是编排框架的定位：不替代业务逻辑，只替代手写的「下一步往哪走」。

核心约束不变：模型只能用检索到的片段回答，片段里没有就说不知道。
"""

import json
import os
from typing import TypedDict

import kb
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
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


# ── 状态：图上所有节点共享、传递的数据 ─────────────────────────────
# LangGraph 的核心：一张图在「状态」上流转。每个节点读状态、改状态，
# 边决定改完后往哪走。状态显式声明，这就是「状态机显式化」的落点。
class AgentState(TypedDict):
    messages: list          # 发给模型的完整消息历史（system + 历史 + 用户 + tool 结果）
    retrieve_count: int     # 已检索几次，用来判「到上限就强制收尾」
    all_hits: list          # 累积所有检索到的片段，最终返回
    total_usage: dict       # 累积 token
    result: dict            # 最终解析出的 answer/ask_more/escalate
    ticket_text: str        # 原始工单（收尾后写会话用）
    thread_id: str | None   # 会话 id


def init_state(ticket_text, thread_id):
    """构造初始状态。对应 v5 里 triage() 开头拼 messages 那段。"""
    history = SESSIONS.get(thread_id, []) if thread_id else []
    query_text = build_query(ticket_text, history)
    messages = [{"role": "system", "content": build_system(bool(history))}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": f"用户反馈：\n{query_text}"})
    return AgentState(
        messages=messages,
        retrieve_count=0,
        all_hits=[],
        total_usage={"prompt_tokens": 0, "completion_tokens": 0},
        result={},
        ticket_text=ticket_text,
        thread_id=thread_id,
    )


# ── 节点 1：让模型决定下一步 ──────────────────────────────────────
# 对应 v5 循环里「调一次模型」那步。节点只负责调模型、更新状态，
# 「接下来去哪」由下面的 route 路由函数判断，节点本身不管。
def call_model(state: AgentState) -> AgentState:
    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=state["messages"],
        tools=TOOLS,
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    state["total_usage"]["prompt_tokens"] += response.usage.prompt_tokens
    state["total_usage"]["completion_tokens"] += response.usage.completion_tokens

    msg = response.choices[0].message
    state["messages"].append(msg.model_dump())
    if not msg.tool_calls:
        state["result"] = parse_result(msg.content or "")
    return state


# ── 节点 2：执行检索 ──────────────────────────────────────────────
# 对应 v5 里「模型要求调工具时执行检索」那段。要不要检索的判断挪到了
# 边（route）上，节点只干「检索」这一件事，职责单一。
def do_retrieve(state: AgentState, top_k: int) -> AgentState:
    msg = state["messages"][-1]
    for call in msg["tool_calls"]:
        args = json.loads(call["function"]["arguments"])
        hits = kb.search(args.get("query", ""), top_k=top_k)
        state["all_hits"].extend(hits)
        state["messages"].append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": format_hits(hits),
        })
    state["retrieve_count"] += 1
    return state


# ── 路由：模型这轮到底想干什么 ────────────────────────────────────
# 这是 v5 里 `if msg.tool_calls ... else ...` 那个分支，从代码搬到了「边」上。
# 每个节点跑完，问一次「下一步去哪」，返回节点名。
def route(state: AgentState):
    msg = state["messages"][-1]
    if msg.get("tool_calls") and state["retrieve_count"] < MAX_RETRIEVE:
        return "retrieve"          # 模型要检索，且没到上限 → 走检索节点
    return "finalize"              # 否则收尾（给结论了，或到上限了）


# ── 节点 3：收尾 ──────────────────────────────────────────────────
# 对应 v5 循环结束后「兜底转人工 + 挂 hits/usage + 会话生命周期」那段。
def finalize(state: AgentState) -> AgentState:
    result = state.get("result") or {
        "category": "其他", "decision": "escalate",
        "reply": "", "reason": "达到检索次数上限仍未给出判断，转人工。",
    }
    result["hits"] = state["all_hits"]
    result["usage"] = state["total_usage"]

    if state["thread_id"]:
        if result.get("decision") == "ask_more":
            remember(state["thread_id"], state["ticket_text"],
                     json.dumps(result, ensure_ascii=False))
        else:
            SESSIONS.pop(state["thread_id"], None)

    state["result"] = result
    return state


# ── 建图 ──────────────────────────────────────────────────────────
# 把节点和边拼成一张图，结构一目了然：
#   START → call_model ──route判断──→ retrieve ──→ call_model（环，最多绕 MAX_RETRIEVE 次）
#                       └──────────→ finalize → END
def build_graph(top_k=5):
    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("retrieve", lambda s: do_retrieve(s, top_k))  # 闭包把 top_k 绑进检索节点
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", route, {"retrieve": "retrieve", "finalize": "finalize"})
    graph.add_edge("retrieve", "call_model")   # 检索完回到模型，形成环
    graph.add_edge("finalize", END)
    return graph


def triage(ticket_text, thread_id=None, top_k=5):
    """处理一条工单（LangGraph 状态图版）。

    签名和 v5 一致，评测脚本和 app.py 不用改。流程和 v5 完全相同：
    模型决定「调 retrieve 检索」还是「直接给最终判断」，最多 MAX_RETRIEVE 次，
    到上限强制收尾。thread_id 语义不变：None 是单轮，不读写会话表。
    """
    graph = build_graph(top_k).compile()
    final = graph.invoke(init_state(ticket_text, thread_id))
    return final["result"]


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
