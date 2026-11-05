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


def triage(ticket_text, top_k=5):
    """处理一条工单，返回分类、决策、回复和用到的知识片段。

    top_k 取 5 不是拍脑袋：实测 BM25 在 741 条知识上 top1 只中 6/10，
    但 top5 能中 9/10。既然正确答案九成都在候选里，就多给模型几条让它自己挑，
    剩下的由"片段里没有就不许答"这条硬规则兜住。
    """
    hits = kb.search(ticket_text, top_k=top_k)

    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_message(ticket_text, hits)},
        ],
        response_format={"type": "json_object"},
        # 温度必须设 0：实测同一句"报错了怎么办"，默认温度下分类会在"报错"和"其他"之间跳。
        # 分类不稳定，后面的评测就不可复现，改 prompt 也看不出是变好还是随机波动。
        temperature=0,
        # 关掉思考模式：这一步是判断"片段够不够答"，不需要长推理，开着只会变慢变贵
        extra_body={"thinking": {"type": "disabled"}},
    )

    result = parse_result(response.choices[0].message.content)
    result["hits"] = hits
    result["usage"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
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
