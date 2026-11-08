"""生成"同源"评测集：问题从知识库片段反推，答案就是那条片段。

为什么需要它：原来的 30 条是真实 GitHub issue，79% 是缺陷报告，文档里根本没有
对应答案。用那种样本测，自动解决率永远是 0，而且分不清是"系统不行"还是"数据不行"。

这里问题和答案同源，所以"答对"可以自动判定——回复里引用了源片段就算对，
不需要人工标注。代价是问题比真实工单干净，数字偏乐观，报告里必须写清。

组成：
  20 条文档题（expect=answer，带 gt=源片段 id）
  10 条真实缺陷/需求（expect=escalate，答了就是错误外发）

用法：python eval/make_golden.py
"""

import json
import os
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
CLIENT = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"))
MODEL = os.getenv("MODEL", "deepseek-flash")

SEED = 20261108
N_DOC = 20
N_REAL = 10

GEN_PROMPT = """下面是一段产品文档。你要扮演一个没看过它的普通用户，去提一条工单。

文档：
{text}

要求：
1. 用大白话描述你遇到的现象或想做的事，像在群里求助
2. 除了 FastGPT，不许出现任何产品名、参数名、函数名、接口路径，也不许照抄文档的小标题
   （照抄了这道题就作废，这条最重要）
3. 只问这段文档能回答的问题
4. 30~100 字

只输出 JSON：{{"title": "一句话标题", "body": "正文"}}"""

# 实测文档能答的工单不能当负样本：#2942 被答且引用正确，
# 它本质是"回复为什么被截断"的咨询，不是缺陷报告。
BLACKLIST = {2942}


def pick_chunks(faq, n):
    """挑够长的片段：太短的（多半是参数表）反推不出像样的问题。"""
    pool = [c for c in faq if len(c["text"]) >= 200]
    random.Random(SEED).shuffle(pool)
    return pool[:n]


def pick_real_tickets(tickets, n):
    """挑陈述式的缺陷/需求：这类本来就该转人工，用来测错误外发。"""
    ask = re.compile(r"如何|怎么|怎样|为什么|能否|能不能|是否|请问|可以吗|\?")
    pool = [t for t in tickets
            if not ask.search(t["title"]) and len(t["title"]) > 8 and t["number"] not in BLACKLIST]
    random.Random(SEED).shuffle(pool)
    return pool[:n]


def gen_one(chunk):
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": GEN_PROMPT.format(text=chunk["text"][:800])}],
        response_format={"type": "json_object"},
        temperature=0.7,   # 生成要多样性；这里不是分类，不需要可复现
        extra_body={"thinking": {"type": "disabled"}},
    )
    data = json.loads(resp.choices[0].message.content)
    return data["title"].strip(), data["body"].strip()


def main():
    faq = json.loads((ROOT / "data/faq.json").read_text(encoding="utf-8"))
    tickets = json.loads((ROOT / "data/tickets.json").read_text(encoding="utf-8"))

    items = []
    print(f"1) 从知识库反推 {N_DOC} 条问题")
    for chunk in pick_chunks(faq, N_DOC):
        title, body = gen_one(chunk)
        items.append({"title": title, "body": body, "expect": "answer", "gt": chunk["id"]})
        print(f'   [{chunk["id"][:32]:<32}] {title[:40]}', flush=True)

    print(f"2) 取 {N_REAL} 条真实缺陷/需求工单")
    for t in pick_real_tickets(tickets, N_REAL):
        items.append({"number": t["number"], "title": t["title"],
                      "body": (t["body"] or "")[:1000], "expect": "escalate", "gt": None})
        print(f'   #{t["number"]:<6} {t["title"][:40]}', flush=True)

    out = ROOT / "eval/golden.json"
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {out}（{len(items)} 条）")


if __name__ == "__main__":
    main()
