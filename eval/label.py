"""逐条标注 30 条工单：标出"它本该走哪条路"。

这是评测集的 ground truth。标的时候不看模型输出，否则会被它的判断锚定，
标出来的东西就只剩"确认模型是对的"，不再是独立标准。

判定树（标之前先读一遍，否则标到第 15 条会开始犹豫）：
1. 看不出用户遇到什么具体问题 -> q（该反问）
2. 看得出，是产品缺陷或功能缺失 -> e b / e f（该转人工）
3. 看得出，是"怎么用"的问题 -> 先按 s 看检索，库里有能直接回答的片段 -> a
   没有 -> e c（覆盖不足）

用法：在项目根目录跑 python eval/label.py
"""

import json
import pathlib
import random
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
sys.stdout.reconfigure(encoding="utf-8")

import kb

TICKETS = BASE.parent / "data" / "tickets.json"
SAMPLE = BASE / "sample.json"
LABELS = BASE / "labels.jsonl"

# 换这个数字就是换一批样本。换过之后，旧基线的数字不可比，要重跑
SEED = 20261105
SAMPLE_SIZE = 30

ESCALATE_SUB = {"b": "bug", "f": "feature", "c": "coverage"}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_sample(tickets):
    """抽 30 条并立刻落盘。

    落盘而不是每次现抽：以后 tickets.json 顺序变了、或者重跑 ingest 换了数据，
    评测集也不能变，否则两版之间的差异分不清是代码改的还是样本换的。
    """
    if SAMPLE.exists():
        return json.loads(SAMPLE.read_text(encoding="utf-8"))

    picked = random.Random(SEED).sample(tickets, SAMPLE_SIZE)
    numbers = sorted(ticket["number"] for ticket in picked)
    SAMPLE.write_text(json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已抽取 {SAMPLE_SIZE} 条样本 -> {SAMPLE}")
    return numbers


def clean_body(body):
    """去掉 issue 模板自带的 checkbox。

    每个 issue 正文开头都是一堆「- [x] 我已确认...」，对判断没帮助，
    但会占掉半屏，让人看不到真正的问题描述。
    """
    lines = []
    for line in body.splitlines():
        if line.strip().startswith("- ["):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def show(ticket, index, total):
    labels = ",".join(ticket["labels"]) or "-"
    print(f"\n{'─' * 72}")
    print(f"[{index}/{total}] #{ticket['number']}  [{labels}]  评论 {ticket['comments']}  {ticket['state_reason']}")
    print(f"{'─' * 72}")
    print(ticket["title"])
    body = clean_body(ticket["body"])
    print()
    print(body[:700] if body else "（正文为空）")


def show_hits(ticket):
    """标 a 之前必须先看这个：库里没有能答的片段，就不该标 answer。"""
    text = ticket["title"] + "\n" + ticket["body"]
    hits = kb.search(text, top_k=3)
    if not hits:
        print("\n  检索：无命中")
        return
    print()
    for hit in hits:
        print(f"  {hit['score']:>7.2f}  {hit['id']}")
        print(f"           {hit['text'][:110]}...")


def parse(line):
    """解析一行输入，返回 (expected, sub, note)。看不懂就返回 None，让主循环重问。"""
    parts = line.strip().split(maxsplit=1)
    if not parts:
        return None

    command = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if command == "a":
        return ("answer", None, rest)
    if command == "q":
        return ("ask_more", None, rest)
    if command == "e":
        sub_key = rest.split(maxsplit=1)[0].lower() if rest else ""
        if sub_key not in ESCALATE_SUB:
            return None
        note = rest[len(sub_key):].strip()
        return ("escalate", ESCALATE_SUB[sub_key], note)
    return None


def main():
    tickets = json.loads(TICKETS.read_text(encoding="utf-8"))
    by_number = {ticket["number"]: ticket for ticket in tickets}
    numbers = ensure_sample(tickets)

    done = {row["number"] for row in read_jsonl(LABELS)}
    if done:
        print(f"已标 {len(done)} 条，继续剩下的")

    for index, number in enumerate(numbers, 1):
        if number in done:
            continue

        ticket = by_number[number]
        show(ticket, index, len(numbers))

        while True:
            line = input(
                "\na=该答  q=该反问  e b/f/c=转人工(缺陷/需求/覆盖不足)  s=看检索  k=跳过"
                "\n> "
            )
            command = line.strip().lower()
            if command == "s":
                show_hits(ticket)
                continue
            if command == "k":
                break

            parsed = parse(line)
            if parsed is None:
                print("看不懂。示例：a    q 只有标题没正文    e b")
                continue

            expected, sub, note = parsed
            record = {"number": number, "expected": expected, "sub": sub, "note": note}
            with LABELS.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  -> {expected}" + (f" / {sub}" if sub else ""))
            done.add(number)
            break

    print(f"\n已标 {len(done)}/{len(numbers)} 条 -> {LABELS}")
    if len(done) < len(numbers):
        print("还有没标的，再跑一次会从断点继续。")


if __name__ == "__main__":
    main()
