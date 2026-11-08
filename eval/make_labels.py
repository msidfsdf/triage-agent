"""从构造好的 tickets.json 生成评测用的 ground truth（labels.jsonl）和样本（sample.json）。

为什么能自动生成：这一版 tickets 是用 make_tickets.py 构造的，每条工单构造时
就知道它「该走哪条路」，写进了字段里，不用再人工逐条标：

- answer_id 非空（咨询类）      -> expected = answer    该答
- labels 含「信息不足」         -> expected = ask_more  该反问
- 其余（bug / feature 缺陷需求）-> expected = escalate  该转人工

这和旧的 label.py（人工逐条标）是两种来源：
- 旧数据：工单来自 GitHub issue，没有标准答案，只能人看检索结果一条条标
- 新数据：工单是构造的，答案写死在 answer_id 上，标注可以程序化、可复现

用法：python eval/make_labels.py
"""

import json
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")

TICKETS = BASE.parent / "data" / "tickets.json"
LABELS = BASE / "labels.jsonl"
SAMPLE = BASE / "sample.json"
SAMPLE_SIZE = 0  # 0 = 全量


def expected_of(ticket):
    if ticket.get("answer_id"):
        return "answer"
    if "信息不足" in ticket.get("labels", []):
        return "ask_more"
    return "escalate"


def main():
    tickets = json.loads(TICKETS.read_text(encoding="utf-8"))

    labels = []
    for t in tickets:
        labels.append({
            "number": t["number"],
            "expected": expected_of(t),
            # 咨询类把 answer_id 记进 sub，核查时能直接对上文档
            "sub": t.get("answer_id") or None,
            "note": t["labels"][0] if t.get("labels") else "",
        })

    LABELS.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in labels) + "\n",
        encoding="utf-8",
    )

    # 样本：默认全量（SAMPLE_SIZE 设成 0 表示全量）。
    # 300 条全量跑，没有抽样偶然性，指标直接是总体真值。
    # 想临时只跑一部分调试，把 SAMPLE_SIZE 改成具体数字（按类别等比例抽）。
    if SAMPLE_SIZE == 0:
        picked = sorted(t["number"] for t in tickets)
    else:
        import random
        rng = random.Random(20261108)
        by_expected = {"answer": [], "ask_more": [], "escalate": []}
        for t in tickets:
            by_expected[expected_of(t)].append(t["number"])
        # 按类别等比例抽：answer 占 60%、ask_more 20%、escalate 20%
        want_answer = round(SAMPLE_SIZE * 0.6)
        want_ask = round(SAMPLE_SIZE * 0.2)
        want_esc = SAMPLE_SIZE - want_answer - want_ask
        picked = []
        picked.extend(rng.sample(by_expected["answer"], want_answer))
        picked.extend(rng.sample(by_expected["ask_more"], want_ask))
        picked.extend(rng.sample(by_expected["escalate"], want_esc))
        picked.sort()

    SAMPLE.write_text(json.dumps(picked, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已生成 {LABELS}（{len(labels)} 条）")
    print(f"已生成 {SAMPLE}（{len(picked)} 条样本）")
    print(f"  其中 answer {sum(1 for p in picked if expected_of({t['number']: t for t in tickets}[p]) == 'answer')} "
          f"/ ask_more {sum(1 for p in picked if expected_of({t['number']: t for t in tickets}[p]) == 'ask_more')} "
          f"/ escalate {sum(1 for p in picked if expected_of({t['number']: t for t in tickets}[p]) == 'escalate')}")


if __name__ == "__main__":
    main()
