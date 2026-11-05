"""读 results + labels，算指标。不调模型，随便重跑。

正类（positive）= 被正确自动解答的工单：该答 + 答了 + 答对。
三个精度指标分子相同，只有分母不同：

- Precision 精确率 = 答对 / 发出      发出去的东西有多准（分母是模型答的）
- Recall    召回率 = 答对 / 该答      该答的捞回来多少（分母是人工认为该答的）
- Resolution 解决率 = 答对 / 全部      业务上替人挡掉了多少工单

为什么不用 F1：发错一条要人工二次处理还赔信任，漏答一条只是多一个人工工单，
两类错误代价不对称，等权平均会把它们算成一样。

用法（项目根目录）：
    python eval/analyze.py            # 读最新一次结果，出报告
    python eval/analyze.py --judge    # 逐条核对 answer，结果写 verdicts.jsonl
    python eval/analyze.py --file run-20261105-1910.jsonl
"""

import argparse
import json
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")

LABELS = BASE / "labels.jsonl"
VERDICTS = BASE / "verdicts.jsonl"
RESULTS = BASE / "results"
TICKETS = BASE.parent / "data" / "tickets.json"
FAQ = BASE.parent / "data" / "faq.json"

ACTIONS = ["answer", "ask_more", "escalate"]


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(part, whole):
    # 分母为 0 必须打 n/a，不能打 0.0%：那会让人以为"零错误"，实际是"没有样本"
    return "  n/a " if whole == 0 else f"{part / whole * 100:5.1f}%"


def display_width(text):
    # 一个汉字在终端占两格，str.ljust 按字符数算，混着中英文就会对不齐
    return sum(2 if ord(char) > 127 else 1 for char in text)


def judge(answer_rows, verdicts):
    """逐条问"这条答对了吗"。

    必须人工判：让模型自评等于让被测对象给自己打分，指标会一路虚高。
    判的标准是"回复里每个关键结论，能不能在下面贴出的片段原文里找到"。
    """
    todo = [row for row in answer_rows if row["number"] not in verdicts]
    if not todo:
        print("所有自动回答都已核对。")
        return verdicts

    tickets = {t["number"]: t for t in json.loads(TICKETS.read_text(encoding="utf-8"))}
    faq = {chunk["id"]: chunk for chunk in json.loads(FAQ.read_text(encoding="utf-8"))}

    print(f"\n有 {len(todo)} 条自动回答待核对。判断标准：回复的结论能否在片段原文里找到。\n")

    for row in todo:
        ticket = tickets.get(row["number"], {})
        print("─" * 72)
        print(f"#{row['number']}  {ticket.get('title', '')}")
        for hit in row["hits"][:3]:
            chunk = faq.get(hit["id"], {})
            print(f"\n  [片段] {hit['id']}")
            print(f"  {chunk.get('text', '（找不到这个片段）')[:400]}")
        print(f"\n  [模型回复] {row['reply'][:500]}")

        line = input("\ny=答对  n=答错  k=跳过 > ").strip().lower()
        if line not in ("y", "n"):
            continue

        record = {"number": row["number"], "ok": line == "y"}
        with VERDICTS.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        verdicts[row["number"]] = record["ok"]

    return verdicts

# 数数
def compute(rows, labels, verdicts):
    labeled = [row for row in rows if row["number"] in labels and "error" not in row]
    answers = [row for row in labeled if row.get("decision") == "answer"]

    should_answer = [row for row in labeled if labels[row["number"]]["expected"] == "answer"]

    # 只统计已核对的回答：没核对的不算对，也不算错，单独报出来
    judged = [row for row in answers if row["number"] in verdicts]
    correct = [row for row in judged if verdicts[row["number"]]]

    # 正类 = 被正确自动解答的工单：该答 + 答了 + 答对
    # 三个指标分子相同、分母不同，所以不会出现"换个名字数字就变好看"
    true_positive = [row for row in correct if labels[row["number"]]["expected"] == "answer"]

    return {
        "n": len(labeled),
        "answered": len(answers),
        "judged": len(judged),
        "tp": len(true_positive),
        "should": len(should_answer),
    }

def percentile(values, q):
    """取分位数。一共才 30 条，用最简单的索引法，不追求插值精度。"""
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * q))
    return ordered[index]


# 输出报告指标（除法计算）
def report(score, rows, labels, result_path):
    print(f"\n结果文件：{result_path.name}")
    print(f"样本 {score['n']} 条，自动回答 {score['answered']} 条（已核对 {score['judged']} 条）")
    print("正类 = 被正确自动解答的工单（该答 + 答了 + 答对）")

    if score["judged"] < score["answered"]:
        print(f"⚠ 还有 {score['answered'] - score['judged']} 条回答没核对，先跑 --judge")

    print("\n核心指标")
    for name, part, whole, note in [
        ("Precision 精确率", score["tp"], score["judged"], "答对 / 发出：发出去的多准"),
        ("Recall 召回率", score["tp"], score["should"], "答对 / 该答：该答的捞回多少"),
        ("Resolution 解决率", score["tp"], score["n"], "答对 / 全部：替人挡掉多少"),
    ]:
        pad = " " * max(0, 22 - display_width(name))
        print(f"  {name}{pad}{part:>3}/{whole:<3} {pct(part, whole)}   {note}")

    print("\n工程指标")
    elapsed = [row["elapsed"] for row in rows if "elapsed" in row]
    if elapsed:
        print(f"  延迟      平均 {sum(elapsed) / len(elapsed):5.1f}s   P90 {percentile(elapsed, 0.9):5.1f}s")
    prompts = [row.get("usage", {}).get("prompt_tokens", 0) for row in rows if row.get("usage")]
    if prompts:
        print(f"  Prompt    平均 {sum(prompts) / len(prompts):5.0f} tokens/条")

    print("\n决策分布（模型）")
    for action in ACTIONS:
        count = len([row for row in rows if row.get("decision") == action and row["number"] in labels])
        print(f"  {action:<10} {count:>3}  {pct(count, score['n'])}")

    print("\n混淆矩阵（行=人工标注，列=模型）")
    matrix = {expected: {actual: 0 for actual in ACTIONS} for expected in ACTIONS}
    for row in rows:
        if row["number"] not in labels or row.get("decision") not in ACTIONS:
            continue
        expected = labels[row["number"]]["expected"]
        matrix[expected][row["decision"]] += 1

    print(" " * 12 + "".join(f"{action:>11}" for action in ACTIONS))
    for expected in ACTIONS:
        cells = "".join(f"{matrix[expected][actual]:>11}" for actual in ACTIONS)
        print(f"{expected:>11} " + cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="逐条核对自动回答")
    parser.add_argument("--file", default="", help="指定结果文件，默认取最新")
    args = parser.parse_args()

    labels = {row["number"]: row for row in read_jsonl(LABELS)}
    if not labels:
        sys.exit("没有 eval/labels.jsonl，先跑 python eval/label.py 标注")

    if args.file:
        result_path = RESULTS / args.file
    else:
        files = sorted(RESULTS.glob("*.jsonl"))
        if not files:
            sys.exit("results/ 里没有结果，先跑 python eval/run_eval.py")
        result_path = files[-1]

    rows = read_jsonl(result_path)
    verdicts = {row["number"]: row["ok"] for row in read_jsonl(VERDICTS)}

    if args.judge:
        answers = [row for row in rows if row.get("decision") == "answer"]
        verdicts = judge(answers, verdicts)

    report(compute(rows, labels, verdicts), rows, labels, result_path)


if __name__ == "__main__":
    main()
