"""跑评测：逐条调 triage.triage()，原始结果落 jsonl。

单独成文件而不是塞进 analyze：调模型慢且花钱，指标算错了可以免费重算，
模型没必要重跑。一次跑，反复读。

用法（项目根目录）：
    python eval/run_eval.py                # 跑 30 条样本，新开一个结果文件
    python eval/run_eval.py --limit 3      # 先跑 3 条试通
    python eval/run_eval.py --out a.jsonl  # 指定文件，已跑过的自动跳过（断点续跑）
    python eval/run_eval.py --all          # 跑全量 100 条
"""

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# 显式指定 .env：默认 load_dotenv() 是从当前工作目录找，
# 从别的目录跑这个脚本就会拿不到 key，报一个看不懂的 401
load_dotenv(BASE.parent / ".env")

import triage
from triage import clean_body

TICKETS = BASE.parent / "data" / "tickets.json"
SAMPLE = BASE / "sample.json"
RESULTS = BASE / "results"


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_one(ticket):
    text = ticket["title"] + "\n" + clean_body(ticket["body"])
    start = time.time()
    try:
        result = triage.triage(text)
    except Exception as error:
        # 单条失败不能让整轮中断：跑到第 60 条崩了，前面 59 条的调用费就白花了
        return {
            "number": ticket["number"],
            "error": f"{type(error).__name__}: {error}",
            "elapsed": round(time.time() - start, 1),
        }

    return {
        "number": ticket["number"],
        # 只留片段 id 和分数：结果文件是拿来看指标的，不是知识库的备份
        "hits": [{"id": hit["id"], "score": hit["score"]} for hit in result.get("hits", [])],
        "category": result.get("category"),
        "decision": result.get("decision"),
        "reply": result.get("reply", ""),
        "reason": result.get("reason", ""),
        "usage": result.get("usage", {}),
        "elapsed": round(time.time() - start, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="跑全量 100 条，默认只跑 30 条样本")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，调试用")
    parser.add_argument("--out", default="", help="结果文件名，已存在则续跑")
    args = parser.parse_args()

    tickets = json.loads(TICKETS.read_text(encoding="utf-8"))
    by_number = {ticket["number"]: ticket for ticket in tickets}

    if args.all:
        numbers = sorted(by_number)
    else:
        if not SAMPLE.exists():
            sys.exit("没有 eval/sample.json，先跑 python eval/label.py 抽样本")
        numbers = json.loads(SAMPLE.read_text(encoding="utf-8"))

    if args.limit:
        numbers = numbers[:args.limit]

    RESULTS.mkdir(exist_ok=True)
    if args.out:
        out_path = RESULTS / args.out
    else:
        out_path = RESULTS / f"run-{datetime.now():%Y%m%d-%H%M}.jsonl"

    done = {row["number"] for row in read_jsonl(out_path)}
    todo = [number for number in numbers if number not in done]

    print(f"共 {len(numbers)} 条，已完成 {len(done)}，本次跑 {len(todo)} 条")
    print(f"输出：{out_path}\n")

    for index, number in enumerate(todo, 1):
        row = run_one(by_number[number])
        # 每跑完一条立刻落盘，中途 Ctrl+C 也不丢
        with out_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

        if "error" in row:
            print(f"[{index}/{len(todo)}] #{number} 失败：{row['error']}")
        else:
            print(f"[{index}/{len(todo)}] #{number}  {row['decision']:<9} {row['elapsed']}s")

    print(f"\n完成，结果写入 {out_path}")
    print("下一步：python eval/analyze.py")


if __name__ == "__main__":
    main()
