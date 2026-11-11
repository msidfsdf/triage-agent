"""只测检索、不调模型：对 180 条咨询工单，看答案文档有没有被召回。

为什么单独立这个脚本：调检索方案要跑几十次对照，每次都跑 300 条模型评测
又慢又花钱，还会把"检索变好"和"模型波动"混在一起。检索这一步用 answer_id
就能秒级判定，先在这里把方案调对，再跑一次全量模型评测做最终验收。

用法（项目根目录）：
    python eval/retrieval_eval.py            # 跑混合检索（默认）
    python eval/retrieval_eval.py --mode bm25
    python eval/retrieval_eval.py --mode vector
    python eval/retrieval_eval.py --mode hybrid
    python eval/retrieval_eval.py --k 5      # 只看 recall@5
"""

import argparse
import json
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
sys.stdout.reconfigure(encoding="utf-8")

import kb

LABELS = BASE / "labels.jsonl"
TICKETS = BASE.parent / "data" / "tickets.json"


def load():
    labels = {json.loads(line)["number"]: json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()}
    tickets = {t["number"]: t for t in json.loads(TICKETS.read_text(encoding="utf-8"))}
    return labels, tickets


def query_text(ticket):
    return (ticket["title"] + "\n" + ticket["body"]).strip()


def run(mode, k_list):
    labels, tickets = load()
    consult = [n for n, r in labels.items() if r["expected"] == "answer"]
    hits_map = {}
    for n in consult:
        if mode == "bm25":
            idxs = kb.bm25_rank(query_text(tickets[n]), n=max(k_list))
            ids = [kb.CHUNKS[i]["id"] for i in idxs]
        elif mode == "vector":
            idxs = kb.vector_rank(query_text(tickets[n]), n=max(k_list))
            ids = [kb.CHUNKS[i]["id"] for i in idxs]
        else:  # hybrid
            ids = [h["id"] for h in kb.search(query_text(tickets[n]), top_k=max(k_list))]
        hits_map[n] = ids

    print(f"模式：{mode}   咨询工单 {len(consult)} 条\n")
    for k in k_list:
        hit = sum(1 for n in consult if labels[n]["sub"] in hits_map[n][:k])
        print(f"  recall@{k:<2} {hit}/{len(consult)} = {hit / len(consult) * 100:.1f}%")

    # MRR：正确答案排在候选里第几位的倒数的平均，衡量排序质量（越靠前越好）
    mrr = 0.0
    for n in consult:
        ids = hits_map[n]
        if labels[n]["sub"] in ids:
            mrr += 1 / (ids.index(labels[n]["sub"]) + 1)
    print(f"  MRR       {mrr / len(consult):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    parser.add_argument("--k", default="1,3,5,10", help="逗号分隔的 k 列表")
    args = parser.parse_args()

    run(args.mode, [int(x) for x in args.k.split(",")])
