"""跑同源评测集并算指标。全部自动判定，不需要人工标注。

判定规则（写死，别改）：
  文档题：decision==answer 且回复里引用了 gt 片段 -> 答对(tp)
  真实题：decision==answer -> 错误外发（这是安全底线，期望是 0）

指标口径：
  Precision = tp / 所有答出去的     发出去的多准
  Recall    = tp / 20 文档题        该答的捞回多少
  Resolution= tp / 30               替人挡掉多少（严格口径：答错不算）
  错误外发率 = 真实题被答的 / 所有答出去的

用法：python eval/golden_eval.py [--limit N]
"""

import argparse
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import triage

GOLDEN = ROOT / "eval/golden.json"
OUT = ROOT / "eval/results/golden.jsonl"


def cited_ids(reply):
    """回复里用方括号标的片段编号，例如 [chat#猜你想问]。"""
    return re.findall(r"\[([^\[\]]+)\]", reply or "")


def run_once(item):
    text = item["title"] + "\n" + item["body"]
    start = time.time()
    r = triage.triage(text)
    return {
        "expect": item["expect"],
        "gt": item["gt"],
        "title": item["title"],
        "decision": r["decision"],
        "category": r.get("category", ""),
        "reply": r.get("reply", ""),
        "reason": r.get("reason", ""),
        "cited": cited_ids(r.get("reply", "")),
        "hit_ids": [h["id"] for h in r["hits"]],
        "elapsed": round(time.time() - start, 2),
        "prompt_tokens": r["usage"]["prompt_tokens"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，调试用")
    args = parser.parse_args()

    items = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if args.limit:
        items = items[:args.limit]

    rows = []
    for i, item in enumerate(items, 1):
        row = run_once(item)
        rows.append(row)
        if row["expect"] == "answer":
            ok = "对" if (row["decision"] == "answer" and row["gt"] in row["cited"]) else "错"
            print(f'[{i}/{len(items)}] 文档题 {ok}  {row["decision"]:<10} {row["title"][:34]}', flush=True)
        else:
            bad = "  <-- 错误外发" if row["decision"] == "answer" else ""
            print(f'[{i}/{len(items)}] 真实题 {row["decision"]:<10} {row["title"][:34]}{bad}', flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    doc = [r for r in rows if r["expect"] == "answer"]
    real = [r for r in rows if r["expect"] == "escalate"]
    emitted = [r for r in rows if r["decision"] == "answer"]
    tp = [r for r in doc if r["decision"] == "answer" and r["gt"] in r["cited"]]
    leaked = [r for r in real if r["decision"] == "answer"]
    top1 = [r for r in doc if r["gt"] in r["hit_ids"][:1]]
    top5 = [r for r in doc if r["gt"] in r["hit_ids"]]

    def frac(a, b):
        return f"{a}/{b} = {a / b * 100:.0f}%" if b else f"{a}/0 = n/a"

    print("\n" + "=" * 62)
    print(f"检索   top1 {frac(len(top1), len(doc))}   top5 {frac(len(top5), len(doc))}")
    print(f"Precision 精确率   {frac(len(tp), len(emitted))}   答对/发出")
    print(f"Recall 召回率      {frac(len(tp), len(doc))}   答对/该答")
    print(f"Resolution 解决率  {frac(len(tp), len(rows))}   答对/全部")
    print(f"错误外发           {frac(len(leaked), len(real))}   真实缺陷被当成能答")
    lat = sorted(r["elapsed"] for r in rows)
    print(f"延迟  中位 {lat[len(lat) // 2]}s   P90 {lat[int(len(lat) * 0.9)]}s")
    print(f"prompt token  平均 {sum(r['prompt_tokens'] for r in rows) // len(rows)}")


if __name__ == "__main__":
    main()
