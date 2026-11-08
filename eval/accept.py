"""v3 多轮验收：全部可自动复现，不依赖人工标注。

指标口径（先说口径再给数字）：
  M1 单轮等价     v2 与 v3 在 30 条工单上构造的 messages 是否逐字节相同（不调模型，零成本）
  M2 会话状态机   会话创建 / 结束清理 / 轮数上限，是否正确
  M3 多轮一致率   「两轮对话的终态」是否等于「同一工单一次性说清」的终态
  M4 token 增量   第 2 轮相对「只发第 2 句」多花多少 prompt token（这是历史的价钱）
  M5 延迟         P90
  M6 引用有效性   decision=answer 的回复里，引用的片段编号是否都能在检索结果里找到

M3 为什么这么设计：多轮的价值是「把信息不足的首轮补到等效于一次性说清」。
如果补完之后系统给出的决策，和用户一开始就说清楚时一样，说明历史没有把判断带偏；
不一样，说明历史在污染。这个对照不需要人工标注，因为「说清楚的那一版」就是真实工单原文。

不测 Recall / Precision：那需要人工标注 labels.jsonl。
让模型给自己的回答打分等于让被测对象批卷子，指标会一路虚高，不做。
"""

import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import triage
from run_eval import clean_body

TICKETS = {t["number"]: t for t in json.loads((ROOT / "data" / "tickets.json").read_text(encoding="utf-8"))}
SAMPLE = json.loads((BASE / "sample.json").read_text(encoding="utf-8"))

# 多轮用例：工单是真的，正文是真的，只有「首轮那句」和「补充那句」是我从正文里摘的。
# first 取自标题（信息不足，该反问），followup 是正文里的关键事实，full 就是工单原文。
FIXTURES = [
    {"number": 640, "first": "connect EPERM",
     "followup": "网页版，在任何输入框输入文字都提示 connect EPERM"},
    {"number": 8012, "first": "fastgpt的cpu资源占用非常高",
     "followup": "工作流和知识库都没在用，cpu 还是持续很高，部署方式是 milvus + minio"},
    {"number": 7989, "first": "应用的数量实际值比显示值少",
     "followup": "个人信息里统计已创建应用时不含 Agent V2，实际是 a+b+c 但只显示 a+b"},
    {"number": 2206, "first": "4.8.8的几个bug",
     "followup": "版本号还显示 4.7.7；插件里点返回箭头一直加载；条件判断后指定回复手动输入提示校验错误"},
    {"number": 1893, "first": "高级编排工具调用不生效",
     "followup": "刚装的最新版本，调试时工具调用直接回复了信息，并没有调用相关的模块"},
    {"number": 7427, "first": "MCP服务名称自动生成不能修改",
     "followup": "MCP 服务名自动生成成 FastGPT-mcp-xxx，在其它 AI 平台调用时不好识别，希望支持改名"},
    # 下面两条是已知「文档里有答案」的工单。加进来是为了让终态里出现 answer，
    # 否则 6 条全 escalate/ask_more，M3 的"一致"就只是在验证"两个都是转人工"，没有信息量。
    {"number": 8113, "first": "循环节点里能用并行节点吗",
     "followup": "我在循环节点内部加并行节点加不上去，这是出于什么目的这样设计的"},
    {"number": 2942, "first": "回复被截断了",
     "followup": "4.8.11-fix 版本，AI 配置里回复上限设的 1950，长文本输入后最大响应 tokens 自动变成 200"},
]


def timed(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return result, round(time.time() - start, 1)


def ticket_text(number):
    return TICKETS[number]["title"] + "\n" + clean_body(TICKETS[number]["body"])


# ── M1 ─────────────────────────────────────────────────────────
FAKE_OUTPUT = '{"category":"其他","decision":"escalate","reply":"","reason":"stub"}'


def load_v2():
    """从 git HEAD 取出 v2 的 triage.py 来比。

    「单轮路径没变」不能靠读代码自己下结论，要拿真的旧代码跑一遍才知道。
    """
    source = subprocess.run(["git", "show", "HEAD:triage.py"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8").stdout
    path = pathlib.Path(tempfile.mkdtemp()) / "triage_v2.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("triage_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_messages(module, text):
    """把要发给模型的参数截下来，一条都不真发，所以这个指标零成本零噪声。"""
    seen = []

    def fake_create(**kwargs):
        seen.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=FAKE_OUTPUT))],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        )

    original = module.CLIENT
    module.CLIENT = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    try:
        module.triage(text)
    finally:
        module.CLIENT = original
    return seen[0]


def metric_single_turn_equivalence(v2):
    same = 0
    for number in SAMPLE:
        text = ticket_text(number)
        if capture_messages(v2, text) == capture_messages(triage, text):
            same += 1
    return same, len(SAMPLE)


# ── M2 ─────────────────────────────────────────────────────────
def metric_state_machine():
    checks = []

    triage.SESSIONS.clear()
    for i in range(3):
        triage.remember("cap", f"用户{i}", f"回复{i}")
    last = triage.SESSIONS["cap"][-1]["user"]
    checks.append(("最多留 2 轮", len(triage.SESSIONS["cap"]) == 2 and last == "用户2", f"实际 {len(triage.SESSIONS['cap'])} 轮"))

    triage.SESSIONS.clear()
    triage.triage("报错了，怎么办")
    checks.append(("单轮不建会话", triage.SESSIONS == {}, f"实际 {len(triage.SESSIONS)} 个会话"))

    first = triage.triage("报错了，怎么办", thread_id="smoke")
    kept = len(triage.SESSIONS.get("smoke", []))
    checks.append(("反问后保留会话", first["decision"] == "ask_more" and kept == 1,
                   f"决策 {first['decision']}，会话 {kept} 轮"))

    second = triage.triage("4.17.3，上传 word 报 413", thread_id="smoke")
    gone = "smoke" not in triage.SESSIONS
    checks.append(("非反问后清会话", second["decision"] != "ask_more" and gone,
                   f"决策 {second['decision']}，残留 {not gone}"))
    return checks


# ── M3 / M4 / M6 ───────────────────────────────────────────────
def cited_ids(reply):
    """只取像片段编号的引用（我们的 id 都带 #），避开 markdown 链接的方括号。"""
    return {x for x in re.findall(r"\[([^\[\]]+)\]", reply or "") if "#" in x}


def metric_multi_turn():
    rows = []
    for fixture in FIXTURES:
        full = ticket_text(fixture["number"])
        thread = f"fx-{fixture['number']}"

        d_first, t1 = timed(triage.triage, fixture["first"])
        d_full, t2 = timed(triage.triage, full)
        triage.triage(fixture["first"], thread_id=thread)
        r2, t3 = timed(triage.triage, fixture["followup"], thread_id=thread)
        solo, t4 = timed(triage.triage, fixture["followup"])

        rows.append({
            "number": fixture["number"],
            "first": d_first["decision"],
            "full": d_full["decision"],
            "multi": r2["decision"],
            "solo": solo["decision"],
            "match": r2["decision"] == d_full["decision"],
            "extra_tokens": r2["usage"]["prompt_tokens"] - solo["usage"]["prompt_tokens"],
            "elapsed": [t1, t2, t3, t4],
            "answer": r2 if r2["decision"] == "answer" else None,
        })
    return rows


def metric_citation(rows):
    """answer 的回复里引用到的片段编号，是否真的在它自己的检索结果里。

    只查「引用了不存在的东西」这一种错，查不了「引用了但结论不对」
    —— 后者必须人工看，不能自动。
    """
    total = 0
    bad = []
    for row in rows:
        if not row["answer"]:
            continue
        total += 1
        available = {hit["id"] for hit in row["answer"]["hits"]}
        wrong = cited_ids(row["answer"]["reply"]) - available
        if wrong:
            bad.append((row["number"], sorted(wrong)))
    return total, bad


def main():
    print("=" * 74)
    print("M1 单轮等价（不调模型）")
    v2 = load_v2()
    same, total = metric_single_turn_equivalence(v2)
    print(f"  messages 逐字节相同  {same}/{total}")
    print(f"  含义：v3 在 thread_id=None 时构造的 prompt 与 v2 完全一致，"
          f"{'单轮行为没有改动' if same == total else '存在差异，需逐条查'}")

    print("\n" + "=" * 74)
    print("M2 会话状态机")
    checks = metric_state_machine()
    for name, ok, detail in checks:
        print(f"  {'通过' if ok else '失败'}  {name:<12} {detail}")
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"  {passed}/{len(checks)} 通过")

    print("\n" + "=" * 74)
    print("M3 多轮一致率 / M4 token 净差")
    rows = metric_multi_turn()
    print(f"  {'工单':<8}{'首轮':<11}{'说清楚':<11}{'两轮终态':<11}{'一致':<6}{'token净差'}")
    for row in rows:
        mark = "是" if row["match"] else "否"
        print(f"  #{row['number']:<7}{row['first']:<11}{row['full']:<11}{row['multi']:<11}{mark:<6}"
              f"{row['extra_tokens']:+d}")
    matched = sum(1 for row in rows if row["match"])
    answered = sum(1 for row in rows if row["multi"] == "answer")
    print(f"  一致 {matched}/{len(rows)}；多轮终态里 answer {answered} 条")
    print(f"  第 2 轮比「只发第 2 句」平均净差 "
          f"{sum(r['extra_tokens'] for r in rows) / len(rows):+.0f} prompt token")
    print("  注意这个净差混了两个因素：加进来的历史（正）+ 检索命中的片段变了（可正可负）。"
          "\n  所以它不能单独当「历史的价钱」，要拆开得比对两次的命中片段。")

    print("\n" + "=" * 74)
    print("M5 延迟")
    all_elapsed = sorted(t for row in rows for t in row["elapsed"])
    print(f"  共 {len(all_elapsed)} 次调用，中位 {all_elapsed[len(all_elapsed) // 2]}s，"
          f"P90 {all_elapsed[int(len(all_elapsed) * 0.9)]}s")

    print("\n" + "=" * 74)
    print("M6 引用有效性")
    answered, bad = metric_citation(rows)
    if answered == 0:
        # 分母为 0 必须打 n/a，不能打「0 条错误」—— 那是没发出去，不是零错误
        print("  n/a（本轮 0 条走 answer，无法评估）")
    else:
        print(f"  answer {answered} 条，引用不存在的片段 {len(bad)} 条"
              + (f"：{bad}" if bad else ""))

    print("\n" + "=" * 74)
    print("未测：Recall / Precision / Resolution —— 需要人工标注 eval/labels.jsonl，本脚本不做。")


if __name__ == "__main__":
    main()
