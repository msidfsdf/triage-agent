"""FastAPI 入口：一个页面 + 三个接口。

这一版不流式：整个分诊只调一次模型，大概 2~4 秒，直接返回体验更好。
等流程里出现"检索 → 判断 → 再检索"这种多步时，再改成 SSE 边跑边推。
"""

import json
import pathlib
import random

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import triage

BASE_DIR = pathlib.Path(__file__).parent
TICKETS = json.loads((BASE_DIR / "data" / "tickets.json").read_text(encoding="utf-8"))

app = FastAPI(title="triage-agent", description="FastGPT 客服工单分诊")


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=5, description="工单原文")
    # 不传就是单轮，和 v2 行为一致，方便 curl 直接试
    thread_id: str | None = Field(None, description="会话 id，多轮时由前端生成并沿用")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/triage")
def run_triage(request: TicketRequest):
    """给一条工单，返回分类、决策、回复和用到的知识片段。

    模型或网络出故障时不让用户看到 500，而是转人工——这本来就是分诊该有的行为：
    系统判断不了，就交给人，而不是把报错甩给用户。
    """
    try:
        return triage.triage(request.text, thread_id=request.thread_id)
    except Exception as error:
        return {
            "category": "其他",
            "decision": "escalate",
            "reply": "系统暂时无法判断这条反馈，已为您转接人工客服。",
            "reason": f"分诊服务异常，已转人工。错误信息：{type(error).__name__}: {error}",
            "hits": [],
        }


@app.get("/random_ticket")
def random_ticket():
    """随机给一条真实工单，方便快速试。

    以前这里只从「正文超过 150 字」的工单里抽，理由是想挑描述完整的。
    实测这个过滤是错的：能答的工单恰恰都是短工单（正文中位 20 出头，
    很多只有标题和一句话），它们一条都进不了池子 —— 池子 27 条和能答的 11 条零重合，
    于是页面上连点三十次也见不到一次「已解决」。
    现在改成全量等概率抽。
    """
    ticket = random.choice(TICKETS)
    return {
        "number": ticket["number"],
        "title": ticket["title"],
        "body": ticket["body"],
        "labels": ticket["labels"],
        # 和评测走同一个清洗：不然页面上跑的是带 "我已确认…" 模板的原文，
        # 评测跑的是清洗后的文本，两边结果对不上
        "text": f"{ticket['title']}\n{triage.clean_body(ticket['body'])}",
    }


@app.get("/stats")
def stats():
    """知识库规模，放在页面上让人一眼知道"它到底看了多少资料"。"""
    faq = json.loads((BASE_DIR / "data" / "faq.json").read_text(encoding="utf-8"))
    return {"knowledge_count": len(faq), "ticket_count": len(TICKETS)}
