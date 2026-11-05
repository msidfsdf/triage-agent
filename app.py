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
        return triage.triage(request.text)
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
    """随机给一条真实工单，方便快速试。优先挑正文长一些的，短工单看不出回答质量。"""
    # 是从/data/tickets.json读取的工单
    longer = [t for t in TICKETS if len(t["body"]) > 150]
    ticket = random.choice(longer or TICKETS)
    return {
        "number": ticket["number"],
        "title": ticket["title"],
        "body": ticket["body"],
        "labels": ticket["labels"],
        "text": f"{ticket['title']}\n{ticket['body']}",
    }


@app.get("/stats")
def stats():
    """知识库规模，放在页面上让人一眼知道"它到底看了多少资料"。"""
    faq = json.loads((BASE_DIR / "data" / "faq.json").read_text(encoding="utf-8"))
    return {"knowledge_count": len(faq), "ticket_count": len(TICKETS)}
