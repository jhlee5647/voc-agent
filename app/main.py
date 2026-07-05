"""FastAPI + slack-bolt(FastAPI adapter) 조립 — POST /slack/events.

실행: uv run uvicorn app.main:api --port 3000  (+ ngrok http 3000)
signing secret 검증과 url_verification challenge는 bolt 기본 동작에 위임.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp

from app.report.scheduler import create_scheduler
from app.slack_handlers import register

load_dotenv()

bolt_app = AsyncApp(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)
register(bolt_app)


@asynccontextmanager
async def lifespan(_api):
    scheduler = create_scheduler()
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


api = FastAPI(lifespan=lifespan)
handler = AsyncSlackRequestHandler(bolt_app)


@api.post("/slack/events")
async def slack_events(request: Request):
    return await handler.handle(request)
