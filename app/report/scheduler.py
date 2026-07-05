"""주간 리포트 스케줄러 — 매주 월 09:00 KST에 생성 → Slack 채널 포스팅."""

import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.report.weekly import build_weekly_report


def run_weekly_job(client, channel: str, *, conn=None, base_date=None, embedder=None) -> None:
    """리포트 생성 → 채널 포스팅. conn 미지정 시 새 커넥션 (스케줄 실행 경로)."""
    if conn is not None:
        text = build_weekly_report(conn, base_date=base_date, embedder=embedder)
    else:
        from app.db import get_connection

        with get_connection() as new_conn:
            text = build_weekly_report(new_conn, base_date=base_date, embedder=embedder)
    client.chat_postMessage(channel=channel, text=text)


def _default_job() -> None:
    from slack_sdk import WebClient

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    run_weekly_job(client, os.environ["SLACK_REPORT_CHANNEL"])


def create_scheduler(job=_default_job) -> AsyncIOScheduler:
    """월 09:00 Asia/Seoul cron 잡이 등록된 스케줄러 반환 (start는 호출자 몫)."""
    scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    scheduler.add_job(
        job,
        CronTrigger(day_of_week="mon", hour=9, minute=0, timezone="Asia/Seoul"),
        id="weekly_report",
    )
    return scheduler
