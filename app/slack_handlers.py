"""Slack app_mention 핸들러 — 즉시 ACK 후 태스크로 에이전트 실행, 스레드 답변.

3초 ACK 규칙: 핸들러는 LLM 작업을 asyncio 태스크로 분리하고 즉시 반환한다.
미준수 시 Slack이 이벤트를 재전송해 중복 답변이 발생한다.
"""

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60.0
TIMEOUT_TEXT = "답변 생성이 제한 시간을 초과해 중단했어요. 질문을 좁혀서 다시 시도해 주세요."
ERROR_TEXT = "답변 생성 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요."

_MENTION = re.compile(r"<@[^>]+>")

# create_task 결과를 참조로 붙잡아 GC로 태스크가 사라지는 것을 방지
_tasks: set[asyncio.Task] = set()


def _default_answer(question: str) -> str:
    """요청마다 새 커넥션으로 에이전트 실행 (동시 요청 간 커넥션 공유 없음)."""
    from app.agent.__main__ import run
    from app.db import get_connection

    with get_connection() as conn:
        return run(question, conn)


def register(app, answer_fn=_default_answer, *, timeout: float = TIMEOUT_SECONDS) -> None:
    """app_mention 핸들러 등록. answer_fn: 질문 → 답변 (동기, 워커 스레드에서 실행)."""

    async def _answer(question: str, thread_ts: str, say) -> None:
        try:
            text = await asyncio.wait_for(asyncio.to_thread(answer_fn, question), timeout)
        except TimeoutError:
            text = TIMEOUT_TEXT
        except Exception:
            logger.exception("에이전트 실행 실패")
            text = ERROR_TEXT
        await say(text=text, thread_ts=thread_ts)

    @app.event("app_mention")
    async def handle_mention(event, say):
        question = _MENTION.sub("", event["text"]).strip()
        thread_ts = event.get("thread_ts") or event["ts"]
        task = asyncio.create_task(_answer(question, thread_ts, say))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
