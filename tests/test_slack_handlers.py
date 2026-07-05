"""slack_handlers 테스트 (게이트 2 승인 6케이스).

실제 bolt AsyncApp 없이 — FakeApp으로 핸들러를 직접 획득하고, say는 기록만 한다.
answer_fn은 asyncio.to_thread로 실행되므로 threading 기반으로 블록/지연시킨다.
"""

import asyncio
import threading
import time

from app.slack_handlers import ERROR_TEXT, TIMEOUT_TEXT, register


class FakeApp:
    def __init__(self):
        self.handlers = {}

    def event(self, name):
        def deco(fn):
            self.handlers[name] = fn
            return fn

        return deco


class FakeSay:
    def __init__(self):
        self.calls = []

    async def __call__(self, text=None, thread_ts=None):
        self.calls.append({"text": text, "thread_ts": thread_ts})


def make_event(text="<@U0BOT> 질문", ts="111.222", thread_ts=None):
    event = {"text": text, "ts": ts, "channel": "C01"}
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


def setup(answer_fn, timeout=5.0):
    app, say = FakeApp(), FakeSay()
    register(app, answer_fn, timeout=timeout)
    return app.handlers["app_mention"], say


async def until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "제한 시간 내에 조건 미충족"
        await asyncio.sleep(0.01)


async def test_mention_stripped():
    received = []
    handler, say = setup(lambda q: received.append(q) or "답변")
    await handler(event=make_event("<@U0BOT> 브랜드별 리뷰 수"), say=say)
    await until(lambda: say.calls)
    assert received == ["브랜드별 리뷰 수"]


async def test_handler_returns_before_answer():
    gate = threading.Event()

    def slow_answer(question):
        gate.wait(3)
        return "늦은 답변"

    handler, say = setup(slow_answer)
    await handler(event=make_event(), say=say)  # answer_fn이 블록돼 있어도 즉시 반환(ACK)
    assert say.calls == []
    gate.set()
    await until(lambda: say.calls)
    assert say.calls[0]["text"] == "늦은 답변"


async def test_answer_posted_to_thread():
    handler, say = setup(lambda q: "답변")
    await handler(event=make_event(ts="111.222"), say=say)
    await until(lambda: say.calls)
    assert say.calls == [{"text": "답변", "thread_ts": "111.222"}]


async def test_reply_in_existing_thread():
    handler, say = setup(lambda q: "답변")
    await handler(event=make_event(ts="222.333", thread_ts="000.111"), say=say)
    await until(lambda: say.calls)
    assert say.calls[0]["thread_ts"] == "000.111"


async def test_timeout_posts_error_once():
    handler, say = setup(lambda q: time.sleep(1) or "답변", timeout=0.05)
    await handler(event=make_event(), say=say)
    await until(lambda: say.calls)
    await asyncio.sleep(0.2)  # 자동 재시도가 있다면 추가 호출이 생길 시간
    assert say.calls == [{"text": TIMEOUT_TEXT, "thread_ts": "111.222"}]


async def test_exception_posts_error():
    def broken_answer(question):
        raise ValueError("에이전트 내부 오류")

    handler, say = setup(broken_answer)
    await handler(event=make_event(), say=say)
    await until(lambda: say.calls)
    assert say.calls == [{"text": ERROR_TEXT, "thread_ts": "111.222"}]
