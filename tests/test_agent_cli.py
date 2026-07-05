"""CLI 테스트 (게이트 2 승인 2케이스).

main()의 실 LLM 경로는 pytest 대상이 아님 — CLI verify(실 DB + gpt-4o-mini)가 담당.
"""

import pytest
from langchain_core.messages import AIMessage

from app.agent.__main__ import main, run
from tests.conftest import ScriptedChatModel


def test_run_returns_final_answer(conn):
    model = ScriptedChatModel(responses=[AIMessage(content="최종 답변")])
    assert run("브랜드별 리뷰 수는?", conn, model=model) == "최종 답변"


def test_cli_requires_question_arg():
    with pytest.raises(SystemExit):
        main([])
