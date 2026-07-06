"""build_agent 그래프 wiring 테스트 (게이트 2 승인 4케이스).

실 LLM 없이 스크립트된 가짜 ChatModel로 그래프 배선만 검증한다.
gpt-4o-mini의 실제 tool 선택 품질은 CLI verify와 평가 파이프라인(6번 단위) 담당.
"""

import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.agent.graph import SYSTEM_PROMPT, build_agent
from tests.conftest import ScriptedChatModel


def _agent(conn, responses):
    model = ScriptedChatModel(responses=responses)
    return model, build_agent(conn, model=model)


AGG_CALL = AIMessage(
    content="",
    tool_calls=[
        {"name": "aggregate_reviews", "args": {"group_by": "brand", "metric": "count"}, "id": "c1"}
    ],
)


def test_binds_three_tools(conn):
    model, agent = _agent(conn, [AIMessage(content="답변")])
    agent.invoke({"messages": [("user", "질문")]})
    assert sorted(t.name for t in model.bound_tools) == [
        "aggregate_reviews",
        "list_metadata",
        "search_reviews",
    ]


def test_system_prompt_prepended(conn):
    model, agent = _agent(conn, [AIMessage(content="답변")])
    agent.invoke({"messages": [("user", "질문")]})
    first = model.received[0][0]
    assert isinstance(first, SystemMessage)
    assert first.content == SYSTEM_PROMPT


def test_tool_call_round_trip(conn):
    model, agent = _agent(conn, [AGG_CALL, AIMessage(content="최종 답변")])
    result = agent.invoke({"messages": [("user", "브랜드별 리뷰 수는?")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert {"group": "브랜드B", "n": 15} in json.loads(tool_messages[0].content)
    assert result["messages"][-1].content == "최종 답변"


def test_default_model_bounded():
    # 기본값(무제한 대기 + 재시도 2회)이면 호출 1번이 Slack 60초 예산을 단독으로 초과 가능
    from app.agent.graph import _default_model

    model = _default_model()
    assert model.request_timeout == 30
    assert model.max_retries == 1


def test_recursion_limit_bounds_tool_loop(conn):
    # tool 호출 10회를 스크립트해도 상한(16 스텝 ≈ tool 7회)에서 루프가 잘려야 한다.
    # langgraph v1 프리빌트는 예외 대신 remaining_steps로 우아하게 종료하므로,
    # 스크립트 11개가 전부 소진되지 않았는지(호출 횟수)로 검증한다.
    # 주의: 같은 AIMessage 객체 재사용 금지 — id 중복 제거로 루프가 조기 붕괴함.
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "aggregate_reviews", "args": {"metric": "count"}, "id": f"c{i}"}],
        )
        for i in range(10)
    ] + [AIMessage(content="답변")]
    model, agent = _agent(conn, responses)
    agent.invoke({"messages": [("user", "질문")]})
    assert model.calls < len(responses)  # 상한 없으면(기본 25) 11회 전부 소진됨 — 실측


def test_system_prompt_no_exact_total():
    # 프롬프트에 정확한 전체 건수가 있으면 LLM이 tool 없이 베껴 답한다 (eval quant-01 실측)
    assert "346,479" not in SYSTEM_PROMPT
    assert "346479" not in SYSTEM_PROMPT


def test_tool_error_fed_back_not_raised(conn):
    bad_call = AIMessage(
        content="",
        tool_calls=[{"name": "aggregate_reviews", "args": {"group_by": "잘못됨"}, "id": "c1"}],
    )
    model, agent = _agent(conn, [bad_call, AIMessage(content="에러를 인지한 답변")])
    result = agent.invoke({"messages": [("user", "질문")]})  # 예외 없이 완료되어야 함
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "group_by" in tool_messages[0].content
    assert result["messages"][-1].content == "에러를 인지한 답변"
