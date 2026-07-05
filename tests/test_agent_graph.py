"""build_agent 그래프 wiring 테스트 (게이트 2 승인 4케이스).

실 LLM 없이 스크립트된 가짜 ChatModel로 그래프 배선만 검증한다.
gpt-4o-mini의 실제 tool 선택 품질은 CLI verify와 평가 파이프라인(6번 단위) 담당.
"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.graph import SYSTEM_PROMPT, build_agent


class ScriptedChatModel(BaseChatModel):
    """응답 큐를 순서대로 뱉고, bind_tools 인자와 수신 메시지를 기록하는 가짜 모델."""

    responses: list[AIMessage]
    bound_tools: list = []
    received: list = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received.append(list(messages))
        message = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


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
