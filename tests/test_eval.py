"""eval 파이프라인 테스트 — 사이클 1: 로더/트래젝토리 (게이트 2 승인 4케이스)."""

from collections import Counter

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eval.run_eval import extract_trajectory, load_golden


def test_load_golden_shape():
    items = load_golden()
    assert len(items) == 20
    assert Counter(i["type"] for i in items) == {"quant": 8, "qual": 8, "multi": 4}
    for item in items:
        assert {"id", "type", "question", "expected_tools"} <= item.keys()


def test_load_golden_ids_unique():
    ids = [i["id"] for i in load_golden()]
    assert len(ids) == len(set(ids))


def test_extract_trajectory_collects_tool_calls():
    messages = [
        HumanMessage("질문"),
        AIMessage(
            "", tool_calls=[{"name": "aggregate_reviews", "args": {"metric": "count"}, "id": "c1"}]
        ),
        ToolMessage("[]", tool_call_id="c1"),
        AIMessage(
            "", tool_calls=[{"name": "search_reviews", "args": {"query": "불만"}, "id": "c2"}]
        ),
        ToolMessage("[]", tool_call_id="c2"),
        AIMessage("최종 답변"),
    ]
    assert extract_trajectory(messages) == [
        {"name": "aggregate_reviews", "args": {"metric": "count"}},
        {"name": "search_reviews", "args": {"query": "불만"}},
    ]


def test_extract_trajectory_ignores_plain_messages():
    assert extract_trajectory([HumanMessage("질문"), AIMessage("답변")]) == []
