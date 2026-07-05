"""eval 파이프라인 테스트 — 사이클 1: 로더/트래젝토리, 사이클 2: 채점기."""

from collections import Counter

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eval.run_eval import (
    aggregate,
    extract_trajectory,
    load_golden,
    score_answer_qual,
    score_answer_quant,
    score_trajectory,
)


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


# --- 사이클 2: 채점기 (게이트 2 승인 13케이스 중 5~13) ---


def _item(**kw):
    base = {
        "id": "x",
        "type": "quant",
        "question": "질문",
        "expected_tools": [["aggregate_reviews"]],
        "arg_constraints": [],
    }
    base.update(kw)
    return base


def _call(name="aggregate_reviews", **args):
    return {"name": name, "args": args}


GRADE_LTE_3 = {"tool": "search_reviews", "arg": "grade_max", "op": "lte", "value": 3}


def test_trajectory_pass_all_requirements():
    item = _item(
        expected_tools=[["aggregate_reviews"], ["search_reviews"]],
        arg_constraints=[GRADE_LTE_3],
        min_tool_calls=2,
    )
    trajectory = [_call(metric="count"), _call("search_reviews", query="불만", grade_max=3)]
    assert score_trajectory(item, trajectory) == {"tools_ok": True, "args_ok": True}


def test_trajectory_alternative_satisfies():
    item = _item(expected_tools=[["list_metadata", "aggregate_reviews"]])
    assert score_trajectory(item, [_call(metric="count")])["tools_ok"] is True


def test_trajectory_missing_tool_fails():
    item = _item(expected_tools=[["aggregate_reviews"], ["search_reviews"]])
    assert score_trajectory(item, [_call(metric="count")])["tools_ok"] is False


def test_trajectory_min_tool_calls():
    item = _item(min_tool_calls=2)
    assert score_trajectory(item, [_call(metric="count")])["tools_ok"] is False


@pytest.mark.parametrize(
    ("constraint", "good_call", "bad_call"),
    [
        (
            {"tool": "aggregate_reviews", "arg": "reviewer_sex", "op": "eq", "value": "여성"},
            _call(reviewer_sex="여성"),
            _call(reviewer_sex="남성"),
        ),
        (
            GRADE_LTE_3,
            _call("search_reviews", grade_max=3),
            _call("search_reviews", grade_max=4),
        ),
        (
            {
                "tool": "search_reviews",
                "arg": "sub_category_name",
                "op": "contains",
                "value": "데님",
            },
            _call("search_reviews", sub_category_name="데님 팬츠"),
            _call("search_reviews", sub_category_name="슈트 팬츠/슬랙스"),
        ),
    ],
)
def test_arg_constraint_ops(constraint, good_call, bad_call):
    item = _item(expected_tools=[[constraint["tool"]]], arg_constraints=[constraint])
    assert score_trajectory(item, [good_call])["args_ok"] is True
    assert score_trajectory(item, [bad_call])["args_ok"] is False


def test_arg_constraint_missing_arg_fails():
    item = _item(expected_tools=[["search_reviews"]], arg_constraints=[GRADE_LTE_3])
    assert score_trajectory(item, [_call("search_reviews", query="불만")])["args_ok"] is False


def test_quant_number_within_tolerance():
    item = _item(ground_truth={"numbers": [{"values": [116874], "rel_tol": 0.01}]})
    assert score_answer_quant(item, "2026년 6월 리뷰는 약 116,000건입니다.") is True


def test_quant_number_comma_parsing():
    item = _item(ground_truth={"numbers": [{"values": [346479], "abs_tol": 0}]})
    assert score_answer_quant(item, "전체 리뷰는 총 346,479건입니다.") is True


def test_quant_number_out_of_tolerance_fails():
    item = _item(ground_truth={"numbers": [{"values": [116874], "rel_tol": 0.01}]})
    assert score_answer_quant(item, "약 120,000건입니다.") is False


def test_quant_string_alternatives():
    item = _item(ground_truth={"strings": [["2026-06", "6월"]]})
    assert score_answer_quant(item, "리뷰가 가장 많은 달은 6월입니다.") is True
    assert score_answer_quant(item, "리뷰가 가장 많은 달은 5월입니다.") is False


def test_quant_percent_alternative_values():
    item = _item(ground_truth={"numbers": [{"values": [3.05, 0.0305], "rel_tol": 0.05}]})
    assert score_answer_quant(item, "부정률은 약 3.1%입니다.") is True
    assert score_answer_quant(item, "부정률은 0.03 수준입니다.") is True


def test_qual_judge_score_reflected():
    item = _item(type="qual", rubric="루브릭")
    calls = []

    def fake_judge(question, rubric, answer):
        calls.append((question, rubric, answer))
        return 4

    assert score_answer_qual(item, "답변", judge=fake_judge) == 4
    assert calls == [("질문", "루브릭", "답변")]


def test_aggregate_report():
    results = [
        {"type": "quant", "tools_ok": True, "args_ok": True, "quant_ok": True, "judge_score": None},
        {
            "type": "quant",
            "tools_ok": False,
            "args_ok": False,
            "quant_ok": False,
            "judge_score": None,
        },
        {"type": "qual", "tools_ok": True, "args_ok": True, "quant_ok": None, "judge_score": 4},
        {"type": "multi", "tools_ok": True, "args_ok": True, "quant_ok": True, "judge_score": 3},
    ]
    report = aggregate(results)
    assert report["tool_select_rate"] == 0.75
    assert report["quant_accuracy"] == pytest.approx(2 / 3)
    assert report["judge_avg"] == 3.5
    assert report["passed"] == {"tool_select": False, "quant": False, "judge": True}
