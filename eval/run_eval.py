"""골든셋 평가 러너 — 2단계 채점: (a) tool 트래젝토리, (b) 최종 답변."""

import json
import re
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"

# 통과 기준 (ROADMAP): tool 선택 ≥90%, 정량 정답률 ≥90%, judge 평균 ≥3.5/5
PASS_CRITERIA = {"tool_select": 0.9, "quant": 0.9, "judge": 3.5}

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """golden.jsonl을 문항 dict 목록으로 로드."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_trajectory(messages) -> list[dict]:
    """에이전트 결과 messages에서 tool 호출을 순서대로 추출 → [{name, args}]."""
    calls = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            calls.append({"name": tool_call["name"], "args": tool_call["args"]})
    return calls


def _constraint_met(call_args: dict, constraint: dict) -> bool:
    actual = call_args.get(constraint["arg"])
    if actual is None:
        return False
    op, value = constraint["op"], constraint["value"]
    if op == "eq":
        return actual == value
    if op == "lte":
        return actual <= value
    if op == "contains":
        return value in str(actual)
    raise ValueError(f"지원하지 않는 op: {op!r}")


def score_trajectory(item: dict, trajectory: list[dict]) -> dict:
    """(a)단계 채점. expected_tools는 AND-of-OR, arg_constraints는 호출 중 1개 만족."""
    called = [call["name"] for call in trajectory]
    tools_ok = all(
        any(name in called for name in alternatives) for alternatives in item["expected_tools"]
    ) and len(trajectory) >= item.get("min_tool_calls", 1)
    args_ok = all(
        any(
            _constraint_met(call["args"], constraint)
            for call in trajectory
            if call["name"] == constraint["tool"]
        )
        for constraint in item.get("arg_constraints", [])
    )
    return {"tools_ok": tools_ok, "args_ok": args_ok}


def _parse_numbers(text: str) -> list[float]:
    return [float(m.group(0).replace(",", "")) for m in _NUMBER.finditer(text)]


def _number_entry_met(entry: dict, numbers: list[float]) -> bool:
    for value in entry["values"]:
        tol = entry["abs_tol"] if "abs_tol" in entry else value * entry.get("rel_tol", 0.0)
        if any(abs(number - value) <= tol for number in numbers):
            return True
    return False


def score_answer_quant(item: dict, answer: str) -> bool:
    """(b)단계 정량 채점: 답변 내 수치가 허용오차 내, 기대 문자열(대안 중 1개) 포함."""
    ground_truth = item["ground_truth"]
    numbers = _parse_numbers(answer)
    numbers_ok = all(_number_entry_met(e, numbers) for e in ground_truth.get("numbers", []))
    strings_ok = all(
        any(alt in answer for alt in alternatives)
        for alternatives in ground_truth.get("strings", [])
    )
    return numbers_ok and strings_ok


def score_answer_qual(item: dict, answer: str, judge) -> int:
    """(b)단계 정성 채점: judge(question, rubric, answer) → 1~5점."""
    return judge(item["question"], item["rubric"], answer)


def aggregate(results: list[dict]) -> dict:
    """문항별 결과 → 지표 3종 + 통과 기준 판정."""
    tool_select_rate = sum(r["tools_ok"] for r in results) / len(results)
    quants = [r["quant_ok"] for r in results if r["quant_ok"] is not None]
    quant_accuracy = sum(quants) / len(quants) if quants else None
    scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
    judge_avg = sum(scores) / len(scores) if scores else None
    return {
        "tool_select_rate": tool_select_rate,
        "quant_accuracy": quant_accuracy,
        "judge_avg": judge_avg,
        "passed": {
            "tool_select": tool_select_rate >= PASS_CRITERIA["tool_select"],
            "quant": quant_accuracy is not None and quant_accuracy >= PASS_CRITERIA["quant"],
            "judge": judge_avg is not None and judge_avg >= PASS_CRITERIA["judge"],
        },
    }
