"""골든셋 평가 러너 — 2단계 채점: (a) tool 트래젝토리, (b) 최종 답변."""

import json
from pathlib import Path

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"


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
