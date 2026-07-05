"""CLI: python -m app.agent "질문" — 에이전트 수동 verify용 진입점."""

import argparse

from app.agent.graph import build_agent
from app.db import get_connection


def run(question: str, conn, *, model=None) -> str:
    """질문 1건을 에이전트에 넣고 최종 답변 텍스트를 반환."""
    agent = build_agent(conn, model=model)
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.agent", description="VoC 분석 에이전트")
    parser.add_argument("question", help="에이전트에 물을 질문")
    args = parser.parse_args(argv)
    with get_connection() as conn:
        print(run(args.question, conn))


if __name__ == "__main__":
    main()
