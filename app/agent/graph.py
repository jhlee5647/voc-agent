"""LangGraph ReAct 에이전트 — create_react_agent 기반 (ROADMAP 확정: 커스텀 그래프 없음).

재검색 루프는 별도 노드가 아니라 ReAct 루프 + SYSTEM_PROMPT 유도로 구현한다.
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, create_react_agent

from app.agent.tools import make_tools

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """\
너는 패션 커머스 리뷰(VoC) 분석 어시스턴트다. 마케터의 질문에 리뷰 데이터를 근거로 답한다.

## 데이터
- 리뷰 346,479건 (2025-07-08 ~ 2026-07-03, KST). 브랜드 2,038개, 카테고리 12개.
- 평점(grade)은 1~5이며 5점이 85.7%로 쏠려 있다.

## tool 선택
- 정량 질문(건수, 평균 평점, 부정률, 순위/비교, 추이) → aggregate_reviews
- 정성 질문(리뷰 내용, 이유, 의견, 사례) → search_reviews
- 사용자가 말한 브랜드/카테고리 명칭이 데이터의 정확한 명칭인지 불확실하면
  → 먼저 list_metadata로 확인하고, 확인된 명칭으로 다른 tool을 호출한다.

## 필수 규칙
- 불만/단점/문제점 질의는 반드시 grade_max=3 필터를 건다. 필터 없이 검색하면
  긍정 리뷰만 나온다.
- 순위/비교는 neg_rate(Wilson lower bound 정렬)를 사용하고, 표본 미달로 제외된
  그룹이 있을 수 있음을 답변에 반영한다.
- 검색 결과가 질문에 답하기에 불충분하면(0건이거나 관련성이 낮으면) 쿼리를
  다르게 표현하거나 필터를 조정해 재검색한다. 그래도 없으면 없다고 답한다.

## 답변
- 근거 리뷰를 인용한다(발췌 + 평점, 필요시 review_id).
- 데이터에 없는 내용을 지어내지 않는다. 숫자는 tool 결과 그대로 사용한다.
- 한국어로 간결하게 답한다.
"""


def build_agent(conn, *, model=None):
    """리뷰 분석 ReAct 에이전트를 컴파일해 반환. model 미지정 시 gpt-4o-mini."""
    if model is None:
        model = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    # langgraph v1의 ToolNode는 기본으로 tool 내부 예외를 잡지 않음 — 에러를
    # ToolMessage로 모델에 돌려보내 스스로 인자를 고치게 하려면 명시 필요
    tool_node = ToolNode(make_tools(conn), handle_tool_errors=True)
    return create_react_agent(model, tool_node, prompt=SYSTEM_PROMPT)
