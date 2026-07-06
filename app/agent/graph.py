"""LangGraph ReAct 에이전트 — create_react_agent 기반 (ROADMAP 확정: 커스텀 그래프 없음).

재검색 루프는 별도 노드가 아니라 ReAct 루프 + SYSTEM_PROMPT 유도로 구현한다.
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, create_react_agent

from app.agent.tools import make_tools

CHAT_MODEL = "gpt-4o-mini"

# Slack 60초 타임아웃이 워커 스레드를 중단하지 못하므로(비용은 계속 발생),
# LLM 호출별 상한 + ReAct 스텝 상한으로 폭주 비용의 총량을 유계로 만든다.
LLM_TIMEOUT_SECONDS = 30  # gpt-4o-mini 정상 응답은 수 초 — 초과는 이상 상황
LLM_MAX_RETRIES = 1  # HTTP 레벨 재시도 (ROADMAP의 "에이전트 자동 재시도 없음"과 별개)
RECURSION_LIMIT = 16  # ≈ tool 7회 — 골든셋 최다 경로(재시도 포함 4~5회)에 여유

SYSTEM_PROMPT = """\
너는 패션 커머스 리뷰(VoC) 분석 어시스턴트다. 마케터의 질문에 리뷰 데이터를 근거로 답한다.

## 데이터
- 리뷰 약 34만 건 (2025-07-08 ~ 2026-07-03, KST). 브랜드 약 2천 개, 카테고리 12개.
- 평점(grade)은 1~5이며 5점이 약 86%로 쏠려 있다.
- 건수·평점·비율 등 모든 수치는 프롬프트가 아니라 반드시 tool로 조회해 답한다.

## tool 선택
- 정량 질문(건수, 평균 평점, 부정률, 순위/비교, 추이) → aggregate_reviews
- 정성 질문(리뷰 내용, 이유, 의견, 사례) → search_reviews
- 브랜드/카테고리/중분류 필터를 걸기 전에는 먼저 list_metadata로 데이터의
  정확한 명칭과 계층을 확인하고, 카테고리는 category_name에, 중분류(예: 데님
  팬츠, 스니커즈)는 sub_category_name에 넣는다. 잘못된 계층에 넣으면 0건이 된다.
- list_metadata는 명칭별 리뷰 수(n)도 반환한다. "리뷰가 가장 많은 X" 판단은
  반환된 n을 그대로 사용한다.

## 필수 규칙
- 불만/단점/문제점 질의는 반드시 grade_max=3 필터를 건다. 필터 없이 검색하면
  긍정 리뷰만 나온다.
- 질문이 특정 브랜드/카테고리/중분류/품목을 지목하면 해당 메타필터를 걸어
  무관한 품목의 리뷰가 섞이지 않게 한다.
- 순위/비교는 neg_rate(Wilson lower bound 정렬)를 사용하고, 표본 미달로 제외된
  그룹이 있을 수 있음을 답변에 반영한다.
- tool 결과가 0건이거나 관련성이 낮으면 그대로 포기하지 말고, list_metadata로
  명칭을 재확인하거나 필터를 완화하고 쿼리를 바꿔 반드시 재시도한다.
  재시도 후에도 없으면 없다고 답한다.

## 답변
- 근거 리뷰를 인용한다(발췌 + 평점, 필요시 review_id).
- 데이터에 없는 내용을 지어내지 않는다. 숫자는 tool 결과 그대로 사용한다.
- 한국어로 간결하게 답한다.
"""


def _default_model() -> ChatOpenAI:
    """호출 상한이 걸린 기본 모델 (무제한 대기 + 재시도 2회 기본값 사용 금지)."""
    return ChatOpenAI(
        model=CHAT_MODEL, temperature=0, timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES
    )


def build_agent(conn, *, model=None):
    """리뷰 분석 ReAct 에이전트를 컴파일해 반환. model 미지정 시 gpt-4o-mini."""
    if model is None:
        model = _default_model()
    # langgraph v1의 ToolNode는 기본으로 tool 내부 예외를 잡지 않음 — 에러를
    # ToolMessage로 모델에 돌려보내 스스로 인자를 고치게 하려면 명시 필요
    tool_node = ToolNode(make_tools(conn), handle_tool_errors=True)
    agent = create_react_agent(model, tool_node, prompt=SYSTEM_PROMPT)
    return agent.with_config(recursion_limit=RECURSION_LIMIT)
