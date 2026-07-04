# CLAUDE.md — VoC 분석 Slack 챗봇

## 프로젝트 개요

리뷰 데이터를 분석하는 LangGraph 기반 Slack 챗봇 + 주간 자동 리포트.
사용자 — 클로드 코드 간 서로 합의된 전체 계획은 `ROADMAP.md` 참조. **ROADMAP.md의 확정 결정사항은 임의로 변경하지 않는다.**
변경이 필요하다고 판단되면 먼저 사용자에게 근거를 제시하고 합의 후 ROADMAP.md를 갱신한다.

## 기술 스택 (확정 — 변경 금지)

- Python 3.12 / **uv** / ruff / pytest (mypy 미사용)
- FastAPI + slack-bolt(FastAPI adapter) + ngrok
- PostgreSQL 16 + pgvector (Docker, `pgvector/pgvector:pg16`)
- OpenAI: gpt-4o-mini, text-embedding-3-small
- LangGraph, LangSmith(tracing/평가)

## 개발 워크플로우 (필수 준수)

TDD(red → green → refactor)로 진행하되, 아래 승인 게이트를 반드시 거친다.
**승인은 사용자의 명시적 답변으로만 성립한다. 승인 없이 다음 단계로 진행하지 않는다.**

### 게이트 1 — 개발 단위 시작 전
ROADMAP.md 구현 순서의 각 항목을 시작할 때, 다음을 제시하고 승인을 받는다:
- 이 단위에서 만들 것 / 만들지 않을 것 (스코프)
- 파일 구조와 주요 함수/클래스 설계
- 예상 사이클 수

### 게이트 2 — red 단계 전
테스트 작성 전에 다음을 제시하고 승인을 받는다:
- 작성할 테스트 케이스 목록 (각 케이스가 검증하는 동작을 한 줄로)
- 엣지 케이스 포함 여부와 이유

### 게이트 2.5 — red 완료 후, green 진행 전
- 작성된 테스트 코드와 red 실패 출력을 제시 → 사용자가 코드 확인 후 승인 → green 진행

### green + refactor
승인된 테스트를 통과시키고 리팩토링까지 연속 실행한다. 중간 승인 불필요.

### 게이트 3 — commit 전
- 변경된 파일과 diff 요약을 제시 → 사용자 승인 → commit + push
- 커밋 메시지는 Conventional Commits (`feat:`, `fix:`, `test:`, `refactor:`, `chore:`)

### 이원화 규칙
- **풀 게이트(1+2+3) 적용**: `app/agent/` (tools, graph), `eval/`, `app/report/`
- **게이트 3만 적용**: docker-compose.yml, DDL, 설정 파일, .github/, 기타 스캐폴딩

## Git

- GitHub Flow: main + 기능 브랜치 (`feat/ingest-pipeline`, `fix/...`, `chore/...`)
- 사이클 끝 commit, push 수시, 기능 단위 완료 시 PR → main merge
- PR 생성 전 로컬에서 `pytest`와 `ruff check` 통과 확인
- CI 미도입 (MVP 범위 제외 — 배포 단계에서 도입 검토). 품질 게이트는 로컬 pytest + ruff

## 데이터 관련 주의사항

- 소스: `ms_reviews.db` (SQLite, 리포 외부 경로 — `.env`의 `SOURCE_DB_PATH`. **git에 커밋 금지** — .gitignore 확인)
- **정적 DB(346,479건), 추가 갱신 없음** — 주간 리포트 기준일은 파라미터(기본: DB 내 최신 create_date)
- 평점 5점 ~85.7% 쏠림 → 불만/단점 질의는 grade<=3 필터 필수
- 표본 판단은 `review_count`(실수집) 기준, `site_review_count`는 과소집계
- 순위/비교 집계는 **부정률(grade<=3)의 Wilson lower bound** + min_n(기본 10) 미달 그룹 제외
- `satisfaction`은 카테고리마다 JSON 키가 다름 — MVP에서는 필터/집계에 사용하지 않음
- `create_date`는 ISO8601(+09:00) → TIMESTAMPTZ로 파싱
- ingest upsert 시 `ON CONFLICT DO UPDATE`에서 embedding 컬럼 제외 (재실행 시 임베딩 보존)

## 시크릿

- `.env`에만 저장: `OPENAI_API_KEY`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `DATABASE_URL`, `SOURCE_DB_PATH`
- 코드/커밋/로그에 시크릿 노출 절대 금지

---

## 일반 행동 지침 (Behavioral Guidelines)

Behavioral guidelines to reduce common LLM coding mistakes.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
