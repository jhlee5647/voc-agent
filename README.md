# VoC 분석 Slack 챗봇

패션 커머스 리뷰(VoC) 34.6만 건을 분석하는 **LangGraph 기반 Slack 챗봇 + 주간 자동 리포트**.

마케터가 매주 수동으로 리뷰를 훑던 반복 업무를 주간 자동 리포트로 대체하고,
심화 분석("데님 팬츠 사이즈 불만 리뷰 찾아줘", "부정률이 가장 높은 브랜드는?")은
Slack 멘션 질의응답으로 셀프서비스화한다.

## 아키텍처

```
Slack (@봇 멘션)
  │  HTTP Events (ngrok)
  ▼
FastAPI + slack-bolt ──── 즉시 ACK(3초 규칙) → asyncio task → 스레드 답변
  │
  ▼
LangGraph ReAct 에이전트 (gpt-4o-mini, 60초 타임아웃 + 16스텝 상한)
  │
  ├─ aggregate_reviews   정량 집계 — 구조화 인자 → 파라미터화 SQL (text-to-SQL 금지)
  ├─ search_reviews      정성 검색 — pgvector 시맨틱 검색 + 메타필터
  └─ list_metadata       브랜드/카테고리/중분류 명칭·리뷰 수 조회 (인자 매칭용)
  │
  ▼
PostgreSQL 16 + pgvector (reviews 346,479행, text-embedding-3-small 1536차원)

APScheduler (매주 월 09:00 KST) ─→ 주간 리포트 생성 → Slack 채널 포스팅
```

## 기술 스택

Python 3.12 · uv · FastAPI · slack-bolt · LangGraph · OpenAI(gpt-4o-mini, text-embedding-3-small) · PostgreSQL 16 + pgvector(IVFFlat) · APScheduler · pytest · ruff

## 셋업

사전 조건: Docker, [uv](https://docs.astral.sh/uv/), 소스 데이터 `ms_reviews.db`(SQLite), Slack 앱(Bot Token + Signing Secret), OpenAI API 키.

```bash
# 1. 환경변수 — .env.example을 복사해 값 채우기
cp .env.example .env

# 2. PostgreSQL + pgvector 기동 (최초 기동 시 db/schema.sql 자동 적용)
docker compose up -d

# 3. 의존성 설치
uv sync

# 4. 데이터 적재: SQLite → PostgreSQL (재실행 안전 — upsert, 임베딩 보존)
uv run python -m ingest.ingest

# 5. 임베딩 생성: 미임베딩 행만 배치 처리 (중단 후 재실행 안전)
uv run python -m ingest.embed
```

## 실행

```bash
# Slack 봇 (주간 리포트 스케줄러 포함, 단일 워커 전제)
uv run uvicorn app.main:api --port 3000
ngrok http 3000   # Slack 앱 Event Subscriptions에 https://.../slack/events 등록

# CLI로 에이전트 단건 질의 (Slack 없이 검증)
uv run python -m app.agent "부정률이 가장 높은 브랜드는 어디야?"

# 평가 (골든셋 20문항, 결과는 eval/results/에 저장)
uv run python -m eval.run_eval

# 테스트 / 린트
uv run pytest
uv run ruff check .
```

> 주간 리포트 기준일은 파라미터(기본: DB 내 최신 create_date)다. 소스가 정적 DB라
> "실행 시점 오늘" 기준이면 리뷰 0건이 되기 때문 — 언제 데모해도 유의미한 리포트가 나온다.

## 평가

골든셋 20문항(정량 8 / 정성 8 / 복합 4)을 2단계로 채점한다:
(a) tool 트래젝토리 — 올바른 tool과 인자를 썼는가, (b) 최종 답변 — 숫자는 허용오차, 서술은 LLM-as-judge(gpt-4o, 에이전트와 분리해 self-preference 편향 방지).

| 지표 | 결과 | 통과 기준 |
|---|---|---|
| tool 선택률 (tool + 인자) | 100% | ≥90% |
| 정량 정답률 | 100% | ≥90% |
| judge 평균 | 4.18 / 5 | ≥3.5 |

## 주요 설계 결정

전체 결정 이력은 [ROADMAP.md](ROADMAP.md) 참조. 하이라이트:

- **자유 text-to-SQL 금지** — tool은 구조화 인자만 받고, SQL 문장 구조에 들어가는
  값(group_by/metric)은 화이트리스트, 나머지는 전부 파라미터 바인딩
- **순위/비교는 부정률(grade≤3)의 Wilson lower bound** — 평점 5점이 ~86%로 쏠린
  데이터에서 소표본 그룹의 순위 왜곡 방지 (min_n 미달 그룹 제외)
- **IVFFlat + iterative_scan(relaxed_order)** — 로컬 4GB RAM에서 346k×1536 HNSW
  빌드 불가를 실측해 IVFFlat 채택. 메타필터로 후보가 고갈될 때의 결과 누락을
  iterative scan으로 방지 (recall 실측 근거는 PR #8)
- **Slack 3초 ACK 규칙** — 이벤트 수신 즉시 ack, LLM 실행은 태스크 분리.
  미준수 시 Slack 재전송으로 중복 응답 발생
- **에이전트 비용 상한** — LLM 호출별 30초 timeout + ReAct 16스텝. 타임아웃 후에도
  워커 스레드가 계속 도는 구조적 한계를 총량 유계로 보완

## 프로젝트 구조

```
app/
  main.py            # FastAPI + slack-bolt adapter (/slack/events) + 스케줄러 lifespan
  slack_handlers.py  # app_mention: 즉시 ack → asyncio task → 스레드 답변
  agent/graph.py     # ReAct 에이전트 (SYSTEM_PROMPT, 비용 상한)
  agent/tools.py     # 3 tools + Wilson lower bound
  report/weekly.py   # 주간 리포트 빌더 (tools 재사용)
  report/scheduler.py
  db.py
ingest/              # SQLite → Postgres 적재 + 임베딩 배치
eval/                # 골든셋 + 평가 러너
tests/               # 결정적 시드 + 스크립트 모델 (실 LLM 없이 96+건)
db/schema.sql
```
