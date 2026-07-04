# ROADMAP.md — VoC 분석 Slack 챗봇

> 사용자 — 클로드 코드 간 최종 합의본. **확정 결정사항은 임의로 변경하지 않는다.**
> 변경이 필요하면 근거를 제시하고 사용자와 합의 후 이 문서를 갱신한다.

## 개요

VoC를 분석하는 LangGraph 기반 Slack 챗봇 + 주간 자동 리포트.

**포트폴리오 서사**: 마케터가 매주 수동으로 리뷰를 훑던 반복 업무를 주간 자동 리포트로 대체하고, 심화 분석은 챗봇 질의응답으로 셀프서비스화. 데이터 수집(크롤링)은 범위 내에서 제외.

**데이터 소스**: `ms_reviews.db` (SQLite, 사전에 크롤링한 리뷰 데이터, 리포 외부 경로 — `.env`의 `SOURCE_DB_PATH`로 지정).
- 리뷰 **346,479건** / 상품 12,067건 / 브랜드 2,038개 / 카테고리 12
- **정적 DB — 추가 갱신 없음.** MVP는 전량 346k로 진행
- `create_date` 범위: 2025-07-08 ~ 2026-07-03 (+09:00)

## 확정 결정 (변경 금지)

- Python 3.12 / uv / ruff(lint+format) / pytest. mypy 미사용
- FastAPI + slack-bolt(FastAPI adapter) + ngrok (HTTP Events 방식)
- PostgreSQL 16 + pgvector (Docker, `pgvector/pgvector:pg16`)
- OpenAI: gpt-4o-mini, text-embedding-3-small
- LangGraph (create_react_agent로 시작 → 재검색 루프 필요 시 커스텀 그래프)
- LangSmith 무료 티어 (tracing + 평가 기록)
- 임베딩: 리뷰 1건 = 벡터 1개, 청킹 없음, content만 벡터화 (메타데이터는 필터로)
- 벡터 인덱스: **IVFFlat(`vector_cosine_ops`, lists=600)를 Phase 0 DDL에 포함**.
  당초 HNSW로 결정했으나 로컬 개발 머신(RAM 4GB, Docker 가용 ~1.8GB)에서 346k×1536차원
  HNSW 빌드가 메모리 초과로 20시간+ 소요됨을 실측 확인 → IVFFlat로 변경 (2026-07-05 합의).
  배포 환경(충분한 RAM)에서는 HNSW 재검토. 메타필터 + ANN 병용 시 recall 저하 가능
  → 필요 시 `ivfflat.probes` 상향 조정
- 테이블: reviews 단일 테이블에 category_name/sub_category_name 비정규화 복사(products에서 goods_no 조인해 복사, 실측 조인 미매칭 0건) + products 보조 테이블 유지
- satisfaction/repurchase: JSONB 저장, 검색 결과 컨텍스트에 포함, 필터/집계에는 미사용 (MVP 이후 검토)
- **순위/비교 지표: 부정률(grade<=3 비율)의 Wilson lower bound** + min_n(기본 10) 미달 그룹 제외
- 에러 정책: 에이전트 60초 타임아웃, 실패 시 스레드에 에러 답변, 자동 재시도 없음
- MCP 미사용 (plain LangChain tool)
- CI 미도입 (MVP 범위 제외 — 이후 클라우드 배포 단계에서 GitHub Actions 도입 검토). 품질 게이트는 커밋 전 로컬 `pytest` + `ruff check`로 유지

## 로드맵 (MVP까지)

- **Phase 0**: SQLite → PostgreSQL 전량(346k) 적재 + 임베딩 파이프라인 (로컬 Docker)
- **Phase 1 (MVP)**: Slack 봇 + 에이전트 + 3 tools + 주간 리포트 + 평가셋 (로컬 + ngrok)

> MVP 완료까지가 본 계획의 범위. 이후 개발(부하 테스트, 배포 등)은 MVP 완료 시점에 사용자가 별도 초안을 제공하면 합의 후 이 문서를 갱신한다.

## Phase 0 상세

1. docker-compose.yml: `pgvector/pgvector:pg16` 컨테이너 1개
2. DDL: reviews(SQLite reviews 컬럼 전체 + category_name/sub_category_name + satisfaction/repurchase JSONB + create_date TIMESTAMPTZ + embedding vector(1536) + IVFFlat 인덱스), products(SQLite 그대로)
3. ingest/: ms_reviews.db 읽기 → 변환 → INSERT. review_id PK upsert로 중복 방지.
   **`ON CONFLICT (review_id) DO UPDATE`에서 embedding 컬럼은 제외** — 재실행 시 기존 임베딩 보존
4. 임베딩 배치 생성: 미임베딩 행(embedding IS NULL)만 처리, 재실행 안전
5. .env + .gitignore (OpenAI 키, Slack 시크릿, ms_reviews.db 제외)

## 디렉토리 구조 (리포 루트 = voc_agent/)

```
voc_agent/
  app/
    main.py            # FastAPI + slack-bolt adapter (/slack/events)
    slack_handlers.py  # app_mention: 즉시 ack → asyncio task → 스레드 답변
    agent/graph.py, agent/tools.py
    report/weekly.py, report/scheduler.py  # APScheduler
    db.py
  ingest/
  eval/golden.jsonl, eval/run_eval.py
  tests/
  docker-compose.yml
```

## Tools (3개)

1. **aggregate_reviews** — 정량 질문. 구조화 인자 → 파라미터화 SQL (자유 text-to-SQL 금지).
   - 인자: brand_name, category_name, sub_category_name, grade_min, grade_max, date_from, date_to, reviewer_sex, group_by(brand|category|sub_category|grade|month), metric(**count|avg_grade|neg_rate**), min_n(기본 10)
   - neg_rate = 그룹 내 grade<=3 비율. 순위/비교는 neg_rate의 Wilson lower bound 기준 정렬, n < min_n 그룹 제외
2. **search_reviews** — 정성 질문. pgvector 시맨틱 검색 + 메타필터(brand_name, category_name, sub_category_name, grade_min/max, reviewer_sex, date_from/to).
   - "불만/단점" 질의는 grade<=3 필터를 시스템 프롬프트로 유도 (5점 85.7% 쏠림 때문)
   - 반환 컨텍스트에 satisfaction, goods_option, reviewer_height/weight 포함
3. **list_metadata** — 브랜드/카테고리/중분류 목록 조회. 부정확한 명칭 입력의 인자 매칭 지원.

## 에이전트 흐름

질문 → tool 선택(정량→SQL, 정성→검색) → 검색 결과 충분성 자기 평가 → 부족하면 쿼리 재작성/필터 조정 후 재검색 → 근거 리뷰 인용 답변.

## 주간 리포트 (MVP 마지막 기능)

- 스케줄: 매주 월 09:00 KST (APScheduler)
- **기준일은 파라미터** — 기본값은 DB 내 최신 create_date (정적 DB이므로 "실행 시점 오늘" 기준이면 리뷰 0건이 됨. 파라미터화로 언제 데모해도 유의미한 리포트 보장)
- 내용: 기준일로부터 지난 7일(create_date 기준) (1) 리뷰 수/평균 평점 요약(전주 대비), (2) grade<=3 부정률 급증 브랜드 상위(neg_rate Wilson LB + min_n 적용), (3) 부정 리뷰 대표 발췌 2~3건 → Slack 채널 포스팅
- 기존 tools/에이전트 재사용

## Slack 핵심 함정

- **3초 ACK 규칙**: 이벤트 수신 즉시 ack(), LLM 실행은 asyncio task 분리, 완료 후 chat.postMessage로 스레드 답변. 미준수 시 Slack 재전송으로 중복 응답 발생
- signing secret 검증(bolt 기본), app_mention만 처리, 대화 메모리는 MVP 제외

## 평가

- eval/golden.jsonl 20문항: 정량 8(정답 SQL 산출→exact/허용오차) / 정성 8(그중 2~3개는 중분류+grade 필터형, LLM-as-judge) / 복합 4(tool 2개 이상)
- 2단계: (a) tool trajectory(올바른 tool·인자, pytest), (b) 최종 답변(숫자 ground truth, 서술 judge)
- 통과 기준: tool 선택 ≥90%, 정량 정답률 ≥90%, judge 평균 ≥3.5/5

## 구현 순서

1. 리포 초기화 + uv + Docker Compose + DDL → verify: docker compose up, 테이블·인덱스 확인, 로컬 pytest·ruff 통과
2. ingest + 임베딩 (전량 346k) → verify: 건수 일치, NULL율, 샘플 벡터 검색
3. 3 tools (테스트 먼저) → verify: pytest (Wilson/neg_rate/min_n 케이스 포함)
4. LangGraph 에이전트 → verify: CLI에서 골든셋 일부 통과
5. FastAPI + Slack 연동(ngrok) → verify: 실 Slack에서 정량/정성/복합 3종
6. 평가 파이프라인 → verify: 통과 기준 충족 리포트
7. 주간 리포트 → verify: 수동 트리거(기준일 파라미터)로 실 채널 포스팅 + 내용 스모크
