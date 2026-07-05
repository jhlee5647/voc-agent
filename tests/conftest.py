"""테스트 DB(voc_test) fixture — 실데이터(voc)와 격리된 결정적 시드 데이터."""

import os
from datetime import date, timedelta
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

load_dotenv()

SCHEMA = (Path(__file__).parent.parent / "db" / "schema.sql").read_text(encoding="utf-8")

# 시드 설계 (게이트 2 승인 내용):
#   브랜드A(상의/티셔츠, 5월): 12건 = 5점x9, 2점x2, 1점x1  → 부정 3/12
#   브랜드B(바지/데님, 6월):   15건 = 5점x14, 3점x1        → 부정 1/15
#   브랜드C(상의/셔츠, 5월):    5건 = 5점x3, 1점x2         → 부정 2/5 (min_n 미달용)
#   reviewer_sex: A는 F 4명, B는 F 5명, 나머지 M → F 총 9건
_BRANDS = [
    ("브랜드A", "상의", "티셔츠", [5] * 9 + [2, 2, 1], 4, date(2026, 5, 5)),
    ("브랜드B", "바지", "데님", [5] * 14 + [3], 5, date(2026, 6, 1)),
    ("브랜드C", "상의", "셔츠", [5, 5, 5, 1, 1], 0, date(2026, 5, 20)),
]


def seed_rows():
    rows = []
    rid = 0
    for brand, cat, sub_cat, grades, n_female, start in _BRANDS:
        for i, grade in enumerate(grades):
            rid += 1
            rows.append(
                {
                    "review_id": rid,
                    "goods_no": 100,
                    "brand_name": brand,
                    "grade": grade,
                    "content": f"리뷰 {rid}",
                    "reviewer_sex": "F" if i < n_female else "M",
                    "create_date": f"{start + timedelta(days=i)}T12:00:00+09:00",
                    "category_name": cat,
                    "sub_category_name": sub_cat,
                }
            )
    return rows


class ScriptedChatModel(BaseChatModel):
    """응답 큐를 순서대로 뱉고, bind_tools 인자와 수신 메시지를 기록하는 가짜 모델."""

    responses: list[AIMessage]
    bound_tools: list = []
    received: list = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received.append(list(messages))
        message = self.responses[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _test_url() -> str:
    return os.environ["DATABASE_URL"].rsplit("/", 1)[0] + "/voc_test"


@pytest.fixture(scope="session")
def test_db_url():
    admin = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    if not admin.execute("SELECT 1 FROM pg_database WHERE datname = 'voc_test'").fetchone():
        admin.execute("CREATE DATABASE voc_test")
    admin.close()
    with psycopg.connect(_test_url()) as c:
        # IVFFlat(lists=600) 인덱스 생성에 ~180MB 필요 (기본 64MB로는 실패)
        c.execute("SET maintenance_work_mem = '256MB'")
        c.execute(SCHEMA)
        c.commit()
    return _test_url()


@pytest.fixture
def conn(test_db_url):
    with psycopg.connect(test_db_url) as c:
        # TRUNCATE는 IVFFlat 인덱스도 재초기화하므로 동일한 메모리 상향 필요
        c.execute("SET maintenance_work_mem = '256MB'")
        c.execute("TRUNCATE reviews, products")
        rows = seed_rows()
        cols = list(rows[0].keys())
        c.cursor().executemany(
            f"INSERT INTO reviews ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})",
            [tuple(r[k] for k in cols) for r in rows],
        )
        c.commit()
        yield c
