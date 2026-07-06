"""make_tools 래퍼 테스트 (게이트 2 승인 6케이스).

plain 함수의 집계/검색 로직은 기존 테스트가 검증하므로, 여기서는 래퍼 계층
(tool 변환, conn/embedder 바인딩, JSON 직렬화)만 검증한다.
"""

import json

import pytest

from app.agent.tools import make_tools

DIM = 1536
QUERY_VEC = [1.0, 0.0] + [0.0] * (DIM - 2)


def fake_embedder(text: str) -> list[float]:
    return QUERY_VEC


@pytest.fixture
def tools(conn):
    """review_id 1에만 쿼리와 동일한 벡터를 심고, 가짜 임베더를 바인딩한 tool dict."""
    vec = "[" + ",".join(map(str, QUERY_VEC)) + "]"
    conn.execute("UPDATE reviews SET embedding = %s::vector WHERE review_id = 1", (vec,))
    conn.commit()
    return {t.name: t for t in make_tools(conn, embedder=fake_embedder)}


def test_returns_three_named_tools(conn):
    names = [t.name for t in make_tools(conn)]
    assert sorted(names) == ["aggregate_reviews", "list_metadata", "search_reviews"]


def test_schema_hides_conn_and_embedder(tools):
    for tool in tools.values():
        assert "conn" not in tool.args
        assert "embedder" not in tool.args


def test_aggregate_invoke_returns_json(tools):
    raw = tools["aggregate_reviews"].invoke({"group_by": "brand", "metric": "count"})
    assert isinstance(raw, str)
    results = json.loads(raw)
    assert {"group": "브랜드B", "n": 15} in results


def test_search_invoke_uses_bound_embedder(tools):
    raw = tools["search_reviews"].invoke({"query": "기장 불만"})
    results = json.loads(raw)
    assert results[0]["review_id"] == 1
    assert results[0]["content"] == "리뷰 1"


def test_json_is_korean_readable_and_serializes_dates(tools):
    raw = tools["search_reviews"].invoke({"query": "기장 불만"})
    assert "리뷰 1" in raw  # ensure_ascii=False — \uXXXX 이스케이프 없음
    results = json.loads(raw)
    assert isinstance(results[0]["create_date"], str)  # datetime → 문자열 직렬화


def test_list_metadata_invoke(tools):
    raw = tools["list_metadata"].invoke({"kind": "brand", "search": "브랜드"})
    assert "브랜드A" in [r["name"] for r in json.loads(raw)]
