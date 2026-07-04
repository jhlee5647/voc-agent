"""search_reviews 테스트 (게이트 2 승인 9케이스).

가짜 임베더 전략: 방향이 명확한 수제 벡터를 시드 리뷰 3건에 심고,
쿼리 벡터를 고정해 코사인 유사도(1.0 / 0.6 / 0.0)가 손으로 계산되게 한다.
"""

import pytest

from app.agent.tools import search_reviews

DIM = 1536


def _vec(x: float, y: float) -> str:
    """앞 2개 차원만 쓰는 1536차원 벡터 리터럴."""
    return "[" + ",".join([str(x), str(y)] + ["0"] * (DIM - 2)) + "]"


# 쿼리 벡터 = e0. 심은 벡터와의 코사인 유사도가 주석의 값이 된다.
QUERY_VEC = [1.0, 0.0] + [0.0] * (DIM - 2)
_SEED_VECTORS = {
    1: _vec(1.0, 0.0),  # 브랜드A, 5점, 2026-05-05 → 유사도 1.0
    10: _vec(0.6, 0.8),  # 브랜드A, 2점, 2026-05-14 → 유사도 0.6
    13: _vec(0.0, 1.0),  # 브랜드B, 5점, 2026-06-01 → 유사도 0.0
}


def fake_embedder(text: str) -> list[float]:
    return QUERY_VEC


@pytest.fixture
def vconn(conn):
    """시드 32건 중 3건에만 수제 벡터 UPDATE (나머지는 embedding NULL)."""
    for review_id, vec in _SEED_VECTORS.items():
        conn.execute(
            "UPDATE reviews SET embedding = %s::vector WHERE review_id = %s", (vec, review_id)
        )
    conn.commit()
    return conn


def _ids(results):
    return [r["review_id"] for r in results]


def test_sorted_by_similarity_desc(vconn):
    results = search_reviews(vconn, "기장 불만", embedder=fake_embedder)
    assert _ids(results) == [1, 10, 13]


def test_top_k_limits_results(vconn):
    results = search_reviews(vconn, "기장 불만", top_k=2, embedder=fake_embedder)
    assert _ids(results) == [1, 10]


def test_brand_filter_overrides_similarity(vconn):
    # 브랜드A(id 1)가 더 유사해도 브랜드B 것만 반환
    results = search_reviews(vconn, "기장 불만", brand_name="브랜드B", embedder=fake_embedder)
    assert _ids(results) == [13]


def test_grade_max_filter(vconn):
    # 불만 검색 시나리오: grade<=3인 임베딩 행은 id 10(2점)뿐
    results = search_reviews(vconn, "기장 불만", grade_max=3, embedder=fake_embedder)
    assert _ids(results) == [10]


def test_date_filter(vconn):
    results = search_reviews(vconn, "기장 불만", date_from="2026-06-01", embedder=fake_embedder)
    assert _ids(results) == [13]


def test_null_embedding_rows_excluded(vconn):
    # 시드 32건 중 벡터를 심은 3건만 검색 대상
    results = search_reviews(vconn, "기장 불만", top_k=50, embedder=fake_embedder)
    assert len(results) == 3


def test_result_fields_contract(vconn):
    result = search_reviews(vconn, "기장 불만", top_k=1, embedder=fake_embedder)[0]
    expected_keys = {
        "review_id",
        "content",
        "brand_name",
        "category_name",
        "sub_category_name",
        "grade",
        "satisfaction",
        "goods_option",
        "reviewer_height",
        "reviewer_weight",
        "create_date",
        "similarity",
    }
    assert expected_keys <= set(result.keys())


def test_no_match_returns_empty_list(vconn):
    results = search_reviews(vconn, "기장 불만", brand_name="없는브랜드", embedder=fake_embedder)
    assert results == []


def test_similarity_values_exact(vconn):
    results = search_reviews(vconn, "기장 불만", embedder=fake_embedder)
    sims = [r["similarity"] for r in results]
    assert sims[0] == pytest.approx(1.0, abs=1e-6)
    assert sims[1] == pytest.approx(0.6, abs=1e-6)
    assert sims[2] == pytest.approx(0.0, abs=1e-6)
