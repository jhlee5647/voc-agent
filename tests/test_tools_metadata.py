"""list_metadata 테스트 (게이트 2 승인 8케이스)."""

import pytest

from app.agent.tools import list_metadata


def test_brand_list_sorted_by_review_count(conn):
    # 리뷰 수 내림차순: B(15) > A(12) > C(5)
    assert list_metadata(conn, "brand") == ["브랜드B", "브랜드A", "브랜드C"]


def test_category_list(conn):
    # 상의(12+5=17) > 바지(15)
    assert list_metadata(conn, "category") == ["상의", "바지"]


def test_sub_category_list(conn):
    # 데님(15) > 티셔츠(12) > 셔츠(5)
    assert list_metadata(conn, "sub_category") == ["데님", "티셔츠", "셔츠"]


def test_search_partial_match(conn):
    assert list_metadata(conn, "brand", search="랜드A") == ["브랜드A"]


def test_search_case_insensitive(conn):
    conn.execute(
        "INSERT INTO reviews (review_id, goods_no, brand_name, grade, content, create_date) "
        "VALUES (999, 100, 'EngBrand', 5, '리뷰', '2026-06-20T12:00:00+09:00')"
    )
    conn.commit()
    assert list_metadata(conn, "brand", search="engbrand") == ["EngBrand"]


def test_search_no_match_returns_empty(conn):
    assert list_metadata(conn, "brand", search="없는것") == []


def test_limit(conn):
    assert list_metadata(conn, "brand", limit=2) == ["브랜드B", "브랜드A"]


def test_invalid_kind_rejected(conn):
    with pytest.raises(ValueError):
        list_metadata(conn, "asdf")
