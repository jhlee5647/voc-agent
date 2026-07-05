"""aggregate_reviews + wilson_lower_bound 테스트 (게이트 2 승인 16케이스)."""

import pytest

from app.agent.tools import aggregate_reviews, wilson_lower_bound

# ── wilson_lower_bound (순수 함수) ──────────────────────────────────


def test_wilson_zero_n_returns_zero():
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_zero_successes_returns_zero():
    assert wilson_lower_bound(0, 10) == pytest.approx(0.0, abs=1e-9)


def test_wilson_all_successes_below_one():
    assert wilson_lower_bound(10, 10) < 1.0


def test_wilson_larger_sample_higher_bound():
    assert wilson_lower_bound(5, 10) < wilson_lower_bound(50, 100)


def test_wilson_known_value():
    assert wilson_lower_bound(3, 10) == pytest.approx(0.1078, abs=1e-3)


# ── aggregate_reviews (테스트 DB) ───────────────────────────────────


def test_count_no_filters(conn):
    assert aggregate_reviews(conn) == [{"n": 32}]


def test_brand_filter(conn):
    assert aggregate_reviews(conn, brand_name="브랜드A") == [{"n": 12}]


def test_grade_range_filter(conn):
    assert aggregate_reviews(conn, grade_min=1, grade_max=3) == [{"n": 6}]


def test_date_filter_boundaries_inclusive(conn):
    # 브랜드B의 첫(6/1)·마지막(6/15) 리뷰가 경계일에 정확히 걸림 → 포함돼야 15
    result = aggregate_reviews(conn, date_from="2026-06-01", date_to="2026-06-15")
    assert result == [{"n": 15}]


def test_reviewer_sex_filter(conn):
    assert aggregate_reviews(conn, reviewer_sex="F") == [{"n": 9}]


def test_group_by_brand_count(conn):
    result = aggregate_reviews(conn, group_by="brand")
    assert result == [
        {"group": "브랜드B", "n": 15},
        {"group": "브랜드A", "n": 12},
        {"group": "브랜드C", "n": 5},
    ]


def test_avg_grade(conn):
    result = aggregate_reviews(conn, brand_name="브랜드B", metric="avg_grade")
    assert result[0]["n"] == 15
    assert result[0]["avg_grade"] == pytest.approx(73 / 15, abs=1e-6)


def test_neg_rate_grouped_wilson_sorted_min_n_excludes(conn):
    result = aggregate_reviews(conn, group_by="brand", metric="neg_rate", min_n=10)
    groups = [r["group"] for r in result]
    assert groups == ["브랜드A", "브랜드B"]  # wilson_lb 내림차순, C(5건) 제외
    a, b = result
    assert a["n"] == 12 and a["neg_rate"] == pytest.approx(3 / 12)
    assert a["wilson_lb"] == pytest.approx(wilson_lower_bound(3, 12), abs=1e-9)
    assert b["n"] == 15 and b["neg_rate"] == pytest.approx(1 / 15)
    assert b["wilson_lb"] == pytest.approx(wilson_lower_bound(1, 15), abs=1e-9)


def test_neg_rate_lower_min_n_includes_small_group(conn):
    result = aggregate_reviews(conn, group_by="brand", metric="neg_rate", min_n=3)
    groups = [r["group"] for r in result]
    assert groups == ["브랜드C", "브랜드A", "브랜드B"]  # C: 2/5=0.4, wilson 최상위


def test_group_by_month(conn):
    result = aggregate_reviews(conn, group_by="month")
    assert result == [
        {"group": "2026-05", "n": 17},
        {"group": "2026-06", "n": 15},
    ]


def test_month_groups_sorted_chronologically(conn):
    # 5/20 이후로 좁히면 5월(5건) < 6월(15건) — 건수 내림차순이라면 [06, 05]로 뒤집힐 상황.
    # 월 그룹은 metric과 무관하게 시계열(오름차순)이어야 추이 질문에 안전하다.
    result = aggregate_reviews(conn, date_from="2026-05-20", group_by="month")
    assert result == [
        {"group": "2026-05", "n": 5},
        {"group": "2026-06", "n": 15},
    ]


def test_invalid_group_by_and_metric_rejected(conn):
    with pytest.raises(ValueError):
        aggregate_reviews(conn, group_by="asdf")
    with pytest.raises(ValueError):
        aggregate_reviews(conn, metric="asdf")
