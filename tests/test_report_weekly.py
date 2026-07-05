"""weekly 리포트 빌더 테스트 (게이트 2 승인 8케이스).

시드 기준 주간 수치(결정적):
  브랜드A 5/05~5/16 (5점9 + 2,2,1) / 브랜드B 6/01~6/15 (5점14 + 3점1) / 브랜드C 5/20~5/24
  → base 6/15: 이번 주(6/09~15) 7건 avg 4.71, 전주(6/02~08) 7건 avg 5.00
  → base 5/16: 이번 주(5/10~16) 브랜드A 7건 중 부정 3건(42.9%), 전주 부정 0%
"""

from datetime import date

from app.report.weekly import build_weekly_report, latest_review_date

DIM = 1536
QUERY_VEC = [1.0, 0.0] + [0.0] * (DIM - 2)


def fake_embedder(text):
    return QUERY_VEC


def test_latest_review_date(conn):
    assert latest_review_date(conn) == date(2026, 6, 15)


def test_default_base_date_is_latest(conn):
    report = build_weekly_report(conn, embedder=fake_embedder)
    assert "2026-06-09" in report
    assert "2026-06-15" in report


def test_summary_this_week_vs_prev(conn):
    report = build_weekly_report(conn, base_date=date(2026, 6, 15), embedder=fake_embedder)
    assert "7건 (전주 7건)" in report
    assert "4.71" in report
    assert "5.00" in report


def test_prev_week_empty_handled(conn):
    report = build_weekly_report(conn, base_date=date(2026, 6, 7), embedder=fake_embedder)
    assert "7건 (전주 0건)" in report


def test_negative_brands_wilson_with_prev_rate(conn):
    report = build_weekly_report(conn, base_date=date(2026, 5, 16), min_n=3, embedder=fake_embedder)
    assert "브랜드A" in report
    assert "42.9%" in report
    assert "전주 0.0%" in report


def test_negative_brands_min_n_default_message(conn):
    report = build_weekly_report(conn, base_date=date(2026, 5, 16), embedder=fake_embedder)
    assert "충분한 브랜드가 없습니다" in report


def test_excerpts_from_low_grade_reviews(conn):
    vec = "[" + ",".join(map(str, QUERY_VEC)) + "]"
    conn.execute("UPDATE reviews SET embedding = %s::vector WHERE review_id = 10", (vec,))
    conn.commit()
    report = build_weekly_report(conn, base_date=date(2026, 5, 16), embedder=fake_embedder)
    assert "리뷰 10" in report  # 05-14, 브랜드A, 2점 — 이번 주 구간의 grade<=3 발췌


def test_empty_week_message(conn):
    report = build_weekly_report(conn, base_date=date(2026, 7, 30), embedder=fake_embedder)
    assert "리뷰가 없습니다" in report
