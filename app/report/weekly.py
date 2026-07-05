"""주간 VoC 리포트 빌더 — 기존 tools(aggregate_reviews/search_reviews) 재사용.

기준일(base_date)은 파라미터, 기본값은 DB 내 최신 create_date.
정적 DB라 "실행 시점 오늘" 기준이면 리뷰 0건이 되기 때문 (ROADMAP 확정).
"""

from datetime import date, timedelta

from app.agent.tools import aggregate_reviews, search_reviews

EXCERPT_QUERY = "품질 사이즈 배송 불만"
TOP_BRANDS = 3
EXCERPT_COUNT = 3
EXCERPT_MAX_CHARS = 100


def latest_review_date(conn) -> date:
    """DB 최신 create_date의 KST 날짜."""
    row = conn.execute(
        "SELECT MAX((create_date AT TIME ZONE 'Asia/Seoul')::date) FROM reviews"
    ).fetchone()
    return row[0]


def _stats(conn, date_from: date, date_to: date) -> dict:
    """구간 리뷰 수 + 평균 평점 (n=0이면 avg_grade None)."""
    return aggregate_reviews(
        conn, date_from=str(date_from), date_to=str(date_to), metric="avg_grade"
    )[0]


def build_weekly_report(
    conn,
    base_date: date | None = None,
    min_n: int = 10,
    embedder=None,
) -> str:
    """기준일 포함 지난 7일 리포트 텍스트(Slack 마크다운) 생성."""
    base = base_date or latest_review_date(conn)
    this_from, this_to = base - timedelta(days=6), base
    prev_from, prev_to = base - timedelta(days=13), base - timedelta(days=7)

    header = f"📊 주간 VoC 리포트 ({this_from} ~ {this_to})"
    this_week = _stats(conn, this_from, this_to)
    if this_week["n"] == 0:
        return f"{header}\n이번 주 리뷰가 없습니다."
    prev_week = _stats(conn, prev_from, prev_to)

    prev_avg = f"{prev_week['avg_grade']:.2f}" if prev_week["n"] else "-"
    lines = [
        header,
        "",
        "*1. 주간 요약*",
        f"- 리뷰 수: {this_week['n']}건 (전주 {prev_week['n']}건)",
        f"- 평균 평점: {this_week['avg_grade']:.2f} (전주 {prev_avg})",
        "",
        f"*2. 부정률 상위 브랜드* (3점 이하 비율, Wilson LB 정렬, n≥{min_n})",
    ]

    brands = aggregate_reviews(
        conn,
        date_from=str(this_from),
        date_to=str(this_to),
        group_by="brand",
        metric="neg_rate",
        min_n=min_n,
    )[:TOP_BRANDS]
    if not brands:
        lines.append("- 표본이 충분한 브랜드가 없습니다.")
    for brand in brands:
        prev_brand = aggregate_reviews(
            conn,
            brand_name=brand["group"],
            date_from=str(prev_from),
            date_to=str(prev_to),
            metric="neg_rate",
        )[0]
        prev_rate = f"{prev_brand['neg_rate']:.1%}" if prev_brand["n"] else "신규"
        lines.append(
            f"- {brand['group']}: {brand['neg_rate']:.1%} (n={brand['n']}, 전주 {prev_rate})"
        )

    lines += ["", "*3. 부정 리뷰 발췌*"]
    excerpts = search_reviews(
        conn,
        EXCERPT_QUERY,
        grade_max=3,
        date_from=str(this_from),
        date_to=str(this_to),
        top_k=EXCERPT_COUNT,
        embedder=embedder,
    )
    if not excerpts:
        lines.append("- 이번 주 부정 리뷰 발췌가 없습니다.")
    for review in excerpts:
        content = review["content"][:EXCERPT_MAX_CHARS]
        lines.append(f"- [{review['brand_name']}/{review['grade']}점] {content}")

    return "\n".join(lines)
