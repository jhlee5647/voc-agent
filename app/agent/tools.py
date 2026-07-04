"""에이전트 tools — 구조화 인자 → 파라미터화 SQL (자유 text-to-SQL 금지).

group_by/metric만 SQL 문장 구조에 들어가므로 화이트리스트로 검증하고,
나머지 값 인자는 전부 psycopg 파라미터 바인딩으로만 전달한다.
"""

import math
from collections.abc import Callable

EMBEDDING_MODEL = "text-embedding-3-small"

# group_by 키 → SQL 표현식 (화이트리스트)
_GROUP_EXPRS = {
    "brand": "brand_name",
    "category": "category_name",
    "sub_category": "sub_category_name",
    "grade": "grade",
    "month": "to_char(create_date AT TIME ZONE 'Asia/Seoul', 'YYYY-MM')",
}
_METRICS = ("count", "avg_grade", "neg_rate")

# 지표 비교(평균/비율)는 소표본 그룹 제외, 단순 건수는 전 그룹 노출
_MIN_N_METRICS = ("avg_grade", "neg_rate")


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """이항 비율의 Wilson score interval 하한 (95% 기본). n=0이면 0."""
    if n == 0:
        return 0.0
    p = successes / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / (1 + z * z / n))


def _build_where(
    brand_name,
    category_name,
    sub_category_name,
    grade_min,
    grade_max,
    date_from,
    date_to,
    reviewer_sex,
) -> tuple[str, list]:
    conds, params = [], []
    for column, value in (
        ("brand_name = %s", brand_name),
        ("category_name = %s", category_name),
        ("sub_category_name = %s", sub_category_name),
        ("grade >= %s", grade_min),
        ("grade <= %s", grade_max),
        ("(create_date AT TIME ZONE 'Asia/Seoul')::date >= %s", date_from),
        ("(create_date AT TIME ZONE 'Asia/Seoul')::date <= %s", date_to),
        ("reviewer_sex = %s", reviewer_sex),
    ):
        if value is not None:
            conds.append(column)
            params.append(value)
    return (" WHERE " + " AND ".join(conds)) if conds else "", params


def aggregate_reviews(
    conn,
    *,
    brand_name: str | None = None,
    category_name: str | None = None,
    sub_category_name: str | None = None,
    grade_min: int | None = None,
    grade_max: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    reviewer_sex: str | None = None,
    group_by: str | None = None,
    metric: str = "count",
    min_n: int = 10,
) -> list[dict]:
    """정량 집계. group_by 시 metric 기준 내림차순(neg_rate는 Wilson LB) 정렬."""
    if group_by is not None and group_by not in _GROUP_EXPRS:
        raise ValueError(f"group_by는 {sorted(_GROUP_EXPRS)} 중 하나여야 함: {group_by!r}")
    if metric not in _METRICS:
        raise ValueError(f"metric은 {_METRICS} 중 하나여야 함: {metric!r}")

    where, params = _build_where(
        brand_name,
        category_name,
        sub_category_name,
        grade_min,
        grade_max,
        date_from,
        date_to,
        reviewer_sex,
    )
    select = "COUNT(*) AS n, AVG(grade)::float AS avg, COUNT(*) FILTER (WHERE grade <= 3) AS neg"

    if group_by is None:
        row = conn.execute(f"SELECT {select} FROM reviews{where}", params).fetchone()
        return [_to_result(None, row, metric)]

    expr = _GROUP_EXPRS[group_by]
    sql = f"SELECT {expr} AS grp, {select} FROM reviews{where} GROUP BY grp"
    rows = conn.execute(sql, params).fetchall()
    results = [_to_result(grp, stats, metric) for grp, *stats in rows]
    if metric in _MIN_N_METRICS:
        results = [r for r in results if r["n"] >= min_n]
    sort_key = {"count": "n", "avg_grade": "avg_grade", "neg_rate": "wilson_lb"}[metric]
    return sorted(results, key=lambda r: r[sort_key], reverse=True)


def embed_query(text: str) -> list[float]:
    """질의 텍스트를 임베딩 (기본 임베더)."""
    from openai import OpenAI

    resp = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


# 검색 결과에 포함할 컬럼 (ROADMAP: satisfaction/goods_option/reviewer_height·weight 포함)
_SEARCH_FIELDS = (
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
)


def search_reviews(
    conn,
    query: str,
    *,
    brand_name: str | None = None,
    category_name: str | None = None,
    sub_category_name: str | None = None,
    grade_min: int | None = None,
    grade_max: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    reviewer_sex: str | None = None,
    top_k: int = 10,
    embedder: Callable[[str], list[float]] | None = None,
) -> list[dict]:
    """시맨틱 검색 + 메타필터. 코사인 유사도 내림차순 top_k."""
    vector = "[" + ",".join(map(str, (embedder or embed_query)(query))) + "]"
    where, params = _build_where(
        brand_name,
        category_name,
        sub_category_name,
        grade_min,
        grade_max,
        date_from,
        date_to,
        reviewer_sex,
    )
    where = f"{where} AND " if where else " WHERE "
    where += "embedding IS NOT NULL"
    sql = (
        f"SELECT {', '.join(_SEARCH_FIELDS)}, 1 - (embedding <=> %s::vector) AS similarity "
        f"FROM reviews{where} ORDER BY embedding <=> %s::vector LIMIT %s"
    )
    conn.execute("SET ivfflat.probes = 10")
    rows = conn.execute(sql, [vector, *params, vector, top_k]).fetchall()
    columns = (*_SEARCH_FIELDS, "similarity")
    return [dict(zip(columns, row)) for row in rows]


# list_metadata kind → 컬럼 (화이트리스트)
_METADATA_COLUMNS = {
    "brand": "brand_name",
    "category": "category_name",
    "sub_category": "sub_category_name",
}


def list_metadata(
    conn,
    kind: str,
    search: str | None = None,
    limit: int = 50,
) -> list[str]:
    """브랜드/카테고리/중분류 명칭 목록 (리뷰 수 내림차순, ILIKE 부분일치)."""
    if kind not in _METADATA_COLUMNS:
        raise ValueError(f"kind는 {sorted(_METADATA_COLUMNS)} 중 하나여야 함: {kind!r}")
    column = _METADATA_COLUMNS[kind]
    conds, params = [f"{column} IS NOT NULL"], []
    if search is not None:
        conds.append(f"{column} ILIKE %s")
        params.append(f"%{search}%")
    sql = (
        f"SELECT {column} FROM reviews WHERE {' AND '.join(conds)} "
        f"GROUP BY {column} ORDER BY COUNT(*) DESC LIMIT %s"
    )
    return [name for (name,) in conn.execute(sql, [*params, limit]).fetchall()]


def _to_result(group, stats, metric: str) -> dict:
    n, avg, neg = stats
    result = {} if group is None else {"group": group}
    result["n"] = n
    if metric == "avg_grade":
        result["avg_grade"] = avg
    elif metric == "neg_rate":
        result["neg_rate"] = neg / n if n else 0.0
        result["wilson_lb"] = wilson_lower_bound(neg, n)
    return result
