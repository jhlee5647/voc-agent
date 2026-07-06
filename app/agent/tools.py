"""에이전트 tools — 구조화 인자 → 파라미터화 SQL (자유 text-to-SQL 금지).

group_by/metric만 SQL 문장 구조에 들어가므로 화이트리스트로 검증하고,
나머지 값 인자는 전부 psycopg 파라미터 바인딩으로만 전달한다.
"""

import json
import math
from collections.abc import Callable

from langchain_core.tools import BaseTool, tool

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
    if group_by == "month":  # 추이 질문 대비 — 월 그룹만 metric 무관 시계열 정렬
        return sorted(results, key=lambda r: r["group"])
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
    # 인덱스 경로에서 메타필터로 후보가 고갈돼도 LIMIT까지 스캔 (pgvector 0.8+).
    # 346k 실측: recall 동등 + 2~4배 빠름, 필터 강제 시 결과 누락(10건 중 5건) 방지
    conn.execute("SET ivfflat.iterative_scan = relaxed_order")
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
) -> list[dict]:
    """브랜드/카테고리/중분류 명칭·리뷰 수 목록 (리뷰 수 내림차순, ILIKE 부분일치).

    리뷰 수(n)를 함께 반환해 "리뷰가 가장 많은 X" 판단을 LLM이 순서 추측이 아닌
    명시적 수치로 하게 한다 (eval multi-03 실측 대응).
    """
    if kind not in _METADATA_COLUMNS:
        raise ValueError(f"kind는 {sorted(_METADATA_COLUMNS)} 중 하나여야 함: {kind!r}")
    column = _METADATA_COLUMNS[kind]
    conds, params = [f"{column} IS NOT NULL"], []
    if search is not None:
        conds.append(f"{column} ILIKE %s")
        params.append(f"%{search}%")
    sql = (
        f"SELECT {column}, COUNT(*) AS n FROM reviews WHERE {' AND '.join(conds)} "
        f"GROUP BY {column} ORDER BY n DESC LIMIT %s"
    )
    rows = conn.execute(sql, [*params, limit]).fetchall()
    return [{"name": name, "n": n} for name, n in rows]


def make_tools(conn, embedder: Callable[[str], list[float]] | None = None) -> list[BaseTool]:
    """plain 함수 3개를 conn/embedder가 클로저에 바인딩된 LangChain tool로 변환.

    반환값은 JSON 문자열(LLM 컨텍스트에 그대로 들어감) — 한글 보존, datetime은 문자열화.
    """

    def _dumps(result) -> str:
        return json.dumps(result, ensure_ascii=False, default=str)

    @tool("aggregate_reviews")
    def aggregate_tool(
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
    ) -> str:
        """리뷰를 조건으로 필터링해 정량 집계한다 (건수/평균 평점/부정률 질문에 사용).

        metric: count(리뷰 수) | avg_grade(평균 평점) | neg_rate(grade<=3 부정률).
        group_by: brand | category | sub_category | grade | month — 지정 시 그룹별 결과를
        metric 내림차순으로 반환. neg_rate 순위는 Wilson lower bound 기준으로 정렬되며
        표본 n < min_n인 그룹은 제외된다. 날짜는 YYYY-MM-DD, reviewer_sex는 '남성'/'여성'.
        """
        return _dumps(
            aggregate_reviews(
                conn,
                brand_name=brand_name,
                category_name=category_name,
                sub_category_name=sub_category_name,
                grade_min=grade_min,
                grade_max=grade_max,
                date_from=date_from,
                date_to=date_to,
                reviewer_sex=reviewer_sex,
                group_by=group_by,
                metric=metric,
                min_n=min_n,
            )
        )

    @tool("search_reviews")
    def search_tool(
        query: str,
        brand_name: str | None = None,
        category_name: str | None = None,
        sub_category_name: str | None = None,
        grade_min: int | None = None,
        grade_max: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        reviewer_sex: str | None = None,
        top_k: int = 10,
    ) -> str:
        """리뷰 본문을 시맨틱 검색한다 (내용/의견/이유 등 정성 질문에 사용).

        query는 찾으려는 내용의 자연어 서술. 메타필터(브랜드/카테고리/중분류/평점/
        날짜/성별)로 좁힐 수 있다. 불만·단점을 찾을 때는 반드시 grade_max=3을 설정할 것
        (평점 5점이 85.7%라 필터 없이는 긍정 리뷰만 나온다). 날짜는 YYYY-MM-DD,
        reviewer_sex는 '남성'/'여성'.
        """
        return _dumps(
            search_reviews(
                conn,
                query,
                brand_name=brand_name,
                category_name=category_name,
                sub_category_name=sub_category_name,
                grade_min=grade_min,
                grade_max=grade_max,
                date_from=date_from,
                date_to=date_to,
                reviewer_sex=reviewer_sex,
                top_k=top_k,
                embedder=embedder,
            )
        )

    @tool("list_metadata")
    def metadata_tool(kind: str, search: str | None = None, limit: int = 50) -> str:
        """브랜드/카테고리/중분류의 명칭과 리뷰 수(n)를 리뷰 수 내림차순으로 반환한다.

        kind: brand | category | sub_category. search로 부분일치(대소문자 무시) 검색.
        다른 tool에 브랜드/카테고리/중분류 필터를 걸기 전에 먼저 이 tool로 정확한
        명칭을 확인할 것. "리뷰가 가장 많은 X" 판단에는 반환된 n을 그대로 사용할 것.
        """
        return _dumps(list_metadata(conn, kind, search=search, limit=limit))

    return [aggregate_tool, search_tool, metadata_tool]


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
