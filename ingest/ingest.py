"""SQLite(ms_reviews.db) → PostgreSQL 적재.

- products 전체 upsert 후, reviews를 products와 조인해 category를 비정규화 복사
- review_id PK upsert — ON CONFLICT DO UPDATE에서 embedding 컬럼 제외 (재실행 시 임베딩 보존)
- 실행: uv run python -m ingest.ingest
"""

import json
import os
import sqlite3
from datetime import datetime

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

BATCH = 5000

PRODUCT_COLS = [
    "goods_no",
    "goods_name",
    "brand_name",
    "category_code",
    "category_name",
    "gender",
    "site_review_count",
    "backfilled",
    "review_count",
    "last_crawled_at",
    "sub_category_code",
    "sub_category_name",
]

REVIEW_COLS = [
    "review_id",
    "goods_no",
    "goods_name",
    "brand_name",
    "brand_en",
    "goods_sex",
    "grade",
    "content",
    "review_type",
    "review_type_name",
    "review_sub_type",
    "goods_option",
    "like_count",
    "has_images",
    "is_staff",
    "user_nickname",
    "user_level",
    "reviewer_sex",
    "reviewer_height",
    "reviewer_weight",
    "skin_type",
    "satisfaction",
    "repurchase",
    "create_date",
    "image_urls",
    "collected_at",
    "category_name",
    "sub_category_name",
]

REVIEW_SELECT = """
    SELECT r.*, p.category_name AS category_name, p.sub_category_name AS sub_category_name
    FROM reviews r LEFT JOIN products p ON r.goods_no = p.goods_no
"""


def _upsert_sql(table: str, cols: list[str], pk: str) -> str:
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != pk)
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk}) DO UPDATE SET {updates}"
    )


def _to_jsonb(value: str | None) -> Jsonb | None:
    return Jsonb(json.loads(value)) if value else None


def _to_bool(value: int | None) -> bool | None:
    return None if value is None else bool(value)


def transform_review(row: sqlite3.Row) -> tuple:
    values = dict(zip(row.keys(), tuple(row)))
    values["has_images"] = _to_bool(values["has_images"])
    values["is_staff"] = _to_bool(values["is_staff"])
    values["satisfaction"] = _to_jsonb(values["satisfaction"])
    values["repurchase"] = _to_jsonb(values["repurchase"])
    values["create_date"] = datetime.fromisoformat(values["create_date"])
    return tuple(values[c] for c in REVIEW_COLS)


def copy_table(src, pg, select_sql, upsert_sql, transform, label):
    cur = src.execute(select_sql)
    total = 0
    while rows := cur.fetchmany(BATCH):
        with pg.cursor() as pg_cur:
            pg_cur.executemany(upsert_sql, [transform(r) for r in rows])
        pg.commit()
        total += len(rows)
        print(f"{label}: {total}건 적재")
    return total


def main():
    load_dotenv()
    src = sqlite3.connect(f"file:{os.environ['SOURCE_DB_PATH']}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    with psycopg.connect(os.environ["DATABASE_URL"]) as pg:
        copy_table(
            src,
            pg,
            f"SELECT {', '.join(PRODUCT_COLS)} FROM products",
            _upsert_sql("products", PRODUCT_COLS, "goods_no"),
            lambda r: tuple(r),
            "products",
        )
        copy_table(
            src,
            pg,
            REVIEW_SELECT,
            _upsert_sql("reviews", REVIEW_COLS, "review_id"),
            transform_review,
            "reviews",
        )


if __name__ == "__main__":
    main()
