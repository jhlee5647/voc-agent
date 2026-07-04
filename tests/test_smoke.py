"""스키마 스모크 테스트 — DDL 파일이 핵심 구조를 유지하는지 가드."""

from pathlib import Path

SCHEMA = (Path(__file__).parent.parent / "db" / "schema.sql").read_text(encoding="utf-8")


def test_schema_defines_core_tables():
    assert "CREATE TABLE IF NOT EXISTS reviews" in SCHEMA
    assert "CREATE TABLE IF NOT EXISTS products" in SCHEMA


def test_schema_has_embedding_and_hnsw_index():
    assert "embedding         vector(1536)" in SCHEMA
    assert "USING hnsw (embedding vector_cosine_ops)" in SCHEMA


def test_schema_has_denormalized_category_columns():
    reviews_ddl = SCHEMA.split("CREATE TABLE IF NOT EXISTS reviews")[1]
    assert "category_name" in reviews_ddl
    assert "sub_category_name" in reviews_ddl
