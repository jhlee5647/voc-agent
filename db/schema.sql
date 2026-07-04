-- VoC 분석용 PostgreSQL 스키마 (ROADMAP.md Phase 0)
-- 멱등: docker-entrypoint-initdb.d(최초 1회) 및 psql -f 재실행 안전

CREATE EXTENSION IF NOT EXISTS vector;

-- products: SQLite products 테이블 그대로 (보조 테이블)
CREATE TABLE IF NOT EXISTS products (
    goods_no          BIGINT PRIMARY KEY,
    goods_name        TEXT,
    brand_name        TEXT,
    category_code     TEXT,
    category_name     TEXT,
    gender            TEXT,
    site_review_count INTEGER,
    backfilled        INTEGER DEFAULT 0,
    review_count      INTEGER DEFAULT 0,
    last_crawled_at   TEXT,
    sub_category_code TEXT,
    sub_category_name TEXT
);

-- reviews: SQLite reviews 컬럼 전체
--   + category_name/sub_category_name 비정규화 복사 (products에서 goods_no 조인)
--   + satisfaction/repurchase JSONB, create_date TIMESTAMPTZ, embedding vector(1536)
CREATE TABLE IF NOT EXISTS reviews (
    review_id         BIGINT PRIMARY KEY,
    goods_no          BIGINT NOT NULL,
    goods_name        TEXT,
    brand_name        TEXT,
    brand_en          TEXT,
    goods_sex         TEXT,
    grade             SMALLINT,
    content           TEXT,
    review_type       TEXT,
    review_type_name  TEXT,
    review_sub_type   TEXT,
    goods_option      TEXT,
    like_count        INTEGER,
    has_images        BOOLEAN,
    is_staff          BOOLEAN,
    user_nickname     TEXT,
    user_level        INTEGER,
    reviewer_sex      TEXT,
    reviewer_height   INTEGER,
    reviewer_weight   INTEGER,
    skin_type         TEXT,
    satisfaction      JSONB,
    repurchase        JSONB,
    create_date       TIMESTAMPTZ,
    image_urls        TEXT,
    collected_at      TEXT,
    category_name     TEXT,
    sub_category_name TEXT,
    embedding         vector(1536)
);

-- 시맨틱 검색용 HNSW 인덱스 (코사인 거리)
-- 메타필터 병용 시 recall 저하 가능 → 필요 시 hnsw.ef_search 상향 (ROADMAP.md 참조)
CREATE INDEX IF NOT EXISTS idx_reviews_embedding_hnsw
    ON reviews USING hnsw (embedding vector_cosine_ops);
