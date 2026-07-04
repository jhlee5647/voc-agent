"""미임베딩 리뷰 배치 임베딩 (text-embedding-3-small).

- embedding IS NULL인 행만 처리 → 재실행 안전
- 실행: uv run python -m ingest.embed
"""

import os

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector

MODEL = "text-embedding-3-small"
MAX_ROWS = 2000  # API 요청당 입력 상한(2048) 이내
MAX_CHARS = 100_000  # 요청당 토큰 한도(300k) 안전 마진


def fetch_batch(pg) -> list[tuple[int, str]]:
    """문자수 예산 내에서 미임베딩 행을 동적 배치로 가져온다."""
    rows = pg.execute(
        "SELECT review_id, content FROM reviews WHERE embedding IS NULL "
        "ORDER BY review_id LIMIT %s",
        (MAX_ROWS,),
    ).fetchall()
    batch, chars = [], 0
    for rid, content in rows:
        chars += len(content)
        if batch and chars > MAX_CHARS:
            break
        batch.append((rid, content))
    return batch


def main():
    load_dotenv()
    client = OpenAI()
    with psycopg.connect(os.environ["DATABASE_URL"]) as pg:
        register_vector(pg)
        remaining = pg.execute("SELECT COUNT(*) FROM reviews WHERE embedding IS NULL").fetchone()[0]
        done = 0
        while True:
            rows = fetch_batch(pg)
            if not rows:
                break
            resp = client.embeddings.create(model=MODEL, input=[content for _, content in rows])
            with pg.cursor() as cur:
                cur.executemany(
                    "UPDATE reviews SET embedding = %s WHERE review_id = %s",
                    [(e.embedding, rid) for e, (rid, _) in zip(resp.data, rows)],
                )
            pg.commit()
            done += len(rows)
            print(f"임베딩: {done}/{remaining}", flush=True)


if __name__ == "__main__":
    main()
