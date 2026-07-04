"""PostgreSQL 연결 헬퍼."""

import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector


def get_connection(url: str | None = None) -> psycopg.Connection:
    load_dotenv()
    conn = psycopg.connect(url or os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn
