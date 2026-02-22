"""
Async MySQL connection pool with company_id isolation.
"""
import aiomysql
from loguru import logger
from typing import Optional
from app.core.config import settings


class DatabasePool:
    def __init__(self):
        self._pool: Optional[aiomysql.Pool] = None

    async def connect(self):
        self._pool = await aiomysql.create_pool(
            host=settings.MYSQL_HOST, port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER, password=settings.MYSQL_PASSWORD,
            db=settings.MYSQL_DATABASE, minsize=2, maxsize=settings.MYSQL_POOL_SIZE,
            pool_recycle=3600, autocommit=True, charset="utf8mb4",
            cursorclass=aiomysql.DictCursor,
        )
        logger.info(f"MySQL connected: {settings.MYSQL_HOST}/{settings.MYSQL_DATABASE}")

    async def disconnect(self):
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()

    async def execute(self, query: str, params: tuple = ()) -> int:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return cur.rowcount

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchone()

    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchall()

    async def insert_get_id(self, query: str, params: tuple = ()) -> int:
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return cur.lastrowid


db = DatabasePool()
