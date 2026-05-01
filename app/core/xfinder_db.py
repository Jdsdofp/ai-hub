"""
Pool MySQL dedicado ao banco xfinderdb_prod.
"""
import aiomysql
from loguru import logger
from typing import Optional
from app.core.config import settings


class XFinderPool:
    def __init__(self):
        self._pool: Optional[aiomysql.Pool] = None

    async def connect(self):
        try:
            self._pool = await aiomysql.create_pool(
                host=settings.XFINDER_MYSQL_HOST,
                port=settings.XFINDER_MYSQL_PORT,
                user=settings.XFINDER_MYSQL_USER,
                password=settings.XFINDER_MYSQL_PASSWORD,
                db=settings.XFINDER_MYSQL_DATABASE,
                minsize=1,
                maxsize=settings.XFINDER_MYSQL_POOL_SIZE,
                pool_recycle=3600,
                autocommit=True,
                charset="utf8mb4",
                cursorclass=aiomysql.DictCursor,
                connect_timeout=5,
            )
            logger.info(f"XFinder MySQL connected: {settings.XFINDER_MYSQL_HOST}/{settings.XFINDER_MYSQL_DATABASE}")
        except Exception as e:
            logger.warning(f"XFinder MySQL NOT connected: {e}. Company validation will be skipped.")
            self._pool = None

    async def disconnect(self):
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()

    @property
    def available(self) -> bool:
        return self._pool is not None

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        if not self._pool:
            return None
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchone()

    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        if not self._pool:
            return []
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchall()


xfinder_db = XFinderPool()
