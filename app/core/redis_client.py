import redis.asyncio as redis_asyncio

from app.core.config import settings

_client: redis_asyncio.Redis | None = None


def connect_to_redis() -> None:
    global _client
    _client = redis_asyncio.from_url(settings.REDIS_URL, decode_responses=True)


async def close_redis_connection() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None


def get_redis() -> redis_asyncio.Redis:
    if _client is None:
        raise RuntimeError("Redis client not initialized — call connect_to_redis() first")
    return _client
