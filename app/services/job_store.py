import json
from typing import Any, Dict, Optional

from app.core.redis_client import get_redis

JOB_TTL = 3600  # 1 hour — matches Flask's job stores


async def set_job(prefix: str, job_id: str, payload: Dict[str, Any]) -> None:
    await get_redis().setex(f"{prefix}{job_id}", JOB_TTL, json.dumps(payload))


async def update_job(prefix: str, job_id: str, patch: Dict[str, Any]) -> None:
    key = f"{prefix}{job_id}"
    redis = get_redis()
    existing = await redis.get(key)
    if existing:
        data = json.loads(existing)
        data.update(patch)
        await redis.setex(key, JOB_TTL, json.dumps(data))
    else:
        await redis.setex(key, JOB_TTL, json.dumps(patch))


async def get_job(prefix: str, job_id: str) -> Optional[Dict[str, Any]]:
    data = await get_redis().get(f"{prefix}{job_id}")
    return json.loads(data) if data else None
