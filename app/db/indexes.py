# ============================================================
# MongoDB index creation, run once on app startup (app/main.py lifespan).
#
# Without these, every lookup below was a full collection scan. All indexes
# are NON-unique on purpose: create_index is idempotent (same spec = no-op)
# and can't fail on existing data the way a unique index over accidental
# duplicates would. Each create is isolated in try/except so one failure
# (e.g. a differently-named legacy index of the same shape) never blocks
# startup. aiUsageEvents indexes live in app/services/ai_usage.py.
# ============================================================
import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger("db.indexes")

# collection -> list of key specs (each spec is a list of (field, direction))
_INDEXES: dict[str, list[list[tuple[str, int]]]] = {
    "users": [
        [("email", 1)],
        [("is_deleted", 1), ("role", 1)],
    ],
    "selfLearnerRoadmaps": [
        [("user_id", 1), ("created_at", -1)],
        [("user_id", 1), ("active", 1)],
    ],
    "mockTests": [
        [("student_id", 1), ("created_at", -1)],
    ],
    "testAttempts": [
        [("student_id", 1)],
        [("test_id", 1), ("student_id", 1)],
    ],
}


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    created = 0
    for collection, specs in _INDEXES.items():
        for spec in specs:
            try:
                await db[collection].create_index(spec)
                created += 1
            except Exception as exc:  # noqa: BLE001 - one bad index must not block startup
                logger.warning("ensure_indexes: %s %s failed: %s", collection, spec, exc)
    logger.info("ensure_indexes: %d indexes ensured across %d collections", created, len(_INDEXES))
