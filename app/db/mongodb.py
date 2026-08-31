import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import settings

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


def connect_to_mongo() -> None:
    global _client, _db
    _client = AsyncIOMotorClient(
        settings.MONGODB_URI,
        # Slightly more generous than the main LMS app's client (5000/5000/20000)
        # — this project's dev environment sees occasional slower round-trips to
        # the shared remote Mongo host on the very first request of a new
        # connection, and the shorter windows were tripping false timeouts.
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=30000,
    )
    _db = _client[settings.DB_NAME]
    logging.info("MongoDB client created for database '%s'", settings.DB_NAME)


async def ping_mongo() -> None:
    if _client is None:
        raise RuntimeError("Mongo client not initialized — call connect_to_mongo() first")
    await _client.admin.command("ping")
    logging.info("MongoDB connection verified")


def close_mongo_connection() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


def get_database() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Mongo client not initialized — call connect_to_mongo() first")
    return _db
