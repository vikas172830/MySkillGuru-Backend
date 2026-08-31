import uuid

import fakeredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings
from app.core.security import hash_password
from app.models.user import create_user_document

# Dedicated, hardcoded, never-production database name. The real .env points
# MONGODB_URI at the same server the Flask backend uses in production
# (settings.DB_NAME == "nexus") — that's the only Mongo reachable in this dev
# environment, so tests reuse the same server but a completely separate
# database. This name must never be one of the real databases on that server.
TEST_DB_NAME = "lms_evaluation_test"
_PROTECTED_DB_NAMES = {"nexus", "nexus_iot", "admin", "config", "local", "lms_evaluation"}

if TEST_DB_NAME in _PROTECTED_DB_NAMES:
    raise RuntimeError(f"Refusing to run tests against a protected database name: {TEST_DB_NAME!r}")


@pytest_asyncio.fixture
async def test_db():
    """
    Function-scoped disposable Mongo database on the same server as the real
    app — recreated and dropped per test (not session-scoped) specifically
    to avoid pytest-asyncio event-loop-scope mismatches: fixtures and test
    functions both default to a function-scoped loop, and Motor clients
    can't be used across different loops. The app's own lifespan
    (connect_to_mongo/connect_to_redis) never runs during tests — httpx's
    ASGITransport doesn't send ASGI lifespan events — so this fixture
    directly populates app.db.mongodb's module globals itself; it's the
    only thing that ever sets them during a test run.
    """
    import app.db.mongodb as mongodb_module

    client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[TEST_DB_NAME]

    mongodb_module._client = client
    mongodb_module._db = db

    yield db

    # Hard guard, right before the destructive call — never drop anything
    # other than the disposable test database.
    assert db.name == TEST_DB_NAME
    assert db.name not in _PROTECTED_DB_NAMES
    await client.drop_database(TEST_DB_NAME)

    client.close()
    mongodb_module._client = None
    mongodb_module._db = None


@pytest_asyncio.fixture
async def fake_redis_server():
    server = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield server
    await server.aclose()


@pytest.fixture(autouse=True)
def _patch_redis(fake_redis_server):
    """
    app.services.job_store talks to Redis via app.core.redis_client.get_redis(),
    a plain module-level function (not a FastAPI dependency), so it can't be
    swapped with app.dependency_overrides — patch the module global directly.
    """
    import app.core.redis_client as redis_client_module

    redis_client_module._client = fake_redis_server
    yield
    redis_client_module._client = None


@pytest.fixture
def app():
    from app.main import app as fastapi_app

    return fastapi_app


@pytest_asyncio.fixture
async def client(app, test_db):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ============================================================
# AUTH HELPERS
#
# /register itself requires an authenticated caller for every role (ported
# 1:1 from Flask's @jwt_required() on the same route) — there is no public
# bootstrap endpoint. seed_superadmin_direct bypasses the API purely to
# create the first user (mirrors how the real system's first superadmin
# must already exist in the database); every other role in tests is created
# through the real /register endpoint using an already-authenticated caller,
# so the register/login flow itself stays under test.
# ============================================================

async def seed_superadmin_direct(test_db) -> tuple[str, str]:
    email = f"superadmin-{uuid.uuid4().hex[:10]}@test.local"
    password = "TestPass123!"
    user_doc = create_user_document(
        {"fullName": "Test Superadmin", "email": email, "role": 1, "is_active": True},
        hash_password(password),
    )
    await test_db["users"].insert_one(user_doc)
    return email, password


async def login(client: AsyncClient, email: str, password: str) -> None:
    resp = await client.post("/login", json={"email": email, "password": password})
    assert resp.status_code == 200, f"login failed for {email}: {resp.status_code} {resp.text}"


async def register(client: AsyncClient, **payload) -> dict:
    resp = await client.post("/register", json=payload)
    # NOTE: auth.py's register() returns the success dict directly (`return
    # body`) instead of `JSONResponse(status_code=code, ...)`, so despite each
    # _register_* handler computing `code = 201`, the actual HTTP status is
    # always 200. Pre-existing behavior, not a Phase 1 item — worth fixing
    # when auth.py gets its Pydantic/response-model retrofit in Phase 2.
    assert resp.status_code == 200, f"register failed for role={payload.get('role')}: {resp.status_code} {resp.text}"
    return resp.json()


@pytest_asyncio.fixture
async def superadmin_client(client: AsyncClient, test_db):
    email, password = await seed_superadmin_direct(test_db)
    await login(client, email, password)
    return client


@pytest_asyncio.fixture
async def client_factory(app):
    """Spin up additional independent authenticated sessions within a test
    (each httpx.AsyncClient has its own cookie jar) — needed whenever a test
    has to compare behavior across two different logged-in users."""
    clients = []

    async def _make() -> AsyncClient:
        transport = ASGITransport(app=app)
        c = AsyncClient(transport=transport, base_url="http://test")
        clients.append(c)
        return c

    yield _make

    for c in clients:
        await c.aclose()
