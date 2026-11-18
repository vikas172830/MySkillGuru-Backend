"""SSRF guard (app/utils/net.py) and startup index creation
(app/db/indexes.py), ported from the LMS."""

import pytest

from app.db.indexes import _INDEXES, ensure_indexes
from app.utils.net import SsrfError, assert_url_allowed


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:6379/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "http:///nohost",
    ],
)
def test_internal_and_non_http_urls_are_blocked(url):
    with pytest.raises(SsrfError):
        assert_url_allowed(url)


def test_allowlist_blocks_unlisted_host(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "OUTBOUND_FETCH_ALLOWED_HOSTS", "cdn.example.com")
    with pytest.raises(SsrfError):
        assert_url_allowed("https://evil.example.org/file.pdf")


async def test_ensure_indexes_creates_expected_indexes(test_db):
    await ensure_indexes(test_db)
    for collection, specs in _INDEXES.items():
        info = await test_db[collection].index_information()
        existing = {tuple(v["key"]) for v in info.values()}
        for spec in specs:
            assert tuple(spec) in existing, f"{collection} missing index {spec}"


async def test_ensure_indexes_is_idempotent(test_db):
    await ensure_indexes(test_db)
    await ensure_indexes(test_db)
