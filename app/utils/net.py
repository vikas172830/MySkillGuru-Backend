# ============================================================
# SSRF guard for server-side outbound HTTP.
#
# Every place the backend fetches a URL that originated (even indirectly)
# from a client — an uploaded answer-script / question-paper URL, an <img
# src> inside AI-generated HTML — must go through safe_get() instead of
# requests.get(). Without this, a caller who controls a stored URL can make
# the server hit 169.254.169.254 (cloud metadata), 127.0.0.1 / 10.x /
# 192.168.x (internal admin services, Redis, Mongo, Qdrant), etc.
#
# Rules enforced:
#   - scheme must be http/https
#   - the host must resolve ONLY to public unicast IPs (no loopback,
#     link-local, private, reserved, multicast, unspecified)
#   - if an allow-list is configured (settings.outbound_fetch_allowed_hosts_list,
#     which always includes the ImageKit host), the target host must match it
#   - redirects are NOT followed (a 3xx to an internal address is the classic
#     bypass); a 3xx response is treated as an error
#   - the response body is size-capped
# ============================================================
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import requests

from app.core.config import settings

logger = logging.getLogger("app.utils.net")

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_BYTES = 30 * 1024 * 1024  # 30 MB — well above any real answer script / report


class SsrfError(ValueError):
    """Raised when a URL fails the outbound-fetch safety checks."""


def _ip_is_public(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _host_matches_allowlist(host: str) -> bool:
    allow = settings.outbound_fetch_allowed_hosts_list
    if not allow:
        return True  # no allow-list configured -> only the public-IP check applies
    host = host.lower()
    return any(host == h or host.endswith("." + h) for h in allow)


def assert_url_allowed(url: str) -> None:
    """Raise SsrfError unless `url` is safe for the server to fetch."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SsrfError(f"blocked URL scheme: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise SsrfError("URL has no host")

    if not _host_matches_allowlist(host):
        raise SsrfError(f"host not in outbound allow-list: {host}")

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SsrfError(f"could not resolve host {host}: {exc}") from exc

    resolved = {info[4][0] for info in infos}
    if not resolved:
        raise SsrfError(f"host {host} resolved to nothing")
    for ip_str in resolved:
        if not _ip_is_public(ip_str):
            raise SsrfError(f"host {host} resolves to non-public address {ip_str}")


def safe_get(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    headers: dict | None = None,
) -> bytes:
    """
    SSRF-checked GET. Returns the response body as bytes on HTTP 200.
    Raises SsrfError for a disallowed URL, a redirect, a non-200 status, or
    an oversized body. Blocking — call via asyncio.to_thread() from async
    code (same convention as every other blocking client in this codebase).
    """
    assert_url_allowed(url)

    resp = requests.get(url, timeout=timeout, allow_redirects=False, stream=True, headers=headers)
    try:
        if resp.is_redirect or resp.is_permanent_redirect or 300 <= resp.status_code < 400:
            raise SsrfError(f"refusing to follow redirect from {url} -> {resp.headers.get('location')!r}")
        if resp.status_code != 200:
            raise SsrfError(f"unexpected status {resp.status_code} from {url}")

        clen = resp.headers.get("content-length")
        if clen and clen.isdigit() and int(clen) > max_bytes:
            raise SsrfError(f"response too large ({clen} bytes) from {url}")

        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise SsrfError(f"response exceeded {max_bytes} bytes while streaming from {url}")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        resp.close()
