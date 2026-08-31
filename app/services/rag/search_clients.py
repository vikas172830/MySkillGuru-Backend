"""
Free, keyless-where-possible search clients for the Learning Resources
feature (roadmap.py's GET /{roadmap_id}/resources). Each function fetches
REAL candidate links from an official search API — no LLM ever generates a
URL for this feature; the resources prompt only ever picks an index into
what these functions already fetched, so a model can never fabricate a
link that reaches a student.

Every function returns [] (never raises) on any failure — missing key,
network error, malformed response — so a resources fetch degrading to
"no video results this time" never blocks or breaks notes generation.
Same graceful-degradation philosophy as rag/singletons.get_vector_store().

These are synchronous (uses `requests`) — call sites must wrap them with
asyncio.to_thread, same as every other blocking AI-client call in this
codebase (see generate_claude_json/generate_gemini_json call sites).
"""
from __future__ import annotations

import html
import logging
import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests

logger = logging.getLogger("app.services.rag.search_clients")

_REQUEST_TIMEOUT = 6

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text or "")).strip()


def search_youtube(query: str, max_results: int = 5) -> list[dict]:
    """
    YouTube Data API v3 search.list — official, free up to the default daily
    quota (each search call costs 100 of the 10,000 daily units, i.e. ~100
    searches/day). Requires YOUTUBE_API_KEY in .env; returns [] without
    making a request if it's unset, so this feature degrades cleanly for
    anyone who hasn't configured a key yet.
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        logger.warning("search_youtube skipped: YOUTUBE_API_KEY not set")
        return []
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "safeSearch": "strict",
                "relevanceLanguage": "en",
                "key": api_key,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as exc:
        logger.warning("search_youtube failed for query=%r: %s", query, exc)
        return []

    results = []
    for item in items:
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if not video_id or not snippet.get("title"):
            continue
        results.append({
            "title": snippet["title"],
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "description": snippet.get("description", "")[:300],
            "source": "YouTube",
        })
    return results


def search_wikipedia(query: str, max_results: int = 3) -> list[dict]:
    """Wikipedia's public search API — no key required."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            },
            headers={"User-Agent": "LMS-LearningResources/1.0"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("query", {}).get("search", [])
    except Exception as exc:
        logger.warning("search_wikipedia failed for query=%r: %s", query, exc)
        return []

    results = []
    for item in items:
        title = item.get("title")
        if not title:
            continue
        results.append({
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            "description": _strip_html(item.get("snippet", ""))[:300],
            "source": "Wikipedia",
        })
    return results


def search_arxiv(query: str, max_results: int = 3) -> list[dict]:
    """arXiv's public Atom API — no key required, open-access papers only."""
    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as exc:
        logger.warning("search_arxiv failed for query=%r: %s", query, exc)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        id_el = entry.find("atom:id", ns)
        summary_el = entry.find("atom:summary", ns)
        if title_el is None or id_el is None or not (title_el.text or "").strip():
            continue
        results.append({
            "title": " ".join((title_el.text or "").split()),
            "url": (id_el.text or "").strip(),
            "description": " ".join((summary_el.text or "").split())[:300] if summary_el is not None else "",
            "source": "arXiv",
        })
    return results
