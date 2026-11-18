# Sync API, not async — the async API launches Chromium via
# asyncio.create_subprocess_exec on whatever event loop is currently running,
# which fails with NotImplementedError under uvicorn's default Windows loop
# (uvicorn forces WindowsSelectorEventLoopPolicy, which has no subprocess
# transport; only WindowsProactorEventLoopPolicy supports it). The sync API
# manages its browser process through Playwright's own driver connection
# instead, so it's unaffected by the calling loop's policy — callers must
# invoke this via asyncio.to_thread(), never awaited directly.
import logging

from playwright.sync_api import sync_playwright

from app.utils.net import assert_url_allowed

logger = logging.getLogger("app.services.pdf_render")


def _guard_request(route):
    """
    SSRF guard for sub-resource loads. The HTML rendered here is AI output built
    from student input, so an <img src>, <iframe> or fetch() inside it must not
    reach 169.254.169.254 (cloud metadata) or an internal service. data: URIs and
    the about:blank main document are always fine; every http(s) URL must clear
    the same public-IP / allow-list check as safe_get(); anything else is aborted.
    """
    url = route.request.url
    if url.startswith("data:") or url == "about:blank":
        route.continue_()
        return
    if url.startswith(("http://", "https://")):
        try:
            assert_url_allowed(url)
        except Exception as exc:
            logger.warning("pdf_render: blocked sub-resource %s (%s)", url, exc)
            route.abort()
            return
        route.continue_()
        return
    logger.warning("pdf_render: blocked non-http sub-resource %s", url)
    route.abort()


def render_html_to_pdf(html_content: str) -> bytes:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.route("**/*", _guard_request)
            page.set_content(html_content, wait_until="networkidle")

            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,
                margin={"top": "15mm", "right": "15mm", "bottom": "15mm", "left": "15mm"},
            )

            browser.close()
            return pdf_bytes

    except Exception as e:
        raise RuntimeError(f"PDF rendering failed: {e}") from e
