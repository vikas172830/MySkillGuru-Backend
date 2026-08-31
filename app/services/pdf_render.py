# Sync API, not async — the async API launches Chromium via
# asyncio.create_subprocess_exec on whatever event loop is currently running,
# which fails with NotImplementedError under uvicorn's default Windows loop
# (uvicorn forces WindowsSelectorEventLoopPolicy, which has no subprocess
# transport; only WindowsProactorEventLoopPolicy supports it). The sync API
# manages its browser process through Playwright's own driver connection
# instead, so it's unaffected by the calling loop's policy — callers must
# invoke this via asyncio.to_thread(), never awaited directly.
from playwright.sync_api import sync_playwright


def render_html_to_pdf(html_content: str) -> bytes:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
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
