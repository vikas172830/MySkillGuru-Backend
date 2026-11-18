import logging
import re
from typing import Dict, Tuple

import anthropic

from app.core.config import settings

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Lazy singleton — avoids failing app startup when ANTHROPIC_API_KEY isn't set yet."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


def generate_html(
    prompt: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 16000
) -> Tuple[str, Dict[str, int]]:
    """
    Blocking call — run via asyncio.to_thread() from async callers.
    Streamed: a full styled HTML document can be tens of thousands of tokens,
    and a non-streaming request with a large max_tokens risks an HTTP read
    timeout. Returns (html_content, token_usage) with any accidental markdown
    fences stripped. If the model hit the token cap, a visible truncation
    banner is appended and a warning is logged.
    """
    try:
        with _get_client().messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()

        html_content = message.content[0].text
        html_content = re.sub(r"^```html\s*", "", html_content, flags=re.IGNORECASE)
        html_content = re.sub(r"^```\s*", "", html_content)
        html_content = re.sub(r"\s*```\s*$", "", html_content)
        html_content = html_content.strip()

        if message.stop_reason == "max_tokens":
            logging.warning(
                "generate_html: output truncated at max_tokens=%d (model=%s) — raise the caller's budget",
                max_tokens, model,
            )
            html_content += (
                '\n<div style="margin:24px;padding:12px 16px;border:1px solid #f59e0b;'
                'background:#fffbeb;color:#92400e;border-radius:6px;font-family:Arial,sans-serif;'
                'font-size:13px">This document was cut off before it finished generating. '
                'Try a shorter length, or regenerate.</div>'
            )

        token_usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        }

        return html_content, token_usage

    except Exception as e:
        raise RuntimeError(f"Content generation failed: {e}") from e


def generate_text(
    prompt: str, model: str = "claude-sonnet-4-6", max_tokens: int = 8000
) -> Tuple[str, Dict[str, int]]:
    """
    Blocking call — run via asyncio.to_thread() from async callers.
    Like generate_html() but returns the raw response text unmodified (no markdown-fence
    stripping) — used where the output is plain text, not an HTML document.
    """
    try:
        message = _get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        text = message.content[0].text.strip()

        token_usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        }

        return text, token_usage

    except Exception as e:
        raise RuntimeError(f"Content generation failed: {e}") from e
