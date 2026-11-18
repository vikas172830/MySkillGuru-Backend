# ============================================================
# Bounded reads for multipart file uploads.
#
# Every upload endpoint buffers the whole file in memory (and several then
# base64-expand it ~1.33x before sending to Gemini / ImageKit). On the
# single-process backend a few concurrent oversized uploads are a
# memory-pressure DoS. Next's proxy caps the request body at 50 MB
# (next.config.mjs `proxyClientMaxBodySize`); this is the server-side
# backstop so a direct call to the API can't bypass it.
# ============================================================
from fastapi import HTTPException, UploadFile, status

MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # 40 MB — comfortably under the 50 MB proxy cap
_CHUNK = 1024 * 1024


def _too_large(max_bytes: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"File too large. Maximum upload size is {max_bytes // (1024 * 1024)} MB.",
    )


async def read_upload_capped(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read an UploadFile fully, aborting with HTTP 413 once it exceeds
    max_bytes instead of letting an unbounded body into RAM."""
    size_hint = getattr(file, "size", None)
    if isinstance(size_hint, int) and size_hint > max_bytes:
        raise _too_large(max_bytes)

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise _too_large(max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)
