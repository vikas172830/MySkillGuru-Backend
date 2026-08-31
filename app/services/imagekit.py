import base64
import hashlib
import hmac
import os
import time
from typing import Any, Dict, List

from app.core.config import settings


def get_imagekit_auth_params() -> Dict[str, Any]:
    """Signed token for the frontend to upload directly to ImageKit. Pure HMAC — no SDK needed."""
    expire = int(time.time()) + 240
    token = base64.urlsafe_b64encode(os.urandom(16)).decode()

    signature = hmac.new(
        settings.IMAGEKIT_PRIVATE_KEY.encode(), f"{token}{expire}".encode(), hashlib.sha1
    ).hexdigest()

    return {
        "token": token,
        "expire": expire,
        "signature": signature,
        "publicKey": settings.IMAGEKIT_PUBLIC_KEY,
    }


def get_imagekit_client():
    from imagekitio import ImageKit

    return ImageKit(
        public_key=settings.IMAGEKIT_PUBLIC_KEY,
        private_key=settings.IMAGEKIT_PRIVATE_KEY,
        url_endpoint=settings.IMAGEKIT_URL_ENDPOINT,
    )


def delete_imagekit_file(file_id: str) -> None:
    """Blocking call — run via asyncio.to_thread() from async callers."""
    get_imagekit_client().delete_file(file_id)


def upload_file_to_imagekit(file_binary: bytes, filename: str, folder: str, tags: List[str]) -> Dict[str, Any]:
    """Blocking call — run via asyncio.to_thread() from async callers. Content-agnostic (PDF/docx/image/...)."""
    from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions

    try:
        file_b64 = base64.b64encode(file_binary).decode("utf-8")
        result = get_imagekit_client().upload_file(
            file=file_b64,
            file_name=filename,
            options=UploadFileRequestOptions(folder=folder, tags=tags, use_unique_file_name=True),
        )
        return {
            "url": result.url if hasattr(result, "url") else result.get("url"),
            "file_id": result.file_id if hasattr(result, "file_id") else result.get("fileId"),
        }
    except Exception as e:
        raise RuntimeError(f"Upload failed: {e}") from e
