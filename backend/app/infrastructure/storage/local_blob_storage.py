"""Filesystem implementation of ``BlobStoragePort``.

Layout (see docs/CONTRACTS.md):

- uploads: ``{uploads_dir}/{document_id}.pdf``
- images:  ``{images_dir}/{document_id}/{image_id}.png``

Ids come from untrusted input (URL path parameters), so every id is validated
against a strict whitelist and every resolved path is checked to stay inside
its root directory (defence in depth against path traversal).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from app.core.logging import get_logger
from app.domain.exceptions import NotFoundError, ValidationError
from app.domain.models import ImageAsset
from app.domain.ports import BlobStoragePort

logger = get_logger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,127}$")

_EXTENSION_BY_MIME = {"image/png": ".png", "image/jpeg": ".jpg"}
_MIME_BY_EXTENSION = {ext: mime for mime, ext in _EXTENSION_BY_MIME.items()}


def _validate_id(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValidationError(f"Invalid {kind}: {value!r}")
    return value


class LocalBlobStorage(BlobStoragePort):
    def __init__(self, uploads_dir: str, images_dir: str) -> None:
        self._uploads_dir = Path(uploads_dir).resolve()
        self._images_dir = Path(images_dir).resolve()
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def save_upload(self, document_id: str, filename: str, data: bytes) -> str:
        """Persist the original PDF. ``filename`` is metadata only and never touches the path."""
        _validate_id(document_id, "document_id")
        path = self._inside(self._uploads_dir, self._uploads_dir / f"{document_id}.pdf")
        self._atomic_write(path, data)
        logger.info("upload_saved", extra={"document_id": document_id, "bytes": len(data), "path": str(path)})
        return str(path)

    def save_image(self, asset: ImageAsset, data: bytes) -> None:
        path = self._image_path(asset.document_id, asset.image_id, asset.mime_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, data)

    def get_image(self, document_id: str, image_id: str) -> tuple[bytes, str]:
        _validate_id(document_id, "document_id")
        _validate_id(image_id, "image_id")
        for mime in _MIME_BY_EXTENSION.values():
            path = self._image_path(document_id, image_id, mime)
            if path.is_file():
                return path.read_bytes(), mime
        raise NotFoundError(f"Image {image_id} not found for document {document_id}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _image_path(self, document_id: str, image_id: str, mime_type: str) -> Path:
        _validate_id(document_id, "document_id")
        _validate_id(image_id, "image_id")
        ext = _EXTENSION_BY_MIME.get(mime_type, ".png")
        return self._inside(self._images_dir, self._images_dir / document_id / f"{image_id}{ext}")

    @staticmethod
    def _inside(root: Path, candidate: Path) -> Path:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValidationError("Resolved path escapes the storage root")
        return resolved

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Write to a temp file and rename, so readers never observe partial files."""
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
