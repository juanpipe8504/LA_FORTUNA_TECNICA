"""Fixtures for the ingestion tests: in-test PDF generation and fakes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest

from app.domain.exceptions import NotFoundError
from app.domain.models import ImageAsset
from app.domain.ports import BlobStoragePort

BODY = (
    "The ingestion service receives PDF files through the REST API. Each file is stored and a job is "
    "queued for asynchronous processing. The worker parses the layout and extracts tables and images. "
    "As shown in Figura 1, the API never blocks on heavy work."
)
SECOND_SECTION = (
    "Retrieval combines dense vectors with sparse keyword vectors. Results are fused with reciprocal "
    "rank fusion. The answer cites its sources."
)


class FakeBlobStorage(BlobStoragePort):
    """In-memory blob storage that records saved images."""

    def __init__(self) -> None:
        self.images: dict[tuple[str, str], bytes] = {}
        self.uploads: dict[str, bytes] = {}

    def save_upload(self, document_id: str, filename: str, data: bytes) -> str:
        self.uploads[document_id] = data
        return f"/uploads/{document_id}.pdf"

    def save_image(self, asset: ImageAsset, data: bytes) -> None:
        self.images[(asset.document_id, asset.image_id)] = data

    def get_image(self, document_id: str, image_id: str) -> tuple[bytes, str]:
        try:
            return self.images[(document_id, image_id)], "image/png"
        except KeyError as exc:
            raise NotFoundError(image_id) from exc


def make_png(width: int = 160, height: int = 100, color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.set_rect(pix.irect, color)
    return pix.tobytes("png")


def draw_table(
    page: pymupdf.Page, x0: float, y0: float, rows: list[list[str]], col_w: float = 120, row_h: float = 20
) -> pymupdf.Rect:
    """Draw a ruled table (so ``find_tables`` with the lines strategy detects it)."""
    n_rows, n_cols = len(rows), len(rows[0])
    for r in range(n_rows + 1):
        page.draw_line((x0, y0 + r * row_h), (x0 + n_cols * col_w, y0 + r * row_h))
    for c in range(n_cols + 1):
        page.draw_line((x0 + c * col_w, y0), (x0 + c * col_w, y0 + n_rows * row_h))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((x0 + c * col_w + 4, y0 + r * row_h + 14), cell, fontsize=10)
    return pymupdf.Rect(x0, y0, x0 + n_cols * col_w, y0 + n_rows * row_h)


def build_technical_pdf(path: Path, *, with_vector_diagram: bool = False, with_footer: bool = True) -> Path:
    doc = pymupdf.open()

    page = doc.new_page()  # A4-ish 595x842
    page.insert_text((72, 80), "1. Architecture Overview", fontsize=18, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 95, 523, 175), BODY, fontsize=11)
    page.insert_image(pymupdf.Rect(72, 185, 312, 335), stream=make_png())
    page.insert_text((72, 352), "Figura 1: Arquitectura del sistema de ingesta", fontsize=9)
    page.insert_text((72, 395), "1.1 Data Model", fontsize=14, fontname="hebo")
    page.insert_textbox(
        pymupdf.Rect(72, 405, 523, 445),
        "The job table below lists the states of an ingestion job.",
        fontsize=11,
    )
    page.insert_text((72, 462), "Table 1: Job states", fontsize=9)
    draw_table(page, 72, 470, [["State", "Meaning"], ["pending", "queued"], ["completed", "indexed"]])
    if with_footer:
        page.insert_text((280, 820), "Page 1", fontsize=8)

    page2 = doc.new_page()
    page2.insert_text((72, 80), "2. Retrieval", fontsize=18, fontname="hebo")
    page2.insert_textbox(pymupdf.Rect(72, 95, 523, 175), SECOND_SECTION, fontsize=11)
    if with_vector_diagram:
        for i in range(3):
            x = 90 + i * 150
            page2.draw_rect(pymupdf.Rect(x, 220, x + 100, 270), color=(0, 0, 1), fill=(0.9, 0.9, 1))
            page2.insert_text((x + 10, 250), f"Stage {i + 1}", fontsize=10)
            if i < 2:
                page2.draw_line((x + 100, 245), (x + 150, 245))
                page2.draw_line((x + 142, 240), (x + 150, 245))
                page2.draw_line((x + 142, 250), (x + 150, 245))
        page2.draw_rect(pymupdf.Rect(90, 280, 490, 320), color=(0, 0.5, 0))
        page2.insert_text((72, 345), "Figure 2: Retrieval pipeline", fontsize=9)
    if with_footer:
        page2.insert_text((280, 820), "Page 2", fontsize=8)

    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def blob_storage() -> FakeBlobStorage:
    return FakeBlobStorage()


@pytest.fixture
def pdf_factory(tmp_path: Path) -> Callable[..., Path]:
    def factory(name: str = "doc.pdf", **kwargs: bool) -> Path:
        return build_technical_pdf(tmp_path / name, **kwargs)

    return factory
