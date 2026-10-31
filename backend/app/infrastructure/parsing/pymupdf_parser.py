"""``DocumentParserPort`` implementation on PyMuPDF.

Pipeline per document:

1. Read every page's text blocks (``get_text("dict")``) and compute the body
   font size (char-weighted median) used by the heading heuristics.
2. Detect running headers/footers (text repeated in page margins, page
   numbers) and drop them so they do not pollute chunks.
3. Per page: tables (``find_tables`` -> Markdown), figures (raster, inline and
   rendered vector diagrams), text/heading elements (text inside tables is
   excluded to avoid duplicates), caption -> figure/table association, and a
   column-aware reading order.
4. Image bytes are persisted through ``BlobStoragePort``; the returned
   ``ParsedDocument`` only carries metadata.

All bounding boxes are PDF points with a top-left origin (PyMuPDF default).
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import pymupdf

from app.core.logging import get_logger
from app.domain.exceptions import DocumentParsingError
from app.domain.models import ElementType, ImageAsset, PageElement, ParsedDocument
from app.domain.ports import BlobStoragePort, DocumentParserPort
from app.infrastructure.parsing.captions import caption_label, is_caption
from app.infrastructure.parsing.image_extractor import ImageExtractor, overlap_ratio
from app.infrastructure.parsing.table_extractor import TableExtractor
from app.infrastructure.parsing.text_layout import (
    HeadingDetector,
    TextBlock,
    block_to_elements,
    body_font_size,
    read_text_blocks,
)

logger = get_logger(__name__)

_MARGIN_RATIO = 0.07  # top/bottom band (fraction of page height) for headers/footers
_CAPTION_MAX_DISTANCE = 60.0  # PDF points between a caption and its figure/table
_PAGE_NUMBER = re.compile(r"^(page|página|pag\.?)?\s*\d+(\s*(of|de|/)\s*\d+)?$", re.IGNORECASE)


@dataclass(slots=True)
class _PageContent:
    number: int  # 1-based
    blocks: list[TextBlock]
    width: float
    height: float


@dataclass(slots=True)
class _Visual:
    """A figure or table on a page that may receive a caption."""

    element: PageElement
    is_table: bool
    asset: ImageAsset | None = None
    captioned: bool = False


class PyMuPDFParser(DocumentParserPort):
    def __init__(self, blob_storage: BlobStoragePort, min_image_size_px: int = 64) -> None:
        self._blob_storage = blob_storage
        self._tables = TableExtractor()
        self._images = ImageExtractor(min_image_size_px=min_image_size_px)

    # ------------------------------------------------------------------ #
    # Port
    # ------------------------------------------------------------------ #
    def parse(self, file_path: str, *, document_id: str, filename: str) -> ParsedDocument:
        started = time.perf_counter()
        doc = self._open(file_path)
        try:
            pages = [self._read_page(page) for page in doc]
            detector = HeadingDetector(body_font_size(b for p in pages for b in p.blocks))
            noise = self._margin_noise(pages)
            parsed = ParsedDocument(document_id=document_id, filename=filename, page_count=doc.page_count)
            for page, content in zip(doc, pages, strict=True):
                elements, images = self._parse_page(doc, page, content, detector, noise, document_id)
                parsed.elements.extend(elements)
                parsed.images.extend(images)
        except DocumentParsingError:
            raise
        except Exception as exc:
            raise DocumentParsingError(f"Failed to parse {filename}: {exc}") from exc
        finally:
            doc.close()

        logger.info(
            "pdf_parsed",
            extra={
                "document_id": document_id,
                "pages": parsed.page_count,
                "elements": len(parsed.elements),
                "images": len(parsed.images),
                "tables": sum(1 for e in parsed.elements if e.element_type == ElementType.TABLE),
                "body_font_size": detector.body_size,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return parsed

    # ------------------------------------------------------------------ #
    # Document level
    # ------------------------------------------------------------------ #
    @staticmethod
    def _open(file_path: str) -> pymupdf.Document:
        try:
            doc = pymupdf.open(file_path, filetype="pdf")
        except Exception as exc:
            raise DocumentParsingError(f"Cannot open PDF: {exc}") from exc
        if doc.needs_pass:
            doc.close()
            raise DocumentParsingError("PDF is password protected")
        if doc.page_count == 0:
            doc.close()
            raise DocumentParsingError("PDF has no pages")
        return doc

    @staticmethod
    def _read_page(page: pymupdf.Page) -> _PageContent:
        return _PageContent(page.number + 1, read_text_blocks(page), page.rect.width, page.rect.height)

    @staticmethod
    def _normalise_margin_text(text: str) -> str:
        return re.sub(r"\d+", "#", text.lower()).strip()

    def _in_margin(self, block: TextBlock, page: _PageContent) -> bool:
        band = page.height * _MARGIN_RATIO
        return block.bbox.y1 <= band or block.bbox.y0 >= page.height - band

    def _margin_noise(self, pages: list[_PageContent]) -> set[str]:
        """Normalised texts repeated in the top/bottom margin of many pages."""
        if len(pages) < 2:
            return set()
        counts: Counter[str] = Counter()
        for page in pages:
            counts.update(
                {self._normalise_margin_text(b.text) for b in page.blocks if self._in_margin(b, page)}
            )
        threshold = max(2, math.ceil(len(pages) * 0.5))
        return {text for text, count in counts.items() if count >= threshold}

    def _is_noise(self, block: TextBlock, page: _PageContent, noise: set[str]) -> bool:
        if not self._in_margin(block, page):
            return False
        text = block.text.strip()
        return bool(_PAGE_NUMBER.match(text)) or self._normalise_margin_text(text) in noise

    # ------------------------------------------------------------------ #
    # Page level
    # ------------------------------------------------------------------ #
    def _parse_page(
        self,
        doc: pymupdf.Document,
        page: pymupdf.Page,
        content: _PageContent,
        detector: HeadingDetector,
        noise: set[str],
        document_id: str,
    ) -> tuple[list[PageElement], list[ImageAsset]]:
        number = content.number
        visuals: list[_Visual] = []

        tables = self._tables.extract(page)
        table_boxes = [t.bbox for t in tables]
        for table in tables:
            element = PageElement(ElementType.TABLE, number, table.bbox, text=table.markdown)
            visuals.append(_Visual(element, is_table=True))

        assets: list[ImageAsset] = []
        for index, image in enumerate(self._images.extract(doc, page, exclude=table_boxes)):
            asset = ImageAsset(
                image_id=f"{document_id}_p{number}_i{index}",
                document_id=document_id,
                page=number,
                bbox=image.bbox,
                width=image.width,
                height=image.height,
            )
            try:
                self._blob_storage.save_image(asset, image.png)
            except Exception:
                logger.warning("image_save_failed", extra={"image_id": asset.image_id}, exc_info=True)
                continue
            assets.append(asset)
            element = PageElement(ElementType.IMAGE, number, image.bbox, image_id=asset.image_id)
            visuals.append(_Visual(element, is_table=False, asset=asset))

        text_elements: list[PageElement] = []
        for block in content.blocks:
            if self._is_noise(block, content, noise):
                continue
            if any(overlap_ratio(block.bbox, box) > 0.5 for box in table_boxes):
                continue  # already represented by the table's markdown
            text_elements.extend(block_to_elements(block, number, detector))

        consumed = self._attach_captions(text_elements, visuals)
        elements = [e for e in text_elements if id(e) not in consumed]
        elements.extend(v.element for v in visuals)
        return self._reading_order(elements, content.width), assets

    @staticmethod
    def _attach_captions(text_elements: list[PageElement], visuals: list[_Visual]) -> set[int]:
        """Greedy nearest matching of caption blocks to figures/tables.

        Figure captions only attach to images; table captions attach to tables
        or to images (tables embedded as pictures). Returns ids of consumed
        caption elements, which are removed from the text flow since they live
        on in the figure/table.
        """
        pairs: list[tuple[float, int, int]] = []
        for ci, caption in enumerate(text_elements):
            label = caption_label(caption.text) if is_caption(caption.text) else None
            if label is None:
                continue
            for vi, visual in enumerate(visuals):
                if visual.is_table and not label.startswith("table:"):
                    continue
                distance = caption.bbox.distance_to(visual.element.bbox)
                if distance <= _CAPTION_MAX_DISTANCE:
                    pairs.append((distance, ci, vi))

        consumed: set[int] = set()
        used_captions: set[int] = set()
        for _, ci, vi in sorted(pairs):
            visual = visuals[vi]
            if ci in used_captions or visual.captioned:
                continue
            caption = text_elements[ci]
            used_captions.add(ci)
            visual.captioned = True
            consumed.add(id(caption))
            if visual.is_table:
                visual.element.text = f"{caption.text}\n\n{visual.element.text}"
            else:
                visual.element.text = caption.text
                if visual.asset is not None:
                    visual.asset.caption = caption.text
        return consumed

    @staticmethod
    def _reading_order(elements: list[PageElement], page_width: float) -> list[PageElement]:
        """Top-to-bottom order, column-aware for two-column layouts.

        If enough elements sit entirely in the left and in the right half, the
        page is treated as two columns: full-width elements and the left column
        first, then the right column (each sorted by y).
        """
        mid = page_width / 2
        left = [e for e in elements if e.bbox.x1 <= mid + 5]
        right = [e for e in elements if e.bbox.x0 >= mid - 5]
        two_columns = (
            sum(e.element_type == ElementType.TEXT for e in left) >= 2
            and sum(e.element_type == ElementType.TEXT for e in right) >= 2
        )

        def key(element: PageElement) -> tuple[int, float, float]:
            column = 1 if two_columns and element.bbox.x0 >= mid - 5 else 0
            return column, round(element.bbox.y0, 0), element.bbox.x0

        return sorted(elements, key=key)
