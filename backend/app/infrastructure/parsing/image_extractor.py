"""Image extraction: embedded raster images, inline images and vector diagrams.

Technical PDFs carry figures in three forms:

1. **XObject raster images** – ``page.get_images`` + ``page.get_image_rects``
   for the on-page placement. Converted to PNG (CMYK -> RGB, soft masks
   applied as alpha).
2. **Inline images** – no xref; recovered from ``get_text("dict")`` image
   blocks only when ``get_image_info`` reports one (avoids decoding every page
   twice).
3. **Vector drawings** – diagrams drawn with paths. When a page has enough
   drawing commands, ``page.cluster_drawings`` groups them into regions; each
   large-enough region not covered by a table or raster image is rendered to
   PNG via ``page.get_pixmap(clip=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pymupdf

from app.core.logging import get_logger
from app.domain.models import BoundingBox

logger = get_logger(__name__)

ImageSource = Literal["raster", "inline", "vector"]


@dataclass(slots=True)
class ExtractedImage:
    bbox: BoundingBox
    png: bytes
    width: int
    height: int
    source: ImageSource


def _bbox(rect: pymupdf.Rect) -> BoundingBox:
    return BoundingBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


def overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection area divided by the area of ``a`` (0..1)."""
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return (ix * iy) / a.area if a.area > 0 else 0.0


def _contains(outer: BoundingBox, inner: BoundingBox, tolerance: float = 1.0) -> bool:
    """Containment test that also works for zero-area boxes (straight lines)."""
    return (
        inner.x0 >= outer.x0 - tolerance
        and inner.y0 >= outer.y0 - tolerance
        and inner.x1 <= outer.x1 + tolerance
        and inner.y1 <= outer.y1 + tolerance
    )


def pixmap_to_png(pix: pymupdf.Pixmap) -> bytes:
    """Encode a pixmap as PNG, converting colourspaces PNG cannot hold (CMYK...)."""
    if pix.colorspace is None:
        raise ValueError("stencil/alpha-only pixmap cannot be encoded")
    if pix.colorspace.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return pix.tobytes("png")


class ImageExtractor:
    def __init__(
        self,
        min_image_size_px: int = 64,
        render_dpi: int = 144,
        min_drawings_per_page: int = 8,
        min_drawings_per_figure: int = 5,
        min_vector_size_pt: float = 48.0,
        max_page_coverage: float = 0.85,
        cluster_tolerance: float = 12.0,
    ) -> None:
        self.min_image_size_px = min_image_size_px
        self.render_dpi = render_dpi
        self.min_drawings_per_page = min_drawings_per_page
        self.min_drawings_per_figure = min_drawings_per_figure
        self.min_vector_size_pt = min_vector_size_pt
        self.max_page_coverage = max_page_coverage
        self.cluster_tolerance = cluster_tolerance

    def extract(
        self, doc: pymupdf.Document, page: pymupdf.Page, exclude: list[BoundingBox] | None = None
    ) -> list[ExtractedImage]:
        """All figures on a page, ordered top-to-bottom / left-to-right.

        ``exclude`` (e.g. table bboxes) prevents table borders from being
        mistaken for vector diagrams.
        """
        images = self._raster_images(doc, page)
        images += self._inline_images(page, images)
        images += self._vector_figures(page, [*(exclude or []), *(i.bbox for i in images)])
        images.sort(key=lambda i: (round(i.bbox.y0), i.bbox.x0))
        return images

    # ------------------------------------------------------------------ #
    # Raster (XObject) images
    # ------------------------------------------------------------------ #
    def _raster_images(self, doc: pymupdf.Document, page: pymupdf.Page) -> list[ExtractedImage]:
        page_rect = page.rect
        seen: set[int] = set()
        result: list[ExtractedImage] = []
        for item in page.get_images(full=True):
            xref, smask, width, height = item[0], item[1], item[2], item[3]
            if xref in seen:
                continue
            seen.add(xref)
            if min(width, height) < self.min_image_size_px:
                continue
            try:
                rects = [r & page_rect for r in page.get_image_rects(xref)]
                rects = [r for r in rects if not r.is_empty and r.width >= 4 and r.height >= 4]
                if not rects:  # image resource present but not displayed on this page
                    continue
                rect = max(rects, key=lambda r: r.width * r.height)
                png = self._xref_to_png(doc, xref, smask)
            except Exception:
                logger.warning(
                    "image_extraction_failed", extra={"page": page.number + 1, "xref": xref}, exc_info=True
                )
                continue
            result.append(ExtractedImage(_bbox(rect), png, width, height, "raster"))
        return result

    @staticmethod
    def _xref_to_png(doc: pymupdf.Document, xref: int, smask: int) -> bytes:
        pix = pymupdf.Pixmap(doc, xref)
        if smask:
            try:
                pix = pymupdf.Pixmap(pix, pymupdf.Pixmap(doc, smask))
            except Exception:  # broken soft mask: keep the opaque image
                logger.debug("smask_apply_failed", extra={"xref": xref})
        return pixmap_to_png(pix)

    # ------------------------------------------------------------------ #
    # Inline images (no xref)
    # ------------------------------------------------------------------ #
    def _inline_images(self, page: pymupdf.Page, existing: list[ExtractedImage]) -> list[ExtractedImage]:
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:
            return []
        min_px = self.min_image_size_px
        if not any(
            info.get("xref", 0) == 0 and min(info.get("width", 0), info.get("height", 0)) >= min_px
            for info in infos
        ):
            return []
        result: list[ExtractedImage] = []
        for block in page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_IMAGES).get("blocks", []):
            if block.get("type") != 1:
                continue
            width, height = int(block.get("width", 0)), int(block.get("height", 0))
            if min(width, height) < self.min_image_size_px:
                continue
            bbox = _bbox(pymupdf.Rect(block["bbox"]) & page.rect)
            if bbox.area <= 0 or any(overlap_ratio(bbox, e.bbox) > 0.5 for e in [*existing, *result]):
                continue
            try:
                png = pixmap_to_png(pymupdf.Pixmap(block["image"]))
            except Exception:
                logger.warning("inline_image_failed", extra={"page": page.number + 1}, exc_info=True)
                continue
            result.append(ExtractedImage(bbox, png, width, height, "inline"))
        return result

    # ------------------------------------------------------------------ #
    # Vector diagrams
    # ------------------------------------------------------------------ #
    def _vector_figures(self, page: pymupdf.Page, exclude: list[BoundingBox]) -> list[ExtractedImage]:
        try:
            drawings = page.get_drawings()
            if len(drawings) < self.min_drawings_per_page:
                return []
            clusters = page.cluster_drawings(
                drawings=drawings, x_tolerance=self.cluster_tolerance, y_tolerance=self.cluster_tolerance
            )
        except Exception:
            logger.warning("drawing_analysis_failed", extra={"page": page.number + 1}, exc_info=True)
            return []

        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        drawing_boxes = [_bbox(pymupdf.Rect(d["rect"])) for d in drawings if d.get("rect") is not None]
        result: list[ExtractedImage] = []
        for cluster in clusters:
            rect = pymupdf.Rect(cluster) & page_rect
            bbox = _bbox(rect)
            if bbox.width < self.min_vector_size_pt or bbox.height < self.min_vector_size_pt:
                continue
            if bbox.area > self.max_page_coverage * page_area:  # page frame / background
                continue
            if any(overlap_ratio(bbox, other) > 0.3 or overlap_ratio(other, bbox) > 0.5 for other in exclude):
                continue
            members = sum(1 for d in drawing_boxes if _contains(bbox, d))
            if members < self.min_drawings_per_figure:
                continue
            try:
                clip = pymupdf.Rect(rect.x0 - 4, rect.y0 - 4, rect.x1 + 4, rect.y1 + 4) & page_rect
                pix = page.get_pixmap(clip=clip, dpi=self.render_dpi)
                png = pixmap_to_png(pix)
            except Exception:
                logger.warning("vector_render_failed", extra={"page": page.number + 1}, exc_info=True)
                continue
            result.append(ExtractedImage(_bbox(clip), png, pix.width, pix.height, "vector"))
            exclude = [*exclude, bbox]
        return result
