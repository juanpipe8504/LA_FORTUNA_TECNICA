"""Text layout analysis on top of ``page.get_text("dict")``.

Turns PyMuPDF blocks/lines/spans into ``PageElement``s, classifying each run of
lines as HEADING or TEXT with font heuristics relative to the document's body
font size (char-weighted median), which is robust across documents that use
different base sizes.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field

import pymupdf

from app.domain.models import BoundingBox, ElementType, PageElement
from app.infrastructure.parsing.captions import is_caption

# Text-only extraction: no image bytes, ligatures expanded ("ﬁ" -> "fi").
TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_PRESERVE_LIGATURES

_BOLD_FONT_HINT = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_DEFAULT_BODY_SIZE = 11.0


@dataclass(slots=True)
class TextLine:
    text: str
    bbox: BoundingBox
    size: float
    bold: bool


@dataclass(slots=True)
class TextBlock:
    bbox: BoundingBox
    lines: list[TextLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return join_lines(line.text for line in self.lines)


def _rect(values: Iterable[float]) -> BoundingBox:
    return BoundingBox.from_list(tuple(values))


def _span_is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & pymupdf.TEXT_FONT_BOLD) or bool(
        _BOLD_FONT_HINT.search(span.get("font", ""))
    )


def _make_line(spans: list[dict]) -> TextLine | None:
    text = _WHITESPACE.sub(" ", "".join(s["text"] for s in spans)).replace("­", "").strip()
    if not text:
        return None
    visible = [s for s in spans if s["text"].strip()]
    weights: dict[float, int] = {}
    for span in visible:
        size = round(float(span.get("size", 0.0)), 1)
        weights[size] = weights.get(size, 0) + len(span["text"].strip())
    bbox = _rect(spans[0]["bbox"])
    for span in spans[1:]:
        bbox = bbox.union(_rect(span["bbox"]))
    return TextLine(
        text=text,
        bbox=bbox,
        size=max(weights, key=lambda k: weights[k]) if weights else _DEFAULT_BODY_SIZE,
        bold=bool(visible) and all(_span_is_bold(s) for s in visible),
    )


def _split_on_gaps(spans: list[dict]) -> list[list[dict]]:
    """Split a visual line where spans are separated by a column-sized gap.

    PyMuPDF may merge side-by-side columns sharing a baseline into one line;
    a horizontal gap of several em is a column gutter, not a word space.
    """
    segments: list[list[dict]] = [[spans[0]]]
    for prev, span in zip(spans, spans[1:], strict=False):
        gap = span["bbox"][0] - prev["bbox"][2]
        if gap > max(3.0 * float(span.get("size", 10.0)), 24.0):
            segments.append([span])
        else:
            segments[-1].append(span)
    return segments


def _to_block(lines: list[TextLine]) -> TextBlock:
    bbox = lines[0].bbox
    for line in lines[1:]:
        bbox = bbox.union(line.bbox)
    return TextBlock(bbox=bbox, lines=lines)


def _split_columns(lines: list[TextLine]) -> list[list[TextLine]]:
    """Split a block whose lines sit side by side (columns sharing baselines).

    Lines are swept left-to-right into clusters of overlapping x-intervals; the
    split is applied only if two clusters actually share a row, so ordinary
    paragraphs (indents, short last lines) are never split.
    """
    if len(lines) < 2:
        return [lines]
    clusters: list[list[TextLine]] = []
    right_edge = float("-inf")
    for line in sorted(lines, key=lambda ln: ln.bbox.x0):
        if line.bbox.x0 > right_edge + 3.0:
            clusters.append([])
        clusters[-1].append(line)
        right_edge = max(right_edge, line.bbox.x1)
    if len(clusters) == 1:
        return [lines]

    def shares_row(a: list[TextLine], b: list[TextLine]) -> bool:
        return any(min(x.bbox.y1, y.bbox.y1) - max(x.bbox.y0, y.bbox.y0) > 1.0 for x in a for y in b)

    if not any(shares_row(a, b) for a, b in zip(clusters, clusters[1:], strict=False)):
        return [lines]
    return [sorted(c, key=lambda ln: (ln.bbox.y0, ln.bbox.x0)) for c in clusters]


def read_text_blocks(page: pymupdf.Page) -> list[TextBlock]:
    """Text blocks of a page with per-line dominant font size and bold flag.

    Blocks that span a column gutter are split into one block per column.
    """
    blocks: list[TextBlock] = []
    for raw_block in page.get_text("dict", flags=TEXT_FLAGS).get("blocks", []):
        if raw_block.get("type") != 0:
            continue
        lines: list[TextLine] = []
        for raw_line in raw_block.get("lines", []):
            spans = [s for s in raw_line.get("spans", []) if s.get("text")]
            if not spans:
                continue
            for segment in _split_on_gaps(spans):
                line = _make_line(segment)
                if line is not None:
                    lines.append(line)
        blocks.extend(_to_block(column) for column in _split_columns(lines) if column)
    return blocks


def body_font_size(blocks: Iterable[TextBlock]) -> float:
    """Char-weighted median font size: the size most of the running text uses."""
    sizes: list[float] = []
    for block in blocks:
        for line in block.lines:
            # Cap the weight so a single huge line cannot dominate memory/time.
            sizes.extend([round(line.size * 2) / 2] * min(len(line.text), 500))
    return statistics.median(sizes) if sizes else _DEFAULT_BODY_SIZE


def join_lines(lines: Iterable[str]) -> str:
    """Join visual lines into a paragraph, repairing end-of-line hyphenation."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}" if out else line
    return out


class HeadingDetector:
    """Decides whether a line is a heading.

    A line is a heading when it is short and either noticeably larger than the
    body text, or entirely bold at (roughly) body size and not sentence-like.
    Caption-like lines ("Figure 2: ...") are never headings.
    """

    def __init__(
        self,
        body_size: float,
        size_ratio: float = 1.15,
        max_chars: int = 150,
        max_words: int = 20,
    ) -> None:
        self.body_size = body_size
        self.size_ratio = size_ratio
        self.max_chars = max_chars
        self.max_words = max_words

    def is_heading(self, line: TextLine) -> bool:
        text = line.text.strip()
        if not text or len(text) > self.max_chars or len(text.split()) > self.max_words:
            return False
        if not any(ch.isalpha() for ch in text) or is_caption(text):
            return False
        if line.size >= self.body_size * self.size_ratio:
            return True
        return line.bold and line.size >= self.body_size * 0.95 and not text.endswith((".", ",", ";"))


def block_to_elements(
    block: TextBlock, page_number: int, detector: HeadingDetector, max_heading_lines: int = 3
) -> list[PageElement]:
    """Split a block into consecutive HEADING / TEXT runs (a block can hold both)."""
    runs: list[tuple[bool, list[TextLine]]] = []
    for line in block.lines:
        heading = detector.is_heading(line)
        if runs and runs[-1][0] == heading:
            runs[-1][1].append(line)
        else:
            runs.append((heading, [line]))

    elements: list[PageElement] = []
    for heading, lines in runs:
        # A long run of "big" lines is a callout/quote, not a heading.
        heading = heading and len(lines) <= max_heading_lines
        bbox = lines[0].bbox
        for line in lines[1:]:
            bbox = bbox.union(line.bbox)
        elements.append(
            PageElement(
                element_type=ElementType.HEADING if heading else ElementType.TEXT,
                page=page_number,
                bbox=bbox,
                text=join_lines(line.text for line in lines),
                font_size=max(line.size for line in lines),
                is_bold=all(line.bold for line in lines),
            )
        )
    # Merge adjacent TEXT runs produced by a heading-classification flip-flop.
    merged: list[PageElement] = []
    for element in elements:
        prev = merged[-1] if merged else None
        if prev and prev.element_type == element.element_type == ElementType.TEXT:
            prev.text = join_lines([prev.text, element.text])
            prev.bbox = prev.bbox.union(element.bbox)
        else:
            merged.append(element)
    return merged
