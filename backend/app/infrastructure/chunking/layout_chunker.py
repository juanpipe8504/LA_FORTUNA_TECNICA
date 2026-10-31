"""Layout-aware semantic chunker.

Why not fixed-size windows: technical PDFs are organised by headings, and
tables/figures are self-contained units. Cutting every N characters splits
tables in half, mixes unrelated sections and detaches figures from the text
that explains them. Rules implemented here:

* **Sections** – headings (classified by the parser) open a new section. The
  heading hierarchy is derived from heading font sizes (bigger = higher level)
  and kept as ``section_path``; the breadcrumb is prefixed to every chunk's
  text so the embedding carries the section context.
* **Text** – paragraphs of a section are split into sentences and packed
  greedily up to ``max_tokens`` (approx. ``words * 1.3``), never cutting a
  sentence; consecutive chunks share ``overlap_sentences`` sentences.
* **Tables** – one atomic ``TABLE`` chunk (Markdown) regardless of size, so the
  LLM gets the whole table as context. Tables with ``table_row_min_rows`` or more
  body rows additionally get one ``TABLE`` chunk per row, rendered as
  self-describing text ("Tabla 2: Plan — Tarea: X; Intervalo: 4000 h"): a single
  row's signal is otherwise diluted among unrelated rows for both the dense
  embedding and BM25. Row chunks are prefixed with the innermost heading only
  (a long breadcrumb swamps a one-line row; measured on the sample manual) and
  share the table's page/bbox (and so its image links), which lets consumers
  group them with their whole-table chunk.
* **Images** – one ``IMAGE`` chunk per figure (caption + section context; the
  ingestion use case later appends the vision-LLM description).
* **Image links** – every chunk gets ``ImageRef``s to images on the same page
  within ``image_link_max_distance`` points, sorted by distance; a textual
  reference ("see Figure 3") to a captioned figure forces a link at distance 0.

Chunk ids are deterministic UUIDv5(document_id, index) so re-ingesting a
document overwrites the same points in the vector store.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

from app.domain.models import (
    BoundingBox,
    Chunk,
    ChunkType,
    ElementType,
    ImageAsset,
    ImageRef,
    PageElement,
    ParsedDocument,
)
from app.domain.ports import ChunkerPort
from app.infrastructure.parsing.captions import caption_label, mentioned_labels

CHUNK_NAMESPACE = uuid.UUID("6f1c3a52-9a0e-4d43-8f7b-2f1e6c1d9b10")

_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
_DELIMITER_CELL = re.compile(r"^:?-{3,}:?$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])[\"')\]]?\s+")
_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "fig.",
    "figs.",
    "eq.",
    "no.",
    "nr.",
    "approx.",
    "p.",
    "pp.",
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "sr.",
    "sra.",
    "ej.",
    "aprox.",
    "núm.",
    "cf.",
    "al.",
    "sec.",
    "ch.",
}


def approx_tokens(text: str) -> int:
    """Cheap tokenizer-agnostic estimate (~1.3 tokens per word for EN/ES)."""
    return math.ceil(len(text.split()) * 1.3)


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter that does not break on common abbreviations
    ("Fig. 3", "e.g.") or before lowercase continuations ("3.5 m. long")."""
    pieces = [p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p and p.strip()]
    sentences: list[str] = []
    for piece in pieces:
        if sentences:
            last_word = sentences[-1].rsplit(maxsplit=1)[-1].lower()
            if last_word in _ABBREVIATIONS or piece[:1].islower():
                sentences[-1] = f"{sentences[-1]} {piece}"
                continue
        sentences.append(piece)
    return sentences


@dataclass(slots=True)
class MarkdownTable:
    caption: str
    header: list[str]
    rows: list[list[str]]


def _split_cells(line: str) -> list[str]:
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") and not inner.endswith("\\|") else inner
    return [" ".join(c.replace("\\|", "|").split()) for c in _UNESCAPED_PIPE.split(inner)]


def markdown_table_rows(text: str) -> MarkdownTable | None:
    """Parse a GFM table (optionally preceded by caption lines) into caption/header/rows.

    Escaped pipes (``\\|``) inside cells are restored. Returns ``None`` when ``text``
    holds no table (no header + delimiter line).
    """
    caption_lines: list[str] = []
    table_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif not table_lines and stripped:
            caption_lines.append(stripped)
    if len(table_lines) < 2 or not all(_DELIMITER_CELL.match(c) for c in _split_cells(table_lines[1])):
        return None
    header = _split_cells(table_lines[0])
    rows = [_split_cells(line) for line in table_lines[2:]]
    return MarkdownTable(" ".join(caption_lines), header, [r for r in rows if any(r)])


def table_row_text(caption: str, header: list[str], row: list[str]) -> str:
    """Render one row as self-describing text: ``caption — Col: value; Col: value``."""
    parts = []
    for i, value in enumerate(row):
        if not value:
            continue
        name = header[i] if i < len(header) else ""
        parts.append(f"{name}: {value}" if name else value)
    body = "; ".join(parts)
    return f"{caption} — {body}" if caption else body


@dataclass(slots=True)
class _Sentence:
    text: str
    page: int
    bbox: BoundingBox


@dataclass(slots=True)
class _Draft:
    """A chunk under construction (before ids / image links)."""

    chunk_type: ChunkType
    body: str
    section_path: list[str]
    boxes: dict[int, BoundingBox] = field(default_factory=dict)  # page -> union bbox
    own_image_id: str | None = None
    # Row chunks are short: the full breadcrumb would dominate their embedding,
    # so only the innermost heading is prefixed (section_path keeps the full path).
    leaf_prefix: bool = False

    def add_box(self, page: int, bbox: BoundingBox) -> None:
        current = self.boxes.get(page)
        self.boxes[page] = bbox if current is None else current.union(bbox)


class LayoutAwareChunker(ChunkerPort):
    def __init__(
        self,
        max_tokens: int = 400,
        overlap_sentences: int = 1,
        image_link_max_distance: float = 150.0,
        table_row_min_rows: int | None = 3,
    ) -> None:
        if max_tokens < 20:
            raise ValueError("max_tokens must be >= 20")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.image_link_max_distance = image_link_max_distance
        # None disables row-level table chunks.
        self.table_row_min_rows = table_row_min_rows

    # ------------------------------------------------------------------ #
    # Port
    # ------------------------------------------------------------------ #
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        levels = self._heading_levels(document.elements)
        assets = {a.image_id: a for a in document.images}
        drafts: list[_Draft] = []
        section: list[tuple[int, str]] = []  # (level, heading text)
        buffer: list[_Sentence] = []
        # IMAGE chunks wait for the surrounding text chunk so chunk order follows reading order.
        pending: list[_Draft] = []

        def path() -> list[str]:
            return [title for _, title in section]

        for element in document.elements:
            kind = element.element_type
            if kind == ElementType.HEADING:
                self._flush(buffer, pending, path(), drafts, keep_overlap=False)
                level = levels.get(self._size_key(element), len(levels))
                while section and section[-1][0] >= level:
                    section.pop()
                section.append((level, element.text.strip()))
            elif kind == ElementType.TEXT:
                for sentence in self._sentences(element):
                    if buffer and self._tokens(buffer + [sentence], path()) > self.max_tokens:
                        self._flush(buffer, pending, path(), drafts, keep_overlap=True)
                    buffer.append(sentence)
            elif kind == ElementType.TABLE:
                self._flush(buffer, pending, path(), drafts, keep_overlap=False)
                drafts.extend(self._table_drafts(element, path()))
            elif kind == ElementType.IMAGE and element.image_id:
                pending.append(self._image_draft(element, assets.get(element.image_id), path()))
        self._flush(buffer, pending, path(), drafts, keep_overlap=False)

        return [self._finalise(i, d, document) for i, d in enumerate(d for d in drafts if d.body)]

    # ------------------------------------------------------------------ #
    # Section hierarchy
    # ------------------------------------------------------------------ #
    @staticmethod
    def _size_key(element: PageElement) -> float:
        return round((element.font_size or 0.0) * 2) / 2

    def _heading_levels(self, elements: list[PageElement]) -> dict[float, int]:
        """Map heading font size -> level (0 = biggest)."""
        sizes = sorted(
            {self._size_key(e) for e in elements if e.element_type == ElementType.HEADING}, reverse=True
        )
        return {size: level for level, size in enumerate(sizes)}

    # ------------------------------------------------------------------ #
    # Text packing
    # ------------------------------------------------------------------ #
    def _sentences(self, element: PageElement) -> list[_Sentence]:
        result: list[_Sentence] = []
        for sentence in split_sentences(element.text):
            # A single "sentence" longer than the budget (e.g. a long list without
            # punctuation) is cut on word boundaries so no chunk explodes.
            words = sentence.split()
            step = max(1, int(self.max_tokens * 0.8 / 1.3))
            for start in range(0, len(words), step):
                result.append(_Sentence(" ".join(words[start : start + step]), element.page, element.bbox))
        return result

    @staticmethod
    def _prefix(section_path: list[str]) -> str:
        return f"Section: {' > '.join(section_path)}\n\n" if section_path else ""

    def _tokens(self, sentences: list[_Sentence], section_path: list[str]) -> int:
        return approx_tokens(self._prefix(section_path)) + approx_tokens(" ".join(s.text for s in sentences))

    def _flush(
        self,
        buffer: list[_Sentence],
        pending: list[_Draft],
        section_path: list[str],
        drafts: list[_Draft],
        *,
        keep_overlap: bool,
    ) -> None:
        if not buffer:
            drafts.extend(pending)
            pending.clear()
            return
        draft = _Draft(ChunkType.TEXT, " ".join(s.text for s in buffer), list(section_path))
        for sentence in buffer:
            draft.add_box(sentence.page, sentence.bbox)
        drafts.append(draft)
        drafts.extend(pending)
        pending.clear()
        overlap = buffer[-self.overlap_sentences :] if keep_overlap and self.overlap_sentences else []
        # Never let the overlap alone fill a chunk (would loop on giant sentences).
        if overlap and self._tokens(overlap, section_path) > self.max_tokens // 2:
            overlap = []
        buffer[:] = overlap

    # ------------------------------------------------------------------ #
    # Tables
    # ------------------------------------------------------------------ #
    def _table_drafts(self, element: PageElement, section_path: list[str]) -> list[_Draft]:
        whole = _Draft(ChunkType.TABLE, element.text.strip(), list(section_path))
        whole.add_box(element.page, element.bbox)
        drafts = [whole]
        min_rows = self.table_row_min_rows
        table = markdown_table_rows(element.text) if min_rows is not None else None
        if table and min_rows is not None and len(table.rows) >= max(1, min_rows):
            for row in table.rows:
                body = table_row_text(table.caption, table.header, row)
                draft = _Draft(ChunkType.TABLE, body, list(section_path), leaf_prefix=True)
                draft.add_box(element.page, element.bbox)
                drafts.append(draft)
        return drafts

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #
    @staticmethod
    def _image_draft(element: PageElement, asset: ImageAsset | None, section_path: list[str]) -> _Draft:
        caption = (asset.caption if asset and asset.caption else element.text).strip()
        location = f"page {element.page}" + (f", section '{section_path[-1]}'" if section_path else "")
        body = f"[Image] {caption}" if caption else f"[Image] Figure on {location}."
        draft = _Draft(ChunkType.IMAGE, body, list(section_path), own_image_id=element.image_id)
        draft.add_box(element.page, element.bbox)
        return draft

    def _image_refs(self, draft: _Draft, document: ParsedDocument) -> list[ImageRef]:
        labels = mentioned_labels(draft.body) if draft.chunk_type != ChunkType.IMAGE else set()
        refs: dict[str, ImageRef] = {}
        for image in document.images:
            distance: float | None = None
            if image.image_id == draft.own_image_id:
                distance = 0.0
            else:
                label = caption_label(image.caption) if image.caption else None
                if label and label in labels:
                    distance = 0.0
                elif image.page in draft.boxes:
                    gap = draft.boxes[image.page].distance_to(image.bbox)
                    if gap <= self.image_link_max_distance:
                        distance = gap
            if distance is not None:
                refs[image.image_id] = ImageRef(
                    image_id=image.image_id,
                    page=image.page,
                    bbox=image.bbox,
                    distance=round(distance, 2),
                    caption=image.caption,
                )
        return sorted(refs.values(), key=lambda r: (r.distance, r.image_id))

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def _finalise(self, index: int, draft: _Draft, document: ParsedDocument) -> Chunk:
        pages = sorted(draft.boxes)
        page_start = pages[0] if pages else 1
        prefix_path = draft.section_path[-1:] if draft.leaf_prefix else draft.section_path
        return Chunk(
            chunk_id=str(uuid.uuid5(CHUNK_NAMESPACE, f"{document.document_id}:{index}")),
            document_id=document.document_id,
            document_name=document.filename,
            text=self._prefix(prefix_path) + draft.body,
            page_start=page_start,
            page_end=pages[-1] if pages else page_start,
            bbox=draft.boxes.get(page_start),
            chunk_type=draft.chunk_type,
            section_path=draft.section_path,
            images=self._image_refs(draft, document),
        )
