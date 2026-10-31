"""Table detection (``page.find_tables``) and Markdown rendering."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

from app.core.logging import get_logger
from app.domain.models import BoundingBox

logger = get_logger(__name__)


@dataclass(slots=True)
class ExtractedTable:
    bbox: BoundingBox
    markdown: str
    rows: int
    cols: int


def _clean_cell(value: object) -> str:
    text = "" if value is None else " ".join(str(value).split())
    return text.replace("|", "\\|")


def rows_to_markdown(rows: list[list[object]]) -> tuple[str, int, int]:
    """Render rows as a GitHub-flavoured Markdown table (first row = header).

    Fully empty rows and columns are dropped. Returns (markdown, rows, cols).
    """
    cleaned = [[_clean_cell(c) for c in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return "", 0, 0
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    keep = [i for i in range(width) if any(row[i] for row in cleaned)]
    cleaned = [[row[i] for i in keep] for row in cleaned]
    if not keep:
        return "", 0, 0
    header, body = cleaned[0], cleaned[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    return "\n".join(lines), len(cleaned), len(keep)


class TableExtractor:
    """Wraps ``find_tables``; tables with fewer than 2 rows or 2 columns are discarded
    as they are almost always false positives (rules, boxes around text)."""

    def __init__(self, min_rows: int = 2, min_cols: int = 2) -> None:
        self.min_rows = min_rows
        self.min_cols = min_cols

    def extract(self, page: pymupdf.Page) -> list[ExtractedTable]:
        try:
            finder = page.find_tables()
        except Exception:  # table detection must never break parsing
            logger.warning("table_detection_failed", extra={"page": page.number + 1}, exc_info=True)
            return []
        tables: list[ExtractedTable] = []
        for table in finder.tables:
            try:
                markdown, rows, cols = rows_to_markdown(table.extract())
            except Exception:
                logger.warning("table_extraction_failed", extra={"page": page.number + 1}, exc_info=True)
                continue
            if rows < self.min_rows or cols < self.min_cols:
                continue
            tables.append(ExtractedTable(BoundingBox.from_list(tuple(table.bbox)), markdown, rows, cols))
        return tables
