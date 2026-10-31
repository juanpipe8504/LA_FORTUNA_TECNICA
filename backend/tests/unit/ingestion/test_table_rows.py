"""Row-level table chunks: markdown parsing helpers and chunker integration."""

from __future__ import annotations

import pymupdf

from app.domain.models import BoundingBox, ChunkType, ElementType, ImageAsset, PageElement, ParsedDocument
from app.infrastructure.chunking.layout_chunker import (
    LayoutAwareChunker,
    MarkdownTable,
    markdown_table_rows,
    table_row_text,
)
from app.infrastructure.parsing.pymupdf_parser import PyMuPDFParser
from app.infrastructure.parsing.table_extractor import rows_to_markdown

from .conftest import draw_table

PLAN = (
    "Tabla 2: Plan de mantenimiento preventivo\n\n"
    "| Tarea | Intervalo | Observaciones |\n"
    "| --- | --- | --- |\n"
    "| Inspección visual | Mensual | Aire seco |\n"
    "| Relubricación de rodamientos | 4000 h | Grasa de litio NLGI 3; 20 g (LA) y 15 g (LOA) |\n"
    "| Medición de aislamiento | 6 meses | Mínimo 10 MOhm |"
)
TABLE_BOX = BoundingBox(56, 180, 539, 316)


def _doc(elements, images=()):
    return ParsedDocument("doc", "doc.pdf", page_count=1, elements=list(elements), images=list(images))


def _heading(text, y):
    bbox = BoundingBox(56, y, 400, y + 18)
    return PageElement(ElementType.HEADING, 1, bbox, text, font_size=18, is_bold=True)


def _table(text=PLAN, bbox=TABLE_BOX):
    return PageElement(ElementType.TABLE, 1, bbox, text)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_markdown_table_rows_parses_caption_header_and_rows():
    table = markdown_table_rows(PLAN)
    assert table == MarkdownTable(
        caption="Tabla 2: Plan de mantenimiento preventivo",
        header=["Tarea", "Intervalo", "Observaciones"],
        rows=[
            ["Inspección visual", "Mensual", "Aire seco"],
            ["Relubricación de rodamientos", "4000 h", "Grasa de litio NLGI 3; 20 g (LA) y 15 g (LOA)"],
            ["Medición de aislamiento", "6 meses", "Mínimo 10 MOhm"],
        ],
    )


def test_markdown_table_rows_restores_escaped_pipes_and_skips_empty_rows():
    markdown, _, _ = rows_to_markdown([["A|B", "C"], ["x | y", "1"], ["", ""], ["z", "2|3"]])
    assert "\\|" in markdown  # the extractor escapes pipes inside cells
    table = markdown_table_rows(markdown)
    assert table is not None
    assert table.caption == ""
    assert table.header == ["A|B", "C"]
    assert table.rows == [["x | y", "1"], ["z", "2|3"]]


def test_markdown_table_rows_accepts_alignment_delimiters_and_multiline_caption():
    text = "Table 9: Limits\n(continued)\n\n| k | v |\n|:---|---:|\n| a | 1 |"
    table = markdown_table_rows(text)
    assert table is not None
    assert table.caption == "Table 9: Limits (continued)"
    assert table.rows == [["a", "1"]]


def test_markdown_table_rows_returns_none_without_a_table():
    assert markdown_table_rows("Just a paragraph.") is None
    assert markdown_table_rows("| a | b |\n| c | d |") is None  # no delimiter row
    assert markdown_table_rows("") is None


def test_table_row_text_is_self_describing():
    text = table_row_text(
        "Tabla 2: Plan de mantenimiento preventivo",
        ["Tarea", "Intervalo", "Observaciones"],
        ["Relubricación de rodamientos", "4000 h", "Grasa de litio NLGI 3"],
    )
    assert text == (
        "Tabla 2: Plan de mantenimiento preventivo — Tarea: Relubricación de rodamientos; "
        "Intervalo: 4000 h; Observaciones: Grasa de litio NLGI 3"
    )


def test_table_row_text_skips_empty_cells_and_unnamed_columns():
    assert table_row_text("", ["k", "", "x"], ["a", "b", ""]) == "k: a; b"
    assert table_row_text("Cap", ["k"], ["a", "extra"]) == "Cap — k: a; extra"


# --------------------------------------------------------------------------- #
# Chunker
# --------------------------------------------------------------------------- #
def test_table_with_three_rows_gets_whole_chunk_plus_row_chunks():
    near = ImageAsset("doc_p1_i0", "doc", 1, BoundingBox(56, 330, 300, 430), caption="Figura 4: Engrasador")
    image_el = PageElement(ElementType.IMAGE, 1, near.bbox, near.caption, image_id=near.image_id)
    doc = _doc([_heading("5. Mantenimiento", 140), _table(), image_el], images=[near])

    chunks = LayoutAwareChunker().chunk(doc)
    tables = [c for c in chunks if c.chunk_type == ChunkType.TABLE]

    assert len(tables) == 4
    whole, *rows = tables
    assert whole.text == "Section: 5. Mantenimiento\n\n" + PLAN
    assert rows[1].text == (
        "Section: 5. Mantenimiento\n\n"
        "Tabla 2: Plan de mantenimiento preventivo — Tarea: Relubricación de rodamientos; "
        "Intervalo: 4000 h; Observaciones: Grasa de litio NLGI 3; 20 g (LA) y 15 g (LOA)"
    )
    for row in rows:
        assert row.section_path == ["5. Mantenimiento"]
        assert (row.page_start, row.page_end, row.bbox) == (1, 1, TABLE_BOX)
        assert [r.image_id for r in row.images] == [r.image_id for r in whole.images] == ["doc_p1_i0"]
    # The whole table comes first, its rows right after it, then the rest of the flow.
    assert chunks[chunks.index(whole) + 1 : chunks.index(whole) + 4] == rows


def test_row_chunks_prefix_only_the_innermost_heading():
    title = PageElement(ElementType.HEADING, 1, BoundingBox(56, 40, 400, 64), "Manual MTX-450", font_size=24)
    doc = _doc([title, _heading("5. Mantenimiento", 140), _table()])

    whole, first_row, *_ = LayoutAwareChunker().chunk(doc)

    assert whole.text.startswith("Section: Manual MTX-450 > 5. Mantenimiento\n\nTabla 2")
    assert first_row.text.startswith(
        "Section: 5. Mantenimiento\n\nTabla 2: Plan de mantenimiento preventivo — "
    )
    assert first_row.section_path == whole.section_path == ["Manual MTX-450", "5. Mantenimiento"]


def test_small_tables_and_disabled_option_keep_a_single_table_chunk():
    two_rows = "| k | v |\n| --- | --- |\n| a | 1 |\n| b | 2 |"
    assert len(LayoutAwareChunker().chunk(_doc([_table(two_rows)]))) == 1
    assert len(LayoutAwareChunker(table_row_min_rows=None).chunk(_doc([_table()]))) == 1
    assert len(LayoutAwareChunker(table_row_min_rows=2).chunk(_doc([_table(two_rows)]))) == 3


def test_unparseable_table_text_falls_back_to_whole_chunk_only():
    chunks = LayoutAwareChunker().chunk(_doc([_table("a b c\nd e f\ng h i\nj k l")]))
    assert [c.text for c in chunks] == ["a b c\nd e f\ng h i\nj k l"]


def test_row_chunk_ids_are_deterministic_and_unique():
    doc = _doc([_heading("5. Mantenimiento", 140), _table()])
    first = [c.chunk_id for c in LayoutAwareChunker().chunk(doc)]
    second = [c.chunk_id for c in LayoutAwareChunker().chunk(doc)]
    assert first == second
    assert len(set(first)) == len(first) == 4


def test_end_to_end_pdf_table_rows_carry_caption(tmp_path, blob_storage):
    path = tmp_path / "plan.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "5. Mantenimiento", fontsize=18, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 95, 523, 135), "Siga el plan de la tabla.", fontsize=11)
    page.insert_text((72, 162), "Tabla 2: Plan de mantenimiento", fontsize=9)
    draw_table(
        page,
        72,
        170,
        [
            ["Tarea", "Intervalo"],
            ["Limpieza", "Mensual"],
            ["Relubricacion", "4000 h"],
            ["Aislamiento", "6 meses"],
        ],
    )
    doc.save(path)
    doc.close()

    parsed = PyMuPDFParser(blob_storage).parse(str(path), document_id="d", filename="plan.pdf")
    tables = [c for c in LayoutAwareChunker().chunk(parsed) if c.chunk_type == ChunkType.TABLE]

    assert len(tables) == 4
    assert tables[2].text.endswith("Tabla 2: Plan de mantenimiento — Tarea: Relubricacion; Intervalo: 4000 h")
    assert len({c.bbox for c in tables}) == 1
