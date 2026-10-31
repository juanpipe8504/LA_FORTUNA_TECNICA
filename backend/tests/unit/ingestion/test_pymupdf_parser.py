from __future__ import annotations

import pymupdf
import pytest

from app.domain.exceptions import DocumentParsingError
from app.domain.models import ElementType
from app.infrastructure.parsing.pymupdf_parser import PyMuPDFParser
from app.infrastructure.parsing.table_extractor import rows_to_markdown
from app.infrastructure.parsing.text_layout import join_lines
from tests.unit.ingestion.conftest import make_png


def _parse(path, blob_storage, **kwargs):
    return PyMuPDFParser(blob_storage, **kwargs).parse(str(path), document_id="doc1", filename="doc.pdf")


def _of_type(document, element_type):
    return [e for e in document.elements if e.element_type == element_type]


def test_detects_headings_and_body_text(pdf_factory, blob_storage):
    document = _parse(pdf_factory(), blob_storage)

    assert document.page_count == 2
    headings = [e.text for e in _of_type(document, ElementType.HEADING)]
    assert headings == ["1. Architecture Overview", "1.1 Data Model", "2. Retrieval"]
    body = " ".join(e.text for e in _of_type(document, ElementType.TEXT))
    assert "asynchronous processing" in body
    assert "reciprocal rank fusion" in body


def test_drops_page_number_footers(pdf_factory, blob_storage):
    document = _parse(pdf_factory(), blob_storage)
    assert not any(e.text.startswith("Page ") for e in document.elements)


def test_extracts_table_as_markdown_without_duplicating_cell_text(pdf_factory, blob_storage):
    document = _parse(pdf_factory(), blob_storage)

    tables = _of_type(document, ElementType.TABLE)
    assert len(tables) == 1
    table = tables[0]
    assert table.text.startswith("Table 1: Job states")  # caption attached to the table
    assert "| State | Meaning |" in table.text
    assert "| pending | queued |" in table.text
    texts = " ".join(e.text for e in document.elements if e.element_type != ElementType.TABLE)
    assert "pending" not in texts and "indexed" not in texts
    assert "Table 1: Job states" not in texts  # consumed caption is not duplicated


def test_extracts_raster_image_with_bbox_caption_and_persists_png(pdf_factory, blob_storage):
    document = _parse(pdf_factory(), blob_storage)

    raster = [i for i in document.images if i.page == 1]
    assert len(raster) == 1
    image = raster[0]
    assert image.image_id == "doc1_p1_i0"
    assert image.caption == "Figura 1: Arquitectura del sistema de ingesta"
    assert image.bbox.to_list() == pytest.approx([72, 185, 312, 335], abs=1)
    assert (image.width, image.height) == (160, 100)
    data = blob_storage.images[("doc1", "doc1_p1_i0")]
    assert data.startswith(b"\x89PNG")
    element = next(e for e in document.elements if e.image_id == "doc1_p1_i0")
    assert element.element_type == ElementType.IMAGE and element.text == image.caption


def test_reading_order_places_image_between_its_paragraphs(pdf_factory, blob_storage):
    document = _parse(pdf_factory(), blob_storage)
    page1 = [e.element_type for e in document.elements if e.page == 1]
    assert page1 == [
        ElementType.HEADING,
        ElementType.TEXT,
        ElementType.IMAGE,
        ElementType.HEADING,
        ElementType.TEXT,
        ElementType.TABLE,
    ]


def test_renders_vector_diagram_as_image(pdf_factory, blob_storage):
    document = _parse(pdf_factory(with_vector_diagram=True), blob_storage)

    vector = [i for i in document.images if i.page == 2]
    assert len(vector) == 1
    assert vector[0].caption == "Figure 2: Retrieval pipeline"
    assert vector[0].bbox.width > 300
    assert blob_storage.images[("doc1", vector[0].image_id)].startswith(b"\x89PNG")


def test_no_vector_figure_for_table_borders(pdf_factory, blob_storage):
    document = _parse(pdf_factory(with_vector_diagram=False), blob_storage)
    assert [i.page for i in document.images] == [1]


def test_skips_images_smaller_than_threshold(tmp_path, blob_storage):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(72, 72, 92, 92), stream=make_png(16, 16))  # icon
    page.insert_image(pymupdf.Rect(72, 200, 272, 300), stream=make_png(200, 100))
    page.insert_text((72, 400), "Body text so the page is not empty.", fontsize=11)
    path = tmp_path / "icons.pdf"
    doc.save(path)

    document = _parse(path, blob_storage, min_image_size_px=64)
    assert len(document.images) == 1
    assert document.images[0].width == 200


def test_same_image_placed_twice_is_deduplicated_per_page(tmp_path, blob_storage):
    doc = pymupdf.open()
    page = doc.new_page()
    xref = page.insert_image(pymupdf.Rect(72, 72, 272, 172), stream=make_png(200, 100))
    page.insert_image(pymupdf.Rect(72, 300, 272, 400), xref=xref)
    path = tmp_path / "dupe.pdf"
    doc.save(path)

    document = _parse(path, blob_storage)
    assert len(document.images) == 1


def test_cmyk_image_is_converted_to_png(tmp_path, blob_storage):
    cmyk = pymupdf.Pixmap(pymupdf.csCMYK, pymupdf.IRect(0, 0, 100, 100), False)
    cmyk.set_rect(cmyk.irect, (0, 255, 255, 0))
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(72, 72, 272, 272), pixmap=cmyk)
    path = tmp_path / "cmyk.pdf"
    doc.save(path)

    document = _parse(path, blob_storage)
    data = blob_storage.images[("doc1", document.images[0].image_id)]
    decoded = pymupdf.Pixmap(data)
    assert decoded.colorspace.n == 3


def test_image_with_alpha_channel(tmp_path, blob_storage):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 120), True)
    pix.set_rect(pix.irect, (10, 200, 10, 128))
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(72, 72, 272, 272), stream=pix.tobytes("png"))
    path = tmp_path / "alpha.pdf"
    doc.save(path)

    document = _parse(path, blob_storage)
    assert len(document.images) == 1
    assert blob_storage.images[("doc1", document.images[0].image_id)].startswith(b"\x89PNG")


def test_bold_body_size_line_is_heading_but_bold_sentence_is_not(tmp_path, blob_storage):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Installation", fontsize=11, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 80, 523, 200), "Run the installer. " * 12, fontsize=11)
    page.insert_text((72, 240), "This bold line ends like a sentence.", fontsize=11, fontname="hebo")
    path = tmp_path / "bold.pdf"
    doc.save(path)

    document = _parse(path, blob_storage)
    assert [e.text for e in _of_type(document, ElementType.HEADING)] == ["Installation"]


def test_corrupt_file_raises_document_parsing_error(tmp_path, blob_storage):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7\nthis is not really a pdf")
    with pytest.raises(DocumentParsingError):
        _parse(path, blob_storage)


def test_missing_file_raises_document_parsing_error(tmp_path, blob_storage):
    with pytest.raises(DocumentParsingError):
        _parse(tmp_path / "nope.pdf", blob_storage)


def test_encrypted_pdf_is_rejected(tmp_path, blob_storage):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    path = tmp_path / "enc.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(DocumentParsingError, match="password"):
        _parse(path, blob_storage)


def test_rows_to_markdown_cleans_cells_and_drops_empty_rows_and_columns():
    markdown, rows, cols = rows_to_markdown([["A", None, "B|C"], [None, None, None], ["1\n2", None, "3"]])
    assert (rows, cols) == (2, 2)
    assert markdown == "| A | B\\|C |\n| --- | --- |\n| 1 2 | 3 |"
    assert rows_to_markdown([[None, ""]]) == ("", 0, 0)


def test_join_lines_repairs_hyphenation_but_keeps_real_dashes():
    lines = ["The inter-", "national standard", "", "uses UTF-", "8."]
    assert join_lines(lines) == "The international standard uses UTF- 8."


def test_two_column_layout_reads_left_column_first(tmp_path, blob_storage):
    doc = pymupdf.open()
    page = doc.new_page()
    for i, y in enumerate((100, 200, 300)):
        page.insert_textbox(pymupdf.Rect(50, y, 280, y + 60), f"Left paragraph {i}. More words.", fontsize=11)
        page.insert_textbox(pymupdf.Rect(320, y, 550, y + 60), f"Right paragraph {i}. More.", fontsize=11)
    path = tmp_path / "columns.pdf"
    doc.save(path)

    texts = [e.text.split(".")[0] for e in _parse(path, blob_storage).elements]
    assert texts == [f"Left paragraph {i}" for i in range(3)] + [f"Right paragraph {i}" for i in range(3)]


def test_table_cells_keep_word_spacing_on_sample_manual():
    """Regression: table.extract() glued words ("Relubricaciónderodamientos")."""
    import pathlib

    import pymupdf

    from app.infrastructure.parsing.table_extractor import TableExtractor

    sample = pathlib.Path(__file__).parents[4] / "docs" / "sample" / "Manual_Motor.pdf"
    if not sample.exists():
        import pytest

        pytest.skip("sample manual not available")
    with pymupdf.open(sample) as doc:
        markdown = "\n".join(t.markdown for t in TableExtractor().extract(doc[4]))
    assert "Relubricación de rodamientos" in markdown
    assert "4000 h" in markdown
