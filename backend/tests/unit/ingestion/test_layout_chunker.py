from __future__ import annotations

import uuid

import pytest

from app.domain.models import (
    BoundingBox,
    ChunkType,
    ElementType,
    ImageAsset,
    PageElement,
    ParsedDocument,
)
from app.infrastructure.chunking.layout_chunker import LayoutAwareChunker, approx_tokens, split_sentences
from app.infrastructure.parsing.captions import caption_label, is_caption, mentioned_labels
from app.infrastructure.parsing.pymupdf_parser import PyMuPDFParser


def heading(text, size, y, page=1):
    return PageElement(
        ElementType.HEADING, page, BoundingBox(72, y, 400, y + size), text, font_size=size, is_bold=True
    )


def para(text, y, page=1, height=40):
    return PageElement(ElementType.TEXT, page, BoundingBox(72, y, 520, y + height), text, font_size=11)


def image(image_id, y, page=1, caption=""):
    bbox = BoundingBox(72, y, 300, y + 100)
    element = PageElement(ElementType.IMAGE, page, bbox, caption, image_id=image_id)
    asset = ImageAsset(image_id, "doc", page, bbox, caption=caption)
    return element, asset


def document(elements, images=()):
    return ParsedDocument("doc", "doc.pdf", page_count=3, elements=list(elements), images=list(images))


def body_of(chunk):
    return chunk.text.split("\n\n", 1)[1] if chunk.section_path else chunk.text


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_split_sentences_respects_abbreviations_and_lowercase_continuations():
    text = "See Fig. 3 for details. The value is approx. 5 units, e.g. the default. Next one! Done?"
    assert split_sentences(text) == [
        "See Fig. 3 for details.",
        "The value is approx. 5 units, e.g. the default.",
        "Next one!",
        "Done?",
    ]


def test_approx_tokens():
    assert approx_tokens("one two three four five six seven eight nine ten") == 13
    assert approx_tokens("") == 0


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Figure 3: Pipeline", "figure:3"),
        ("Fig. 3 - Pipeline", "figure:3"),
        ("Figura 3. Arquitectura", "figure:3"),
        ("Diagram 2.1 Flow", "figure:2.1"),
        ("Tabla 4: Estados", "table:4"),
        ("Table of contents", None),
    ],
)
def test_caption_labels(text, label):
    assert caption_label(text) == label
    assert is_caption(text) is (label is not None)


def test_mentioned_labels_normalise_languages():
    assert mentioned_labels("As Fig. 3 and la figura 1 show (see Table 2)") == {
        "figure:3",
        "figure:1",
        "table:2",
    }


# --------------------------------------------------------------------------- #
# Chunking rules
# --------------------------------------------------------------------------- #
def test_section_path_follows_heading_hierarchy():
    doc = document(
        [
            heading("1 Intro", 18, 50),
            para("Intro text.", 80),
            heading("1.1 Scope", 14, 130),
            para("Scope text.", 160),
            heading("1.2 Goals", 14, 210),
            para("Goals text.", 240),
            heading("2 Design", 18, 300),
            para("Design text.", 330),
        ]
    )
    chunks = LayoutAwareChunker().chunk(doc)

    assert [c.section_path for c in chunks] == [
        ["1 Intro"],
        ["1 Intro", "1.1 Scope"],
        ["1 Intro", "1.2 Goals"],
        ["2 Design"],
    ]
    assert chunks[1].text == "Section: 1 Intro > 1.1 Scope\n\nScope text."


def test_packs_sentences_up_to_max_tokens_with_overlap():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(30)]
    doc = document([heading("Long", 18, 40), para(" ".join(sentences), 70, height=400)])

    chunks = LayoutAwareChunker(max_tokens=60, overlap_sentences=1).chunk(doc)

    assert len(chunks) > 3
    for chunk in chunks:
        assert approx_tokens(chunk.text) <= 60
        assert chunk.text.endswith(".")  # never cut mid-sentence
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_sentence = split_sentences(body_of(previous))[-1]
        assert body_of(current).startswith(last_sentence)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)


def test_no_overlap_when_disabled():
    sentences = [f"Sentence {i} is here." for i in range(40)]
    doc = document([para(" ".join(sentences), 70, height=400)])
    chunks = LayoutAwareChunker(max_tokens=30, overlap_sentences=0).chunk(doc)
    assert " ".join(c.text for c in chunks).count("Sentence 5 is here.") == 1


def test_very_long_sentence_is_split_on_words():
    doc = document([para("word " * 1000, 70)])
    chunks = LayoutAwareChunker(max_tokens=100).chunk(doc)
    assert len(chunks) >= 10
    assert all(approx_tokens(c.text) <= 100 for c in chunks)


def test_tables_are_atomic_even_when_large():
    rows = "\n".join(f"| row {i} | value {i} |" for i in range(200))
    table = PageElement(
        ElementType.TABLE, 1, BoundingBox(72, 100, 500, 700), f"| a | b |\n| --- | --- |\n{rows}"
    )
    doc = document([heading("Data", 18, 40), para("Before the table.", 60), table, para("After it.", 720)])

    chunks = LayoutAwareChunker(max_tokens=50, table_row_min_rows=None).chunk(doc)

    assert [c.chunk_type for c in chunks] == [ChunkType.TEXT, ChunkType.TABLE, ChunkType.TEXT]
    assert "| row 199 | value 199 |" in chunks[1].text
    assert chunks[1].text.startswith("Section: Data\n\n| a | b |")


def test_image_chunk_and_spatial_links_sorted_by_distance():
    near_el, near = image("doc_p1_i0", 200, caption="Figure 1: Flow")
    far_el, far = image("doc_p1_i1", 600)
    doc = document(
        [heading("Arch", 18, 40), para("Explains the flow.", 120, height=50), near_el, far_el],
        [near, far],
    )
    chunks = LayoutAwareChunker(image_link_max_distance=150).chunk(doc)

    text_chunk = chunks[0]
    assert [(r.image_id, r.distance) for r in text_chunk.images] == [("doc_p1_i0", 30.0)]
    assert text_chunk.images[0].caption == "Figure 1: Flow"

    image_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMAGE]
    assert len(image_chunks) == 2
    assert image_chunks[0].text == "Section: Arch\n\n[Image] Figure 1: Flow"
    own = image_chunks[0].images[0]
    assert (own.image_id, own.distance) == ("doc_p1_i0", 0.0)
    assert "[Image] Figure on page 1, section 'Arch'." in image_chunks[1].text


def test_images_on_other_pages_are_not_linked_spatially():
    img_el, img = image("doc_p2_i0", 100, page=2)
    doc = document([para("Page one text.", 100), img_el], [img])
    chunks = LayoutAwareChunker().chunk(doc)
    assert chunks[0].images == []


def test_caption_mention_forces_link_with_distance_zero_across_pages():
    img_el, img = image("doc_p3_i0", 100, page=3, caption="Figura 7: Despliegue")
    doc = document([para("The deployment is shown in Fig. 7 below.", 100, page=1), img_el], [img])
    chunks = LayoutAwareChunker().chunk(doc)
    assert [(r.image_id, r.distance) for r in chunks[0].images] == [("doc_p3_i0", 0.0)]


def test_chunk_spanning_pages_reports_page_range_and_first_page_bbox():
    doc = document([para("First part.", 700, page=1), para("Second part.", 60, page=2)])
    chunk = LayoutAwareChunker().chunk(doc)[0]
    assert (chunk.page_start, chunk.page_end) == (1, 2)
    assert chunk.bbox == BoundingBox(72, 700, 520, 740)


def test_chunk_ids_are_deterministic_uuids():
    doc = document([heading("A", 18, 40), para("Text one.", 70), heading("B", 18, 140), para("Two.", 170)])
    first = LayoutAwareChunker().chunk(doc)
    second = LayoutAwareChunker().chunk(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    for chunk in first:
        assert uuid.UUID(chunk.chunk_id).version == 5
        assert (chunk.document_id, chunk.document_name) == ("doc", "doc.pdf")


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        LayoutAwareChunker(max_tokens=5)
    with pytest.raises(ValueError):
        LayoutAwareChunker(overlap_sentences=-1)


# --------------------------------------------------------------------------- #
# End to end: real PDF -> parser -> chunker
# --------------------------------------------------------------------------- #
def test_end_to_end_pdf_chunking(pdf_factory, blob_storage):
    parsed = PyMuPDFParser(blob_storage).parse(
        str(pdf_factory(with_vector_diagram=True)), document_id="doc1", filename="doc.pdf"
    )
    chunks = LayoutAwareChunker(max_tokens=400).chunk(parsed)

    by_type = {t: [c for c in chunks if c.chunk_type == t] for t in ChunkType}
    assert len(by_type[ChunkType.TABLE]) == 1
    assert len(by_type[ChunkType.IMAGE]) == 2
    assert by_type[ChunkType.TABLE][0].section_path == ["1. Architecture Overview", "1.1 Data Model"]

    mention = next(c for c in by_type[ChunkType.TEXT] if "Figura 1" in c.text)
    assert (mention.images[0].image_id, mention.images[0].distance) == ("doc1_p1_i0", 0.0)
    assert mention.images[0].caption.startswith("Figura 1")

    retrieval = next(c for c in by_type[ChunkType.TEXT] if "reciprocal rank fusion" in c.text)
    assert retrieval.section_path == ["2. Retrieval"]
    assert [r.image_id for r in retrieval.images] == ["doc1_p2_i0"]
