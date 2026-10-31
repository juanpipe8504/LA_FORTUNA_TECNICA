"""Figure/table caption detection and label normalisation.

Shared by the parser (which attaches captions to images/tables) and the chunker
(which links chunks that *mention* "Figure 3" to the image captioned "Figure 3").
Supports English and Spanish labels.
"""

from __future__ import annotations

import re

_FIGURE_WORDS = r"fig(?:ure|ura)?\.?|diagram(?:a)?|chart|gr[aá]fico|image|imagen|illustration|ilustraci[oó]n"
_TABLE_WORDS = r"table|tabla"
_NUMBER = r"\d+(?:[.\-]\d+)*"

CAPTION_START = re.compile(
    rf"^\s*(?P<kind>{_FIGURE_WORDS}|{_TABLE_WORDS})\s*(?P<num>{_NUMBER})\s*(?:[:.\-–—)]|\s|$)",
    re.IGNORECASE,
)
LABEL_MENTION = re.compile(rf"\b(?P<kind>{_FIGURE_WORDS}|{_TABLE_WORDS})\s*(?P<num>{_NUMBER})", re.IGNORECASE)


def _kind(raw: str) -> str:
    return "table" if raw.lower().startswith("tab") else "figure"


def _normalise_number(raw: str) -> str:
    return raw.rstrip(".-")


def is_caption(text: str) -> bool:
    """True if ``text`` looks like a caption, e.g. ``"Figura 1: Arquitectura"``."""
    return bool(CAPTION_START.match(text)) and len(text) <= 500


def caption_label(text: str) -> str | None:
    """Normalised label of a caption (``"Fig. 3 - Foo"`` -> ``"figure:3"``) or None."""
    match = CAPTION_START.match(text)
    if not match:
        return None
    return f"{_kind(match['kind'])}:{_normalise_number(match['num'])}"


def mentioned_labels(text: str) -> set[str]:
    """All normalised labels referenced in free text ("as shown in Fig. 3" -> {"figure:3"})."""
    return {f"{_kind(m['kind'])}:{_normalise_number(m['num'])}" for m in LABEL_MENTION.finditer(text)}
