"""Question answering over the indexed documents (retrieve -> ground -> generate -> cite)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence

from app.application.prompts import (
    INSUFFICIENT_CONTEXT_MARKER,
    SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)
from app.core.logging import get_logger
from app.domain.exceptions import ValidationError
from app.domain.models import (
    Answer,
    ChatMessage,
    ChunkType,
    ImageCitation,
    RetrievedChunk,
    Role,
    SourceCitation,
)
from app.domain.ports import EmbeddingPort, LLMPort, VectorStorePort

logger = get_logger(__name__)

NO_INFO_ES = (
    "No tengo suficiente información en los documentos indexados para responder a esa pregunta. "
    "Prueba a reformularla o a subir el documento que contenga esa información."
)
NO_INFO_EN = (
    "I don't have enough information in the indexed documents to answer that question. "
    "Try rephrasing it or upload the document that contains this information."
)

SNIPPET_CHARS = 240
CITED_IMAGE_BOOST = 1.5
FIGURE_MENTION_BOOST = 1.25
FIGURE_NUMBER_BOOST = 1.5

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_FIGURE_WORDS_RE = re.compile(
    r"\b(fig(?:ure|ura)?s?\.?|diagrama?s?|diagrams?|imagen(?:es)?|images?|esquemas?|schematics?|"
    r"gr[aá]fic[oa]s?|charts?)\b",
    re.IGNORECASE,
)
_FIGURE_NUMBER_RE = re.compile(r"\bfig(?:ure|ura)?\.?\s*(\d+)", re.IGNORECASE)
_SPANISH_HINT_RE = re.compile(
    r"[¿¡ñáéíóú]|\b(qu[eé]|c[oó]mo|cu[aá]l(?:es)?|cu[aá]nto|d[oó]nde|por qu[eé]|el|la|los|las|"
    r"de|del|para|es|son|un|una)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-testable)
# --------------------------------------------------------------------------- #
def looks_spanish(text: str) -> bool:
    return len(_SPANISH_HINT_RE.findall(text)) >= 2 or bool(re.search(r"[¿¡ñ]", text))


def no_information_message(question: str) -> str:
    return NO_INFO_ES if looks_spanish(question) else NO_INFO_EN


def parse_citations(text: str, n_blocks: int) -> list[int]:
    """Return the distinct, valid 1-based block numbers cited as [n] / [n, m], in order."""
    seen: list[int] = []
    for group in _CITATION_RE.findall(text):
        for raw in group.split(","):
            number = int(raw.strip())
            if 1 <= number <= n_blocks and number not in seen:
                seen.append(number)
    return seen


def renumber_citations(text: str, cited: Sequence[int]) -> str:
    """Rewrite [n] markers so that the k-th cited block becomes [k].

    Sources are returned in ``cited`` order, so after renumbering ``[n]`` in the
    answer always points to ``sources[n - 1]``. Invalid numbers are dropped.
    """
    mapping = {old: new for new, old in enumerate(cited, start=1)}

    def _replace(match: re.Match[str]) -> str:
        numbers = [mapping.get(int(raw.strip())) for raw in match.group(1).split(",")]
        valid = [str(n) for n in numbers if n is not None]
        return f"[{', '.join(valid)}]" if valid else ""

    return _CITATION_RE.sub(_replace, text)


def strip_marker(text: str) -> tuple[str, bool]:
    """Remove the insufficient-context marker. Returns (clean_text, had_marker)."""
    had_marker = INSUFFICIENT_CONTEXT_MARKER in text
    return text.replace(INSUFFICIENT_CONTEXT_MARKER, "").strip(), had_marker


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "…"


def trim_history(history: Sequence[ChatMessage] | None, max_turns: int) -> list[ChatMessage]:
    """Keep the last ``max_turns`` turns (1 turn = user + assistant message), no system msgs."""
    if not history or max_turns <= 0:
        return []
    messages = [
        ChatMessage(m.role, m.content)  # images are never replayed from history
        for m in history
        if m.role in (Role.USER, Role.ASSISTANT) and m.content.strip()
    ]
    return messages[-2 * max_turns :]


def rank_images(
    retrieved: Sequence[RetrievedChunk],
    *,
    cited: Sequence[int],
    answer_text: str,
    max_images: int,
) -> list[ImageCitation]:
    """Score images linked to the retrieved chunks and return the top ``max_images``.

    base score = chunk score / (1 + distance / 100)  (IMAGE chunks: distance 0)
    x1.5 if the owning chunk was cited, x1.25 if the answer talks about a figure and
    the chunk is an IMAGE chunk, x1.5 if the answer names "Figura N" and the caption matches.
    """
    if max_images <= 0:
        return []
    cited_set = set(cited)
    mentions_figure = bool(_FIGURE_WORDS_RE.search(answer_text))
    figure_numbers = set(_FIGURE_NUMBER_RE.findall(answer_text))

    best: dict[str, ImageCitation] = {}
    for position, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        is_image_chunk = chunk.chunk_type is ChunkType.IMAGE
        for ref in chunk.images:
            distance = 0.0 if is_image_chunk else max(0.0, ref.distance)
            score = item.score / (1.0 + distance / 100.0)
            if position in cited_set:
                score *= CITED_IMAGE_BOOST
            if mentions_figure and is_image_chunk:
                score *= FIGURE_MENTION_BOOST
            caption_numbers = set(_FIGURE_NUMBER_RE.findall(ref.caption))
            if figure_numbers & caption_numbers:
                score *= FIGURE_NUMBER_BOOST
            current = best.get(ref.image_id)
            if current is None or score > current.relevance:
                best[ref.image_id] = ImageCitation(
                    image_id=ref.image_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page=ref.page,
                    caption=ref.caption or (chunk.text if is_image_chunk else ""),
                    bbox=ref.bbox,
                    related_chunk_id=chunk.chunk_id,
                    relevance=score,
                )
    result = sorted(best.values(), key=lambda c: c.relevance, reverse=True)[:max_images]
    for citation in result:
        citation.relevance = round(citation.relevance, 6)
        citation.caption = make_snippet(citation.caption, 300)
    return result


# --------------------------------------------------------------------------- #
# Use case
# --------------------------------------------------------------------------- #
class AnswerQuestionUseCase:
    def __init__(
        self,
        embeddings: EmbeddingPort,
        vector_store: VectorStorePort,
        llm: LLMPort,
        top_k: int = 6,
        min_score: float = 0.0,
        max_images: int = 2,
        history_max_turns: int = 6,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score
        self._max_images = max_images
        self._history_max_turns = history_max_turns
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._clock = clock

    def execute(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        document_ids: list[str] | None = None,
    ) -> Answer:
        question = (question or "").strip()
        if not question:
            raise ValidationError("The question must not be empty")
        started = self._clock()

        query_vector = self._embeddings.embed_query(question)
        hits = self._vector_store.hybrid_search(
            question, query_vector, top_k=self._top_k, document_ids=document_ids or None
        )
        retrieved = [h for h in hits if h.score >= self._min_score]

        if not retrieved:
            answer = Answer(text=no_information_message(question), has_sufficient_context=False)
            self._log(started, n_hits=len(hits), retrieved=retrieved, answer=answer, llm_called=False)
            return answer

        messages = self._build_messages(question, history, retrieved)
        raw = self._llm.generate(messages, temperature=self._temperature, max_tokens=self._max_tokens)
        text, had_marker = strip_marker(raw)
        sufficient = not had_marker and bool(text)
        if not text:
            text = no_information_message(question)

        cited = parse_citations(text, len(retrieved))
        source_hits = [retrieved[i - 1] for i in cited] if cited else retrieved
        images = (
            rank_images(retrieved, cited=cited, answer_text=text, max_images=self._max_images)
            if sufficient
            else []
        )
        if cited:
            text = renumber_citations(text, cited)
        answer = Answer(
            text=text,
            has_sufficient_context=sufficient,
            sources=[self._to_source(h) for h in source_hits],
            images=images,
        )
        self._log(started, n_hits=len(hits), retrieved=retrieved, answer=answer, llm_called=True)
        return answer

    # ---------------------------------------------------------------- helpers
    def _build_messages(
        self,
        question: str,
        history: Sequence[ChatMessage] | None,
        retrieved: Sequence[RetrievedChunk],
    ) -> list[ChatMessage]:
        language = "Spanish" if looks_spanish(question) else None
        user_prompt = build_user_prompt(question, build_context_block(retrieved), language)
        return [
            ChatMessage(Role.SYSTEM, SYSTEM_PROMPT),
            *trim_history(history, self._history_max_turns),
            ChatMessage(Role.USER, user_prompt),
        ]

    @staticmethod
    def _to_source(hit: RetrievedChunk) -> SourceCitation:
        chunk = hit.chunk
        return SourceCitation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page=chunk.page_start,
            score=round(hit.score, 6),
            snippet=make_snippet(chunk.text),
            bbox=chunk.bbox,
        )

    def _log(
        self,
        started: float,
        *,
        n_hits: int,
        retrieved: Sequence[RetrievedChunk],
        answer: Answer,
        llm_called: bool,
    ) -> None:
        logger.info(
            "question_answered",
            extra={
                "latency_ms": round((self._clock() - started) * 1000, 1),
                "n_hits": n_hits,
                "n_chunks": len(retrieved),
                "n_sources": len(answer.sources),
                "n_images": len(answer.images),
                "sufficient": answer.has_sufficient_context,
                "llm_called": llm_called,
            },
        )
