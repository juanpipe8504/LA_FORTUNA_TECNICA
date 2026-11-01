"""Question answering over the indexed documents (retrieve -> ground -> generate -> cite)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from app.application.prompts import (
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
FIGURE_NUMBER_BOOST = 2.0  # the answer names "Figura N" and the image caption is "Figura N"
# A marker-flagged reply still counts as a (partial) grounded answer when, besides the
# refusal sentence(s), it has at least this many characters of content and a valid [n].
MIN_PARTIAL_ANSWER_CHARS = 80

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_FIGURE_WORDS_RE = re.compile(
    r"\b(fig(?:ure|ura)?s?\.?|diagrama?s?|diagrams?|imagen(?:es)?|images?|esquemas?|schematics?|"
    r"gr[aá]fic[oa]s?|charts?)\b",
    re.IGNORECASE,
)
_FIGURE_NUMBER_RE = re.compile(r"\bfig(?:ure|ura)?\.?\s*(\d+)", re.IGNORECASE)
# Lenient: small models sometimes emit "[NO_CONTEXT]" or "[[ NO_CONTEXT ]]".
_MARKER_RE = re.compile(r"\[{1,2}\s*NO[_ ]?CONTEXT\s*\]{1,2}", re.IGNORECASE)
_TRAILING_MARKER_RE = re.compile(_MARKER_RE.pattern + r"$", re.IGNORECASE)
_SENTENCE_SPLIT_RE =re.compile(r"(?<=[.!?\n])\s+")
_REFUSAL_RE = re.compile(
    r"(no\s+(?:tengo|hay|dispongo\s+de|contiene[n]?|incluye[n]?|proporciona[n]?|se\s+encuentra)\b"
    r".{0,80}?\b(?:informaci[oó]n|datos|detalles)"
    r"|(?:informaci[oó]n|datos|contexto)\s+(?:es\s+|son\s+)?(?:in)?suficiente"
    r"|\bnot?\s+(?:have|contain|include|provide)s?\b.{0,60}?\b(?:enough|sufficient|information|details)"
    r"|\b(?:insufficient|not\s+enough)\s+(?:information|context|details))",
    re.IGNORECASE,
)
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


def collapse_table_hits(hits: Sequence[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
    """Keep only the best-scoring chunk per table and return at most ``limit`` hits.

    A table is indexed as a whole-table chunk plus one chunk per row, all sharing
    document, page and bbox; without collapsing, one table could fill the context
    and show up several times in the sources.
    """
    seen: set[tuple[str, int, tuple[float, ...] | None]] = set()
    result: list[RetrievedChunk] = []
    for hit in hits:  # already sorted by score desc
        chunk = hit.chunk
        if chunk.chunk_type is ChunkType.TABLE:
            key = (chunk.document_id, chunk.page_start, tuple(chunk.bbox.to_list()) if chunk.bbox else None)
            if key in seen:
                continue
            seen.add(key)
        result.append(hit)
        if len(result) == limit:
            break
    return result


def strip_marker(text: str) -> tuple[str, bool]:
    """Remove the insufficient-context marker (lenient match). Returns (clean_text, had_marker)."""
    cleaned, count = _MARKER_RE.subn("", text)
    return cleaned.strip(), count > 0


def substantive_content(text: str) -> str:
    """``text`` without its refusal sentences ("no tengo suficiente información...")."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return " ".join(s for s in sentences if s and not _REFUSAL_RE.search(s)).strip()


def is_grounded_answer(text: str, had_marker: bool, n_blocks: int) -> bool:
    """Decide sufficiency from evidence, not only from the marker.

    * no marker -> sufficient when there is any text;
    * marker -> sufficient only when the reply is in fact a (partial) grounded
      answer: beyond the refusal sentence(s) it has substantive content
      (``MIN_PARTIAL_ANSWER_CHARS``) with at least one valid citation [n].
      Small models (e.g. gemma3:4b) often prepend the marker / a disclaimer and
      then answer correctly with citations; flagging that as "no context" would
      hide the sources and figures of a good answer.
    """
    if not text:
        return False
    if not had_marker:
        return True
    body = substantive_content(text)
    if not parse_citations(body, n_blocks):
        return False
    return len(_CITATION_RE.sub("", body).strip()) >= MIN_PARTIAL_ANSWER_CHARS


class MarkerFilter:
    """Incremental marker remover for streamed text.

    Holds back a trailing, still-unclosed ``[...`` run (a possible marker prefix
    such as ``[[NO_CON``) until it can tell whether it is the marker, so the
    marker never reaches the client. Leading whitespace is dropped so the
    streamed text lines up with the final (stripped) answer.
    """

    _MAX_HOLD = 24  # longer than any marker variant

    def __init__(self) -> None:
        self._buffer = ""
        self._started = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        hold_from = len(self._buffer)
        open_at = self._buffer.find("[", self._buffer.rfind("]") + 1)
        if open_at != -1 and len(self._buffer) - open_at <= self._MAX_HOLD:
            hold_from = open_at
        # "[[NO_CONTEXT]" already matches the lenient pattern but may still grow a "]".
        trailing = _TRAILING_MARKER_RE.search(self._buffer)
        if trailing is not None:
            hold_from = min(hold_from, trailing.start())
        ready, self._buffer = self._buffer[:hold_from], self._buffer[hold_from:]
        return self._emit(ready)

    def flush(self) -> str:
        ready, self._buffer = self._buffer, ""
        return self._emit(ready)

    def _emit(self, text: str) -> str:
        text = _MARKER_RE.sub("", text)
        if not self._started:
            text = text.lstrip()
            self._started = bool(text)
        return text


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

    base score = chunk score / (1 + distance / 100)  (an IMAGE chunk's own figure: distance 0)
    x1.5 if the owning chunk was cited, x1.25 if the answer talks about a figure and the
    image is an IMAGE chunk's own figure, x2 if the answer names "Figura N" and the caption
    (or the IMAGE chunk text) is "Figura N".
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
            # An IMAGE chunk links its own figure at distance 0 plus neighbouring images
            # on the page at their real distance: only the former is "this figure".
            distance = max(0.0, ref.distance)
            own_image = is_image_chunk and distance == 0.0
            score = item.score / (1.0 + distance / 100.0)
            if position in cited_set:
                score *= CITED_IMAGE_BOOST
            if mentions_figure and own_image:
                score *= FIGURE_MENTION_BOOST
            caption = ref.caption or (chunk.text if own_image else "")
            if figure_numbers & set(_FIGURE_NUMBER_RE.findall(caption)):
                score *= FIGURE_NUMBER_BOOST
            current = best.get(ref.image_id)
            if current is None or score > current.relevance:
                best[ref.image_id] = ImageCitation(
                    image_id=ref.image_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page=ref.page,
                    caption=caption,
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
# Stream events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TokenEvent:
    """A text delta of the answer being generated (marker already removed).

    Citation numbers are the model's raw ones; the renumbered text (matching
    ``sources``) is in the final event, which clients should render at the end.
    """

    text: str


@dataclass(frozen=True, slots=True)
class FinalEvent:
    """The complete answer: identical to what ``execute`` returns for the same LLM output."""

    answer: Answer


StreamEvent = TokenEvent | FinalEvent


@dataclass(slots=True)
class _Prepared:
    question: str
    started: float
    n_hits: int
    retrieved: list[RetrievedChunk]
    messages: list[ChatMessage]


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
        prepared = self._prepare(question, history, document_ids)
        if not prepared.retrieved:
            return self._no_context_answer(prepared)
        raw = self._llm.generate(
            prepared.messages, temperature=self._temperature, max_tokens=self._max_tokens
        )
        return self._finalize(prepared, raw)

    def stream(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        document_ids: list[str] | None = None,
    ) -> Iterator[StreamEvent]:
        """Same pipeline as :meth:`execute`, yielding ``TokenEvent``s then one ``FinalEvent``.

        The insufficient-context marker never appears in token events. If the
        model produced no visible text (e.g. only the marker), the fallback text
        is emitted as a token, so the streamed text is never empty.
        """
        prepared = self._prepare(question, history, document_ids)
        if not prepared.retrieved:
            answer = self._no_context_answer(prepared)
            yield TokenEvent(answer.text)
            yield FinalEvent(answer)
            return

        marker_filter = MarkerFilter()
        parts: list[str] = []
        emitted = False
        for chunk in self._llm.stream(
            prepared.messages, temperature=self._temperature, max_tokens=self._max_tokens
        ):
            parts.append(chunk)
            delta = marker_filter.feed(chunk)
            if delta:
                emitted = True
                yield TokenEvent(delta)
        tail = marker_filter.flush()
        if tail:
            emitted = True
            yield TokenEvent(tail)

        answer = self._finalize(prepared, "".join(parts))
        if not emitted:
            yield TokenEvent(answer.text)
        yield FinalEvent(answer)

    # ---------------------------------------------------------------- pipeline
    def _prepare(
        self,
        question: str,
        history: Sequence[ChatMessage] | None,
        document_ids: list[str] | None,
    ) -> _Prepared:
        """Validate, retrieve and build the prompt (shared by ``execute`` and ``stream``)."""
        question = (question or "").strip()
        if not question:
            raise ValidationError("The question must not be empty")
        started = self._clock()

        query_vector = self._embeddings.embed_query(question)
        hits = self._vector_store.hybrid_search(
            question, query_vector, top_k=self._top_k * 2, document_ids=document_ids or None
        )
        retrieved = collapse_table_hits([h for h in hits if h.score >= self._min_score], self._top_k)
        messages = self._build_messages(question, history, retrieved) if retrieved else []
        return _Prepared(question, started, len(hits), retrieved, messages)

    def _no_context_answer(self, prepared: _Prepared) -> Answer:
        answer = Answer(text=no_information_message(prepared.question), has_sufficient_context=False)
        self._log(prepared, answer=answer, llm_called=False)
        return answer

    def _finalize(self, prepared: _Prepared, raw: str) -> Answer:
        """Post-process the raw LLM output: marker, sufficiency, citations, sources, images."""
        retrieved = prepared.retrieved
        text, had_marker = strip_marker(raw)
        sufficient = is_grounded_answer(text, had_marker, len(retrieved))
        if not text:
            text = no_information_message(prepared.question)

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
        self._log(prepared, answer=answer, llm_called=True, had_marker=had_marker)
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
        prepared: _Prepared,
        *,
        answer: Answer,
        llm_called: bool,
        had_marker: bool = False,
    ) -> None:
        logger.info(
            "question_answered",
            extra={
                "latency_ms": round((self._clock() - prepared.started) * 1000, 1),
                "n_hits": prepared.n_hits,
                "n_chunks": len(prepared.retrieved),
                "n_sources": len(answer.sources),
                "n_images": len(answer.images),
                "sufficient": answer.has_sufficient_context,
                "had_marker": had_marker,
                "llm_called": llm_called,
            },
        )
