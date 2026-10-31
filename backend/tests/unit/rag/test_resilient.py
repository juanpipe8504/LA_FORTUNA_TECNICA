from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.domain.exceptions import CircuitOpenError, ExternalServiceError, TransientError
from app.domain.models import ChatMessage, Role
from app.domain.ports import EmbeddingPort, LLMPort
from app.infrastructure.resilience.circuit_breaker import CircuitBreaker, CircuitState
from app.infrastructure.resilience.resilient import ResilientEmbeddings, ResilientLLM
from app.infrastructure.resilience.retry import RetryPolicy


class ScriptedLLM(LLMPort):
    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple] = []

    def _next(self):
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def generate(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0,
                 max_tokens: int | None = None) -> str:
        self.calls.append(("generate", temperature, max_tokens))
        return self._next()

    def describe_image(self, image: bytes, *, context: str = "") -> str:
        self.calls.append(("describe", image, context))
        return self._next()

    @property
    def supports_vision(self) -> bool:
        return True


class ScriptedEmbeddings(EmbeddingPort):
    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def _next(self, value):
        self.calls += 1
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        return value

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._next([[1.0] for _ in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._next([1.0])

    @property
    def dimension(self) -> int:
        return self._next(1)


POLICY = RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=1.0, jitter=False)
MSG = [ChatMessage(Role.USER, "hi")]


def test_llm_retries_transient_errors_and_forwards_arguments(clock, fake_sleep) -> None:
    inner = ScriptedLLM([TransientError("429"), "answer"])
    llm = ResilientLLM(inner, CircuitBreaker("llm", clock=clock), POLICY, sleep=fake_sleep)

    assert llm.generate(MSG, temperature=0.3, max_tokens=50) == "answer"
    assert inner.calls == [("generate", 0.3, 50)] * 2
    assert fake_sleep.calls == [0.1]
    assert llm.supports_vision is True


def test_exhausted_retries_count_as_a_single_breaker_failure(clock, fake_sleep) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=2, clock=clock)
    inner = ScriptedLLM([TransientError(str(i)) for i in range(3)])
    llm = ResilientLLM(inner, breaker, POLICY, sleep=fake_sleep)

    with pytest.raises(TransientError):
        llm.describe_image(b"img", context="ctx")
    assert len(inner.calls) == 3
    assert breaker.failure_count == 1
    assert breaker.state is CircuitState.CLOSED


def test_open_circuit_fails_fast_without_retrying_or_sleeping(clock, fake_sleep) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=1, recovery_timeout=30, clock=clock)
    inner = ScriptedLLM([ExternalServiceError("400")])
    llm = ResilientLLM(inner, breaker, POLICY, sleep=fake_sleep)

    with pytest.raises(ExternalServiceError):
        llm.generate(MSG)
    assert breaker.state is CircuitState.OPEN

    with pytest.raises(CircuitOpenError):
        llm.generate(MSG)
    assert len(inner.calls) == 1  # non-transient: no retry; open: no call
    assert fake_sleep.calls == []


def test_embeddings_wrapper(clock, fake_sleep) -> None:
    inner = ScriptedEmbeddings([TransientError("timeout")])
    emb = ResilientEmbeddings(inner, CircuitBreaker("emb", clock=clock), POLICY, sleep=fake_sleep)

    assert emb.embed_documents(["a", "b"]) == [[1.0], [1.0]]
    assert emb.embed_query("q") == [1.0]
    assert emb.dimension == 1
    assert inner.calls == 4
    assert emb.breaker.state is CircuitState.CLOSED
