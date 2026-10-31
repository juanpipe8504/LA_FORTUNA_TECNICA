"""Resilience decorators for the LLM and embedding ports.

Composition order: ``breaker.call(retrying_fn)`` - the retry loop runs *inside*
the circuit breaker. Rationale:

* one logical operation (including all of its retries) counts as a single
  breaker failure, so a short burst of 429s absorbed by retries never trips it;
* once the circuit is open no retries (and no sleeps) happen at all: callers
  fail fast with ``CircuitOpenError`` (HTTP 503) instead of piling up threads;
* ``CircuitOpenError`` is never retried because it is raised outside the loop.

Both classes implement the same port as the object they wrap (Decorator
pattern), so the application layer is unaware of resilience concerns.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

from app.domain.models import ChatMessage
from app.domain.ports import EmbeddingPort, LLMPort
from app.infrastructure.resilience.circuit_breaker import CircuitBreaker
from app.infrastructure.resilience.retry import RandomFn, RetryPolicy, SleepFn, call_with_retry

R = TypeVar("R")


class ResilientCaller:
    """Runs a zero-arg callable as ``breaker.call(retry(fn))``."""

    def __init__(
        self,
        breaker: CircuitBreaker,
        policy: RetryPolicy,
        sleep: SleepFn = time.sleep,
        rng: RandomFn = random.random,
    ) -> None:
        self.breaker = breaker
        self.policy = policy
        self._sleep = sleep
        self._rng = rng

    def __call__(self, fn: Callable[[], R]) -> R:
        return self.breaker.call(call_with_retry, self.policy, fn, sleep=self._sleep, rng=self._rng)


class ResilientLLM(LLMPort):
    def __init__(
        self,
        inner: LLMPort,
        breaker: CircuitBreaker,
        policy: RetryPolicy,
        *,
        sleep: SleepFn = time.sleep,
        rng: RandomFn = random.random,
    ) -> None:
        self._inner = inner
        self._call = ResilientCaller(breaker, policy, sleep, rng)

    @property
    def breaker(self) -> CircuitBreaker:
        return self._call.breaker

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        return self._call(
            lambda: self._inner.generate(messages, temperature=temperature, max_tokens=max_tokens)
        )

    def describe_image(self, image: bytes, *, context: str = "") -> str:
        return self._call(lambda: self._inner.describe_image(image, context=context))

    @property
    def supports_vision(self) -> bool:
        return self._inner.supports_vision


class ResilientEmbeddings(EmbeddingPort):
    def __init__(
        self,
        inner: EmbeddingPort,
        breaker: CircuitBreaker,
        policy: RetryPolicy,
        *,
        sleep: SleepFn = time.sleep,
        rng: RandomFn = random.random,
    ) -> None:
        self._inner = inner
        self._call = ResilientCaller(breaker, policy, sleep, rng)

    @property
    def breaker(self) -> CircuitBreaker:
        return self._call.breaker

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._call(lambda: self._inner.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._call(lambda: self._inner.embed_query(text))

    @property
    def dimension(self) -> int:
        # May trigger a remote probe (Ollama): protect it like any other call.
        return self._call(lambda: self._inner.dimension)
