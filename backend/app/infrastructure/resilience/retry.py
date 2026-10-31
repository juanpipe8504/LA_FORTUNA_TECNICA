"""Retry with exponential backoff and *full jitter*.

Delay after failed attempt ``n`` (1-based) is
``uniform(0, min(max_delay, base_delay * 2**(n-1)))`` (AWS "full jitter"), which
spreads retries from concurrent workers and avoids thundering herds against a
rate-limited provider. When the raised error carries a ``retry_after`` attribute
(parsed from the ``Retry-After`` header by the adapters) that value is used as a
lower bound, capped at ``max_delay`` so an interactive request never blocks for
minutes.
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import ParamSpec, TypeVar

from app.core.logging import get_logger
from app.domain.exceptions import TransientError

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

SleepFn = Callable[[float], None]
RandomFn = Callable[[], float]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: bool = True
    retry_on: tuple[type[BaseException], ...] = field(default=(TransientError,))

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be >= 0")

    def backoff(self, attempt: int, rng: RandomFn = random.random) -> float:
        """Delay to wait after the failed ``attempt`` (1-based), before retry hints."""
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return rng() * ceiling if self.jitter else ceiling

    def delay_for(self, attempt: int, error: BaseException, rng: RandomFn = random.random) -> float:
        delay = self.backoff(attempt, rng)
        retry_after = getattr(error, "retry_after", None)
        if isinstance(retry_after, int | float) and retry_after > 0:
            delay = max(delay, min(float(retry_after), self.max_delay))
        return delay


def call_with_retry(
    policy: RetryPolicy,
    fn: Callable[P, R],
    *args: P.args,
    sleep: SleepFn = time.sleep,
    rng: RandomFn = random.random,
    **kwargs: P.kwargs,
) -> R:
    """Invoke ``fn`` retrying the errors listed in ``policy.retry_on``."""
    name = getattr(fn, "__qualname__", repr(fn))
    attempt = 1
    while True:
        try:
            return fn(*args, **kwargs)
        except policy.retry_on as exc:
            if attempt >= policy.max_attempts:
                logger.warning(
                    "retry_exhausted",
                    extra={"operation": name, "attempts": attempt, "error": str(exc)},
                )
                raise
            delay = policy.delay_for(attempt, exc, rng)
            logger.info(
                "retrying_after_error",
                extra={
                    "operation": name,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "delay_s": round(delay, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            sleep(delay)
            attempt += 1


def retry_with_backoff(
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[BaseException], ...] = (TransientError,),
    *,
    jitter: bool = True,
    sleep: SleepFn = time.sleep,
    rng: RandomFn = random.random,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator form of :func:`call_with_retry`."""
    policy = RetryPolicy(max_attempts, base_delay, max_delay, jitter, retry_on)

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return call_with_retry(policy, fn, *args, sleep=sleep, rng=rng, **kwargs)

        return wrapper

    return decorator


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) into seconds."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - datetime.now(UTC)).total_seconds()
    return max(0.0, seconds)


def transient_error(message: str, retry_after: float | None = None) -> TransientError:
    """Build a ``TransientError`` carrying an optional ``retry_after`` hint (seconds)."""
    error = TransientError(message)
    error.retry_after = retry_after  # type: ignore[attr-defined]
    return error
