"""Thread-safe circuit breaker (CLOSED -> OPEN -> HALF_OPEN -> CLOSED/OPEN).

* CLOSED: calls pass; consecutive ``ExternalServiceError`` failures are counted
  and ``failure_threshold`` of them open the circuit.
* OPEN: calls fail fast with ``CircuitOpenError`` until ``recovery_timeout``
  seconds have elapsed.
* HALF_OPEN: exactly one trial call is let through; success closes the circuit,
  failure re-opens it. Concurrent callers are rejected while the trial runs.

Only dependency failures (``ExternalServiceError`` and subclasses) count: a
programming or validation error must not take a healthy provider offline.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import ParamSpec, TypeVar

from app.core.logging import get_logger
from app.domain.exceptions import CircuitOpenError, ExternalServiceError

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._trial_in_flight = False

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failures

    def _transition(self, new_state: CircuitState) -> None:
        if new_state is self._state:
            return
        logger.warning(
            "circuit_state_change",
            extra={
                "circuit": self.name,
                "from_state": self._state.value,
                "to_state": new_state.value,
                "failures": self._failures,
            },
        )
        self._state = new_state

    def _maybe_half_open(self) -> None:
        if (
            self._state is CircuitState.OPEN
            and self._clock() - self._opened_at >= self._recovery_timeout
        ):
            self._trial_in_flight = False
            self._transition(CircuitState.HALF_OPEN)

    # ------------------------------------------------------------------- call
    def _before_call(self) -> bool:
        """Admit or reject a call. Returns True when it is the half-open trial."""
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.OPEN:
                remaining = self._recovery_timeout - (self._clock() - self._opened_at)
                raise CircuitOpenError(
                    f"Service '{self.name}' is temporarily unavailable "
                    f"(circuit open, retry in {max(0.0, remaining):.0f}s)"
                )
            if self._state is CircuitState.HALF_OPEN:
                if self._trial_in_flight:
                    raise CircuitOpenError(
                        f"Service '{self.name}' is temporarily unavailable (recovery probe in progress)"
                    )
                self._trial_in_flight = True
                return True
            return False

    def _on_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._trial_in_flight = False
            self._transition(CircuitState.CLOSED)

    def _on_failure(self, is_trial: bool) -> None:
        with self._lock:
            self._failures += 1
            self._trial_in_flight = False
            if is_trial or self._failures >= self._failure_threshold:
                self._opened_at = self._clock()
                self._transition(CircuitState.OPEN)

    def _on_ignored_error(self, is_trial: bool) -> None:
        if is_trial:
            with self._lock:
                self._trial_in_flight = False

    def call(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        is_trial = self._before_call()
        try:
            result = fn(*args, **kwargs)
        except ExternalServiceError:
            self._on_failure(is_trial)
            raise
        except BaseException:
            self._on_ignored_error(is_trial)
            raise
        self._on_success()
        return result

    def record_failure(self) -> None:
        """Count a dependency failure observed outside :meth:`call`.

        Used for streamed responses: :meth:`call` only covers opening the stream
        (up to the first chunk), so a failure in the middle of the stream is
        reported here.
        """
        self._on_failure(is_trial=False)

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._trial_in_flight = False
            self._transition(CircuitState.CLOSED)
