from __future__ import annotations

import threading

import pytest

from app.domain.exceptions import CircuitOpenError, ExternalServiceError, TransientError, ValidationError
from app.infrastructure.resilience.circuit_breaker import CircuitBreaker, CircuitState


def _fail(exc: BaseException):
    def fn():
        raise exc

    return fn


def _trip(breaker: CircuitBreaker, n: int) -> None:
    for _ in range(n):
        with pytest.raises(ExternalServiceError):
            breaker.call(_fail(TransientError("down")))


def test_closed_passes_calls_and_resets_failures_on_success(clock) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=10, clock=clock)
    _trip(breaker, 2)
    assert breaker.failure_count == 2
    assert breaker.call(lambda x: x * 2, 21) == 42
    assert breaker.failure_count == 0
    assert breaker.state is CircuitState.CLOSED


def test_opens_after_threshold_and_fails_fast(clock) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=10, clock=clock)
    _trip(breaker, 3)
    assert breaker.state is CircuitState.OPEN

    called = []
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: called.append(1))
    assert called == []


def test_half_open_after_timeout_then_closes_on_success(clock) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=1, recovery_timeout=10, clock=clock)
    _trip(breaker, 1)
    clock.advance(9.9)
    assert breaker.state is CircuitState.OPEN
    clock.advance(0.1)
    assert breaker.state is CircuitState.HALF_OPEN

    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.state is CircuitState.CLOSED


def test_half_open_failure_reopens_immediately(clock) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=3, recovery_timeout=10, clock=clock)
    _trip(breaker, 3)
    clock.advance(10)
    assert breaker.state is CircuitState.HALF_OPEN

    _trip(breaker, 1)  # one trial failure is enough
    assert breaker.state is CircuitState.OPEN
    clock.advance(5)
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "ok")  # the recovery timer restarted


def test_half_open_allows_a_single_concurrent_trial(clock) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=1, recovery_timeout=1, clock=clock)
    _trip(breaker, 1)
    clock.advance(1)

    started, release = threading.Event(), threading.Event()
    results: list[str] = []

    def slow_trial() -> str:
        started.set()
        release.wait(2)
        return "trial"

    worker = threading.Thread(target=lambda: results.append(breaker.call(slow_trial)))
    worker.start()
    started.wait(2)
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "second")
    release.set()
    worker.join(2)
    assert results == ["trial"]
    assert breaker.state is CircuitState.CLOSED


@pytest.mark.parametrize("error", [ValidationError("bad"), ValueError("bug")])
def test_non_dependency_errors_do_not_count(clock, error) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=1, clock=clock)
    with pytest.raises(type(error)):
        breaker.call(_fail(error))
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_non_dependency_error_in_half_open_releases_trial(clock) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=1, recovery_timeout=1, clock=clock)
    _trip(breaker, 1)
    clock.advance(1)
    with pytest.raises(ValueError):
        breaker.call(_fail(ValueError("bug")))
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.call(lambda: 1) == 1


def test_transitions_are_logged(clock, caplog) -> None:
    breaker = CircuitBreaker("embeddings", failure_threshold=1, recovery_timeout=1, clock=clock)
    with caplog.at_level("WARNING"):
        _trip(breaker, 1)
    record = next(r for r in caplog.records if r.getMessage() == "circuit_state_change")
    assert record.circuit == "embeddings"
    assert (record.from_state, record.to_state) == ("closed", "open")


def test_reset(clock) -> None:
    breaker = CircuitBreaker("llm", failure_threshold=1, clock=clock)
    _trip(breaker, 1)
    breaker.reset()
    assert breaker.state is CircuitState.CLOSED


def test_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker("x", failure_threshold=0)
