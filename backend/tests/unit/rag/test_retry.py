from __future__ import annotations

import pytest

from app.domain.exceptions import ExternalServiceError, TransientError, ValidationError
from app.infrastructure.resilience.retry import (
    RetryPolicy,
    call_with_retry,
    parse_retry_after,
    retry_with_backoff,
    transient_error,
)


class Flaky:
    def __init__(self, failures: list[BaseException], result: str = "ok") -> None:
        self.failures = list(failures)
        self.calls = 0
        self.result = result

    def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.result


def test_exponential_backoff_without_jitter(fake_sleep) -> None:
    fn = Flaky([TransientError("a"), TransientError("b"), TransientError("c")])
    decorated = retry_with_backoff(4, base_delay=0.5, max_delay=1.5, jitter=False, sleep=fake_sleep)(fn)

    assert decorated() == "ok"
    assert fn.calls == 4
    assert fake_sleep.calls == [0.5, 1.0, 1.5]  # 0.5, 1.0, 2.0 capped at 1.5


def test_full_jitter_uses_random_fraction_of_ceiling(fake_sleep) -> None:
    fn = Flaky([TransientError("a"), TransientError("b")])
    policy = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=10.0, jitter=True)

    call_with_retry(policy, fn, sleep=fake_sleep, rng=lambda: 0.5)

    assert fake_sleep.calls == [0.5, 1.0]


def test_gives_up_after_max_attempts(fake_sleep) -> None:
    fn = Flaky([TransientError(str(i)) for i in range(5)])
    decorated = retry_with_backoff(3, base_delay=0.1, max_delay=1, sleep=fake_sleep)(fn)

    with pytest.raises(TransientError, match="2"):
        decorated()
    assert fn.calls == 3
    assert len(fake_sleep.calls) == 2


@pytest.mark.parametrize("error", [ExternalServiceError("bad request"), ValidationError("x"), KeyError("k")])
def test_non_retryable_errors_pass_through_immediately(fake_sleep, error) -> None:
    fn = Flaky([error])
    decorated = retry_with_backoff(5, sleep=fake_sleep)(fn)

    with pytest.raises(type(error)):
        decorated()
    assert fn.calls == 1
    assert fake_sleep.calls == []


def test_retry_after_is_honoured_as_lower_bound_and_capped(fake_sleep) -> None:
    fn = Flaky([transient_error("429", retry_after=3.0), transient_error("429", retry_after=60.0)])
    policy = RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=8.0, jitter=False)

    call_with_retry(policy, fn, sleep=fake_sleep)

    assert fake_sleep.calls == [3.0, 8.0]


def test_custom_retry_on() -> None:
    sleeps: list[float] = []
    fn = Flaky([ConnectionError("reset")])
    decorated = retry_with_backoff(2, base_delay=0, retry_on=(ConnectionError,), sleep=sleeps.append)(fn)
    assert decorated() == "ok"
    assert fn.calls == 2


def test_decorator_preserves_metadata_and_arguments(fake_sleep) -> None:
    @retry_with_backoff(2, sleep=fake_sleep)
    def add(a: int, b: int = 0) -> int:
        """Adds."""
        return a + b

    assert add(1, b=2) == 3
    assert add.__name__ == "add"
    assert add.__doc__ == "Adds."


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, None), ("", None), ("5", 5.0), ("1.5", 1.5), ("-3", 0.0), ("garbage", None),
     ("Wed, 21 Oct 2015 07:28:00 GMT", 0.0)],
)
def test_parse_retry_after(header, expected) -> None:
    assert parse_retry_after(header) == expected
