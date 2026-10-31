from __future__ import annotations

import pytest

from tests.unit.api.conftest import API


@pytest.mark.parametrize("path", ["/health", f"{API}/health"])
def test_liveness(client, path):
    response = client.get(path)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_all_checks_pass(client):
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"redis": True, "qdrant": True}}


def test_not_ready_when_a_check_fails_or_raises(client, container):
    def boom() -> bool:
        raise ConnectionError("qdrant down")

    container.health_checks.update({"redis": lambda: False, "qdrant": boom})

    response = client.get(f"{API}/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "checks": {"redis": False, "qdrant": False}}
