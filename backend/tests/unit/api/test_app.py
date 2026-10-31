"""App factory, middleware (request id, access log) and lifespan behaviour."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.api import main as main_module
from app.api.main import create_app
from app.api.middleware import resolve_request_id
from app.core import logging as logging_module
from tests.unit.api.conftest import API


def test_generates_request_id_when_missing(client):
    response = client.get("/health")

    assert len(response.headers["x-request-id"]) == 32


def test_propagates_valid_incoming_request_id(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-42"})

    assert response.headers["x-request-id"] == "trace-42"


@pytest.mark.parametrize("bad", ["", "with space", "x" * 200, "inject\nline"])
def test_rejects_malformed_request_ids(bad):
    assert resolve_request_id(bad) != bad


def test_access_log_contains_method_path_status_and_duration(client, caplog):
    with caplog.at_level(logging.INFO, logger="app.api.access"):
        client.get("/health", headers={"X-Request-ID": "log-1"})

    record = next(r for r in caplog.records if r.name == "app.api.access")
    assert record.method == "GET"
    assert record.path == "/health"
    assert record.status_code == 200
    assert record.duration_ms >= 0


def test_request_id_is_bound_to_log_context_inside_handlers(client, container):
    seen: list[dict] = []

    def capture() -> list:
        seen.append(dict(logging_module._context.get()))
        return []

    container.list_documents.execute.side_effect = capture

    client.get(f"{API}/documents", headers={"X-Request-ID": "ctx-1"})

    assert seen == [{"request_id": "ctx-1"}]


def test_unknown_route_uses_error_envelope(client):
    response = client.get(f"{API}/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_cors_allows_configured_origin(client, settings):
    origin = settings.cors_origins[0]

    response = client.options(
        f"{API}/chat",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_openapi_metadata(client, settings):
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == settings.app_name
    assert f"{API}/chat" in schema["paths"]
    assert "/health/ready" in schema["paths"]


def test_injected_container_is_used_and_not_closed(container, monkeypatch):
    def fail() -> None:
        raise AssertionError("must not build the real container")

    monkeypatch.setattr(main_module, "get_container", fail)
    closed: list[bool] = []
    container.closers.append(lambda: closed.append(True))
    app = create_app(container=container)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert app.state.container is container
    assert closed == []


def test_lifespan_builds_container_lazily_and_closes_it(container, monkeypatch):
    calls: list[str] = []
    container.closers.append(lambda: calls.append("closed"))

    def fake_get_container():
        calls.append("built")
        return container

    monkeypatch.setattr(main_module, "get_container", fake_get_container)
    monkeypatch.setattr(main_module, "get_settings", lambda: container.settings)

    app = create_app()
    assert calls == []  # nothing is built at import/creation time

    with TestClient(app) as client:
        assert calls == ["built"]
        assert app.state.container is container
        assert client.get("/health").status_code == 200
    assert calls == ["built", "closed"]
