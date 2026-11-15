"""Веб-приложение (авторизация) и резервный провайдер LLM — без сетевых вызовов."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app import config, llm
from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "APP_USER", "admin")
    monkeypatch.setattr(config, "APP_PASSWORD", "secret")
    return TestClient(app)


def test_pages_require_password(client):
    assert client.get("/").status_code == 401
    assert client.get("/", auth=("admin", "wrong")).status_code == 401
    assert client.get("/", auth=("admin", "secret")).status_code == 200


def test_health_is_open(client):
    assert client.get("/health").json() == {"ok": True}


def test_no_password_means_local_mode(monkeypatch):
    monkeypatch.setattr(config, "APP_PASSWORD", "")
    assert TestClient(app).get("/").status_code == 200


def test_bad_run_id_is_rejected(client):
    resp = client.get("/editor/..%2F..%2Fapp", auth=("admin", "secret"))
    assert resp.status_code in (400, 404)


class Answer(BaseModel):
    value: str


def _failing_openai():
    def parse(**_kwargs):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))

    return SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    )


def test_fallback_to_anthropic_on_openai_outage(monkeypatch):
    calls = {}

    def anthropic_parse(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=Answer(value="ok"))

    fake_anthropic = SimpleNamespace(messages=SimpleNamespace(parse=anthropic_parse))
    monkeypatch.setattr(llm, "_client", _failing_openai)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: fake_anthropic)

    result = llm.call_structured("gpt-4o", "system", "user", Answer)

    assert result == Answer(value="ok")
    assert calls["output_format"] is Answer
    assert "temperature" not in calls  # текущие модели Claude не принимают сэмплинг


def test_fallback_refusal_raises(monkeypatch):
    fake_anthropic = SimpleNamespace(
        messages=SimpleNamespace(
            parse=lambda **_: SimpleNamespace(stop_reason="refusal", parsed_output=None)
        )
    )
    monkeypatch.setattr(llm, "_client", _failing_openai)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: fake_anthropic)
    with pytest.raises(RuntimeError):
        llm.call_structured("gpt-4o", "system", "user", Answer)


def test_without_anthropic_key_openai_error_propagates(monkeypatch):
    monkeypatch.setattr(llm, "_client", _failing_openai)
    monkeypatch.setattr(llm, "_anthropic_client", lambda: None)
    with pytest.raises(openai.APIConnectionError):
        llm.call_structured("gpt-4o", "system", "user", Answer)
