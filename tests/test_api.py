import pytest
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def fake_agent(question, file_path=None, task_id=None):
    """Replaces the real agent so tests are instant and free."""
    return f"echo: {question}"


@pytest.fixture(autouse=True)
def use_fake_agent(monkeypatch):
    monkeypatch.setattr(api, "agent", fake_agent)
    monkeypatch.delenv("API_KEY", raising=False)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_answer_returns_agent_output():
    response = client.post("/answer", json={"question": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "echo: hello"
    assert body["duration_seconds"] >= 0


def test_empty_question_is_rejected():
    response = client.post("/answer", json={"question": ""})
    assert response.status_code == 422  # validation error from Pydantic


def test_missing_question_is_rejected():
    response = client.post("/answer", json={})
    assert response.status_code == 422


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/answer", json={"question": "hi"}).status_code == 401
    ok = client.post("/answer", json={"question": "hi"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_upload_passes_file_to_agent_and_cleans_up(monkeypatch):
    seen = {}

    def agent_reading_file(question, file_path=None, task_id=None):
        with open(file_path) as f:
            seen["content"] = f.read()
        seen["path"] = file_path
        return "read it"

    monkeypatch.setattr(api, "agent", agent_reading_file)
    response = client.post(
        "/answer/upload",
        data={"question": "What does the file say?"},
        files={"file": ("note.txt", b"hello from the file", "text/plain")},
    )
    assert response.status_code == 200
    assert seen["content"] == "hello from the file"
    assert seen["path"].endswith(".txt")   # extension preserved for the tools

    import os
    assert not os.path.exists(seen["path"])  # temp file deleted after the request