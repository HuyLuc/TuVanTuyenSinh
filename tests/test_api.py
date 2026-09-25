import pytest
from fastapi.testclient import TestClient

from tuyensinh.api import main
from tuyensinh.config import Settings
from tuyensinh.telemetry.query_log import QueryLogger


class FakeLLM:
    def __init__(self):
        self.contents = None

    def stream(self, contents, **kw):
        self.contents = contents
        yield "Chưa có "
        yield "dữ liệu."


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake = FakeLLM()
    qlog = QueryLogger(Settings(_env_file=None, data_dir=tmp_path))
    monkeypatch.setattr(main, "get_llm", lambda: fake)
    monkeypatch.setattr(main, "get_query_logger", lambda: qlog)
    return TestClient(main.app), fake, qlog


def test_health(client):
    c, _, _ = client
    assert c.get("/health").json() == {"status": "ok"}


def test_chat_streams_answer_and_logs(client):
    c, fake, qlog = client
    r = c.post(
        "/api/chat",
        json={
            "question": "Còn năm 2025 thì sao?",
            "history": [
                {"role": "user", "content": "Điểm chuẩn CNTT Bách khoa 2026?"},
                {"role": "assistant", "content": "Chưa có dữ liệu."},
            ],
        },
    )
    assert r.status_code == 200
    assert r.text == "Chưa có dữ liệu."
    assert [x.role for x in fake.contents] == ["user", "model", "user"]
    assert "Còn năm 2025" in qlog.path.read_text(encoding="utf-8")


def test_chat_rejects_empty_question(client):
    c, _, _ = client
    assert c.post("/api/chat", json={"question": ""}).status_code == 422


def test_gradio_mounted(client):
    c, _, _ = client
    assert c.get("/").status_code == 200
