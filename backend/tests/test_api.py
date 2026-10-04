import os
os.environ["KUMAGPT_DATABASE_URL"] = "sqlite:///./data/test_kumagpt.db"
os.environ["KUMAGPT_MODEL_BACKEND"] = "demo"

from fastapi.testclient import TestClient
from app.main import app
from app.config import Settings
from app.services.kumagpt_model import KumaGPTInference


def test_prompt_matches_notebook_chat_template():
    model = KumaGPTInference(Settings(model_backend="demo"))
    prompt = model._build_prompt([
        {"role": "user", "content": "日本の首都は？"},
        {"role": "assistant", "content": "日本の首都は東京です。"},
        {"role": "user", "content": "人口は？"},
    ])
    assert prompt == (
        "<user>日本の首都は？\n"
        "<assistant>日本の首都は東京です。<|endoftext|>\n"
        "<user>人口は？\n"
        "<assistant>"
    )


def test_saved_and_temporary_chat():
    with TestClient(app) as client:
        chat = client.post("/api/v1/chats", json={}).json()
        reply = client.post(f"/api/v1/chats/{chat['id']}/messages", json={"content": "こんにちは"})
        assert reply.status_code == 201
        assert client.get(f"/api/v1/chats/{chat['id']}/messages").json()["total"] == 2
        before = client.get("/api/v1/chats").json()["total"]
        temporary = client.post("/api/v1/temporary/messages", json={"content": "一時会話", "history": []})
        assert temporary.status_code == 200
        assert client.get("/api/v1/chats").json()["total"] == before
        assert client.delete(f"/api/v1/chats/{chat['id']}").status_code == 204


def test_sampling_parameters_are_validated_for_both_endpoints():
    with TestClient(app) as client:
        chat = client.post("/api/v1/chats", json={}).json()
        valid = {"content": "こんにちは", "temperature": 1.2, "top_k": 50}
        assert client.post(f"/api/v1/chats/{chat['id']}/messages", json=valid).status_code == 201
        assert client.post("/api/v1/temporary/messages", json={**valid, "history": []}).status_code == 200
        assert client.post(
            "/api/v1/temporary/messages",
            json={"content": "範囲外", "history": [], "temperature": 0, "top_k": 101},
        ).status_code == 422
