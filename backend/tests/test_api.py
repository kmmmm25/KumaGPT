import os
os.environ["KUMAGPT_DATABASE_URL"] = "sqlite:///./data/test_kumagpt.db"
os.environ["KUMAGPT_MODEL_BACKEND"] = "demo"

from fastapi.testclient import TestClient
from app.main import app


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
