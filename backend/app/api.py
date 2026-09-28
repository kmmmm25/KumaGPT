from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import get_db
from .models import Chat, Message, utcnow
from .schemas import (
    ChatCreate, ChatRead, ChatUpdate, MessageRead, Page, SendMessage,
    SendMessageResponse, TemporaryMessage, TemporaryMessageRead,
)

router = APIRouter()


def find_chat(db: Session, chat_id: int) -> Chat:
    chat = db.get(Chat, chat_id)
    if chat is None:
        raise HTTPException(404, "チャットが見つかりません")
    return chat


@router.post("/chats", response_model=ChatRead, status_code=201)
def create_chat(payload: ChatCreate, db: Session = Depends(get_db)):
    chat = Chat(title=(payload.title or "新しいチャット").strip() or "新しいチャット")
    db.add(chat); db.commit(); db.refresh(chat)
    return chat


@router.get("/chats", response_model=Page)
def list_chats(limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    items = db.scalars(select(Chat).order_by(Chat.updated_at.desc(), Chat.id.desc()).limit(limit).offset(offset)).all()
    total = db.scalar(select(func.count()).select_from(Chat)) or 0
    return Page(items=[ChatRead.model_validate(item) for item in items], total=total, limit=limit, offset=offset)


@router.get("/chats/{chat_id}", response_model=ChatRead)
def get_chat(chat_id: int, db: Session = Depends(get_db)):
    return find_chat(db, chat_id)


@router.patch("/chats/{chat_id}", response_model=ChatRead)
def update_chat(chat_id: int, payload: ChatUpdate, db: Session = Depends(get_db)):
    chat = find_chat(db, chat_id); chat.title = payload.title; chat.updated_at = utcnow()
    db.commit(); db.refresh(chat); return chat


@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    db.delete(find_chat(db, chat_id)); db.commit(); return Response(status_code=204)


@router.get("/chats/{chat_id}/messages", response_model=Page)
def list_messages(chat_id: int, limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    find_chat(db, chat_id)
    condition = Message.chat_id == chat_id
    items = db.scalars(select(Message).where(condition).order_by(Message.created_at, Message.id).limit(limit).offset(offset)).all()
    total = db.scalar(select(func.count()).select_from(Message).where(condition)) or 0
    return Page(items=[MessageRead.model_validate(item) for item in items], total=total, limit=limit, offset=offset)


@router.post("/chats/{chat_id}/messages", response_model=SendMessageResponse, status_code=201)
async def send_message(chat_id: int, payload: SendMessage, request: Request, db: Session = Depends(get_db)):
    chat = find_chat(db, chat_id)
    history = db.scalars(select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.desc(), Message.id.desc()).limit(10)).all()
    context = [{"role": item.role, "content": item.content} for item in reversed(history)]
    try:
        answer = await request.app.state.inference.generate(context + [{"role": "user", "content": payload.content}])
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "回答の生成に失敗しました") from exc
    if not history and chat.title == "新しいチャット":
        chat.title = payload.content[:40]
    user = Message(chat=chat, role="user", content=payload.content)
    assistant = Message(chat=chat, role="assistant", content=answer)
    chat.updated_at = utcnow(); db.add_all((user, assistant)); db.commit(); db.refresh(user); db.refresh(assistant)
    return SendMessageResponse(user=user, assistant=assistant)


@router.post("/temporary/messages", response_model=TemporaryMessageRead)
async def temporary_message(payload: TemporaryMessage, request: Request):
    context = [item.model_dump() for item in payload.history] + [{"role": "user", "content": payload.content}]
    try:
        answer = await request.app.state.inference.generate(context)
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "回答の生成に失敗しました") from exc
    return TemporaryMessageRead(content=answer)
