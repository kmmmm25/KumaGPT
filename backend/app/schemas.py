from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageInput(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("本文を入力してください")
        return value.strip()


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("本文を入力してください")
        return value.strip()


class TemporaryMessage(SendMessage):
    history: list[MessageInput] = Field(default_factory=list, max_length=10)


class ChatCreate(BaseModel):
    title: str | None = Field(default=None, max_length=80)


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=80)

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("タイトルを入力してください")
        return value.strip()


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    chat_id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class TemporaryMessageRead(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class Page(BaseModel):
    items: list
    total: int
    limit: int
    offset: int


class SendMessageResponse(BaseModel):
    user: MessageRead
    assistant: MessageRead
