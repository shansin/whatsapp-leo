"""Data models for WhatsApp Leo agent."""

from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class ReceivedMessage:
    """Data structure for incoming WhatsApp messages."""

    chat_jid: str
    chat_name: str
    content: str
    file_length: int
    filename: str
    id: str
    is_from_me: bool
    media_type: str
    phone_number: str
    sender: str
    sender_jid: str
    timestamp: str
    url: str

    @classmethod
    def from_dict(cls, data: dict) -> "ReceivedMessage":
        return cls(
            chat_jid=data.get("chat_jid", ""),
            chat_name=data.get("chat_name", ""),
            content=data.get("content", ""),
            file_length=data.get("file_length", 0),
            filename=data.get("filename", ""),
            id=data.get("id", ""),
            is_from_me=data.get("is_from_me", False),
            media_type=data.get("media_type", ""),
            phone_number=data.get("phone_number", ""),
            sender=data.get("sender", ""),
            sender_jid=data.get("sender_jid", ""),
            timestamp=data.get("timestamp", ""),
            url=data.get("url", ""),
        )


class ReminderParsed(BaseModel):
    """Structured output model for reminder parsing."""

    reminder_message: str
    remind_at: str
