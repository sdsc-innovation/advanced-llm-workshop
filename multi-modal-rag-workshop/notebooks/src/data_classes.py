from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field


class DataType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class Chunk(BaseModel):
    chunk_id: int
    content: str
    metadata: dict = Field(default_factory=dict)
    data_type: Optional[DataType] = DataType.TEXT
    score: Optional[float] = None


class Roles(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMMessage(BaseModel):
    content: Optional[str] = None
    role: Optional[Roles] = None
