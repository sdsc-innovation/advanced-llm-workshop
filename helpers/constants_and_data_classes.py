from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field

API_BASE_URL = (
    "https://vlm2vec.runai-innovation-clement.inference.compute.datascience.ch"
)
API_TOKEN = "sdsc_vlm2vec"

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
    DEVELOPER = "developer"  # Previously, system
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMMessage(BaseModel):
    content: Optional[str] = None
    role: Optional[Roles] = None
