"""Embedding data models."""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class EmbeddingType(str, Enum):
    """Embedding type enumeration."""
    VISUAL = "visual"  # ColPALI visual embeddings
    TEXT = "text"  # Text embeddings
    MULTIMODAL = "multimodal"  # Combined embeddings


class VisualEmbedding(BaseModel):
    """Visual embedding from ColPALI."""
    element_id: str = Field(..., description="Element ID")
    embeddings: List[List[float]] = Field(
        ...,
        description="Multi-vector embeddings (1030 x 128)"
    )
    image_path: str = Field(..., description="Path to image")
    page_number: int = Field(..., description="Page number")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextEmbedding(BaseModel):
    """Text embedding."""
    element_id: str = Field(..., description="Element ID")
    embedding: List[float] = Field(..., description="Dense embedding vector")
    text: str = Field(..., description="Original text")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    """Embedding generation request."""
    content: str = Field(..., description="Content to embed")
    embedding_type: EmbeddingType = Field(..., description="Type of embedding")
    model: Optional[str] = Field(None, description="Model to use")


class EmbeddingResponse(BaseModel):
    """Embedding generation response."""
    embedding: List[float] = Field(..., description="Generated embedding")
    model: str = Field(..., description="Model used")
    dimension: int = Field(..., description="Embedding dimension")
