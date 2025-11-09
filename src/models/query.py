"""Query request and response models."""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Query type enumeration."""
    SEMANTIC = "semantic"  # Pure semantic search
    SQL = "sql"  # SQL-based numerical queries
    HYBRID = "hybrid"  # Combination of both


class RetrievalMode(str, Enum):
    """Retrieval mode enumeration."""
    VISUAL = "visual"  # Visual embeddings only
    TEXT = "text"  # Text embeddings only
    MULTIMODAL = "multimodal"  # Both visual and text


class Citation(BaseModel):
    """Citation with source information."""
    document_id: str = Field(..., description="Source document ID")
    element_id: str = Field(..., description="Source element ID")
    page_number: int = Field(..., description="Page number")
    content: str = Field(..., description="Cited content")
    score: float = Field(..., description="Relevance score")
    bbox: Optional[Dict[str, float]] = Field(None, description="Bounding box coordinates")
    image_url: Optional[str] = Field(None, description="Image URL if applicable")


class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="User query")
    document_ids: Optional[List[str]] = Field(
        None,
        description="Filter by specific document IDs"
    )
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.MULTIMODAL,
        description="Retrieval mode"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    include_images: bool = Field(default=True, description="Include images in response")
    include_tables: bool = Field(default=True, description="Include tables in response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")


class QueryResponse(BaseModel):
    """Query response model."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    query_type: QueryType = Field(..., description="Detected query type")
    citations: List[Citation] = Field(..., description="Source citations")
    images: List[str] = Field(default_factory=list, description="Relevant image URLs")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Relevant tables")
    confidence: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")


class AgentStep(BaseModel):
    """Agent reasoning step."""
    step_number: int = Field(..., description="Step number")
    action: str = Field(..., description="Action taken")
    observation: str = Field(..., description="Observation")
    thought: str = Field(..., description="Agent thought process")


class AgentResponse(BaseModel):
    """Agentic query response with reasoning trace."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Final answer")
    steps: List[AgentStep] = Field(..., description="Reasoning steps")
    citations: List[Citation] = Field(..., description="Citations")
    processing_time: float = Field(..., description="Total processing time")
