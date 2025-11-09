"""Document data models."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Document type enumeration."""
    PDF = "pdf"
    IMAGE = "image"
    SCANNED = "scanned"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ElementType(str, Enum):
    """Document element type."""
    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    page: int = Field(..., description="Page number")


class DocumentElement(BaseModel):
    """Document element extracted from PDF."""
    element_id: str = Field(..., description="Unique element ID")
    element_type: ElementType = Field(..., description="Element type")
    content: str = Field(..., description="Element content")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box")
    page_number: int = Field(..., description="Page number")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentMetadata(BaseModel):
    """Document metadata."""
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    num_pages: int = Field(..., description="Number of pages")
    doc_type: DocumentType = Field(..., description="Document type")
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class DocumentUploadRequest(BaseModel):
    """Document upload request."""
    extract_images: bool = Field(default=True, description="Extract images from document")
    extract_tables: bool = Field(default=True, description="Extract tables from document")
    perform_ocr: bool = Field(default=True, description="Perform OCR on images")
    use_layout_detection: bool = Field(default=True, description="Use layout detection")


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    document_id: str = Field(..., description="Unique document ID")
    status: ProcessingStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")


class DocumentProcessingResult(BaseModel):
    """Document processing result."""
    document_id: str = Field(..., description="Document ID")
    status: ProcessingStatus = Field(..., description="Processing status")
    elements: List[DocumentElement] = Field(..., description="Extracted elements")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    error: Optional[str] = Field(None, description="Error message if failed")
