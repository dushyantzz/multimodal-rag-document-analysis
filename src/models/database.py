"""SQLAlchemy database models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

from src.models.document import ProcessingStatus, DocumentType, ElementType

Base = declarative_base()


class Document(Base):
    """Document table."""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)
    num_pages = Column(Integer, nullable=False)
    doc_type = Column(SQLEnum(DocumentType), nullable=False)
    status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.PENDING)
    upload_date = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float, nullable=True)
    metadata = Column(JSON, default={})
    error_message = Column(Text, nullable=True)


class DocumentElement(Base):
    """Document element table."""
    __tablename__ = "document_elements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    element_type = Column(SQLEnum(ElementType), nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


class TableData(Base):
    """Table data for SQL-RAG."""
    __tablename__ = "table_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    element_id = Column(UUID(as_uuid=True), nullable=False)
    table_name = Column(String(255), nullable=False)
    column_names = Column(JSON, nullable=False)
    data = Column(JSON, nullable=False)
    page_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryLog(Base):
    """Query log table."""
    __tablename__ = "query_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False)
    answer = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    processing_time = Column(Float, nullable=False)
    num_citations = Column(Integer, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
