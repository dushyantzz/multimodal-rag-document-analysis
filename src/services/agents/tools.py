"""Agent tools for RAG operations."""

from typing import Optional
import numpy as np

from src.core.logger import get_logger
from src.services.vector_store import QdrantVectorStore
from src.services.sql_rag import SQLStore, TextToSQLGenerator
from src.services.embeddings import EmbeddingService

logger = get_logger(__name__)


class AgentTools:
    """Tools available to the query agent.
    
    Provides unified interface to:
    - Vector store (Qdrant)
    - SQL store (PostgreSQL/DuckDB)
    - Text-to-SQL generator
    - Embedding generation
    - OCR and document processing
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        sql_store: SQLStore,
        text_to_sql: TextToSQLGenerator,
        embedding_service: EmbeddingService,
    ):
        """Initialize agent tools.
        
        Args:
            vector_store: Qdrant vector store instance
            sql_store: SQL store instance
            text_to_sql: Text-to-SQL generator instance
            embedding_service: Embedding service instance
        """
        self.vector_store = vector_store
        self.sql_store = sql_store
        self.text_to_sql = text_to_sql
        self.embedding_service = embedding_service
        
        logger.info("Initialized AgentTools")

    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        return self.embedding_service.embed_text([text])[0]

    def embed_image(self, image_path: str) -> np.ndarray:
        """Generate visual embedding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector
        """
        from PIL import Image
        image = Image.open(image_path)
        return self.embedding_service.embed_images([image])[0]

    def execute_sql(self, sql: str) -> list:
        """Execute SQL query.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query results
        """
        return self.sql_store.execute_query(sql, return_dataframe=False)

    def generate_sql(self, query: str, document_ids: Optional[list] = None) -> dict:
        """Generate SQL from natural language.
        
        Args:
            query: Natural language query
            document_ids: Optional document filters
            
        Returns:
            Generated SQL and metadata
        """
        return self.text_to_sql.generate_sql(query, document_ids)
