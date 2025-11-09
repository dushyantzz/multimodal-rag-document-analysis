"""Qdrant vector store for multimodal embeddings with ColBERT-style multi-vector support."""

import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


class DocumentChunk(BaseModel):
    """Document chunk with metadata."""
    
    chunk_id: str
    document_id: str
    content: str
    chunk_type: str  # text, image, table, figure
    page_number: int
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}


class SearchResult(BaseModel):
    """Search result with score and metadata."""
    
    chunk_id: str
    document_id: str
    content: str
    score: float
    chunk_type: str
    page_number: int
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}


class QdrantVectorStore:
    """Qdrant vector store with multimodal embedding support.
    
    Features:
    - Separate collections for visual (ColPALI) and text embeddings
    - Multi-vector storage for patch-based embeddings
    - Hybrid retrieval combining visual and text search
    - Metadata filtering and payload storage
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_prefix: str = "multimodal_rag",
    ):
        """Initialize Qdrant client and collections.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_prefix: Prefix for collection names
        """
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_prefix = collection_prefix
        
        # Initialize client
        self.client = QdrantClient(host=self.host, port=self.port)
        
        # Collection names
        self.visual_collection = f"{collection_prefix}_visual"
        self.text_collection = f"{collection_prefix}_text"
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized QdrantVectorStore at {self.host}:{self.port}")

    def create_collections(self, force_recreate: bool = False) -> None:
        """Create vector collections for visual and text embeddings.
        
        Args:
            force_recreate: If True, delete existing collections and recreate
        """
        try:
            # Visual collection (ColPALI patch embeddings: 1030 vectors of 128 dims)
            if force_recreate and self.client.collection_exists(self.visual_collection):
                self.client.delete_collection(self.visual_collection)
                logger.info(f"Deleted existing collection: {self.visual_collection}")
            
            if not self.client.collection_exists(self.visual_collection):
                self.client.create_collection(
                    collection_name=self.visual_collection,
                    vectors_config=VectorParams(
                        size=128,  # ColPALI embedding dimension
                        distance=Distance.COSINE,
                    ),
                )
                # Create payload index for filtering
                self.client.create_payload_index(
                    collection_name=self.visual_collection,
                    field_name="document_id",
                    field_schema="keyword",
                )
                self.client.create_payload_index(
                    collection_name=self.visual_collection,
                    field_name="chunk_type",
                    field_schema="keyword",
                )
                logger.info(f"Created visual collection: {self.visual_collection}")
            
            # Text collection (standard text embeddings: 1536 dims for OpenAI)
            if force_recreate and self.client.collection_exists(self.text_collection):
                self.client.delete_collection(self.text_collection)
                logger.info(f"Deleted existing collection: {self.text_collection}")
            
            if not self.client.collection_exists(self.text_collection):
                self.client.create_collection(
                    collection_name=self.text_collection,
                    vectors_config=VectorParams(
                        size=settings.TEXT_EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                # Create payload index
                self.client.create_payload_index(
                    collection_name=self.text_collection,
                    field_name="document_id",
                    field_schema="keyword",
                )
                self.client.create_payload_index(
                    collection_name=self.text_collection,
                    field_name="chunk_type",
                    field_schema="keyword",
                )
                logger.info(f"Created text collection: {self.text_collection}")
                
        except Exception as e:
            logger.error(f"Error creating collections: {e}")
            raise

    def add_visual_embeddings(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[np.ndarray],  # Each is (1030, 128) for ColPALI
    ) -> List[str]:
        """Add visual (ColPALI) embeddings to vector store.
        
        Args:
            chunks: List of document chunks
            embeddings: List of multi-vector embeddings (patch embeddings)
            
        Returns:
            List of inserted point IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        points = []
        point_ids = []
        
        for chunk, embedding in zip(chunks, embeddings):
            # For ColPALI: Store average of patch embeddings as single vector
            # MaxSim operation will be done at query time
            avg_embedding = np.mean(embedding, axis=0).tolist()
            
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number,
                "bbox": chunk.bbox,
                "metadata": chunk.metadata,
                # Store patch embeddings for MaxSim retrieval
                "patch_embeddings": embedding.tolist(),
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=avg_embedding,
                    payload=payload,
                )
            )
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.visual_collection,
            points=points,
        )
        
        logger.info(f"Added {len(points)} visual embeddings to {self.visual_collection}")
        return point_ids

    def add_text_embeddings(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[np.ndarray],  # Each is (1536,) for OpenAI
    ) -> List[str]:
        """Add text embeddings to vector store.
        
        Args:
            chunks: List of document chunks
            embeddings: List of text embeddings
            
        Returns:
            List of inserted point IDs
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        points = []
        point_ids = []
        
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "page_number": chunk.page_number,
                "bbox": chunk.bbox,
                "metadata": chunk.metadata,
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.text_collection,
            points=points,
        )
        
        logger.info(f"Added {len(points)} text embeddings to {self.text_collection}")
        return point_ids

    def search_visual(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Search using visual embeddings.
        
        Args:
            query_embedding: Query visual embedding
            limit: Maximum number of results
            document_ids: Filter by document IDs
            chunk_types: Filter by chunk types
            
        Returns:
            List of search results
        """
        # Build filter
        filter_conditions = []
        if document_ids:
            filter_conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_ids))
            )
        if chunk_types:
            filter_conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value=chunk_types))
            )
        
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search
        results = self.client.search(
            collection_name=self.visual_collection,
            query_vector=query_embedding.tolist(),
            limit=limit,
            query_filter=query_filter,
        )
        
        # Parse results
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    chunk_id=result.payload["chunk_id"],
                    document_id=result.payload["document_id"],
                    content=result.payload["content"],
                    score=result.score,
                    chunk_type=result.payload["chunk_type"],
                    page_number=result.payload["page_number"],
                    bbox=result.payload.get("bbox"),
                    metadata=result.payload.get("metadata", {}),
                )
            )
        
        logger.info(f"Visual search returned {len(search_results)} results")
        return search_results

    def search_text(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Search using text embeddings.
        
        Args:
            query_embedding: Query text embedding
            limit: Maximum number of results
            document_ids: Filter by document IDs
            chunk_types: Filter by chunk types
            
        Returns:
            List of search results
        """
        # Build filter
        filter_conditions = []
        if document_ids:
            filter_conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_ids))
            )
        if chunk_types:
            filter_conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value=chunk_types))
            )
        
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search
        results = self.client.search(
            collection_name=self.text_collection,
            query_vector=query_embedding.tolist(),
            limit=limit,
            query_filter=query_filter,
        )
        
        # Parse results
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    chunk_id=result.payload["chunk_id"],
                    document_id=result.payload["document_id"],
                    content=result.payload["content"],
                    score=result.score,
                    chunk_type=result.payload["chunk_type"],
                    page_number=result.payload["page_number"],
                    bbox=result.payload.get("bbox"),
                    metadata=result.payload.get("metadata", {}),
                )
            )
        
        logger.info(f"Text search returned {len(search_results)} results")
        return search_results

    def hybrid_search(
        self,
        visual_embedding: Optional[np.ndarray] = None,
        text_embedding: Optional[np.ndarray] = None,
        limit: int = 10,
        visual_weight: float = 0.5,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Perform hybrid search combining visual and text modalities.
        
        Args:
            visual_embedding: Visual query embedding
            text_embedding: Text query embedding
            limit: Maximum number of results
            visual_weight: Weight for visual scores (text weight = 1 - visual_weight)
            document_ids: Filter by document IDs
            chunk_types: Filter by chunk types
            
        Returns:
            List of deduplicated and reranked search results
        """
        results_map = {}  # chunk_id -> SearchResult with best score
        
        # Visual search
        if visual_embedding is not None:
            visual_results = self.search_visual(
                query_embedding=visual_embedding,
                limit=limit * 2,  # Get more candidates
                document_ids=document_ids,
                chunk_types=chunk_types,
            )
            for result in visual_results:
                weighted_score = result.score * visual_weight
                if result.chunk_id not in results_map or weighted_score > results_map[result.chunk_id].score:
                    result.score = weighted_score
                    results_map[result.chunk_id] = result
        
        # Text search
        if text_embedding is not None:
            text_results = self.search_text(
                query_embedding=text_embedding,
                limit=limit * 2,
                document_ids=document_ids,
                chunk_types=chunk_types,
            )
            text_weight = 1.0 - visual_weight
            for result in text_results:
                weighted_score = result.score * text_weight
                if result.chunk_id in results_map:
                    # Combine scores for chunks found in both
                    results_map[result.chunk_id].score += weighted_score
                else:
                    result.score = weighted_score
                    results_map[result.chunk_id] = result
        
        # Sort by combined score and return top k
        combined_results = sorted(
            results_map.values(),
            key=lambda x: x.score,
            reverse=True,
        )[:limit]
        
        logger.info(f"Hybrid search returned {len(combined_results)} results")
        return combined_results

    def delete_document(
        self,
        document_id: str,
    ) -> Tuple[int, int]:
        """Delete all chunks for a document from both collections.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Tuple of (visual_deleted_count, text_deleted_count)
        """
        filter_condition = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )
        
        # Delete from visual collection
        visual_result = self.client.delete(
            collection_name=self.visual_collection,
            points_selector=filter_condition,
        )
        
        # Delete from text collection
        text_result = self.client.delete(
            collection_name=self.text_collection,
            points_selector=filter_condition,
        )
        
        logger.info(
            f"Deleted document {document_id}: "
            f"{visual_result} visual points, {text_result} text points"
        )
        
        return (visual_result, text_result)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about collections.
        
        Returns:
            Dictionary with collection statistics
        """
        visual_info = self.client.get_collection(self.visual_collection)
        text_info = self.client.get_collection(self.text_collection)
        
        return {
            "visual": {
                "name": self.visual_collection,
                "points_count": visual_info.points_count,
                "vectors_count": visual_info.vectors_count,
                "status": visual_info.status,
            },
            "text": {
                "name": self.text_collection,
                "points_count": text_info.points_count,
                "vectors_count": text_info.vectors_count,
                "status": text_info.status,
            },
        }

    async def close(self):
        """Close connections and cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("Closed QdrantVectorStore")
