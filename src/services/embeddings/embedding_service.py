"""Main embedding service coordinating visual and text embeddings."""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np

from .colpali_embedder import ColPALIEmbedder
from .text_embedder import TextEmbedder
from ...models.document import Document

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Coordinate multimodal embedding generation.
    
    Generates:
    1. Visual embeddings using ColPALI (page images)
    2. Text embeddings using OpenAI/Cohere (text chunks)
    3. Table embeddings (hybrid visual + text)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.colpali_embedder = ColPALIEmbedder(config.get("colpali", {}))
        self.text_embedder = TextEmbedder(config.get("text_embeddings", {}))
        self.use_visual = config.get("use_visual_embeddings", True)
        self.use_text = config.get("use_text_embeddings", True)
        self.batch_size = config.get("batch_size", 4)
        
    async def initialize(self):
        """Initialize embedding models."""
        tasks = []
        
        if self.use_visual:
            tasks.append(self.colpali_embedder.initialize())
        
        if self.use_text:
            tasks.append(self.text_embedder.initialize())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        logger.info("Embedding service initialized")
    
    async def embed_document(self, document: Document) -> Dict[str, Any]:
        """Generate all embeddings for a document.
        
        Args:
            document: Processed document
            
        Returns:
            Dict containing visual and text embeddings
        """
        logger.info(f"Generating embeddings for document: {document.metadata.filename}")
        
        results = {
            "document_id": document.id,
            "visual_embeddings": [],
            "text_embeddings": [],
            "table_embeddings": []
        }
        
        tasks = []
        
        # Visual embeddings (ColPALI)
        if self.use_visual and document.page_images:
            tasks.append(self._generate_visual_embeddings(document))
        
        # Text embeddings
        if self.use_text:
            tasks.append(self._generate_text_embeddings(document))
        
        # Execute in parallel
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in embedding task {i}: {str(result)}")
                elif isinstance(result, dict):
                    results.update(result)
        
        return results
    
    async def _generate_visual_embeddings(self, document: Document) -> Dict[str, List]:
        """Generate ColPALI embeddings for page images."""
        logger.info("Generating visual embeddings with ColPALI")
        
        try:
            # Get page images
            page_images = []
            for page_num, img_data in sorted(document.page_images.items()):
                page_images.append(img_data)
            
            # Generate embeddings
            embeddings = await self.colpali_embedder.embed_images(page_images)
            
            # Structure results
            visual_embeddings = []
            for page_num, embedding in enumerate(embeddings, start=1):
                visual_embeddings.append({
                    "page_number": page_num,
                    "embedding": embedding,
                    "embedding_type": "colpali",
                    "dimensions": len(embedding) if isinstance(embedding, list) else embedding.shape[0]
                })
            
            logger.info(f"Generated {len(visual_embeddings)} visual embeddings")
            return {"visual_embeddings": visual_embeddings}
            
        except Exception as e:
            logger.error(f"Error generating visual embeddings: {str(e)}")
            return {"visual_embeddings": []}
    
    async def _generate_text_embeddings(self, document: Document) -> Dict[str, List]:
        """Generate text embeddings for document chunks."""
        logger.info("Generating text embeddings")
        
        try:
            # Extract text chunks
            chunks = self._create_text_chunks(document)
            
            if not chunks:
                return {"text_embeddings": []}
            
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                texts = [chunk["text"] for chunk in batch]
                
                embeddings = await self.text_embedder.embed_texts(texts)
                
                for j, embedding in enumerate(embeddings):
                    chunk_idx = i + j
                    all_embeddings.append({
                        "chunk_id": chunk_idx,
                        "text": chunks[chunk_idx]["text"],
                        "page_number": chunks[chunk_idx]["page_number"],
                        "element_type": chunks[chunk_idx]["element_type"],
                        "embedding": embedding,
                        "embedding_type": "text",
                        "dimensions": len(embedding)
                    })
            
            logger.info(f"Generated {len(all_embeddings)} text embeddings")
            return {"text_embeddings": all_embeddings}
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {str(e)}")
            return {"text_embeddings": []}
    
    def _create_text_chunks(self, document: Document, chunk_size: int = 500) -> List[Dict]:
        """Create text chunks from document elements."""
        chunks = []
        current_chunk = ""
        current_page = 1
        current_type = "text"
        
        for element in document.elements:
            text = element.content.strip()
            
            if not text:
                continue
            
            # For tables and images, keep as separate chunks
            if element.element_type in ["Table", "Image", "Figure"]:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "page_number": current_page,
                        "element_type": current_type
                    })
                    current_chunk = ""
                
                chunks.append({
                    "text": text,
                    "page_number": element.page_number,
                    "element_type": element.element_type
                })
                continue
            
            # Add to current chunk
            if len(current_chunk) + len(text) < chunk_size:
                current_chunk += " " + text
                current_page = element.page_number
                current_type = element.element_type
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "page_number": current_page,
                        "element_type": current_type
                    })
                current_chunk = text
                current_page = element.page_number
                current_type = element.element_type
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "page_number": current_page,
                "element_type": current_type
            })
        
        return chunks
    
    async def embed_query(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        """Embed a query for retrieval.
        
        Args:
            query: Query text
            mode: "text", "visual", or "hybrid"
            
        Returns:
            Query embeddings
        """
        results = {}
        
        if mode in ["text", "hybrid"]:
            text_emb = await self.text_embedder.embed_texts([query])
            results["text_embedding"] = text_emb[0] if text_emb else None
        
        if mode in ["visual", "hybrid"]:
            # For visual queries, could use CLIP or ColPALI query encoder
            # For now, use text embedding as approximation
            results["visual_embedding"] = results.get("text_embedding")
        
        return results
