"""PDF parsing using Unstructured.io for advanced element detection."""

import logging
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.staging.base import elements_to_json
except ImportError:
    partition_pdf = None
    elements_to_json = None

import fitz  # PyMuPDF as fallback

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse PDF documents with element detection.
    
    Uses Unstructured.io for advanced parsing with fallback to PyMuPDF.
    Extracts: text, tables, images, titles, headers, lists.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = config.get("strategy", "hi_res")  # hi_res, fast, ocr_only
        self.extract_images = config.get("extract_images", True)
        self.extract_tables = config.get("extract_tables", True)
        self.chunk_elements = config.get("chunk_elements", True)
        self.max_characters = config.get("max_characters", 1500)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse PDF and extract structured elements.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of document elements with metadata
        """
        logger.info(f"Parsing PDF: {file_path}")
        
        if partition_pdf is None:
            logger.warning("Unstructured.io not installed, using PyMuPDF fallback")
            return await self._parse_with_pymupdf(file_path)
        
        try:
            # Run blocking partition_pdf in thread pool
            loop = asyncio.get_event_loop()
            elements = await loop.run_in_executor(
                self.executor,
                self._partition_pdf_sync,
                file_path
            )
            
            # Convert to structured format
            structured_elements = await self._structure_elements(elements)
            
            logger.info(f"Extracted {len(structured_elements)} elements")
            return structured_elements
            
        except Exception as e:
            logger.error(f"Error parsing with Unstructured.io: {str(e)}")
            logger.info("Falling back to PyMuPDF")
            return await self._parse_with_pymupdf(file_path)
    
    def _partition_pdf_sync(self, file_path: Path) -> List:
        """Synchronous PDF partitioning (runs in thread pool)."""
        return partition_pdf(
            filename=str(file_path),
            strategy=self.strategy,
            extract_images_in_pdf=self.extract_images,
            extract_image_block_types=["Image", "Table"],
            infer_table_structure=self.extract_tables,
            chunking_strategy="by_title" if self.chunk_elements else None,
            max_characters=self.max_characters,
            include_page_breaks=True,
            languages=["eng"],  # Add more languages as needed
        )
    
    async def _structure_elements(self, elements: List) -> List[Dict[str, Any]]:
        """Convert Unstructured.io elements to standardized format."""
        structured = []
        
        for element in elements:
            # Get element type
            element_type = element.category if hasattr(element, 'category') else 'Unknown'
            
            # Extract coordinates
            coordinates = {}
            if hasattr(element, 'metadata') and hasattr(element.metadata, 'coordinates'):
                coords = element.metadata.coordinates
                if coords:
                    coordinates = {
                        "points": coords.points if hasattr(coords, 'points') else [],
                        "system": coords.system if hasattr(coords, 'system') else "PixelSpace"
                    }
            
            # Build structured element
            structured_elem = {
                "type": element_type,
                "text": str(element),
                "page_number": getattr(element.metadata, 'page_number', 1) if hasattr(element, 'metadata') else 1,
                "coordinates": coordinates,
                "metadata": {}
            }
            
            # Add type-specific metadata
            if hasattr(element, 'metadata'):
                metadata = element.metadata
                
                # Table metadata
                if element_type == "Table" and hasattr(metadata, 'text_as_html'):
                    structured_elem["metadata"]["text_as_html"] = metadata.text_as_html
                
                # Image metadata
                if element_type in ["Image", "Figure"] and hasattr(metadata, 'image_path'):
                    structured_elem["metadata"]["image_path"] = metadata.image_path
                
                # General metadata
                if hasattr(metadata, 'filename'):
                    structured_elem["metadata"]["filename"] = metadata.filename
                if hasattr(metadata, 'file_directory'):
                    structured_elem["metadata"]["file_directory"] = metadata.file_directory
            
            structured.append(structured_elem)
        
        return structured
    
    async def _parse_with_pymupdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Fallback parser using PyMuPDF."""
        elements = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        for line in block.get("lines", []):
                            text = " ".join([span["text"] for span in line.get("spans", [])])
                            if text.strip():
                                elements.append({
                                    "type": "text",
                                    "text": text,
                                    "page_number": page_num + 1,
                                    "coordinates": {
                                        "points": block["bbox"],
                                        "system": "PixelSpace"
                                    },
                                    "metadata": {}
                                })
                    
                    elif block["type"] == 1:  # Image block
                        elements.append({
                            "type": "Image",
                            "text": "",
                            "page_number": page_num + 1,
                            "coordinates": {
                                "points": block["bbox"],
                                "system": "PixelSpace"
                            },
                            "metadata": {"image_index": block.get("number", 0)}
                        })
            
            doc.close()
            logger.info(f"PyMuPDF extracted {len(elements)} elements")
            
        except Exception as e:
            logger.error(f"Error parsing with PyMuPDF: {str(e)}")
            raise
        
        return elements
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
