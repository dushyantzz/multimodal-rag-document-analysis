"""Main document processor orchestrating the entire pipeline."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from .pdf_parser import PDFParser
from .layout_detector import LayoutDetector
from .ocr_engine import OCREngine
from ...models.document import Document, DocumentElement, DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrates document processing pipeline.
    
    Pipeline stages:
    1. PDF parsing with Unstructured.io
    2. Layout detection with DocLayout-YOLO
    3. OCR for scanned content
    4. Element extraction and structuring
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pdf_parser = PDFParser(config.get("pdf_parser", {}))
        self.layout_detector = LayoutDetector(config.get("layout_detector", {}))
        self.ocr_engine = OCREngine(config.get("ocr", {}))
        
    async def process_document(self, file_path: Path) -> Document:
        """Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the PDF document
            
        Returns:
            Processed Document object with all elements
        """
        logger.info(f"Processing document: {file_path}")
        start_time = datetime.now()
        
        try:
            # Stage 1: Parse PDF with Unstructured.io
            logger.info("Stage 1: PDF parsing")
            parsed_elements = await self.pdf_parser.parse(file_path)
            
            # Stage 2: Layout detection
            logger.info("Stage 2: Layout detection")
            layout_results = await self.layout_detector.detect(file_path)
            
            # Stage 3: OCR for scanned pages or images
            logger.info("Stage 3: OCR processing")
            ocr_results = await self.ocr_engine.process(file_path, parsed_elements)
            
            # Stage 4: Merge and structure results
            logger.info("Stage 4: Merging results")
            document = await self._merge_results(
                file_path, parsed_elements, layout_results, ocr_results
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Document processed in {processing_time:.2f}s")
            
            document.metadata.processing_time = processing_time
            return document
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
    
    async def _merge_results(
        self,
        file_path: Path,
        parsed_elements: List[Dict],
        layout_results: List[Dict],
        ocr_results: List[Dict]
    ) -> Document:
        """Merge results from all processing stages."""
        
        elements = []
        page_images = {}
        tables = []
        
        # Process parsed elements
        for elem in parsed_elements:
            element = DocumentElement(
                element_type=elem.get("type", "text"),
                content=elem.get("text", ""),
                page_number=elem.get("page_number", 1),
                bbox=elem.get("coordinates", {}),
                metadata=elem.get("metadata", {})
            )
            elements.append(element)
            
            # Extract tables separately
            if elem.get("type") == "Table":
                tables.append({
                    "page": elem.get("page_number", 1),
                    "content": elem.get("text", ""),
                    "html": elem.get("metadata", {}).get("text_as_html", ""),
                    "bbox": elem.get("coordinates", {})
                })
        
        # Add layout detection results
        for layout in layout_results:
            # Enhance elements with layout info
            pass
        
        # Add OCR results for scanned content
        for ocr_result in ocr_results:
            if ocr_result.get("is_scanned"):
                element = DocumentElement(
                    element_type="text",
                    content=ocr_result.get("text", ""),
                    page_number=ocr_result.get("page", 1),
                    bbox=ocr_result.get("bbox", {}),
                    metadata={"source": "ocr", "confidence": ocr_result.get("confidence")}
                )
                elements.append(element)
        
        # Create document metadata
        metadata = DocumentMetadata(
            filename=file_path.name,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            page_count=len(set(e.page_number for e in elements)),
            processing_date=datetime.now(),
        )
        
        # Create final document
        document = Document(
            id=None,  # Will be set when stored
            metadata=metadata,
            elements=elements,
            tables=tables,
            page_images=page_images
        )
        
        return document
    
    async def extract_images(self, document: Document) -> List[Dict]:
        """Extract images from document for vision processing."""
        images = []
        
        for element in document.elements:
            if element.element_type in ["Image", "Figure"]:
                images.append({
                    "page": element.page_number,
                    "bbox": element.bbox,
                    "content": element.content,
                    "metadata": element.metadata
                })
        
        return images
    
    async def extract_tables_to_sql(self, document: Document) -> List[Dict]:
        """Extract tables for SQL database insertion."""
        structured_tables = []
        
        for table in document.tables:
            # Parse HTML table to structured format
            structured = await self._parse_table_html(table["html"])
            structured_tables.append({
                "page": table["page"],
                "columns": structured["columns"],
                "rows": structured["rows"],
                "bbox": table["bbox"]
            })
        
        return structured_tables
    
    async def _parse_table_html(self, html: str) -> Dict:
        """Parse HTML table into structured format."""
        # Use BeautifulSoup or pandas to parse HTML tables
        import pandas as pd
        from io import StringIO
        
        try:
            df = pd.read_html(StringIO(html))[0]
            return {
                "columns": df.columns.tolist(),
                "rows": df.values.tolist()
            }
        except:
            return {"columns": [], "rows": []}
