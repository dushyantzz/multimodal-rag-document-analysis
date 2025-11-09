"""OCR engine for extracting text from scanned documents.

Supports multiple OCR backends:
- PaddleOCR: 80+ languages, table recognition
- Tesseract: Fallback OCR engine
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

logger = logging.getLogger(__name__)


class OCREngine:
    """Multi-backend OCR engine for scanned documents."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = config.get("backend", "paddle")  # paddle, tesseract, auto
        self.languages = config.get("languages", ["en"])
        self.detect_tables = config.get("detect_tables", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.paddle_ocr = None
        
    async def initialize(self):
        """Initialize OCR engines."""
        if self.backend in ["paddle", "auto"] and PaddleOCR is not None:
            try:
                logger.info("Initializing PaddleOCR")
                loop = asyncio.get_event_loop()
                self.paddle_ocr = await loop.run_in_executor(
                    self.executor,
                    self._init_paddle_sync
                )
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing PaddleOCR: {str(e)}")
                self.paddle_ocr = None
        
        if self.backend == "tesseract" and pytesseract is None:
            logger.warning("Tesseract not installed")
    
    def _init_paddle_sync(self):
        """Initialize PaddleOCR synchronously."""
        return PaddleOCR(
            use_angle_cls=True,
            lang="en",  # Will support multi-language later
            use_gpu=True,
            show_log=False
        )
    
    async def process(self, pdf_path: Path, parsed_elements: List[Dict]) -> List[Dict[str, Any]]:
        """Process document for OCR if needed.
        
        Args:
            pdf_path: Path to PDF
            parsed_elements: Already parsed elements to check for scanned pages
            
        Returns:
            OCR results for scanned content
        """
        logger.info("Checking for scanned pages")
        
        # Detect scanned pages (pages with few/no text elements)
        scanned_pages = await self._detect_scanned_pages(pdf_path, parsed_elements)
        
        if not scanned_pages:
            logger.info("No scanned pages detected")
            return []
        
        logger.info(f"Found {len(scanned_pages)} scanned pages, running OCR")
        
        # Run OCR on scanned pages
        ocr_results = []
        for page_num in scanned_pages:
            page_results = await self._ocr_page(pdf_path, page_num)
            ocr_results.extend(page_results)
        
        return ocr_results
    
    async def _detect_scanned_pages(self, pdf_path: Path, parsed_elements: List[Dict]) -> List[int]:
        """Detect which pages are scanned (low text content)."""
        # Count text elements per page
        page_text_counts = {}
        for elem in parsed_elements:
            page_num = elem.get("page_number", 1)
            if elem.get("type") in ["text", "Title", "NarrativeText"]:
                page_text_counts[page_num] = page_text_counts.get(page_num, 0) + 1
        
        # Get total pages
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        # Pages with < 5 text elements are likely scanned
        scanned_pages = []
        for page_num in range(1, total_pages + 1):
            text_count = page_text_counts.get(page_num, 0)
            if text_count < 5:
                scanned_pages.append(page_num)
        
        return scanned_pages
    
    async def _ocr_page(self, pdf_path: Path, page_num: int) -> List[Dict[str, Any]]:
        """Run OCR on a specific page."""
        # Convert page to image
        image = await self._extract_page_image(pdf_path, page_num)
        
        # Choose OCR backend
        if self.paddle_ocr is not None:
            return await self._ocr_with_paddle(image, page_num)
        elif pytesseract is not None:
            return await self._ocr_with_tesseract(image, page_num)
        else:
            logger.warning("No OCR backend available")
            return []
    
    async def _extract_page_image(self, pdf_path: Path, page_num: int) -> np.ndarray:
        """Extract page as high-resolution image."""
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # 0-indexed
        
        # High DPI for better OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x scale = ~300 DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        doc.close()
        return np.array(img)
    
    async def _ocr_with_paddle(self, image: np.ndarray, page_num: int) -> List[Dict[str, Any]]:
        """Run PaddleOCR on image."""
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                self.paddle_ocr.ocr,
                image,
                cls=True
            )
            
            ocr_results = []
            
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        bbox_points = line[0]
                        text_info = line[1]
                        text = text_info[0] if isinstance(text_info, tuple) else text_info
                        confidence = text_info[1] if isinstance(text_info, tuple) and len(text_info) > 1 else 1.0
                        
                        if confidence >= self.confidence_threshold:
                            # Convert bbox points to x1, y1, x2, y2
                            bbox = self._points_to_bbox(bbox_points)
                            
                            ocr_results.append({
                                "page": page_num,
                                "text": text,
                                "bbox": bbox,
                                "confidence": confidence,
                                "is_scanned": True,
                                "backend": "paddle"
                            })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"PaddleOCR error: {str(e)}")
            return []
    
    async def _ocr_with_tesseract(self, image: np.ndarray, page_num: int) -> List[Dict[str, Any]]:
        """Run Tesseract OCR on image."""
        loop = asyncio.get_event_loop()
        
        try:
            # Get OCR data with bounding boxes
            data = await loop.run_in_executor(
                self.executor,
                pytesseract.image_to_data,
                Image.fromarray(image),
                output_type=pytesseract.Output.DICT
            )
            
            ocr_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > self.confidence_threshold * 100:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    ocr_results.append({
                        "page": page_num,
                        "text": text,
                        "bbox": {
                            "x1": x,
                            "y1": y,
                            "x2": x + w,
                            "y2": y + h
                        },
                        "confidence": conf / 100.0,
                        "is_scanned": True,
                        "backend": "tesseract"
                    })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Tesseract error: {str(e)}")
            return []
    
    def _points_to_bbox(self, points: List[List[float]]) -> Dict[str, float]:
        """Convert polygon points to bounding box."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        return {
            "x1": min(xs),
            "y1": min(ys),
            "x2": max(xs),
            "y2": max(ys)
        }
    
    async def recognize_table_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """Recognize table structure in image (PaddleOCR only)."""
        if self.paddle_ocr is None or not self.detect_tables:
            return {}
        
        # PaddleOCR table recognition
        # This requires PaddleOCR with table recognition model
        # Implementation depends on specific model availability
        logger.info("Table structure recognition not yet implemented")
        return {}
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
