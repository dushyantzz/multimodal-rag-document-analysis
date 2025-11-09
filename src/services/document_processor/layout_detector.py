"""Layout detection using DocLayout-YOLO v12.

Detects 11 document element types:
- Title, Plain Text, Abandon, Figure, Figure_caption,
  Table, Table_caption, Table_footnote, Isolate_formula,
  Formula_caption, List
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
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


class LayoutDetector:
    """Detect document layout using DocLayout-YOLO v12.
    
    Provides precise bounding boxes and element classification
    for document understanding.
    """

    # DocLayNet label mapping
    ELEMENT_LABELS = {
        0: "Title",
        1: "Plain_Text",
        2: "Abandon",
        3: "Figure",
        4: "Figure_caption",
        5: "Table",
        6: "Table_caption",
        7: "Table_footnote",
        8: "Isolate_formula",
        9: "Formula_caption",
        10: "List"
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("model_path", "doclayout-yolo-v12-base")
        self.confidence_threshold = config.get("confidence_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.device = config.get("device", "cuda:0")  # or "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model = None
        
    async def initialize(self):
        """Load YOLO model."""
        if YOLO is None:
            logger.warning("Ultralytics not installed, layout detection disabled")
            return
        
        try:
            logger.info(f"Loading DocLayout-YOLO model: {self.model_path}")
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model_sync
            )
            logger.info("DocLayout-YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            self.model = None
    
    def _load_model_sync(self) -> Optional[Any]:
        """Load model synchronously in thread pool."""
        try:
            # Try loading from Hugging Face or local path
            if self.model_path.startswith("hf://"):
                # Load from Hugging Face Hub
                model_id = self.model_path.replace("hf://", "")
                model = YOLO(f"hf://{model_id}")
            elif Path(self.model_path).exists():
                # Load from local path
                model = YOLO(self.model_path)
            else:
                # Try DocLayout-YOLO pretrained
                logger.info("Attempting to load pretrained DocLayout-YOLO")
                # You can download from: https://github.com/opendatalab/DocLayout-YOLO
                model = YOLO("yolov8x.pt")  # Fallback to base YOLOv8
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
    
    async def detect(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Detect layout elements in PDF pages.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of detected elements with bounding boxes
        """
        if self.model is None:
            logger.warning("YOLO model not loaded, skipping layout detection")
            return []
        
        logger.info(f"Detecting layout in: {pdf_path}")
        
        try:
            # Convert PDF pages to images
            page_images = await self._pdf_to_images(pdf_path)
            
            # Run detection on each page
            all_detections = []
            for page_num, img in enumerate(page_images, start=1):
                detections = await self._detect_page(img, page_num)
                all_detections.extend(detections)
            
            logger.info(f"Detected {len(all_detections)} layout elements")
            return all_detections
            
        except Exception as e:
            logger.error(f"Error in layout detection: {str(e)}")
            return []
    
    async def _pdf_to_images(self, pdf_path: Path) -> List[np.ndarray]:
        """Convert PDF pages to images for YOLO processing."""
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Render page at high DPI for better detection
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(np.array(img))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
        
        return images
    
    async def _detect_page(self, image: np.ndarray, page_num: int) -> List[Dict[str, Any]]:
        """Run YOLO detection on a single page."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor,
            self._run_detection_sync,
            image
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # Process each detection
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "page_number": page_num,
                        "element_type": self.ELEMENT_LABELS.get(class_id, "Unknown"),
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        },
                        "confidence": confidence,
                        "class_id": class_id
                    })
        
        return detections
    
    def _run_detection_sync(self, image: np.ndarray):
        """Run YOLO detection synchronously."""
        return self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
    
    async def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """Visualize detected layout elements."""
        import cv2
        
        img_copy = image.copy()
        
        # Color map for different element types
        colors = {
            "Title": (255, 0, 0),
            "Plain_Text": (0, 255, 0),
            "Figure": (0, 0, 255),
            "Table": (255, 255, 0),
            "List": (255, 0, 255),
        }
        
        for det in detections:
            bbox = det["bbox"]
            element_type = det["element_type"]
            confidence = det["confidence"]
            
            # Draw bounding box
            color = colors.get(element_type, (128, 128, 128))
            cv2.rectangle(
                img_copy,
                (int(bbox["x1"]), int(bbox["y1"])),
                (int(bbox["x2"]), int(bbox["y2"])),
                color,
                2
            )
            
            # Draw label
            label = f"{element_type}: {confidence:.2f}"
            cv2.putText(
                img_copy,
                label,
                (int(bbox["x1"]), int(bbox["y1"]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        if output_path:
            cv2.imwrite(str(output_path), cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        
        return img_copy
    
    def __del__(self):
        """Cleanup executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
