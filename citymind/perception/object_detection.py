"""
Layer 2B: Structural Object Detection
Detects structural elements (beams, columns, walls, rebar, scaffolding).

AMD Tech: Ryzen AI NPU inference via ONNX Runtime + Vitis AI EP
Model: RT-DETR / YOLOv8 fine-tuned on construction data
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# Structural element classes for construction
STRUCTURAL_CLASSES = {
    0: "beam",
    1: "column", 
    2: "wall",
    3: "slab",
    4: "foundation",
    5: "rebar",
    6: "scaffolding",
    7: "railing",
    8: "joint",
    9: "pipe",
    10: "window",
    11: "door",
    12: "staircase",
}

# COCO classes relevant to construction (for pre-trained model fallback)
COCO_CONSTRUCTION_RELEVANT = {
    0: "person",       # Worker detection (PPE check)
    56: "chair",       # Furniture detection
    57: "couch",
    58: "potted plant",
    60: "dining table",
    62: "tv",
    72: "refrigerator",
}


class ObjectDetector:
    """
    Structural element detection for infrastructure inspection.
    
    Uses YOLOv8/RT-DETR with construction-specific fine-tuning.
    Falls back to COCO pre-trained model with construction-relevant filtering.
    
    AMD Technology:
    - Training: Fine-tuned on AMD Instinct MI300X with ROCm 7.2 + PyTorch TunableOp
    - Optimization: Exported to ONNX, quantized INT8 via Vitis AI Quantizer
    - Inference: ONNX Runtime + Vitis AI EP on Ryzen AI NPU (XDNA)
    """
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence: float = 0.35,
        use_npu: bool = False,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.use_npu = use_npu
        self.model = None
        self._loaded = False
    
    def load_model(self):
        """Load detection model."""
        if self._loaded:
            return
        
        if self.use_npu:
            self._load_onnx_npu()
        else:
            self._load_ultralytics()
        
        self._loaded = True
    
    def _load_ultralytics(self):
        """Load YOLOv8 via ultralytics library."""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading detection model: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Detection model loaded successfully")
            
        except ImportError:
            logger.warning("ultralytics not available, using OpenCV DNN fallback")
            self.model = "opencv_fallback"
    
    def _load_onnx_npu(self):
        """Load ONNX model for NPU inference."""
        try:
            import onnxruntime as ort
            
            onnx_path = Path("models/detector_int8.onnx")
            vaip_config = "models/vaip_config.json"
            
            self.model = ort.InferenceSession(
                str(onnx_path),
                providers=["VitisAIExecutionProvider"],
                provider_options=[{"config_file": vaip_config}],
            )
            logger.info("Detection model loaded on NPU")
            
        except Exception as e:
            logger.warning(f"NPU loading failed ({e}), falling back to ultralytics")
            self.use_npu = False
            self._load_ultralytics()
    
    def detect(self, image_path: str) -> Dict:
        """
        Detect structural elements in a single image.
        
        Returns:
            dict with detections list, each containing:
            - bbox: [x1, y1, x2, y2]
            - class_name: structural element type
            - confidence: detection confidence
            - area: bounding box area (pixels)
        """
        self.load_model()
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        h, w = img.shape[:2]
        
        if self.model == "opencv_fallback":
            return self._detect_opencv(img, image_path)
        elif self.use_npu:
            return self._detect_npu(img, image_path)
        else:
            return self._detect_ultralytics(img, image_path)
    
    def _detect_ultralytics(self, img: np.ndarray, image_path: str) -> Dict:
        """Run detection via ultralytics YOLO."""
        results = self.model(img, conf=self.confidence, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = r.names.get(cls_id, f"class_{cls_id}")
                
                # Map COCO class to structural element (best-effort)
                structural_type = self._map_to_structural(cls_name)
                
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
                detections.append({
                    "bbox": [round(x, 1) for x in bbox],
                    "class_name": cls_name,
                    "structural_type": structural_type,
                    "confidence": round(conf, 3),
                    "area": round(area, 1),
                    "class_id": cls_id,
                })
        
        return {
            "detections": detections,
            "count": len(detections),
            "source_image": image_path,
            "model": self.model_path,
            "image_size": list(img.shape[:2]),
        }
    
    def _detect_opencv(self, img: np.ndarray, image_path: str) -> Dict:
        """
        Fallback detection using OpenCV edge analysis + heuristics.
        Identifies rectangular structural patterns typical of construction.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Skip tiny contours
                continue
            
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / max(ch, 1)
            
            # Classify based on geometry
            if aspect_ratio > 3.0:
                structural_type = "beam"
            elif aspect_ratio < 0.3:
                structural_type = "column"
            elif 0.8 < aspect_ratio < 1.2 and area > 5000:
                structural_type = "wall"
            else:
                structural_type = "structural_element"
            
            detections.append({
                "bbox": [float(x), float(y), float(x + cw), float(y + ch)],
                "class_name": structural_type,
                "structural_type": structural_type,
                "confidence": round(min(area / 10000, 0.9), 3),
                "area": float(area),
                "class_id": -1,
            })
        
        # Keep top detections by area
        detections.sort(key=lambda d: d["area"], reverse=True)
        detections = detections[:20]
        
        return {
            "detections": detections,
            "count": len(detections),
            "source_image": image_path,
            "model": "opencv_heuristic",
            "image_size": [h, w],
        }
    
    def _detect_npu(self, img: np.ndarray, image_path: str) -> Dict:
        """NPU inference path via ONNX Runtime + Vitis AI EP."""
        # Preprocess for YOLO
        img_resized = cv2.resize(img, (640, 640))
        img_float = img_resized.astype(np.float32) / 255.0
        img_float = np.transpose(img_float, (2, 0, 1))
        img_float = np.expand_dims(img_float, 0)
        
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: img_float})
        
        # Parse YOLO output (simplified)
        detections = self._parse_yolo_output(outputs[0], img.shape[:2])
        
        return {
            "detections": detections,
            "count": len(detections),
            "source_image": image_path,
            "model": "detector_int8.onnx (NPU)",
            "image_size": list(img.shape[:2]),
        }
    
    def _parse_yolo_output(self, output: np.ndarray, original_size: Tuple) -> List[Dict]:
        """Parse raw YOLO ONNX output into detection list."""
        detections = []
        # Simplified parsing — in production, use proper NMS
        if len(output.shape) == 3:
            output = output[0]
        
        for det in output:
            if len(det) < 6:
                continue
            conf = det[4]
            if conf < self.confidence:
                continue
            
            cls_scores = det[5:]
            cls_id = int(np.argmax(cls_scores))
            
            x_center, y_center, w, h = det[0:4]
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            # Scale to original image size
            scale_y = original_size[0] / 640
            scale_x = original_size[1] / 640
            
            detections.append({
                "bbox": [
                    round(x1 * scale_x, 1), round(y1 * scale_y, 1),
                    round(x2 * scale_x, 1), round(y2 * scale_y, 1),
                ],
                "class_name": f"class_{cls_id}",
                "structural_type": "structural_element",
                "confidence": round(float(conf), 3),
                "area": round(float(w * h * scale_x * scale_y), 1),
                "class_id": cls_id,
            })
        
        return detections
    
    def _map_to_structural(self, coco_class: str) -> str:
        """Map COCO class name to construction structural type."""
        mapping = {
            "person": "worker",
            "truck": "construction_vehicle",
            "car": "vehicle",
            "bench": "structural_support",
            "chair": "furniture",
            "tv": "equipment",
            "laptop": "equipment",
            "cell phone": "equipment",
            "bottle": "debris",
            "cup": "debris",
        }
        return mapping.get(coco_class, "structural_element")
    
    def batch_detect(
        self, 
        image_paths: List[str], 
        output_dir: str
    ) -> List[Dict]:
        """Run detection on multiple images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, img_path in enumerate(image_paths):
            try:
                logger.info(f"  Object detection: frame {i+1}/{len(image_paths)}")
                result = self.detect(img_path)
                results.append(result)
                
                # Save annotated image
                self._save_annotated(
                    img_path, result["detections"],
                    str(output_dir / f"detection_{i:04d}.png")
                )
                
            except Exception as e:
                logger.error(f"  Detection failed for {img_path}: {e}")
                results.append({"error": str(e), "source_image": img_path})
        
        # Save summary
        with open(output_dir / "detection_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _save_annotated(
        self, 
        image_path: str, 
        detections: List[Dict], 
        output_path: str
    ):
        """Save image with detection bounding boxes overlaid."""
        img = cv2.imread(image_path)
        if img is None:
            return
        
        for det in detections:
            bbox = [int(x) for x in det["bbox"]]
            label = f"{det['structural_type']} {det['confidence']:.2f}"
            
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(
                img, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        
        cv2.imwrite(output_path, img)
