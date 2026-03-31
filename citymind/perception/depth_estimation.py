"""
Layer 2A: Monocular Depth Estimation
Uses Depth Anything v2 (from AMD CVML Library concept) for dense depth maps.

AMD Tech: Ryzen AI NPU (XDNA), CVML Library, ONNX Runtime + Vitis AI EP
Reuse: Inspired by AMD's Feb 9, 2026 blog on CVML + ROS 2 edge perception
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
import cv2

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything v2.
    
    Production path: Model quantized via Vitis AI → INT8 ONNX → Ryzen AI NPU
    Demo path: HuggingFace transformers pipeline on CPU/GPU
    
    AMD Technology:
    - Training: AMD Instinct MI300X with ROCm 7.2
    - Optimization: Vitis AI Quantizer (INT8)
    - Inference: ONNX Runtime + Vitis AI Execution Provider on Ryzen AI NPU
    - Library: AMD CVML Library (pre-optimized Depth Anything v2)
    """
    
    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        use_npu: bool = False,
        device: str = "cpu",
    ):
        self.model_id = model_id
        self.use_npu = use_npu
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False
    
    def load_model(self):
        """Load depth estimation model."""
        if self._loaded:
            return
        
        if self.use_npu:
            self._load_npu_model()
        else:
            self._load_transformers_model()
        
        self._loaded = True
    
    def _load_transformers_model(self):
        """Load model via HuggingFace transformers."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            import torch
            
            logger.info(f"Loading depth model: {self.model_id}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            
            self.model.eval()
            logger.info("Depth model loaded successfully")
            
        except ImportError:
            logger.warning("transformers not available, using lightweight depth estimation")
            self.model = "lightweight"
    
    def _load_npu_model(self):
        """
        Load quantized model for AMD Ryzen AI NPU via ONNX Runtime + Vitis AI EP.
        
        This is the production deployment path:
        1. Model trained on MI300X with ROCm
        2. Exported to ONNX
        3. Quantized INT8 via Vitis AI Quantizer
        4. Deployed on Ryzen AI NPU via ONNX Runtime with Vitis AI EP
        """
        try:
            import onnxruntime as ort
            
            onnx_path = Path("models/depth_anything_v2_int8.onnx")
            vaip_config = "models/vaip_config.json"
            
            providers = ["VitisAIExecutionProvider"]
            provider_options = [{"config_file": vaip_config}]
            
            self.model = ort.InferenceSession(
                str(onnx_path),
                providers=providers,
                provider_options=provider_options,
            )
            logger.info("Depth model loaded on NPU via Vitis AI EP")
            
        except Exception as e:
            logger.warning(f"NPU loading failed ({e}), falling back to CPU")
            self.use_npu = False
            self._load_transformers_model()
    
    def estimate_depth(self, image_path: str) -> Dict:
        """
        Estimate depth for a single image.
        
        Returns:
            dict with depth_map (numpy array), visualization path, stats
        """
        self.load_model()
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.model == "lightweight":
            depth_map = self._lightweight_depth(img_rgb)
        elif self.use_npu:
            depth_map = self._infer_npu(img_rgb)
        else:
            depth_map = self._infer_transformers(img_rgb)
        
        # Normalize depth to 0-1
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        return {
            "depth_map": depth_normalized,
            "raw_depth": depth_map,
            "stats": {
                "min_depth": float(depth_map.min()),
                "max_depth": float(depth_map.max()),
                "mean_depth": float(depth_map.mean()),
                "std_depth": float(depth_map.std()),
            },
            "source_image": image_path,
        }
    
    def _infer_transformers(self, img_rgb: np.ndarray) -> np.ndarray:
        """Run inference via HuggingFace transformers."""
        import torch
        from PIL import Image
        
        pil_img = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        elif self.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth
        
        # Interpolate to original size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        return depth.cpu().numpy()
    
    def _infer_npu(self, img_rgb: np.ndarray) -> np.ndarray:
        """Run inference on AMD Ryzen AI NPU via ONNX Runtime."""
        # Preprocess
        img_resized = cv2.resize(img_rgb, (518, 518))
        img_float = img_resized.astype(np.float32) / 255.0
        img_float = np.transpose(img_float, (2, 0, 1))
        img_float = np.expand_dims(img_float, 0)
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        result = self.model.run(None, {input_name: img_float})
        
        depth = result[0].squeeze()
        
        # Resize to original
        depth = cv2.resize(depth, (img_rgb.shape[1], img_rgb.shape[0]))
        return depth
    
    def _lightweight_depth(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Lightweight depth estimation using edge-based heuristics.
        Fallback when deep learning models are not available.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale edge analysis for depth cues
        blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
        blur2 = cv2.GaussianBlur(gray, (15, 15), 0)
        blur3 = cv2.GaussianBlur(gray, (31, 31), 0)
        
        # Edges at different scales indicate different depths
        edges1 = cv2.Laplacian(blur1, cv2.CV_64F)
        edges2 = cv2.Laplacian(blur2, cv2.CV_64F)
        edges3 = cv2.Laplacian(blur3, cv2.CV_64F)
        
        # Combine: sharp edges = close, blurry = far
        depth = np.abs(edges1) * 0.5 + np.abs(edges2) * 0.3 + np.abs(edges3) * 0.2
        
        # Add vertical gradient (objects lower in frame tend to be closer)
        h, w = gray.shape
        vertical_gradient = np.linspace(0.3, 1.0, h).reshape(-1, 1)
        vertical_gradient = np.tile(vertical_gradient, (1, w))
        
        depth = depth * 0.7 + vertical_gradient * np.max(depth) * 0.3
        
        return depth.astype(np.float32)
    
    def visualize_depth(
        self, 
        depth_map: np.ndarray, 
        output_path: str,
        colormap: int = cv2.COLORMAP_INFERNO
    ) -> str:
        """Save depth map as colorized visualization."""
        # Normalize to 0-255
        depth_vis = (depth_map * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, colormap)
        cv2.imwrite(output_path, depth_colored)
        return output_path
    
    def batch_estimate(
        self, 
        image_paths: List[str], 
        output_dir: str
    ) -> List[Dict]:
        """Process multiple frames and save depth maps."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, img_path in enumerate(image_paths):
            try:
                logger.info(f"  Depth estimation: frame {i+1}/{len(image_paths)}")
                result = self.estimate_depth(img_path)
                
                # Save depth map
                depth_filename = f"depth_{i:04d}.npy"
                np.save(str(output_dir / depth_filename), result["depth_map"])
                
                # Save visualization
                vis_filename = f"depth_vis_{i:04d}.png"
                self.visualize_depth(
                    result["depth_map"],
                    str(output_dir / vis_filename)
                )
                
                result["depth_file"] = str(output_dir / depth_filename)
                result["visualization"] = str(output_dir / vis_filename)
                # Don't store large arrays in results list
                del result["depth_map"]
                del result["raw_depth"]
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"  Depth estimation failed for {img_path}: {e}")
                results.append({"error": str(e), "source_image": img_path})
        
        # Save summary
        summary_path = output_dir / "depth_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
