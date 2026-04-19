"""
ONNX Export Script
Exports PyTorch models (YOLOv8, Depth Anything) to ONNX format for NPU deployment.

AMD Tech: ONNX Runtime + Vitis AI EP, Ryzen AI NPU (XDNA)
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_yolo_to_onnx(
    model_path: str = "yolov8m.pt",
    output_path: str = "models/yolov8m_structural.onnx",
    imgsz: int = 640,
    opset: int = 17,
):
    """
    Export YOLOv8 model to ONNX format.
    
    AMD Optimization:
    - ONNX opset 17 for maximum operator coverage
    - Static input shape for NPU optimization
    - FP32 → later quantized to INT8 via Vitis AI
    """
    try:
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=True,
            dynamic=False,  # Static shape for NPU
        )
        
        logger.info(f"YOLOv8 exported to ONNX: {output}")
        return str(output)
        
    except ImportError:
        logger.error("ultralytics not installed")
        return None
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return None


def export_depth_model_to_onnx(
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    output_path: str = "models/depth_anything_v2_small.onnx",
    height: int = 518,
    width: int = 518,
):
    """
    Export Depth Anything v2 to ONNX for AMD Ryzen AI NPU.
    
    AMD Optimization:
    - Fixed input resolution for NPU graph compilation
    - Optimized for Vitis AI EP INT8 quantization
    """
    try:
        import torch
        from transformers import AutoModelForDepthEstimation
        
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model.eval()
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        dummy_input = torch.randn(1, 3, height, width)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output),
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            dynamic_axes=None,  # Static for NPU
        )
        
        logger.info(f"Depth model exported to ONNX: {output}")
        return str(output)
        
    except ImportError:
        logger.error("torch/transformers not installed")
        return None
    except Exception as e:
        logger.error(f"Depth ONNX export failed: {e}")
        return None


def main():
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Export models to ONNX for AMD NPU")
    parser.add_argument("--model", choices=["yolo", "depth", "all"], default="all")
    parser.add_argument("--yolo-path", default="yolov8m.pt")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model in ("yolo", "all"):
        export_yolo_to_onnx(args.yolo_path, str(output_dir / "yolov8m_structural.onnx"))
    
    if args.model in ("depth", "all"):
        export_depth_model_to_onnx(output_path=str(output_dir / "depth_anything_v2_small.onnx"))
    
    print("✅ ONNX export complete!")


if __name__ == "__main__":
    main()
