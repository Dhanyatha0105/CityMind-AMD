"""
Layer 2D: NPU Inference Engine
ONNX Runtime wrapper with AMD Vitis AI Execution Provider for Ryzen AI NPU.

AMD Tech: ONNX Runtime + Vitis AI EP, Ryzen AI NPU (XDNA), Ryzen AI Software v1.7

This module provides the bridge between trained models and AMD hardware:
1. Models trained on MI300X with ROCm 7.2
2. Exported to ONNX format
3. Quantized INT8 via Vitis AI Quantizer
4. Deployed here via ONNX Runtime with Vitis AI Execution Provider
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging
import time

logger = logging.getLogger(__name__)


class NPUInferenceEngine:
    """
    AMD Ryzen AI NPU inference engine using ONNX Runtime + Vitis AI EP.
    
    Supports multiple execution providers with automatic fallback:
    1. VitisAIExecutionProvider (AMD Ryzen AI NPU — target production)
    2. DmlExecutionProvider (DirectML — Windows AMD GPU fallback)
    3. CoreMLExecutionProvider (macOS — development fallback)
    4. CPUExecutionProvider (universal fallback)
    
    AMD Technology Stack:
    - Ryzen AI NPU (XDNA architecture) — 40+ TOPS AI performance
    - ONNX Runtime with Vitis AI Execution Provider
    - Vitis AI Quantizer for INT8/INT4 optimization
    - Ryzen AI Software v1.7 (16K context, MoE support, BF16)
    """
    
    def __init__(
        self,
        vaip_config_path: str = "models/vaip_config.json",
        preferred_provider: str = "auto",
    ):
        self.vaip_config_path = vaip_config_path
        self.preferred_provider = preferred_provider
        self._sessions = {}
        self._available_providers = self._detect_providers()
    
    def _detect_providers(self) -> List[str]:
        """Detect available ONNX Runtime execution providers."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {available}")
            return available
        except ImportError:
            logger.warning("ONNX Runtime not installed")
            return []
    
    def get_best_provider(self) -> tuple:
        """
        Select the best available execution provider.
        Priority: VitisAI > DML > CoreML > CPU
        """
        if self.preferred_provider != "auto":
            return self.preferred_provider, {}
        
        if "VitisAIExecutionProvider" in self._available_providers:
            return "VitisAIExecutionProvider", {"config_file": self.vaip_config_path}
        elif "DmlExecutionProvider" in self._available_providers:
            return "DmlExecutionProvider", {}
        elif "CoreMLExecutionProvider" in self._available_providers:
            return "CoreMLExecutionProvider", {}
        else:
            return "CPUExecutionProvider", {}
    
    def load_model(self, model_path: str, model_name: str = None) -> str:
        """
        Load an ONNX model with the best available provider.
        
        Args:
            model_path: Path to .onnx model file
            model_name: Optional name for caching. Defaults to filename.
            
        Returns:
            Session key for later inference calls
        """
        import onnxruntime as ort
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        key = model_name or model_path.stem
        
        provider, options = self.get_best_provider()
        logger.info(f"Loading {key} on {provider}")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = [provider]
        provider_options = [options] if options else None
        
        # Always add CPU as fallback
        if provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
            if provider_options:
                provider_options.append({})
        
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
        
        self._sessions[key] = {
            "session": session,
            "provider": provider,
            "model_path": str(model_path),
            "input_names": [i.name for i in session.get_inputs()],
            "output_names": [o.name for o in session.get_outputs()],
            "input_shapes": [i.shape for i in session.get_inputs()],
        }
        
        logger.info(f"Model '{key}' loaded: inputs={self._sessions[key]['input_names']}")
        return key
    
    def infer(
        self, 
        model_key: str, 
        inputs: Dict[str, np.ndarray],
        benchmark: bool = False
    ) -> Dict:
        """
        Run inference on loaded model.
        
        Args:
            model_key: Key returned by load_model
            inputs: Dict mapping input names to numpy arrays
            benchmark: If True, measure and return latency
            
        Returns:
            Dict with outputs and optional timing info
        """
        if model_key not in self._sessions:
            raise ValueError(f"Model '{model_key}' not loaded. Call load_model first.")
        
        session_info = self._sessions[model_key]
        session = session_info["session"]
        
        start_time = time.time() if benchmark else 0
        
        outputs = session.run(None, inputs)
        
        elapsed = (time.time() - start_time) * 1000 if benchmark else 0
        
        result = {
            "outputs": outputs,
            "model": model_key,
            "provider": session_info["provider"],
        }
        
        if benchmark:
            result["latency_ms"] = round(elapsed, 2)
            result["throughput_fps"] = round(1000 / max(elapsed, 0.01), 1)
        
        return result
    
    def benchmark_model(
        self, 
        model_key: str, 
        sample_input: Dict[str, np.ndarray],
        warmup_runs: int = 3,
        benchmark_runs: int = 10,
    ) -> Dict:
        """
        Benchmark a model's inference performance.
        
        Returns latency statistics (mean, p50, p95, p99, throughput).
        """
        if model_key not in self._sessions:
            raise ValueError(f"Model '{model_key}' not loaded")
        
        session = self._sessions[model_key]["session"]
        
        # Warmup
        for _ in range(warmup_runs):
            session.run(None, sample_input)
        
        # Benchmark
        latencies = []
        for _ in range(benchmark_runs):
            start = time.time()
            session.run(None, sample_input)
            latencies.append((time.time() - start) * 1000)
        
        latencies.sort()
        
        return {
            "model": model_key,
            "provider": self._sessions[model_key]["provider"],
            "runs": benchmark_runs,
            "mean_ms": round(np.mean(latencies), 2),
            "median_ms": round(np.median(latencies), 2),
            "p95_ms": round(np.percentile(latencies, 95), 2),
            "p99_ms": round(np.percentile(latencies, 99), 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "throughput_fps": round(1000 / np.mean(latencies), 1),
        }
    
    def get_model_info(self, model_key: str) -> Dict:
        """Get information about a loaded model."""
        if model_key not in self._sessions:
            return {"error": "Model not loaded"}
        
        info = self._sessions[model_key]
        return {
            "model": model_key,
            "provider": info["provider"],
            "model_path": info["model_path"],
            "inputs": info["input_names"],
            "outputs": info["output_names"],
            "input_shapes": [str(s) for s in info["input_shapes"]],
        }
    
    def list_models(self) -> List[str]:
        """List all loaded model keys."""
        return list(self._sessions.keys())
    
    @staticmethod
    def get_system_info() -> Dict:
        """Get AMD hardware and software information."""
        info = {
            "amd_target_platform": "AMD Ryzen AI (XDNA NPU)",
            "amd_npu_performance": "40+ TOPS (INT8)",
            "amd_software_stack": {
                "onnx_runtime": "1.16+",
                "vitis_ai_ep": "Vitis AI Execution Provider",
                "ryzen_ai_software": "v1.7",
                "features": [
                    "16K context window for LLMs",
                    "MoE (Mixture of Experts) support",
                    "BF16 pipeline optimization",
                    "VLM (Vision Language Model) support",
                ],
            },
            "deployment_pipeline": {
                "step_1": "Train on AMD Instinct MI300X with ROCm 7.2",
                "step_2": "Export model to ONNX format",
                "step_3": "Quantize INT8 via Vitis AI Quantizer",
                "step_4": "Deploy on Ryzen AI NPU via ONNX RT + Vitis AI EP",
            },
        }
        
        try:
            import onnxruntime as ort
            info["onnxruntime_version"] = ort.__version__
            info["available_providers"] = ort.get_available_providers()
        except ImportError:
            info["onnxruntime_version"] = "not installed"
        
        return info
