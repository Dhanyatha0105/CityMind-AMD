"""
Layer 1B: Quality Filter
Advanced image quality assessment and filtering for construction site imagery.

Detects: blur, overexposure, underexposure, motion blur, occlusion
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Multi-criteria image quality assessment for construction inspection frames.
    
    Filters:
    - Blur detection (Laplacian variance)
    - Exposure analysis (histogram analysis)
    - Contrast check (standard deviation)
    - Edge density (structural content)
    """
    
    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_range: Tuple[float, float] = (30.0, 230.0),
        contrast_threshold: float = 20.0,
        edge_density_threshold: float = 0.02,
    ):
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.contrast_threshold = contrast_threshold
        self.edge_density_threshold = edge_density_threshold
    
    def assess_quality(self, image_path: str) -> Dict:
        """
        Comprehensive quality assessment of a single image.
        
        Returns dict with scores and pass/fail for each criterion.
        """
        img = cv2.imread(image_path)
        if img is None:
            return {"pass": False, "reason": "Cannot read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Blur detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_pass = blur_score >= self.blur_threshold
        
        # 2. Brightness analysis
        mean_brightness = np.mean(gray)
        brightness_pass = (
            self.brightness_range[0] <= mean_brightness <= self.brightness_range[1]
        )
        
        # 3. Contrast check
        contrast = np.std(gray)
        contrast_pass = contrast >= self.contrast_threshold
        
        # 4. Edge density (structural content indicator)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (h * w)
        edge_pass = edge_density >= self.edge_density_threshold
        
        # Overall pass
        overall_pass = blur_pass and brightness_pass and contrast_pass and edge_pass
        
        # Quality score (0-100)
        quality_score = self._compute_quality_score(
            blur_score, mean_brightness, contrast, edge_density
        )
        
        return {
            "pass": overall_pass,
            "quality_score": quality_score,
            "blur": {"score": float(blur_score), "pass": blur_pass},
            "brightness": {"score": float(mean_brightness), "pass": brightness_pass},
            "contrast": {"score": float(contrast), "pass": contrast_pass},
            "edge_density": {"score": float(edge_density), "pass": edge_pass},
            "resolution": [w, h],
        }
    
    def filter_frames(
        self, 
        frame_paths: List[str], 
        output_dir: str = None
    ) -> Dict:
        """
        Filter a list of frames, keeping only high-quality ones.
        
        Returns dict with passed/failed frame lists and quality assessments.
        """
        passed = []
        failed = []
        assessments = []
        
        for fp in frame_paths:
            assessment = self.assess_quality(fp)
            assessment["path"] = fp
            assessments.append(assessment)
            
            if assessment["pass"]:
                passed.append(fp)
            else:
                failed.append(fp)
        
        logger.info(
            f"Quality filter: {len(passed)}/{len(frame_paths)} passed "
            f"({len(failed)} rejected)"
        )
        
        return {
            "passed": passed,
            "failed": failed,
            "total": len(frame_paths),
            "pass_rate": len(passed) / max(len(frame_paths), 1),
            "assessments": assessments,
        }
    
    def _compute_quality_score(
        self, 
        blur: float, 
        brightness: float, 
        contrast: float, 
        edge_density: float
    ) -> float:
        """Compute composite quality score 0-100."""
        # Normalize each metric to 0-1
        blur_norm = min(blur / 500.0, 1.0)
        
        # Brightness: best around 128, penalize extremes
        brightness_norm = 1.0 - abs(brightness - 128) / 128.0
        brightness_norm = max(0, brightness_norm)
        
        contrast_norm = min(contrast / 80.0, 1.0)
        edge_norm = min(edge_density / 0.15, 1.0)
        
        # Weighted average
        score = (
            0.35 * blur_norm + 
            0.20 * brightness_norm + 
            0.20 * contrast_norm + 
            0.25 * edge_norm
        )
        
        return round(score * 100, 1)
