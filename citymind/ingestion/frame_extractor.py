"""
Layer 1A: Frame Extractor
Extracts keyframes from video at specified FPS with quality filtering.

Hardware: AMD Ryzen CPU (Zen 5)
Software: OpenCV, FFmpeg
Reuse: Adapted from Differentiable-Scan-to-BIM-v2 video_utils.py
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extracts high-quality keyframes from video for 3D reconstruction pipeline.
    
    Pipeline:
    1. Open video → extract frames at target FPS
    2. Compute blur score (Laplacian variance)
    3. Compute similarity (SSIM-based dedup)
    4. Output clean, non-redundant keyframes
    
    AMD Tech: Runs on AMD Ryzen CPU (Zen 5 architecture)
    """
    
    def __init__(
        self,
        target_fps: int = 3,
        blur_threshold: float = 100.0,
        similarity_threshold: float = 0.95,
        max_frames: int = 50,
        min_resolution: Tuple[int, int] = (320, 240),
    ):
        self.target_fps = target_fps
        self.blur_threshold = blur_threshold
        self.similarity_threshold = similarity_threshold
        self.max_frames = max_frames
        self.min_resolution = min_resolution
    
    def extract_frames(
        self, 
        video_path: str, 
        output_dir: str,
        return_metadata: bool = True
    ) -> dict:
        """
        Extract keyframes from video file.
        
        Args:
            video_path: Path to input video (MP4, AVI, MOV)
            output_dir: Directory to save extracted frames
            return_metadata: Whether to return frame metadata
            
        Returns:
            dict with keys: frames (list of paths), metadata (dict)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        logger.info(f"Video: {video_path.name}")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {video_fps:.1f}, Duration: {duration:.1f}s, Frames: {total_frames}")
        
        # Calculate frame interval
        frame_interval = max(1, int(video_fps / self.target_fps))
        
        extracted_frames = []
        frame_metadata = []
        prev_frame_gray = None
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process at target FPS interval
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # Check resolution
            h, w = frame.shape[:2]
            if w < self.min_resolution[0] or h < self.min_resolution[1]:
                frame_idx += 1
                continue
            
            # Compute blur score (Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score < self.blur_threshold:
                logger.debug(f"  Frame {frame_idx}: SKIP (blur={blur_score:.1f} < {self.blur_threshold})")
                frame_idx += 1
                continue
            
            # Check similarity to previous frame (SSIM-based dedup)
            if prev_frame_gray is not None:
                similarity = self._compute_similarity(prev_frame_gray, gray)
                if similarity > self.similarity_threshold:
                    logger.debug(f"  Frame {frame_idx}: SKIP (similarity={similarity:.3f})")
                    frame_idx += 1
                    continue
            
            # Save frame
            frame_name = f"frame_{saved_count:04d}.png"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            
            extracted_frames.append(str(frame_path))
            frame_metadata.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / video_fps if video_fps > 0 else 0,
                "blur_score": float(blur_score),
                "resolution": [w, h],
                "filename": frame_name,
            })
            
            prev_frame_gray = gray.copy()
            saved_count += 1
            
            if saved_count >= self.max_frames:
                logger.info(f"  Reached max frames ({self.max_frames})")
                break
            
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"  Extracted {saved_count} keyframes from {total_frames} total frames")
        
        # Build result
        result = {
            "frames": extracted_frames,
            "count": saved_count,
            "video_info": {
                "path": str(video_path),
                "filename": video_path.name,
                "resolution": [width, height],
                "fps": video_fps,
                "duration": duration,
                "total_frames": total_frames,
            },
            "extraction_params": {
                "target_fps": self.target_fps,
                "blur_threshold": self.blur_threshold,
                "similarity_threshold": self.similarity_threshold,
                "max_frames": self.max_frames,
            },
        }
        
        if return_metadata:
            result["frame_metadata"] = frame_metadata
        
        # Save metadata
        meta_path = output_dir / "extraction_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def _compute_similarity(self, img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
        """
        Compute structural similarity between two grayscale frames.
        Uses histogram correlation as a fast approximation of SSIM.
        """
        # Resize to common size for comparison
        target_size = (256, 256)
        img1_resized = cv2.resize(img1_gray, target_size)
        img2_resized = cv2.resize(img2_gray, target_size)
        
        # Histogram comparison (fast SSIM approximation)
        hist1 = cv2.calcHist([img1_resized], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2_resized], [0], None, [256], [0, 256])
        
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, correlation)
    
    @staticmethod
    def create_thumbnail_grid(
        frame_paths: List[str], 
        output_path: str,
        cols: int = 5,
        thumb_size: Tuple[int, int] = (200, 150)
    ) -> str:
        """Create a grid thumbnail of all extracted frames for visualization."""
        frames = []
        for fp in frame_paths[:25]:  # Max 25 in grid
            img = cv2.imread(fp)
            if img is not None:
                thumb = cv2.resize(img, thumb_size)
                frames.append(thumb)
        
        if not frames:
            return ""
        
        rows = (len(frames) + cols - 1) // cols
        # Pad with black frames
        while len(frames) < rows * cols:
            frames.append(np.zeros_like(frames[0]))
        
        grid_rows = []
        for r in range(rows):
            row_frames = frames[r * cols:(r + 1) * cols]
            grid_rows.append(np.hstack(row_frames))
        
        grid = np.vstack(grid_rows)
        cv2.imwrite(output_path, grid)
        return output_path
