"""
Layer 1C: Metadata Extractor
Extracts GPS coordinates, timestamps, and camera info from video/images.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extract metadata from video and image files for georeferencing
    and temporal tracking of inspection data.
    """
    
    def extract_video_metadata(self, video_path: str) -> Dict:
        """Extract metadata from video file using ffprobe."""
        video_path = Path(video_path)
        
        if not video_path.exists():
            return {
                "filename": video_path.name,
                "path": str(video_path),
                "error": "File not found",
            }
        
        metadata = {
            "filename": video_path.name,
            "path": str(video_path),
            "file_size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
            "extraction_timestamp": datetime.now().isoformat(),
            "gps": None,
            "camera": None,
        }
        
        # Try ffprobe for detailed metadata
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", "-show_streams",
                    str(video_path)
                ],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                
                # Extract video stream info
                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        metadata["video_codec"] = stream.get("codec_name")
                        metadata["width"] = stream.get("width")
                        metadata["height"] = stream.get("height")
                        metadata["fps"] = eval(stream.get("r_frame_rate", "0/1"))
                        metadata["duration"] = float(stream.get("duration", 0))
                        break
                
                # Extract format tags (may contain GPS)
                tags = probe_data.get("format", {}).get("tags", {})
                if "location" in tags:
                    metadata["gps"] = self._parse_gps_string(tags["location"])
                elif "com.apple.quicktime.location.ISO6709" in tags:
                    metadata["gps"] = self._parse_gps_string(
                        tags["com.apple.quicktime.location.ISO6709"]
                    )
                
                if "com.apple.quicktime.model" in tags:
                    metadata["camera"] = tags["com.apple.quicktime.model"]
                elif "model" in tags:
                    metadata["camera"] = tags["model"]
                    
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"ffprobe not available or failed: {e}")
        
        return metadata
    
    def _parse_gps_string(self, gps_str: str) -> Optional[Dict]:
        """Parse GPS string from video metadata (ISO 6709 format)."""
        try:
            # Format: +37.7749-122.4194+0.000/
            gps_str = gps_str.strip().rstrip("/")
            parts = []
            current = ""
            for char in gps_str:
                if char in "+-" and current:
                    parts.append(float(current))
                    current = char
                else:
                    current += char
            if current:
                parts.append(float(current))
            
            if len(parts) >= 2:
                return {
                    "latitude": parts[0],
                    "longitude": parts[1],
                    "altitude": parts[2] if len(parts) > 2 else None,
                }
        except (ValueError, IndexError):
            pass
        return None
    
    def create_inspection_metadata(
        self,
        video_metadata: Dict,
        frame_count: int,
        structure_type: str = "unknown",
        inspector_notes: str = "",
    ) -> Dict:
        """Create a complete inspection metadata record."""
        return {
            "inspection_id": f"INS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "source_video": video_metadata,
            "frame_count": frame_count,
            "structure_type": structure_type,
            "inspector_notes": inspector_notes,
            "processing_platform": {
                "target": "AMD Ryzen AI (XDNA NPU)",
                "framework": "CityMind v1.0",
                "amd_technologies": [
                    "Ryzen AI NPU", "CVML Library", "Vitis AI EP",
                    "ONNX Runtime", "GAIA Framework"
                ],
            },
        }
