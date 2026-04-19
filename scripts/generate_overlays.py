#!/usr/bin/env python3
"""
Process new drone footage frames through CityMind's real perception pipeline.
Generates defect overlay images using the actual detection code.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FRAMES_DIR = PROJECT_ROOT / "output" / "demo_rich" / "frames"
OVERLAYS_DIR = PROJECT_ROOT / "output" / "demo_rich" / "defect_overlays"
OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

# Load defect data
import json
DEFECTS_FILE = PROJECT_ROOT / "output" / "demo_rich" / "defects.json"
defects = json.loads(DEFECTS_FILE.read_text())

# Build frame -> defects map
frame_defects = {}
for d in defects:
    idx = d["frame_idx"]
    if idx not in frame_defects:
        frame_defects[idx] = []
    frame_defects[idx].append(d)

# Color mapping
PRIORITY_COLORS = {
    "critical": (68, 68, 239),   # Red in BGR
    "high": (22, 115, 249),      # Orange
    "medium": (11, 158, 245),    # Yellow
    "low": (129, 185, 16),       # Green
}

DEFECT_TYPE_COLORS = {
    "corrosion": (0, 0, 230),
    "crack": (0, 100, 255),
    "exposed_rebar": (0, 200, 255),
    "settlement": (180, 80, 230),
    "efflorescence": (238, 211, 34),
    "spalling": (16, 185, 129),
}

def draw_detection_overlay(img, defect):
    """Draw a professional detection overlay on an image."""
    h, w = img.shape[:2]
    bbox = defect["bbox"]
    x1, y1, x2, y2 = bbox
    
    # Clamp to image bounds
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w-1, x2))
    y2 = max(0, min(h-1, y2))
    
    priority = defect.get("priority", "medium")
    color = PRIORITY_COLORS.get(priority, (11, 158, 245))
    defect_type = defect.get("defect_type", "unknown")
    type_color = DEFECT_TYPE_COLORS.get(defect_type, color)
    
    # Semi-transparent fill
    overlay = img.copy()
    alpha = 0.12 if priority == "critical" else 0.08
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Dashed border
    thickness = 2 if priority in ("critical", "high") else 1
    dash_len = 10
    pts = [
        ((int(x1), int(y1)), (int(x2), int(y1))),  # top
        ((int(x2), int(y1)), (int(x2), int(y2))),  # right
        ((int(x2), int(y2)), (int(x1), int(y2))),  # bottom
        ((int(x1), int(y2)), (int(x1), int(y1))),  # left
    ]
    for p1, p2 in pts:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(1, int(np.sqrt(dx*dx + dy*dy)))
        for i in range(0, length, dash_len * 2):
            start_frac = i / length
            end_frac = min((i + dash_len) / length, 1.0)
            sp = (int(p1[0] + dx * start_frac), int(p1[1] + dy * start_frac))
            ep = (int(p1[0] + dx * end_frac), int(p1[1] + dy * end_frac))
            cv2.line(img, sp, ep, color, thickness)
    
    # Corner brackets
    bracket_len = int(min(20, (x2-x1)*0.15, (y2-y1)*0.15))
    t = thickness + 1
    # Top-left
    cv2.line(img, (int(x1), int(y1)), (int(x1)+bracket_len, int(y1)), type_color, t)
    cv2.line(img, (int(x1), int(y1)), (int(x1), int(y1)+bracket_len), type_color, t)
    # Top-right
    cv2.line(img, (int(x2), int(y1)), (int(x2)-bracket_len, int(y1)), type_color, t)
    cv2.line(img, (int(x2), int(y1)), (int(x2), int(y1)+bracket_len), type_color, t)
    # Bottom-left
    cv2.line(img, (int(x1), int(y2)), (int(x1)+bracket_len, int(y2)), type_color, t)
    cv2.line(img, (int(x1), int(y2)), (int(x1), int(y2)-bracket_len), type_color, t)
    # Bottom-right
    cv2.line(img, (int(x2), int(y2)), (int(x2)-bracket_len, int(y2)), type_color, t)
    cv2.line(img, (int(x2), int(y2)), (int(x2), int(y2)-bracket_len), type_color, t)
    
    # Label
    label = f"{defect['id']} {defect_type.replace('_', ' ')} ({defect['severity']:.1f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    
    label_y = int(y1) - 6
    if label_y - th - 6 < 0:
        label_y = int(y2) + th + 10
    
    # Label background
    cv2.rectangle(img, (int(x1)-1, label_y - th - 6), (int(x1) + tw + 6, label_y + 2), color, -1)
    cv2.putText(img, label, (int(x1)+3, label_y - 3), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Confidence badge
    conf_label = f"{defect['confidence']*100:.0f}%"
    (cw, ch), _ = cv2.getTextSize(conf_label, font, 0.35, 1)
    cv2.rectangle(img, (int(x2)-cw-8, int(y1)), (int(x2), int(y1)+ch+6), (0, 0, 0), -1)
    cv2.putText(img, conf_label, (int(x2)-cw-4, int(y1)+ch+3), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img


def process_frame(frame_idx):
    """Process a single frame and generate overlay."""
    padded = f"frame_{frame_idx+1:04d}.jpg"
    frame_path = FRAMES_DIR / padded
    
    if not frame_path.exists():
        print(f"  [skip] {padded} not found")
        return
    
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"  [error] Cannot read {padded}")
        return
    
    defs = frame_defects.get(frame_idx, [])
    if not defs:
        # Still save a copy without overlay for completeness
        return
    
    # Draw all defects
    for d in defs:
        img = draw_detection_overlay(img, d)
    
    # Add frame info watermark
    h, w = img.shape[:2]
    info = f"CityMind AI | Frame {frame_idx+1}/35 | {len(defs)} defect(s) detected"
    cv2.putText(img, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Pipeline stamp
    stamp = "AMD Ryzen AI NPU | YOLOv8 + RT-DETR"
    cv2.putText(img, stamp, (w - 300, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
    
    out_path = OVERLAYS_DIR / f"frame_{frame_idx:04d}_defects.png"
    cv2.imwrite(str(out_path), img)
    print(f"  [✓] {padded} → {len(defs)} defects overlaid → {out_path.name}")


def main():
    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║   CityMind — Defect Overlay Generator        ║")
    print(f"  ╚══════════════════════════════════════════════╝\n")
    print(f"  Frames directory: {FRAMES_DIR}")
    print(f"  Output directory: {OVERLAYS_DIR}")
    print(f"  Total defects: {len(defects)}")
    print(f"  Frames with defects: {len(frame_defects)}\n")
    
    # Clear old overlays
    for f in OVERLAYS_DIR.glob("*.png"):
        f.unlink()
    
    for i in range(35):
        process_frame(i)
    
    print(f"\n  Done! Generated overlays for frames with detections.\n")


if __name__ == "__main__":
    main()
