#!/usr/bin/env python3
"""
CityMind — Smart Defect Generator
Generates realistic, contextually-aware defects that target actual
building/infrastructure regions, NOT sky or vegetation.

Uses frame-level analysis to place bounding boxes only on built structures.
"""

import json
import random
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FRAMES_DIR = PROJECT_ROOT / "output" / "demo_rich" / "frames"
OUTPUT_DIR = PROJECT_ROOT / "output" / "demo_rich"

random.seed(2026)  # Reproducible

# Urban infrastructure defect types with realistic subtypes
DEFECT_TYPES = [
    {
        "type": "facade_crack",
        "subtypes": ["hairline", "structural", "diagonal_shear", "horizontal"],
        "severity_range": (4.0, 9.5),
        "priority_map": {9.0: "critical", 7.0: "high", 5.0: "medium"},
        "methods": ["yolov8", "rt-detr"],
        "description_templates": [
            "Vertical crack on {zone} facade, {w:.1f}m estimated length",
            "Diagonal shear crack across {zone} wall, penetrating render layer",
            "Hairline fractures on {zone} exterior, moisture ingress risk",
        ]
    },
    {
        "type": "spalling",
        "subtypes": ["concrete_delamination", "render_detachment", "surface_pop_out"],
        "severity_range": (3.5, 8.5),
        "priority_map": {8.0: "critical", 6.0: "high", 4.5: "medium"},
        "methods": ["yolov8", "rt-detr"],
        "description_templates": [
            "Concrete spalling on {zone} column, {area:.2f}m² affected area",
            "Render delamination on {zone} parapet, exposing substrate",
            "Surface pop-out on {zone} wall panel, freeze-thaw damage pattern",
        ]
    },
    {
        "type": "corrosion_stain",
        "subtypes": ["rust_bleeding", "efflorescence", "water_stain"],
        "severity_range": (2.5, 7.0),
        "priority_map": {6.5: "high", 4.5: "medium"},
        "methods": ["yolov8"],
        "description_templates": [
            "Rust staining below {zone} balcony railing, indicating rebar corrosion",
            "Efflorescence deposits on {zone} wall, salt crystallization active",
            "Water damage stain on {zone} soffit, drainage system failure",
        ]
    },
    {
        "type": "structural_settlement",
        "subtypes": ["differential", "uniform", "tilting"],
        "severity_range": (6.0, 9.8),
        "priority_map": {8.5: "critical", 7.0: "high"},
        "methods": ["rt-detr", "depth_analysis"],
        "description_templates": [
            "Differential settlement detected at {zone} foundation, {delta:.1f}mm offset",
            "Structural tilting of {zone} facade, plumb deviation {delta:.1f}mm/m",
        ]
    },
    {
        "type": "window_damage",
        "subtypes": ["frame_degradation", "seal_failure", "glass_crack"],
        "severity_range": (2.0, 5.5),
        "priority_map": {5.0: "medium", 3.0: "low"},
        "methods": ["yolov8"],
        "description_templates": [
            "Window frame deterioration on {zone}, sealant degradation",
            "Glazing seal failure on {zone} fenestration, thermal bridge risk",
        ]
    },
    {
        "type": "roof_deterioration",
        "subtypes": ["membrane_damage", "flashing_failure", "ponding"],
        "severity_range": (4.0, 8.0),
        "priority_map": {7.5: "critical", 5.5: "high", 4.0: "medium"},
        "methods": ["yolov8", "rt-detr"],
        "description_templates": [
            "Roof membrane damage on {zone}, UV degradation pattern",
            "Flashing failure at {zone} parapet junction, water ingress active",
        ]
    },
    {
        "type": "vegetation_intrusion",
        "subtypes": ["root_damage", "moss_growth", "vine_penetration"],
        "severity_range": (2.0, 6.0),
        "priority_map": {5.5: "medium", 3.0: "low"},
        "methods": ["yolov8"],
        "description_templates": [
            "Biological growth on {zone} facade, moisture retention risk",
            "Root intrusion at {zone} foundation, structural concern",
        ]
    },
]

# Zone definitions matching drone footage of a town/district
ZONES = [
    {"id": "block_a", "name": "Block A — Residential", "type": "residential_complex"},
    {"id": "block_b", "name": "Block B — Commercial", "type": "commercial_building"},
    {"id": "tower_1", "name": "Tower 1 — Mixed Use", "type": "high_rise"},
    {"id": "tower_2", "name": "Tower 2 — Office", "type": "office_tower"},
    {"id": "civic_center", "name": "Civic Center", "type": "public_building"},
    {"id": "infrastructure", "name": "Infrastructure — Roads & Utilities", "type": "infrastructure"},
]


def detect_building_regions(img):
    """
    Analyze a frame to find regions that contain buildings/infrastructure.
    Returns list of (x1, y1, x2, y2) bounding boxes for valid detection regions.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Mask out sky (top portion, low saturation, high value)
    sky_mask = np.zeros((h, w), dtype=np.uint8)
    # Scan from top: sky is typically bright, low-saturation
    for row in range(h):
        row_hsv = hsv[row, :, :]
        mean_s = row_hsv[:, 1].mean()
        mean_v = row_hsv[:, 2].mean()
        if mean_s < 50 and mean_v > 160:
            sky_mask[row, :] = 255
        else:
            # Once we hit non-sky, everything below is valid
            break

    # Step 2: Find texture-rich regions (buildings have edges)
    edges = cv2.Canny(gray, 50, 150)
    # Dilate edges to create contiguous regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    edge_regions = cv2.dilate(edges, kernel, iterations=3)

    # Step 3: Combine - valid region = not sky AND has texture
    valid_mask = (sky_mask == 0) & (edge_regions > 0)
    
    # Step 4: Find contours of valid regions
    valid_uint8 = valid_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(valid_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    min_area = w * h * 0.01  # At least 1% of frame
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch > min_area:
            # Ensure not too close to top (sky zone)
            if y > h * 0.1:
                regions.append((x, y, x + cw, y + ch))
    
    # If no good regions found, use lower 2/3 of frame (safe fallback for urban footage)
    if not regions:
        margin_top = int(h * 0.35)
        margin_side = int(w * 0.05)
        regions.append((margin_side, margin_top, w - margin_side, h - int(h * 0.05)))

    return regions


def generate_defect_in_region(region, img_w, img_h, defect_type_info, zone, defect_id, frame_idx):
    """Generate a single defect within a valid building region."""
    x1, y1, x2, y2 = region
    rw = x2 - x1
    rh = y2 - y1

    # Defect size: 5-35% of region size (realistic for drone inspection)
    bbox_w = random.randint(int(rw * 0.08), int(rw * 0.35))
    bbox_h = random.randint(int(rh * 0.08), int(rh * 0.30))

    # Random position within region
    bx1 = random.randint(x1, max(x1, x2 - bbox_w))
    by1 = random.randint(y1, max(y1, y2 - bbox_h))
    bx2 = min(bx1 + bbox_w, img_w - 1)
    by2 = min(by1 + bbox_h, img_h - 1)

    severity = round(random.uniform(*defect_type_info["severity_range"]), 1)
    confidence = round(random.uniform(0.72, 0.97), 2)

    # Determine priority from severity
    priority = "low"
    for threshold, prio in sorted(defect_type_info["priority_map"].items(), reverse=True):
        if severity >= threshold:
            priority = prio
            break

    subtype = random.choice(defect_type_info["subtypes"])
    method = random.choice(defect_type_info["methods"])
    desc_template = random.choice(defect_type_info["description_templates"])
    description = desc_template.format(
        zone=zone["name"],
        w=round(random.uniform(0.3, 2.5), 1),
        area=round(random.uniform(0.1, 1.8), 2),
        delta=round(random.uniform(2.0, 15.0), 1),
    )

    return {
        "id": f"DEF-{defect_id:03d}",
        "defect_type": defect_type_info["type"],
        "subtype": subtype,
        "severity": severity,
        "confidence": confidence,
        "bbox": [bx1, by1, bx2, by2],
        "frame_idx": frame_idx,
        "detection_method": method,
        "zone": zone["id"],
        "zone_name": zone["name"],
        "location_description": description,
        "priority": priority,
    }


def generate_all_defects():
    """Generate contextually-aware defects across all frames."""
    defects = []
    defect_id = 1

    # Frames that should have defects (not every frame — more realistic)
    # Cluster defects in certain areas of the flight path
    active_frames = [2, 4, 5, 7, 8, 10, 11, 14, 15, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34]

    for frame_idx in active_frames:
        padded = f"frame_{frame_idx + 1:04d}.jpg"
        frame_path = FRAMES_DIR / padded

        if not frame_path.exists():
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        regions = detect_building_regions(img)

        if not regions:
            continue

        # 1-3 defects per active frame
        n_defects = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]

        for _ in range(n_defects):
            region = random.choice(regions)
            defect_type_info = random.choice(DEFECT_TYPES)
            zone = random.choice(ZONES)

            defect = generate_defect_in_region(
                region, w, h, defect_type_info, zone, defect_id, frame_idx
            )
            defects.append(defect)
            defect_id += 1

    return defects


def generate_twin_data(defects):
    """Generate comprehensive digital twin data."""
    # Zone analysis
    zone_analysis = {}
    for zone in ZONES:
        zone_defects = [d for d in defects if d["zone"] == zone["id"]]
        if zone_defects:
            avg_sev = sum(d["severity"] for d in zone_defects) / len(zone_defects)
            health = max(10, int(100 - avg_sev * 8 - len(zone_defects) * 3))
        else:
            health = random.randint(75, 95)
        zone_analysis[zone["id"]] = {
            "health": health,
            "defects": len(zone_defects),
            "name": zone["name"],
            "type": zone["type"],
        }

    # Overall health
    all_health = [z["health"] for z in zone_analysis.values()]
    avg_health = sum(all_health) / len(all_health) if all_health else 60
    critical = sum(1 for d in defects if d["priority"] == "critical")
    high = sum(1 for d in defects if d["priority"] == "high")
    medium = sum(1 for d in defects if d["priority"] == "medium")
    low = sum(1 for d in defects if d["priority"] == "low")

    grade = "A" if avg_health >= 85 else "B" if avg_health >= 70 else "C" if avg_health >= 55 else "D" if avg_health >= 40 else "F"
    status = "Good" if avg_health >= 85 else "Fair" if avg_health >= 70 else "Needs Attention" if avg_health >= 55 else "Poor" if avg_health >= 40 else "Critical"

    # Defect type analysis
    by_type = {}
    for d in defects:
        t = d["defect_type"]
        if t not in by_type:
            by_type[t] = {"count": 0, "total_severity": 0}
        by_type[t]["count"] += 1
        by_type[t]["total_severity"] += d["severity"]
    for t in by_type:
        by_type[t]["avg_severity"] = round(by_type[t]["total_severity"] / by_type[t]["count"], 1)
        del by_type[t]["total_severity"]

    return {
        "twin_id": "TWIN-DISTRICT-20260228",
        "structure_type": "Urban District — Mixed Use Infrastructure",
        "inspection_type": "Aerial Drone Survey (DJI Mavic 3)",
        "health_index": {
            "score": round(avg_health, 1),
            "grade": grade,
            "status": status,
            "defect_count": len(defects),
            "critical_count": critical,
            "high_count": high,
            "medium_count": medium,
            "low_count": low,
        },
        "zone_analysis": zone_analysis,
        "defect_analysis": {
            "total_defects": len(defects),
            "critical_defects": [d for d in defects if d["priority"] == "critical"],
            "by_type": by_type,
        },
    }


def generate_agents_data(defects):
    """Generate AI agent analysis data."""
    critical_defects = [d for d in defects if d["priority"] == "critical"]
    high_defects = [d for d in defects if d["priority"] == "high"]

    classified = []
    for d in defects[:12]:  # Top 12 most relevant
        actions = {
            "facade_crack": "Schedule structural engineer assessment within 72 hours. Install crack monitors for ongoing measurement.",
            "spalling": "Commission concrete repair contractor. Remove loose material and apply protective coating.",
            "corrosion_stain": "Investigate source of moisture. Test rebar for active corrosion using half-cell potential measurement.",
            "structural_settlement": "URGENT: Install settlement markers immediately. Restrict occupancy until geotechnical review completed.",
            "window_damage": "Schedule window replacement during next maintenance cycle. Apply temporary weatherproofing.",
            "roof_deterioration": "Inspect roof membrane and flashings. Schedule waterproofing repair before next rainfall.",
            "vegetation_intrusion": "Remove biological growth and apply biocide treatment. Improve drainage to reduce moisture.",
        }
        classified.append({
            "id": d["id"],
            "recommendation": actions.get(d["defect_type"], "Schedule detailed inspection and repair assessment."),
        })

    return {
        "inspector": {
            "agent": "Inspector Agent",
            "model": "Llama 3.1 8B (INT4 via Ryzen AI NPU)",
            "summary": f"Aerial survey detected {len(defects)} defects across the district. "
                       f"{len(critical_defects)} critical findings require immediate action: "
                       + ", ".join(d["id"] for d in critical_defects[:3])
                       + ". Priority areas: facade integrity, structural settlement monitoring, and water ingress mitigation.",
            "classified_defects": classified,
        },
        "safety": {
            "agent": "Safety Assessment Agent",
            "model": "Mistral 7B (INT4 via AMD NPU)",
            "risk_score": min(95, 30 + len(critical_defects) * 20 + len(high_defects) * 5),
            "risk_level": "High" if critical_defects else "Moderate",
            "immediate_actions": [
                "Cordon off areas below spalling zones — falling debris risk",
                "Install temporary shoring at settlement-affected structures",
                "Deploy moisture barriers at active water ingress points",
                "Schedule emergency structural review for critical-rated buildings",
            ][:2 + len(critical_defects)],
            "monitoring_recommendations": [
                "Install IoT crack-width sensors on critical facades",
                "Deploy tilt sensors on structures showing settlement",
                "Schedule monthly drone re-surveys for change detection",
                "Establish weather-triggered inspection protocols",
            ],
            "risk_factors": [
                {"factor": "Structural Integrity", "score": min(90, 20 + len([d for d in defects if d["defect_type"] in ("facade_crack", "structural_settlement")]) * 12)},
                {"factor": "Water Damage", "score": min(85, 15 + len([d for d in defects if d["defect_type"] in ("corrosion_stain", "roof_deterioration")]) * 10)},
                {"factor": "Material Degradation", "score": min(80, 10 + len([d for d in defects if d["defect_type"] in ("spalling", "corrosion_stain")]) * 10)},
                {"factor": "Public Safety", "score": min(75, 25 + len(critical_defects) * 15)},
            ],
        },
        "compliance": {
            "overall_compliance": "68% — Non-Compliant",
            "total_violations": 5,
            "violations": [
                {"code": "IBC 1604.5", "description": "Structural integrity compromised — visible settlement and facade cracking exceed allowable limits", "severity": "critical", "defect_id": critical_defects[0]["id"] if critical_defects else "DEF-001"},
                {"code": "IBC 1503.2", "description": "Roof drainage and waterproofing failures — active water ingress detected", "severity": "high", "defect_id": "DEF-003"},
                {"code": "ACI 318-19 §20.5", "description": "Concrete cover insufficient — corrosion staining indicates rebar exposure risk", "severity": "high", "defect_id": "DEF-005"},
                {"code": "ASCE 7-22 §2.3", "description": "Facade elements at risk of detachment — spalling concrete overhead", "severity": "medium", "defect_id": "DEF-008"},
                {"code": "IBC 2403.2", "description": "Fenestration seal failures — compromised thermal envelope integrity", "severity": "low", "defect_id": "DEF-010"},
            ],
            "compliant_elements": [
                "Foundation bearing capacity (no major settlement beyond spec)",
                "Fire egress pathways — clear and accessible",
                "Electrical infrastructure — no exposed hazards detected",
            ],
        },
    }


def main():
    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  CityMind — Smart Defect Generator                      ║")
    print("  ║  Context-aware detection: buildings only, no sky/trees   ║")
    print("  ╚══════════════════════════════════════════════════════════╝\n")

    # Generate defects
    defects = generate_all_defects()
    print(f"  Generated {len(defects)} contextually-aware defects")

    # Stats
    types = {}
    for d in defects:
        types[d["defect_type"]] = types.get(d["defect_type"], 0) + 1
    print(f"  Types: {dict(sorted(types.items()))}")

    priorities = {}
    for d in defects:
        priorities[d["priority"]] = priorities.get(d["priority"], 0) + 1
    print(f"  Priorities: {dict(sorted(priorities.items()))}")

    frames_used = set(d["frame_idx"] for d in defects)
    print(f"  Frames with defects: {len(frames_used)}/{35}")

    # Save defects
    defects_path = OUTPUT_DIR / "defects.json"
    defects_path.write_text(json.dumps(defects, indent=2))
    print(f"\n  Saved: {defects_path}")

    # Generate twin data
    twin = generate_twin_data(defects)
    twin_path = OUTPUT_DIR / "twin.json"
    twin_path.write_text(json.dumps(twin, indent=2))
    print(f"  Saved: {twin_path}")

    # Generate agents data
    agents = generate_agents_data(defects)
    agents_path = OUTPUT_DIR / "agents.json"
    agents_path.write_text(json.dumps(agents, indent=2))
    print(f"  Saved: {agents_path}")

    print("\n  ✓ All data regenerated with smart building-aware detection!\n")


if __name__ == "__main__":
    main()
