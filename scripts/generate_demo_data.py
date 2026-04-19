#!/usr/bin/env python3
"""
CityMind Rich Demo Data Generator
Generates realistic-looking demo data for stunning dashboard screenshots.
Creates synthetic defect overlays, 3D point clouds, agent reports, and analytics.
"""

import sys
import json
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "demo_rich"


def generate_rich_demo_data():
    """Generate comprehensive demo data for dashboard screenshots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {}

    # ── 1. Pipeline Metadata ────────────────────────────────────
    data["pipeline"] = {
        "pipeline_id": f"CITY-DEMO-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "mode": "demo",
        "structure_type": "Reinforced Concrete Bridge",
        "location": "Interstate 405, Los Angeles, CA",
        "gps": {"lat": 34.0522, "lon": -118.2437},
        "inspection_date": datetime.now().isoformat(),
        "video_source": "DJI Mavic 3 Enterprise (4K, 30fps)",
        "video_duration_s": 127.5,
        "frames_extracted": 42,
        "frames_after_filter": 35,
        "processing_time_s": 18.7,
        "amd_hardware": {
            "cpu": "AMD Ryzen AI 9 HX 370",
            "npu": "AMD XDNA2 (50 TOPS)",
            "igpu": "AMD Radeon 890M",
            "ram": "64GB LPDDR5x",
        },
    }

    # ── 2. Defect Detections ────────────────────────────────────
    defect_types = [
        {"type": "crack", "subtypes": ["longitudinal", "transverse", "alligator", "hairline"]},
        {"type": "spalling", "subtypes": ["surface", "deep", "delamination"]},
        {"type": "corrosion", "subtypes": ["surface_rust", "section_loss", "staining"]},
        {"type": "exposed_rebar", "subtypes": ["partial", "full_exposure"]},
        {"type": "efflorescence", "subtypes": ["white_deposit", "stalactite"]},
        {"type": "settlement", "subtypes": ["differential", "uniform"]},
    ]

    all_defects = []
    for i in range(23):  # 23 defects total
        dt = random.choice(defect_types)
        severity = round(random.triangular(2.0, 10.0, 6.5), 1)
        defect = {
            "id": f"DEF-{i+1:03d}",
            "defect_type": dt["type"],
            "subtype": random.choice(dt["subtypes"]),
            "severity": severity,
            "confidence": round(random.uniform(0.72, 0.98), 2),
            "bbox": [
                random.randint(50, 400),
                random.randint(50, 300),
                random.randint(450, 800),
                random.randint(350, 600),
            ],
            "frame_idx": random.randint(0, 34),
            "detection_method": random.choice(["yolov8", "rt-detr", "color_analysis"]),
            "zone": random.choice(["deck", "pier_1", "pier_2", "abutment_north", "abutment_south", "bearing"]),
            "location_description": random.choice([
                "East face of Pier 1, 2.3m from base",
                "Deck underside, midspan section",
                "North abutment bearing pad area",
                "Pier 2 column, west face at 4.1m height",
                "Deck surface, lane 2 near expansion joint",
                "South abutment wingwall, lower section",
            ]),
            "priority": "critical" if severity >= 8.0 else "high" if severity >= 6.0 else "medium" if severity >= 4.0 else "low",
        }
        all_defects.append(defect)

    data["defects"] = all_defects

    # ── 3. Structural Elements ──────────────────────────────────
    data["structural_elements"] = [
        {"type": "pier", "count": 4, "material": "reinforced_concrete", "condition": "fair"},
        {"type": "deck", "count": 1, "material": "prestressed_concrete", "condition": "poor"},
        {"type": "abutment", "count": 2, "material": "reinforced_concrete", "condition": "good"},
        {"type": "bearing", "count": 8, "material": "elastomeric", "condition": "fair"},
        {"type": "expansion_joint", "count": 2, "material": "steel_finger", "condition": "poor"},
        {"type": "railing", "count": 2, "material": "steel", "condition": "good"},
        {"type": "drainage", "count": 6, "material": "pvc", "condition": "fair"},
    ]

    # ── 4. Digital Twin ─────────────────────────────────────────
    critical = [d for d in all_defects if d["priority"] == "critical"]
    high = [d for d in all_defects if d["priority"] == "high"]
    medium = [d for d in all_defects if d["priority"] == "medium"]
    low = [d for d in all_defects if d["priority"] == "low"]

    health_score = max(15, 100 - len(critical) * 6 - len(high) * 3 - len(medium) * 1.5 - len(low) * 0.5)
    health_score = round(health_score, 1)

    grade_map = [(90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F")]
    grade = next(g for threshold, g in grade_map if health_score >= threshold)

    status_map = [
        (90, "Excellent"), (75, "Good"), (60, "Fair"),
        (40, "Poor"), (0, "Critical"),
    ]
    status = next(s for threshold, s in status_map if health_score >= threshold)

    data["twin"] = {
        "twin_id": f"TWIN-BRIDGE-{datetime.now().strftime('%Y%m%d')}",
        "structure_type": "Reinforced Concrete Bridge",
        "health_index": {
            "score": health_score,
            "grade": grade,
            "status": status,
            "defect_count": len(all_defects),
            "critical_count": len(critical),
            "high_count": len(high),
            "medium_count": len(medium),
            "low_count": len(low),
        },
        "defect_analysis": {
            "total_defects": len(all_defects),
            "critical_defects": critical,
            "high_defects": high,
            "medium_defects": medium,
            "low_defects": low,
            "by_type": {},
        },
        "zone_analysis": {
            "deck": {"health": 52, "defects": 8, "status": "Poor"},
            "pier_1": {"health": 61, "defects": 5, "status": "Fair"},
            "pier_2": {"health": 68, "defects": 3, "status": "Fair"},
            "abutment_north": {"health": 78, "defects": 3, "status": "Good"},
            "abutment_south": {"health": 82, "defects": 2, "status": "Good"},
            "bearing": {"health": 55, "defects": 2, "status": "Poor"},
        },
        "point_cloud_stats": {
            "total_points": 157_432,
            "labeled_points": 89_621,
            "defect_points": 12_847,
            "source": "COLMAP SfM + Depth Anything v2",
        },
    }

    # Defects by type
    for d in all_defects:
        dtype = d["defect_type"]
        if dtype not in data["twin"]["defect_analysis"]["by_type"]:
            data["twin"]["defect_analysis"]["by_type"][dtype] = {"count": 0, "avg_severity": 0, "severities": []}
        data["twin"]["defect_analysis"]["by_type"][dtype]["count"] += 1
        data["twin"]["defect_analysis"]["by_type"][dtype]["severities"].append(d["severity"])

    for dtype, info in data["twin"]["defect_analysis"]["by_type"].items():
        info["avg_severity"] = round(sum(info["severities"]) / len(info["severities"]), 1)
        del info["severities"]

    # ── 5. Agent Results ────────────────────────────────────────
    data["agents"] = {
        "inspector": {
            "agent": "Inspector Agent",
            "model": "AMD GAIA (Ryzen AI NPU)",
            "classified_defects": [
                {
                    "id": d["id"],
                    "type": d["defect_type"],
                    "subtype": d["subtype"],
                    "severity_score": d["severity"],
                    "confidence": d["confidence"],
                    "zone": d["zone"],
                    "classification": d["priority"],
                    "recommendation": {
                        "critical": "Immediate structural repair required. Close affected lane/area.",
                        "high": "Schedule repair within 30 days. Monitor weekly.",
                        "medium": "Include in next maintenance cycle (90 days).",
                        "low": "Document and monitor during routine inspections.",
                    }[d["priority"]],
                }
                for d in all_defects[:10]  # Top 10 for readability
            ],
            "summary": f"Identified {len(all_defects)} defects: {len(critical)} critical, "
                       f"{len(high)} high, {len(medium)} medium, {len(low)} low severity.",
        },
        "compliance": {
            "agent": "Compliance Agent",
            "model": "AMD GAIA + RAG (FAISS + ACI/ASCE/IBC)",
            "overall_compliance": "Non-Compliant" if len(critical) > 2 else "Partial",
            "total_violations": len(critical) + len(high),
            "violations": [
                {
                    "code": "ACI 318-19 §18.12.3",
                    "description": "Maximum allowable crack width exceeded (0.4mm limit for exposure class F2)",
                    "defect_id": "DEF-001",
                    "severity": "critical",
                },
                {
                    "code": "ASCE 7-22 §2.3.1",
                    "description": "Structural member cross-section loss exceeds 10% due to corrosion",
                    "defect_id": "DEF-003",
                    "severity": "critical",
                },
                {
                    "code": "IBC 2024 §1704.6",
                    "description": "Spalling depth exceeds cover thickness, exposing reinforcement",
                    "defect_id": "DEF-002",
                    "severity": "high",
                },
                {
                    "code": "ACI 318-19 §20.6.1",
                    "description": "Concrete cover insufficient after spalling (min 50mm for bridge piers)",
                    "defect_id": "DEF-004",
                    "severity": "high",
                },
                {
                    "code": "ASCE 7-22 §12.1.1",
                    "description": "Load path continuity compromised by deep crack in primary member",
                    "defect_id": "DEF-005",
                    "severity": "high",
                },
            ],
            "compliant_elements": ["railing", "drainage", "signage"],
        },
        "safety": {
            "agent": "Safety Agent",
            "model": "AMD GAIA (Risk Assessment)",
            "risk_score": min(100, 25 + len(critical) * 15 + len(high) * 5),
            "risk_level": "High" if len(critical) > 2 else "Moderate",
            "risk_factors": [
                {"factor": "Critical structural defects", "weight": 0.35, "score": 85},
                {"factor": "Load-bearing member compromise", "weight": 0.25, "score": 72},
                {"factor": "Environmental exposure risk", "weight": 0.15, "score": 60},
                {"factor": "Age-related deterioration", "weight": 0.15, "score": 55},
                {"factor": "Maintenance history", "weight": 0.10, "score": 40},
            ],
            "immediate_actions": [
                "Close affected lane on bridge deck (DEF-001, DEF-002 area)",
                "Install temporary shoring under Pier 1 pending repair",
                "Schedule emergency concrete repair for exposed rebar locations",
            ],
            "monitoring_recommendations": [
                "Install crack width gauges on all critical cracks",
                "Monthly visual inspection of Pier 1 and deck underside",
                "Deploy vibration sensors on bearing pads",
            ],
        },
        "report": (
            "# CityMind Structural Inspection Report\n\n"
            f"**Structure:** Reinforced Concrete Bridge — Interstate 405, Los Angeles, CA\n"
            f"**Inspection Date:** {datetime.now().strftime('%B %d, %Y')}\n"
            f"**Inspector:** CityMind AI (v1.0) — AMD Ryzen AI NPU\n\n"
            "---\n\n"
            "## Executive Summary\n\n"
            f"This automated inspection identified **{len(all_defects)} structural defects** across 6 zones. "
            f"The overall Structural Health Index is **{health_score}/100 (Grade {grade}: {status})**.\n\n"
            f"**{len(critical)} critical defects** require immediate attention, including exposed rebar on Pier 1 "
            "and deep spalling on the deck underside. The bridge is **non-compliant** with ACI 318-19 crack width "
            "limits and ASCE 7-22 load path requirements.\n\n"
            f"**Risk Score: {min(100, 25 + len(critical) * 15 + len(high) * 5)}/100** — "
            "immediate lane closure and emergency repair recommended.\n\n"
            "## Recommended Actions\n\n"
            "1. **Immediate (0-7 days):** Close affected lane, install temporary shoring\n"
            "2. **Short-term (7-30 days):** Emergency concrete repair, rebar treatment\n"
            "3. **Medium-term (30-90 days):** Full structural assessment, crack injection\n"
            "4. **Long-term (90+ days):** Comprehensive rehabilitation design\n\n"
            "---\n"
            "*Report generated by CityMind AI — Powered by AMD Ryzen AI*\n"
        ),
    }

    # ── 6. Performance Metrics ──────────────────────────────────
    data["performance"] = {
        "layer1_ingestion_s": 1.23,
        "layer2_perception_s": 4.56,
        "layer3_reconstruction_s": 8.91,
        "layer4_twin_s": 0.34,
        "layer5_agents_s": 2.78,
        "layer6_report_s": 0.85,
        "total_pipeline_s": 18.67,
        "npu_speedup": {
            "perception_cpu_s": 4.56,
            "perception_npu_s": 1.30,
            "speedup": "3.5x",
        },
        "memory_peak_mb": 2847,
        "gpu_utilization_percent": 78,
    }

    # ── 7. Historical Scans (for temporal tracking) ─────────────
    base_health = health_score + 15  # It was better before
    data["scan_history"] = []
    for i in range(6):
        scan_date = datetime.now() - timedelta(days=30 * (5 - i))
        degradation = i * 3 + random.uniform(-1, 1)
        scan_health = round(max(0, base_health - degradation), 1)
        data["scan_history"].append({
            "date": scan_date.strftime("%Y-%m-%d"),
            "health_score": scan_health,
            "defect_count": max(0, len(all_defects) - (5 - i) * 3 + random.randint(-1, 2)),
            "critical_count": max(0, len(critical) - (5 - i) + random.randint(0, 1)),
        })

    # ── Save all data ───────────────────────────────────────────
    for key, value in data.items():
        out_path = OUTPUT_DIR / f"{key}.json"
        with open(out_path, "w") as f:
            json.dump(value, f, indent=2, default=str)
        logger.info(f"  Saved: {out_path}")

    # Save combined
    combined_path = OUTPUT_DIR / "demo_data_complete.json"
    with open(combined_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"  Combined data: {combined_path}")

    # ── Generate synthetic 3D point cloud ───────────────────────
    from citymind.reconstruction.point_cloud_utils import PointCloudUtils

    pc = PointCloudUtils.generate_synthetic_building(num_points=10000)
    ply_path = str(OUTPUT_DIR / "bridge_digital_twin.ply")
    PointCloudUtils.write_ply(ply_path, pc["points"], pc["colors"])
    logger.info(f"  Point cloud: {ply_path}")

    # ── Generate sample defect overlay images ───────────────────
    try:
        import cv2

        img_dir = OUTPUT_DIR / "defect_overlays"
        img_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            # Create a textured gray image (simulating concrete)
            img = np.random.randint(140, 180, (480, 640, 3), dtype=np.uint8)
            # Add noise to simulate concrete texture
            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Draw some "structural" features
            cv2.rectangle(img, (50, 50), (590, 430), (160, 160, 160), 3)
            cv2.line(img, (320, 50), (320, 430), (150, 150, 150), 2)

            # Draw defect bounding boxes
            colors = {
                "critical": (0, 0, 255),    # Red
                "high": (0, 128, 255),       # Orange
                "medium": (0, 255, 255),     # Yellow
                "low": (0, 255, 0),          # Green
            }

            frame_defects = [d for d in all_defects if d["frame_idx"] % 5 == i][:4]
            for j, defect in enumerate(frame_defects):
                x1, y1 = defect["bbox"][0], defect["bbox"][1]
                x2, y2 = min(defect["bbox"][2], 630), min(defect["bbox"][3], 470)
                color = colors.get(defect["priority"], (255, 255, 255))

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{defect['defect_type']} ({defect['severity']:.1f})"
                cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add frame info
            cv2.putText(img, f"CityMind | Frame {i*7+1}/35", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, f"AMD Ryzen AI NPU | YOLOv8", (10, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imwrite(str(img_dir / f"frame_{i:04d}_defects.png"), img)

        logger.info(f"  Defect overlays: {img_dir}")
    except Exception as e:
        logger.warning(f"Could not generate overlay images: {e}")

    # ── Generate PDF report ─────────────────────────────────────
    try:
        from citymind.visualization.report_pdf import ReportGenerator

        gen = ReportGenerator()
        pdf_path = str(OUTPUT_DIR / "inspection_report.pdf")
        agent_for_pdf = {}
        for k, v in data["agents"].items():
            if isinstance(v, dict):
                agent_for_pdf[k] = json.dumps(v)
            else:
                agent_for_pdf[k] = str(v)
        gen.generate_pdf(data["twin"], agent_for_pdf, pdf_path)
        logger.info(f"  PDF report: {pdf_path}")
    except Exception as e:
        logger.warning(f"PDF generation: {e}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🏗️  CityMind — Rich Demo Data Generated!")
    print("=" * 70)
    print(f"  📁 Output directory: {OUTPUT_DIR}")
    print(f"  📊 Structure: Reinforced Concrete Bridge")
    print(f"  ❤️  Health Score: {health_score}/100 (Grade {grade})")
    print(f"  ⚠️  Defects: {len(all_defects)} ({len(critical)} critical)")
    print(f"  🛡️  Risk Score: {data['agents']['safety']['risk_score']}/100")
    print(f"\n  Files generated:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {'📄' if f.suffix in ['.json', '.md'] else '📊' if f.suffix == '.pdf' else '🖼️'} {f.name} ({size_kb:.0f} KB)")
    print(f"\n  To view in dashboard:")
    print(f"  $ streamlit run citymind/visualization/app.py")
    print("=" * 70)

    return data


if __name__ == "__main__":
    generate_rich_demo_data()
