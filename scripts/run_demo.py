#!/usr/bin/env python3
"""
CityMind Quick Demo Script
Runs the synthetic demo pipeline and prints results.
No GPU/NPU or video files required.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    print("=" * 70)
    print("  🏗️  CityMind — Quick Demo")
    print("  AI-Powered Infrastructure Digital Twin")
    print("  Powered by AMD Ryzen AI | GAIA Framework")
    print("=" * 70)
    
    # Run synthetic pipeline
    from citymind.pipeline import CityMindPipeline
    
    output_dir = str(Path(__file__).parent.parent / "output" / "demo")
    
    pipeline = CityMindPipeline(
        output_dir=output_dir,
        llm_provider="deterministic",
        max_frames=10,
    )
    
    results = pipeline._run_synthetic("Reinforced Concrete Building")
    
    # Print summary
    twin = results.get("twin", {})
    health = twin.get("health_index", {})
    agents = results.get("agents", {})
    
    print("\n" + "=" * 70)
    print("  📊 Results Summary")
    print("=" * 70)
    
    print(f"\n  🏗️  Twin ID: {twin.get('twin_id', 'N/A')}")
    print(f"  ❤️  Health Index: {health.get('score', 'N/A')}/100 (Grade: {health.get('grade', 'N/A')})")
    print(f"  📋 Status: {health.get('status', 'N/A')}")
    print(f"  ⚠️  Defects: {health.get('defect_count', 0)} total, {health.get('critical_count', 0)} critical")
    
    # Safety agent results
    safety = agents.get("safety", {})
    if isinstance(safety, str):
        import json as _json
        try:
            safety = _json.loads(safety)
        except Exception:
            safety = {}
    risk_score = safety.get("risk_score", "N/A")
    risk_level = safety.get("risk_level", "N/A")
    print(f"  🛡️  Risk Score: {risk_score}/100 ({risk_level})")
    
    # Compliance agent results
    compliance = agents.get("compliance", {})
    if isinstance(compliance, str):
        try:
            compliance = _json.loads(compliance)
        except Exception:
            compliance = {}
    print(f"  📋 Compliance: {compliance.get('overall_compliance', 'N/A')} ({compliance.get('total_violations', 0)} violations)")
    
    print(f"\n  📁 Output: {output_dir}")
    print(f"  📄 Report: {results.get('report_path', 'N/A')}")
    
    print("\n  ✅ Demo complete!")
    print("\n  To launch the interactive dashboard:")
    print("  $ streamlit run citymind/visualization/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
