"""
Layer 6: PDF Report Generation
Generates professional PDF inspection reports using FPDF2.

AMD Tech: Report generation pipeline part of AMD GAIA agent output
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates professional PDF inspection reports.
    
    Uses FPDF2 for lightweight PDF generation without heavy dependencies.
    Markdown reports are also supported as the primary format.
    """
    
    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Sanitize text for FPDF Helvetica (Latin-1 compatible)."""
        replacements = {
            '\u2014': '--',   # em dash
            '\u2013': '-',    # en dash
            '\u2018': "'",    # left single quote
            '\u2019': "'",    # right single quote
            '\u201c': '"',    # left double quote
            '\u201d': '"',    # right double quote
            '\u2026': '...',  # ellipsis
            '\u2022': '*',    # bullet
            '\u00a7': 'S',    # section sign §
            '\u2265': '>=',   # ≥
            '\u2264': '<=',   # ≤
            '\u00b2': '2',    # ²
            '\u00b3': '3',    # ³
            '\u221a': 'sqrt', # √
            '\u03bb': 'lambda', # λ
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        # Final pass: replace any remaining non-latin-1 chars
        return text.encode('latin-1', errors='replace').decode('latin-1')
    
    def generate_pdf(
        self,
        twin_data: Dict,
        pipeline_results: Dict,
        output_path: str = None,
    ) -> str:
        """
        Generate a PDF inspection report.
        
        Args:
            twin_data: Digital twin data
            pipeline_results: Multi-agent pipeline results
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to generated PDF file
        """
        if output_path is None:
            output_path = f"citymind_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            from fpdf import FPDF
            return self._generate_fpdf(twin_data, pipeline_results, str(output_path))
        except ImportError:
            logger.warning("fpdf2 not installed, generating Markdown report instead")
            md_path = output_path.with_suffix(".md")
            self.generate_markdown(twin_data, pipeline_results, str(md_path))
            return str(md_path)
    
    def _generate_fpdf(self, twin_data: Dict, pipeline_results: Dict, output_path: str) -> str:
        """Generate PDF using FPDF2."""
        from fpdf import FPDF
        
        sanitize = self._sanitize_text
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # ── Page 1: Cover ───────────────────────────────────────
        pdf.add_page()
        
        # Header
        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(237, 28, 36)  # AMD Red
        pdf.cell(0, 20, "CityMind", new_x="LMARGIN", new_y="NEXT", align="C")
        
        pdf.set_font("Helvetica", "", 14)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, "AI-Powered Infrastructure Digital Twin", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.cell(0, 8, "Structural Inspection Report", new_x="LMARGIN", new_y="NEXT", align="C")
        
        pdf.ln(15)
        
        # Report info
        health = twin_data.get("health_index", {})
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 0, 0)
        
        info_items = [
            ("Report ID", twin_data.get("twin_id", "N/A")),
            ("Date", datetime.now().strftime("%B %d, %Y")),
            ("Structure", twin_data.get("structure_info", {}).get("type", "N/A")),
            ("Health Index", f"{health.get('score', 'N/A')}/100 (Grade: {health.get('grade', 'N/A')})"),
            ("Status", health.get("status", "N/A")),
        ]
        
        for label, value in info_items:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(50, 8, f"{label}:", new_x="RIGHT")
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, sanitize(str(value)), new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(10)
        
        # Executive summary
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(237, 28, 36)
        pdf.cell(0, 10, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        summary = f"{health.get('description', '')}. {health.get('recommendation', '')}"
        pdf.multi_cell(0, 6, self._sanitize_text(summary))
        
        # ── Page 2: Defects ─────────────────────────────────────
        pdf.add_page()
        
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(237, 28, 36)
        pdf.cell(0, 10, "Defect Analysis", new_x="LMARGIN", new_y="NEXT")
        
        defects = twin_data.get("defect_analysis", {})
        all_defects = defects.get("all_defects", defects.get("critical_defects", []))
        
        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(40, 40, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(20, 7, "ID", border=1, fill=True)
        pdf.cell(30, 7, "Type", border=1, fill=True)
        pdf.cell(25, 7, "Severity", border=1, fill=True)
        pdf.cell(25, 7, "Confidence", border=1, fill=True)
        pdf.cell(30, 7, "Method", border=1, fill=True)
        pdf.cell(60, 7, "Code Reference", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        
        # Table rows
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(0, 0, 0)
        
        for d in all_defects[:15]:
            sev = d.get("severity", 0)
            if sev >= 7:
                pdf.set_fill_color(255, 200, 200)
            elif sev >= 5:
                pdf.set_fill_color(255, 240, 200)
            else:
                pdf.set_fill_color(255, 255, 255)
            
            pdf.cell(20, 6, sanitize(str(d.get("id", ""))), border=1, fill=True)
            pdf.cell(30, 6, sanitize(str(d.get("defect_type", ""))), border=1, fill=True)
            pdf.cell(25, 6, f"{sev:.1f}/10", border=1, fill=True)
            pdf.cell(25, 6, f"{d.get('confidence', 0):.0%}", border=1, fill=True)
            pdf.cell(30, 6, sanitize(str(d.get("detection_method", ""))[:15]), border=1, fill=True)
            pdf.cell(60, 6, sanitize(str(d.get("code_reference", ""))[:35]), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        
        # ── Page 3: Compliance & Safety ─────────────────────────
        pdf.add_page()
        
        compliance = pipeline_results.get("compliance", {})
        safety = pipeline_results.get("safety", {})
        
        # Parse JSON strings if needed
        import json as _json
        if isinstance(compliance, str):
            try:
                compliance = _json.loads(compliance)
            except Exception:
                compliance = {}
        if isinstance(safety, str):
            try:
                safety = _json.loads(safety)
            except Exception:
                safety = {}
        
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(237, 28, 36)
        pdf.cell(0, 10, "Code Compliance & Risk Assessment", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "Compliance Status:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, str(compliance.get("overall_compliance", "N/A")), new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "Total Violations:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, str(compliance.get("total_violations", 0)), new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "Risk Score:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 11)
        risk_score = safety.get("risk_score", "N/A")
        pdf.cell(0, 8, f"{risk_score}/100", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "Risk Level:", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, str(safety.get("risk_level", "N/A")), new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(10)
        
        # AMD Technology section
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(237, 28, 36)
        pdf.cell(0, 10, "AMD Technology Stack", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        
        amd_techs = [
            ("AMD Instinct MI300X", "Model training (ROCm 7.2)"),
            ("AMD Ryzen AI NPU (XDNA)", "Edge inference for perception models"),
            ("Vitis AI Quantizer", "INT8 model optimization"),
            ("ONNX Runtime + Vitis AI EP", "NPU deployment runtime"),
            ("AMD GAIA Framework", "Multi-agent orchestration"),
            ("Lemonade (TurnkeyML)", "LLM optimization pipeline"),
            ("Ryzen AI Software v1.7", "Local AI runtime (16K context)"),
            ("AMD CVML Library", "Depth estimation pipeline"),
            ("Ryzen AI Max+ iGPU", "3D reconstruction acceleration"),
            ("Genesis Simulation Engine", "Digital twin physics"),
            ("Vulkan Backend", "GPU rendering optimization"),
            ("PyTorch (ROCm) + TunableOp", "Training kernel optimization"),
        ]
        
        for tech, role in amd_techs:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(55, 6, tech, border=0)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 6, role, new_x="LMARGIN", new_y="NEXT")
        
        # Footer
        pdf.ln(15)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.multi_cell(0, 5,
            "Disclaimer: This report is generated by CityMind AI and is intended as a preliminary "
            "screening tool. All findings should be verified by a licensed Professional Engineer (PE). "
            "CityMind does not replace professional engineering judgment.\n\n"
            f"Generated by CityMind v1.0 | Powered by AMD Ryzen AI | {datetime.now().year}"
        )
        
        # Save
        pdf.output(output_path)
        logger.info(f"PDF report generated: {output_path}")
        return output_path
    
    def generate_markdown(
        self,
        twin_data: Dict,
        pipeline_results: Dict,
        output_path: str = None,
    ) -> str:
        """
        Generate Markdown inspection report.
        
        Falls back to the report agent's output if available.
        """
        report = pipeline_results.get("report", {})
        md_content = report.get("report_markdown", "")
        
        if not md_content:
            md_content = self._build_markdown_report(twin_data, pipeline_results)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md_content)
            logger.info(f"Markdown report saved: {output_path}")
        
        return md_content
    
    def _build_markdown_report(self, twin_data: Dict, pipeline_results: Dict) -> str:
        """Build markdown report from data."""
        health = twin_data.get("health_index", {})
        defects = twin_data.get("defect_analysis", {})
        
        return f"""# 🏗️ CityMind Infrastructure Inspection Report

## Executive Summary
- **Report ID:** {twin_data.get('twin_id', 'N/A')}
- **Date:** {datetime.now().strftime('%B %d, %Y')}
- **Health Index:** {health.get('score', 'N/A')}/100 (Grade: {health.get('grade', 'N/A')})
- **Status:** {health.get('status', 'N/A')}

{health.get('description', '')}. {health.get('recommendation', '')}

## Defect Summary
- **Total Defects:** {defects.get('total_defects', 0)}
- **Critical:** {len(defects.get('critical_defects', []))}
- **High:** {len(defects.get('high_defects', []))}

---
*Generated by CityMind v1.0 — Powered by AMD*
"""
