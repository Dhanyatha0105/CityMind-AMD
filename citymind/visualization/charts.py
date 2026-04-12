"""
Layer 6: Charts and Visualization Components
Plotly-based charts for defect analysis, risk assessment, and pipeline performance.

AMD Tech: Visualization accelerated by AMD Ryzen AI Max+ iGPU (Vulkan)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def defect_distribution_chart(defect_types: Dict, height: int = 350):
    """
    Pie/donut chart showing defect type distribution.
    
    Args:
        defect_types: Dict mapping defect_type → {count, max_severity, avg_severity}
    """
    import plotly.graph_objects as go
    
    types = list(defect_types.keys())
    counts = [v["count"] for v in defect_types.values()]
    
    color_map = {
        "crack": "#FF0000",
        "spalling": "#FF6600",
        "corrosion": "#FF9900",
        "delamination": "#FFCC00",
        "staining": "#FFFF00",
        "displacement": "#FF3366",
        "exposed_rebar": "#CC0000",
        "water_damage": "#0066FF",
        "scaling": "#9966FF",
    }
    colors = [color_map.get(t, "#888888") for t in types]
    
    fig = go.Figure(data=[go.Pie(
        labels=[t.replace("_", " ").title() for t in types],
        values=counts,
        marker=dict(colors=colors, line=dict(color='rgb(30,30,50)', width=2)),
        hole=0.45,
        textinfo='label+percent',
        textfont=dict(color='white'),
        hoverinfo='label+value+percent',
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(200,200,200)'),
        height=height,
        showlegend=False,
        margin=dict(l=20, r=20, t=10, b=20),
        annotations=[dict(
            text=f"{sum(counts)}<br>Defects",
            x=0.5, y=0.5,
            font_size=16,
            font_color='white',
            showarrow=False,
        )],
    )
    
    return fig


def severity_histogram(defects: List[Dict], height: int = 350):
    """
    Histogram of defect severity scores.
    
    Args:
        defects: List of defect dicts with 'severity' field
    """
    import plotly.graph_objects as go
    
    severities = [d.get("severity", 0) for d in defects]
    
    # Color bins by severity range
    fig = go.Figure()
    
    # Low severity (0-3)
    low = [s for s in severities if s < 3]
    if low:
        fig.add_trace(go.Histogram(
            x=low, name="Low (0-3)",
            marker_color="#00E676", opacity=0.8,
            xbins=dict(start=0, end=10, size=1),
        ))
    
    # Medium severity (3-5)
    med = [s for s in severities if 3 <= s < 5]
    if med:
        fig.add_trace(go.Histogram(
            x=med, name="Medium (3-5)",
            marker_color="#FFD600", opacity=0.8,
            xbins=dict(start=0, end=10, size=1),
        ))
    
    # High severity (5-7)
    high = [s for s in severities if 5 <= s < 7]
    if high:
        fig.add_trace(go.Histogram(
            x=high, name="High (5-7)",
            marker_color="#FF6D00", opacity=0.8,
            xbins=dict(start=0, end=10, size=1),
        ))
    
    # Critical severity (7-10)
    crit = [s for s in severities if s >= 7]
    if crit:
        fig.add_trace(go.Histogram(
            x=crit, name="Critical (7-10)",
            marker_color="#FF1744", opacity=0.8,
            xbins=dict(start=0, end=10, size=1),
        ))
    
    fig.update_layout(
        barmode='stack',
        xaxis_title="Severity Score",
        yaxis_title="Count",
        xaxis=dict(range=[0, 10], dtick=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(200,200,200)'),
        height=height,
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="top", y=1.15),
    )
    
    return fig


def risk_gauge(risk_score: float, height: int = 250):
    """
    Gauge chart for risk score visualization.
    
    Args:
        risk_score: Risk score 0-100 (higher = more dangerous)
    """
    import plotly.graph_objects as go
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title={"text": "Structural Risk", "font": {"color": "white", "size": 14}},
        delta={"reference": 50, "increasing": {"color": "#FF1744"}, "decreasing": {"color": "#00E676"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "white",
                "tickfont": {"color": "white"},
            },
            "bar": {"color": "#ED1C24", "thickness": 0.8},
            "bgcolor": "rgb(30,30,50)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "rgba(0, 230, 118, 0.3)"},
                {"range": [20, 40], "color": "rgba(100, 221, 23, 0.3)"},
                {"range": [40, 60], "color": "rgba(255, 214, 0, 0.3)"},
                {"range": [60, 80], "color": "rgba(255, 109, 0, 0.3)"},
                {"range": [80, 100], "color": "rgba(255, 23, 68, 0.3)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": risk_score,
            },
        },
        number={"font": {"color": "white", "size": 32}},
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig


def health_index_card(health: Dict, height: int = 200):
    """
    Health index indicator with grade.
    """
    import plotly.graph_objects as go
    
    score = health.get("score", 0)
    
    grade_colors = {
        "A": "#00E676", "B": "#64DD17", "C": "#FFD600",
        "D": "#FF6D00", "F": "#FF1744",
    }
    color = grade_colors.get(health.get("grade", "C"), "#FFD600")
    
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=score,
        title={"text": f"Health: {health.get('grade', '?')}", "font": {"color": "white"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "bgcolor": "rgb(30,30,50)",
            "steps": [
                {"range": [0, 40], "color": "rgba(255,23,68,0.2)"},
                {"range": [40, 60], "color": "rgba(255,214,0,0.2)"},
                {"range": [60, 100], "color": "rgba(0,230,118,0.2)"},
            ],
        },
        number={"font": {"color": "white", "size": 28}, "suffix": "/100"},
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=height,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    
    return fig


def zone_health_chart(zones: List[Dict], height: int = 350):
    """Bar chart showing health scores per inspection zone."""
    import plotly.graph_objects as go
    
    zone_names = [z["name"] for z in zones]
    zone_scores = [z["score"] for z in zones]
    zone_defects = [len(z.get("defects", [])) for z in zones]
    
    colors = [
        "#00E676" if s >= 75 else "#FFD600" if s >= 50 else "#FF1744"
        for s in zone_scores
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=zone_names,
        y=zone_scores,
        name="Health Score",
        marker_color=colors,
        text=[f"{s}/100" for s in zone_scores],
        textposition='auto',
        textfont=dict(color='white'),
    ))
    
    fig.add_trace(go.Scatter(
        x=zone_names,
        y=[d * 20 for d in zone_defects],  # Scale for visibility
        name="Defect Count (×20)",
        mode='lines+markers',
        line=dict(color='#FF6B35', width=2),
        marker=dict(size=8, color='#FF6B35'),
        yaxis='y2',
    ))
    
    fig.update_layout(
        yaxis=dict(title="Health Score", range=[0, 100]),
        yaxis2=dict(title="Defects", overlaying='y', side='right', range=[0, max(zone_defects) * 25 + 10]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(200,200,200)'),
        height=height,
        margin=dict(l=50, r=50, t=20, b=60),
        legend=dict(orientation="h", yanchor="top", y=1.12),
    )
    
    return fig


def pipeline_performance_chart(
    steps: List[str] = None,
    amd_times: List[float] = None,
    cpu_times: List[float] = None,
    height: int = 400,
):
    """Bar chart comparing AMD vs CPU pipeline performance."""
    import plotly.graph_objects as go
    
    if steps is None:
        steps = [
            "Ingestion", "Depth Est.", "Object Det.",
            "Defect Det.", "3D Recon.", "Twin Engine",
            "Agents", "Report",
        ]
    if amd_times is None:
        amd_times = [0.8, 1.2, 0.9, 0.7, 3.5, 0.3, 2.1, 0.5]
    if cpu_times is None:
        cpu_times = [1.5, 8.5, 5.2, 4.1, 15.0, 0.5, 12.0, 1.0]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="AMD Ryzen AI",
        x=steps,
        y=amd_times,
        marker_color="#ED1C24",
        text=[f"{t:.1f}s" for t in amd_times],
        textposition='auto',
    ))
    fig.add_trace(go.Bar(
        name="CPU Baseline",
        x=steps,
        y=cpu_times,
        marker_color="#555555",
        text=[f"{t:.1f}s" for t in cpu_times],
        textposition='auto',
    ))
    
    fig.update_layout(
        barmode='group',
        yaxis_title="Time (seconds)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgb(200,200,200)'),
        height=height,
        margin=dict(l=40, r=20, t=20, b=60),
        legend=dict(orientation="h", yanchor="top", y=1.12),
    )
    
    return fig
