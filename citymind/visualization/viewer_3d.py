"""
Layer 6: 3D Point Cloud Viewer
Plotly-based interactive 3D visualization for the digital twin.

AMD Tech: Ryzen AI Max+ iGPU for GPU-accelerated rendering via Vulkan
"""

import numpy as np
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


def render_point_cloud_plotly(
    points: np.ndarray,
    colors: np.ndarray = None,
    labels: np.ndarray = None,
    defect_points: np.ndarray = None,
    title: str = "CityMind 3D Digital Twin",
    point_size: int = 2,
    max_points: int = 5000,
    height: int = 600,
):
    """
    Render an interactive 3D point cloud using Plotly.
    
    Args:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (0-1)
        labels: N array of semantic labels
        defect_points: Mx3 array of defect locations to highlight
        title: Plot title
        point_size: Marker size
        max_points: Maximum points to display (downsample if exceeded)
        height: Plot height in pixels
    
    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go
    
    # Downsample for performance
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
        if labels is not None:
            labels = labels[indices]
    
    traces = []
    
    # Main point cloud
    if colors is not None:
        color_strings = [
            f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
            for c in colors
        ]
    else:
        # Default: color by height
        y_vals = points[:, 1]
        color_strings = y_vals
    
    # Hover text with coordinates
    hover_text = [
        f"X: {p[0]:.2f}m<br>Y: {p[1]:.2f}m<br>Z: {p[2]:.2f}m"
        for p in points
    ]
    
    if labels is not None:
        label_names = {
            0: "Unknown", 1: "Wall", 2: "Column", 3: "Beam",
            4: "Slab", 5: "Crack", 6: "Spalling", 7: "Corrosion",
        }
        hover_text = [
            f"{label_names.get(int(l), 'Unknown')}<br>{h}"
            for l, h in zip(labels, hover_text)
        ]
    
    traces.append(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 2],  # Swap Y/Z for architectural convention
        z=points[:, 1],
        mode='markers',
        marker=dict(
            size=point_size,
            color=color_strings,
            opacity=0.8,
            colorscale='Viridis' if colors is None else None,
        ),
        text=hover_text,
        hoverinfo='text',
        name='Structure',
    ))
    
    # Highlight defect points
    if defect_points is not None and len(defect_points) > 0:
        traces.append(go.Scatter3d(
            x=defect_points[:, 0],
            y=defect_points[:, 2],
            z=defect_points[:, 1],
            mode='markers',
            marker=dict(
                size=point_size * 3,
                color='red',
                symbol='diamond',
                opacity=1.0,
                line=dict(width=1, color='white'),
            ),
            name='🔴 Defects',
        ))
    
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='white'),
        ),
        scene=dict(
            xaxis=dict(
                title="X (m)",
                gridcolor='rgb(50,50,70)',
                zerolinecolor='rgb(70,70,90)',
            ),
            yaxis=dict(
                title="Z (m)",
                gridcolor='rgb(50,50,70)',
                zerolinecolor='rgb(70,70,90)',
            ),
            zaxis=dict(
                title="Height (m)",
                gridcolor='rgb(50,50,70)',
                zerolinecolor='rgb(70,70,90)',
            ),
            bgcolor='rgb(15,15,25)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
            ),
        ),
        paper_bgcolor='rgb(15,15,25)',
        font=dict(color='rgb(200,200,200)'),
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(0,0,0,0.5)',
        ),
    )
    
    return fig


def render_semantic_twin(
    points: np.ndarray,
    labels: np.ndarray,
    defects: List[Dict] = None,
    title: str = "Semantic Digital Twin",
    height: int = 600,
):
    """
    Render point cloud colored by semantic labels + defect overlay.
    """
    import plotly.graph_objects as go
    
    label_config = {
        0: ("Unknown", "rgb(128,128,128)"),
        1: ("Wall", "rgb(50,50,200)"),
        2: ("Column", "rgb(50,200,50)"),
        3: ("Beam", "rgb(200,200,50)"),
        4: ("Slab", "rgb(150,150,150)"),
        5: ("Crack", "rgb(255,0,0)"),
        6: ("Spalling", "rgb(255,100,0)"),
        7: ("Corrosion", "rgb(200,0,200)"),
    }
    
    traces = []
    
    # One trace per label for legend
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        name, color = label_config.get(int(lbl), ("Unknown", "rgb(128,128,128)"))
        pts = points[mask]
        
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 2], z=pts[:, 1],
            mode='markers',
            marker=dict(size=2, color=color, opacity=0.7),
            name=name,
        ))
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            bgcolor='rgb(15,15,25)',
            xaxis=dict(title="X (m)", gridcolor='rgb(50,50,70)'),
            yaxis=dict(title="Z (m)", gridcolor='rgb(50,50,70)'),
            zaxis=dict(title="Height (m)", gridcolor='rgb(50,50,70)'),
        ),
        paper_bgcolor='rgb(15,15,25)',
        font=dict(color='rgb(200,200,200)'),
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    return fig


def render_depth_heatmap(
    depth_map: np.ndarray,
    title: str = "Depth Estimation",
    height: int = 400,
):
    """Render a 2D depth map as a heatmap."""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=go.Heatmap(
        z=depth_map,
        colorscale='Viridis',
        reversescale=True,
        colorbar=dict(title="Depth (m)", tickfont=dict(color='white')),
    ))
    
    fig.update_layout(
        title=title,
        paper_bgcolor='rgb(15,15,25)',
        font=dict(color='rgb(200,200,200)'),
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x'),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig
