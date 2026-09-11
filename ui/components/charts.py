"""Reusable Plotly chart helpers for the dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

JOINT_COLOR = {
    "R_KNEE": "#ef4444",
    "L_KNEE": "#3b82f6",
    "R_HIP": "#f97316",
    "L_HIP": "#8b5cf6",
    "R_ANKLE": "#10b981",
    "L_ANKLE": "#06b6d4",
}

VEL_COLOR = {
    "R_KNEE_vel_deg_s": "#ef4444",
    "L_KNEE_vel_deg_s": "#3b82f6",
    "R_HIP_vel_deg_s": "#f97316",
    "L_HIP_vel_deg_s": "#8b5cf6",
}

FOOT_COLOR = {
    "R_FOOT_speed_norm": "#ef4444",
    "L_FOOT_speed_norm": "#3b82f6",
}

_DARK = dict(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="#fafafa", size=14),
    xaxis=dict(gridcolor="#333", title_font=dict(size=15), tickfont=dict(size=12)),
    yaxis=dict(gridcolor="#333", title_font=dict(size=15), tickfont=dict(size=12)),
)


def _readable_layout(
    fig: go.Figure,
    *,
    title: str,
    x_title: str,
    y_title: str,
    height: int = 360,
    note: str | None = None,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#fafafa", size=18)),
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        margin=dict(l=70, r=30, t=70 if note else 55, b=65),
        legend=dict(orientation="h", y=-0.28, font=dict(size=13)),
        **_DARK,
    )
    if note:
        fig.add_annotation(
            text=note,
            xref="paper",
            yref="paper",
            x=0,
            y=1.12,
            showarrow=False,
            align="left",
            font=dict(color="#cbd5e1", size=12),
        )
    return fig

def event_shapes(events: list[dict]) -> list[dict]:
    return [
        dict(
            type="rect", xref="x", yref="paper",
            x0=float(ev.get("start_time_sec", 0)),
            x1=float(ev.get("end_time_sec", 0)),
            y0=0, y1=1,
            fillcolor="rgba(255,220,0,0.12)", line_width=0,
        )
        for ev in events
    ]

def plot_time_series(
    df: pd.DataFrame,
    columns: list[str],
    colors: dict,
    events: list[dict],
    title: str,
    y_label: str,
) -> go.Figure:
    fig = go.Figure()
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time_sec"], y=df[col], name=col,
                line=dict(color=colors.get(col, "#aaa"), width=1.5), mode="lines",
            ))
    fig.update_layout(
        title=title, xaxis_title="Zaman (sn)", yaxis_title=y_label,
        height=300, margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.25),
        shapes=event_shapes(events),
        **_DARK,
    )
    return fig

def radar_chart(pre_vals: list[float], post_vals: list[float], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pre_vals + [pre_vals[0]], theta=labels + [labels[0]],
        name="Pre", line=dict(color="#3b82f6", width=2),
        fill="toself", fillcolor="rgba(59,130,246,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=post_vals + [post_vals[0]], theta=labels + [labels[0]],
        name="Post", line=dict(color="#ef4444", width=2),
        fill="toself", fillcolor="rgba(239,68,68,0.15)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, gridcolor="#333"), bgcolor="#0e1117"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), height=420,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

def bar_comparison(
    pre_vals: list[float],
    post_vals: list[float],
    labels: list[str],
    pct_changes: list[float | None],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Pre", x=labels, y=pre_vals, marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="Post", x=labels, y=post_vals, marker_color="#ef4444"))
    fig.update_layout(
        barmode="group", height=350,
        margin=dict(l=40, r=20, t=30, b=80),
        xaxis_tickangle=-35,
        legend=dict(orientation="h", y=-0.4),
        **_DARK,
    )
    return fig

def readable_bar_comparison(
    labels: list[str],
    pre_vals: list[float],
    post_vals: list[float],
    title: str,
    y_title: str,
    note: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Pre-antrenman",
        x=labels,
        y=pre_vals,
        marker_color="#2563eb",
        text=[f"{v:.1f}" for v in pre_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Post-antrenman",
        x=labels,
        y=post_vals,
        marker_color="#dc2626",
        text=[f"{v:.1f}" for v in post_vals],
        textposition="outside",
    ))
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-20)
    return _readable_layout(fig, title=title, x_title="Metrik", y_title=y_title, height=390, note=note)
