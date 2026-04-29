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
    font=dict(color="#fafafa"),
    xaxis=dict(gridcolor="#333"),
    yaxis=dict(gridcolor="#333"),
)


def kick_event_shapes(events: list[dict]) -> list[dict]:
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
        shapes=kick_event_shapes(events),
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


def gauge(value: float, title: str) -> go.Figure:
    color = "#22c55e" if value < 33 else ("#f59e0b" if value < 66 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        title={"text": title, "font": {"color": "#fafafa", "size": 14}},
        number={"suffix": "/100", "font": {"color": "#fafafa", "size": 22}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#555", tickfont=dict(color="#aaa")),
            bar=dict(color=color),
            bgcolor="#1e2030",
            steps=[
                dict(range=[0, 33], color="#1a2a1a"),
                dict(range=[33, 66], color="#2a2a1a"),
                dict(range=[66, 100], color="#2a1a1a"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    return fig


def overlay_chart(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    column: str,
    label: str,
    pre_events: list[dict],
    post_events: list[dict],
) -> go.Figure:
    fig = go.Figure()
    for ev in pre_events:
        fig.add_vrect(x0=float(ev["start_time_sec"]), x1=float(ev["end_time_sec"]),
                      fillcolor="rgba(59,130,246,0.08)", line_width=0)
    for ev in post_events:
        fig.add_vrect(x0=float(ev["start_time_sec"]), x1=float(ev["end_time_sec"]),
                      fillcolor="rgba(239,68,68,0.08)", line_width=0)
    if column in pre_df.columns:
        fig.add_trace(go.Scatter(
            x=pre_df["time_sec"], y=pre_df[column],
            name=f"Pre — {label}", mode="lines",
            line=dict(color="#3b82f6", width=1.5, dash="dot"),
        ))
    if column in post_df.columns:
        fig.add_trace(go.Scatter(
            x=post_df["time_sec"], y=post_df[column],
            name=f"Post — {label}", mode="lines",
            line=dict(color="#ef4444", width=1.5),
        ))
    is_velocity = "vel" in column or "speed" in column
    fig.update_layout(
        height=240,
        xaxis_title="Zaman (sn)",
        yaxis_title="Hız (°/s)" if is_velocity else "Açı (°)",
        margin=dict(l=50, r=20, t=30, b=35),
        legend=dict(orientation="h", y=-0.35, font=dict(size=11)),
        title=dict(text=label, font=dict(color="#fafafa", size=13)),
        **_DARK,
    )
    return fig


def per_kick_trend(
    pre_events: list[dict],
    post_events: list[dict],
    col: str,
    label: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Pre",
        x=[f"T{int(e['kick_id'])}" for e in pre_events],
        y=[float(e[col]) if e.get(col) is not None else 0 for e in pre_events],
        marker_color="#3b82f6",
    ))
    fig.add_trace(go.Bar(
        name="Post",
        x=[f"T{int(e['kick_id'])}" for e in post_events],
        y=[float(e[col]) if e.get(col) is not None else 0 for e in post_events],
        marker_color="#ef4444",
    ))
    fig.update_layout(
        barmode="group", height=240,
        title=dict(text=label, font=dict(color="#fafafa", size=13)),
        margin=dict(l=40, r=10, t=35, b=35),
        legend=dict(orientation="h", y=-0.35, font=dict(size=11)),
        **_DARK,
    )
    return fig
