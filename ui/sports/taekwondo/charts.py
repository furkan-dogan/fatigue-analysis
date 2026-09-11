"""Kick-specific chart annotations and legacy fatigue gauge."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from ui.components.charts import _DARK, _readable_layout, event_shapes

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
    fig.add_annotation(
        text="Mavi kesikli çizgi pre-antrenman, kırmızı düz çizgi post-antrenman verisini gösterir. Renkli aralıklar otomatik tespit edilen tekme zamanlarıdır.",
        xref="paper", yref="paper", x=0, y=1.18, showarrow=False,
        align="left", font=dict(color="#cbd5e1", size=12),
    )
    fig.update_layout(
        height=330,
        xaxis_title="Video zamanı (saniye)",
        yaxis_title="Açısal hız (derece/saniye)" if is_velocity else "Eklem açısı (derece)",
        margin=dict(l=70, r=25, t=70, b=60),
        legend=dict(orientation="h", y=-0.28, font=dict(size=12)),
        title=dict(text=label, font=dict(color="#fafafa", size=17)),
        **_DARK,
    )
    fig.update_traces(hovertemplate="%{x:.2f} sn<br>%{y:.2f}<extra>%{fullData.name}</extra>")
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
    fig.update_layout(barmode="group")
    _readable_layout(
        fig,
        title=f"Tekme Bazlı Karşılaştırma: {label}",
        x_title="Tekme numarası",
        y_title=label,
        height=330,
        note="Her bar bir tekmeyi temsil eder. Aynı numaradaki pre ve post tekmeler sıra bazlı karşılaştırılır; bire bir aynı hareket olmak zorunda değildir.",
    )
    return fig
