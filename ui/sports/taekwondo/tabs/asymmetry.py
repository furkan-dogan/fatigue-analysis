from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from ui.sports.taekwondo.presentation import render_metric_help
from ui.components.comparison import comparison_panel


def _render_asymmetry_tab(pre_events: list[dict], post_events: list[dict]) -> None:
    st.subheader("Bilateral Asimetri İndeksi (ASI)")
    st.info(
        "ASI sağ ve sol taraf arasındaki yüzde farkı gösterir. Pozitif değer sağ tarafın, negatif değer sol tarafın daha baskın olduğunu belirtir. "
        "Mutlak değerin 10% üzerine çıkması klinik ve sportif performans açısından dikkat gerektirir."
    )
    render_metric_help(["knee_asi", "hip_asi"], "ASI nasıl hesaplandı ve nasıl yorumlanır?")

    for asi_key, asi_lbl in [("knee_asi", "Diz ASI (%)"), ("hip_asi", "Kalça ASI (%)")]:
        if not any(e.get(asi_key) is not None for e in pre_events + post_events):
            continue
        fig = go.Figure()
        for evs, clr, lbl in [(pre_events, "#3b82f6", "Pre"), (post_events, "#ef4444", "Post")]:
            vals  = [float(e[asi_key]) if e.get(asi_key) is not None else None for e in evs]
            x_lbl = [f"T{int(e['kick_id'])}" for e in evs]
            fig.add_trace(go.Bar(name=lbl, x=x_lbl, y=vals, marker_color=clr))
        fig.add_hline(y=10,  line_dash="dash", line_color="orange", annotation_text="+10% eşik")
        fig.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="-10% eşik")
        fig.add_hline(y=0,   line_color="#555", line_width=1)
        fig.update_layout(
            barmode="group", height=280,
            title=dict(text=asi_lbl, font=dict(color="#fafafa", size=13)),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", title="%"),
            margin=dict(l=40, r=10, t=35, b=30),
            legend=dict(orientation="h", y=-0.35),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Oturum Ortalama ASI")
    ac1, ac2, ac3, ac4 = st.columns(4)
    for col_w, label, evs in [
        (ac1, "Pre Diz",    pre_events), (ac2, "Post Diz",    post_events),
        (ac3, "Pre Kalça",  pre_events), (ac4, "Post Kalça",  post_events),
    ]:
        k    = "knee_asi" if "Diz" in label else "hip_asi"
        vals = [float(e[k]) for e in evs if e.get(k) is not None]
        if vals:
            mean_asi = sum(vals) / len(vals)
            col_w.metric(label, f"{mean_asi:+.1f}%",
                         delta="⚠️" if abs(mean_asi) > 10 else "✅", delta_color="off")
