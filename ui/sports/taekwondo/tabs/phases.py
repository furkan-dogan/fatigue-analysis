from __future__ import annotations

import pandas as pd
import streamlit as st
from ui.sports.taekwondo.presentation import comparison_rows, render_metric_help
from ui.sports.taekwondo.charts import per_kick_trend
from ui.sports.taekwondo.components import phase_bars
from ui.components.comparison import comparison_panel


def _render_phase_tab(pre_events: list[dict], post_events: list[dict]) -> None:
    st.subheader("Tekme Faz Analizi")
    st.info(
        "Her tekme üç faza ayrılır: yüklenme, uzatma ve geri çekim. "
        "Yorgunlukta özellikle uzatma hızı düşebilir ve geri çekim süresi uzayabilir; bu durum sporcunun savunmaya dönüşünü geciktirir."
    )
    render_metric_help(["extension_dur_sec", "retraction_dur_sec", "extension_peak_vel_deg_s", "retraction_peak_vel_deg_s"])

    phase_rows = comparison_rows(
        pre_events,
        post_events,
        ["extension_dur_sec", "retraction_dur_sec", "extension_peak_vel_deg_s", "retraction_peak_vel_deg_s"],
    )
    comparison_panel(phase_rows)

    fc1, fc2 = st.columns(2)
    with fc1:
        phase_bars(pre_events,  "Pre — Faz Süreleri", "#3b82f6")
    with fc2:
        phase_bars(post_events, "Post — Faz Süreleri", "#ef4444")

    st.subheader("Faz Hızları — Pre vs Post")
    for pk, plbl in [
        ("extension_peak_vel_deg_s",  "Uzatma Peak Hızı (°/s)"),
        ("retraction_peak_vel_deg_s", "Geri Çekim Peak Hızı (°/s)"),
        ("loading_peak_vel_deg_s",    "Yüklenme Peak Hızı (°/s)"),
    ]:
        if any(e.get(pk) is not None for e in pre_events + post_events):
            st.plotly_chart(per_kick_trend(pre_events, post_events, pk, plbl), use_container_width=True)

    st.subheader("Geri Çekim / Uzatma Oranı")
    st.caption("Yorgunlukla geri çekim yavaşlar → oran artar. >1.5 = belirgin yavaşlama.")
    for label, evs in [("Pre", pre_events), ("Post", post_events)]:
        ratios = []
        for ev in evs:
            ext = ev.get("extension_dur_sec")
            ret = ev.get("retraction_dur_sec")
            if ext and ret and float(ext) > 0.001:
                ratios.append(round(float(ret) / float(ext), 3))
        if ratios:
            mean_ratio = sum(ratios) / len(ratios)
            st.metric(
                f"{label} — ort. geri çekim/uzatma oranı", f"{mean_ratio:.2f}",
                delta="⚠️ yavaş geri çekim" if mean_ratio > 1.5 else "✅ normal",
                delta_color="off",
            )
