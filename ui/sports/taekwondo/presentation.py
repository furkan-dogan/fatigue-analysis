"""Taekwondo help/quality renderers and compatibility exports for prepared data."""

from __future__ import annotations

from typing import Iterable
import streamlit as st
from src.sports.taekwondo.fatigue import FATIGUE_WEIGHTS
from src.sports.taekwondo.reporting.quality import quality_notices
from ui.components.quality import quality_panel
from ui.components.models import QualityNotice
from src.sports.taekwondo.reporting.catalog import EMG_CH1_GROUP, EMG_CH1_MUSCLE, EMG_CH2_GROUP, EMG_CH2_MUSCLE, EMG_MUSCLE_GROUP, EMG_MUSCLE_NAME, EMG_PLACEMENT, METRICS, MetricInfo, PRIMARY_METRICS
from src.sports.taekwondo.reporting.formatting import change_label, fmt_value, mean_for, metric_info, readable_emg_frequency, readable_emg_rms, readable_metric_value, status_badge
from src.sports.taekwondo.reporting.summaries import comparison_rows, emg_device_rows, emg_summary_text, movement_summary_rows, readable_kick_rows, session_summary_rows
from src.sports.taekwondo.reporting.findings import action_recommendations, analysis_paragraph, readable_findings, top_findings

def render_metric_help(keys: Iterable[str], title: str = "Metrikler nasıl okunmalı?") -> None:
    with st.expander(title, expanded=False):
        for key in keys:
            info = metric_info(key)
            weight = FATIGUE_WEIGHTS.get(key)
            weight_text = f" Yorgunluk indeksindeki ağırlığı: {weight:.0%}." if weight else ""
            st.markdown(f"**{info.label}**")
            st.markdown(f"- Ne ölçer: {info.plain}")
            st.markdown(f"- Nasıl hesaplandı: {info.calculation}")
            st.markdown(f"- Nasıl yorumlanır: {info.interpretation}{weight_text}")


def render_data_quality(pre_events: list[dict], post_events: list[dict]) -> None:
    quality_panel([QualityNotice(message, level) for level, message in quality_notices(pre_events, post_events)])
