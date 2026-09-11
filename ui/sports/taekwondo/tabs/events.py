from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
from src.sports.taekwondo.fatigue import FATIGUE_METRICS
from ui.sports.taekwondo.presentation import readable_kick_rows, render_metric_help
from ui.sports.taekwondo.charts import per_kick_trend
from ui.sports.taekwondo.components import kick_video_section
from ui.components.comparison import comparison_panel


def _render_per_kick_tab(
    pre_events: list[dict],
    post_events: list[dict],
    tmp: Path | None,
) -> None:
    st.caption("Her tekme ayrı satırda gösterilir. Bu bölüm hangi tekmenin analizi güçlendirdiğini veya zayıflattığını görmek içindir.")
    pk_col = st.selectbox(
        "Grafikte gösterilecek metrik",
        [k for k in FATIGUE_METRICS if any(e.get(k) is not None for e in pre_events + post_events)],
        format_func=lambda k: FATIGUE_METRICS[k][0],
        key="dv_pk_col",
        help="Seçilen metrik her tekme için pre ve post barlarıyla çizilir. Tekme numarası sıra numarasıdır; aynı numara bire bir aynı teknik tekrarını garanti etmez.",
    )
    if pk_col:
        st.plotly_chart(
            per_kick_trend(pre_events, post_events, pk_col, FATIGUE_METRICS[pk_col][0]),
            use_container_width=True,
        )
        render_metric_help([pk_col], "Seçilen metrik ne anlama geliyor?")

    pre_tbl = pd.DataFrame(readable_kick_rows(pre_events))
    post_tbl = pd.DataFrame(readable_kick_rows(post_events))
    st.markdown("#### Pre/Post Tekme Tablosu")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**Pre-antrenman tekmeleri**")
        if not pre_tbl.empty:
            st.dataframe(pre_tbl, use_container_width=True, hide_index=True)
    with tc2:
        st.markdown("**Post-antrenman tekmeleri**")
        if not post_tbl.empty:
            st.dataframe(post_tbl, use_container_width=True, hide_index=True)

    st.divider()
    kc1, kc2 = st.columns(2)
    with kc1:
        kick_video_section(
            pre_events,
            str(tmp / "pre_annotated.mp4") if tmp else "",
            "Pre Tekmeleri", tmp,
        )
    with kc2:
        kick_video_section(
            post_events,
            str(tmp / "post_annotated.mp4") if tmp else "",
            "Post Tekmeleri", tmp,
        )
