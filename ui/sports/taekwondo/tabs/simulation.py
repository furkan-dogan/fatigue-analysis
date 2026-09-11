from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from ui.components.comparison import comparison_panel


def _render_sensor_tab(
    sensor_ok: bool,
    ps: dict,
    pos: dict,
    pre_events: list[dict],
    post_events: list[dict],
    pre_freq_drop: float,
    post_freq_drop: float,
) -> None:
    st.subheader("EMG Detayları")
    st.info(
        "Pre → Post arası EMG değişimi nöromüsküler yorgunluk perspektifinden gösterilir. "
        "Rectus femoris ve biceps femoris RMS değerleri kas aktivasyonunu, median frekans düşüşü kas yorgunluğu eğilimini anlatır."
    )

    if not sensor_ok:
        st.warning("Sensör verisi üretilemedi. Analizi yeniden çalıştırın.")
        return

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("EMG Frekans — Pre sonu",  f"{ps['freq_end']:.0f} Hz",
               delta=f"{ps['freq_end'] - ps['freq_start']:.0f} Hz",  delta_color="inverse")
    sc2.metric("EMG Frekans — Post sonu", f"{pos['freq_end']:.0f} Hz",
               delta=f"{pos['freq_end'] - pos['freq_start']:.0f} Hz", delta_color="inverse")
    sc3.metric("SmO2 — Pre sonu",  f"%{ps['smo2_end']:.0f}",
               delta=f"{ps['smo2_end'] - ps['smo2_start']:.0f}%",  delta_color="inverse")
    sc4.metric("SmO2 — Post sonu", f"%{pos['smo2_end']:.0f}",
               delta=f"{pos['smo2_end'] - pos['smo2_start']:.0f}%", delta_color="inverse")
    with st.expander("EMG ve NIRS değerleri nasıl yorumlanır?", expanded=False):
        st.markdown(
            "**EMG median frekans:** Kas kasılması sırasında güç spektrumunun orta frekansıdır. "
            "Yorgunlukta motor ünite iletimi ve hızlı kasılan lif katkısı azaldığı için frekans düşebilir."
        )
        st.markdown(
            "**EMG RMS:** Kas aktivasyon genliğini gösterir. Artış daha fazla kas aktivasyonu veya kompansasyon anlamına gelebilir; tek başına yorgunluk bulgusu değildir."
        )
        st.markdown(
            "**SmO2:** Kas oksijen satürasyonudur. Düşüş, çalışan kasın oksijen tüketiminin arttığını veya oksijenlenmenin yetersiz kaldığını gösterir."
        )

    st.markdown("---")

    # EMG median frequency overlay
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=ps["t_arr"],  y=ps["freq_arr"],
        name="Pre — EMG Median Frekans",  line=dict(color="#3b82f6", width=2, dash="dot")))
    fig_freq.add_trace(go.Scatter(x=pos["t_arr"], y=pos["freq_arr"],
        name="Post — EMG Median Frekans", line=dict(color="#ef4444", width=2)))
    for ev in pre_events:
        fig_freq.add_vline(x=float(ev["peak_time_sec"]),
                           line_dash="dot", line_color="rgba(59,130,246,0.3)", line_width=1)
    for ev in post_events:
        fig_freq.add_vline(x=float(ev["peak_time_sec"]),
                           line_dash="dot", line_color="rgba(239,68,68,0.3)", line_width=1)
    fig_freq.update_layout(
        title="EMG Median Frekans — Nöromüsküler Yorgunluk Trendi",
        height=270, xaxis_title="Zaman (sn)",
        yaxis=dict(title="Frekans (Hz)", gridcolor="#333"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=50, r=20, t=40, b=45),
        legend=dict(orientation="h", y=-0.4),
    )
    st.plotly_chart(fig_freq, use_container_width=True)

    # SmO2 overlay
    fig_smo2 = go.Figure()
    fig_smo2.add_trace(go.Scatter(x=ps["t_nirs"],  y=ps["smo2_arr"],
        name="Pre — SmO2 (%)",  line=dict(color="#22c55e", width=2, dash="dot")))
    fig_smo2.add_trace(go.Scatter(x=pos["t_nirs"], y=pos["smo2_arr"],
        name="Post — SmO2 (%)", line=dict(color="#f97316", width=2)))
    for ev in pre_events:
        fig_smo2.add_vline(x=float(ev["peak_time_sec"]),
                           line_dash="dot", line_color="rgba(34,197,94,0.3)", line_width=1)
    for ev in post_events:
        fig_smo2.add_vline(x=float(ev["peak_time_sec"]),
                           line_dash="dot", line_color="rgba(249,115,22,0.3)", line_width=1)
    fig_smo2.update_layout(
        title="NIRS — Kas Oksijen Satürasyonu (SmO2)",
        height=270, xaxis_title="Zaman (sn)",
        yaxis=dict(title="SmO2 (%)", gridcolor="#333", range=[0, 100]),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=50, r=20, t=40, b=45),
        legend=dict(orientation="h", y=-0.4),
    )
    st.plotly_chart(fig_smo2, use_container_width=True)

    # EMG RMS overlay
    fig_rms = go.Figure()
    fig_rms.add_trace(go.Scatter(x=ps["t_arr"],   y=ps["rms_arr"],
        name="Pre CH1 — Rectus femoris",   line=dict(color="#3b82f6", width=1.5, dash="dot")))
    fig_rms.add_trace(go.Scatter(x=pos["t_arr"],  y=pos["rms_arr"],
        name="Post CH1 — Rectus femoris",  line=dict(color="#ef4444", width=1.5)))
    fig_rms.add_trace(go.Scatter(x=ps["t_arr"],   y=ps["rms2_arr"],
        name="Pre CH2 — Biceps femoris",  line=dict(color="#60a5fa", width=1, dash="dot")))
    fig_rms.add_trace(go.Scatter(x=pos["t_arr"],  y=pos["rms2_arr"],
        name="Post CH2 — Biceps femoris", line=dict(color="#fca5a5", width=1)))
    fig_rms.update_layout(
        title="EMG RMS — Kas Aktivasyon Büyüklüğü",
        height=260, xaxis_title="Zaman (sn)",
        yaxis=dict(title="RMS (mV)", gridcolor="#333"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=50, r=20, t=40, b=45),
        legend=dict(orientation="h", y=-0.5, font=dict(size=10)),
    )
    st.plotly_chart(fig_rms, use_container_width=True)
    st.caption("Detaylı yorum için → 📄 Sporcu Raporu sekmesi.")
