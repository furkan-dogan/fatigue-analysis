from __future__ import annotations

import pandas as pd
import streamlit as st
from ui.sports.taekwondo.presentation import PRIMARY_METRICS, comparison_rows, emg_device_rows, fmt_value, metric_info
from ui.components.comparison import comparison_panel
from src.sports.taekwondo.reporting.tables import _emg_frame_export_rows


def _render_export_tab(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    pre_events: list[dict],
    post_events: list[dict],
    fatigue_data: dict,
    fi: float,
    pre_emg_rows: list[dict],
    post_emg_rows: list[dict],
) -> None:
    st.subheader("Dışa Aktar")
    st.info(
        "Ham CSV dosyaları tüm ölçümleri içerir. Normal kullanım için 'okunabilir özet' dosyalarını kullanmak daha uygundur; "
        "bu dosyalarda metrik adı, pre/post değeri, değişim yüzdesi ve bilimsel yorum birlikte yer alır."
    )
    exp1, exp2, exp3 = st.columns(3)

    with exp1:
        st.markdown("**Pre Frame Metrikleri**")
        if not pre_df.empty:
            st.download_button(
                "📥 pre_frame_metrics.csv",
                pre_df.to_csv(index=False).encode("utf-8"),
                "pre_frame_metrics.csv", "text/csv",
            )
    with exp2:
        st.markdown("**Post Frame Metrikleri**")
        if not post_df.empty:
            st.download_button(
                "📥 post_frame_metrics.csv",
                post_df.to_csv(index=False).encode("utf-8"),
                "post_frame_metrics.csv", "text/csv",
            )
    with exp3:
        st.markdown("**Kick Events**")
        if pre_events:
            st.download_button(
                "📥 pre_kick_events.csv",
                pd.DataFrame(pre_events).to_csv(index=False).encode("utf-8"),
                "pre_kick_events.csv", "text/csv",
            )
        if post_events:
            st.download_button(
                "📥 post_kick_events.csv",
                pd.DataFrame(post_events).to_csv(index=False).encode("utf-8"),
                "post_kick_events.csv", "text/csv",
            )

    st.markdown("**Yorgunluk Raporu**")
    report_rows = comparison_rows(pre_events, post_events, PRIMARY_METRICS)
    for key, m in fatigue_data["metrics"].items():
        if m["pre"] is None or key in PRIMARY_METRICS:
            continue
        info = metric_info(key)
        report_rows.append({
            "Metrik": info.label,
            "Pre Ortalama": fmt_value(m["pre"], key),
            "Post Ortalama": fmt_value(m["post"], key),
            "Değişim": f"{m['pct']:+.1f}%" if m["pct"] is not None else "-",
            "Sonuç": "Yorgunluk yönünde" if m["fatigue_contribution"] and m["fatigue_contribution"] > 0 else "Yorgunluk yönünde değil",
            "Bu Ne Anlama Geliyor?": info.interpretation,
            "Nasıl Hesaplandı": info.calculation,
        })
    report_rows.append({
        "Metrik": "Yorgunluk İndeksi",
        "Pre Ortalama": "-",
        "Post Ortalama": f"{fi:.1f}/100",
        "Değişim": "-",
        "Sonuç": "Düşük" if fi < 33 else ("Orta" if fi < 66 else "Yüksek"),
        "Bu Ne Anlama Geliyor?": "Pre ve post biyomekanik metriklerin ağırlıklı değişiminden hesaplanan bileşik yorgunluk skorudur.",
        "Nasıl Hesaplandı": "Diz ROM, peak diz hızı, peak hıza süre, tekme yüksekliği, ayak hızı, tekme süresi ve ortalama hız ağırlıklı olarak birleştirilir.",
    })
    comparison_panel(report_rows)
    st.download_button(
        "📥 okunabilir_yorgunluk_raporu.csv",
        pd.DataFrame(report_rows).to_csv(index=False).encode("utf-8"),
        "okunabilir_yorgunluk_raporu.csv", "text/csv",
    )

    st.markdown("**EMG Cihaz Çıktıları**")
    st.caption(
        "Bu dosyalar video zamanıyla aynı eksendedir. Frame çıktısında her satır bir video frame'ine, tekme çıktısında her satır bir tekme penceresine karşılık gelir."
    )
    emg_exp1, emg_exp2 = st.columns(2)
    pre_emg_frame_df = pd.DataFrame(_emg_frame_export_rows(pre_df, pre_emg_rows, pre_events, "Pre"))
    post_emg_frame_df = pd.DataFrame(_emg_frame_export_rows(post_df, post_emg_rows, post_events, "Post"))
    pre_emg_kick_df = pd.DataFrame(emg_device_rows(pre_events, pre_emg_rows, "Pre"))
    post_emg_kick_df = pd.DataFrame(emg_device_rows(post_events, post_emg_rows, "Post"))
    with emg_exp1:
        st.markdown("**Frame senkron EMG**")
        if not pre_emg_frame_df.empty:
            st.download_button(
                "📥 pre_emg_frame_sync.csv",
                pre_emg_frame_df.to_csv(index=False).encode("utf-8"),
                "pre_emg_frame_sync.csv", "text/csv",
            )
        if not post_emg_frame_df.empty:
            st.download_button(
                "📥 post_emg_frame_sync.csv",
                post_emg_frame_df.to_csv(index=False).encode("utf-8"),
                "post_emg_frame_sync.csv", "text/csv",
            )
    with emg_exp2:
        st.markdown("**Tekme bazlı EMG**")
        if not pre_emg_kick_df.empty:
            st.download_button(
                "📥 pre_emg_kick_summary.csv",
                pre_emg_kick_df.to_csv(index=False).encode("utf-8"),
                "pre_emg_kick_summary.csv", "text/csv",
            )
        if not post_emg_kick_df.empty:
            st.download_button(
                "📥 post_emg_kick_summary.csv",
                post_emg_kick_df.to_csv(index=False).encode("utf-8"),
                "post_emg_kick_summary.csv", "text/csv",
            )
    with st.expander("EMG CSV kolonları nasıl okunur?", expanded=False):
        st.markdown("**frame_idx / video_time_sec:** EMG satırının hangi video frame'i ve saniyesiyle eşleştiğini gösterir.")
        st.markdown("**kick_id:** Satır bir tekme penceresindeyse tekme numarasıdır; boşsa tekme dışında kalan zamandır.")
        st.markdown("**kick_phase:** Frame'in tekmenin hazırlık, vuruşa yükselme, vuruş anı veya geri çekme bölümünde olduğunu gösterir.")
        st.markdown("**CH1_RectusFemoris_mV:** Rectus femoris kanalının mV cinsinden aktivasyon değeridir.")
        st.markdown("**CH2_BicepsFemoris_mV:** Biceps femoris kanalının mV cinsinden aktivasyon değeridir.")
        st.markdown("**CH1/CH2 median freq Hz:** Her kas kanalının frekans bilgisidir; tekrarlar ilerledikçe düşmesi yorgunlukla uyumludur.")
