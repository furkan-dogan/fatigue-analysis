from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui.components.comparison import comparison_panel


def _render_stats_tab(pre_events: list[dict], post_events: list[dict]) -> None:
    st.subheader("İstatistiksel Karşılaştırma")
    st.info(
        "Bu bölüm ortalama farkın yanında etki büyüklüğünü gösterir. Cohen's d mutlak değeri 0.2 küçük, 0.5 orta, 0.8 ve üstü büyük etki olarak yorumlanır. "
        "Tekme sayısı düşükse güven aralıkları genişler; bu durumda sonuç ön bulgu olarak raporlanmalıdır."
    )

    from src.core.statistics import compare_metric

    STAT_METRICS = {
        "active_knee_rom_deg":         "Aktif Diz ROM (°)",
        "active_peak_knee_vel_deg_s":  "Peak Diz Hızı (°/s)",
        "active_mean_knee_vel_deg_s":  "Ort. Diz Hızı (°/s)",
        "time_to_peak_knee_vel_sec":   "Peak Hıza Süre (sn)",
        "peak_kick_height_norm":       "Tekme Yüksekliği",
        "extension_peak_vel_deg_s":    "Uzatma Hızı (°/s)",
        "retraction_peak_vel_deg_s":   "Geri Çekim Hızı (°/s)",
        "retraction_dur_sec":          "Geri Çekim Süresi (sn)",
        "knee_asi":                    "Diz ASI (%)",
        "duration_sec":                "Tekme Süresi (sn)",
    }

    stat_rows = []
    for col_k, col_lbl in STAT_METRICS.items():
        pre_v  = [float(e[col_k]) for e in pre_events  if e.get(col_k) is not None]
        post_v = [float(e[col_k]) for e in post_events if e.get(col_k) is not None]
        if not pre_v or not post_v:
            continue
        res    = compare_metric(pre_v, post_v)
        ci_pre = res["pre_ci"]
        ci_post= res["post_ci"]
        d      = res["cohens_d"]
        stat_rows.append({
            "Metrik":          col_lbl,
            "Pre ort. ± std":  f"{res['pre_mean']:.2f} ± {res['pre_std']:.2f}" if res["pre_std"] else f"{res['pre_mean']:.2f}",
            "Post ort. ± std": f"{res['post_mean']:.2f} ± {res['post_std']:.2f}" if res["post_std"] else f"{res['post_mean']:.2f}",
            "%95 CI (Pre)":    f"[{ci_pre[0]:.2f}, {ci_pre[1]:.2f}]"  if ci_pre  else "—",
            "%95 CI (Post)":   f"[{ci_post[0]:.2f}, {ci_post[1]:.2f}]" if ci_post else "—",
            "Δ":               f"{res['delta']:+.2f}" if res["delta"] is not None else "—",
            "% Değişim":       f"{res['pct_change']:+.1f}%" if res["pct_change"] is not None else "—",
            "Cohen's d":       f"{d:.2f}" if d is not None else "—",
            "Etki Büyüklüğü":  res["effect_label"],
            "n (pre/post)":    f"{res['n_pre']} / {res['n_post']}",
            "Sonuç Yorumu":    "Güçlü değişim" if d is not None and abs(d) >= 0.8 and res["n_pre"] >= 5 and res["n_post"] >= 5 else ("Ön bulgu; tekme sayısı artırılmalı" if res["n_pre"] < 5 or res["n_post"] < 5 else "Destekleyici bulgu"),
        })

    if stat_rows:
        comparison_panel(stat_rows)

        d_labels = [r["Metrik"]     for r in stat_rows if r["Cohen's d"] != "—"]
        d_vals   = [float(r["Cohen's d"]) for r in stat_rows if r["Cohen's d"] != "—"]
        d_colors = ["#ef4444" if v < 0 else "#22c55e" for v in d_vals]
        if d_vals:
            fig = go.Figure(go.Bar(
                x=d_vals, y=d_labels, orientation="h",
                marker_color=d_colors,
                text=[f"{v:+.2f}" for v in d_vals], textposition="outside",
            ))
            fig.add_vline(x=0,    line_color="#888",   line_width=1)
            fig.add_vline(x=0.8,  line_dash="dot", line_color="#f59e0b", annotation_text="büyük")
            fig.add_vline(x=-0.8, line_dash="dot", line_color="#f59e0b")
            fig.add_vline(x=0.5,  line_dash="dot", line_color="#64748b", annotation_text="orta")
            fig.add_vline(x=-0.5, line_dash="dot", line_color="#64748b")
            fig.update_layout(
                height=380, xaxis_title="Cohen's d  (negatif = post < pre)",
                margin=dict(l=200, r=80, t=20, b=30),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="#333", range=[-3, 3]),
                yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Güven Uyarıları")
    low_pre  = [e for e in pre_events  if e.get("confidence_flag") == "low"]
    low_post = [e for e in post_events if e.get("confidence_flag") == "low"]

    if not low_pre and not low_post:
        st.success("Tüm tekmelerde pose güveni yeterli (≥0.60)")
    else:
        if low_pre:
            ids = ", ".join(f"T{int(e['kick_id'])} ({e.get('pose_confidence','?'):.2f})" for e in low_pre)
            st.warning(f"**Pre** — düşük güvenli tekmeler: {ids}")
        if low_post:
            ids = ", ".join(f"T{int(e['kick_id'])} ({e.get('pose_confidence','?'):.2f})" for e in low_post)
            st.warning(f"**Post** — düşük güvenli tekmeler: {ids}")
        st.caption("Düşük güvenli tekmelerin metrikleri gürültülü olabilir — karşılaştırmada dikkate alın.")

    if len(pre_events) < 5 or len(post_events) < 5:
        st.info(
            f"Pre: {len(pre_events)} tekme, Post: {len(post_events)} tekme. "
            "İstatistiksel karşılaştırma için her oturumda en az 5 tekme önerilir. "
            "Cohen's d ve CI değerleri düşük n'de geniş belirsizlik taşır."
        )
