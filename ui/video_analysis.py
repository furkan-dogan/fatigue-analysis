"""Video Analizi sayfası — pre/post karşılaştırma."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.fatigue import FATIGUE_METRICS, compute_fatigue
from src.utils import events_mean
from ui.charts import gauge, overlay_chart, per_kick_trend
from ui.components import kick_video_section, phase_bars, sensor_stats, video_player
from ui.report import render_athlete_report


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _render_fatigue_tab(fatigue_data: dict, fi: float, fi_label: str) -> None:
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(gauge(fi, "Yorgunluk İndeksi"), use_container_width=True)
        fi_color = "🟢" if fi < 33 else ("🟡" if fi < 66 else "🔴")
        st.markdown(f"**{fi_color} {fi_label} yorgunluk** — {fi:.1f}/100")
        st.caption(
            "0–33: Düşük  |  33–66: Orta  |  66–100: Yüksek\n\n"
            "Ağırlıklar: Diz ROM ×0.25, Peak hız ×0.25, "
            "Peak hıza süre ×0.15, Tekme yüksekliği ×0.15, Ayak hızı ×0.10"
        )
    with g2:
        st.subheader("Metrik Bazlı Yorgunluk Katkıları")
        mets = fatigue_data["metrics"]
        rows_ft = []
        for m in mets.values():
            if m["pre"] is None or m["post"] is None or m["pct"] is None:
                continue
            trend = "▲" if m["pct"] > 0 else "▼"
            is_fatigue = (m["direction"] * m["pct"]) > 0
            rows_ft.append({
                "Metrik":      f"{m['label']} ({m['unit']})" if m["unit"] else m["label"],
                "Pre":         f"{m['pre']:.3f}",
                "Post":        f"{m['post']:.3f}",
                "Δ":           f"{m['delta']:+.3f}",
                "% Değişim":   f"{trend} {abs(m['pct']):.1f}%",
                "Yorgunluk":   "🔴 Evet" if is_fatigue else "🟢 Hayır",
                "Katkı":       f"{m['fatigue_contribution']:+.1f}",
            })
        st.dataframe(pd.DataFrame(rows_ft).set_index("Metrik"), use_container_width=True)

    st.subheader("Yorgunluk Katkı Grafiği")
    labels, vals, colors = [], [], []
    for m in mets.values():
        if m.get("fatigue_contribution") is None:
            continue
        labels.append(m["label"])
        vals.append(round(m["fatigue_contribution"], 1))
        colors.append("#ef4444" if m["fatigue_contribution"] > 0 else "#22c55e")

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in vals], textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#888", line_width=1)
    fig.update_layout(
        height=340, xaxis_title="Yorgunluk katkısı (+ = yorgunluk, − = iyileşme)",
        margin=dict(l=180, r=60, t=20, b=30),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#333", range=[-110, 110]),
        yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_per_kick_tab(
    pre_events: list[dict],
    post_events: list[dict],
    tmp: Path | None,
) -> None:
    st.caption("Her tekme için pre vs post karşılaştırması")
    pk_col = st.selectbox(
        "Metrik",
        [k for k in FATIGUE_METRICS if any(e.get(k) is not None for e in pre_events + post_events)],
        format_func=lambda k: FATIGUE_METRICS[k][0],
        key="dv_pk_col",
    )
    if pk_col:
        st.plotly_chart(
            per_kick_trend(pre_events, post_events, pk_col, FATIGUE_METRICS[pk_col][0]),
            use_container_width=True,
        )

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


def _render_phase_tab(pre_events: list[dict], post_events: list[dict]) -> None:
    st.subheader("Tekme Faz Analizi")
    st.caption("Her tekme 3 faza ayrılır: Yüklenme (chamber) → Uzatma (extension) → Geri çekim (retraction)")

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


def _render_asymmetry_tab(pre_events: list[dict], post_events: list[dict]) -> None:
    st.subheader("Bilateral Asimetri İndeksi (ASI)")
    st.caption("ASI = (Sağ − Sol) / Ort(Sağ, Sol) × 100  |  Pozitif → Sağ dominant  |  |ASI| > 10% klinik olarak anlamlı")

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


def _render_stats_tab(pre_events: list[dict], post_events: list[dict]) -> None:
    st.subheader("İstatistiksel Karşılaştırma")
    st.caption("Cohen's d etki büyüklüğü + %95 güven aralığı — düşük tekme sayısında yorumla dikkatli")

    from src.stats import compare_metric

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
        })

    if stat_rows:
        st.dataframe(pd.DataFrame(stat_rows).set_index("Metrik"), use_container_width=True)

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
            f"ℹ️ Pre: {len(pre_events)} tekme, Post: {len(post_events)} tekme. "
            "İstatistiksel karşılaştırma için her oturumda en az 5 tekme önerilir. "
            "Cohen's d ve CI değerleri düşük n'de geniş belirsizlik taşır."
        )


def _render_export_tab(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    pre_events: list[dict],
    post_events: list[dict],
    fatigue_data: dict,
    fi: float,
) -> None:
    st.subheader("Dışa Aktar")
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
    report_rows = []
    for key, m in fatigue_data["metrics"].items():
        if m["pre"] is None:
            continue
        report_rows.append({
            "metrik_kodu":         key,
            "metrik_label":        m["label"],
            "birim":               m["unit"],
            "pre":                 round(m["pre"],  4) if m["pre"]  else "",
            "post":                round(m["post"], 4) if m["post"] else "",
            "delta":               round(m["delta"], 4) if m["delta"] is not None else "",
            "pct_degisim":         round(m["pct"],  2)  if m["pct"]  is not None else "",
            "yorgunluk_katkisi":   round(m["fatigue_contribution"], 2) if m["fatigue_contribution"] is not None else "",
        })
    report_rows.append({
        "metrik_kodu": "YORGUNLUK_INDEKSI", "metrik_label": "Yorgunluk İndeksi",
        "birim": "/100", "pre": "", "post": "", "delta": "",
        "pct_degisim": "", "yorgunluk_katkisi": round(fi, 2),
    })
    st.download_button(
        "📥 yorgunluk_raporu.csv",
        pd.DataFrame(report_rows).to_csv(index=False).encode("utf-8"),
        "yorgunluk_raporu.csv", "text/csv",
    )


def _render_sensor_tab(
    sensor_ok: bool,
    ps: dict,
    pos: dict,
    pre_events: list[dict],
    post_events: list[dict],
    pre_freq_drop: float,
    post_freq_drop: float,
) -> None:
    st.subheader("Sensör Simülasyonu — EMG + NIRS")
    st.caption(
        "Video analizinden fizyolojik model ile üretilen simüle EMG ve NIRS karşılaştırması. "
        "Pre → Post arası yorgunluk değişimi, nöromüsküler ve metabolik perspektiften. "
        "*(Model tabanlı simülasyon — gerçek K-Myo + Moxy verisi değil)*"
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
        name="Pre CH1 — Aktif Bacak",   line=dict(color="#3b82f6", width=1.5, dash="dot")))
    fig_rms.add_trace(go.Scatter(x=pos["t_arr"],  y=pos["rms_arr"],
        name="Post CH1 — Aktif Bacak",  line=dict(color="#ef4444", width=1.5)))
    fig_rms.add_trace(go.Scatter(x=ps["t_arr"],   y=ps["rms2_arr"],
        name="Pre CH2 — Stance Bacak",  line=dict(color="#60a5fa", width=1, dash="dot")))
    fig_rms.add_trace(go.Scatter(x=pos["t_arr"],  y=pos["rms2_arr"],
        name="Post CH2 — Stance Bacak", line=dict(color="#fca5a5", width=1)))
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


def render() -> None:
    st.title("Video Analizi — Yorgunluk Değerlendirmesi")
    st.caption(
        "Pre ve post antrenman videolarını yükle, analiz et, "
        "yorgunluk metriklerini ve sensör simülasyonunu incele."
    )

    # ── Sidebar params ─────────────────────────────────────────────────────────
    st.sidebar.subheader("Analiz Parametreleri")
    dv_show_labels = st.sidebar.checkbox(
        "Eklem etiketleri", value=False, key="dv_labels",
        help="Annotated videoda her eklemin üstüne kısa isim yazar. "
             "Pose takibini görsel kontrol etmek için açın.",
    )
    dv_prominence = st.sidebar.slider(
        "Event prominence", 0.02, 0.20, 0.06, 0.01, key="dv_prom",
        help="Ayağın baseline'ın ne kadar üstüne çıkınca 'tekme başladı' sayılsın.",
    )
    dv_min_dist = st.sidebar.slider(
        "Min peak mesafe (sn)", 0.1, 1.5, 0.25, 0.05, key="dv_dist",
        help="İki ayrı tekme arasındaki minimum süre.",
    )
    dv_min_dur = st.sidebar.slider(
        "Min event süresi (sn)", 0.05, 0.5, 0.10, 0.05, key="dv_dur",
        help="Bu süreden kısa hareketler tekme sayılmaz.",
    )
    dv_max_dur = st.sidebar.slider(
        "Max event süresi (sn)", 1.0, 10.0, 6.0, 0.5, key="dv_maxdur",
        help="Bu süreden uzun hareketler tekme sayılmaz.",
    )
    st.sidebar.subheader("Kick Doğrulama")
    dv_min_rom = st.sidebar.slider(
        "Min diz ROM (°)", 0, 60, 12, 5, key="dv_rom",
        help="Tekme sayılması için dizin en az bu kadar açılıp kapanması gerekir.",
    )
    dv_min_height = st.sidebar.slider(
        "Min peak yükseklik", -1.0, 0.5, -0.5, 0.05, key="dv_height",
        help="Tekme anında ayağın ulaşması gereken minimum normalize yükseklik.",
    )
    dv_vel_assist = st.sidebar.slider(
        "Hız yardımı eşiği (°/s)", 50, 500, 100, 25, key="dv_vel_assist",
        help="Diz açısal hızı bu değeri geçen anlarda ikincil tekme adayı oluşturulur.",
    )

    # ── Upload ─────────────────────────────────────────────────────────────────
    col_pre, col_post = st.columns(2)
    with col_pre:
        st.markdown("#### Pre-antrenman")
        pre_upload = st.file_uploader("Pre video", type=["mp4", "avi", "mov"], key="dv_pre")
    with col_post:
        st.markdown("#### Post-antrenman")
        post_upload = st.file_uploader("Post video", type=["mp4", "avi", "mov"], key="dv_post")

    both_ready = pre_upload is not None and post_upload is not None
    run_dual   = st.button("▶ Her İkisini Analiz Et", disabled=not both_ready, type="primary")

    # ── Session state ──────────────────────────────────────────────────────────
    if "dv_pre_result" not in st.session_state:
        st.session_state.update({
            "dv_pre_result":  None,
            "dv_post_result": None,
            "dv_pre_df":      None,
            "dv_post_df":     None,
            "dv_tmp":         None,
        })

    if run_dual and both_ready:
        from src.pipeline import run_analysis

        tmp = Path(tempfile.mkdtemp())
        pre_in,   post_in   = tmp / "pre_input.mp4",     tmp / "post_input.mp4"
        pre_out,  post_out  = tmp / "pre_annotated.mp4", tmp / "post_annotated.mp4"
        pre_fcsv, post_fcsv = tmp / "pre_frames.csv",    tmp / "post_frames.csv"
        pre_ecsv, post_ecsv = tmp / "pre_events.csv",    tmp / "post_events.csv"

        pre_in.write_bytes(pre_upload.read())
        post_in.write_bytes(post_upload.read())

        prog = st.progress(0, text="Pre video analiz ediliyor…")

        def _prog_pre(cur, total):
            prog.progress(int(min(cur / max(total, 1), 1.0) * 50), text=f"PRE — Frame {cur}/{total}")

        def _prog_post(cur, total):
            prog.progress(50 + int(min(cur / max(total, 1), 1.0) * 50), text=f"POST — Frame {cur}/{total}")

        try:
            kw = dict(
                show_joint_labels=dv_show_labels,
                event_peak_prominence_norm=dv_prominence,
                event_min_distance_sec=dv_min_dist,
                event_min_duration_sec=dv_min_dur,
                event_max_duration_sec=dv_max_dur,
                event_min_knee_rom_deg=float(dv_min_rom),
                event_min_peak_kick_height_norm=float(dv_min_height),
                vel_assist_threshold=float(dv_vel_assist),
            )
            pre_res  = run_analysis(pre_in,  pre_out,  pre_fcsv,  pre_ecsv,  progress_callback=_prog_pre,  **kw)
            post_res = run_analysis(post_in, post_out, post_fcsv, post_ecsv, progress_callback=_prog_post, **kw)
            prog.progress(100, text="Tamamlandı!")
        except Exception as exc:
            st.error(f"Analiz hatası: {exc}")
            st.stop()

        st.session_state.update({
            "dv_pre_result":  pre_res,
            "dv_post_result": post_res,
            "dv_pre_df":      _read_csv(str(pre_fcsv)),
            "dv_post_df":     _read_csv(str(post_fcsv)),
            "dv_tmp":         tmp,
        })

    pre_res  = st.session_state.get("dv_pre_result")
    post_res = st.session_state.get("dv_post_result")
    pre_df   = st.session_state.get("dv_pre_df",  pd.DataFrame())
    post_df  = st.session_state.get("dv_post_df", pd.DataFrame())

    if pre_res is None or post_res is None:
        st.info("Her iki videoyu yükleyip analiz başlatın.")
        st.stop()

    pre_events, post_events = pre_res.events, post_res.events

    # ── Sensor stats ───────────────────────────────────────────────────────────
    _pre_emg   = pre_res.synthetic_emg_rows   or []
    _pre_nirs  = pre_res.synthetic_nirs_rows  or []
    _post_emg  = post_res.synthetic_emg_rows  or []
    _post_nirs = post_res.synthetic_nirs_rows or []
    _sensor_ok = bool(_pre_emg and _post_emg)

    if _sensor_ok:
        _win5           = max(1, int(pre_res.fps * 5))
        _ps             = sensor_stats(_pre_emg,  _pre_nirs,  _win5)
        _pos            = sensor_stats(_post_emg, _post_nirs, _win5)
        _pre_freq_drop  = _ps["freq_start"]  - _ps["freq_end"]
        _post_freq_drop = _pos["freq_start"] - _pos["freq_end"]
        _pre_smo2_drop  = _ps["smo2_start"]  - _ps["smo2_end"]
        _post_smo2_drop = _pos["smo2_start"] - _pos["smo2_end"]
    else:
        _ps = _pos = {}
        _pre_freq_drop = _post_freq_drop = _pre_smo2_drop = _post_smo2_drop = 0.0

    # ── Summary header ─────────────────────────────────────────────────────────
    fatigue_data = compute_fatigue(pre_events, post_events)
    fi = fatigue_data["fatigue_index"]

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Pre — Tespit Edilen Tekme", len(pre_events))
    h1.metric("Pre — Toplam Frame",        pre_res.total_frames)
    h2.metric("Post — Tespit Edilen Tekme", len(post_events))
    h2.metric("Post — Toplam Frame",        post_res.total_frames)

    pre_mean_vel  = events_mean(pre_events,  "active_peak_knee_vel_deg_s")
    post_mean_vel = events_mean(post_events, "active_peak_knee_vel_deg_s")
    if pre_mean_vel and post_mean_vel:
        h3.metric("Ort. Peak Diz Hızı — Pre",  f"{pre_mean_vel:.0f} °/s")
        h3.metric("Ort. Peak Diz Hızı — Post", f"{post_mean_vel:.0f} °/s",
                  delta=f"{post_mean_vel - pre_mean_vel:+.0f} °/s", delta_color="inverse")

    pre_mean_rom  = events_mean(pre_events,  "active_knee_rom_deg")
    post_mean_rom = events_mean(post_events, "active_knee_rom_deg")
    if pre_mean_rom and post_mean_rom:
        h4.metric("Ort. Diz ROM — Pre",  f"{pre_mean_rom:.1f}°")
        h4.metric("Ort. Diz ROM — Post", f"{post_mean_rom:.1f}°",
                  delta=f"{post_mean_rom - pre_mean_rom:+.1f}°", delta_color="inverse")

    fi_label = "Düşük" if fi < 33 else ("Orta" if fi < 66 else "Yüksek")
    h5.metric("Yorgunluk İndeksi", f"{fi:.1f}/100", delta=fi_label, delta_color="off")
    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📹 Videolar", "📐 Açı Karşılaştırma", "⚡ Hız Karşılaştırma",
        "🔥 Yorgunluk Analizi", "📊 Tekme Bazlı", "🏃 Faz Analizi",
        "📏 Asimetri", "🔬 İstatistik", "💾 Export",
        "🧪 Sensör Simülasyonu", "📄 Sporcu Raporu",
    ])

    with tabs[0]:
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**Pre-antrenman (annotated)**")
            pre_vid = Path(pre_res.output_video_path)
            if pre_vid.exists():
                video_player(pre_vid)
            else:
                st.warning("Pre video çıktısı bulunamadı.")
        with vc2:
            st.markdown("**Post-antrenman (annotated)**")
            post_vid = Path(post_res.output_video_path)
            if post_vid.exists():
                video_player(post_vid)
            else:
                st.warning("Post video çıktısı bulunamadı.")

    with tabs[1]:
        st.caption("Mavi kesikli = Pre  |  Kırmızı düz = Post  |  Renkli bantlar = tekme eventleri")
        for col_name, lbl in [
            ("R_KNEE",  "Sağ Diz Açısı"),       ("L_KNEE",  "Sol Diz Açısı"),
            ("R_HIP",   "Sağ Kalça Açısı"),      ("L_HIP",   "Sol Kalça Açısı"),
            ("R_ANKLE", "Sağ Ayak Bileği Açısı"),("L_ANKLE", "Sol Ayak Bileği Açısı"),
        ]:
            if col_name in pre_df.columns or col_name in post_df.columns:
                st.plotly_chart(
                    overlay_chart(pre_df, post_df, col_name, lbl, pre_events, post_events),
                    use_container_width=True,
                )

    with tabs[2]:
        st.caption("Açısal hız ve ayak hızı — yorgunluk en çok hızda görünür")
        for col_name, lbl in [
            ("R_KNEE_vel_deg_s",  "Sağ Diz Açısal Hızı"),
            ("L_KNEE_vel_deg_s",  "Sol Diz Açısal Hızı"),
            ("R_HIP_vel_deg_s",   "Sağ Kalça Açısal Hızı"),
            ("R_FOOT_speed_norm", "Sağ Ayak Hızı (normalize)"),
            ("L_FOOT_speed_norm", "Sol Ayak Hızı (normalize)"),
        ]:
            if col_name in pre_df.columns or col_name in post_df.columns:
                st.plotly_chart(
                    overlay_chart(pre_df, post_df, col_name, lbl, pre_events, post_events),
                    use_container_width=True,
                )

    with tabs[3]:
        _render_fatigue_tab(fatigue_data, fi, fi_label)

    with tabs[4]:
        _render_per_kick_tab(pre_events, post_events, st.session_state.get("dv_tmp"))

    with tabs[5]:
        _render_phase_tab(pre_events, post_events)

    with tabs[6]:
        _render_asymmetry_tab(pre_events, post_events)

    with tabs[7]:
        _render_stats_tab(pre_events, post_events)

    with tabs[8]:
        _render_export_tab(pre_df, post_df, pre_events, post_events, fatigue_data, fi)

    with tabs[9]:
        _render_sensor_tab(_sensor_ok, _ps, _pos, pre_events, post_events,
                           _pre_freq_drop, _post_freq_drop)

    with tabs[10]:
        render_athlete_report(
            pre_events, post_events, pre_res, post_res, fi,
            _ps, _pos,
            _pre_freq_drop, _post_freq_drop,
            _pre_smo2_drop, _post_smo2_drop,
            _sensor_ok,
        )
