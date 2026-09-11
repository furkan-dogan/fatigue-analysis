from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui.paths import SAMPLE_DATA_DIR
from ui.components.comparison import comparison_panel


def _render_real_sensor_tab(
    pre_res,
    post_res,
    pre_events: list[dict],
    post_events: list[dict],
) -> bool:
    """EMG+NIRS CSV yükleme, senkron ve görselleştirme. CSV yüklüyse True döner."""
    import tempfile
    from src.sports.taekwondo.sensor_sync import (
        load_emg_csv, load_nirs_csv,
        resample_to_video_times, resample_nirs_to_video_times,
        compute_rms_per_kick, compute_nirs_per_kick,
    )

    st.subheader("Sensör CSV Verisi — EMG + NIRS")
    st.info(
        "K-Myo / Delsys EMG ve Moxy NIRS cihazlarından alınan CSV çıktılarını yükleyin. "
        "Sistem sensör zaman serisini video frame zamanlarına yeniden örnekler, tekme pencereleriyle eşleştirir ve her tekme için RMS/SmO2 özeti üretir."
    )

    sample_emg  = SAMPLE_DATA_DIR / "emg_sample.csv"
    sample_nirs = SAMPLE_DATA_DIR / "nirs_moxy_sample.csv"
    dl1, dl2, _ = st.columns([1, 1, 3])
    if sample_emg.exists():
        dl1.download_button(
            "📥 Örnek EMG CSV", sample_emg.read_bytes(),
            "emg_sample.csv", "text/csv", key="dl_emg_sample",
        )
    if sample_nirs.exists():
        dl2.download_button(
            "📥 Örnek NIRS CSV", sample_nirs.read_bytes(),
            "nirs_moxy_sample.csv", "text/csv", key="dl_nirs_sample",
        )

    st.divider()
    pre_col, post_col = st.columns(2)
    with pre_col:
        st.markdown("**Pre Oturum**")
        pre_emg_f  = st.file_uploader("EMG CSV",         type=["csv"], key="rs_pre_emg")
        pre_nirs_f = st.file_uploader("NIRS CSV (Moxy)",  type=["csv"], key="rs_pre_nirs")
        pre_off    = st.slider("Sync gecikmesi — Pre (sn)", -30.0, 30.0, 0.0, 0.5, key="rs_pre_off",
                               help="EMG/NIRS kaydı videodan kaç saniye sonra başladı.")
    with post_col:
        st.markdown("**Post Oturum**")
        post_emg_f  = st.file_uploader("EMG CSV",         type=["csv"], key="rs_post_emg")
        post_nirs_f = st.file_uploader("NIRS CSV (Moxy)",  type=["csv"], key="rs_post_nirs")
        post_off    = st.slider("Sync gecikmesi — Post (sn)", -30.0, 30.0, 0.0, 0.5, key="rs_post_off",
                                help="EMG/NIRS kaydı videodan kaç saniye sonra başladı.")

    any_uploaded = any([pre_emg_f, pre_nirs_f, post_emg_f, post_nirs_f])
    if not any_uploaded:
        st.info("Sensör CSV dosyalarını yükleyin veya yukarıdaki örnek dosyaları indirip deneyin.")
        return False

    def _save(uf) -> Path:
        p = Path(tempfile.mktemp(suffix=".csv"))
        p.write_bytes(uf.read())
        return p

    color_map  = {"Pre": "#3b82f6", "Post": "#ef4444"}
    ev_map     = {"Pre": pre_events, "Post": post_events}
    sessions   = [
        ("Pre",  pre_emg_f,  pre_nirs_f,  pre_events,  pre_res.fps,  pre_res.total_frames,  pre_off),
        ("Post", post_emg_f, post_nirs_f, post_events, post_res.fps, post_res.total_frames, post_off),
    ]

    results: dict[str, dict] = {}
    for lbl, emg_f, nirs_f, events, fps, total_frames, offset in sessions:
        if not emg_f and not nirs_f:
            continue
        vtimes = [i / fps for i in range(total_frames)]
        sr: dict = {"t": vtimes, "offset": offset}

        if emg_f:
            try:
                emg_data          = load_emg_csv(_save(emg_f))
                emg_res           = resample_to_video_times(emg_data, vtimes, offset)
                sr["emg"]         = emg_data
                sr["emg_res"]     = emg_res
                sr["kick_emg"]    = compute_rms_per_kick(emg_res, events, fps)
            except Exception as exc:
                st.error(f"{lbl} EMG yüklenemedi: {exc}")

        if nirs_f:
            try:
                nirs_data         = load_nirs_csv(_save(nirs_f))
                nirs_res          = resample_nirs_to_video_times(nirs_data, vtimes, offset)
                sr["nirs"]        = nirs_data
                sr["nirs_res"]    = nirs_res
                sr["kick_nirs"]   = compute_nirs_per_kick(nirs_res, events)
            except Exception as exc:
                st.error(f"{lbl} NIRS yüklenemedi: {exc}")

        if len(sr) > 2:
            results[lbl] = sr

    if not results:
        return False

    # ── EMG grafiği ────────────────────────────────────────────────────────────
    if any("emg" in v for v in results.values()):
        st.subheader("EMG — Kas Aktivasyonu")
        fig_emg = go.Figure()
        for lbl, sr in results.items():
            if "emg_res" not in sr:
                continue
            ch_name = list(sr["emg_res"].keys())[0]
            vals = [v if v is not None else float("nan") for v in sr["emg_res"][ch_name]]
            fig_emg.add_trace(go.Scatter(
                x=sr["t"], y=vals, name=f"{lbl} — {ch_name}",
                line=dict(color=color_map[lbl], width=1.2), opacity=0.85,
            ))
        for ev in pre_events:
            fig_emg.add_vline(x=float(ev["peak_time_sec"]),
                              line_dash="dot", line_color="rgba(59,130,246,0.35)", line_width=1)
        for ev in post_events:
            fig_emg.add_vline(x=float(ev["peak_time_sec"]),
                              line_dash="dot", line_color="rgba(239,68,68,0.35)", line_width=1)
        fig_emg.update_layout(
            height=260, xaxis_title="Zaman (sn)", yaxis_title="mV",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
            margin=dict(l=50, r=20, t=30, b=40), legend=dict(orientation="h", y=-0.4),
        )
        st.plotly_chart(fig_emg, use_container_width=True)

    # ── NIRS grafiği ───────────────────────────────────────────────────────────
    if any("nirs" in v for v in results.values()):
        st.subheader("NIRS — Kas Oksijen Satürasyonu (SmO2)")
        fig_nirs = go.Figure()
        for lbl, sr in results.items():
            if "nirs" not in sr:
                continue
            nd = sr["nirs"]
            adj_t = [t + sr["offset"] for t in nd["time_s"]]
            fig_nirs.add_trace(go.Scatter(
                x=adj_t, y=nd["smo2"], name=f"{lbl} — SmO2",
                line=dict(color=color_map[lbl], width=2),
            ))
        for ev in pre_events:
            fig_nirs.add_vline(x=float(ev["peak_time_sec"]),
                               line_dash="dot", line_color="rgba(59,130,246,0.35)", line_width=1)
        for ev in post_events:
            fig_nirs.add_vline(x=float(ev["peak_time_sec"]),
                               line_dash="dot", line_color="rgba(239,68,68,0.35)", line_width=1)
        fig_nirs.update_layout(
            height=260, xaxis_title="Zaman (sn)", yaxis_title="SmO2 (%)",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", range=[30, 100]),
            margin=dict(l=50, r=20, t=30, b=40), legend=dict(orientation="h", y=-0.4),
        )
        st.plotly_chart(fig_nirs, use_container_width=True)

    # ── Per-kick tablo ─────────────────────────────────────────────────────────
    st.subheader("Tekme Bazlı Sensör Özeti")
    for lbl, sr in results.items():
        st.markdown(f"**{lbl} Oturum**")
        rows = []
        for ev in ev_map[lbl]:
            kid = int(ev["kick_id"])
            row: dict = {"Tekme": f"T{kid}", "Süre (sn)": round(float(ev.get("duration_sec", 0)), 2)}
            if "kick_emg" in sr:
                ke = next((k for k in sr["kick_emg"] if k["kick_id"] == kid), {})
                for ck in [k for k in ke if k.endswith("_rms")][:2]:
                    row[ck.replace("_rms", " RMS (mV)")] = ke.get(ck)
            if "kick_nirs" in sr:
                kn = next((k for k in sr["kick_nirs"] if k["kick_id"] == kid), {})
                row["Ort. SmO2 (%)"] = kn.get("mean_smo2")
                if kn.get("mean_thb") is not None:
                    row["Ort. THb"] = kn.get("mean_thb")
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Tekme"), use_container_width=True)

    st.session_state["rs_real_sensor_ok"] = True
    return True
