"""Taekwondo upload and analysis-tab composition."""
from __future__ import annotations

from src.adapters.csv_read import read_optional_csv as _read_csv
from ui.components.upload import video_uploader
from ui.sports.taekwondo.tabs.summary import _render_general_summary
from ui.sports.taekwondo.tabs.fatigue import _render_fatigue_tab
from ui.sports.taekwondo.tabs.events import _render_per_kick_tab
from ui.sports.taekwondo.tabs.phases import _render_phase_tab
from ui.sports.taekwondo.tabs.asymmetry import _render_asymmetry_tab
from ui.sports.taekwondo.tabs.statistics import _render_stats_tab
from ui.sports.taekwondo.tabs.export import _render_export_tab
from ui.sports.taekwondo.tabs.simulation import _render_sensor_tab
from ui.sports.taekwondo.tabs.sensors import _render_real_sensor_tab


import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from src.sports.taekwondo.fatigue import compute_fatigue
from src.core.numeric import events_mean
from ui.sports.taekwondo.presentation import render_metric_help
from ui.sports.taekwondo.charts import overlay_chart
from src.sports.taekwondo.sensor_summary import sensor_stats
from ui.sports.taekwondo.report import render_athlete_report


def render() -> None:
    st.title("Video Analizi — Yorgunluk Değerlendirmesi")
    st.info(
        "Pre ve post antrenman videolarını yükle, analiz et, "
        "yorgunluk metriklerini ve EMG değerlerini incele. "
        "Ekrandaki her tablo/grafik, sporcu veya antrenörün kolay okuyacağı şekilde 'ne ölçüldü, nasıl hesaplandı, nasıl yorumlanır' açıklamalarıyla verilir."
    )

    # Fixed analysis parameters keep demo and repeated analyses consistent.
    dv_show_labels = False
    dv_prominence = 0.06
    dv_min_dist = 0.25
    dv_min_dur = 0.10
    dv_max_dur = 6.0
    dv_min_rom = 12
    dv_min_height = -0.5
    dv_vel_assist = 100

    # ── Upload ─────────────────────────────────────────────────────────────────
    col_pre, col_post = st.columns(2)
    with col_pre:
        st.markdown("#### Pre-antrenman")
        pre_upload = video_uploader("Pre video", key="dv_pre")
    with col_post:
        st.markdown("#### Post-antrenman")
        post_upload = video_uploader("Post video", key="dv_post")

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
        from src.sports.taekwondo.pipeline import run_analysis

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
        "Özet", "📐 Açı Karşılaştırma", "⚡ Hız Karşılaştırma",
        "🔥 Yorgunluk Analizi", "📊 Tekme Bazlı", "🏃 Faz Analizi",
        "📏 Asimetri", "🔬 İstatistik", "💾 Export",
        "🧪 EMG Detayları", "📡 Sensör CSV", "📄 Sporcu Raporu",
    ])

    with tabs[0]:
        _render_general_summary(
            pre_events,
            post_events,
            pre_res,
            post_res,
            fatigue_data,
            fi,
            fi_label,
            _pre_emg,
            _post_emg,
        )

    with tabs[1]:
        st.info(
            "Açı grafikleri eklemlerin zaman içindeki konum değişimini gösterir. "
            "Diz ve kalça açılarındaki post düşüşleri hareket genişliği ve teknik stabilite açısından değerlendirilir."
        )
        render_metric_help(["active_knee_rom_deg"], "Açı grafiklerini nasıl okumalıyım?")
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
        st.info(
            "Hız grafikleri yorgunluk etkisini en hızlı gösteren bölümdür. "
            "Post çizgisinde peak hızların azalması, sporcunun patlayıcı hareket üretimini koruyamadığını gösterebilir."
        )
        render_metric_help(["active_peak_knee_vel_deg_s", "active_peak_foot_speed_norm"], "Hız metrikleri nasıl hesaplandı?")
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
        _render_export_tab(pre_df, post_df, pre_events, post_events, fatigue_data, fi, _pre_emg, _post_emg)

    with tabs[9]:
        _render_sensor_tab(_sensor_ok, _ps, _pos, pre_events, post_events,
                           _pre_freq_drop, _post_freq_drop)

    with tabs[10]:
        _real_sensor_loaded = _render_real_sensor_tab(pre_res, post_res, pre_events, post_events)

    with tabs[11]:
        _real_ok = st.session_state.get("rs_real_sensor_ok", False)
        render_athlete_report(
            pre_events, post_events, pre_res, post_res, fi,
            _ps, _pos,
            _pre_freq_drop, _post_freq_drop,
            _pre_smo2_drop, _post_smo2_drop,
            _sensor_ok,
            real_sensor=_real_ok,
        )
