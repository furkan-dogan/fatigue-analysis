"""Kick-specific tables and phase panels."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.adapters.video_clips import trim_clip
from ui.components.video_player import video_player
from ui.components.models import TimelineEvent
from ui.components.timeline import event_timeline
from ui.sports.taekwondo.presentation import readable_kick_rows, render_metric_help

def kick_video_section(
    events_list: list[dict],
    video_path: str,
    label: str,
    tmp_dir: Path | None,
) -> None:
    st.markdown(f"#### {label}")
    if not events_list:
        st.info("Tekme tespit edilemedi.")
        return

    ev_df = pd.DataFrame(readable_kick_rows(events_list))
    st.dataframe(
        ev_df.set_index("Tekme"),
        use_container_width=True,
    )
    render_metric_help(
        [
            "duration_sec",
            "active_knee_rom_deg",
            "active_peak_knee_vel_deg_s",
            "peak_kick_height_norm",
            "pose_confidence",
        ],
        "Bu tekme tablosundaki sütunlar ne anlama geliyor?",
    )

    vid_path = Path(video_path)
    timeline_events = []
    for event in events_list:
        try:
            timeline_events.append(TimelineEvent(
                str(event['kick_id']), f"Tekme {event['kick_id']}",
                float(event['start_time_sec']), float(event['end_time_sec']),
                float(event['peak_time_sec']) if event.get('peak_time_sec') is not None else None,
            ))
        except (KeyError, TypeError, ValueError):
            st.warning('Zaman bilgisi eksik veya geçersiz bir olay çizelgeye eklenemedi.')
    selected = event_timeline(timeline_events, key=f'taekwondo_{label}_timeline')
    if selected and vid_path.is_file():
        video_player(vid_path, start_time=selected.start_seconds)

    for ev in events_list:
        kid     = int(ev["kick_id"])
        leg     = ev.get("active_leg", "?")
        t_start = float(ev["start_time_sec"])
        t_end   = float(ev["end_time_sec"])
        dur     = float(ev.get("duration_sec", 0))
        rom     = ev.get("active_knee_rom_deg")
        vel     = ev.get("active_peak_knee_vel_deg_s")
        height  = ev.get("peak_kick_height_norm")

        exp_title = (
            f"Tekme {kid}  |  {leg} bacak  |  {t_start:.2f}s – {t_end:.2f}s  |  ROM {float(rom):.1f}°"
            if rom else
            f"Tekme {kid}  |  {leg} bacak  |  {t_start:.2f}s – {t_end:.2f}s"
        )
        with st.expander(exp_title, expanded=(kid == 1)):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Süre",       f"{dur:.2f} sn")
            mc2.metric("Diz ROM",    f"{float(rom):.1f}°"    if rom    else "—")
            mc3.metric("Peak Hız",   f"{float(vel):.0f} °/s" if vel    else "—")
            mc4.metric("Yükseklik",  f"{float(height):.3f}"  if height else "—")

            if vid_path.exists() and tmp_dir is not None:
                clip_key  = f"clip_{label.replace(' ', '_')}_{kid}"
                clip_path = tmp_dir / f"{clip_key}.mp4"
                if not clip_path.exists():
                    with st.spinner("Video kırpılıyor…"):
                        trim_clip(vid_path, t_start, t_end, clip_path)
                if clip_path.exists() and clip_path.stat().st_size > 1000:
                    video_player(clip_path)
                else:
                    video_player(vid_path, start_time=t_start)
            elif vid_path.exists():
                video_player(vid_path, start_time=t_start)

def phase_bars(events_list: list[dict], label: str, color: str) -> None:
    phases    = ["Yüklenme", "Uzatma", "Geri Çekim"]
    dur_keys  = ["loading_dur_sec", "extension_dur_sec", "retraction_dur_sec"]
    fig = go.Figure()
    for ev in events_list:
        kick_lbl = f"T{int(ev['kick_id'])}"
        durs = [float(ev.get(k) or 0) for k in dur_keys]
        for phase, dur in zip(phases, durs):
            fig.add_trace(go.Bar(
                name=phase, x=[kick_lbl], y=[dur],
                legendgroup=phase, showlegend=(int(ev["kick_id"]) == 1),
            ))
    fig.update_layout(
        barmode="stack", height=300,
        title=dict(text=label, font=dict(color="#fafafa", size=17)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=13),
        xaxis=dict(gridcolor="#333", title="Tekme numarası"),
        yaxis=dict(gridcolor="#333", title="Süre (sn)"),
        margin=dict(l=65, r=20, t=55, b=55),
        legend=dict(orientation="h", y=-0.30),
    )
    st.plotly_chart(fig, use_container_width=True)
