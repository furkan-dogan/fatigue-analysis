"""EMG Senkronizasyon sayfası."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.charts import kick_event_shapes


def render() -> None:
    st.title("EMG Senkronizasyon")
    st.caption(
        "EMG CSV + frame metrics CSV yükle → video zamanıyla hizala → tekme bazlı RMS hesapla."
    )

    st.sidebar.subheader("EMG Ayarları")
    emg_offset    = st.sidebar.number_input("EMG zaman offset (sn)", value=0.0, step=0.01,
                                             format="%.3f",
                                             help="EMG kaydı videodan geç başladıysa pozitif girin.")
    emg_delimiter = st.sidebar.selectbox("CSV delimiter", [",", ";", "\t"], index=0)
    emg_skip_rows = st.sidebar.number_input("Atlanacak başlık satırı", min_value=0, value=0, step=1)

    col_emg, col_frame, col_ev = st.columns(3)
    with col_emg:
        emg_file = st.file_uploader("EMG CSV", type=["csv", "txt"], key="emg_file")
    with col_frame:
        frame_file = st.file_uploader("Frame metrics CSV", type=["csv"], key="frame_file")
    with col_ev:
        ev_file = st.file_uploader("Kick events CSV", type=["csv"], key="ev_file")

    if emg_file and frame_file:
        try:
            from src.emg_sync import (
                compute_rms_per_kick,
                export_kick_emg_csv,
                export_synced_frame_csv,
                load_emg_csv,
                resample_to_video_times,
            )

            with tempfile.TemporaryDirectory() as tmp_str:
                tmp = Path(tmp_str)
                emg_path   = tmp / "emg.csv"
                frame_path = tmp / "frame.csv"
                emg_path.write_bytes(emg_file.read())
                frame_path.write_bytes(frame_file.read())

                frame_df = pd.read_csv(str(frame_path))
                emg_data = load_emg_csv(emg_path, delimiter=emg_delimiter,
                                        skip_rows=int(emg_skip_rows))

                st.success(
                    f"EMG yüklendi — {len(emg_data['channels'])} kanal, "
                    f"~{emg_data['sample_rate_hz']:.0f} Hz, "
                    f"{len(emg_data['time_s'])} sample"
                )

                all_channels = list(emg_data["channels"].keys())
                selected_channels = st.multiselect(
                    "Gösterilecek kanallar",
                    all_channels,
                    default=all_channels[:min(4, len(all_channels))],
                )

                video_times = frame_df["time_sec"].tolist() if "time_sec" in frame_df.columns else []
                if not video_times:
                    st.error("Frame CSV'de 'time_sec' sütunu bulunamadı.")
                    st.stop()

                emg_resampled = resample_to_video_times(emg_data, video_times,
                                                        emg_time_offset_sec=emg_offset)

                # ── EMG + diz açısı overlay ───────────────────────────────────
                st.subheader("EMG + Diz Açısı (senkron)")

                kick_events: list[dict] = []
                if ev_file:
                    ev_df = pd.read_csv(ev_file)
                    kick_events = ev_df.to_dict("records")

                shapes = kick_event_shapes(kick_events)
                fig_sync = go.Figure()

                for col, color in [("R_KNEE", "#ef4444"), ("L_KNEE", "#3b82f6")]:
                    if col in frame_df.columns:
                        fig_sync.add_trace(go.Scatter(
                            x=video_times, y=frame_df[col].tolist(),
                            name=f"{col} açısı (°)", yaxis="y2",
                            line=dict(color=color, width=1.5, dash="dot"), mode="lines",
                        ))

                for ch in selected_channels:
                    fig_sync.add_trace(go.Scatter(
                        x=video_times, y=emg_resampled.get(ch, []),
                        name=f"EMG: {ch}", mode="lines", line=dict(width=1),
                    ))

                fig_sync.update_layout(
                    height=380, xaxis_title="Zaman (sn)",
                    yaxis=dict(title="EMG (a.u.)", gridcolor="#333"),
                    yaxis2=dict(title="Açı (°)", overlaying="y", side="right", gridcolor="#444"),
                    shapes=shapes,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"),
                    margin=dict(l=50, r=60, t=30, b=40),
                    legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(fig_sync, use_container_width=True)

                # ── Per-kick RMS ──────────────────────────────────────────────
                if kick_events:
                    st.subheader("Tekme Bazlı EMG RMS")
                    kick_emg = compute_rms_per_kick(
                        {ch: emg_resampled[ch] for ch in selected_channels if ch in emg_resampled},
                        kick_events,
                        video_fps=30.0,
                    )
                    rms_df = pd.DataFrame(kick_emg).set_index("kick_id")
                    st.dataframe(rms_df, use_container_width=True)

                    rms_cols = [c for c in rms_df.columns if c.endswith("_rms")]
                    if rms_cols:
                        fig_rms = go.Figure()
                        for rc in rms_cols:
                            fig_rms.add_trace(go.Bar(
                                name=rc.replace("_rms", ""),
                                x=[f"Tekme {int(r)}" for r in rms_df.index],
                                y=rms_df[rc].tolist(),
                            ))
                        fig_rms.update_layout(
                            barmode="group", title="Tekme Bazlı RMS per Kanal",
                            height=320,
                            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font=dict(color="#fafafa"),
                            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(fig_rms, use_container_width=True)

                # ── Export ────────────────────────────────────────────────────
                st.subheader("Dışa Aktar")
                synced_path = tmp / "emg_synced_frames.csv"
                export_synced_frame_csv(
                    synced_path,
                    frame_df.to_dict("records"),
                    {ch: emg_resampled[ch] for ch in selected_channels if ch in emg_resampled},
                )
                st.download_button(
                    "📥 Frame-level senkron CSV indir",
                    synced_path.read_bytes(),
                    "emg_synced_frames.csv", "text/csv",
                )

                if kick_events:
                    kick_emg_path = tmp / "kick_emg_rms.csv"
                    export_kick_emg_csv(kick_emg_path, kick_emg)
                    st.download_button(
                        "📥 Tekme EMG RMS CSV indir",
                        kick_emg_path.read_bytes(),
                        "kick_emg_rms.csv", "text/csv",
                    )

        except Exception as exc:
            st.error(f"EMG işlem hatası: {exc}")
            st.exception(exc)

    else:
        st.info("EMG CSV ve Frame metrics CSV dosyalarını yükleyin.")
        st.subheader("Beklenen EMG CSV formatı")
        st.code(
            "time_s,bicep_femoris,rectus_femoris,gastrocnemius\n"
            "0.000,0.012,-0.003,0.008\n"
            "0.001,0.015, 0.001,0.010\n"
            "...",
            language="text",
        )
        st.markdown("""
**Desteklenen formatlar:**
- İlk sütun: zaman (saniye cinsinden — `time_s`, `time`, `t` gibi başlıklar otomatik tanınır)
- Kalan sütunlar: EMG kanalları (sayısal değerler)
- Noraxon / Delsys için fazladan metadata satırlarını *sidebar'dan* atlayabilirsiniz
""")
