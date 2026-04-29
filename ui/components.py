"""Reusable Streamlit UI components: video player, kick sections, phase bars, sensor stats."""
from __future__ import annotations

import http.server as _http_server
import socketserver as _socketserver
import threading as _threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components

_video_servers: dict[str, int] = {}


def get_video_server(directory: Path) -> int:
    """Start a range-capable HTTP server for the given directory (one per dir)."""
    key = str(directory.resolve())
    if key in _video_servers:
        return _video_servers[key]

    class _RangeHandler(_http_server.BaseHTTPRequestHandler):
        _root = directory.resolve()

        def log_message(self, *_): pass

        def do_GET(self):
            fname = self.path.lstrip("/").split("?")[0]
            fpath = self.__class__._root / fname
            if not fpath.exists() or not fpath.is_file():
                self.send_error(404); return
            size = fpath.stat().st_size
            rng  = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                parts = rng[6:].split("-")
                start = int(parts[0]) if parts[0] else 0
                end   = int(parts[1]) if len(parts) > 1 and parts[1] else size - 1
                end   = min(end, size - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type",   "video/mp4")
                self.send_header("Content-Range",  f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges",  "bytes")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(fpath, "rb") as f:
                    f.seek(start); self.wfile.write(f.read(length))
            else:
                self.send_response(200)
                self.send_header("Content-Type",   "video/mp4")
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges",  "bytes")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())

    server = _socketserver.ThreadingTCPServer(("127.0.0.1", 0), _RangeHandler)
    server.daemon_threads = True
    port = server.server_address[1]
    _threading.Thread(target=server.serve_forever, daemon=True).start()
    _video_servers[key] = port
    return port


def video_player(path: Path | str, start_time: float = 0.0, height: int = 480) -> None:
    """HTML5 video player with range support and seek-on-load."""
    p = Path(path)
    if not p.exists():
        st.warning("Video bulunamadı.")
        return
    port = get_video_server(p.parent)
    t_frag = f"#t={start_time:.3f}" if start_time > 0 else ""
    url = f"http://127.0.0.1:{port}/{p.name}{t_frag}"
    uid = abs(hash(str(p) + str(start_time))) % 999999

    html = f"""
<style>
  #w{uid}{{background:#000;line-height:0;position:relative}}
  #v{uid}{{width:100%;display:block;max-height:{height}px;cursor:pointer}}
  #v{uid}::-webkit-media-controls{{opacity:0;transition:opacity .2s}}
  #w{uid}:hover #v{uid}::-webkit-media-controls{{opacity:1}}
</style>
<div id="w{uid}">
  <video id="v{uid}" controls preload="auto">
    <source src="{url}" type="video/mp4">
  </video>
</div>
<script>
(function(){{
  var v = document.getElementById('v{uid}');
  var w = document.getElementById('w{uid}');
  var t = {start_time};
  if(t > 0){{
    v.addEventListener('loadedmetadata', function(){{ v.currentTime = t; }}, {{once:true}});
  }}
  if(!CSS.supports('-webkit-appearance','none')){{
    v.removeAttribute('controls');
    w.addEventListener('mouseenter',()=>v.setAttribute('controls',''));
    w.addEventListener('mouseleave',()=>v.removeAttribute('controls'));
  }}
}})();
</script>
"""
    _components.html(html, height=height + 8)


def trim_clip(src: Path, start: float, end: float, out: Path) -> bool:
    """Cut [start, end] seconds from src using OpenCV. Returns True on success."""
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(str(src))
        fps_v = cap.get(_cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = _cv2.VideoWriter_fourcc(*"avc1")
        writer = _cv2.VideoWriter(str(out), fourcc, fps_v, (w, h))
        if not writer.isOpened():
            fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
            writer = _cv2.VideoWriter(str(out), fourcc, fps_v, (w, h))
        pad = 0.4
        f_start = max(0, int((start - pad) * fps_v))
        f_end   = int((end + pad) * fps_v)
        cap.set(_cv2.CAP_PROP_POS_FRAMES, f_start)
        for _ in range(f_end - f_start + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        return out.exists() and out.stat().st_size > 1000
    except Exception:
        return False


_KICK_DISP_COLS = [
    "kick_id", "active_leg", "duration_sec", "active_knee_rom_deg",
    "active_peak_knee_vel_deg_s", "time_to_peak_knee_vel_sec",
    "peak_kick_height_norm", "active_peak_foot_speed_norm",
]


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

    ev_df = pd.DataFrame(events_list)
    st.dataframe(
        ev_df[[c for c in _KICK_DISP_COLS if c in ev_df.columns]].set_index("kick_id"),
        use_container_width=True,
    )

    vid_path = Path(video_path)
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
                clip_key  = f"clip_{label}_{kid}"
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
        barmode="stack", height=220,
        title=dict(text=label, font=dict(color="#fafafa", size=13)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333", title="Süre (sn)"),
        margin=dict(l=40, r=10, t=35, b=30),
        legend=dict(orientation="h", y=-0.35),
    )
    st.plotly_chart(fig, use_container_width=True)


def sensor_stats(emg_rows: list[dict], nirs_rows: list[dict], win: int) -> dict:
    """Compute summary stats for a single session's EMG + NIRS rows."""
    freq = [r["EMG_median_freq_Hz"] for r in emg_rows]
    smo2 = [r["SmO2"]              for r in nirs_rows]
    rms  = [r["EMG_RMS_mV"]        for r in emg_rows]
    return {
        "freq_start": float(np.mean(freq[:win])),
        "freq_end":   float(np.mean(freq[-win:])),
        "freq_arr":   freq,
        "smo2_start": float(np.mean(smo2[:win])),
        "smo2_end":   float(np.mean(smo2[-win:])),
        "smo2_min":   float(min(smo2)),
        "smo2_arr":   smo2,
        "rms_arr":    rms,
        "rms2_arr":   [r["EMG_CH2_RMS_mV"] for r in emg_rows],
        "thb_arr":    [r["THb"] for r in nirs_rows],
        "t_arr":      [r["time_sec"] for r in emg_rows],
        "t_nirs":     [r["time_sec"] for r in nirs_rows],
    }
