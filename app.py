"""Streamlit dashboard for taekwondo fatigue analysis.

Run:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import base64
import csv
import math
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components

# Check optional deps once at startup
try:
    import ultralytics  # noqa: F401
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kickboks Yorgunluk Analizi",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🥊 Kickboks Analizi")
page = st.sidebar.radio(
    "Sayfa",
    ["Tek Video Analizi", "Çift Video Analizi", "EMG Sync"],
    label_visibility="collapsed",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

JOINT_COLOR = {
    "R_KNEE": "#ef4444",
    "L_KNEE": "#3b82f6",
    "R_HIP": "#f97316",
    "L_HIP": "#8b5cf6",
    "R_ANKLE": "#10b981",
    "L_ANKLE": "#06b6d4",
}

VEL_COLOR = {
    "R_KNEE_vel_deg_s": "#ef4444",
    "L_KNEE_vel_deg_s": "#3b82f6",
    "R_HIP_vel_deg_s": "#f97316",
    "L_HIP_vel_deg_s": "#8b5cf6",
}

FOOT_COLOR = {
    "R_FOOT_speed_norm": "#ef4444",
    "L_FOOT_speed_norm": "#3b82f6",
}


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _kick_event_lines(events: list[dict]) -> list[dict]:
    """Return plotly shape dicts for kick event start/end bands."""
    shapes = []
    for ev in events:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=float(ev.get("start_time_sec", 0)),
                x1=float(ev.get("end_time_sec", 0)),
                y0=0,
                y1=1,
                fillcolor="rgba(255,220,0,0.12)",
                line_width=0,
            )
        )
    return shapes


def _plot_time_series(df: pd.DataFrame, columns: list[str], colors: dict, events: list[dict], title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    shapes = _kick_event_lines(events)
    for col in columns:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["time_sec"],
                    y=df[col],
                    name=col,
                    line=dict(color=colors.get(col, "#aaa"), width=1.5),
                    mode="lines",
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Zaman (sn)",
        yaxis_title=y_label,
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.25),
        shapes=shapes,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    return fig


def _radar_chart(pre_vals: list[float], post_vals: list[float], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=pre_vals + [pre_vals[0]], theta=labels + [labels[0]], name="Pre", line=dict(color="#3b82f6", width=2), fill="toself", fillcolor="rgba(59,130,246,0.15)"))
    fig.add_trace(go.Scatterpolar(r=post_vals + [post_vals[0]], theta=labels + [labels[0]], name="Post", line=dict(color="#ef4444", width=2), fill="toself", fillcolor="rgba(239,68,68,0.15)"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, gridcolor="#333"), bgcolor="#0e1117"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        height=420,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def _bar_comparison(pre_vals: list[float], post_vals: list[float], labels: list[str], pct_changes: list[float | None]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Pre", x=labels, y=pre_vals, marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="Post", x=labels, y=post_vals, marker_color="#ef4444"))
    fig.update_layout(
        barmode="group",
        height=350,
        margin=dict(l=40, r=20, t=30, b=80),
        xaxis_tickangle=-35,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
        legend=dict(orientation="h", y=-0.4),
    )
    return fig


# ── Fatigue helpers (shared by pages 1 & 2) ───────────────────────────────────

# Metric definition: (label, unit, direction)
# direction=+1  → increase means fatigue  (duration, time_to_peak)
# direction=-1  → decrease means fatigue  (ROM, velocity, height)
FATIGUE_METRICS: dict[str, tuple[str, str, int]] = {
    "active_knee_rom_deg":         ("Aktif Diz ROM",          "°",      -1),
    "active_peak_knee_vel_deg_s":  ("Peak Diz Hızı",          "°/s",    -1),
    "active_mean_knee_vel_deg_s":  ("Ort. Diz Hızı",          "°/s",    -1),
    "time_to_peak_knee_vel_sec":   ("Peak Hıza Süre",         "sn",     +1),
    "peak_kick_height_norm":       ("Peak Tekme Yüksekliği",  "",       -1),
    "active_peak_foot_speed_norm": ("Peak Ayak Hızı",         "t/s",    -1),
    "duration_sec":                ("Tekme Süresi",           "sn",     +1),
    "R_HIP_rom":                   ("Sağ Kalça ROM",          "°",      -1),
    "L_HIP_rom":                   ("Sol Kalça ROM",          "°",      -1),
    "R_ANKLE_rom":                 ("Sağ Ayak Bileği ROM",    "°",      -1),
}

FATIGUE_WEIGHTS: dict[str, float] = {
    "active_knee_rom_deg":         0.25,
    "active_peak_knee_vel_deg_s":  0.25,
    "time_to_peak_knee_vel_sec":   0.15,
    "peak_kick_height_norm":       0.15,
    "active_peak_foot_speed_norm": 0.10,
    "duration_sec":                0.05,
    "active_mean_knee_vel_deg_s":  0.05,
}


def _events_mean(events: list[dict], col: str) -> float | None:
    vals = [float(e[col]) for e in events if e.get(col) is not None]
    return sum(vals) / len(vals) if vals else None


def _compute_fatigue(pre_events: list[dict], post_events: list[dict]) -> dict:
    """Return per-metric deltas + composite fatigue index (0–100)."""
    results = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for key, (label, unit, direction) in FATIGUE_METRICS.items():
        pre_v = _events_mean(pre_events, key)
        post_v = _events_mean(post_events, key)
        if pre_v is None or post_v is None or abs(pre_v) < 1e-8:
            results[key] = dict(label=label, unit=unit, direction=direction,
                                pre=pre_v, post=post_v, delta=None, pct=None, fatigue_contribution=None)
            continue

        delta = post_v - pre_v
        pct = delta / abs(pre_v) * 100.0
        # Contribution: positive = fatigue signal, clamped to [-100, 100]
        contribution = float(direction * pct)
        contribution_clamped = max(-100.0, min(100.0, contribution))

        w = FATIGUE_WEIGHTS.get(key, 0.0)
        if w > 0:
            weighted_sum += contribution_clamped * w
            weight_total += w

        results[key] = dict(label=label, unit=unit, direction=direction,
                            pre=pre_v, post=post_v, delta=delta, pct=pct,
                            fatigue_contribution=contribution_clamped)

    composite = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    # Normalize to 0–100: composite=-100 → no fatigue, composite=+100 → max fatigue
    fatigue_index = max(0.0, min(100.0, (composite + 100.0) / 2.0))
    return {"metrics": results, "fatigue_index": fatigue_index, "composite_raw": composite}


import http.server as _http_server
import socketserver as _socketserver
import threading as _threading

_video_servers: dict[str, int] = {}  # dir → port


def _get_video_server(directory: Path) -> int:
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
                self.send_header("Content-Type",  "video/mp4")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
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


def _video_player(path: Path | str, start_time: float = 0.0, height: int = 480) -> None:
    """Proper HTML5 video player with range support. Controls show only on hover."""
    p = Path(path)
    if not p.exists():
        st.warning("Video bulunamadı.")
        return
    port = _get_video_server(p.parent)
    url  = f"http://127.0.0.1:{port}/{p.name}"
    uid  = abs(hash(str(p))) % 999999

    html = f"""
<style>
  #w{uid}{{background:#000;line-height:0;position:relative}}
  #v{uid}{{width:100%;display:block;max-height:{height}px;cursor:pointer}}
  #v{uid}::-webkit-media-controls{{opacity:0;transition:opacity .2s}}
  #w{uid}:hover #v{uid}::-webkit-media-controls{{opacity:1}}
</style>
<div id="w{uid}">
  <video id="v{uid}" controls preload="metadata">
    <source src="{url}" type="video/mp4">
  </video>
</div>
<script>
(function(){{
  var v = document.getElementById('v{uid}');
  var w = document.getElementById('w{uid}');
  v.currentTime = {start_time};
  // fallback for non-webkit: toggle controls on hover
  if(!CSS.supports('-webkit-appearance','none')){{
    v.removeAttribute('controls');
    w.addEventListener('mouseenter',()=>v.setAttribute('controls',''));
    w.addEventListener('mouseleave',()=>v.removeAttribute('controls'));
  }}
}})();
</script>
"""
    _components.html(html, height=height + 8)


def _gauge(value: float, title: str) -> go.Figure:
    color = "#22c55e" if value < 33 else ("#f59e0b" if value < 66 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        title={"text": title, "font": {"color": "#fafafa", "size": 14}},
        number={"suffix": "/100", "font": {"color": "#fafafa", "size": 22}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#555", tickfont=dict(color="#aaa")),
            bar=dict(color=color),
            bgcolor="#1e2030",
            steps=[
                dict(range=[0, 33], color="#1a2a1a"),
                dict(range=[33, 66], color="#2a2a1a"),
                dict(range=[66, 100], color="#2a1a1a"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=value),
        ),
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
    )
    return fig


def _overlay_angle_chart(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    column: str,
    label: str,
    pre_events: list[dict],
    post_events: list[dict],
) -> go.Figure:
    fig = go.Figure()
    # kick bands
    for ev in pre_events:
        fig.add_vrect(x0=float(ev["start_time_sec"]), x1=float(ev["end_time_sec"]),
                      fillcolor="rgba(59,130,246,0.08)", line_width=0)
    for ev in post_events:
        fig.add_vrect(x0=float(ev["start_time_sec"]), x1=float(ev["end_time_sec"]),
                      fillcolor="rgba(239,68,68,0.08)", line_width=0)
    if column in pre_df.columns:
        fig.add_trace(go.Scatter(x=pre_df["time_sec"], y=pre_df[column],
                                  name=f"Pre — {label}", mode="lines",
                                  line=dict(color="#3b82f6", width=1.5, dash="dot")))
    if column in post_df.columns:
        fig.add_trace(go.Scatter(x=post_df["time_sec"], y=post_df[column],
                                  name=f"Post — {label}", mode="lines",
                                  line=dict(color="#ef4444", width=1.5)))
    fig.update_layout(
        height=240, xaxis_title="Zaman (sn)", yaxis_title="Açı (°)" if "vel" not in column else "Hız (°/s)",
        margin=dict(l=50, r=20, t=30, b=35),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        legend=dict(orientation="h", y=-0.35, font=dict(size=11)),
        title=dict(text=label, font=dict(color="#fafafa", size=13)),
    )
    return fig


def _per_kick_trend(pre_events: list[dict], post_events: list[dict], col: str, label: str) -> go.Figure:
    """Bar chart showing metric per kick — side by side pre/post."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Pre",
        x=[f"T{int(e['kick_id'])}" for e in pre_events],
        y=[float(e[col]) if e.get(col) is not None else 0 for e in pre_events],
        marker_color="#3b82f6",
    ))
    fig.add_trace(go.Bar(
        name="Post",
        x=[f"T{int(e['kick_id'])}" for e in post_events],
        y=[float(e[col]) if e.get(col) is not None else 0 for e in post_events],
        marker_color="#ef4444",
    ))
    fig.update_layout(
        barmode="group", height=240, title=dict(text=label, font=dict(color="#fafafa", size=13)),
        margin=dict(l=40, r=10, t=35, b=35),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
        legend=dict(orientation="h", y=-0.35, font=dict(size=11)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Tek Video Analizi
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Tek Video Analizi":
    st.title("Tek Video Analizi")
    st.caption("Video yükle → analiz çalıştır → eklem açıları, hız ve tekme eventlerini incele.")

    # ── Sidebar params ────────────────────────────────────────────────────────
    st.sidebar.subheader("Parametreler")
    _backend_opts = ["mediapipe", "yolo"] if _YOLO_AVAILABLE else ["mediapipe"]
    backend = st.sidebar.selectbox(
        "Pose backend",
        _backend_opts,
        index=0,
        help="MediaPipe: hızlı, 33 nokta, kurulumu kolay.\n"
             "YOLO: 17 COCO nokta, hızlı tekmelerde daha kararlı tracking — "
             "ilk çalıştırmada model (~6 MB) indirilir."
             + ("" if _YOLO_AVAILABLE else "\n\n⚠️ YOLO için: pip install ultralytics"),
    )
    yolo_model = st.sidebar.selectbox(
        "YOLO model",
        ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolov8n-pose.pt", "yolov8s-pose.pt"],
        index=0,
        help="n=nano (en hızlı), s=small (daha doğru). İlk kullanımda indirilir.",
        disabled=(backend != "yolo"),
    )
    show_labels = st.sidebar.checkbox(
        "Eklem etiketleri", value=False,
        help="Annotated videoda her eklemin üstüne kısa isim yazar (R_KNE, L_ANK vb.). "
             "Pose'un doğru takip edilip edilmediğini görsel olarak kontrol etmek için açın.",
    )
    prominence = st.sidebar.slider(
        "Event prominence (norm)", 0.02, 0.20, 0.06, 0.01,
        help="Ayağın baseline'ın ne kadar üstüne çıkınca 'tekme başladı' sayılsın (normalize torso birimi). "
             "Düşürürsen küçük hareketler de event olur; artırırsan yalnızca net yüksek tekmeler yakalanır.",
    )
    min_dist = st.sidebar.slider(
        "Min peak mesafe (sn)", 0.1, 1.5, 0.25, 0.05,
        help="İki ayrı tekme arasındaki minimum süre. "
             "Hızlı kombinasyon varsa düşür (0.1 sn); tek tekme alıştırması ise yüksek tut.",
    )
    min_dur = st.sidebar.slider(
        "Min event süresi (sn)", 0.05, 0.5, 0.10, 0.05,
        help="Bu süreden kısa hareketler tekme sayılmaz. "
             "Çok kısa titremeleri veya anlık sarsılmaları eler.",
    )
    max_dur = st.sidebar.slider(
        "Max event süresi (sn)", 1.0, 10.0, 6.0, 0.5,
        help="Bu süreden uzun hareketler tekme sayılmaz. "
             "Uzun duruş değişikliklerini veya bozuk pose tespitlerini dışarıda bırakır.",
    )

    st.sidebar.subheader("Kick Doğrulama Filtreleri")
    min_knee_rom = st.sidebar.slider(
        "Min diz ROM (°)", 0, 60, 12, 5,
        help="Tekme sayılması için dizin en az bu kadar açılıp kapanması gerekir.\n\n"
             "Gerçek tekmelerde ROM genellikle 60–120°, weight-shift'te ise 3–5°.\n\n"
             "20° varsayılanı sahte tespitlerin neredeyse tamamını eler.",
    )
    min_peak_height = st.sidebar.slider(
        "Min peak yüksekliği (norm)", -1.0, 0.5, -0.5, 0.05,
        help="Tekme anında ayağın ulaşması gereken minimum yükseklik (torso uzunluğuna normalize).\n\n"
             "0.0 = kalça hizası (orta-yüksek tekme)\n"
             "−0.3 = kalçanın biraz altı (düşük tekme)\n"
             "−1.0 = neredeyse yerde (filtre kapalı)\n\n"
             "Atılan tekme alçaksa −0.5'e çekin.",
    )
    vel_assist = st.sidebar.slider(
        "Hız yardımı eşiği (°/s)", 50, 500, 100, 25,
        help="Diz açısal hızı bu değeri geçen anlarda tekme adayı oluşturulur.\n\n"
             "Hızlı tekmeleri yakalayan ikincil sinyal — ayak yüksekliği sinyali kaçırdığında devreye girer.\n\n"
             "Düşürürsen çok hassas olur (duruş değişiklikleri de yakalanabilir).\n"
             "Artırırsan sadece patlayıcı hızlı tekmelerde aktif olur.",
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Video seç (MP4/AVI)", type=["mp4", "avi", "mov"])
    run_btn = st.button("▶ Analizi Çalıştır", disabled=uploaded is None, type="primary")

    if run_btn and uploaded is not None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            input_path = tmp / "input.mp4"
            output_path = tmp / "annotated.mp4"
            frame_csv = tmp / "frame_metrics.csv"
            events_csv = tmp / "kick_events.csv"

            input_path.write_bytes(uploaded.read())

            progress_bar = st.progress(0, text="Pose analizi yapılıyor…")

            def _progress(cur: int, total: int) -> None:
                pct = int(min(cur / max(total, 1), 1.0) * 100)
                progress_bar.progress(pct, text=f"Frame {cur}/{total} işleniyor…")

            try:
                from src.pipeline import run_analysis
                result = run_analysis(
                    input_path=input_path,
                    output_path=output_path,
                    frame_csv_path=frame_csv,
                    events_csv_path=events_csv,
                    show_joint_labels=show_labels,
                    event_peak_prominence_norm=prominence,
                    event_min_distance_sec=min_dist,
                    event_min_duration_sec=min_dur,
                    event_max_duration_sec=max_dur,
                    event_min_knee_rom_deg=float(min_knee_rom),
                    event_min_peak_kick_height_norm=float(min_peak_height),
                    progress_callback=_progress,
                    backend=backend,
                    yolo_model=yolo_model,
                    vel_assist_threshold=float(vel_assist),
                )
                progress_bar.progress(100, text="Tamamlandı!")
            except Exception as exc:
                st.error(f"Hata: {exc}")
                st.stop()

            df = _read_csv(str(frame_csv))
            events = result.events

            # ── Summary metrics ───────────────────────────────────────────────
            st.subheader("Özet")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Toplam Frame", result.total_frames)
            c2.metric("FPS", f"{result.fps:.1f}")
            c3.metric("Tespit Edilen Tekme", len(events))
            if events:
                mean_dur = sum(float(e["duration_sec"]) for e in events) / len(events)
                c4.metric("Ort. Tekme Süresi", f"{mean_dur:.2f} sn")
                peak_vels = [float(e["active_peak_knee_vel_deg_s"]) for e in events if e.get("active_peak_knee_vel_deg_s") is not None]
                c5.metric("Ort. Peak Diz Hızı", f"{sum(peak_vels)/len(peak_vels):.0f} °/s" if peak_vels else "N/A")

            # ── Tabs ──────────────────────────────────────────────────────────
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📹 Annotated Video", "📈 Açı Grafikleri", "⚡ Hız Grafikleri", "📋 Tekme Eventleri", "🔍 Kick İnceleme"])

            with tab1:
                if output_path.exists():
                    _video_player(output_path)
                else:
                    st.warning("Video çıktısı oluşturulamadı.")

            with tab2:
                if not df.empty:
                    fig_angles = _plot_time_series(
                        df,
                        ["R_KNEE", "L_KNEE", "R_HIP", "L_HIP", "R_ANKLE", "L_ANKLE"],
                        JOINT_COLOR,
                        events,
                        "Eklem Açıları",
                        "Açı (°)",
                    )
                    st.plotly_chart(fig_angles, use_container_width=True)

                    fig_height = _plot_time_series(
                        df,
                        ["R_KICK_HEIGHT", "L_KICK_HEIGHT", "KICK_HEIGHT_ACTIVE"],
                        {"R_KICK_HEIGHT": "#ef4444", "L_KICK_HEIGHT": "#3b82f6", "KICK_HEIGHT_ACTIVE": "#fbbf24"},
                        events,
                        "Normalize Tekme Yüksekliği",
                        "Yükseklik (torso uzunluğu)",
                    )
                    st.plotly_chart(fig_height, use_container_width=True)
                else:
                    st.warning("Frame verisi bulunamadı.")

            with tab3:
                if not df.empty:
                    fig_vel = _plot_time_series(
                        df,
                        ["R_KNEE_vel_deg_s", "L_KNEE_vel_deg_s", "R_HIP_vel_deg_s", "L_HIP_vel_deg_s"],
                        VEL_COLOR,
                        events,
                        "Açısal Hız",
                        "Hız (°/sn)",
                    )
                    st.plotly_chart(fig_vel, use_container_width=True)

                    fig_foot = _plot_time_series(
                        df,
                        ["R_FOOT_speed_norm", "L_FOOT_speed_norm"],
                        FOOT_COLOR,
                        events,
                        "Ayak Hızı (normalize)",
                        "Hız (torso/sn)",
                    )
                    st.plotly_chart(fig_foot, use_container_width=True)
                else:
                    st.warning("Hız verisi bulunamadı.")

            with tab4:
                if events:
                    display_cols = [
                        "kick_id", "active_leg", "start_time_sec", "peak_time_sec", "end_time_sec",
                        "duration_sec", "peak_kick_height_norm",
                        "active_peak_knee_angle_deg", "active_knee_rom_deg",
                        "active_peak_knee_vel_deg_s", "active_mean_knee_vel_deg_s",
                        "time_to_peak_knee_vel_sec",
                        "active_peak_foot_speed_norm",
                    ]
                    ev_df = pd.DataFrame(events)
                    show_cols = [c for c in display_cols if c in ev_df.columns]
                    st.dataframe(ev_df[show_cols].set_index("kick_id"), use_container_width=True)

                    # Per-kick velocity bar chart
                    if "active_peak_knee_vel_deg_s" in ev_df.columns:
                        fig_k = go.Figure()
                        fig_k.add_trace(go.Bar(
                            x=[f"Tekme {int(e['kick_id'])}" for e in events],
                            y=[float(e.get("active_peak_knee_vel_deg_s") or 0) for e in events],
                            marker_color="#ef4444",
                            name="Peak Diz Hızı (°/s)",
                        ))
                        fig_k.add_trace(go.Bar(
                            x=[f"Tekme {int(e['kick_id'])}" for e in events],
                            y=[float(e.get("active_knee_rom_deg") or 0) for e in events],
                            marker_color="#3b82f6",
                            name="Diz ROM (°)",
                        ))
                        fig_k.update_layout(
                            barmode="group", height=300,
                            title="Tekme Bazlı: Peak Hız vs ROM",
                            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font=dict(color="#fafafa"),
                            xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
                            margin=dict(l=40, r=20, t=40, b=40),
                            legend=dict(orientation="h", y=-0.3),
                        )
                        st.plotly_chart(fig_k, use_container_width=True)

                    # CSV indirme
                    ev_csv_bytes = ev_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Events CSV indir", ev_csv_bytes, "kick_events.csv", "text/csv")
                else:
                    st.info("Tekme eventi tespit edilemedi. Parametreleri ayarlayın.")

            with tab5:
                st.subheader("Kick İnceleme — her tekme ayrı ayrı")
                if not df.empty and events:
                    for ev in events:
                        kid = int(ev["kick_id"])
                        t_start = float(ev["start_time_sec"])
                        t_end = float(ev["end_time_sec"])
                        t_peak = float(ev["peak_time_sec"])
                        dur = float(ev["duration_sec"])
                        leg = ev.get("active_leg", "?")
                        rom = ev.get("active_knee_rom_deg")
                        height = ev.get("peak_kick_height_norm")
                        peak_vel = ev.get("active_peak_knee_vel_deg_s")
                        ttp = ev.get("time_to_peak_knee_vel_sec")

                        with st.expander(
                            f"Tekme {kid}  |  {leg} bacak  |  {t_start:.2f}–{t_end:.2f}s  |  ROM {float(rom):.1f}°  |  Height {float(height):.3f}" if rom and height else f"Tekme {kid}",
                            expanded=(kid == 1),
                        ):
                            # Mini stats
                            mc1, mc2, mc3, mc4 = st.columns(4)
                            mc1.metric("Aktif Bacak", leg)
                            mc2.metric("Süre", f"{dur:.3f} sn")
                            mc3.metric("Diz ROM", f"{float(rom):.1f}°" if rom is not None else "—")
                            mc4.metric("Peak Yükseklik", f"{float(height):.3f}" if height is not None else "—")
                            mc1b, mc2b, mc3b, mc4b = st.columns(4)
                            mc1b.metric("Peak Diz Hızı", f"{float(peak_vel):.0f} °/s" if peak_vel is not None else "—")
                            mc2b.metric("Peak Hıza Süre", f"{float(ttp):.3f} sn" if ttp is not None else "—")
                            mc3b.metric("R_HIP ROM", f"{float(ev['R_HIP_rom']):.1f}°" if ev.get('R_HIP_rom') is not None else "—")
                            mc4b.metric("L_HIP ROM", f"{float(ev['L_HIP_rom']):.1f}°" if ev.get('L_HIP_rom') is not None else "—")

                            # Zoom in on kick window with padding
                            pad = 0.5
                            mask = (df["time_sec"] >= max(0, t_start - pad)) & (df["time_sec"] <= t_end + pad)
                            df_zoom = df[mask].copy()

                            knee_col = "R_KNEE" if leg == "R" else "L_KNEE"
                            hip_col = "R_HIP" if leg == "R" else "L_HIP"
                            height_col = "R_KICK_HEIGHT" if leg == "R" else "L_KICK_HEIGHT"
                            vel_col = f"{knee_col}_vel_deg_s"

                            fig_kick = go.Figure()
                            for col, color, name in [
                                (knee_col, "#ef4444", f"{knee_col} açısı (°)"),
                                (hip_col, "#f97316", f"{hip_col} açısı (°)"),
                            ]:
                                if col in df_zoom.columns:
                                    fig_kick.add_trace(go.Scatter(
                                        x=df_zoom["time_sec"], y=df_zoom[col],
                                        name=name, line=dict(color=color, width=2), mode="lines",
                                    ))
                            if height_col in df_zoom.columns:
                                fig_kick.add_trace(go.Scatter(
                                    x=df_zoom["time_sec"], y=df_zoom[height_col],
                                    name=f"Ayak yüksekliği ({leg})", line=dict(color="#fbbf24", width=1.5, dash="dot"),
                                    yaxis="y2", mode="lines",
                                ))
                            if vel_col in df_zoom.columns:
                                fig_kick.add_trace(go.Scatter(
                                    x=df_zoom["time_sec"], y=df_zoom[vel_col],
                                    name="Açısal hız (°/s)", line=dict(color="#a78bfa", width=1.5),
                                    yaxis="y3", mode="lines",
                                ))
                            # Event window band
                            fig_kick.add_vrect(x0=t_start, x1=t_end, fillcolor="rgba(255,220,0,0.15)", line_width=0)
                            # Peak marker
                            fig_kick.add_vline(x=t_peak, line_dash="dash", line_color="#fbbf24", line_width=1.5)

                            fig_kick.update_layout(
                                height=280,
                                margin=dict(l=50, r=80, t=15, b=30),
                                xaxis_title="Zaman (sn)",
                                yaxis=dict(title="Açı (°)", gridcolor="#333"),
                                yaxis2=dict(title="Yükseklik", overlaying="y", side="right", gridcolor="#333", showgrid=False),
                                yaxis3=dict(title="Hız (°/s)", overlaying="y", side="right", anchor="free", position=1.0, showgrid=False),
                                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                font=dict(color="#fafafa"),
                                legend=dict(orientation="h", y=-0.4, font=dict(size=10)),
                            )

                            vcol, gcol = st.columns([1, 1])
                            with gcol:
                                st.plotly_chart(fig_kick, use_container_width=True)
                            with vcol:
                                clip_path = tmp / f"clip_{kid}.mp4"
                                if not clip_path.exists():
                                    import cv2 as _cv2
                                    try:
                                        cap_c = _cv2.VideoCapture(str(output_path))
                                        fps_c = cap_c.get(_cv2.CAP_PROP_FPS) or 30.0
                                        w_c = int(cap_c.get(_cv2.CAP_PROP_FRAME_WIDTH))
                                        h_c = int(cap_c.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                                        fcc = _cv2.VideoWriter_fourcc(*"avc1")
                                        wri_c = _cv2.VideoWriter(str(clip_path), fcc, fps_c, (w_c, h_c))
                                        if not wri_c.isOpened():
                                            fcc = _cv2.VideoWriter_fourcc(*"mp4v")
                                            wri_c = _cv2.VideoWriter(str(clip_path), fcc, fps_c, (w_c, h_c))
                                        pad_c = 0.4
                                        f0 = max(0, int((t_start - pad_c) * fps_c))
                                        f1 = int((t_end + pad_c) * fps_c)
                                        cap_c.set(_cv2.CAP_PROP_POS_FRAMES, f0)
                                        for _ in range(f1 - f0 + 1):
                                            ok_c, fr_c = cap_c.read()
                                            if not ok_c:
                                                break
                                            wri_c.write(fr_c)
                                        cap_c.release()
                                        wri_c.release()
                                    except Exception:
                                        pass
                                if clip_path.exists() and clip_path.stat().st_size > 1000:
                                    _video_player(clip_path)
                                else:
                                    _video_player(output_path, start_time=t_start)
                elif df.empty:
                    st.warning("Önce analiz çalıştırın.")
                else:
                    st.info("Tekme eventi tespit edilemedi.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Çift Video Analizi
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Çift Video Analizi":
    st.title("Çift Video Analizi — Yorgunluk Değerlendirmesi")
    st.caption("Pre ve post antrenman videolarını yükle, analiz et, açısal yorgunluk metriklerini karşılaştır.")

    # ── Sidebar params ────────────────────────────────────────────────────────
    st.sidebar.subheader("Analiz Parametreleri")
    dv_backend = st.sidebar.selectbox(
        "Pose backend",
        ["mediapipe", "yolo"] if _YOLO_AVAILABLE else ["mediapipe"],
        index=0,
        key="dv_backend",
        help="MediaPipe: hızlı, 33 nokta, kurulumu kolay.\n"
             "YOLO: 17 COCO nokta, hızlı tekmelerde daha kararlı tracking — "
             "ilk çalıştırmada model (~6 MB) indirilir."
             + ("" if _YOLO_AVAILABLE else "\n\n⚠️ YOLO için: pip install ultralytics"),
    )
    dv_yolo_model = st.sidebar.selectbox(
        "YOLO model",
        ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolov8n-pose.pt", "yolov8s-pose.pt"],
        index=0,
        key="dv_yolo_model",
        help="n=nano (en hızlı), s=small (daha doğru). İlk kullanımda indirilir.",
        disabled=(dv_backend != "yolo"),
    )
    dv_show_labels = st.sidebar.checkbox(
        "Eklem etiketleri", value=False, key="dv_labels",
        help="Annotated videoda her eklemin üstüne kısa isim yazar (R_KNE, L_ANK vb.). "
             "Pose'un doğru takip edilip edilmediğini görsel olarak kontrol etmek için açın.",
    )
    dv_prominence = st.sidebar.slider(
        "Event prominence", 0.02, 0.20, 0.06, 0.01, key="dv_prom",
        help="Ayağın baseline'ın ne kadar üstüne çıkınca 'tekme başladı' sayılsın (normalize torso birimi). "
             "Düşürürsen küçük hareketler de event olur; artırırsan yalnızca net yüksek tekmeler yakalanır.",
    )
    dv_min_dist = st.sidebar.slider(
        "Min peak mesafe (sn)", 0.1, 1.5, 0.25, 0.05, key="dv_dist",
        help="İki ayrı tekme arasındaki minimum süre. "
             "Hızlı kombinasyon varsa düşür (0.1 sn); tek tekme alıştırması ise yüksek tut.",
    )
    dv_min_dur = st.sidebar.slider(
        "Min event süresi (sn)", 0.05, 0.5, 0.10, 0.05, key="dv_dur",
        help="Bu süreden kısa hareketler tekme sayılmaz. "
             "Çok kısa titremeleri veya anlık sarsılmaları eler.",
    )
    dv_max_dur = st.sidebar.slider(
        "Max event süresi (sn)", 1.0, 10.0, 6.0, 0.5, key="dv_maxdur",
        help="Bu süreden uzun hareketler tekme sayılmaz. "
             "Uzun duruş değişikliklerini veya bozuk pose tespitlerini dışarıda bırakır.",
    )
    st.sidebar.subheader("Kick Doğrulama")
    dv_min_rom = st.sidebar.slider(
        "Min diz ROM (°)", 0, 60, 12, 5, key="dv_rom",
        help="Tekme sayılması için dizin en az bu kadar açılıp kapanması gerekir.\n\n"
             "Gerçek tekmelerde ROM genellikle 60–120°, weight-shift'te ise 3–5°.\n\n"
             "20° varsayılanı sahte tespitlerin neredeyse tamamını eler.",
    )
    dv_min_height = st.sidebar.slider(
        "Min peak yükseklik", -1.0, 0.5, -0.5, 0.05, key="dv_height",
        help="Tekme anında ayağın ulaşması gereken minimum yükseklik (torso uzunluğuna normalize).\n\n"
             "0.0 = kalça hizası (orta-yüksek tekme)\n"
             "−0.3 = kalçanın biraz altı (düşük tekme)\n"
             "−1.0 = neredeyse yerde (filtre kapalı)\n\n"
             "Atılan tekme alçaksa −0.5'e çekin.",
    )
    dv_vel_assist = st.sidebar.slider(
        "Hız yardımı eşiği (°/s)", 50, 500, 100, 25, key="dv_vel_assist",
        help="Diz açısal hızı bu değeri geçen anlarda tekme adayı oluşturulur.\n\n"
             "Hızlı tekmeleri yakalayan ikincil sinyal — ayak yüksekliği sinyali kaçırdığında devreye girer.",
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    col_up_pre, col_up_post = st.columns(2)
    with col_up_pre:
        st.markdown("#### Pre-antrenman")
        pre_upload = st.file_uploader("Pre video", type=["mp4", "avi", "mov"], key="dv_pre")
    with col_up_post:
        st.markdown("#### Post-antrenman")
        post_upload = st.file_uploader("Post video", type=["mp4", "avi", "mov"], key="dv_post")

    both_ready = pre_upload is not None and post_upload is not None
    run_dual = st.button("▶ Her İkisini Analiz Et", disabled=not both_ready, type="primary")

    # ── Session state ─────────────────────────────────────────────────────────
    if "dv_pre_result" not in st.session_state:
        st.session_state["dv_pre_result"] = None
        st.session_state["dv_post_result"] = None
        st.session_state["dv_pre_df"] = None
        st.session_state["dv_post_df"] = None
        st.session_state["dv_tmp"] = None

    if run_dual and both_ready:
        from src.pipeline import run_analysis

        tmp_dir = tempfile.mkdtemp()
        tmp = Path(tmp_dir)

        pre_in   = tmp / "pre_input.mp4"
        post_in  = tmp / "post_input.mp4"
        pre_out  = tmp / "pre_annotated.mp4"
        post_out = tmp / "post_annotated.mp4"
        pre_fcsv = tmp / "pre_frames.csv"
        post_fcsv = tmp / "post_frames.csv"
        pre_ecsv = tmp / "pre_events.csv"
        post_ecsv = tmp / "post_events.csv"

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
                backend=dv_backend,
                yolo_model=dv_yolo_model,
                vel_assist_threshold=float(dv_vel_assist),
            )
            pre_res  = run_analysis(pre_in,  pre_out,  pre_fcsv,  pre_ecsv,  progress_callback=_prog_pre,  **kw)
            post_res = run_analysis(post_in, post_out, post_fcsv, post_ecsv, progress_callback=_prog_post, **kw)
            prog.progress(100, text="Tamamlandı!")
        except Exception as exc:
            st.error(f"Analiz hatası: {exc}")
            st.stop()

        st.session_state["dv_pre_result"]  = pre_res
        st.session_state["dv_post_result"] = post_res
        st.session_state["dv_pre_df"]      = _read_csv(str(pre_fcsv))
        st.session_state["dv_post_df"]     = _read_csv(str(post_fcsv))
        st.session_state["dv_tmp"]         = tmp

    pre_res  = st.session_state.get("dv_pre_result")
    post_res = st.session_state.get("dv_post_result")
    pre_df   = st.session_state.get("dv_pre_df",  pd.DataFrame())
    post_df  = st.session_state.get("dv_post_df", pd.DataFrame())

    if pre_res is None or post_res is None:
        st.info("Her iki videoyu yükleyip analiz başlatın.")
        st.stop()

    pre_events  = pre_res.events
    post_events = post_res.events

    # ── Summary header ────────────────────────────────────────────────────────
    fatigue_data = _compute_fatigue(pre_events, post_events)
    fi = fatigue_data["fatigue_index"]

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Pre — Tespit Edilen Tekme", len(pre_events))
    h1.metric("Pre — Toplam Frame", pre_res.total_frames)
    h2.metric("Post — Tespit Edilen Tekme", len(post_events))
    h2.metric("Post — Toplam Frame", post_res.total_frames)

    pre_mean_vel  = _events_mean(pre_events,  "active_peak_knee_vel_deg_s")
    post_mean_vel = _events_mean(post_events, "active_peak_knee_vel_deg_s")
    if pre_mean_vel and post_mean_vel:
        vel_delta = post_mean_vel - pre_mean_vel
        h3.metric("Ort. Peak Diz Hızı — Pre",  f"{pre_mean_vel:.0f} °/s")
        h3.metric("Ort. Peak Diz Hızı — Post", f"{post_mean_vel:.0f} °/s", delta=f"{vel_delta:+.0f} °/s", delta_color="inverse")

    pre_mean_rom  = _events_mean(pre_events,  "active_knee_rom_deg")
    post_mean_rom = _events_mean(post_events, "active_knee_rom_deg")
    if pre_mean_rom and post_mean_rom:
        rom_delta = post_mean_rom - pre_mean_rom
        h4.metric("Ort. Diz ROM — Pre",  f"{pre_mean_rom:.1f}°")
        h4.metric("Ort. Diz ROM — Post", f"{post_mean_rom:.1f}°", delta=f"{rom_delta:+.1f}°", delta_color="inverse")

    fi_label = "Düşük" if fi < 33 else ("Orta" if fi < 66 else "Yüksek")
    h5.metric("Yorgunluk İndeksi", f"{fi:.1f}/100", delta=fi_label, delta_color="off")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["📹 Videolar", "📐 Açı Karşılaştırma", "⚡ Hız Karşılaştırma",
                    "🔥 Yorgunluk Analizi", "📊 Tekme Bazlı", "🏃 Faz Analizi",
                    "📏 Asimetri", "🔬 İstatistik", "💾 Export"])

    # ── Tab 0: Videos ─────────────────────────────────────────────────────────
    with tabs[0]:
        vc1, vc2 = st.columns(2)
        with vc1:
            st.markdown("**Pre-antrenman (annotated)**")
            pre_vid = Path(pre_res.output_video_path)
            if pre_vid.exists():
                _video_player(pre_vid)
            else:
                st.warning("Pre video çıktısı bulunamadı.")
        with vc2:
            st.markdown("**Post-antrenman (annotated)**")
            post_vid = Path(post_res.output_video_path)
            if post_vid.exists():
                _video_player(post_vid)
            else:
                st.warning("Post video çıktısı bulunamadı.")

    # ── Tab 1: Angle comparison ───────────────────────────────────────────────
    with tabs[1]:
        st.caption("Mavi kesikli = Pre  |  Kırmızı düz = Post  |  Renkli bantlar = tekme eventleri")
        angle_pairs = [
            ("R_KNEE", "Sağ Diz Açısı"),
            ("L_KNEE", "Sol Diz Açısı"),
            ("R_HIP",  "Sağ Kalça Açısı"),
            ("L_HIP",  "Sol Kalça Açısı"),
            ("R_ANKLE","Sağ Ayak Bileği Açısı"),
            ("L_ANKLE","Sol Ayak Bileği Açısı"),
        ]
        for col_name, lbl in angle_pairs:
            if col_name in pre_df.columns or col_name in post_df.columns:
                st.plotly_chart(
                    _overlay_angle_chart(pre_df, post_df, col_name, lbl, pre_events, post_events),
                    use_container_width=True,
                )

    # ── Tab 2: Velocity comparison ────────────────────────────────────────────
    with tabs[2]:
        st.caption("Açısal hız ve ayak hızı — yorgunluk en çok hızda görünür")
        vel_pairs = [
            ("R_KNEE_vel_deg_s", "Sağ Diz Açısal Hızı"),
            ("L_KNEE_vel_deg_s", "Sol Diz Açısal Hızı"),
            ("R_HIP_vel_deg_s",  "Sağ Kalça Açısal Hızı"),
            ("R_FOOT_speed_norm","Sağ Ayak Hızı (normalize)"),
            ("L_FOOT_speed_norm","Sol Ayak Hızı (normalize)"),
        ]
        for col_name, lbl in vel_pairs:
            if col_name in pre_df.columns or col_name in post_df.columns:
                st.plotly_chart(
                    _overlay_angle_chart(pre_df, post_df, col_name, lbl, pre_events, post_events),
                    use_container_width=True,
                )

    # ── Tab 3: Fatigue analysis ───────────────────────────────────────────────
    with tabs[3]:
        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(_gauge(fi, "Yorgunluk İndeksi"), use_container_width=True)
            fi_color = "🟢" if fi < 33 else ("🟡" if fi < 66 else "🔴")
            st.markdown(f"**{fi_color} {fi_label} yorgunluk** — {fi:.1f}/100")
            st.caption(
                "0–33: Düşük  |  33–66: Orta  |  66–100: Yüksek\n\n"
                "Ağırlıklar: Diz ROM ×0.25, Peak hız ×0.25, "
                "Peak hıza süre ×0.15, Tekme yüksekliği ×0.15, "
                "Ayak hızı ×0.10"
            )

        with g2:
            st.subheader("Metrik Bazlı Yorgunluk Katkıları")
            mets = fatigue_data["metrics"]
            rows_ft = []
            for key, m in mets.items():
                if m["pre"] is None or m["post"] is None:
                    continue
                if m["pct"] is None:
                    continue
                trend = "▲" if m["pct"] > 0 else "▼"
                is_fatigue = (m["direction"] * m["pct"]) > 0
                rows_ft.append({
                    "Metrik": f"{m['label']} ({m['unit']})" if m["unit"] else m["label"],
                    "Pre":    f"{m['pre']:.3f}",
                    "Post":   f"{m['post']:.3f}",
                    "Δ":      f"{m['delta']:+.3f}",
                    "% Değişim": f"{trend} {abs(m['pct']):.1f}%",
                    "Yorgunluk": "🔴 Evet" if is_fatigue else "🟢 Hayır",
                    "Katkı": f"{m['fatigue_contribution']:+.1f}",
                })
            st.dataframe(pd.DataFrame(rows_ft).set_index("Metrik"), use_container_width=True)

        # Horizontal bar chart of fatigue contributions
        st.subheader("Yorgunluk Katkı Grafiği")
        contrib_labels, contrib_vals, contrib_colors = [], [], []
        for key, m in mets.items():
            if m.get("fatigue_contribution") is None:
                continue
            contrib_labels.append(m["label"])
            contrib_vals.append(round(m["fatigue_contribution"], 1))
            contrib_colors.append("#ef4444" if m["fatigue_contribution"] > 0 else "#22c55e")

        fig_contrib = go.Figure(go.Bar(
            x=contrib_vals, y=contrib_labels, orientation="h",
            marker_color=contrib_colors,
            text=[f"{v:+.1f}" for v in contrib_vals],
            textposition="outside",
        ))
        fig_contrib.add_vline(x=0, line_color="#888", line_width=1)
        fig_contrib.update_layout(
            height=340, xaxis_title="Yorgunluk katkısı (+ = yorgunluk, − = iyileşme)",
            margin=dict(l=180, r=60, t=20, b=30),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            xaxis=dict(gridcolor="#333", range=[-110, 110]),
            yaxis=dict(gridcolor="#333"),
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

    # ── Tab 4: Per-kick breakdown ─────────────────────────────────────────────
    with tabs[4]:
        st.caption("Her tekme için pre vs post karşılaştırması")
        pk_col = st.selectbox(
            "Metrik",
            [k for k in FATIGUE_METRICS if
             any(e.get(k) is not None for e in pre_events + post_events)],
            format_func=lambda k: FATIGUE_METRICS[k][0],
            key="dv_pk_col",
        )
        if pk_col:
            st.plotly_chart(
                _per_kick_trend(pre_events, post_events, pk_col, FATIGUE_METRICS[pk_col][0]),
                use_container_width=True,
            )

        st.divider()

        # ── Kick detail cards with video jump ────────────────────────────────
        disp_cols = ["kick_id", "active_leg", "duration_sec", "active_knee_rom_deg",
                     "active_peak_knee_vel_deg_s", "time_to_peak_knee_vel_sec",
                     "peak_kick_height_norm", "active_peak_foot_speed_norm"]

        def _trim_clip(src: Path, start: float, end: float, out: Path) -> bool:
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
                pad = 0.4  # seconds before/after kick
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

        def _kick_video_section(events_list: list[dict], video_path: str, label: str, tmp_dir: Path | None) -> None:
            st.markdown(f"#### {label}")
            if not events_list:
                st.info("Tekme tespit edilemedi.")
                return

            ev_df = pd.DataFrame(events_list)
            st.dataframe(
                ev_df[[c for c in disp_cols if c in ev_df.columns]].set_index("kick_id"),
                use_container_width=True,
            )

            vid_path = Path(video_path)

            for ev in events_list:
                kid = int(ev["kick_id"])
                leg = ev.get("active_leg", "?")
                t_start = float(ev["start_time_sec"])
                t_end   = float(ev["end_time_sec"])
                dur     = float(ev.get("duration_sec", 0))
                rom     = ev.get("active_knee_rom_deg")
                vel     = ev.get("active_peak_knee_vel_deg_s")
                height  = ev.get("peak_kick_height_norm")

                with st.expander(
                    f"Tekme {kid}  |  {leg} bacak  |  {t_start:.2f}s – {t_end:.2f}s  |  ROM {float(rom):.1f}°" if rom else f"Tekme {kid}  |  {leg} bacak  |  {t_start:.2f}s – {t_end:.2f}s",
                    expanded=(kid == 1),
                ):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Süre", f"{dur:.2f} sn")
                    mc2.metric("Diz ROM", f"{float(rom):.1f}°" if rom else "—")
                    mc3.metric("Peak Hız", f"{float(vel):.0f} °/s" if vel else "—")
                    mc4.metric("Yükseklik", f"{float(height):.3f}" if height else "—")

                    # Trim clip for this kick
                    if vid_path.exists() and tmp_dir is not None:
                        clip_key = f"clip_{label}_{kid}"
                        clip_path = tmp_dir / f"{clip_key}.mp4"
                        if not clip_path.exists():
                            with st.spinner("Video kırpılıyor…"):
                                _trim_clip(vid_path, t_start, t_end, clip_path)
                        if clip_path.exists() and clip_path.stat().st_size > 1000:
                            _video_player(clip_path)
                        else:
                            _video_player(vid_path, start_time=t_start)
                    elif vid_path.exists():
                        _video_player(vid_path, start_time=t_start)

        tmp = st.session_state.get("dv_tmp")
        kc1, kc2 = st.columns(2)
        with kc1:
            _kick_video_section(
                pre_events,
                str(tmp / "pre_annotated.mp4") if tmp else "",
                "Pre Tekmeleri", tmp,
            )
        with kc2:
            _kick_video_section(
                post_events,
                str(tmp / "post_annotated.mp4") if tmp else "",
                "Post Tekmeleri", tmp,
            )

    # ── Tab 5: Faz analizi ───────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("Tekme Faz Analizi")
        st.caption("Her tekme 3 faza ayrılır: Yüklenme (chamber) → Uzatma (extension) → Geri çekim (retraction)")

        PHASE_COLS = {
            "loading_dur_sec":           ("Yüklenme Süresi", "sn"),
            "extension_dur_sec":         ("Uzatma Süresi", "sn"),
            "retraction_dur_sec":        ("Geri Çekim Süresi", "sn"),
            "loading_peak_vel_deg_s":    ("Yüklenme Peak Hız", "°/s"),
            "extension_peak_vel_deg_s":  ("Uzatma Peak Hız", "°/s"),
            "retraction_peak_vel_deg_s": ("Geri Çekim Peak Hız", "°/s"),
        }

        def _phase_bars(events_list: list[dict], label: str, color: str) -> None:
            phases = ["Yüklenme", "Uzatma", "Geri Çekim"]
            dur_keys = ["loading_dur_sec", "extension_dur_sec", "retraction_dur_sec"]
            fig = go.Figure()
            for ev in events_list:
                kick_lbl = f"T{int(ev['kick_id'])}"
                durs = [float(ev.get(k) or 0) for k in dur_keys]
                for phase, dur in zip(phases, durs):
                    fig.add_trace(go.Bar(name=phase, x=[kick_lbl], y=[dur],
                                         legendgroup=phase, showlegend=(int(ev['kick_id']) == 1)))
            fig.update_layout(barmode="stack", height=220, title=dict(text=label, font=dict(color="#fafafa", size=13)),
                              plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
                              xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", title="Süre (sn)"),
                              margin=dict(l=40, r=10, t=35, b=30),
                              legend=dict(orientation="h", y=-0.35))
            st.plotly_chart(fig, use_container_width=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            _phase_bars(pre_events,  "Pre — Faz Süreleri", "#3b82f6")
        with fc2:
            _phase_bars(post_events, "Post — Faz Süreleri", "#ef4444")

        # Phase velocity comparison
        st.subheader("Faz Hızları — Pre vs Post")
        vel_phase_keys = [("extension_peak_vel_deg_s", "Uzatma Peak Hızı (°/s)"),
                          ("retraction_peak_vel_deg_s", "Geri Çekim Peak Hızı (°/s)"),
                          ("loading_peak_vel_deg_s", "Yüklenme Peak Hızı (°/s)")]
        for pk, plbl in vel_phase_keys:
            if any(e.get(pk) is not None for e in pre_events + post_events):
                st.plotly_chart(_per_kick_trend(pre_events, post_events, pk, plbl), use_container_width=True)

        # Retraction ratio — key fatigue marker
        st.subheader("Geri Çekim / Uzatma Oranı")
        st.caption("Yorgunlukla geri çekim yavaşlar → oran artar. >1.5 = belirgin yavaşlama.")
        for label, evs, color in [("Pre", pre_events, "#3b82f6"), ("Post", post_events, "#ef4444")]:
            ratios = []
            for ev in evs:
                ext = ev.get("extension_dur_sec")
                ret = ev.get("retraction_dur_sec")
                if ext and ret and float(ext) > 0.001:
                    ratios.append(round(float(ret) / float(ext), 3))
            if ratios:
                mean_ratio = sum(ratios) / len(ratios)
                st.metric(f"{label} — ort. geri çekim/uzatma oranı", f"{mean_ratio:.2f}",
                          delta="⚠️ yavaş geri çekim" if mean_ratio > 1.5 else "✅ normal", delta_color="off")

    # ── Tab 6: Asimetri ──────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Bilateral Asimetri İndeksi (ASI)")
        st.caption("ASI = (Sağ − Sol) / Ort(Sağ, Sol) × 100  |  Pozitif → Sağ dominant  |  |ASI| > 10% klinik olarak anlamlı")

        asi_cols = [("knee_asi", "Diz ASI (%)"), ("hip_asi", "Kalça ASI (%)")]
        for asi_key, asi_lbl in asi_cols:
            if not any(e.get(asi_key) is not None for e in pre_events + post_events):
                continue
            fig_asi = go.Figure()
            for evs, clr, lbl in [(pre_events, "#3b82f6", "Pre"), (post_events, "#ef4444", "Post")]:
                vals = [float(e[asi_key]) if e.get(asi_key) is not None else None for e in evs]
                x_lbl = [f"T{int(e['kick_id'])}" for e in evs]
                fig_asi.add_trace(go.Bar(name=lbl, x=x_lbl, y=vals, marker_color=clr))
            fig_asi.add_hline(y=10,  line_dash="dash", line_color="orange", annotation_text="+10% eşik")
            fig_asi.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="-10% eşik")
            fig_asi.add_hline(y=0,   line_color="#555", line_width=1)
            fig_asi.update_layout(barmode="group", height=280,
                                  title=dict(text=asi_lbl, font=dict(color="#fafafa", size=13)),
                                  plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                  font=dict(color="#fafafa"),
                                  xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", title="%"),
                                  margin=dict(l=40, r=10, t=35, b=30),
                                  legend=dict(orientation="h", y=-0.35))
            st.plotly_chart(fig_asi, use_container_width=True)

        # Mean ASI summary
        st.subheader("Oturum Ortalama ASI")
        ac1, ac2, ac3, ac4 = st.columns(4)
        for col_w, label, evs in [(ac1, "Pre Diz", pre_events), (ac2, "Post Diz", post_events),
                                   (ac3, "Pre Kalça", pre_events), (ac4, "Post Kalça", post_events)]:
            k = "knee_asi" if "Diz" in label else "hip_asi"
            vals = [float(e[k]) for e in evs if e.get(k) is not None]
            if vals:
                mean_asi = sum(vals) / len(vals)
                flag = "⚠️" if abs(mean_asi) > 10 else "✅"
                col_w.metric(label, f"{mean_asi:+.1f}%", delta=flag, delta_color="off")

    # ── Tab 7: İstatistik ────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("İstatistiksel Karşılaştırma")
        st.caption("Cohen's d etki büyüklüğü + %95 güven aralığı — düşük tekme sayısında yorumla dikkatli")

        from src.stats import compare_metric, effect_label

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
            res = compare_metric(pre_v, post_v)
            ci_pre  = res["pre_ci"]
            ci_post = res["post_ci"]
            d = res["cohens_d"]
            stat_rows.append({
                "Metrik": col_lbl,
                "Pre ort. ± std": f"{res['pre_mean']:.2f} ± {res['pre_std']:.2f}" if res["pre_std"] else f"{res['pre_mean']:.2f}",
                "Post ort. ± std": f"{res['post_mean']:.2f} ± {res['post_std']:.2f}" if res["post_std"] else f"{res['post_mean']:.2f}",
                "%95 CI (Pre)":  f"[{ci_pre[0]:.2f}, {ci_pre[1]:.2f}]"  if ci_pre  else "—",
                "%95 CI (Post)": f"[{ci_post[0]:.2f}, {ci_post[1]:.2f}]" if ci_post else "—",
                "Δ": f"{res['delta']:+.2f}" if res["delta"] is not None else "—",
                "% Değişim": f"{res['pct_change']:+.1f}%" if res["pct_change"] is not None else "—",
                "Cohen's d": f"{d:.2f}" if d is not None else "—",
                "Etki Büyüklüğü": res["effect_label"],
                "n (pre/post)": f"{res['n_pre']} / {res['n_post']}",
            })

        if stat_rows:
            stat_df = pd.DataFrame(stat_rows).set_index("Metrik")
            st.dataframe(stat_df, use_container_width=True)

            # Cohen's d bar chart
            d_labels = [r["Metrik"] for r in stat_rows if r["Cohen's d"] != "—"]
            d_vals   = [float(r["Cohen's d"]) for r in stat_rows if r["Cohen's d"] != "—"]
            d_colors = ["#ef4444" if v < 0 else "#22c55e" for v in d_vals]
            if d_vals:
                fig_d = go.Figure(go.Bar(x=d_vals, y=d_labels, orientation="h",
                                          marker_color=d_colors,
                                          text=[f"{v:+.2f}" for v in d_vals],
                                          textposition="outside"))
                fig_d.add_vline(x=0,    line_color="#888", line_width=1)
                fig_d.add_vline(x=0.8,  line_dash="dot", line_color="#f59e0b", annotation_text="büyük")
                fig_d.add_vline(x=-0.8, line_dash="dot", line_color="#f59e0b")
                fig_d.add_vline(x=0.5,  line_dash="dot", line_color="#64748b", annotation_text="orta")
                fig_d.add_vline(x=-0.5, line_dash="dot", line_color="#64748b")
                fig_d.update_layout(
                    height=380, xaxis_title="Cohen's d  (negatif = post < pre)",
                    margin=dict(l=200, r=80, t=20, b=30),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"),
                    xaxis=dict(gridcolor="#333", range=[-3, 3]),
                    yaxis=dict(gridcolor="#333"),
                )
                st.plotly_chart(fig_d, use_container_width=True)

        # Güven uyarıları
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

        # n < 5 uyarısı
        if len(pre_events) < 5 or len(post_events) < 5:
            st.info(
                f"ℹ️ Pre: {len(pre_events)} tekme, Post: {len(post_events)} tekme. "
                "İstatistiksel karşılaştırma için her oturumda en az 5 tekme önerilir. "
                "Cohen's d ve CI değerleri düşük n'de geniş belirsizlik taşır."
            )

    # ── Tab 8: Export ─────────────────────────────────────────────────────────
    with tabs[8]:
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
                "metrik_kodu":   key,
                "metrik_label":  m["label"],
                "birim":         m["unit"],
                "pre":           round(m["pre"], 4) if m["pre"] else "",
                "post":          round(m["post"], 4) if m["post"] else "",
                "delta":         round(m["delta"], 4) if m["delta"] is not None else "",
                "pct_degisim":   round(m["pct"], 2) if m["pct"] is not None else "",
                "yorgunluk_katkisi": round(m["fatigue_contribution"], 2) if m["fatigue_contribution"] is not None else "",
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


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EMG Sync
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "EMG Sync":
    st.title("EMG Senkronizasyon")
    st.caption(
        "EMG CSV + frame metrics CSV yükle → video zamanıyla hizala → tekme bazlı RMS hesapla."
    )

    st.sidebar.subheader("EMG Ayarları")
    emg_offset = st.sidebar.number_input("EMG zaman offset (sn)", value=0.0, step=0.01, format="%.3f", help="EMG kaydı videodan geç başladıysa pozitif girin.")
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

            # Save uploads to temp
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                emg_path = tmp / "emg.csv"
                frame_path = tmp / "frame.csv"
                emg_path.write_bytes(emg_file.read())
                frame_path.write_bytes(frame_file.read())

                frame_df = pd.read_csv(str(frame_path))

                # Load EMG
                emg_data = load_emg_csv(
                    emg_path,
                    delimiter=emg_delimiter,
                    skip_rows=int(emg_skip_rows),
                )

                st.success(
                    f"EMG yüklendi — {len(emg_data['channels'])} kanal, "
                    f"~{emg_data['sample_rate_hz']:.0f} Hz, "
                    f"{len(emg_data['time_s'])} sample"
                )

                # Channel selection
                all_channels = list(emg_data["channels"].keys())
                selected_channels = st.multiselect(
                    "Gösterilecek kanallar",
                    all_channels,
                    default=all_channels[:min(4, len(all_channels))],
                )

                # Resample
                video_times = frame_df["time_sec"].tolist() if "time_sec" in frame_df.columns else []
                if not video_times:
                    st.error("Frame CSV'de 'time_sec' sütunu bulunamadı.")
                    st.stop()

                emg_resampled = resample_to_video_times(emg_data, video_times, emg_time_offset_sec=emg_offset)

                # ── EMG overlay on video timeline ────────────────────────────
                st.subheader("EMG + Diz Açısı (senkron)")

                kick_events: list[dict] = []
                if ev_file:
                    ev_df = pd.read_csv(ev_file)
                    kick_events = ev_df.to_dict("records")

                shapes = _kick_event_lines(kick_events)

                fig_sync = go.Figure()
                # Add knee angle if available
                for col, color in [("R_KNEE", "#ef4444"), ("L_KNEE", "#3b82f6")]:
                    if col in frame_df.columns:
                        fig_sync.add_trace(go.Scatter(
                            x=video_times,
                            y=frame_df[col].tolist(),
                            name=f"{col} açısı (°)",
                            yaxis="y2",
                            line=dict(color=color, width=1.5, dash="dot"),
                            mode="lines",
                        ))

                for ch in selected_channels:
                    vals = emg_resampled.get(ch, [])
                    fig_sync.add_trace(go.Scatter(
                        x=video_times,
                        y=vals,
                        name=f"EMG: {ch}",
                        mode="lines",
                        line=dict(width=1),
                    ))

                fig_sync.update_layout(
                    height=380,
                    xaxis_title="Zaman (sn)",
                    yaxis=dict(title="EMG (a.u.)", gridcolor="#333"),
                    yaxis2=dict(title="Açı (°)", overlaying="y", side="right", gridcolor="#444"),
                    shapes=shapes,
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"),
                    margin=dict(l=50, r=60, t=30, b=40),
                    legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(fig_sync, use_container_width=True)

                # ── Per-kick RMS table ────────────────────────────────────────
                if kick_events:
                    st.subheader("Tekme Bazlı EMG RMS")
                    kick_emg = compute_rms_per_kick(
                        {ch: emg_resampled[ch] for ch in selected_channels if ch in emg_resampled},
                        kick_events,
                        video_fps=30.0,
                    )
                    rms_df = pd.DataFrame(kick_emg).set_index("kick_id")
                    st.dataframe(rms_df, use_container_width=True)

                    # Per-channel RMS bar chart across kicks
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
                            barmode="group",
                            title="Tekme Bazlı RMS per Kanal",
                            height=320,
                            plot_bgcolor="#0e1117",
                            paper_bgcolor="#0e1117",
                            font=dict(color="#fafafa"),
                            xaxis=dict(gridcolor="#333"),
                            yaxis=dict(gridcolor="#333"),
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
                    "emg_synced_frames.csv",
                    "text/csv",
                )

                if kick_events:
                    kick_emg_path = tmp / "kick_emg_rms.csv"
                    export_kick_emg_csv(kick_emg_path, kick_emg)
                    st.download_button(
                        "📥 Tekme EMG RMS CSV indir",
                        kick_emg_path.read_bytes(),
                        "kick_emg_rms.csv",
                        "text/csv",
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
