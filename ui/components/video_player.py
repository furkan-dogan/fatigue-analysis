"""Shared local HTML5 player. Remote serving is tracked in upgrade.md."""

from __future__ import annotations

import http.server as _http_server
import socketserver as _socketserver
import threading as _threading
import urllib.parse as _urllib_parse
from pathlib import Path
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
            fname = _urllib_parse.unquote(self.path.lstrip("/").split("?")[0])
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
    url = f"http://127.0.0.1:{port}/{_urllib_parse.quote(p.name)}{t_frag}"
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
