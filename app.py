#!/usr/bin/env python3
"""
app.py — Flask web UI for face_timelapse.py.

Run:
    python app.py
Browser opens automatically at http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import os
import queue
import threading
import webbrowser
from pathlib import Path
from typing import Any, Dict

from flask import Flask, Response, jsonify, request, send_file, stream_with_context

from face_timelapse import (
    detect_all_faces,
    discover_images,
    smooth_eye_coords,
    write_video,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global job state — only one job at a time
# ---------------------------------------------------------------------------

_progress_q: queue.Queue = queue.Queue(maxsize=512)
_cancel_event = threading.Event()
_job_lock = threading.Lock()
_job_active: bool = False
_output_file: str = ""

# ---------------------------------------------------------------------------
# Embedded UI
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Timelapse Generator</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh;display:flex;justify-content:center;padding:2rem 1rem}
.container{width:100%;max-width:680px}
h1{font-size:1.75rem;font-weight:700;margin-bottom:1.5rem;color:#f8fafc}
.card{background:#1a1d27;border:1px solid #2d3148;border-radius:12px;padding:1.5rem;margin-bottom:1.25rem}
.field{margin-bottom:1rem}
.field:last-child{margin-bottom:0}
label.lbl{display:block;font-size:.75rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.35rem}
input[type=text],input[type=number]{width:100%;background:#0f1117;border:1px solid #2d3148;border-radius:8px;padding:.6rem .75rem;color:#e2e8f0;font-size:.95rem;outline:none;transition:border-color .15s}
input:focus{border-color:#6366f1}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem}
.toggle-group{display:flex;background:#0f1117;border:1px solid #2d3148;border-radius:8px;overflow:hidden}
.toggle-group input[type=radio]{display:none}
.toggle-group .tlbl{flex:1;text-align:center;padding:.6rem;cursor:pointer;font-size:.875rem;color:#64748b;margin:0;transition:all .15s}
.toggle-group input:checked+.tlbl{background:#6366f1;color:#fff}
.btn-primary{display:inline-flex;align-items:center;gap:.5rem;background:#6366f1;color:#fff;border:none;border-radius:8px;padding:.75rem 1.5rem;font-size:1rem;font-weight:600;cursor:pointer;transition:background .15s,opacity .15s}
.btn-primary:hover{background:#4f46e5}
.btn-primary:disabled{opacity:.45;cursor:not-allowed}
.btn-cancel{display:inline-flex;align-items:center;gap:.5rem;background:transparent;color:#ef4444;border:1px solid #ef4444;border-radius:8px;padding:.75rem 1.5rem;font-size:1rem;font-weight:600;cursor:pointer;transition:all .15s}
.btn-cancel:hover{background:#ef444422}
.btn-row{display:flex;gap:.75rem;align-items:center;margin-top:1.25rem}
#progress-section{display:none}
.phase-lbl{font-size:.875rem;color:#94a3b8;margin-bottom:.5rem}
.track{background:#2d3148;border-radius:999px;height:8px;overflow:hidden;margin-bottom:.4rem}
.fill{height:100%;background:linear-gradient(90deg,#6366f1,#818cf8);border-radius:999px;transition:width .3s ease;width:0%}
.prog-meta{display:flex;justify-content:space-between;font-size:.8rem;color:#64748b}
.err{color:#ef4444;font-size:.875rem;margin-top:.6rem;display:none}
#video-section{display:none}
video{width:100%;border-radius:8px;background:#000;margin-top:.75rem}
.ok-lbl{font-size:.875rem;color:#4ade80;margin-bottom:.25rem}
</style>
</head>
<body>
<div class="container">
  <h1>&#127916; Face Timelapse Generator</h1>

  <div class="card">
    <div class="field">
      <label class="lbl">Input Folder</label>
      <input type="text" id="input_dir" placeholder="/path/to/photos">
    </div>
    <div class="field">
      <label class="lbl">Output File</label>
      <input type="text" id="output_file" value="timelapse_output.mp4">
    </div>
    <div class="grid-2 field">
      <div>
        <label class="lbl">FPS</label>
        <input type="number" id="fps" value="24" min="1" max="120" step="1">
      </div>
      <div>
        <label class="lbl">Resolution</label>
        <input type="text" id="resolution" value="1920x1080">
      </div>
    </div>
    <div class="grid-3 field">
      <div>
        <label class="lbl">Zoom Factor</label>
        <input type="number" id="zoom_factor" value="1.0" min="0.1" max="5" step="0.1">
      </div>
      <div>
        <label class="lbl">EMA Alpha</label>
        <input type="number" id="ema_alpha" value="0.25" min="0.01" max="1" step="0.01">
      </div>
      <div>
        <label class="lbl">Eye Span</label>
        <input type="number" id="eye_span" value="0.35" min="0.05" max="0.9" step="0.01">
      </div>
    </div>
    <div class="field">
      <label class="lbl">Background Fill</label>
      <div class="toggle-group">
        <input type="radio" name="blur_bg" id="bg_blur" value="true" checked>
        <label class="tlbl" for="bg_blur">Blurred</label>
        <input type="radio" name="blur_bg" id="bg_black" value="false">
        <label class="tlbl" for="bg_black">Black</label>
      </div>
    </div>
    <div class="btn-row">
      <button class="btn-primary" id="btn-gen" onclick="startJob()">&#9654; Generate Timelapse</button>
      <button class="btn-cancel" id="btn-cancel" style="display:none" onclick="cancelJob()">&#10005; Cancel</button>
    </div>
  </div>

  <div class="card" id="progress-section">
    <div class="phase-lbl" id="phase-lbl">Initialising&#8230;</div>
    <div class="track"><div class="fill" id="fill"></div></div>
    <div class="prog-meta">
      <span id="prog-msg"></span>
      <span id="prog-pct">0%</span>
    </div>
    <div class="err" id="err-msg"></div>
  </div>

  <div class="card" id="video-section">
    <div class="ok-lbl">&#10003; Timelapse complete</div>
    <video id="vid" controls></video>
  </div>
</div>
<script>
let es = null;

function startJob() {
  const inputDir = document.getElementById('input_dir').value.trim();
  const outputFile = document.getElementById('output_file').value.trim();
  if (!inputDir) { alert('Please enter an input folder path.'); return; }

  const payload = {
    input_dir: inputDir,
    output_file: outputFile || 'timelapse_output.mp4',
    fps: parseFloat(document.getElementById('fps').value),
    resolution: document.getElementById('resolution').value.trim(),
    zoom_factor: parseFloat(document.getElementById('zoom_factor').value),
    ema_alpha: parseFloat(document.getElementById('ema_alpha').value),
    eye_span: parseFloat(document.getElementById('eye_span').value),
    blur_bg: document.querySelector('input[name=blur_bg]:checked').value === 'true',
  };

  resetUI();

  fetch('/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  })
  .then(r => r.json())
  .then(d => { if (d.error) { showErr(d.error); resetButtons(); } else { listenProgress(); } })
  .catch(e => { showErr(e.toString()); resetButtons(); });
}

function listenProgress() {
  if (es) es.close();
  es = new EventSource('/progress');
  es.onmessage = function(e) {
    const ev = JSON.parse(e.data);
    if (ev.phase === 'ping') return;
    if (ev.phase === 'done') {
      setProg(100, 'Complete', ev.msg);
      showVideo();
      resetButtons();
      es.close(); es = null;
    } else if (ev.phase === 'error') {
      showErr(ev.msg);
      resetButtons();
      es.close(); es = null;
    } else {
      const lbl = ev.phase === 'detection' ? 'Detecting faces…' : 'Writing video…';
      setProg(ev.pct, lbl, ev.msg);
    }
  };
  es.onerror = function() { es.close(); es = null; };
}

function setProg(pct, lbl, msg) {
  document.getElementById('phase-lbl').textContent = lbl;
  document.getElementById('fill').style.width = pct + '%';
  document.getElementById('prog-pct').textContent = pct + '%';
  document.getElementById('prog-msg').textContent = msg;
}

function showErr(msg) {
  const el = document.getElementById('err-msg');
  el.textContent = msg;
  el.style.display = 'block';
}

function showVideo() {
  document.getElementById('video-section').style.display = 'block';
  const v = document.getElementById('vid');
  v.src = '/video?t=' + Date.now();
  v.load();
}

function cancelJob() {
  fetch('/cancel', {method: 'POST'});
}

function resetUI() {
  document.getElementById('progress-section').style.display = 'block';
  document.getElementById('video-section').style.display = 'none';
  document.getElementById('err-msg').style.display = 'none';
  document.getElementById('fill').style.width = '0%';
  document.getElementById('prog-pct').textContent = '0%';
  document.getElementById('prog-msg').textContent = '';
  document.getElementById('phase-lbl').textContent = 'Starting…';
  document.getElementById('btn-gen').disabled = true;
  document.getElementById('btn-cancel').style.display = 'inline-flex';
}

function resetButtons() {
  document.getElementById('btn-gen').disabled = false;
  document.getElementById('btn-cancel').style.display = 'none';
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> Response:
    return Response(_HTML, status=200, mimetype="text/html")


@app.post("/run")
def run() -> Response:
    """Start a timelapse job. Returns 409 if one is already running."""
    global _job_active, _output_file, _cancel_event

    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}

    input_dir = data.get("input_dir", "").strip()
    if not input_dir:
        return jsonify({"error": "input_dir is required."}), 400

    resolution = data.get("resolution", "1920x1080")
    parts = resolution.lower().split("x")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        return jsonify({"error": f"Invalid resolution '{resolution}'. Use WxH, e.g. 1920x1080."}), 400

    fps_val = float(data.get("fps", 24))
    if fps_val <= 0:
        return jsonify({"error": "fps must be greater than 0."}), 400

    with _job_lock:
        if _job_active:
            return jsonify({"error": "A job is already running."}), 409
        _job_active = True

    # Reset shared state
    _cancel_event = threading.Event()
    while not _progress_q.empty():
        try:
            _progress_q.get_nowait()
        except queue.Empty:
            break
    raw_output = data.get("output_file", "timelapse_output.mp4") or "timelapse_output.mp4"
    resolved_output = Path(raw_output).resolve()
    allowed_root = Path.cwd().resolve()
    if not str(resolved_output).startswith(str(allowed_root)):
        with _job_lock:
            _job_active = False
        return jsonify({"error": "output_file must be within the current working directory."}), 400
    _output_file = str(resolved_output)
    data["output_file"] = _output_file

    t = threading.Thread(target=_run_pipeline, args=(data,), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.get("/progress")
def progress() -> Response:
    """SSE stream of job progress events."""

    def generate():
        while True:
            try:
                event = _progress_q.get(timeout=30)
                try:
                    yield f"data: {json.dumps(event)}\n\n"
                except GeneratorExit:
                    return
                if event.get("phase") in ("done", "error"):
                    break
            except queue.Empty:
                try:
                    yield 'data: {"phase":"ping"}\n\n'
                except GeneratorExit:
                    return

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/video")
def video() -> Response:
    """Serve the finished video for in-browser playback."""
    if not _output_file or not Path(_output_file).exists():
        return jsonify({"error": "No video available yet."}), 404
    return send_file(_output_file, mimetype="video/mp4")


@app.post("/scan")
def scan() -> Response:
    """Validate path and count images, optionally recursing subfolders."""
    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}

    raw_path = data.get("path", "").strip()
    if not raw_path:
        return jsonify({"error": "path is required."}), 400

    recursive = bool(data.get("recursive", True))

    try:
        paths = discover_images(raw_path, recursive=recursive)
        return jsonify({"count": len(paths), "path": str(Path(raw_path).resolve())})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/cancel")
def cancel() -> Response:
    """Signal the running job to stop."""
    _cancel_event.set()
    return jsonify({"status": "cancel signal sent"})


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(data: Dict[str, Any]) -> None:
    """Execute the timelapse pipeline on a background thread."""
    global _job_active

    try:
        w, h = [int(x) for x in data["resolution"].lower().split("x")]
        num_workers = max(1, os.cpu_count() - 1)  # type: ignore[operator]

        image_paths = discover_images(
            data["input_dir"],
            recursive=bool(data.get("recursive", True)),
        )
        total = len(image_paths)

        def det_cb(phase: str, current: int, total_count: int) -> None:
            _progress_q.put({
                "phase": "detection",
                "pct": int(100 * current / max(total_count, 1)),
                "msg": f"{current:,} / {total_count:,} images",
                "eta": "",
            })

        detections = detect_all_faces(
            image_paths,
            num_workers=num_workers,
            progress_callback=det_cb,
            cancel_event=_cancel_event,
        )

        if _cancel_event.is_set():
            _emit_cancelled(data.get("output_file", ""))
            return

        smoothed = smooth_eye_coords(detections, alpha=float(data.get("ema_alpha", 0.25)))

        detected = sum(1 for d in detections if d.eyes is not None)
        if detected == 0:
            _progress_q.put({
                "phase": "error", "pct": 0,
                "msg": "No faces detected in any image.", "eta": "",
            })
            return

        def write_cb(phase: str, current: int, total_count: int) -> None:
            _progress_q.put({
                "phase": "writing",
                "pct": int(100 * current / max(total_count, 1)),
                "msg": f"{current:,} / {total_count:,} frames",
                "eta": "",
            })

        out = data.get("output_file", "timelapse_output.mp4")
        write_video(
            image_paths=image_paths,
            smoothed=smoothed,
            output_file=out,
            fps=float(data.get("fps", 24)),
            dst_w=w,
            dst_h=h,
            zoom=float(data.get("zoom_factor", 1.0)),
            eye_span_fraction=float(data.get("eye_span", 0.35)),
            blur_bg=bool(data.get("blur_bg", True)),
            progress_callback=write_cb,
            cancel_event=_cancel_event,
        )

        if _cancel_event.is_set():
            _emit_cancelled(out)
            return

        _progress_q.put({
            "phase": "done", "pct": 100,
            "msg": f"Saved → {out}", "eta": "",
        })

    except Exception as exc:  # noqa: BLE001
        _progress_q.put({"phase": "error", "pct": 0, "msg": str(exc), "eta": ""})

    finally:
        with _job_lock:
            _job_active = False


def _emit_cancelled(output_file: str) -> None:
    if output_file and Path(output_file).exists():
        try:
            os.remove(output_file)
        except OSError:
            pass
    _progress_q.put({"phase": "error", "pct": 0, "msg": "Cancelled by user.", "eta": ""})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
