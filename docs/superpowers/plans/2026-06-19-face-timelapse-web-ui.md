# Face Timelapse Web UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Flask web UI to `face_timelapse.py` so users can configure, run, monitor progress, and preview the output video in Chrome.

**Architecture:** A single `app.py` runs a Flask server that imports pipeline functions from `face_timelapse.py`, executes them on a background thread, and streams progress to the browser via Server-Sent Events (SSE). The UI is a dark-themed HTML page embedded as a string in `app.py` — no build step required.

**Tech Stack:** Python 3.10+, Flask 3.x, threading, queue.Queue, SSE (text/event-stream), vanilla HTML/CSS/JS.

## Global Constraints

- Python ≥ 3.10 required (uses `match`, `slots=True`, walrus operator in face_timelapse.py)
- Flask ≥ 3.0
- Only one job may run at a time; `/run` returns HTTP 409 if a job is already active
- Cancel deletes any partial output file
- `face_timelapse.py` CLI behaviour must be unchanged (new params default to `None`)
- No Node.js, no npm, no build step

---

## Task 1: Add `progress_callback` and `cancel_event` to `face_timelapse.py`

**Files:**
- Modify: `face_timelapse.py` — `detect_all_faces()` and `write_video()`
- Create: `tests/test_callbacks.py`

**Interfaces:**
- Produces:
  - `detect_all_faces(..., progress_callback: Optional[Callable[[str, int, int], None]] = None, cancel_event: Optional[threading.Event] = None)`
  - `write_video(..., progress_callback: Optional[Callable[[str, int, int], None]] = None, cancel_event: Optional[threading.Event] = None)`
  - Both callback signatures: `callback(phase: str, current: int, total: int) -> None`

---

- [ ] **Step 1: Create the test file**

Create `tests/__init__.py` (empty) and `tests/test_callbacks.py`:

```python
"""Tests for progress_callback and cancel_event integration."""
import threading
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from face_timelapse import write_video, EyeCoords


def test_write_video_calls_progress_callback(tmp_path):
    """write_video calls progress_callback with ('writing', n, total) for each frame."""
    calls = []

    def cb(phase, current, total):
        calls.append((phase, current, total))

    # Create a tiny valid image
    import cv2, numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = tmp_path / "frame_001.jpg"
    cv2.imwrite(str(p), img)

    eyes = EyeCoords(lx=0.4, ly=0.5, rx=0.6, ry=0.5)
    out = str(tmp_path / "out.mp4")

    write_video(
        image_paths=[str(p)],
        smoothed=[eyes],
        output_file=out,
        fps=1.0,
        dst_w=320,
        dst_h=240,
        zoom=1.0,
        eye_span_fraction=0.35,
        blur_bg=False,
        progress_callback=cb,
    )

    assert len(calls) == 1
    assert calls[0] == ("writing", 1, 1)


def test_write_video_cancel_stops_early(tmp_path):
    """write_video stops writing when cancel_event is set before the loop."""
    import cv2, numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    paths = []
    for i in range(5):
        p = tmp_path / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(str(p))

    eyes = EyeCoords(lx=0.4, ly=0.5, rx=0.6, ry=0.5)
    out = str(tmp_path / "out.mp4")

    cancel = threading.Event()
    cancel.set()  # cancelled before we even start

    written_count = []
    def cb(phase, current, total):
        written_count.append(current)

    write_video(
        image_paths=paths,
        smoothed=[eyes] * 5,
        output_file=out,
        fps=1.0,
        dst_w=320,
        dst_h=240,
        zoom=1.0,
        eye_span_fraction=0.35,
        blur_bg=False,
        progress_callback=cb,
        cancel_event=cancel,
    )

    # No frames written because cancel was set upfront
    assert len(written_count) == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/nicholas
python -m pytest tests/test_callbacks.py -v
```

Expected: `FAILED` — `write_video()` does not yet accept `progress_callback` or `cancel_event`.

- [ ] **Step 3: Add imports to `face_timelapse.py`**

At the top of `face_timelapse.py`, the `typing` import already includes `Optional` and `Callable`. Confirm this line exists (it does in the current file):

```python
from typing import Dict, List, Optional, Tuple
```

Add `Callable` to that import:

```python
from typing import Callable, Dict, List, Optional, Tuple
```

Also add `threading` to the stdlib imports block (near the top):

```python
import threading
```

- [ ] **Step 4: Modify `detect_all_faces()` signature and body**

Find the current signature in `face_timelapse.py`:

```python
def detect_all_faces(
    image_paths: List[str],
    num_workers: int,
    batch_size: int = 128,
) -> List[DetectionResult]:
```

Replace with:

```python
def detect_all_faces(
    image_paths: List[str],
    num_workers: int,
    batch_size: int = 128,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[DetectionResult]:
```

Inside the function body, find the block that starts with:

```python
    with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init) as pool:
        with tqdm(total=len(tasks), desc="Detecting faces", unit="img", dynamic_ncols=True) as pbar:
            for start in range(0, len(tasks), batch_size):
                batch = tasks[start : start + batch_size]
                futures = {pool.submit(_worker_detect, t): t for t in batch}

                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        item = futures[fut]
                        res = DetectionResult(
                            path=item[0],
                            index=item[1],
                            error=str(exc),
                        )
                    bucket[res.index] = res
                    pbar.update(1)
```

Replace with:

```python
    use_tqdm = progress_callback is None
    pbar = (
        tqdm(total=len(tasks), desc="Detecting faces", unit="img", dynamic_ncols=True)
        if use_tqdm
        else None
    )

    try:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init) as pool:
            for start in range(0, len(tasks), batch_size):
                if cancel_event and cancel_event.is_set():
                    break
                batch = tasks[start : start + batch_size]
                futures = {pool.submit(_worker_detect, t): t for t in batch}

                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        item = futures[fut]
                        res = DetectionResult(
                            path=item[0],
                            index=item[1],
                            error=str(exc),
                        )
                    bucket[res.index] = res
                    if use_tqdm:
                        pbar.update(1)  # type: ignore[union-attr]
                    else:
                        progress_callback("detection", len(bucket), len(tasks))  # type: ignore[misc]
    finally:
        if pbar:
            pbar.close()
```

- [ ] **Step 5: Modify `write_video()` signature and body**

Find the current signature:

```python
def write_video(
    image_paths: List[str],
    smoothed: List[Optional[EyeCoords]],
    output_file: str,
    fps: float,
    dst_w: int,
    dst_h: int,
    zoom: float,
    eye_span_fraction: float,
    blur_bg: bool,
) -> None:
```

Replace with:

```python
def write_video(
    image_paths: List[str],
    smoothed: List[Optional[EyeCoords]],
    output_file: str,
    fps: float,
    dst_w: int,
    dst_h: int,
    zoom: float,
    eye_span_fraction: float,
    blur_bg: bool,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
```

Inside `write_video`, find the inner loop (inside the `with tqdm(...)` block):

```python
        with tqdm(
            total=len(image_paths),
            desc="Writing frames",
            unit="frame",
            dynamic_ncols=True,
        ) as pbar:
            for path, eyes in zip(image_paths, smoothed):
                pbar.update(1)

                if eyes is None:
```

Replace with:

```python
        use_tqdm = progress_callback is None
        total_frames = len(image_paths)
        pbar = (
            tqdm(total=total_frames, desc="Writing frames", unit="frame", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        frame_index = 0
        try:
          for path, eyes in zip(image_paths, smoothed):
            if cancel_event and cancel_event.is_set():
                break
            frame_index += 1
            if use_tqdm:
                pbar.update(1)  # type: ignore[union-attr]

            if eyes is None:
```

Then find the two `skipped += 1` and `written += 1` lines at the end of the loop body and add callback call after `written += 1`:

```python
                    writer.write(frame)
                    written += 1
                    if not use_tqdm:
                        progress_callback("writing", written + skipped, total_frames)  # type: ignore[misc]
```

Close the try/finally around the loop (before `writer.release()`):

```python
        finally:
            if pbar:
                pbar.close()
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
python -m pytest tests/test_callbacks.py -v
```

Expected output:
```
tests/test_callbacks.py::test_write_video_calls_progress_callback PASSED
tests/test_callbacks.py::test_write_video_cancel_stops_early PASSED
2 passed in Xs
```

- [ ] **Step 7: Verify CLI still works unchanged**

```bash
python face_timelapse.py --help
```

Expected: help text prints with all arguments listed. No import errors.

- [ ] **Step 8: Commit**

```bash
git -C /Users/nicholas init 2>/dev/null; \
git -C /Users/nicholas add face_timelapse.py tests/test_callbacks.py tests/__init__.py && \
git -C /Users/nicholas commit -m "feat: add progress_callback and cancel_event to pipeline functions"
```

---

## Task 2: Create `app.py` — Flask server with embedded UI

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes:
  - `face_timelapse.discover_images(input_dir: str) -> List[str]`
  - `face_timelapse.detect_all_faces(image_paths, num_workers, progress_callback, cancel_event) -> List[DetectionResult]`
  - `face_timelapse.smooth_eye_coords(detections, alpha) -> List[Optional[EyeCoords]]`
  - `face_timelapse.write_video(image_paths, smoothed, output_file, fps, dst_w, dst_h, zoom, eye_span_fraction, blur_bg, progress_callback, cancel_event) -> None`
- Produces:
  - `GET  /`          → 200 text/html
  - `POST /run`       → 200 `{"status":"started"}` | 400 `{"error":"..."}` | 409 `{"error":"A job is already running."}`
  - `GET  /progress`  → SSE stream of `{"phase","pct","msg","eta"}` JSON events
  - `GET  /video`     → video/mp4 stream | 404
  - `POST /cancel`    → 200 `{"status":"cancel signal sent"}`

---

- [ ] **Step 1: Create `app.py`**

Create `/Users/nicholas/app.py` with the following complete content:

```python
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

_progress_q: queue.Queue = queue.Queue()
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
    _output_file = data.get("output_file", "timelapse_output.mp4")

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
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("phase") in ("done", "error"):
                    break
            except queue.Empty:
                # keepalive ping so the browser connection stays open
                yield 'data: {"phase":"ping"}\n\n'

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

        image_paths = discover_images(data["input_dir"])
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
```

- [ ] **Step 2: Write Flask route tests**

Create `tests/test_app.py`:

```python
"""Smoke tests for Flask routes using the test client."""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import app as app_module
from app import app as flask_app


@pytest.fixture()
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_index_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Face Timelapse Generator" in r.data
    assert b"text/html" in r.content_type.encode()


def test_run_missing_input_dir(client):
    r = client.post("/run", json={})
    assert r.status_code == 400
    assert "input_dir is required" in r.get_json()["error"]


def test_run_invalid_resolution(client):
    r = client.post("/run", json={"input_dir": "/tmp", "resolution": "bad"})
    assert r.status_code == 400
    assert "resolution" in r.get_json()["error"].lower()


def test_video_404_when_no_output(client):
    app_module._output_file = ""
    r = client.get("/video")
    assert r.status_code == 404


def test_cancel_returns_ok(client):
    r = client.post("/cancel")
    assert r.status_code == 200
    assert r.get_json()["status"] == "cancel signal sent"
```

- [ ] **Step 3: Run route tests**

```bash
cd /Users/nicholas
python -m pytest tests/test_app.py -v
```

Expected:
```
tests/test_app.py::test_index_returns_html PASSED
tests/test_app.py::test_run_missing_input_dir PASSED
tests/test_app.py::test_run_invalid_resolution PASSED
tests/test_app.py::test_video_404_when_no_output PASSED
tests/test_app.py::test_cancel_returns_ok PASSED
5 passed in Xs
```

- [ ] **Step 4: Commit**

```bash
git -C /Users/nicholas add app.py tests/test_app.py && \
git -C /Users/nicholas commit -m "feat: add Flask web UI with SSE progress and embedded HTML"
```

---

## Task 3: Update `requirements.txt` and smoke test in browser

**Files:**
- Modify: `requirements.txt`

---

- [ ] **Step 1: Update `requirements.txt`**

Replace the contents of `/Users/nicholas/requirements.txt` with:

```
# face_timelapse.py + app.py — Python >= 3.10 required
opencv-python>=4.9.0
mediapipe>=0.10.14
numpy>=1.26.0
tqdm>=4.66.0
flask>=3.0.0
```

- [ ] **Step 2: Install flask**

```bash
pip install flask>=3.0.0
```

Expected: `Successfully installed flask-X.X.X` (or "already satisfied").

- [ ] **Step 3: Run all tests**

```bash
cd /Users/nicholas
python -m pytest tests/ -v
```

Expected: All 7 tests pass (2 callback + 5 route).

- [ ] **Step 4: Launch the app**

```bash
cd /Users/nicholas
python app.py
```

Expected: Browser opens at `http://127.0.0.1:5000` showing the dark-themed UI with the form, Generate button, and all settings fields.

- [ ] **Step 5: Smoke test the UI manually**

1. Enter a folder path that **does not exist** → click Generate → confirm red error appears: `"Input directory does not exist: ..."`
2. Enter a valid folder of photos → click Generate → confirm the progress section appears, phase label reads "Detecting faces…", and the progress bar animates.
3. While the job runs, click Cancel → confirm "Cancelled by user." error message appears and the Generate button re-enables.
4. Run a job to completion → confirm the video player appears and the video plays back correctly.

- [ ] **Step 6: Commit**

```bash
git -C /Users/nicholas add requirements.txt && \
git -C /Users/nicholas commit -m "chore: add flask to requirements"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Pick input folder → `input_dir` field in form
- ✅ Configure settings → all 7 fields (fps, resolution, zoom, ema_alpha, eye_span, blur_bg, output_file)
- ✅ Live progress bar → SSE `/progress` + JS `setProg()`
- ✅ Preview output video → `/video` route + `<video>` element shown on `done` event
- ✅ Video saved to disk path → `write_video()` writes to `data["output_file"]`
- ✅ Error handling: empty folder, job collision (409), no faces, cancel, corrupt images
- ✅ CLI unchanged → `progress_callback=None` default preserves tqdm behaviour
- ✅ Cancel deletes partial output → `_emit_cancelled()` calls `os.remove()`
- ✅ No Node.js, no build step

**Placeholder scan:** No TBDs, no TODOs, no "similar to Task N" references. All code blocks are complete.

**Type consistency:**
- `progress_callback(phase: str, current: int, total: int)` — used identically in Task 1 (definition) and Task 2 (call sites `det_cb`, `write_cb`)
- `cancel_event: Optional[threading.Event]` — consistent across both modified functions and `_run_pipeline`
- `EyeCoords` imported correctly in test via `from face_timelapse import write_video, EyeCoords`
