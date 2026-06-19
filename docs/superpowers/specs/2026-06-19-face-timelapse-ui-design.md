# Face Timelapse — Web UI Design

**Date:** 2026-06-19
**Status:** Approved

---

## Overview

Add a browser-based UI to `face_timelapse.py` so non-CLI users can configure and run the pipeline, watch real-time progress, and preview the finished video — all in Chrome.

---

## Architecture

```
app.py (Flask server)
├── GET  /          → serves the single-page UI (HTML embedded in app.py)
├── POST /run       → validates args, spawns pipeline in a background thread
├── GET  /progress  → SSE stream — emits JSON progress events
├── GET  /video     → streams the finished .mp4 for in-browser playback
└── POST /cancel    → signals the background thread to stop
```

- `face_timelapse.py` is imported as a module; its two passes (detection + write) run on a background thread so Flask stays responsive.
- Progress is passed via a thread-safe `queue.Queue`; the `/progress` SSE endpoint drains it and pushes events to the browser.
- The UI is a single `index.html` string embedded in `app.py` — no separate files, no build step.
- On startup, `app.py` calls `webbrowser.open("http://localhost:5000")` to open Chrome automatically.

---

## UI Layout

Single dark-themed scrollable page:

```
┌─────────────────────────────────────────────────────┐
│  🎬  Face Timelapse Generator                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Input Folder   [ /path/to/photos          ] Browse │
│  Output File    [ timelapse_output.mp4     ] Browse │
│                                                     │
│  FPS     [ 24  ]   Resolution  [ 1920x1080 ]        │
│  Zoom    [ 1.0 ]   EMA Alpha   [ 0.25      ]        │
│  Eye Span [ 0.35]  Background  ( Blurred ● Black )  │
│                                                     │
│  [ ▶  Generate Timelapse ]   [ ✕ Cancel ]          │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Phase: Detecting faces...                          │
│  ████████████░░░░░░░░  62%   1,240 / 2,000 imgs    │
│  ETA: ~1m 23s                                       │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐   │
│  │          (video player — shown on done)     │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

- Progress section is hidden until a job starts.
- Video player appears only when the job completes successfully.
- Errors appear inline below the progress bar in red.
- Generate button is disabled while a job is active.

---

## SSE Event Schema

Every event pushed from `/progress` is a JSON line:

```json
{ "phase": "detection", "pct": 62,  "msg": "1,240 / 2,000 images",              "eta": "1m 23s" }
{ "phase": "writing",   "pct": 88,  "msg": "1,760 / 2,000 frames",              "eta": "0m 14s" }
{ "phase": "done",      "pct": 100, "msg": "Output saved → timelapse_output.mp4", "eta": "" }
{ "phase": "error",     "pct": 0,   "msg": "No faces detected in any image.",    "eta": "" }
```

---

## Error Handling

| Scenario | Handling |
|---|---|
| Invalid / empty folder | Form validation before job starts; red inline message |
| Job already running | Generate button disabled; `/run` returns 409 |
| No faces detected | `error` SSE event shown in red below progress bar |
| Corrupt images | Silently skipped; count reported in completion message |
| User cancels | `threading.Event` signals background thread; partial output deleted |

---

## Changes to `face_timelapse.py`

- Add optional `progress_callback: Optional[Callable[[str, int, int], None]]` parameter to `detect_all_faces()` and `write_video()`.
- When `None`, behaviour is unchanged — tqdm runs as before (CLI mode unaffected).
- When provided, callback is called with `(phase, current, total)` on each step; the Flask layer converts this to SSE events.

---

## Files Produced

| File | Purpose |
|---|---|
| `app.py` | Flask server + embedded HTML/CSS/JS UI |
| `face_timelapse.py` | Modified to accept optional progress callback |
| `requirements.txt` | Add `flask>=3.0` |

---

## Run Instructions

```bash
pip install flask
python app.py
# Browser opens automatically at http://localhost:5000
```
