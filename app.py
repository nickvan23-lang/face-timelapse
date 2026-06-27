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
  :root {
    --bg-color: #09090b;
    --card-bg: #18181b;
    --card-border: #27272a;
    --text-main: #f4f4f5;
    --text-muted: #a1a1aa;
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --danger: #ef4444;
    --danger-hover: #dc2626;
    --success: #10b981;
    --input-bg: #09090b;
    --input-border: #3f3f46;
    --ring-color: rgba(99, 102, 241, 0.3);
  }

  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-color);
    color: var(--text-main);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    padding: 3rem 1rem;
    line-height: 1.5;
  }

  .container {
    width: 100%;
    max-width: 720px;
    animation: fade-in-up 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }

  @keyframes fade-in-up {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
  }

  h1 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 2rem;
    color: #fff;
    text-align: center;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
  }

  .card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px -4px rgba(0,0,0,0.5);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }

  .field { margin-bottom: 1.25rem; }
  .field:last-child { margin-bottom: 0; }

  label.lbl {
    display: block;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }

  input[type=text], input[type=number] {
    width: 100%;
    background: var(--input-bg);
    border: 1px solid var(--input-border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: var(--text-main);
    font-size: 0.95rem;
    outline: none;
    transition: all 0.2s ease;
  }

  input:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--ring-color);
  }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.25rem; }

  .toggle-group {
    display: flex;
    background: var(--input-bg);
    border: 1px solid var(--input-border);
    border-radius: 10px;
    overflow: hidden;
  }

  .toggle-group input[type=radio] { display: none; }
  .toggle-group .tlbl {
    flex: 1;
    text-align: center;
    padding: 0.75rem;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-muted);
    transition: all 0.2s ease;
  }
  .toggle-group input:checked + .tlbl {
    background: var(--primary);
    color: #fff;
  }

  button {
    font-family: inherit;
    outline: none;
  }

  .btn-primary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background: var(--primary);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.875rem 1.75rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
    width: 100%;
  }

  .btn-primary:hover:not(:disabled) {
    background: var(--primary-hover);
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
  }

  .btn-primary:active:not(:disabled) {
    transform: translateY(1px);
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    box-shadow: none;
  }

  .btn-cancel {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background: transparent;
    color: var(--danger);
    border: 1px solid var(--danger);
    border-radius: 10px;
    padding: 0.875rem 1.75rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    width: 100%;
  }

  .btn-cancel:hover {
    background: rgba(239, 68, 68, 0.1);
    transform: translateY(-1px);
  }
  .btn-cancel:active {
    transform: translateY(1px);
  }

  .btn-row {
    display: flex;
    gap: 1rem;
    align-items: center;
    margin-top: 2rem;
  }

  #progress-section, #video-section {
    display: none;
    animation: fade-in 0.4s ease forwards;
  }

  @keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .phase-lbl {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text-main);
    margin-bottom: 0.75rem;
  }

  .track {
    background: var(--input-bg);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 0.75rem;
    border: 1px solid var(--card-border);
  }

  .fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    border-radius: 999px;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    width: 0%;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
  }

  .prog-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: var(--text-muted);
  }

  .err {
    color: var(--danger);
    font-size: 0.9rem;
    margin-top: 1rem;
    padding: 0.75rem;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 8px;
    border: 1px solid rgba(239, 68, 68, 0.2);
    display: none;
  }

  video {
    width: 100%;
    border-radius: 12px;
    background: #000;
    margin-top: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }

  .ok-lbl {
    font-size: 1rem;
    font-weight: 500;
    color: var(--success);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .drop-zone {
    border: 2px dashed var(--input-border);
    border-radius: 12px;
    padding: 2.5rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    background: var(--input-bg);
    user-select: none;
    position: relative;
    overflow: hidden;
  }

  .drop-zone:hover, .drop-zone.dz-hover {
    border-color: var(--primary);
    background: rgba(99, 102, 241, 0.05);
  }

  .dz-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    transition: transform 0.2s ease;
  }
  .drop-zone:hover .dz-icon {
    transform: scale(1.1);
  }

  .dz-text {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-main);
  }

  .dz-sub {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
  }

  .dz-picked {
    display: none;
    align-items: center;
    gap: 0.75rem;
    justify-content: center;
    font-size: 1.05rem;
    color: var(--text-main);
  }

  .dz-badge {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #818cf8;
  }

  .scan-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 0.75rem;
    min-height: 2.25rem;
  }

  .btn-scan {
    background: #27272a;
    color: var(--text-main);
    border: 1px solid #3f3f46;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn-scan:hover:not(:disabled) {
    background: #3f3f46;
  }

  .btn-scan:active:not(:disabled) {
    transform: translateY(1px);
  }

  .btn-scan:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .scan-ok { color: var(--success); font-size: 0.9rem; font-weight: 500; display:flex; align-items:center; gap:0.25rem; }
  .scan-err { color: var(--danger); font-size: 0.9rem; font-weight: 500; }
</style>
</head>
<body>
<div class="container">
  <h1>&#127916; Face Timelapse Generator</h1>

  <div class="card">
    <div class="field">
      <label class="lbl">Input Folder</label>
      <div class="drop-zone" id="drop-zone"
           onclick="dzClick(event)"
           ondragover="dzOver(event)"
           ondragleave="dzLeave(event)"
           ondrop="dzDrop(event)">
        <input type="file" id="folder-input" webkitdirectory multiple
               style="display:none" onchange="dzPicked(this)">
        <div id="dz-idle">
          <div class="dz-icon">&#128193;</div>
          <div class="dz-text">Drag a folder here</div>
          <div class="dz-sub">or click to browse</div>
        </div>
        <div class="dz-picked" id="dz-picked">
          <span>&#128193;</span>
          <span id="dz-name"></span>
          <span class="dz-badge" id="dz-client-count"></span>
        </div>
      </div>
      <input type="text" id="input_dir" placeholder="/full/path/to/folder"
             style="margin-top:0.75rem" oninput="dzResetScan()">
      <div class="scan-row">
        <div id="scan-status"></div>
        <button class="btn-scan" id="btn-scan" onclick="scanFolder()">Scan Folder</button>
      </div>
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
      <button class="btn-primary" id="btn-gen" onclick="startJob()" disabled>
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        Generate Timelapse
      </button>
      <button class="btn-cancel" id="btn-cancel" style="display:none" onclick="cancelJob()">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
        Cancel
      </button>
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
    <div class="ok-lbl">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
      Timelapse complete
    </div>
    <video id="vid" controls></video>
  </div>
</div>
<script>
let es = null;
let _scanned = false;
const _IMG_EXT = new Set(['.jpg','.jpeg','.png','.bmp','.tiff','.tif','.webp']);

function dzClick(e) {
  if (e.target.id === 'input_dir' || e.target.id === 'btn-scan') return;
  document.getElementById('folder-input').click();
}
function dzOver(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.add('dz-hover');
}
function dzLeave(e) {
  document.getElementById('drop-zone').classList.remove('dz-hover');
}
function dzDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('dz-hover');

  if (e.dataTransfer && e.dataTransfer.files.length) {
    dzPicked(e.dataTransfer);
  }
}
function dzPicked(input) {
  const files = Array.from(input.files);
  if (!files.length) return;

  // Try to get folder name from webkitRelativePath, fallback to looking at file paths if dropped, etc.
  let folderName = 'Folder';
  if (files[0].webkitRelativePath) {
    folderName = files[0].webkitRelativePath.split('/')[0];
  } else if (input.webkitEntries && input.webkitEntries.length) {
      folderName = input.webkitEntries[0].name;
  } else if (files[0].name) {
      // Just a rough fallback
      folderName = 'Selected items';
  }

  const imgCount = files.filter(f => _IMG_EXT.has('.' + f.name.split('.').pop().toLowerCase())).length;
  document.getElementById('dz-idle').style.display = 'none';
  const picked = document.getElementById('dz-picked');
  picked.style.display = 'flex';
  document.getElementById('dz-name').textContent = folderName;
  document.getElementById('dz-client-count').textContent = imgCount.toLocaleString() + ' images';

  // Update the input field if possible to guide the backend scanning (since browser security limits absolute path sharing)
  // Note: For a true desktop app disguised as a web app (which python + flask often is for local tools),
  // users typically still type the path. We'll rely on the manual path entry for the backend.
  // We can't automatically fill in `input_dir` with the absolute path because browsers hide it.
  // If the backend has a folder browser dialog endpoint, we could use that.

  dzResetScan();
  // We can try to auto-scan if they filled in the path
  if(document.getElementById('input_dir').value.trim() !== '') {
      scanFolder();
  }
}
function dzResetScan() {
  _scanned = false;
  document.getElementById('scan-status').innerHTML = '';
  document.getElementById('btn-gen').disabled = true;
}
async function scanFolder() {
  const path = document.getElementById('input_dir').value.trim();
  if (!path) { alert('Enter the full path to the folder first.'); return; }
  const btn = document.getElementById('btn-scan');
  btn.disabled = true;
  document.getElementById('scan-status').innerHTML = '<span style="color:var(--text-muted)">Scanning…</span>';
  try {
    const r = await fetch('/scan', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({path, recursive: true}),
    });
    const d = await r.json();
    if (d.error) {
      document.getElementById('scan-status').innerHTML = `<span class="scan-err">
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 4px;"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
        ${d.error}
      </span>`;
      document.getElementById('btn-gen').disabled = true;
      _scanned = false;
    } else {
      document.getElementById('scan-status').innerHTML =
        `<span class="scan-ok">
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
        ${d.count.toLocaleString()} photos found</span>`;
      document.getElementById('btn-gen').disabled = false;
      _scanned = true;
    }
  } catch(e) {
    document.getElementById('scan-status').innerHTML = `<span class="scan-err">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 4px;"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
      Connection error
    </span>`;
    document.getElementById('btn-gen').disabled = true;
  } finally {
    btn.disabled = false;
  }
}

function startJob() {
  const inputDir = document.getElementById('input_dir').value.trim();
  const outputFile = document.getElementById('output_file').value.trim();
  if (!inputDir) { alert('Please enter the full path to the folder and click Scan Folder first.'); return; }

  const payload = {
    input_dir: inputDir,
    output_file: outputFile || 'timelapse_output.mp4',
    fps: parseFloat(document.getElementById('fps').value),
    resolution: document.getElementById('resolution').value.trim(),
    zoom_factor: parseFloat(document.getElementById('zoom_factor').value),
    ema_alpha: parseFloat(document.getElementById('ema_alpha').value),
    eye_span: parseFloat(document.getElementById('eye_span').value),
    blur_bg: document.querySelector('input[name=blur_bg]:checked').value === 'true',
    recursive: true,
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
  if (es) {
    es.close();
  }

  es = new EventSource('/progress');

  es.addEventListener('message', function(e) {
    try {
      const ev = JSON.parse(e.data);
      if (ev.phase === 'ping') return;

      if (ev.phase === 'done') {
        setProg(100, 'Complete', ev.msg);
        showVideo();
        resetButtons();
        if (es) { es.close(); es = null; }
      } else if (ev.phase === 'error') {
        showErr(ev.msg);
        resetButtons();
        if (es) { es.close(); es = null; }
      } else {
        const lbl = ev.phase === 'detection' ? 'Detecting faces…' : 'Writing video…';
        setProg(ev.pct, lbl, ev.msg);
      }
    } catch (err) {
      console.error('Error parsing SSE data', err);
    }
  });

  es.addEventListener('error', function(e) {
    console.error('SSE Error', e);
    // Usually means the connection closed. We can clean up if we want, but EventSource auto-reconnects by default.
    // If we want to prevent auto-reconnect on server crash, we could do this:
    if (es && es.readyState === EventSource.CLOSED) {
        showErr("Connection lost. Job may have failed or finished unexpectedly.");
        resetButtons();
        es.close();
        es = null;
    }
  });
}

function setProg(pct, lbl, msg) {
  document.getElementById('phase-lbl').textContent = lbl;
  document.getElementById('fill').style.width = pct + '%';
  document.getElementById('prog-pct').textContent = pct + '%';
  document.getElementById('prog-msg').textContent = msg;
}

function showErr(msg) {
  const el = document.getElementById('err-msg');
  el.innerHTML = `
    <div style="display:flex; align-items:center; gap: 0.5rem; font-weight: 600; margin-bottom: 0.25rem;">
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
      Error
    </div>
    ${msg}
  `;
  el.style.display = 'block';
}

function showVideo() {
  document.getElementById('video-section').style.display = 'block';
  const v = document.getElementById('vid');
  v.src = '/video?t=' + Date.now();
  v.load();
}

function cancelJob() {
  fetch('/cancel', {method: 'POST'})
    .then(r => r.json())
    .then(d => {
       console.log('Cancellation requested');
       document.getElementById('btn-cancel').disabled = true;
       document.getElementById('btn-cancel').innerHTML = 'Cancelling...';
    })
    .catch(e => console.error(e));
}

function resetUI() {
  document.getElementById('progress-section').style.display = 'block';
  document.getElementById('video-section').style.display = 'none';
  document.getElementById('err-msg').style.display = 'none';
  document.getElementById('fill').style.width = '0%';
  document.getElementById('prog-pct').textContent = '0%';
  document.getElementById('prog-msg').textContent = '';
  document.getElementById('phase-lbl').textContent = 'Starting…';

  document.getElementById('btn-gen').style.display = 'none';

  const btnCancel = document.getElementById('btn-cancel');
  btnCancel.style.display = 'inline-flex';
  btnCancel.disabled = false;
  btnCancel.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg> Cancel';
}

function resetButtons() {
  document.getElementById('btn-gen').style.display = 'inline-flex';
  document.getElementById('btn-gen').disabled = !_scanned;
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
