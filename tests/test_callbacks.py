"""Tests for progress_callback and cancel_event integration."""
import threading
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from facelapse import write_video, EyeCoords


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
