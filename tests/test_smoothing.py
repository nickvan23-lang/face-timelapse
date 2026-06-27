import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from facelapse.models import DetectionResult, EyeCoords
from facelapse.smoothing import smooth_eye_coords, EMASmoother

def test_ema_smoother_initialization():
    smoother = EMASmoother(alpha=0.5)
    assert smoother.val_x is None
    assert smoother.val_y is None

    # First update should just set the value
    x, y = smoother.update(100.0, 200.0)
    assert x == 100.0
    assert y == 200.0

def test_ema_smoother_update():
    smoother = EMASmoother(alpha=0.5)
    smoother.update(100.0, 100.0)

    # Second update should blend 50/50
    x, y = smoother.update(200.0, 0.0)
    assert x == 150.0  # 0.5 * 200 + 0.5 * 100
    assert y == 50.0   # 0.5 * 0 + 0.5 * 100

def test_smooth_eye_coords():
    detections = [
        DetectionResult(path="1.jpg", index=0, eyes=EyeCoords(lx=10, ly=10, rx=20, ry=20)),
        DetectionResult(path="2.jpg", index=1, eyes=EyeCoords(lx=30, ly=30, rx=40, ry=40)),
        DetectionResult(path="3.jpg", index=2, eyes=None), # Missing detection
        DetectionResult(path="4.jpg", index=3, eyes=EyeCoords(lx=50, ly=50, rx=60, ry=60))
    ]

    smoothed = smooth_eye_coords(detections, alpha=0.5)

    assert len(smoothed) == 4

    # First frame should be exact
    assert smoothed[0].lx == 10

    # Second frame should be smoothed
    assert smoothed[1].lx == 20 # 0.5 * 30 + 0.5 * 10

    # Third frame should carry over the previous smoothed value
    assert smoothed[2].lx == 20

    # Fourth frame should resume smoothing from the carried over value
    assert smoothed[3].lx == 35 # 0.5 * 50 + 0.5 * 20
