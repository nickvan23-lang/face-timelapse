from typing import List, Optional, Tuple
from facelapse.models import DetectionResult, EyeCoords

class EMASmoother:
    def __init__(self, alpha: float = 0.25):
        self.alpha = max(0.01, min(1.0, alpha))
        self.val_x: Optional[float] = None
        self.val_y: Optional[float] = None

    def update(self, new_x: float, new_y: float) -> Tuple[float, float]:
        if self.val_x is None or self.val_y is None:
            self.val_x = new_x
            self.val_y = new_y
        else:
            self.val_x = self.alpha * new_x + (1 - self.alpha) * self.val_x
            self.val_y = self.alpha * new_y + (1 - self.alpha) * self.val_y
        return self.val_x, self.val_y

def smooth_eye_coords(
    detections: List[DetectionResult],
    alpha: float = 0.25
) -> List[Optional[EyeCoords]]:
    left_smoother = EMASmoother(alpha)
    right_smoother = EMASmoother(alpha)

    smoothed: List[Optional[EyeCoords]] = []

    for det in detections:
        if det.eyes is None:
            if left_smoother.val_x is not None:
                smoothed.append(EyeCoords(
                    lx=left_smoother.val_x,
                    ly=left_smoother.val_y,
                    rx=right_smoother.val_x,  # type: ignore
                    ry=right_smoother.val_y   # type: ignore
                ))
            else:
                smoothed.append(None)
            continue

        lx, ly = left_smoother.update(det.eyes.lx, det.eyes.ly)
        rx, ry = right_smoother.update(det.eyes.rx, det.eyes.ry)

        smoothed.append(EyeCoords(
            lx=lx, ly=ly,
            rx=rx, ry=ry
        ))

    return smoothed
