from dataclasses import dataclass
from typing import Optional

@dataclass
class EyeCoords:
    lx: float
    ly: float
    rx: float
    ry: float

    def to_pixels(self, img_w: int, img_h: int) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.lx * img_w, self.ly * img_h), (self.rx * img_w, self.ry * img_h))

@dataclass
class DetectionResult:
    path: str
    index: int
    eyes: Optional[EyeCoords] = None
    error: str = ""
