from facelapse.models import EyeCoords, DetectionResult
from facelapse.utils import discover_images
from facelapse.detection import detect_all_faces
from facelapse.smoothing import smooth_eye_coords
from facelapse.video import write_video
