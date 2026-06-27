#!/usr/bin/env python3
"""
face_timelapse.py — Production-grade face-centered timelapse generator.

Pipeline:
  Pass 1 (parallel)  — MediaPipe face/iris detection across all images
  Smooth             — EMA over raw landmarks to eliminate jitter
  Pass 2 (streaming) — Per-frame affine warp + video write (no full dataset in RAM)

Usage:
  python face_timelapse.py --input_dir ./photos --output_file out.mp4 --fps 24
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import threading

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXT: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)

# MediaPipe Face Mesh landmark indices (478-point model with refine_landmarks=True)
# Iris centres (only available when refine_landmarks=True)
_LM_LEFT_IRIS = 468
_LM_RIGHT_IRIS = 473

# Fallback: eye-corner pairs to average when iris isn't available
_LM_LEFT_INNER = 133
_LM_LEFT_OUTER = 33
_LM_RIGHT_INNER = 362
_LM_RIGHT_OUTER = 263


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EyeCoords:
    """
    Normalised eye coordinates in [0, 1] image space.

    Attributes:
        lx, ly: Left pupil/iris centre (normalised).
        rx, ry: Right pupil/iris centre (normalised).
    """

    lx: float
    ly: float
    rx: float
    ry: float

    # ------------------------------------------------------------------
    def to_pixels(self, w: int, h: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Convert to absolute pixel coordinates for an image of size (w, h)."""
        return (self.lx * w, self.ly * h), (self.rx * w, self.ry * h)

    def midpoint(self) -> Tuple[float, float]:
        """Normalised midpoint between the two eyes."""
        return (self.lx + self.rx) / 2.0, (self.ly + self.ry) / 2.0

    def distance(self) -> float:
        """Euclidean eye distance in normalised space."""
        return math.hypot(self.rx - self.lx, self.ry - self.ly)


@dataclass(slots=True)
class DetectionResult:
    """
    Output of face detection on a single image.

    Attributes:
        path:  Absolute image path.
        index: Original sort-order index (used to re-sort after parallel scatter).
        eyes:  Detected eye coordinates, or None if detection failed.
        error: Human-readable error string, or None on success.
    """

    path: str
    index: int
    eyes: Optional[EyeCoords] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# EMA Smoother
# ---------------------------------------------------------------------------

class EMASmoother:
    """
    Exponential Moving Average over four eye-coordinate scalars.

    α close to 1  → fast response, little smoothing (more jitter).
    α close to 0  → heavy smoothing, slower response (more lag).
    """

    __slots__ = ("alpha", "_lx", "_ly", "_rx", "_ry", "_ready")

    def __init__(self, alpha: float) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"EMA alpha must be in (0, 1]; got {alpha}")
        self.alpha = alpha
        self._lx = self._ly = self._rx = self._ry = 0.0
        self._ready = False

    def update(self, eyes: EyeCoords) -> None:
        """Incorporate a new detection into the smoothed state."""
        a = self.alpha
        if not self._ready:
            self._lx, self._ly = eyes.lx, eyes.ly
            self._rx, self._ry = eyes.rx, eyes.ry
            self._ready = True
        else:
            self._lx = a * eyes.lx + (1 - a) * self._lx
            self._ly = a * eyes.ly + (1 - a) * self._ly
            self._rx = a * eyes.rx + (1 - a) * self._rx
            self._ry = a * eyes.ry + (1 - a) * self._ry

    @property
    def ready(self) -> bool:
        return self._ready

    def value(self) -> EyeCoords:
        """Current smoothed state as EyeCoords."""
        return EyeCoords(self._lx, self._ly, self._rx, self._ry)


# ---------------------------------------------------------------------------
# Worker — face detection (runs inside subprocess)
# ---------------------------------------------------------------------------

# Module-level FaceMesh instance, one per worker process.
_worker_face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None  # type: ignore[name-defined]


def _worker_init() -> None:
    """Called once per worker process to initialise MediaPipe."""
    global _worker_face_mesh
    _worker_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,          # enables 478-point model with iris
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _worker_detect(args: Tuple[str, int]) -> DetectionResult:
    """
    Detect face landmarks in a single image.

    This function executes in a subprocess (no shared state with the main
    process).  Each process owns its own ``_worker_face_mesh`` instance.

    Args:
        args: (absolute_image_path, sort_index)

    Returns:
        DetectionResult populated with eye coordinates or an error string.
    """
    path, idx = args
    result = DetectionResult(path=path, index=idx)

    try:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            result.error = "cv2.imread returned None (corrupt or unreadable)"
            return result

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_result = _worker_face_mesh.process(img_rgb)  # type: ignore[union-attr]

        if not mp_result.multi_face_landmarks:
            result.error = "No face detected"
            return result

        lms = mp_result.multi_face_landmarks[0].landmark

        if len(lms) > _LM_RIGHT_IRIS:
            # 478-point model: use iris centres (most accurate)
            l_iris = lms[_LM_LEFT_IRIS]
            r = lms[_LM_RIGHT_IRIS]
            result.eyes = EyeCoords(lx=l_iris.x, ly=l_iris.y, rx=r.x, ry=r.y)
        else:
            # 468-point fallback: average inner+outer eye corners
            li = lms[_LM_LEFT_INNER]
            lo = lms[_LM_LEFT_OUTER]
            ri = lms[_LM_RIGHT_INNER]
            ro = lms[_LM_RIGHT_OUTER]
            result.eyes = EyeCoords(
                lx=(li.x + lo.x) / 2.0,
                ly=(li.y + lo.y) / 2.0,
                rx=(ri.x + ro.x) / 2.0,
                ry=(ri.y + ro.y) / 2.0,
            )

    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)

    return result


# ---------------------------------------------------------------------------
# Pass 1 — parallel face detection
# ---------------------------------------------------------------------------

def detect_all_faces(
    image_paths: List[str],
    num_workers: int,
    batch_size: int = 128,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[DetectionResult]:
    """
    Run face detection across every image using a process pool.

    Images are dispatched in batches to keep the IPC queue from ballooning.
    Results are re-sorted to match the original path order.

    Args:
        image_paths: Ordered list of absolute image paths.
        num_workers:  Number of parallel worker processes.
        batch_size:   Futures batch size (tune for memory vs. throughput).

    Returns:
        List[DetectionResult] in the same order as ``image_paths``.
    """
    tasks = [(p, i) for i, p in enumerate(image_paths)]
    bucket: Dict[int, DetectionResult] = {}

    log.info("Pass 1 — face detection  (%d workers, %d images)", num_workers, len(tasks))

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
                batch = tasks[start:start + batch_size]
                futures = {pool.submit(_worker_detect, t): t for t in batch}

                for fut in as_completed(futures):
                    original_task = futures[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        res = DetectionResult(
                            path=original_task[0],
                            index=original_task[1],
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

    # Fill any indices that were never collected (cancel mid-batch)
    for i in range(len(tasks)):
        if i not in bucket:
            bucket[i] = DetectionResult(path=tasks[i][0], index=i, error="Cancelled")

    return [bucket[i] for i in range(len(image_paths))]


# ---------------------------------------------------------------------------
# Smooth — EMA over raw detections
# ---------------------------------------------------------------------------

def smooth_eye_coords(
    detections: List[DetectionResult],
    alpha: float,
) -> List[Optional[EyeCoords]]:
    """
    Apply EMA smoothing to produce a stable coordinate per frame.

    Frames with no detection inherit the previous smoothed state so the
    video doesn't jump or crash.  Frames before any detection are None
    (skipped during video write).

    Args:
        detections: Raw detection results in temporal order.
        alpha:      EMA alpha  [0.0 < alpha ≤ 1.0].

    Returns:
        List of smoothed EyeCoords (or None for pre-first-detection frames).
    """
    smoother = EMASmoother(alpha)
    smoothed: List[Optional[EyeCoords]] = []
    no_face = 0

    for det in detections:
        if det.eyes is not None:
            smoother.update(det.eyes)
        else:
            no_face += 1

        smoothed.append(smoother.value() if smoother.ready else None)

    if no_face:
        log.warning(
            "%d/%d frames had no face detected — using smoothed fallback.",
            no_face,
            len(detections),
        )

    return smoothed


# ---------------------------------------------------------------------------
# Affine transformation helpers
# ---------------------------------------------------------------------------

def build_warp_matrix(
    eyes: EyeCoords,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    zoom: float,
    eye_span_fraction: float,
) -> np.ndarray:
    """
    Build a 2×3 affine matrix that:

    1. Rotates the image so the two eyes are perfectly horizontal.
    2. Scales so the inter-eye distance equals ``eye_span_fraction * dst_w * zoom``.
    3. Translates so the eye midpoint lands exactly at the output frame centre.

    Math derivation
    ---------------
    Let angle = atan2(ry - ly, rx - lx) be the tilt of the eye axis.
    We want to apply R(−angle) (rotate by negative angle) and scale ``s``
    around the source eye midpoint (cx_s, cy_s), then translate the result
    to the destination frame centre (cx_d, cy_d).

    For a source point (x, y):
        x' = s·cos(angle)·(x − cx_s) + s·sin(angle)·(y − cy_s) + cx_d
        y' = −s·sin(angle)·(x − cx_s) + s·cos(angle)·(y − cy_s) + cy_d

    Which gives the 2×3 affine matrix:
        M = [[s·cos θ,  s·sin θ,  cx_d − s·cos θ·cx_s − s·sin θ·cy_s],
             [−s·sin θ, s·cos θ,  cy_d + s·sin θ·cx_s − s·cos θ·cy_s]]

    Args:
        eyes:             Smoothed eye coordinates in normalised [0, 1] space.
        src_w, src_h:     Source image pixel dimensions.
        dst_w, dst_h:     Output frame pixel dimensions.
        zoom:             Additional zoom multiplier (1.0 = default framing).
        eye_span_fraction: Desired inter-eye distance as fraction of ``dst_w``.

    Returns:
        2×3 float64 affine matrix ready for cv2.warpAffine.
    """
    # Eye positions in source pixels
    (lx, ly), (rx, ry) = eyes.to_pixels(src_w, src_h)

    # Rotation angle (tilt of the eye-to-eye vector)
    theta = math.atan2(ry - ly, rx - lx)

    # Scale: map current inter-eye pixel distance → target
    eye_dist_px = math.hypot(rx - lx, ry - ly)
    target_dist = eye_span_fraction * dst_w * zoom
    s = target_dist / max(eye_dist_px, 1e-6)

    # Clamp to avoid absurdly large transformations on partial faces
    s = min(s, 20.0)

    # Source and destination centres
    cx_s = (lx + rx) / 2.0
    cy_s = (ly + ry) / 2.0
    cx_d = dst_w / 2.0
    cy_d = dst_h / 2.0

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    M = np.array(
        [
            [
                s * cos_t,
                s * sin_t,
                cx_d - s * cos_t * cx_s - s * sin_t * cy_s,
            ],
            [
                -s * sin_t,
                s * cos_t,
                cy_d + s * sin_t * cx_s - s * cos_t * cy_s,
            ],
        ],
        dtype=np.float64,
    )
    return M


def apply_warp(
    img: np.ndarray,
    M: np.ndarray,
    dst_w: int,
    dst_h: int,
    blur_bg: bool,
) -> np.ndarray:
    """
    Warp ``img`` with affine matrix ``M`` into a (dst_h × dst_w) frame.

    Out-of-bounds areas (from rotation/scale) are filled with:
    - a darkened, blurred version of the source image  (``blur_bg=True``)
    - solid black                                       (``blur_bg=False``)

    Args:
        img:     Source BGR image.
        M:       2×3 affine matrix.
        dst_w:   Output frame width in pixels.
        dst_h:   Output frame height in pixels.
        blur_bg: Whether to use a blurred fill instead of black.

    Returns:
        Output BGR frame of shape (dst_h, dst_w, 3).
    """
    if blur_bg:
        # Build a blurred, scaled background plate
        bg = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=35, sigmaY=35)
        bg = (bg.astype(np.float32) * 0.35).astype(np.uint8)  # darken

        # Warp on top; BORDER_TRANSPARENT leaves bg pixels untouched where
        # the source has no coverage.
        frame = cv2.warpAffine(
            img,
            M,
            (dst_w, dst_h),
            dst=bg.copy(),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
    else:
        frame = cv2.warpAffine(
            img,
            M,
            (dst_w, dst_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    return frame


# ---------------------------------------------------------------------------
# Pass 2 — streaming video write
# ---------------------------------------------------------------------------

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
    """
    Read images one-by-one (streaming), apply the affine warp, write frames.

    At no point is more than one decoded image held in RAM alongside the
    output frame.

    Args:
        image_paths:       Ordered list of image paths.
        smoothed:          Per-frame smoothed eye coordinates (None = skip).
        output_file:       Destination .mp4 path.
        fps:               Output frames per second.
        dst_w, dst_h:      Output frame dimensions.
        zoom:              Zoom multiplier.
        eye_span_fraction: Inter-eye fraction of output width.
        blur_bg:           Use blurred background fill.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (dst_w, dst_h))

    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open '{output_file}'. "
            "Check codec support and output directory permissions."
        )

    written = skipped = 0

    try:
        log.info("Pass 2 — writing video → %s", output_file)
        use_tqdm = progress_callback is None
        total_frames = len(image_paths)
        pbar = (
            tqdm(total=total_frames, desc="Writing frames", unit="frame", dynamic_ncols=True)
            if use_tqdm
            else None
        )
        try:
            for path, eyes in zip(image_paths, smoothed):
                if cancel_event and cancel_event.is_set():
                    break
                if use_tqdm:
                    pbar.update(1)  # type: ignore[union-attr]

                if eyes is None:
                    # Before first detection — no valid reference state yet
                    skipped += 1
                    if not use_tqdm:
                        progress_callback("writing", written + skipped, total_frames)  # type: ignore[misc]
                    continue

                try:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is None:
                        log.debug("Skipping unreadable image: %s", path)
                        skipped += 1
                        if not use_tqdm:
                            progress_callback("writing", written + skipped, total_frames)  # type: ignore[misc]
                        continue

                    h, w = img.shape[:2]
                    M = build_warp_matrix(eyes, w, h, dst_w, dst_h, zoom, eye_span_fraction)
                    frame = apply_warp(img, M, dst_w, dst_h, blur_bg)
                    writer.write(frame)
                    written += 1
                    if not use_tqdm:
                        progress_callback("writing", written + skipped, total_frames)  # type: ignore[misc]

                except Exception as exc:  # noqa: BLE001
                    log.debug("Error processing %s: %s", path, exc)
                    skipped += 1
                    if not use_tqdm:
                        progress_callback("writing", written + skipped, total_frames)  # type: ignore[misc]
        finally:
            if pbar:
                pbar.close()
    finally:
        writer.release()

    log.info(
        "Done — %d frames written, %d skipped → %s",
        written,
        skipped,
        output_file,
    )


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def _natural_sort_key(s: str) -> List:
    """Split into numeric and text chunks for human-natural ordering."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def discover_images(input_dir: str, recursive: bool = False) -> List[str]:
    """
    Scan ``input_dir`` for supported images and return a naturally sorted list.

    Args:
        input_dir: Path to the directory containing source images.
        recursive: If True, walk all subfolders via rglob.

    Returns:
        Sorted list of absolute image path strings.

    Raises:
        ValueError: If the directory doesn't exist or contains no images.
    """
    p = Path(input_dir).resolve()
    if not p.is_dir():
        raise ValueError(f"Input directory does not exist: {p}")

    glob = p.rglob("*") if recursive else p.iterdir()
    paths = [str(f) for f in glob if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]

    if not paths:
        suffix = " (including subfolders)" if recursive else f". Supported extensions: {sorted(SUPPORTED_EXT)}"
        raise ValueError(f"No supported images found in '{p}'{suffix}")

    paths.sort(key=_natural_sort_key)
    log.info("Found %d images in %s (recursive=%s)", len(paths), p, recursive)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    cpu_default = max(1, multiprocessing.cpu_count() - 1)

    parser = argparse.ArgumentParser(
        prog="face_timelapse",
        description=(
            "Generate a stabilised, eye-centred timelapse video from a "
            "large directory of photos."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--input_dir", "-i",
        required=True,
        metavar="DIR",
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output_file", "-o",
        default="timelapse_output.mp4",
        metavar="FILE",
        help="Output .mp4 file path.",
    )

    # Video params
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=24.0,
        help="Output frames per second.",
    )
    parser.add_argument(
        "--resolution", "-r",
        default="1920x1080",
        metavar="WxH",
        help="Output resolution, e.g. 1920x1080 or 1080x1080.",
    )
    parser.add_argument(
        "--zoom_factor", "-z",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help=(
            "Zoom multiplier on the face. "
            "1.0 = default framing; 1.5 = 50 %% closer."
        ),
    )

    # Stabilisation
    parser.add_argument(
        "--ema_alpha", "-a",
        type=float,
        default=0.25,
        metavar="FLOAT",
        help=(
            "EMA smoothing factor (0 < α ≤ 1). "
            "Lower values give smoother but laggier stabilisation."
        ),
    )
    parser.add_argument(
        "--eye_span",
        type=float,
        default=0.35,
        metavar="FLOAT",
        help=(
            "Desired inter-eye distance as a fraction of the output width. "
            "Controls how large the face appears. "
            "Overridden by --zoom_factor."
        ),
    )

    # Performance
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=cpu_default,
        metavar="N",
        help="Number of parallel face-detection worker processes.",
    )

    # Aesthetics
    parser.add_argument(
        "--no_blur_bg",
        action="store_true",
        help=(
            "Fill letterbox/pillarbox areas with solid black instead of a "
            "blurred background."
        ),
    )

    # Debug
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    return parser


def parse_resolution(spec: str) -> Tuple[int, int]:
    """
    Parse a resolution string like '1920x1080' into (width, height).

    Raises:
        argparse.ArgumentTypeError: On invalid format.
    """
    parts = re.split(r"[xX×]", spec.strip())
    try:
        w, h = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{spec}'. Expected format: WxH (e.g. 1920x1080)"
        )
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive integers.")
    return w, h


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full detection → smooth → write pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate resolution
    try:
        dst_w, dst_h = parse_resolution(args.resolution)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if not (0.0 < args.ema_alpha <= 1.0):
        parser.error("--ema_alpha must be in the range (0, 1].")
    if args.zoom_factor <= 0:
        parser.error("--zoom_factor must be positive.")
    if not (0.01 < args.eye_span < 1.0):
        parser.error("--eye_span must be between 0.01 and 1.0.")

    log.info(
        "Config: %dx%d @ %.1f fps | zoom=%.2f | ema_alpha=%.2f | "
        "eye_span=%.2f | workers=%d",
        dst_w, dst_h, args.fps, args.zoom_factor,
        args.ema_alpha, args.eye_span, args.workers,
    )

    # ── Step 1: Discover images ──────────────────────────────────────────
    try:
        image_paths = discover_images(args.input_dir)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # ── Step 2: Parallel face detection ─────────────────────────────────
    t0 = time.perf_counter()
    detections = detect_all_faces(image_paths, num_workers=args.workers)
    log.info("Detection pass: %.1fs", time.perf_counter() - t0)

    detected = sum(1 for d in detections if d.eyes is not None)
    log.info("Faces detected in %d / %d images (%.1f%%)",
             detected, len(detections), 100.0 * detected / len(detections))

    if detected == 0:
        log.error(
            "No faces detected in any image. "
            "Check that photos contain a clearly visible face."
        )
        sys.exit(1)

    # ── Step 3: EMA smoothing ────────────────────────────────────────────
    smoothed = smooth_eye_coords(detections, alpha=args.ema_alpha)

    # ── Step 4: Streaming video write ────────────────────────────────────
    t1 = time.perf_counter()
    try:
        write_video(
            image_paths=image_paths,
            smoothed=smoothed,
            output_file=args.output_file,
            fps=args.fps,
            dst_w=dst_w,
            dst_h=dst_h,
            zoom=args.zoom_factor,
            eye_span_fraction=args.eye_span,
            blur_bg=not args.no_blur_bg,
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info("Video write: %.1fs", time.perf_counter() - t1)
    log.info("Output saved → %s", args.output_file)


if __name__ == "__main__":
    # Required guard for ProcessPoolExecutor on Windows / PyInstaller
    multiprocessing.freeze_support()
    main()
