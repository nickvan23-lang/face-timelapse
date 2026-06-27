import threading
from typing import Callable, List, Optional
import cv2
import numpy as np
from pathlib import Path
import math
from tqdm import tqdm

from facelapse.models import EyeCoords
import logging
log = logging.getLogger(__name__)

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
