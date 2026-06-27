import multiprocessing as mp
import threading
from typing import Callable, List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict
from tqdm import tqdm

import cv2
import mediapipe as mp
import mediapipe.python.solutions


from facelapse.models import DetectionResult, EyeCoords
import logging
log = logging.getLogger(__name__)

_LM_LEFT_IRIS = 468
_LM_RIGHT_IRIS = 473
_LM_LEFT_INNER = 133
_LM_LEFT_OUTER = 33
_LM_RIGHT_INNER = 362
_LM_RIGHT_OUTER = 263

_face_mesh_local = threading.local()


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
            l = lms[_LM_LEFT_IRIS]
            r = lms[_LM_RIGHT_IRIS]
            result.eyes = EyeCoords(lx=l.x, ly=l.y, rx=r.x, ry=r.y)
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
                batch = tasks[start : start + batch_size]
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
