import sys, os
import pytest
import math
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from face_timelapse import (
    EyeCoords, EMASmoother, _worker_init, _worker_detect,
    detect_all_faces, smooth_eye_coords, build_warp_matrix,
    apply_warp, write_video, _natural_sort_key, discover_images,
    build_parser, parse_resolution, main, DetectionResult
)

def test_eye_coords():
    eyes = EyeCoords(0.1, 0.2, 0.9, 0.2)
    assert eyes.to_pixels(100, 200) == ((10.0, 40.0), (90.0, 40.0))
    assert eyes.midpoint() == (0.5, 0.2)
    assert eyes.distance() == 0.8

def test_ema_smoother():
    with pytest.raises(ValueError):
        EMASmoother(0.0)
    with pytest.raises(ValueError):
        EMASmoother(1.1)

    smoother = EMASmoother(0.5)
    assert not smoother.ready

    eyes1 = EyeCoords(0.0, 0.0, 1.0, 1.0)
    smoother.update(eyes1)
    assert smoother.ready
    assert smoother.value() == eyes1

    eyes2 = EyeCoords(1.0, 1.0, 2.0, 2.0)
    smoother.update(eyes2)
    assert smoother.value() == EyeCoords(0.5, 0.5, 1.5, 1.5)

@patch('face_timelapse.cv2.imread')
def test_worker_detect_no_image(mock_imread):
    mock_imread.return_value = None
    res = _worker_detect(("bad.jpg", 0))
    assert res.error == "cv2.imread returned None (corrupt or unreadable)"

@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.cv2.cvtColor')
def test_worker_detect_no_face(mock_cvt, mock_imread):
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    import face_timelapse
    # Just mock it out so we bypass init issue
    face_timelapse._worker_face_mesh = MagicMock()
    face_timelapse._worker_face_mesh.process.return_value.multi_face_landmarks = None

    res = _worker_detect(("img.jpg", 0))
    assert res.error == "No face detected"

@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.cv2.cvtColor')
def test_worker_detect_exception(mock_cvt, mock_imread):
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    import face_timelapse
    face_timelapse._worker_face_mesh = MagicMock()
    face_timelapse._worker_face_mesh.process.side_effect = Exception("Test Error")

    res = _worker_detect(("img.jpg", 0))
    assert res.error == "Test Error"

@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.cv2.cvtColor')
def test_worker_detect_success(mock_cvt, mock_imread):
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    import face_timelapse
    face_timelapse._worker_face_mesh = MagicMock()

    # Create fake landmarks
    class Landmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    lms = [Landmark(0, 0) for _ in range(500)]
    lms[face_timelapse._LM_LEFT_IRIS] = Landmark(0.1, 0.2)
    lms[face_timelapse._LM_RIGHT_IRIS] = Landmark(0.9, 0.2)

    face_timelapse._worker_face_mesh.process.return_value.multi_face_landmarks = [
        MagicMock(landmark=lms)
    ]

    res = _worker_detect(("img.jpg", 0))
    assert res.eyes.lx == 0.1
    assert res.eyes.rx == 0.9
    assert res.error is None

    # 468 point fallback
    lms = [Landmark(0, 0) for _ in range(470)]
    lms[face_timelapse._LM_LEFT_INNER] = Landmark(0.2, 0.2)
    lms[face_timelapse._LM_LEFT_OUTER] = Landmark(0.1, 0.2)
    lms[face_timelapse._LM_RIGHT_INNER] = Landmark(0.8, 0.2)
    lms[face_timelapse._LM_RIGHT_OUTER] = Landmark(0.9, 0.2)

    face_timelapse._worker_face_mesh.process.return_value.multi_face_landmarks = [
        MagicMock(landmark=lms)
    ]

    res = _worker_detect(("img.jpg", 0))
    assert math.isclose(res.eyes.lx, 0.15)
    assert math.isclose(res.eyes.rx, 0.85)
    assert res.error is None

def test_smooth_eye_coords():
    res1 = DetectionResult("p1", 0, EyeCoords(0.0, 0.0, 1.0, 1.0))
    res2 = DetectionResult("p2", 1, None)
    res3 = DetectionResult("p3", 2, EyeCoords(1.0, 1.0, 2.0, 2.0))

    smoothed = smooth_eye_coords([res2, res1, res2, res3], 0.5)

    assert smoothed[0] is None
    assert smoothed[1] == EyeCoords(0.0, 0.0, 1.0, 1.0)
    assert smoothed[2] == EyeCoords(0.0, 0.0, 1.0, 1.0)
    assert smoothed[3] == EyeCoords(0.5, 0.5, 1.5, 1.5)

def test_build_warp_matrix():
    eyes = EyeCoords(0.4, 0.5, 0.6, 0.5)
    M = build_warp_matrix(eyes, 1000, 1000, 500, 500, 1.0, 0.35)
    assert M.shape == (2, 3)

def test_apply_warp():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    out1 = apply_warp(img, M, 50, 50, False)
    assert out1.shape == (50, 50, 3)

    out2 = apply_warp(img, M, 50, 50, True)
    assert out2.shape == (50, 50, 3)

def test_natural_sort_key():
    assert _natural_sort_key("img10.jpg") > _natural_sort_key("img2.jpg")

def test_parse_resolution():
    assert parse_resolution("1920x1080") == (1920, 1080)
    assert parse_resolution(" 800X600 ") == (800, 600)

    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_resolution("bad")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_resolution("-10x10")

def test_build_parser():
    parser = build_parser()
    args = parser.parse_args(["-i", "foo"])
    assert args.input_dir == "foo"

@patch('face_timelapse.write_video')
@patch('face_timelapse.detect_all_faces')
@patch('face_timelapse.discover_images')
@patch('sys.argv', ['face_timelapse.py', '-i', 'foo'])
def test_main(mock_discover, mock_detect, mock_write):
    mock_discover.return_value = ["img1.jpg"]
    mock_detect.return_value = [DetectionResult("img1.jpg", 0, EyeCoords(0.0, 0.0, 1.0, 1.0))]

    main()
    mock_discover.assert_called_once_with("foo")
    mock_detect.assert_called_once()
    mock_write.assert_called_once()

@patch('face_timelapse.write_video')
@patch('face_timelapse.detect_all_faces')
@patch('face_timelapse.discover_images')
@patch('sys.argv', ['face_timelapse.py', '-i', 'foo', '-v'])
def test_main_verbose(mock_discover, mock_detect, mock_write):
    mock_discover.return_value = ["img1.jpg"]
    mock_detect.return_value = [DetectionResult("img1.jpg", 0, EyeCoords(0.0, 0.0, 1.0, 1.0))]
    main()

@patch('face_timelapse.discover_images')
@patch('sys.argv', ['face_timelapse.py', '-i', 'foo'])
def test_main_discover_fails(mock_discover):
    mock_discover.side_effect = ValueError("Error")
    with pytest.raises(SystemExit):
        main()

@patch('face_timelapse.detect_all_faces')
@patch('face_timelapse.discover_images')
@patch('sys.argv', ['face_timelapse.py', '-i', 'foo'])
def test_main_no_faces(mock_discover, mock_detect):
    mock_discover.return_value = ["img1.jpg"]
    mock_detect.return_value = [DetectionResult("img1.jpg", 0, None)]
    with pytest.raises(SystemExit):
        main()

@patch('face_timelapse.write_video')
@patch('face_timelapse.detect_all_faces')
@patch('face_timelapse.discover_images')
@patch('sys.argv', ['face_timelapse.py', '-i', 'foo'])
def test_main_write_fails(mock_discover, mock_detect, mock_write):
    mock_discover.return_value = ["img1.jpg"]
    mock_detect.return_value = [DetectionResult("img1.jpg", 0, EyeCoords(0.0, 0.0, 1.0, 1.0))]
    mock_write.side_effect = RuntimeError("Failed")
    with pytest.raises(SystemExit):
        main()

def test_main_args_validation():
    with patch('sys.argv', ['face_timelapse.py', '-i', 'foo', '-r', 'bad']):
        with pytest.raises(SystemExit):
            main()
    with patch('sys.argv', ['face_timelapse.py', '-i', 'foo', '-a', '1.5']):
        with pytest.raises(SystemExit):
            main()
    with patch('sys.argv', ['face_timelapse.py', '-i', 'foo', '-z', '-1']):
        with pytest.raises(SystemExit):
            main()
    with patch('sys.argv', ['face_timelapse.py', '-i', 'foo', '--eye_span', '2']):
        with pytest.raises(SystemExit):
            main()

@patch('face_timelapse.cv2.VideoWriter')
def test_write_video_writer_fails(mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = False
    mock_vw.return_value = mock_writer

    with pytest.raises(RuntimeError):
        write_video(
            image_paths=["test.jpg"],
            smoothed=[EyeCoords(0,0,1,1)],
            output_file=str(tmp_path / "out.mp4"),
            fps=24,
            dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False
        )

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.build_warp_matrix')
@patch('face_timelapse.apply_warp')
def test_write_video_with_tqdm(mock_apply, mock_build, mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer

    mock_imread.return_value = np.zeros((10,10,3), dtype=np.uint8)

    # eyes is None -> skipped
    write_video(
        image_paths=["test1.jpg", "test2.jpg"],
        smoothed=[None, EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24,
        dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=None
    )
    assert mock_writer.write.call_count == 1

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
def test_write_video_imread_fails(mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer

    mock_imread.return_value = None

    write_video(
        image_paths=["test1.jpg"],
        smoothed=[EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24,
        dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=None
    )
    assert mock_writer.write.call_count == 0

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.build_warp_matrix')
def test_write_video_warp_exception(mock_build, mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer

    mock_imread.return_value = np.zeros((10,10,3), dtype=np.uint8)
    mock_build.side_effect = Exception("Test Exception")

    write_video(
        image_paths=["test1.jpg"],
        smoothed=[EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24,
        dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=None
    )
    assert mock_writer.write.call_count == 0

def test_worker_init():
    import face_timelapse
    with patch('face_timelapse.mp') as mock_mp:
        face_timelapse._worker_init()
        mock_mp.solutions.face_mesh.FaceMesh.assert_called_once()

def test_detect_all_faces_cancel(tmp_path):
    import threading
    p = tmp_path / 'img.jpg'
    import cv2, numpy as np
    cv2.imwrite(str(p), np.zeros((10,10,3), dtype=np.uint8))

    cancel_event = threading.Event()
    cancel_event.set()

    with patch('face_timelapse.mp') as mock_mp:
        res = detect_all_faces([str(p)], 1, cancel_event=cancel_event)

    assert res[0].error == "Cancelled"

@patch('face_timelapse.cv2.VideoWriter')
def test_write_video_none_eyes_with_callback(mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer

    cb = MagicMock()
    write_video(
        image_paths=["test1.jpg"],
        smoothed=[None],
        output_file=str(tmp_path / "out.mp4"),
        fps=24, dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=cb
    )
    cb.assert_called_with("writing", 1, 1)

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
def test_write_video_unreadable_image_with_callback(mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer
    mock_imread.return_value = None

    cb = MagicMock()
    write_video(
        image_paths=["test1.jpg"],
        smoothed=[EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24, dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=cb
    )
    cb.assert_called_with("writing", 1, 1)

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.build_warp_matrix')
def test_write_video_warp_exception_with_callback(mock_build, mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer
    mock_imread.return_value = np.zeros((10,10,3), dtype=np.uint8)
    mock_build.side_effect = Exception("Test Error")

    cb = MagicMock()
    write_video(
        image_paths=["test1.jpg"],
        smoothed=[EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24, dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=cb
    )
    cb.assert_called_with("writing", 1, 1)

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.build_warp_matrix')
def test_write_video_exception_no_tqdm_no_callback(mock_build, mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer
    mock_imread.return_value = np.zeros((10,10,3), dtype=np.uint8)
    mock_build.side_effect = Exception("Test Error")

    write_video(
        image_paths=["test1.jpg"],
        smoothed=[EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24, dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=None
    )

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.build_warp_matrix')
def test_write_video_exception_with_tqdm_no_callback(mock_build, mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer
    mock_imread.return_value = np.zeros((10,10,3), dtype=np.uint8)
    mock_build.side_effect = Exception("Test Error")

    with patch('face_timelapse.tqdm') as mock_tqdm:
        write_video(
            image_paths=["test1.jpg"],
            smoothed=[EyeCoords(0,0,1,1)],
            output_file=str(tmp_path / "out.mp4"),
            fps=24, dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
            progress_callback=None
        )

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.build_warp_matrix')
def test_write_video_exception_with_callback_not_tqdm(mock_build, mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer
    mock_imread.return_value = np.zeros((10,10,3), dtype=np.uint8)
    mock_build.side_effect = Exception("Test Error")

    cb = MagicMock()
    write_video(
        image_paths=["test1.jpg"],
        smoothed=[EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24, dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=cb
    )
    cb.assert_called_with("writing", 1, 1)

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.build_warp_matrix')
def test_write_video_exception_with_tqdm_no_callback_hit_exception(mock_build, mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer
    mock_imread.return_value = np.zeros((10,10,3), dtype=np.uint8)
    mock_build.side_effect = Exception("Test Error")

    write_video(
        image_paths=["test1.jpg"],
        smoothed=[EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24, dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False
    )

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.build_warp_matrix')
def test_write_video_exception_with_tqdm_no_callback_hit_exception_with_cb(mock_build, mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer
    mock_imread.return_value = np.zeros((10,10,3), dtype=np.uint8)
    mock_build.side_effect = Exception("Test Error")

    cb = MagicMock()
    write_video(
        image_paths=["test1.jpg"],
        smoothed=[EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24, dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=cb
    )
    cb.assert_called_with("writing", 1, 1)

@patch('face_timelapse.cv2.VideoWriter')
@patch('face_timelapse.cv2.imread')
@patch('face_timelapse.build_warp_matrix')
def test_write_video_exception_with_tqdm_no_callback_hit_exception_with_cb2(mock_build, mock_imread, mock_vw, tmp_path):
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_vw.return_value = mock_writer
    mock_imread.return_value = np.zeros((10,10,3), dtype=np.uint8)
    mock_build.side_effect = Exception("Test Error")

    cb = MagicMock()
    write_video(
        image_paths=["test1.jpg"],
        smoothed=[EyeCoords(0,0,1,1)],
        output_file=str(tmp_path / "out.mp4"),
        fps=24, dst_w=100, dst_h=100, zoom=1, eye_span_fraction=0.35, blur_bg=False,
        progress_callback=cb
    )
    cb.assert_called_with("writing", 1, 1)
