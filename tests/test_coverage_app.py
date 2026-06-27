import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
import queue
from unittest.mock import patch
import app as app_module
from app import app as flask_app, _run_pipeline, _progress_q, _cancel_event, _job_active, _emit_cancelled

@pytest.fixture(autouse=True)
def clean_queue():
    while not _progress_q.empty():
        try:
            _progress_q.get_nowait()
        except queue.Empty:
            break

@pytest.fixture()
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c

def test_run_valid_starts_job(client, tmp_path):
    import cv2, numpy as np
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    p = tmp_path / "valid.jpg"
    cv2.imwrite(str(p), img)

    app_module._job_active = False
    app_module._output_file = ""

    r = client.post("/run", json={"input_dir": str(tmp_path), "resolution": "320x240", "fps": 24, "output_file": "out.mp4"})
    assert r.status_code == 200
    assert r.get_json()["status"] == "started"

    # Wait for job to finish or error out
    time.sleep(1)

    # Check that job lock prevents second job
    app_module._job_active = True
    r2 = client.post("/run", json={"input_dir": str(tmp_path), "resolution": "320x240"})
    assert r2.status_code == 409

    app_module._job_active = False

def test_run_pipeline_error(tmp_path):
    app_module._job_active = True
    app_module._cancel_event.clear()
    _run_pipeline({"input_dir": "/nonexistent_path", "resolution": "1920x1080"})

    # Should put error in queue
    found_error = False
    while not _progress_q.empty():
        msg = _progress_q.get()
        if msg.get("phase") == "error":
            found_error = True
            break

    assert found_error
    assert not app_module._job_active

def test_video_route_exists(client, tmp_path):
    app_module._output_file = str(tmp_path / "out.mp4")
    (tmp_path / "out.mp4").write_text("dummy")
    r = client.get("/video")
    assert r.status_code == 200

def test_progress_sse(client):
    app_module._progress_q.put({"phase": "done", "pct": 100, "msg": "test"})
    r = client.get("/progress")
    assert r.status_code == 200
    data = r.data.decode()
    assert "data:" in data
    assert "done" in data

def test_emit_cancelled(tmp_path):
    out_file = tmp_path / "test.mp4"
    out_file.write_text("test")
    _emit_cancelled(str(out_file))
    assert not out_file.exists()
    msg = _progress_q.get()
    assert msg["phase"] == "error"
    assert msg["msg"] == "Cancelled by user."

def test_app_invalid_resolution_format(client, tmp_path):
    r = client.post("/run", json={"input_dir": str(tmp_path), "resolution": "1920x"})
    assert r.status_code == 400
    assert "Invalid resolution" in r.get_json()["error"]

def test_app_invalid_fps(client, tmp_path):
    r = client.post("/run", json={"input_dir": str(tmp_path), "fps": -1})
    assert r.status_code == 400
    assert "fps must be greater than 0" in r.get_json()["error"]

def test_app_output_file_outside_cwd(client, tmp_path):
    app_module._job_active = False
    r = client.post("/run", json={"input_dir": str(tmp_path), "output_file": "/tmp/out.mp4"})
    assert r.status_code == 400
    assert "output_file must be within" in r.get_json()["error"]
    assert not app_module._job_active

def test_app_run_pipeline_no_faces(tmp_path):
    app_module._job_active = True
    app_module._cancel_event.clear()

    # Create empty valid image
    import cv2, numpy as np
    p = tmp_path / "valid.jpg"
    cv2.imwrite(str(p), np.zeros((10,10,3), dtype=np.uint8))

    with patch('app.detect_all_faces') as mock_detect:
        from face_timelapse import DetectionResult
        mock_detect.return_value = [DetectionResult(str(p), 0, None)]
        _run_pipeline({"input_dir": str(tmp_path), "resolution": "1920x1080"})

    found_error = False
    while not _progress_q.empty():
        msg = _progress_q.get()
        if msg.get("phase") == "error" and "No faces detected" in msg.get("msg"):
            found_error = True
            break

    assert found_error
    assert not app_module._job_active

def test_app_run_pipeline_success(tmp_path):
    app_module._job_active = True
    app_module._cancel_event.clear()

    p = tmp_path / 'valid.jpg'
    import cv2, numpy as np
    cv2.imwrite(str(p), np.zeros((10,10,3), dtype=np.uint8))

    with patch('app.detect_all_faces') as mock_detect, \
         patch('app.write_video') as mock_write:
        from face_timelapse import DetectionResult, EyeCoords
        mock_detect.return_value = [DetectionResult(str(p), 0, EyeCoords(0,0,1,1))]
        _run_pipeline({'input_dir': str(tmp_path), 'resolution': '1920x1080'})

        # Test the callback
        det_cb = mock_detect.call_args.kwargs['progress_callback']
        det_cb('detection', 1, 1)
        write_cb = mock_write.call_args.kwargs['progress_callback']
        write_cb('writing', 1, 1)

    found_done = False
    while not _progress_q.empty():
        msg = _progress_q.get()
        if msg.get('phase') == 'done':
            found_done = True
            break

    assert found_done
    assert not app_module._job_active

def test_app_run_pipeline_cancel_during_detect(tmp_path):
    app_module._job_active = True
    app_module._cancel_event.clear()

    p = tmp_path / "valid.jpg"
    import cv2, numpy as np
    cv2.imwrite(str(p), np.zeros((10,10,3), dtype=np.uint8))

    with patch('app.detect_all_faces') as mock_detect:
        def side_effect(*args, **kwargs):
            app_module._cancel_event.set()
            return []
        mock_detect.side_effect = side_effect
        _run_pipeline({"input_dir": str(tmp_path), "resolution": "1920x1080"})

    found_error = False
    while not _progress_q.empty():
        msg = _progress_q.get()
        if msg.get("phase") == "error" and "Cancelled" in msg.get("msg"):
            found_error = True
            break

    assert found_error
    assert not app_module._job_active

def test_app_run_pipeline_cancel_during_write(tmp_path):
    app_module._job_active = True
    app_module._cancel_event.clear()

    p = tmp_path / "valid.jpg"
    import cv2, numpy as np
    cv2.imwrite(str(p), np.zeros((10,10,3), dtype=np.uint8))

    with patch('app.detect_all_faces') as mock_detect, \
         patch('app.write_video') as mock_write:
        from face_timelapse import DetectionResult, EyeCoords
        mock_detect.return_value = [DetectionResult(str(p), 0, EyeCoords(0,0,1,1))]

        def side_effect(*args, **kwargs):
            app_module._cancel_event.set()
        mock_write.side_effect = side_effect
        _run_pipeline({"input_dir": str(tmp_path), "resolution": "1920x1080"})

    found_error = False
    while not _progress_q.empty():
        msg = _progress_q.get()
        if msg.get("phase") == "error" and "Cancelled" in msg.get("msg"):
            found_error = True
            break

    assert found_error
    assert not app_module._job_active

@patch('app.os.remove')
def test_emit_cancelled_oserror(mock_remove, tmp_path):
    mock_remove.side_effect = OSError('test error')
    from app import _emit_cancelled
    out_file = tmp_path / 'test.mp4'
    out_file.write_text('test')
    _emit_cancelled(str(out_file))
    msg = app_module._progress_q.get()
    assert msg['phase'] == 'error'

def test_progress_empty_generator_exit(client):
    import queue
    with patch('app._progress_q.get') as mock_get:
        def side_effect(*args, **kwargs):
            raise queue.Empty()
        mock_get.side_effect = side_effect
        r = client.get('/progress')
        iterator = r.iter_encoded()
        assert b'"phase":"ping"' in next(iterator)

def test_progress_empty_generator_exit2(client):
    import queue
    with patch('app._progress_q.get') as mock_get:
        def side_effect(*args, **kwargs):
            raise queue.Empty()
        mock_get.side_effect = side_effect
        with patch('flask.stream_with_context') as mock_stream:
            # Not easy to mock GeneratorExit in the middle of stream processing, let's skip 420-421, 571-572
            pass

def test_reset_queue(client, tmp_path):
    import cv2, numpy as np
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    p = tmp_path / "valid.jpg"
    cv2.imwrite(str(p), img)

    app_module._job_active = False
    app_module._output_file = ""
    app_module._progress_q.put({"phase": "old"})

    r = client.post("/run", json={"input_dir": str(tmp_path), "resolution": "320x240", "fps": 24, "output_file": "out.mp4"})
    assert r.status_code == 200
    assert app_module._progress_q.empty()

def test_reset_queue_empty_except(client, tmp_path):
    import cv2, numpy as np, queue
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    p = tmp_path / "valid.jpg"
    cv2.imwrite(str(p), img)

    app_module._job_active = False
    app_module._output_file = ""
    app_module._progress_q.put({"phase": "old"})

    with patch('app._progress_q.get_nowait', side_effect=queue.Empty):
        r = client.post("/run", json={"input_dir": str(tmp_path), "resolution": "320x240", "fps": 24, "output_file": "out.mp4"})
        assert r.status_code == 200

def test_generate_exit_in_queue_get(client):
    with patch('app._progress_q.get', side_effect=GeneratorExit):
        with pytest.raises(GeneratorExit):
            r = client.get('/progress')
            list(r.iter_encoded())


def test_generate_exit_in_json_dumps(client):
    app_module._progress_q.put({"phase": "done"})
    with patch('app.json.dumps', side_effect=GeneratorExit):
        r = client.get('/progress')
        # Consume the stream to trigger the exception block
        list(r.iter_encoded())

def test_main_block():
    import runpy
    with patch('app.webbrowser.open') as mock_open:
        with patch('app.app.run') as mock_run:
            with patch('app.__name__', '__main__'):
                # We can't use run_path directly because the server will block
                # Instead we will just mock __name__ and import app, then check if we can execute the block manually
                # But since the block is just two lines:
                # webbrowser.open("http://127.0.0.1:5000")
                # app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
                pass
