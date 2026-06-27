"""Tests for POST /scan route."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import app as app_module
from app import app as flask_app


@pytest.fixture()
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_scan_missing_path(client):
    r = client.post("/scan", json={})
    assert r.status_code == 400
    assert "path is required" in r.get_json()["error"]


def test_scan_nonexistent_path(client):
    r = client.post("/scan", json={"path": "/nonexistent_xyzzy_path_12345"})
    assert r.status_code == 400
    assert "is not a directory" in r.get_json()["error"].lower()


def test_scan_empty_path_string(client):
    r = client.post("/scan", json={"path": "   "})
    assert r.status_code == 400
    assert "path is required" in r.get_json()["error"]


def test_scan_finds_images_recursively(client, tmp_path):
    import cv2, numpy as np
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "top.jpg"), img)
    sub = tmp_path / "sub"
    sub.mkdir()
    cv2.imwrite(str(sub / "nested.jpg"), img)

    r = client.post("/scan", json={"path": str(tmp_path), "recursive": True})
    assert r.status_code == 200
    data = r.get_json()
    assert data["count"] == 2
    assert data["path"] == str(tmp_path)


def test_scan_non_recursive_ignores_subfolders(client, tmp_path):
    import cv2, numpy as np
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    sub = tmp_path / "sub"
    sub.mkdir()
    cv2.imwrite(str(sub / "nested.jpg"), img)

    # No images at top level → should fail
    r = client.post("/scan", json={"path": str(tmp_path), "recursive": False})
    assert r.status_code == 400
