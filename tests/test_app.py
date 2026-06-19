"""Smoke tests for Flask routes using the test client."""
import json
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


def test_index_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Face Timelapse Generator" in r.data
    assert b"text/html" in r.content_type.encode()


def test_run_missing_input_dir(client):
    r = client.post("/run", json={})
    assert r.status_code == 400
    assert "input_dir is required" in r.get_json()["error"]


def test_run_invalid_resolution(client):
    r = client.post("/run", json={"input_dir": "/tmp", "resolution": "bad"})
    assert r.status_code == 400
    assert "resolution" in r.get_json()["error"].lower()


def test_video_404_when_no_output(client):
    app_module._output_file = ""
    r = client.get("/video")
    assert r.status_code == 404


def test_cancel_returns_ok(client):
    r = client.post("/cancel")
    assert r.status_code == 200
    assert r.get_json()["status"] == "cancel signal sent"
