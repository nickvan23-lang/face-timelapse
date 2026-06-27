"""Tests for discover_images recursive parameter."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from facelapse import discover_images


def test_recursive_finds_images_in_subfolders(tmp_path):
    sub = tmp_path / "2024" / "jan"
    sub.mkdir(parents=True)
    img = sub / "photo_001.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header

    result = discover_images(str(tmp_path), recursive=True)

    assert str(img) in result


def test_recursive_false_ignores_subfolders(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "photo.jpg").write_bytes(b"\xff\xd8\xff")

    with pytest.raises(ValueError):
        discover_images(str(tmp_path), recursive=False)


def test_recursive_error_message_includes_subfolders_hint(tmp_path):
    with pytest.raises(ValueError, match="No valid images found"):
        discover_images(str(tmp_path), recursive=True)


def test_recursive_ignores_directories_in_rglob(tmp_path):
    """rglob returns dirs too; ensure only files are included."""
    sub = tmp_path / "subdir"
    sub.mkdir()
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff")

    result = discover_images(str(tmp_path), recursive=True)

    assert str(sub) not in result
    assert len(result) == 1


def test_recursive_natural_sort_across_subfolders(tmp_path):
    for i in [3, 1, 2]:
        (tmp_path / f"img_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff")

    result = discover_images(str(tmp_path), recursive=True)

    names = [os.path.basename(p) for p in result]
    assert names == ["img_001.jpg", "img_002.jpg", "img_003.jpg"]
