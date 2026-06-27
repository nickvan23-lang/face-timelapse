import os
import re
from pathlib import Path
from typing import List

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def _natural_sort_key(s: str) -> List:
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def discover_images(input_dir: str, recursive: bool = False) -> List[str]:
    p = Path(input_dir)
    if not p.is_dir():
        raise ValueError(f"Input path '{input_dir}' is not a directory.")

    matches = []
    if recursive:
        for root, _, files in os.walk(p):
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in _IMG_EXTENSIONS:
                    matches.append(os.path.join(root, f))
    else:
        for item in p.iterdir():
            if item.is_file() and item.suffix.lower() in _IMG_EXTENSIONS:
                matches.append(str(item))

    if not matches:
        raise ValueError(f"No valid images found in '{input_dir}'.")

    return sorted(matches, key=_natural_sort_key)
