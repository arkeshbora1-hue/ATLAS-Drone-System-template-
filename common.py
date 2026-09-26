"""Shared dataset utilities for the drop-zone model.

Label format (labels.csv, one row per image):
    filename,present,cx,cy,w,h
    img_0001.jpg,1,0.512,0.430,0.18,0.17
    img_0002.jpg,0,0,0,0,0
Box values are normalised to the image (0..1). Negative images (no marker)
are essential: they teach the presence head what "not a drop zone" looks
like (grass, roofs, shadows, the vehicle's own legs in frame).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

IMG_SIZE = 160


def read_labels(root: str | Path) -> list[dict]:
    root = Path(root)
    rows = []
    with open(root / "labels.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "path": str(root / "images" / r["filename"]),
                "present": float(r["present"]),
                "box": [float(r["cx"]), float(r["cy"]), float(r["w"]), float(r["h"])],
            })
    return rows


def write_labels(root: str | Path, rows: list[dict]) -> None:
    root = Path(root)
    with open(root / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "present", "cx", "cy", "w", "h"])
        for r in rows:
            w.writerow([r["filename"], int(r["present"]), *[f"{v:.5f}" for v in r["box"]]])


def load_image(path: str, size: int = IMG_SIZE) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
