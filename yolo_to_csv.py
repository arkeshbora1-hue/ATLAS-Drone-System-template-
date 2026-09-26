"""Convert a YOLO-format export (Roboflow, CVAT, Label Studio) to labels.csv.

YOLO layout:  <root>/images/*.jpg  and  <root>/labels/*.txt
              each txt: "<class> <cx> <cy> <w> <h>" (normalised), empty = negative
Only the first box of class ``--cls`` is used (one marker per image).

    python training/yolo_to_csv.py --root data/field_captures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_labels  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cls", type=int, default=0)
    a = ap.parse_args(argv)
    root = Path(a.root)
    rows = []
    for img in sorted((root / "images").iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = root / "labels" / (img.stem + ".txt")
        box, present = [0.0] * 4, False
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) == a.cls:
                    box, present = [float(v) for v in parts[1:]], True
                    break
        rows.append({"filename": img.name, "present": present, "box": box})
    write_labels(root, rows)
    print(f"{len(rows)} images, {sum(r['present'] for r in rows)} positive")


if __name__ == "__main__":
    main()
