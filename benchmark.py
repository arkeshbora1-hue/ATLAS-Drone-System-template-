"""Latency / FPS benchmark for the TFLite model. Run it ON THE PI.

    python training/benchmark.py --model models/dropzone_mnv2_int8.tflite

Reports inference-only latency for 1-4 threads and an end-to-end figure
including resize from a 640x480 frame, which is what the 15 fps target is
measured against. Keep the Pi on a heatsink + fan: a throttling Pi 4 loses
~30% throughput after a few minutes (check `vcgencmd get_throttled`).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.perception.detector import TFLiteDropZoneDetector, _load_interpreter  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/dropzone_mnv2_int8.tflite")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args(argv)
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    for threads in (1, 2, 3, 4):
        det = TFLiteDropZoneDetector(a.model, a.size, interpreter=_load_interpreter(a.model, threads))
        for _ in range(10):
            det.detect(frame, 0.0)
        lat, t0 = [], time.perf_counter()
        for _ in range(a.iters):
            lat.append(det.detect(frame, 0.0).latency)
        wall = time.perf_counter() - t0
        p50 = statistics.median(lat) * 1000
        p95 = sorted(lat)[int(0.95 * len(lat))] * 1000
        print(f"threads={threads}: p50 {p50:5.1f} ms  p95 {p95:5.1f} ms  -> {a.iters / wall:5.1f} fps end-to-end")


if __name__ == "__main__":
    main()
