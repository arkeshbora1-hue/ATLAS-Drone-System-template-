"""Synthetic detector for the simulator.

Projects the true drop-zone position into the virtual nadir camera and
returns a detection with pixel noise, misses, false positives and a
realistic inference latency, so the tracker, visual servo and state
machine are exercised exactly as with the real model.
"""
from __future__ import annotations

import random

from atlas.core.state import Detection
from atlas.perception.detector import Detector
from atlas.perception.geometry import ground_to_pixel


class SimDropZoneDetector(Detector):
    def __init__(self, world, vehicle, hfov: float, vfov: float, miss_rate=0.12, fp_rate=0.01,
                 pixel_noise=0.01, max_detect_agl=25.0, latency=0.038, seed=3):
        self.world, self.vehicle = world, vehicle
        self.hfov, self.vfov = hfov, vfov
        self.miss, self.fp = miss_rate, fp_rate
        self.noise = pixel_noise
        self.max_agl = max_detect_agl
        self.latency = latency
        self.rng = random.Random(seed)

    def detect(self, frame_rgb, now: float) -> Detection:
        v, r = self.vehicle, self.rng
        dz = self.world.dropzone
        lat = self.latency * r.uniform(0.9, 1.15)
        if dz is not None and 0.5 < v.alt <= self.max_agl:
            px = ground_to_pixel(dz[0] - v.n, dz[1] - v.e, v.alt, 0.0, 0.0, v.yaw, self.hfov, self.vfov)
            if px and 0.02 < px[0] < 0.98 and 0.02 < px[1] < 0.98 and r.random() > self.miss:
                cx = px[0] + r.gauss(0, self.noise)
                cy = px[1] + r.gauss(0, self.noise)
                size = min(0.9, 1.5 / max(v.alt, 0.5))
                return Detection(now, True, r.uniform(0.72, 0.97), cx, cy, size, size, lat)
        if r.random() < self.fp:
            return Detection(now, True, r.uniform(0.6, 0.7), r.random(), r.random(), 0.1, 0.1, lat)
        return Detection(now, False, r.uniform(0.0, 0.3), 0.5, 0.5, 0.0, 0.0, lat)
