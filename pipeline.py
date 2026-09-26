"""Perception pipeline: camera -> detector -> tracker -> Latest[TargetEstimate].

Runs in its own thread capped at ``perception_fps`` (15). Only called from
the executive through the ``Latest`` slots, so a slow frame can never stall
the 10 Hz control loop. Records per-frame latency for the FPS report.
"""
from __future__ import annotations

import logging
from collections import deque

from atlas.core.geo import LatLon, ll_to_ne
from atlas.core.state import Detection, Latest, RangeSnapshot, Sector, TargetEstimate, VehicleState
from atlas.perception.detector import Detector
from atlas.perception.tracker import TargetTracker

log = logging.getLogger("atlas.perception")


class PerceptionPipeline:
    def __init__(self, cfg, detector: Detector, frame_source, home_fn, vehicle_slot: Latest,
                 range_slot: Latest, out_slot: Latest, clock):
        p = cfg.perception
        self.detector = detector
        self.frames = frame_source          # callable -> (frame, stamp)
        self.home_fn = home_fn
        self.vehicle_slot, self.range_slot, self.out = vehicle_slot, range_slot, out_slot
        self.clock = clock
        self.tracker = TargetTracker(p.confirm_frames, p.confirm_window,
                                     p.camera.hfov_deg, p.camera.vfov_deg)
        self.enabled = True
        self.latencies: deque[float] = deque(maxlen=150)
        self.frame_times: deque[float] = deque(maxlen=150)
        self.last_detection: Detection | None = None

    def reset(self) -> None:
        p = self.tracker
        self.tracker = TargetTracker(p.n, p.m, p.hfov, p.vfov, p.alpha, p.reset_after)
        self.out.put(None)

    @staticmethod
    def agl(vs: VehicleState, rs: RangeSnapshot | None) -> float:
        """Height above ground: down-ToF when in range (<4 m), else baro/GPS rel. alt."""
        if rs is not None and Sector.DOWN not in rs.unhealthy:
            d = rs.distance(Sector.DOWN)
            if d != float("inf"):
                return d
        return max(vs.alt_rel, 0.0)

    def step(self) -> None:
        if not self.enabled:
            return
        now = self.clock.now()
        vs: VehicleState | None = self.vehicle_slot.get()
        home: LatLon | None = self.home_fn()
        if vs is None or home is None:
            return
        frame, _ = self.frames()
        det = self.detector.detect(frame, now)
        self.last_detection = det
        self.latencies.append(det.latency)
        self.frame_times.append(now)
        n, e = ll_to_ne(home, LatLon(vs.lat, vs.lon))
        est: TargetEstimate | None = self.tracker.update(det, vs, n, e, self.agl(vs, self.range_slot.get()))
        self.out.put(est)

    def fps(self) -> float:
        t = self.frame_times
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    def mean_latency_ms(self) -> float:
        return 1000 * sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
