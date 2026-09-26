"""Temporal confirmation and ground localisation of the drop zone.

A single-frame detection is never acted on. A target is *confirmed* once at
least N of the last M frames were positive (5 of 8 by default, ~0.5 s at
15 fps), which suppresses one-off false positives from glare or clutter.
Positive frames are projected to the ground using the vehicle attitude and
height at that instant and smoothed with an exponential moving average in
the home-relative north/east frame, so the estimate stays put while the
vehicle moves.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from atlas.core.state import Detection, TargetEstimate, VehicleState
from atlas.perception.geometry import pixel_to_ground


class TargetTracker:
    def __init__(self, confirm_frames: int, confirm_window: int, hfov: float, vfov: float,
                 alpha: float = 0.3, reset_after: float = 3.0):
        self.n, self.m = confirm_frames, confirm_window
        self.hfov, self.vfov = hfov, vfov
        self.alpha = alpha
        self.reset_after = reset_after
        self.hist: deque[bool] = deque(maxlen=confirm_window)
        self.est_n: Optional[float] = None
        self.est_e: Optional[float] = None
        self.last_hit = -1e9
        self.conf = 0.0
        self.rel_n = self.rel_e = 0.0

    def update(self, det: Detection, vs: VehicleState, veh_n: float, veh_e: float,
               agl: float) -> Optional[TargetEstimate]:
        self.hist.append(det.present)
        if det.present and agl > 0.5:
            off = pixel_to_ground(det.cx, det.cy, agl, vs.roll, vs.pitch, vs.yaw, self.hfov, self.vfov)
            if off is not None:
                tn, te = veh_n + off[0], veh_e + off[1]
                if self.est_n is None or det.stamp - self.last_hit > self.reset_after:
                    self.est_n, self.est_e = tn, te
                    self.rel_n, self.rel_e = off
                else:
                    self.est_n += self.alpha * (tn - self.est_n)
                    self.est_e += self.alpha * (te - self.est_e)
                    self.rel_n += 0.5 * (off[0] - self.rel_n)
                    self.rel_e += 0.5 * (off[1] - self.rel_e)
                self.last_hit = det.stamp
                self.conf = det.confidence
        if self.est_n is None:
            return None
        hits = sum(self.hist)
        return TargetEstimate(stamp=self.last_hit, north=self.est_n, east=self.est_e,
                              confirmed=hits >= self.n, confidence=self.conf, hits=hits,
                              rel_n=self.rel_n, rel_e=self.rel_e)
