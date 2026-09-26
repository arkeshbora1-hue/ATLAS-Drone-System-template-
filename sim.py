"""Simulated range sensors that ray-cast into a ``sim.world.World``."""
from __future__ import annotations

import random

from atlas.core.state import SECTOR_BEARING, RangeReading, Sector
from atlas.sensors.base import RangeSensor


class SimRangeSensor(RangeSensor):
    def __init__(self, name, sector, kind, world, vehicle, max_range=4.0, min_range=0.03,
                 noise=0.02, dropout=0.0, spike=0.0, beam_deg=15.0, seed=0):
        super().__init__(name, sector, max_range, min_range)
        self.kind = kind
        self.world, self.vehicle = world, vehicle
        self.noise, self.dropout, self.spike = noise, dropout, spike
        self.beam = beam_deg
        self.failed = False
        self._rng = random.Random(seed)

    def read(self, now: float) -> RangeReading:
        if self.failed or self._rng.random() < self.dropout:
            return self._fail(now)
        v = self.vehicle
        if self.sector == Sector.DOWN:
            h = v.alt + 0.08          # sensor sits 8 cm above the skids
            d = h if h <= self.max_range else float("inf")
        else:
            bearing = (v.yaw + SECTOR_BEARING[self.sector]) % 360
            # sample the beam cone with 3 rays; the closest return wins, like a real sensor
            half = self.beam / 2
            d = min(self.world.raycast(v.n, v.e, v.alt, (bearing + off) % 360, self.max_range)
                    for off in (-half, 0.0, half))
        if d != float("inf"):
            if self._rng.random() < self.spike:
                d = self._rng.uniform(0.1, self.max_range)       # spurious echo
            d = max(0.0, d + self._rng.gauss(0, self.noise))
        return self._ok(now, d)


def build_sim_array(cfg, world, vehicle, seed=0, **kw):
    from atlas.sensors.array import SensorArray

    sc = cfg.sensors
    sensors = []
    for i, u in enumerate(sc.ultrasonic):
        sensors.append(SimRangeSensor(u.name, u.sector, "ultrasonic", world, vehicle,
                                      sc.ultrasonic_max_range, sc.min_range, noise=0.03,
                                      beam_deg=30.0, seed=seed + i, **kw))
    for i, t in enumerate(sc.tof):
        sensors.append(SimRangeSensor(t.name, t.sector, "tof", world, vehicle,
                                      sc.tof_max_range, sc.min_range, noise=0.01,
                                      beam_deg=27.0, seed=seed + 100 + i, **kw))
    return SensorArray(sensors, sc.median_window, sc.stale_after)
