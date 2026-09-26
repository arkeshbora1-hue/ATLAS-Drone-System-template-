"""Sensor array: samples every range sensor at 10 Hz and fuses per sector.

Per cycle (100 ms budget):
  1. Ultrasonics fired sequentially (≤ ~25 ms each at 4 m -> ≤ 75 ms for 3).
  2. ToF results fetched (non-blocking; they range continuously).
  3. Each sensor's value goes through a short median filter (window 3),
     which rejects single-sample spikes from prop-wash noise or multipath
     while adding only one sample (~100 ms) of lag.
  4. Sector distance = MIN of the filtered, fresh, valid sensors covering it
     (conservative: when ultrasonic and ToF disagree, trust the closer one).
  5. Sectors with no valid fresh sensor are reported as ``unhealthy`` so the
     safety monitor can hold position instead of flying blind.
"""
from __future__ import annotations

import statistics
from collections import deque
from typing import Iterable

from atlas.core.state import RangeReading, RangeSnapshot, Sector
from atlas.sensors.base import RangeSensor


class _Filter:
    def __init__(self, window: int):
        self.buf: deque[float] = deque(maxlen=window)
        self.last_valid = -1e9

    def push(self, r: RangeReading) -> None:
        if r.valid:
            self.buf.append(r.distance)
            self.last_valid = r.stamp

    def value(self) -> float:
        return statistics.median(self.buf) if self.buf else float("inf")


class SensorArray:
    def __init__(self, sensors: Iterable[RangeSensor], median_window: int = 3, stale_after: float = 0.5):
        self.sensors = list(sensors)
        # ultrasonics first: they block, ToF reads are instant
        self.sensors.sort(key=lambda s: 0 if s.kind == "ultrasonic" else 1)
        self.stale_after = stale_after
        self._f = {s.name: _Filter(median_window) for s in self.sensors}
        self.cycles = 0

    def open(self) -> None:
        for s in self.sensors:
            s.open()

    def close(self) -> None:
        for s in self.sensors:
            s.close()

    def sample(self, now: float) -> RangeSnapshot:
        raw = []
        for s in self.sensors:
            r = s.read(now)
            raw.append(r)
            self._f[s.name].push(r)
        self.cycles += 1
        return self.fuse(now, tuple(raw))

    def fuse(self, now: float, raw: tuple = ()) -> RangeSnapshot:
        sectors: dict[Sector, float] = {}
        covered: set[Sector] = set()
        healthy: set[Sector] = set()
        for s in self.sensors:
            covered.add(s.sector)
            f = self._f[s.name]
            if now - f.last_valid > self.stale_after or not f.buf:
                continue
            healthy.add(s.sector)
            v = f.value()
            sectors[s.sector] = min(v, sectors.get(s.sector, float("inf")))
        for sec in covered - healthy:
            sectors.setdefault(sec, float("inf"))
        return RangeSnapshot(stamp=now, sectors=sectors, raw=raw,
                             unhealthy=frozenset(covered - healthy))


def build_hardware_array(cfg) -> SensorArray:
    """Instantiate the real sensors from config (vehicle only)."""
    from atlas.sensors.hcsr04 import HCSR04
    from atlas.sensors.vl53l1x import VL53L1XSensor, assign_addresses

    sc = cfg.sensors
    us = [HCSR04(u.name, u.sector, u.trig_gpio, u.echo_gpio, sc.ultrasonic_max_range,
                 sc.min_range, sc.temperature_c) for u in sc.ultrasonic]
    tof = [VL53L1XSensor(t.name, t.sector, t.xshut_gpio, t.address, sc.tof_max_range, sc.min_range)
           for t in sc.tof]
    assign_addresses(tof)
    return SensorArray(us + tof, sc.median_window, sc.stale_after)
