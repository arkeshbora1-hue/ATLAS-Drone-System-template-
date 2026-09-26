"""Shared data types passed between subsystems.

Each producer thread publishes an immutable snapshot into a ``Latest`` slot;
the executive reads all slots once per tick. There are no queues between
fast producers and the 10 Hz consumer, so a slow consumer never sees stale
backlog, and staleness is always explicit via timestamps.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class Latest(Generic[T]):
    """Thread-safe single-value mailbox."""

    def __init__(self, value: Optional[T] = None):
        self._value = value
        self._lock = threading.Lock()

    def put(self, value: T) -> None:
        with self._lock:
            self._value = value

    def get(self) -> Optional[T]:
        with self._lock:
            return self._value


class Sector(str, Enum):
    FRONT = "FRONT"
    FRONT_LEFT = "FRONT_LEFT"
    FRONT_RIGHT = "FRONT_RIGHT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DOWN = "DOWN"


# Bearing of each horizontal sector relative to the nose (+ clockwise).
SECTOR_BEARING = {
    Sector.FRONT: 0.0,
    Sector.FRONT_LEFT: -35.0,
    Sector.FRONT_RIGHT: 35.0,
    Sector.LEFT: -90.0,
    Sector.RIGHT: 90.0,
}

FORWARD_SECTORS = (Sector.FRONT, Sector.FRONT_LEFT, Sector.FRONT_RIGHT)


@dataclass(frozen=True)
class RangeReading:
    sensor: str
    sector: Sector
    distance: float          # metres; +inf when no return within range
    valid: bool
    stamp: float


@dataclass(frozen=True)
class RangeSnapshot:
    """Fused per-sector distances produced at sensor_hz."""

    stamp: float
    sectors: dict            # Sector -> float (m), inf when clear
    raw: tuple = ()          # RangeReading tuple for logging
    unhealthy: frozenset = frozenset()   # sectors with no valid sensor this cycle

    def distance(self, sector: Sector) -> float:
        return self.sectors.get(sector, float("inf"))

    def min_forward(self) -> float:
        return min(self.distance(s) for s in FORWARD_SECTORS)


@dataclass(frozen=True)
class VehicleState:
    stamp: float
    lat: float
    lon: float
    alt_rel: float           # m above home
    vn: float                # m/s north
    ve: float
    vd: float
    roll: float              # deg
    pitch: float
    yaw: float               # deg, [0,360)
    armed: bool
    mode: str
    battery_v: float
    battery_pct: float
    gps_fix: int             # 0-6 (3 = 3D fix)
    satellites: int
    hdop: float
    fc_heartbeat: float      # stamp of last FC heartbeat


@dataclass(frozen=True)
class Detection:
    """Drop-zone detection in normalised image coordinates."""

    stamp: float
    present: bool
    confidence: float
    cx: float = 0.5          # 0..1 across image width
    cy: float = 0.5          # 0..1 down image height
    w: float = 0.0
    h: float = 0.0
    latency: float = 0.0     # inference time (s)


@dataclass(frozen=True)
class TargetEstimate:
    """Drop zone projected to the ground, in the home tangent plane."""

    stamp: float             # time of the last positive detection
    north: float             # m north of home
    east: float              # m east of home
    confirmed: bool
    confidence: float
    hits: int = 0
    rel_n: float = 0.0       # latest smoothed camera-measured offset vehicle->marker (m)
    rel_e: float = 0.0       # (immune to GPS noise; used for the final visual servo)


@dataclass
class Command:
    """Output of one executive tick, consumed by the flight-controller link."""

    kind: str                          # "velocity" | "hold" | "land" | "rtl" | "none"
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0
    yaw: Optional[float] = None        # absolute heading (deg); None = keep
    note: str = ""
    extras: dict = field(default_factory=dict)
