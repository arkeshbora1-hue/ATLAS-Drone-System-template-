"""Geodesy helpers.

ATLAS flies short missions (< 1 km), so a local tangent plane (north/east
metres relative to a home origin) is accurate to centimetres and keeps the
avoidance and navigation maths simple. Great-circle helpers are provided for
logging and sanity checks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS = 6_371_000.0  # mean radius, m


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float


def haversine(a: LatLon, b: LatLon) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp = p2 - p1
    dl = math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS * math.asin(math.sqrt(h))


def bearing(a: LatLon, b: LatLon) -> float:
    """Initial bearing a->b in degrees [0, 360)."""
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dl = math.radians(b.lon - a.lon)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def ll_to_ne(origin: LatLon, point: LatLon) -> tuple[float, float]:
    """Lat/lon -> (north, east) metres in the tangent plane at origin."""
    dn = math.radians(point.lat - origin.lat) * EARTH_RADIUS
    de = math.radians(point.lon - origin.lon) * EARTH_RADIUS * math.cos(math.radians(origin.lat))
    return dn, de


def ne_to_ll(origin: LatLon, north: float, east: float) -> LatLon:
    lat = origin.lat + math.degrees(north / EARTH_RADIUS)
    lon = origin.lon + math.degrees(east / (EARTH_RADIUS * math.cos(math.radians(origin.lat))))
    return LatLon(lat, lon)


def wrap_180(deg: float) -> float:
    """Wrap an angle to (-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def wrap_360(deg: float) -> float:
    return deg % 360.0


def body_to_ne(forward: float, right: float, yaw_deg: float) -> tuple[float, float]:
    """Rotate a body-frame (forward, right) vector into (north, east)."""
    y = math.radians(yaw_deg)
    n = forward * math.cos(y) - right * math.sin(y)
    e = forward * math.sin(y) + right * math.cos(y)
    return n, e


def ne_to_body(north: float, east: float, yaw_deg: float) -> tuple[float, float]:
    """Rotate a (north, east) vector into body (forward, right)."""
    y = math.radians(yaw_deg)
    f = north * math.cos(y) + east * math.sin(y)
    r = -north * math.sin(y) + east * math.cos(y)
    return f, r


def heading_ne(north: float, east: float) -> float:
    """Heading in degrees [0,360) of a (north, east) vector."""
    return (math.degrees(math.atan2(east, north)) + 360.0) % 360.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def limit_norm(n: float, e: float, max_norm: float) -> tuple[float, float]:
    mag = math.hypot(n, e)
    if mag <= max_norm or mag == 0.0:
        return n, e
    s = max_norm / mag
    return n * s, e * s
