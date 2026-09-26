"""Companion-side geofence (the FC runs its own fence as a second layer)."""
from __future__ import annotations

import math

from atlas.core.geo import LatLon, ll_to_ne


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


class Geofence:
    def __init__(self, home: LatLon, max_alt: float, max_radius: float, polygon=None):
        self.max_alt = max_alt
        self.max_radius = max_radius
        self.poly = [ll_to_ne(home, LatLon(p[0], p[1])) for p in (polygon or [])]

    def check(self, n: float, e: float, alt: float) -> tuple[bool, str]:
        if alt > self.max_alt:
            return False, f"altitude {alt:.1f} > {self.max_alt}"
        r = math.hypot(n, e)
        if r > self.max_radius:
            return False, f"radius {r:.0f} > {self.max_radius:.0f}"
        if len(self.poly) >= 3 and not point_in_polygon(n, e, self.poly):
            return False, "outside polygon"
        return True, ""

    def contains_route(self, pts: list[tuple[float, float, float]]) -> tuple[bool, str]:
        for n, e, a in pts:
            ok, why = self.check(n, e, a)
            if not ok:
                return False, why
        return True, ""
