"""2.5-D simulation world: cylinders (trees, poles) and boxes (walls, buildings).

Coordinates are north/east metres relative to home. An obstacle only blocks
a ray if the vehicle is below its height, so climbing over short obstacles
works as in reality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Cylinder:
    n: float
    e: float
    r: float
    h: float

    def hit(self, on, oe, dn, de) -> float:
        fn, fe = on - self.n, oe - self.e
        b = fn * dn + fe * de
        c = fn * fn + fe * fe - self.r * self.r
        disc = b * b - c
        if disc < 0:
            return math.inf
        s = math.sqrt(disc)
        for t in (-b - s, -b + s):
            if t >= 0:
                return t
        return math.inf


@dataclass
class Box:
    n0: float
    e0: float
    n1: float
    e1: float
    h: float

    def hit(self, on, oe, dn, de) -> float:
        tmin, tmax = -math.inf, math.inf
        for o, d, lo, hi in ((on, dn, min(self.n0, self.n1), max(self.n0, self.n1)),
                             (oe, de, min(self.e0, self.e1), max(self.e0, self.e1))):
            if abs(d) < 1e-12:
                if o < lo or o > hi:
                    return math.inf
                continue
            t1, t2 = (lo - o) / d, (hi - o) / d
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
        if tmax < max(tmin, 0.0):
            return math.inf
        return max(tmin, 0.0)


@dataclass
class World:
    obstacles: list = field(default_factory=list)
    dropzone: Optional[tuple[float, float]] = None   # true marker position (n, e)

    def raycast(self, n, e, alt, bearing_deg, max_range) -> float:
        b = math.radians(bearing_deg)
        dn, de = math.cos(b), math.sin(b)
        best = math.inf
        for ob in self.obstacles:
            if alt >= ob.h:
                continue
            best = min(best, ob.hit(n, e, dn, de))
        return best if best <= max_range else math.inf

    def collides(self, n, e, alt, radius=0.35) -> bool:
        for ob in self.obstacles:
            if alt >= ob.h:
                continue
            if isinstance(ob, Cylinder) and math.hypot(n - ob.n, e - ob.e) < ob.r + radius:
                return True
            if isinstance(ob, Box):
                if (min(ob.n0, ob.n1) - radius <= n <= max(ob.n0, ob.n1) + radius and
                        min(ob.e0, ob.e1) - radius <= e <= max(ob.e0, ob.e1) + radius):
                    return True
        return False

    @classmethod
    def load(cls, path: str | Path) -> "World":
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        obs = []
        for o in d.get("obstacles", []):
            if o["type"] == "cylinder":
                obs.append(Cylinder(o["n"], o["e"], o["r"], o["h"]))
            elif o["type"] == "box":
                obs.append(Box(o["n0"], o["e0"], o["n1"], o["e1"], o["h"]))
        dz = d.get("dropzone")
        return cls(obs, (dz["n"], dz["e"]) if dz else None)
