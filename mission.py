"""Mission definition, route and waypoint bookkeeping.

A mission file (YAML) lists transit waypoints and the approximate drop-zone
location. Everything is converted once into the local north/east frame at
home, and the route is a mutable list so the avoidance layer can splice
detour waypoints in front of the active one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from atlas.core.geo import LatLon, ll_to_ne


@dataclass
class Waypoint:
    n: float
    e: float
    alt: float
    tag: str = "transit"          # transit | detour | climb | dropzone | home

    def dist_to(self, n: float, e: float) -> float:
        return math.hypot(self.n - n, self.e - e)


@dataclass
class MissionSpec:
    name: str
    waypoints: list[LatLon]
    dropzone: LatLon
    search_radius: float = 20.0
    alt: Optional[float] = None
    home: Optional[LatLon] = None

    @classmethod
    def load(cls, path: str | Path) -> "MissionSpec":
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        wps = [LatLon(w["lat"], w["lon"]) for w in d.get("waypoints", [])]
        dz = d["dropzone"]
        home = LatLon(d["home"]["lat"], d["home"]["lon"]) if d.get("home") else None
        return cls(d.get("name", Path(path).stem), wps, LatLon(dz["lat"], dz["lon"]),
                   dz.get("search_radius", 20.0), d.get("cruise_alt"), home)


@dataclass
class Route:
    wps: list[Waypoint] = field(default_factory=list)
    index: int = 0

    @property
    def active(self) -> Optional[Waypoint]:
        return self.wps[self.index] if self.index < len(self.wps) else None

    @property
    def finished(self) -> bool:
        return self.index >= len(self.wps)

    def advance(self) -> None:
        self.index += 1

    def insert_before_active(self, new: list[Waypoint]) -> None:
        self.wps[self.index:self.index] = new

    def drop_detours(self) -> None:
        """Discard pending detour waypoints (e.g. after a climb resolved the block)."""
        keep = self.wps[:self.index] + [w for w in self.wps[self.index:] if w.tag not in ("detour",)]
        self.wps = keep

    def remaining(self) -> list[Waypoint]:
        return self.wps[self.index:]


def build_outbound(spec: MissionSpec, home: LatLon, cruise_alt: float) -> Route:
    wps = []
    for ll in spec.waypoints:
        n, e = ll_to_ne(home, ll)
        wps.append(Waypoint(n, e, cruise_alt, "transit"))
    n, e = ll_to_ne(home, spec.dropzone)
    wps.append(Waypoint(n, e, cruise_alt, "dropzone"))
    return Route(wps)


def build_return(spec: MissionSpec, home: LatLon, cruise_alt: float) -> Route:
    wps = []
    for ll in reversed(spec.waypoints):
        n, e = ll_to_ne(home, ll)
        wps.append(Waypoint(n, e, cruise_alt, "transit"))
    wps.append(Waypoint(0.0, 0.0, cruise_alt, "home"))
    return Route(wps)


def expanding_square(cn: float, ce: float, alt: float, leg: float, max_radius: float) -> list[Waypoint]:
    """Classic SAR expanding-square pattern centred on (cn, ce).

    Legs grow by ``leg`` every two turns: N, E, S, S, W, W, N, N, N, ...
    """
    pts = [Waypoint(cn, ce, alt, "search")]
    n, e = cn, ce
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    length, k = leg, 0
    while True:
        for _ in range(2):
            dn, de = dirs[k % 4]
            n, e = n + dn * length, e + de * length
            if math.hypot(n - cn, e - ce) > max_radius * math.sqrt(2):
                return pts
            pts.append(Waypoint(n, e, alt, "search"))
            k += 1
        length += leg
