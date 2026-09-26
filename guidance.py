"""Waypoint guidance: turns (pose, waypoint) into a velocity + yaw setpoint.

Yaw-first behaviour: the obstacle sensors are body-fixed and face forward,
so the vehicle turns toward the waypoint before accelerating. Forward speed
is scaled by cos(yaw error) and is zero when the error exceeds 45°; a large
turn is only started once the vehicle has braked below 0.8 m/s. Together
these keep the FRONT sectors looking along the direction of travel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from atlas.core.geo import clamp, heading_ne, wrap_180
from atlas.navigation.mission import Waypoint


@dataclass
class GuidanceOutput:
    vn: float
    ve: float
    vd: float
    yaw: float
    dist: float
    reached: bool


class WaypointGuidance:
    def __init__(self, cruise_speed: float, climb_speed: float, descent_speed: float,
                 accept_radius: float, slow_radius: float = 8.0, yaw_gate_deg: float = 45.0,
                 alt_tolerance: float = 1.0):
        self.cruise = cruise_speed
        self.climb = climb_speed
        self.descent = descent_speed
        self.accept = accept_radius
        self.slow_radius = slow_radius
        self.yaw_gate = yaw_gate_deg
        self.alt_tol = alt_tolerance

    def vertical(self, alt: float, target_alt: float) -> float:
        """NED down-velocity for an altitude P-controller."""
        err = target_alt - alt                     # + means climb
        v = clamp(0.8 * err, -self.descent, self.climb)
        return -v

    def update(self, n: float, e: float, alt: float, yaw: float, wp: Waypoint,
               speed_limit: float | None = None, ground_speed: float = 0.0) -> GuidanceOutput:
        dn, de = wp.n - n, wp.e - e
        dist = math.hypot(dn, de)
        vd = self.vertical(alt, wp.alt)
        reached = dist <= self.accept and abs(wp.alt - alt) <= self.alt_tol
        if dist < 0.3:
            return GuidanceOutput(0.0, 0.0, vd, yaw, dist, reached)
        want_yaw = heading_ne(dn, de) if dist > self.accept else yaw
        yaw_err = abs(wrap_180(want_yaw - yaw))
        vmax = min(self.cruise, speed_limit) if speed_limit else self.cruise
        # slow down approaching the waypoint (linear ramp) but keep >= 0.5 m/s
        speed = vmax * clamp(dist / self.slow_radius, 0.1, 1.0)
        speed = max(speed, min(0.5, dist))
        # climb before transit if far below the target altitude
        if wp.alt - alt > 3.0:
            speed *= 0.3
        if yaw_err > self.yaw_gate:
            speed = 0.0
            if ground_speed > 0.8:
                # brake BEFORE turning: while still moving, keep the nose (and
                # the sensors) on the velocity vector until the vehicle slows
                want_yaw = yaw
        else:
            speed *= math.cos(math.radians(yaw_err))
        vn, ve = dn / dist * speed, de / dist * speed
        return GuidanceOutput(vn, ve, vd, want_yaw, dist, reached)
