"""Pinhole projection between image coordinates and the ground plane.

Camera is mounted nadir (looking straight down) with the top of the image
toward the nose. A pixel therefore maps to a body-frame ray
    forward = -(cy - 0.5) * 2 tan(vfov/2)
    right   =  (cx - 0.5) * 2 tan(hfov/2)
    down    =  1
which is rotated by the vehicle attitude (roll, pitch, yaw; ZYX) into NED and
intersected with flat ground ``agl`` metres below. Attitude compensation
matters: at 12 m AGL an uncorrected 5° pitch shifts the estimate ~1 m.
"""
from __future__ import annotations

import math


def _rot_body_to_ned(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _mul(R, v):
    return tuple(sum(R[i][j] * v[j] for j in range(3)) for i in range(3))


def _transpose(R):
    return tuple(tuple(R[j][i] for j in range(3)) for i in range(3))


def pixel_to_ground(cx: float, cy: float, agl: float, roll: float, pitch: float, yaw: float,
                    hfov_deg: float, vfov_deg: float) -> tuple[float, float] | None:
    """Normalised pixel -> (north, east) offset from the vehicle in metres."""
    tx = math.tan(math.radians(hfov_deg) / 2)
    ty = math.tan(math.radians(vfov_deg) / 2)
    ray_b = (-(cy - 0.5) * 2 * ty, (cx - 0.5) * 2 * tx, 1.0)
    n, e, d = _mul(_rot_body_to_ned(roll, pitch, yaw), ray_b)
    if d <= 1e-3:           # ray at or above the horizon
        return None
    s = agl / d
    return n * s, e * s


def ground_to_pixel(north: float, east: float, agl: float, roll: float, pitch: float, yaw: float,
                    hfov_deg: float, vfov_deg: float) -> tuple[float, float] | None:
    """Inverse of ``pixel_to_ground`` (used by the simulator)."""
    Rt = _transpose(_rot_body_to_ned(roll, pitch, yaw))
    f, r, d = _mul(Rt, (north, east, agl))
    if d <= 1e-3:
        return None
    tx = math.tan(math.radians(hfov_deg) / 2)
    ty = math.tan(math.radians(vfov_deg) / 2)
    cx = 0.5 + (r / d) / (2 * tx)
    cy = 0.5 - (f / d) / (2 * ty)
    return cx, cy


def footprint(agl: float, hfov_deg: float, vfov_deg: float) -> tuple[float, float]:
    """Ground footprint (width, height) in metres for a level nadir camera."""
    return (2 * agl * math.tan(math.radians(hfov_deg) / 2),
            2 * agl * math.tan(math.radians(vfov_deg) / 2))
