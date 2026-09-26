"""Kinematic multirotor model implementing ``FlightLink``.

Good enough to exercise the full onboard stack (navigation, avoidance,
visual servoing, failsafes) without hardware: first-order velocity
response with an acceleration limit, yaw-rate limit, ArduPilot-like mode
behaviour (GUIDED timeout, LAND auto-disarm, RTL climb-return-land), a
battery model and optional wind and GPS noise.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from atlas.comms.base import FlightLink
from atlas.core.geo import LatLon, clamp, ne_to_ll, wrap_180
from atlas.core.state import VehicleState
from atlas.core.timing import SimClock


@dataclass
class SimParams:
    tau: float = 0.45               # velocity response time constant (s)
    max_accel: float = 3.0          # m/s^2
    yaw_rate: float = 90.0          # deg/s
    guided_timeout: float = 3.0     # stop if no setpoint (ArduPilot GUID_TIMEOUT)
    land_speed: float = 0.5
    rtl_alt: float = 45.0            # ArduPilot RTL_ALT: above the tallest surveyed obstacle
    rtl_speed: float = 5.0
    battery_drain_hover: float = 0.20   # % per second
    battery_drain_per_ms: float = 0.02  # extra % per second per m/s
    wind_n: float = 0.0             # mean wind (m/s). ArduPilot's velocity loop (with its
    wind_e: float = 0.0             # integrator) rejects the steady part; what leaks through is
    gust_frac: float = 0.35         # gusts: OU process, sigma = gust_frac*|wind|, tau = gust_tau,
    gust_tau: float = 3.0           # of which `gust_leak` appears as position drift
    gust_leak: float = 0.15
    gps_noise: float = 0.0          # 1-sigma metres
    seed: int = 1


class SimLink(FlightLink):
    def __init__(self, home: LatLon, clock: SimClock, params: SimParams | None = None):
        self.p = params or SimParams()
        self.clock = clock
        self._home = home
        self._rng = random.Random(self.p.seed)
        self.n = self.e = 0.0
        self.alt = 0.0
        self.vn = self.ve = self.vd = 0.0
        self.yaw = 0.0
        self.armed = False
        self.mode = "STABILIZE"
        self.battery = 100.0
        self._sp = (0.0, 0.0, 0.0)
        self._sp_yaw: Optional[float] = None
        self._sp_time = -1e9
        self._takeoff_alt: Optional[float] = None
        self.servos: dict[int, int] = {}
        self.servo_log: list[tuple[float, int, int]] = []
        self.gps_fix, self.satellites, self.hdop = 3, 14, 0.8
        self.fc_alive = True
        self.trace: list[tuple] = []
        self._gust_n = self._gust_e = 0.0

    # FlightLink ----------------------------------------------------------
    def connect(self, timeout: float = 30.0) -> None:
        pass

    def close(self) -> None:
        pass

    def home(self) -> Optional[LatLon]:
        return self._home

    def state(self) -> Optional[VehicleState]:
        noise = self.p.gps_noise
        n = self.n + (self._rng.gauss(0, noise) if noise else 0.0)
        e = self.e + (self._rng.gauss(0, noise) if noise else 0.0)
        ll = ne_to_ll(self._home, n, e)
        now = self.clock.now()
        cells_v = 3.3 + 0.9 * self.battery / 100.0
        return VehicleState(
            stamp=now, lat=ll.lat, lon=ll.lon, alt_rel=self.alt,
            vn=self.vn, ve=self.ve, vd=self.vd, roll=0.0, pitch=0.0, yaw=self.yaw % 360.0,
            armed=self.armed, mode=self.mode, battery_v=4 * cells_v, battery_pct=self.battery,
            gps_fix=self.gps_fix, satellites=self.satellites, hdop=self.hdop,
            fc_heartbeat=now if self.fc_alive else 0.0,
        )

    def set_mode(self, mode: str) -> bool:
        if mode not in ("GUIDED", "LAND", "RTL", "LOITER", "BRAKE", "STABILIZE"):
            return False
        self.mode = mode
        if mode in ("LOITER", "BRAKE"):
            self._sp = (0.0, 0.0, 0.0)
        return True

    def arm(self) -> bool:
        if self.gps_fix < 3:
            return False
        self.armed = True
        return True

    def disarm(self) -> bool:
        if self.alt > 0.2:
            return False
        self.armed = False
        return True

    def takeoff(self, alt: float) -> bool:
        if not self.armed or self.mode != "GUIDED":
            return False
        self._takeoff_alt = alt
        return True

    def send_velocity(self, vn, ve, vd, yaw=None) -> None:
        self._sp = (vn, ve, vd)
        self._sp_yaw = yaw
        self._sp_time = self.clock.now()
        self._takeoff_alt = None

    def set_servo(self, channel: int, pwm: int) -> bool:
        self.servos[channel] = pwm
        self.servo_log.append((self.clock.now(), channel, pwm))
        return True

    # physics -------------------------------------------------------------
    def _target_velocity(self, now: float) -> tuple[float, float, float, Optional[float]]:
        if not self.armed:
            return 0.0, 0.0, 0.0, None
        if self.mode == "LAND":
            return 0.0, 0.0, self.p.land_speed, None
        if self.mode == "RTL":
            dist = math.hypot(self.n, self.e)
            if self.alt < self.p.rtl_alt - 0.3 and dist > 2.0:
                return 0.0, 0.0, -1.5, None
            if dist > 1.0:
                s = min(self.p.rtl_speed, dist)
                yaw = math.degrees(math.atan2(-self.e, -self.n)) % 360
                return -self.n / dist * s, -self.e / dist * s, 0.0, yaw
            return 0.0, 0.0, self.p.land_speed, None
        if self.mode == "GUIDED":
            if self._takeoff_alt is not None:
                err = self._takeoff_alt - self.alt
                return 0.0, 0.0, -clamp(err, -1.5, 1.5) if abs(err) > 0.05 else 0.0, None
            if now - self._sp_time > self.p.guided_timeout:
                return 0.0, 0.0, 0.0, None
            vn, ve, vd = self._sp
            return vn, ve, vd, self._sp_yaw
        return 0.0, 0.0, 0.0, None  # LOITER/BRAKE/STABILIZE-hover

    def step(self, dt: float) -> None:
        now = self.clock.now()
        tvn, tve, tvd, tyaw = self._target_velocity(now)
        # first-order response with accel limit
        a = []
        for cur, tgt in ((self.vn, tvn), (self.ve, tve), (self.vd, tvd)):
            acc = (tgt - cur) / self.p.tau
            a.append(clamp(acc, -self.p.max_accel, self.p.max_accel))
        self.vn += a[0] * dt
        self.ve += a[1] * dt
        self.vd += a[2] * dt
        # gusts: Ornstein-Uhlenbeck process; the mean wind is rejected by the FC
        sig = self.p.gust_frac * math.hypot(self.p.wind_n, self.p.wind_e)
        if sig > 0:
            k = dt / self.p.gust_tau
            self._gust_n += -k * self._gust_n + sig * math.sqrt(2 * k) * self._rng.gauss(0, 1)
            self._gust_e += -k * self._gust_e + sig * math.sqrt(2 * k) * self._rng.gauss(0, 1)
        if self.armed:
            self.n += (self.vn + self.p.gust_leak * self._gust_n) * dt
            self.e += (self.ve + self.p.gust_leak * self._gust_e) * dt
            self.alt -= self.vd * dt
        if self.alt <= 0.0:
            self.alt = 0.0
            self.vd = min(self.vd, 0.0)
            if self.mode in ("LAND", "RTL") and self.armed and math.hypot(self.vn, self.ve) < 0.3:
                self.armed = False  # ArduPilot auto-disarms after touchdown
        if tyaw is not None:
            err = wrap_180(tyaw - self.yaw)
            self.yaw = (self.yaw + clamp(err, -self.p.yaw_rate * dt, self.p.yaw_rate * dt)) % 360
        if self.armed:
            speed = math.hypot(self.vn, self.ve)
            self.battery = max(0.0, self.battery - (self.p.battery_drain_hover + self.p.battery_drain_per_ms * speed) * dt)
        self.trace.append((now, self.n, self.e, self.alt, self.yaw, self.mode))
