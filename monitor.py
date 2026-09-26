"""Safety monitor: pre-flight gate and in-flight failsafe arbitration.

Evaluated every executive tick *before* the mission state machine. The most
severe triggered action wins:  LAND > RTL > HOLD > CONTINUE.

In-flight rules
---------------
  battery <= land threshold ............................ LAND
  GPS degraded (no 3D fix / few sats / high HDOP) ...... LAND  (can't navigate home)
  battery <= RTL threshold ............................. RTL
  geofence breach ...................................... RTL
  CPU over temperature ................................. RTL
  range data stale or a forward sector unhealthy ....... HOLD  (escalates to RTL after timeout)
  FC heartbeat lost .................................... HOLD  (nothing we can command;
                                                                FC failsafes take over)
The flight controller's own failsafes (battery, GCS/RC loss, EKF, fence,
GUIDED timeout) remain enabled underneath as an independent layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from atlas.core.geo import LatLon, ll_to_ne
from atlas.core.state import FORWARD_SECTORS, RangeSnapshot, VehicleState
from atlas.safety.geofence import Geofence


class Action(IntEnum):
    CONTINUE = 0
    HOLD = 1
    RTL = 2
    LAND = 3


@dataclass
class Verdict:
    action: Action = Action.CONTINUE
    reasons: list[str] = field(default_factory=list)

    def escalate(self, action: Action, reason: str) -> None:
        self.reasons.append(reason)
        if action > self.action:
            self.action = action


def cpu_temp_c() -> float | None:
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return int(p.read_text()) / 1000.0
    except (OSError, ValueError):
        return None


class SafetyMonitor:
    def __init__(self, cfg, home: LatLon, temp_fn=cpu_temp_c):
        s = cfg.safety
        self.cfg = cfg
        self.home = home
        self.fence = Geofence(home, s.geofence.max_alt, s.geofence.max_radius, s.geofence.polygon)
        self.temp_fn = temp_fn
        self.link_timeout = cfg.mavlink.link_timeout
        self.sensor_stale = cfg.sensors.stale_after

    def gps_ok(self, vs: VehicleState) -> tuple[bool, str]:
        s = self.cfg.safety
        if vs.gps_fix < 3:
            return False, f"GPS fix {vs.gps_fix}"
        if vs.satellites < s.min_satellites:
            return False, f"{vs.satellites} satellites < {s.min_satellites}"
        if vs.hdop > s.max_hdop:
            return False, f"HDOP {vs.hdop:.1f} > {s.max_hdop}"
        return True, ""

    def preflight(self, vs: VehicleState | None, rs: RangeSnapshot | None, now: float,
                  route_pts, perception_ready: bool) -> list[str]:
        s = self.cfg.safety
        fails = []
        if vs is None:
            return ["no telemetry from flight controller"]
        if now - vs.fc_heartbeat > self.link_timeout:
            fails.append("FC heartbeat stale")
        ok, why = self.gps_ok(vs)
        if not ok:
            fails.append(why)
        if 0 <= vs.battery_pct < s.battery_rtl_pct + 20:
            fails.append(f"battery {vs.battery_pct:.0f}% too low to launch")
        if vs.battery_v and vs.battery_v / s.battery_cells < s.min_cell_voltage + 0.25:
            fails.append(f"cell voltage {vs.battery_v / s.battery_cells:.2f} V too low")
        if rs is None or now - rs.stamp > self.sensor_stale:
            fails.append("range sensors not reporting")
        elif rs.unhealthy:
            fails.append("unhealthy sectors: " + ",".join(sorted(x.value for x in rs.unhealthy)))
        ok, why = self.fence.contains_route(route_pts)
        if not ok:
            fails.append("route breaches geofence: " + why)
        if not perception_ready:
            fails.append("perception pipeline not running")
        t = self.temp_fn()
        if t is not None and t > s.max_cpu_temp_c - 10:
            fails.append(f"CPU {t:.0f} C")
        return fails

    def evaluate(self, now: float, vs: VehicleState | None, rs: RangeSnapshot | None,
                 airborne: bool) -> Verdict:
        v = Verdict()
        if vs is None or now - vs.fc_heartbeat > self.link_timeout:
            v.escalate(Action.HOLD, "flight-controller link lost")
            return v
        if not airborne:
            return v
        s = self.cfg.safety
        if 0 <= vs.battery_pct <= s.battery_land_pct:
            v.escalate(Action.LAND, f"battery {vs.battery_pct:.0f}% <= {s.battery_land_pct}%")
        elif 0 <= vs.battery_pct <= s.battery_rtl_pct:
            v.escalate(Action.RTL, f"battery {vs.battery_pct:.0f}% <= {s.battery_rtl_pct}%")
        ok, why = self.gps_ok(vs)
        if not ok:
            v.escalate(Action.LAND, "GPS degraded: " + why)
        n, e = ll_to_ne(self.home, LatLon(vs.lat, vs.lon))
        ok, why = self.fence.check(n, e, vs.alt_rel)
        if not ok:
            v.escalate(Action.RTL, "geofence: " + why)
        t = self.temp_fn()
        if t is not None and t > s.max_cpu_temp_c:
            v.escalate(Action.RTL, f"CPU temperature {t:.0f} C")
        if rs is None or now - rs.stamp > self.sensor_stale:
            v.escalate(Action.HOLD, "range data stale")
        elif any(sec in rs.unhealthy for sec in FORWARD_SECTORS):
            v.escalate(Action.HOLD, "forward range sector unhealthy")
        return v
