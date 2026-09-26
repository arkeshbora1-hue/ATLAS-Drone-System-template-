"""Mission executive: the 10 Hz loop that ties every subsystem together.

Each ``step()``:
  1. read the latest VehicleState, RangeSnapshot and TargetEstimate
  2. SafetyMonitor.evaluate -> may pre-empt the mission with a failsafe
  3. run the current phase handler -> a Command
  4. send the Command to the flight controller
  5. write one blackbox row

Phases
------
  PREFLIGHT -> ARMING -> TAKEOFF -> TRANSIT -> SEARCH -> APPROACH -> DESCEND
            -> RELEASE -> CLIMB -> RETURN -> LAND -> DONE
  any airborne phase -> FAILSAFE(HOLD | RTL | LAND)
  HOLD that clears within the timeout resumes the interrupted phase.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from enum import Enum
from typing import Optional

from atlas.avoidance.reactive import Action as AvoidAction
from atlas.avoidance.reactive import ReactiveAvoider
from atlas.core.geo import LatLon, clamp, limit_norm, ll_to_ne
from atlas.core.state import Command, Latest, RangeSnapshot, Sector, TargetEstimate, VehicleState
from atlas.navigation.guidance import WaypointGuidance
from atlas.navigation.mission import MissionSpec, Route, build_outbound, build_return, expanding_square
from atlas.payload.release import PayloadRelease, ReleaseState
from atlas.safety.monitor import Action as SafeAction
from atlas.safety.monitor import SafetyMonitor
from atlas.telemetry.blackbox import Blackbox

log = logging.getLogger("atlas.exec")


class Phase(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    ARMING = "ARMING"
    TAKEOFF = "TAKEOFF"
    TRANSIT = "TRANSIT"
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    RELEASE = "RELEASE"
    CLIMB = "CLIMB"
    RETURN = "RETURN"
    LAND = "LAND"
    FAILSAFE = "FAILSAFE"
    DONE = "DONE"
    ABORTED = "ABORTED"


AIRBORNE = {Phase.TAKEOFF, Phase.TRANSIT, Phase.SEARCH, Phase.APPROACH, Phase.DESCEND,
            Phase.RELEASE, Phase.CLIMB, Phase.RETURN}
HORIZONTAL_SECTORS = (Sector.FRONT, Sector.FRONT_LEFT, Sector.FRONT_RIGHT, Sector.LEFT, Sector.RIGHT)


class Executive:
    def __init__(self, cfg, link, spec: MissionSpec, clock, range_slot: Latest, target_slot: Latest,
                 blackbox: Optional[Blackbox] = None, perception=None, temp_fn=None):
        self.cfg, self.link, self.spec, self.clock = cfg, link, spec, clock
        self.range_slot, self.target_slot = range_slot, target_slot
        self.bb = blackbox or Blackbox("logs", enabled=False)
        self.perception = perception
        self.temp_fn = temp_fn
        v = cfg.vehicle
        self.guidance = WaypointGuidance(v.cruise_speed, v.climb_speed, v.descent_speed,
                                         v.waypoint_accept_radius)
        self.payload = PayloadRelease(cfg, link, clock)
        self.phase = Phase.IDLE
        self.phase_t = clock.now()
        self.home: Optional[LatLon] = None
        self.safety: Optional[SafetyMonitor] = None
        self.avoider: Optional[ReactiveAvoider] = None
        self.route: Optional[Route] = None
        self.dz_n = self.dz_e = 0.0
        self.delivered = False
        self.failsafe_action: Optional[SafeAction] = None
        self.resume_phase: Optional[Phase] = None
        self.hold_clear_since: Optional[float] = None
        self._blocked_since: Optional[float] = None
        self._centred = 0
        self._crumbs: deque[tuple[float, float]] = deque(maxlen=30)   # breadcrumb trail, 1 m spacing
        self._prog_key, self._prog_best, self._prog_t = None, math.inf, 0.0
        self._egress_until: Optional[float] = None
        self._i_n = self._i_e = 0.0
        self._approach_started = 0.0
        self._preflight_reported: tuple = ()
        self._search_route: Optional[Route] = None
        self.last_cmd = Command("none")
        self.last_avoid = ""
        self.events: list[tuple] = []

    # ---------------------------------------------------------------- helpers
    def _event(self, kind: str, **data) -> None:
        now = self.clock.now()
        self.events.append((round(now, 2), kind, data))
        self.bb.event(now, kind, **data)
        log.info("%s %s", kind, data)

    def _goto(self, phase: Phase, **why) -> None:
        self._event("phase", frm=self.phase.value, to=phase.value, **why)
        self.phase, self.phase_t = phase, self.clock.now()

    def _in_phase(self) -> float:
        return self.clock.now() - self.phase_t

    def start(self) -> None:
        self._goto(Phase.PREFLIGHT)

    @property
    def finished(self) -> bool:
        return self.phase in (Phase.DONE, Phase.ABORTED)

    def _init_home(self, home: LatLon) -> None:
        self.home = home
        cfg = self.cfg
        alt = self.spec.alt or cfg.vehicle.cruise_alt
        self.cruise_alt = alt
        self.safety = SafetyMonitor(cfg, home, **({"temp_fn": self.temp_fn} if self.temp_fn else {}))
        self.avoider = ReactiveAvoider(cfg, cfg.safety.geofence.max_alt)
        self.route = build_outbound(self.spec, home, alt)
        self.dz_n, self.dz_e = ll_to_ne(home, self.spec.dropzone)
        self._event("home", lat=home.lat, lon=home.lon, legs=len(self.route.wps))

    # ------------------------------------------------------------------- tick
    def step(self) -> Command:
        now = self.clock.now()
        vs: Optional[VehicleState] = self.link.state()
        rs: Optional[RangeSnapshot] = self.range_slot.get()
        tgt: Optional[TargetEstimate] = self.target_slot.get()
        if self.home is None:
            h = self.link.home()
            if h is None:
                return Command("none", note="waiting for home")
            self._init_home(h)
        if vs is None:
            return Command("none", note="no telemetry")
        n, e = ll_to_ne(self.home, LatLon(vs.lat, vs.lon))

        # --- safety arbitration ------------------------------------------
        verdict = self.safety.evaluate(now, vs, rs, airborne=vs.armed and self.phase in AIRBORNE)
        if self.phase in AIRBORNE and verdict.action > SafeAction.CONTINUE:
            self._enter_failsafe(verdict.action, verdict.reasons)

        handler = getattr(self, f"_ph_{self.phase.value.lower()}")
        cmd: Command = handler(now, vs, rs, tgt, n, e)
        self._apply(cmd)
        self._log(now, vs, rs, tgt, n, e, cmd, verdict)
        return cmd

    def _apply(self, cmd: Command) -> None:
        self.last_cmd = cmd
        vmax = self.cfg.vehicle.max_speed
        if cmd.kind == "velocity":
            vn, ve = limit_norm(cmd.vn, cmd.ve, vmax)
            self.link.send_velocity(vn, ve, cmd.vd, cmd.yaw)
        elif cmd.kind == "hold":
            self.link.send_velocity(0.0, 0.0, 0.0, None)

    # -------------------------------------------------------------- failsafe
    def _enter_failsafe(self, action: SafeAction, reasons: list[str]) -> None:
        if self.phase == Phase.FAILSAFE and self.failsafe_action is not None and action <= self.failsafe_action:
            return
        if self.phase != Phase.FAILSAFE:
            self.resume_phase = self.phase
        self.failsafe_action = action
        self.hold_clear_since = None
        self._event("failsafe", action=action.name, reasons=reasons)
        if action == SafeAction.RTL:
            rs = self.range_slot.get()
            near = min((rs.distance(x) for x in HORIZONTAL_SECTORS), default=math.inf) if rs else math.inf
            if near < self.cfg.avoidance.safe_distance and self._crumbs:
                # RTL starts with a vertical climb that has no obstacle sensing;
                # first retrace our trail into open space (max 8 s)
                self._egress_until = self.clock.now() + 8.0
                self._event("rtl_egress", nearest=round(near, 2))
            else:
                self.link.rtl()
        elif action == SafeAction.LAND:
            self.link.land()
        self.phase, self.phase_t = Phase.FAILSAFE, self.clock.now()

    def _ph_failsafe(self, now, vs, rs, tgt, n, e) -> Command:
        if not vs.armed:
            self._goto(Phase.DONE if self.delivered else Phase.ABORTED, reason="landed after failsafe")
            return Command("none")
        verdict = self.safety.evaluate(now, vs, rs, airborne=True)
        if verdict.action > self.failsafe_action:
            self._enter_failsafe(verdict.action, verdict.reasons)
        if self.failsafe_action == SafeAction.HOLD:
            if verdict.action == SafeAction.CONTINUE:
                self.hold_clear_since = self.hold_clear_since or now
                if now - self.hold_clear_since > 1.0:
                    self._event("failsafe_cleared", resume=self.resume_phase.value)
                    self.phase, self.phase_t = self.resume_phase, now
                    self.failsafe_action = None
                    return Command("hold")
            else:
                self.hold_clear_since = None
            if self._in_phase() > self.cfg.avoidance.blocked_hold_timeout:
                self._enter_failsafe(SafeAction.RTL, ["hold timeout"])
            return Command("hold", note="failsafe hold")
        if self.failsafe_action == SafeAction.RTL and self._egress_until is not None:
            near = min(rs.distance(x) for x in HORIZONTAL_SECTORS) if rs else 0.0
            if near > self.cfg.avoidance.clear_distance or now > self._egress_until or not self._crumbs:
                self._egress_until = None
                self.link.rtl()
                return Command("none", note="RTL")
            bn, be = self._retrace_velocity(n, e, (0.0, 0.0))
            return Command("velocity", bn, be, 0.0, None, note="egress before RTL")
        return Command("none", note=f"failsafe {self.failsafe_action.name}")

    # ----------------------------------------------------------- ground ops
    def _ph_idle(self, *a) -> Command:
        return Command("none")

    def _ph_preflight(self, now, vs, rs, tgt, n, e) -> Command:
        pts = [(w.n, w.e, w.alt) for w in self.route.wps]
        ready = self.perception is None or getattr(self.perception, "enabled", True)
        fails = self.safety.preflight(vs, rs, now, pts, ready)
        if not fails and not self.link.authorized():
            fails = [f"waiting for pilot MISSION-GO switch (RC{self.cfg.mavlink.auth_rc_channel} high)"]
        if fails:
            if tuple(fails) != self._preflight_reported:
                self._preflight_reported = tuple(fails)
                self._event("preflight_fail", fails=fails)
            return Command("none", note="preflight blocked")
        self._event("preflight_pass")
        self.payload.arm_lock()
        self.link.set_mode("GUIDED")
        self.link.arm()
        self._goto(Phase.ARMING)
        return Command("none")

    def _ph_arming(self, now, vs, rs, tgt, n, e) -> Command:
        if vs.armed and vs.mode == "GUIDED":
            self.link.takeoff(self.cruise_alt)
            self._goto(Phase.TAKEOFF)
        elif self._in_phase() > 10.0:
            self._goto(Phase.ABORTED, reason="arming timeout")
        elif int(self._in_phase() * 10) % 20 == 19:   # retry every 2 s
            self.link.set_mode("GUIDED")
            self.link.arm()
        return Command("none")

    def _ph_takeoff(self, now, vs, rs, tgt, n, e) -> Command:
        if vs.alt_rel >= 0.95 * self.cruise_alt:
            self._goto(Phase.TRANSIT)
            return Command("hold")
        if self._in_phase() > 40.0:
            self._enter_failsafe(SafeAction.LAND, ["takeoff timeout"])
        return Command("none", note="climbing")

    # ------------------------------------------------------- route following
    def _fly_route(self, route: Route, now, vs, rs, n, e, speed_limit: float | None = None) -> tuple[Command, bool]:
        """Follow ``route`` with reactive avoidance. Returns (command, finished)."""
        wp = route.active
        if wp is None:
            return Command("hold"), True
        if rs is None:
            return Command("hold", note="no range data"), False
        # the avoider reasons about the next *real* waypoint (skip detours)
        goal = next((w for w in route.remaining() if w.tag not in ("detour", "climb")), wp)
        d = self.avoider.evaluate(now, rs, n, e, vs.alt_rel, vs.yaw, wp if wp.tag == "climb" else goal)
        self.last_avoid = d.action.value
        if d.action != AvoidAction.BLOCKED:
            self._blocked_since = None
        if d.action == AvoidAction.BACKOFF:
            bn, be = self._retrace_velocity(n, e, d.backoff_vel)
            return Command("velocity", bn, be, 0.0, None, note=d.reason), False
        if d.action == AvoidAction.REPLAN:
            route.insert_before_active(d.detour)
            self._event("replan", side=d.side, front=round(d.front, 2),
                        detour=[(round(w.n, 1), round(w.e, 1)) for w in d.detour])
            wp = route.active
        elif d.action == AvoidAction.CLIMB:
            route.drop_detours()              # lateral detours are moot if we go over
            route.insert_before_active(d.detour)
            new_alt = d.detour[0].alt
            for w in route.remaining():
                if w.tag not in ("detour", "climb"):
                    w.alt = max(w.alt, new_alt)
                    break
            self._event("avoid_climb", to_alt=new_alt)
            wp = route.active
        elif d.action == AvoidAction.BLOCKED:
            self._blocked_since = self._blocked_since or now
            if now - self._blocked_since > self.cfg.avoidance.blocked_hold_timeout:
                self._enter_failsafe(SafeAction.RTL, ["path blocked: " + d.reason])
            return Command("hold", note="blocked"), False

        limit = self.avoider.speed_limit(self.avoider.front_threat(rs))
        if speed_limit is not None:
            limit = min(limit, speed_limit)
        g = self.guidance.update(n, e, vs.alt_rel, vs.yaw, wp, speed_limit=max(limit, 0.3),
                                 ground_speed=math.hypot(vs.vn, vs.ve))
        self._drop_crumb(n, e, math.hypot(g.vn, g.ve))
        rn, re_ = self.avoider.side_repulsion(rs, vs.yaw)
        g.vn, g.ve = g.vn + rn, g.ve + re_
        self._check_progress(now, n, e, goal)
        if g.reached:
            if wp.tag not in ("detour", "climb"):
                self.avoider.new_leg()
            route.advance()
            self._event("waypoint", tag=wp.tag, n=round(wp.n, 1), e=round(wp.e, 1))
            return Command("velocity", g.vn, g.ve, g.vd, g.yaw), route.finished
        return Command("velocity", g.vn, g.ve, g.vd, g.yaw), False

    def _check_progress(self, now: float, n: float, e: float, goal) -> None:
        """Stuck detector: reactive planners can oscillate (back off, replan,
        back off...) without ever tripping a single rule. If the distance to
        the current real waypoint has not improved by 1 m in
        ``no_progress_timeout`` seconds, give up and RTL (RTL_ALT clears
        every surveyed obstacle)."""
        d = math.hypot(goal.n - n, goal.e - e)
        key = (round(goal.n, 1), round(goal.e, 1))
        if self._prog_key != key or d < self._prog_best - 1.0:
            self._prog_key, self._prog_best, self._prog_t = key, d, now
        elif now - self._prog_t > self.cfg.avoidance.no_progress_timeout:
            self._prog_t = now
            self._enter_failsafe(SafeAction.RTL, [f"no progress toward waypoint for "
                                                  f"{self.cfg.avoidance.no_progress_timeout:.0f} s"])

    def _drop_crumb(self, n: float, e: float, speed_cmd: float) -> None:
        """Record the trail only while genuinely flying forward, so hovering
        and GPS jitter near an obstacle never plant crumbs next to it."""
        if speed_cmd < 0.8:
            return
        if not self._crumbs or math.hypot(n - self._crumbs[-1][0], e - self._crumbs[-1][1]) >= 1.0:
            self._crumbs.append((n, e))

    def _retrace_velocity(self, n: float, e: float, fallback: tuple[float, float]) -> tuple[float, float]:
        """Back away along our own breadcrumb trail.

        There is no rear-facing sensor, so reversing blindly along -nose can
        hit something we never saw (e.g. after yawing onto a detour). The
        trail we just flew is the only space known to be free, so BACKOFF
        heads for the most recent crumb at least 1.5 m away.
        """
        speed = self.cfg.avoidance.backoff_speed
        # consume crumbs we have already backed past, then head for the next one
        while self._crumbs and math.hypot(self._crumbs[-1][0] - n, self._crumbs[-1][1] - e) < 0.7:
            self._crumbs.pop()
        if self._crumbs:
            dn, de = self._crumbs[-1][0] - n, self._crumbs[-1][1] - e
            dist = math.hypot(dn, de)
            return dn / dist * speed, de / dist * speed
        return fallback

    def _ph_transit(self, now, vs, rs, tgt, n, e) -> Command:
        cmd, done = self._fly_route(self.route, now, vs, rs, n, e)
        if done:
            self._begin_search()
        return cmd

    # ----------------------------------------------------- drop-zone phases
    def _begin_search(self) -> None:
        a = self.cfg.approach
        self._search_route = Route(expanding_square(self.dz_n, self.dz_e, self.cfg.vehicle.search_alt,
                                                    a.search_leg, min(a.search_max_radius, self.spec.search_radius)))
        self._search_started = self.clock.now()
        self._goto(Phase.SEARCH, waypoints=len(self._search_route.wps))

    def _target_fresh(self, tgt: Optional[TargetEstimate], now: float) -> bool:
        """Entry test for APPROACH: confirmed (N of M) and recently seen."""
        return (tgt is not None and tgt.confirmed
                and now - tgt.stamp < self.cfg.approach.lost_target_timeout)

    def _target_held(self, tgt: Optional[TargetEstimate], now: float) -> bool:
        """Keep-alive test once locked on: any positive frame within the timeout.
        (Requiring N-of-M every tick would drop lock on brief occlusions.)"""
        return tgt is not None and now - tgt.stamp < self.cfg.approach.lost_target_timeout

    def _ph_search(self, now, vs, rs, tgt, n, e) -> Command:
        at_alt = abs(vs.alt_rel - self.cfg.vehicle.search_alt) < 2.0
        if at_alt and self._target_fresh(tgt, now):
            self._centred = 0
            self._i_n = self._i_e = 0.0
            if self._approach_started == 0.0:
                self._approach_started = now
            self._goto(Phase.APPROACH, tgt_n=round(tgt.north, 2), tgt_e=round(tgt.east, 2), hits=tgt.hits)
            return Command("hold")
        cmd, done = self._fly_route(self._search_route, now, vs, rs, n, e, speed_limit=2.0)
        if done or now - self._search_started > self.cfg.approach.search_timeout:
            self._event("dropzone_not_found")
            self._goto(Phase.CLIMB)
        return cmd

    def _servo_to_target(self, tgt: TargetEstimate, n: float, e: float, rs) -> tuple[float, float, float]:
        """PI visual servo on the horizontal offset to the marker.

        P alone leaves a steady-state offset of (disturbance / kp) - e.g. 0.5 m/s
        of wind-induced drift at kp 0.6 sits ~0.8 m off-centre, outside the
        0.6 m release radius forever. The clamped integral term removes it.
        """
        a = self.cfg.approach
        now = self.clock.now()
        dt = 1.0 / self.cfg.loop.executive_hz
        if now - tgt.stamp < 0.3:
            # fresh camera measurement: servo on the image-relative offset
            on, oe = tgt.rel_n, tgt.rel_e
        else:
            # brief gap in detections: steer to the remembered ground position
            on, oe = tgt.north - n, tgt.east - e
        self._i_n, self._i_e = limit_norm(self._i_n + a.ki * on * dt, self._i_e + a.ki * oe * dt, a.i_limit)
        vn, ve = limit_norm(a.kp * on + self._i_n, a.kp * oe + self._i_e, a.max_speed)
        # never creep horizontally into something the side sensors can see
        if rs is not None and min(rs.distance(s) for s in HORIZONTAL_SECTORS) < self.cfg.avoidance.critical_distance:
            vn = ve = 0.0
        return vn, ve, math.hypot(on, oe)

    def _approach_timed_out(self, now) -> bool:
        if now - self._approach_started > self.cfg.approach.approach_timeout:
            self._event("approach_timeout")
            self._goto(Phase.CLIMB)
            return True
        return False

    def _ph_approach(self, now, vs, rs, tgt, n, e) -> Command:
        a = self.cfg.approach
        if self._approach_timed_out(now):
            return Command("hold")
        if not self._target_held(tgt, now):
            self._event("target_lost", phase="APPROACH")
            self._goto(Phase.SEARCH)
            return Command("hold")
        vn, ve, off = self._servo_to_target(tgt, n, e, rs)
        vd = self.guidance.vertical(vs.alt_rel, self.cfg.vehicle.search_alt)
        # leaky counter: a single noisy frame costs 2 ticks instead of a full reset
        self._centred = self._centred + 1 if off < a.centred_radius else max(0, self._centred - 2)
        if self._centred >= a.centred_frames:
            self._goto(Phase.DESCEND, offset=round(off, 2))
        return Command("velocity", vn, ve, vd, None, note=f"offset {off:.2f} m")

    def _agl(self, vs, rs) -> float:
        if rs is not None and Sector.DOWN not in rs.unhealthy and rs.distance(Sector.DOWN) != float("inf"):
            return rs.distance(Sector.DOWN)
        return vs.alt_rel

    def _ph_descend(self, now, vs, rs, tgt, n, e) -> Command:
        a, v = self.cfg.approach, self.cfg.vehicle
        if self._approach_timed_out(now):
            return Command("hold")
        if not self._target_held(tgt, now):
            self._event("target_lost", phase="DESCEND")
            self._goto(Phase.SEARCH)
            return Command("hold")
        vn, ve, off = self._servo_to_target(tgt, n, e, rs)
        agl = self._agl(vs, rs)
        err = agl - v.release_alt_agl
        if err <= 0.15 and off < a.centred_radius:
            self._goto(Phase.RELEASE, agl=round(agl, 2), offset=round(off, 2))
            return Command("velocity", vn, ve, 0.0, None)
        # descend only while centred; pause and re-centre if drifted
        vd = clamp(0.5 * err, 0.2, v.descent_speed) if off < 2 * a.centred_radius else 0.0
        if err <= 0.15:
            vd = 0.0
        return Command("velocity", vn, ve, vd, None, note=f"agl {agl:.2f} off {off:.2f}")

    def _ph_release(self, now, vs, rs, tgt, n, e) -> Command:
        if self.payload.state == ReleaseState.LOCKED:
            self.payload.release()
            self._event("payload_released", n=round(n, 2), e=round(e, 2),
                        err_to_dz=round(math.hypot(n - self.dz_n, e - self.dz_e), 2))
        if self.payload.update() == ReleaseState.DONE and self._in_phase() > self.cfg.payload.open_time + 0.5:
            self.delivered = True
            self._goto(Phase.CLIMB)
        return Command("hold")

    def _ph_climb(self, now, vs, rs, tgt, n, e) -> Command:
        vd = self.guidance.vertical(vs.alt_rel, self.cruise_alt)
        if abs(vs.alt_rel - self.cruise_alt) < 1.0:
            self.route = build_return(self.spec, self.home, self.cruise_alt)
            self.avoider.new_leg()
            self._goto(Phase.RETURN, legs=len(self.route.wps))
        return Command("velocity", 0.0, 0.0, vd, None)

    def _ph_return(self, now, vs, rs, tgt, n, e) -> Command:
        cmd, done = self._fly_route(self.route, now, vs, rs, n, e)
        if done:
            self.link.land()
            self._goto(Phase.LAND)
        return cmd

    def _ph_land(self, now, vs, rs, tgt, n, e) -> Command:
        if not vs.armed:
            self._goto(Phase.DONE if self.delivered else Phase.ABORTED, delivered=self.delivered)
        return Command("none", note="landing")

    def _ph_done(self, *a) -> Command:
        return Command("none")

    _ph_aborted = _ph_done

    # ------------------------------------------------------------------ log
    def _log(self, now, vs, rs, tgt, n, e, cmd, verdict) -> None:
        r = rs.sectors if rs else {}
        g = lambda s: r.get(s, float("nan"))
        self.bb.row(
            t=now, state=self.phase.value, mode=vs.mode, armed=int(vs.armed), n=n, e=e, alt=vs.alt_rel,
            yaw=vs.yaw, vn=vs.vn, ve=vs.ve, vd=vs.vd, battery_pct=vs.battery_pct, sats=vs.satellites,
            hdop=vs.hdop, r_front=g(Sector.FRONT), r_front_left=g(Sector.FRONT_LEFT),
            r_front_right=g(Sector.FRONT_RIGHT), r_left=g(Sector.LEFT), r_right=g(Sector.RIGHT),
            r_down=g(Sector.DOWN), avoid=self.last_avoid,
            tgt_n=tgt.north if tgt else "", tgt_e=tgt.east if tgt else "",
            tgt_confirmed=int(tgt.confirmed) if tgt else "", cmd=cmd.kind, cmd_vn=cmd.vn, cmd_ve=cmd.ve,
            cmd_vd=cmd.vd, cmd_yaw=cmd.yaw if cmd.yaw is not None else "",
            safety=";".join(verdict.reasons),
        )
