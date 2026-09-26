"""Reactive obstacle avoidance with local path re-planning.

Runs every executive tick (10 Hz) on the fused sector distances.

    front = min(FRONT, FRONT_LEFT, FRONT_RIGHT)

    front < critical_distance      -> BACKOFF  (reverse along body X until > safe)
    front < safe_distance          -> REPLAN:
        pick the side (left/right) with the most clearance, biased toward
        the goal; splice two detour waypoints into the route:
            D1 = lateral offset of `detour_lateral` metres to that side
                 (perpendicular to the current heading)
            D2 = D1 + `detour_forward` metres along the current heading
        then continue to the original waypoint. Because guidance is
        yaw-first, the front sensors re-check each detour leg as it is flown,
        so a detour that is itself blocked triggers another replan.
    same side blocks twice more    -> CLIMB once by `climb_step` (maybe it is short)
    both sides blocked             -> CLIMB by `climb_step` (if under ceiling)
    otherwise / too many replans   -> BLOCKED (hold; executive escalates to RTL)

A short cooldown after each replan stops the sensors, which sweep across
the obstacle while the vehicle yaws onto the detour, from re-triggering on
the same obstacle. The critical-distance check ignores the cooldown.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from atlas.core.geo import body_to_ne, heading_ne, ne_to_body, wrap_180
from atlas.core.state import RangeSnapshot, Sector
from atlas.navigation.mission import Waypoint


class Action(str, Enum):
    CLEAR = "CLEAR"
    BACKOFF = "BACKOFF"
    REPLAN = "REPLAN"
    CLIMB = "CLIMB"
    BLOCKED = "BLOCKED"


@dataclass
class Decision:
    action: Action
    detour: list[Waypoint] = field(default_factory=list)
    backoff_vel: tuple[float, float] = (0.0, 0.0)
    front: float = float("inf")
    side: str = ""
    reason: str = ""


class ReactiveAvoider:
    def __init__(self, cfg, max_alt: float):
        a = cfg.avoidance
        self.safe = a.safe_distance
        self.critical = a.critical_distance
        self.clear = a.clear_distance
        self.lateral = a.detour_lateral
        self.forward = a.detour_forward
        self.backoff_speed = a.backoff_speed
        self.climb_step = a.climb_step
        self.max_replans = a.max_replans
        self.max_alt = max_alt
        self.cooldown = a.replan_cooldown
        self.decel = a.brake_decel
        self.reaction = a.reaction_time
        self.sensor_range = min(cfg.sensors.ultrasonic_max_range, cfg.sensors.tof_max_range)
        self.corridor = a.corridor_half_width
        self._diag_sin = math.sin(math.radians(35.0 - 13.5))
        self._last_replan = -1e9
        self._backing_off = False
        self.replans_this_leg = 0
        self.leg_side: str | None = None
        self.side_streak = 0
        self._climbed_this_leg = False
        self.events: list[tuple] = []

    def speed_limit(self, front: float) -> float:
        """Speed governor: fastest forward speed that can still stop before
        ``critical_distance`` given reaction time t and braking decel a:

            d = v*t + v^2 / (2a)   ->   v = a * (-t + sqrt(t^2 + 2d/a))

        A clear reading counts as the sensor's maximum range, because beyond
        that the vehicle is blind. With 4 m sensors this caps cruise at
        ~2.65 m/s (0.6 s reaction, 2.5 m/s^2), which is why ATLAS cruises at 2.5 m/s.
        """
        d = min(front, self.sensor_range) - self.critical
        if d <= 0:
            return 0.0
        a, t = self.decel, self.reaction
        return a * (-t + math.sqrt(t * t + 2 * d / a))

    def side_repulsion(self, snap: RangeSnapshot, yaw: float, radius: float = 1.8,
                       gain: float = 0.8) -> tuple[float, float]:
        """Small sideways push away from anything the side or diagonal sensors
        place within ``radius`` laterally; keeps clearance when sliding past a corner that the
        forward sectors have already passed. Returns an (north, east) velocity."""
        push_right = 0.0
        # lateral distance: LEFT/RIGHT directly; diagonals projected (d * sin 35°)
        s35 = math.sin(math.radians(35.0))
        dl = min(snap.distance(Sector.LEFT), snap.distance(Sector.FRONT_LEFT) * s35)
        dr = min(snap.distance(Sector.RIGHT), snap.distance(Sector.FRONT_RIGHT) * s35)
        if dl < radius:
            push_right += gain * (radius - dl)
        if dr < radius:
            push_right -= gain * (radius - dr)
        return body_to_ne(0.0, push_right, yaw) if push_right else (0.0, 0.0)

    def front_threat(self, snap: RangeSnapshot) -> float:
        """Distance to the nearest return that is actually IN the flight corridor.

        FRONT counts as-is. A diagonal sector (±35°, ~27° beam) only counts if
        the return could lie within ``corridor`` metres of the flight line,
        using the beam's inner edge (worst case): d * sin(35° - 13.5°) < corridor.
        Without this, the diagonal sensors sweeping along a building we are
        already sliding past keep re-triggering replans.
        """
        threat = snap.distance(Sector.FRONT)
        for sec in (Sector.FRONT_LEFT, Sector.FRONT_RIGHT):
            d = snap.distance(sec)
            if d * self._diag_sin < self.corridor:
                threat = min(threat, d)
        return threat

    def new_leg(self) -> None:
        """Called when the route reaches a real (non-detour) waypoint."""
        self.replans_this_leg = 0
        self.leg_side = None
        self.side_streak = 0
        self._climbed_this_leg = False

    def evaluate(self, now: float, snap: RangeSnapshot, n: float, e: float, alt: float,
                 yaw: float, goal: Waypoint) -> Decision:
        front = self.front_threat(snap)
        goal_dist = math.hypot(goal.n - n, goal.e - e)

        # --- critical: too close, back away regardless of cooldown ----------
        if front < self.critical or (self._backing_off and front < self.safe):
            self._backing_off = True
            bn, be = body_to_ne(-self.backoff_speed, 0.0, yaw)
            return Decision(Action.BACKOFF, backoff_vel=(bn, be), front=front,
                            reason=f"front {front:.2f} m < critical {self.critical} m")
        self._backing_off = False

        # obstacle is beyond the goal (e.g. a wall behind the drop zone): ignore
        if front >= self.safe or goal_dist < front - 0.5:
            return Decision(Action.CLEAR, front=front)
        if now - self._last_replan < self.cooldown:
            return Decision(Action.CLEAR, front=front, reason="cooldown")

        # --- replan -------------------------------------------------------
        if self.replans_this_leg >= self.max_replans:
            return self._event(now, Decision(Action.BLOCKED, front=front,
                                             reason=f"{self.replans_this_leg} replans on this leg"))
        left = min(snap.distance(Sector.FRONT_LEFT), snap.distance(Sector.LEFT))
        right = min(snap.distance(Sector.FRONT_RIGHT), snap.distance(Sector.RIGHT))
        # goal side bias: + means goal is to the right of the nose
        goal_rel = wrap_180(heading_ne(goal.n - n, goal.e - e) - yaw)
        left_ok = left > self.safe
        right_ok = right > self.safe
        # escalation ladder: side-step, wider side-step, then try going over
        escalate = self.side_streak >= 1 and self.replans_this_leg >= 2
        can_climb = alt + self.climb_step <= self.max_alt - 2.0
        if escalate and can_climb and not self._climbed_this_leg:
            self._climbed_this_leg = True
            self._last_replan = now
            self.replans_this_leg += 1
            return self._event(now, Decision(Action.CLIMB, detour=[Waypoint(n, e, alt + self.climb_step, "climb")],
                                             front=front, reason="side-steps not clearing; trying over"))
        if left_ok or right_ok:
            committed = {"left": left_ok, "right": right_ok}.get(self.leg_side, False)
            if committed:
                # stay on the side already chosen for this obstacle; flip-flopping
                # between sides is how reactive planners get stuck on wide objects
                side = self.leg_side
            elif left_ok and right_ok:
                if abs(left - right) < 0.5 or (math.isinf(left) and math.isinf(right)):
                    side = "right" if goal_rel >= 0 else "left"
                else:
                    side = "left" if left > right else "right"
            else:
                side = "left" if left_ok else "right"
            # widen the side-step each time the same obstacle blocks us again
            # (4 m, 6 m, 8 m ...): wide objects are cleared in a few replans
            # instead of nibbling around their corner
            self.side_streak = self.side_streak + 1 if side == self.leg_side else 0
            self.leg_side = side
            lateral = self.lateral * (1.0 + 0.5 * self.side_streak)
            sign = -1.0 if side == "left" else 1.0
            # detour geometry in the body frame the sensors just measured in
            # (yaw-first guidance keeps the nose on the direction of travel)
            course = yaw
            d1n, d1e = body_to_ne(0.0, sign * lateral, course)
            fwd = min(self.forward, max(goal_dist - 1.0, 0.0))
            d2n, d2e = body_to_ne(fwd, sign * lateral, course)
            target_alt = goal.alt if goal.tag != "climb" else alt
            detour = [Waypoint(n + d1n, e + d1e, max(alt, target_alt), "detour")]
            if fwd > 1.0:
                detour.append(Waypoint(n + d2n, e + d2e, max(alt, target_alt), "detour"))
            self._last_replan = now
            self.replans_this_leg += 1
            return self._event(now, Decision(Action.REPLAN, detour=detour, front=front, side=side,
                                             reason=f"front {front:.2f} m; L {left:.1f} R {right:.1f}"))
        # both sides blocked -> try going over
        if can_climb:
            self._last_replan = now
            self.replans_this_leg += 1
            return self._event(now, Decision(Action.CLIMB, detour=[Waypoint(n, e, alt + self.climb_step, "climb")],
                                             front=front, reason="both sides blocked"))
        return self._event(now, Decision(Action.BLOCKED, front=front, reason="boxed in at ceiling"))

    def _event(self, now: float, d: Decision) -> Decision:
        self.events.append((now, d.action.value, round(d.front, 2), d.side, d.reason))
        return d


def body_goal_bearing(n, e, yaw, goal: Waypoint) -> float:
    f, r = ne_to_body(goal.n - n, goal.e - e, yaw)
    return math.degrees(math.atan2(r, f))
