"""Closed-loop software-in-the-loop harness.

Runs the *real* onboard stack (Executive, SensorArray fusion, ReactiveAvoider,
PerceptionPipeline + TargetTracker, SafetyMonitor, PayloadRelease) against
the kinematic SimLink, ray-cast range sensors and a synthetic detector, on a
SimClock so a 4-minute mission completes in a couple of seconds.

Scheduling mirrors the vehicle:
    physics      50 Hz
    sensors      10 Hz  -> range slot
    executive    10 Hz
    perception   15 Hz  -> target slot
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from atlas.comms.sim_link import SimLink, SimParams
from atlas.core.state import Latest
from atlas.core.timing import SimClock
from atlas.mission.executive import Executive, Phase
from atlas.navigation.mission import MissionSpec
from atlas.perception.pipeline import PerceptionPipeline
from atlas.perception.sim_detector import SimDropZoneDetector
from atlas.sensors.sim import build_sim_array
from atlas.telemetry.blackbox import Blackbox
from sim.world import World


@dataclass
class Fault:
    at: float
    apply: Callable[["SimContext"], None]
    label: str = ""


@dataclass
class SimContext:
    clock: SimClock
    link: SimLink
    world: World
    array: object
    detector: SimDropZoneDetector
    executive: Executive
    pipeline: PerceptionPipeline


@dataclass
class SimResult:
    phase: str
    delivered: bool
    duration: float
    collided: bool
    min_clearance: float
    release_error: Optional[float]
    replans: int
    events: list
    trace: list
    detections_fps: float
    battery_end: float
    ctx: SimContext = field(repr=False, default=None)

    def summary(self) -> str:
        err = f"{self.release_error:.2f} m" if self.release_error is not None else "n/a"
        return (f"final phase     : {self.phase}\n"
                f"delivered       : {self.delivered}\n"
                f"mission time    : {self.duration:.1f} s\n"
                f"collision       : {self.collided}\n"
                f"min clearance   : {self.min_clearance:.2f} m\n"
                f"avoidance replans: {self.replans}\n"
                f"release error   : {err} from true marker\n"
                f"perception rate : {self.detections_fps:.1f} fps\n"
                f"battery at end  : {self.battery_end:.0f}%")


def _clearance(world: World, n: float, e: float, alt: float) -> float:
    best = math.inf
    for ob in world.obstacles:
        if alt >= ob.h:
            continue
        if hasattr(ob, "r"):
            best = min(best, math.hypot(n - ob.n, e - ob.e) - ob.r)
        else:
            dn = max(min(ob.n0, ob.n1) - n, 0, n - max(ob.n0, ob.n1))
            de = max(min(ob.e0, ob.e1) - e, 0, e - max(ob.e0, ob.e1))
            best = min(best, math.hypot(dn, de))
    return best


def run_mission(cfg, spec: MissionSpec, world: World, faults: list[Fault] | None = None,
                max_time: float = 900.0, sim_params: SimParams | None = None,
                log_dir: str | None = None, seed: int = 0, detector_kw: dict | None = None) -> SimResult:
    clock = SimClock()
    link = SimLink(spec.home, clock, sim_params or SimParams(seed=seed))
    array = build_sim_array(cfg, world, link, seed=seed)
    cam = cfg.perception.camera
    detector = SimDropZoneDetector(world, link, cam.hfov_deg, cam.vfov_deg, seed=seed + 7, **(detector_kw or {}))
    vehicle_slot, range_slot, target_slot = Latest(), Latest(), Latest()
    pipeline = PerceptionPipeline(cfg, detector, lambda: (None, clock.now()), link.home,
                                  vehicle_slot, range_slot, target_slot, clock)
    bb = Blackbox(log_dir, run_name="sim", enabled=log_dir is not None) if log_dir else None
    ex = Executive(cfg, link, spec, clock, range_slot, target_slot, bb, pipeline, temp_fn=lambda: 55.0)
    ctx = SimContext(clock, link, world, array, detector, ex, pipeline)
    pending = sorted(faults or [], key=lambda f: f.at)

    dt = 0.02
    ex_period = 1.0 / cfg.loop.executive_hz
    se_period = 1.0 / cfg.loop.sensor_hz
    pe_period = 1.0 / cfg.loop.perception_fps
    next_ex = next_se = next_pe = 0.0
    collided, min_clear = False, math.inf
    ex.start()
    while clock.now() < max_time and not ex.finished:
        now = clock.now()
        while pending and pending[0].at <= now:
            f = pending.pop(0)
            ex._event("fault_injected", label=f.label)
            f.apply(ctx)
        vehicle_slot.put(link.state())
        if now >= next_se:
            range_slot.put(array.sample(now))
            next_se += se_period
        if now >= next_pe:
            pipeline.step()
            next_pe += pe_period
        if now >= next_ex:
            ex.step()
            next_ex += ex_period
        link.step(dt)
        clock.advance(dt)
        if link.armed and link.alt > 0.5:
            c = _clearance(world, link.n, link.e, link.alt)
            min_clear = min(min_clear, c)
            if world.collides(link.n, link.e, link.alt):
                collided = True
    release_err = None
    for t, kind, data in ex.events:
        if kind == "payload_released" and world.dropzone:
            release_err = math.hypot(data["n"] - world.dropzone[0], data["e"] - world.dropzone[1])
    if bb:
        bb.close()
    replans = sum(1 for _, k, _ in ex.events if k in ("replan", "avoid_climb"))
    return SimResult(ex.phase.value, ex.delivered and ex.phase == Phase.DONE, clock.now(), collided,
                     min_clear, release_err, replans, ex.events, link.trace, pipeline.fps(),
                     link.battery, ctx)
