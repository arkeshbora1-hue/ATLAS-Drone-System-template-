"""ATLAS onboard application entry point.

    python -m atlas run  --mission missions/site_a.yaml            # on the vehicle
    python -m atlas sitl --mission sim/missions/demo.yaml \\
                         --connection tcp:127.0.0.1:5760           # ArduPilot SITL
    python -m atlas check                                          # bench self-test

Thread layout on the Raspberry Pi 4 (4 cores):
    mav-rx / mav-hb     MAVLink receive + 1 Hz heartbeat
    sensors  10 Hz      ultrasonic + ToF sampling and fusion
    perception 15 fps   camera -> INT8 MobileNetV2 (4 TFLite threads) -> tracker
    executive 10 Hz     safety -> state machine -> setpoint -> blackbox
    watchdog  4 Hz      stall detection -> LOITER
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from atlas.core.config import load_config
from atlas.core.geo import LatLon, ll_to_ne
from atlas.core.state import Latest
from atlas.core.timing import Clock, RateLoop
from atlas.mission.executive import AIRBORNE, Executive
from atlas.navigation.mission import MissionSpec
from atlas.perception.pipeline import PerceptionPipeline
from atlas.safety.watchdog import Watchdog
from atlas.telemetry.blackbox import Blackbox

log = logging.getLogger("atlas")


class VehicleView:
    """Adapter giving simulated sensors (SITL mode) the pose of the real/SITL vehicle."""

    def __init__(self, link):
        self.link = link

    def _pose(self):
        s, h = self.link.state(), self.link.home()
        if s is None or h is None:
            return 0.0, 0.0, 0.0, 0.0
        n, e = ll_to_ne(h, LatLon(s.lat, s.lon))
        return n, e, s.alt_rel, s.yaw

    n = property(lambda self: self._pose()[0])
    e = property(lambda self: self._pose()[1])
    alt = property(lambda self: self._pose()[2])
    yaw = property(lambda self: self._pose()[3])


def build(args, cfg):
    from atlas.comms.mavlink_link import MavlinkLink

    clock = Clock()
    if args.connection:
        cfg = cfg.with_overrides({"mavlink.connection": args.connection})
    link = MavlinkLink(cfg, clock)
    link.connect()
    spec = MissionSpec.load(args.mission)
    vehicle_slot, range_slot, target_slot = Latest(), Latest(), Latest()

    if args.cmd == "sitl":
        # real MAVLink autopilot (SITL), simulated sensors and detector
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from atlas.perception.sim_detector import SimDropZoneDetector
        from atlas.sensors.sim import build_sim_array
        from sim.world import World
        world = World.load(args.world)
        view = VehicleView(link)
        array = build_sim_array(cfg, world, view)
        cam = cfg.perception.camera
        detector = SimDropZoneDetector(world, view, cam.hfov_deg, cam.vfov_deg)
        frames = lambda: (None, clock.now())
        camera = None
    else:
        from atlas.perception.camera import Camera
        from atlas.perception.detector import TFLiteDropZoneDetector
        from atlas.sensors.array import build_hardware_array
        array = build_hardware_array(cfg)
        p = cfg.perception
        detector = TFLiteDropZoneDetector(p.model_path, p.input_size, p.num_threads, p.conf_threshold)
        camera = Camera(*p.camera.resolution)
        camera.open()
        frames = camera.latest

    array.open()
    pipeline = PerceptionPipeline(cfg, detector, frames, link.home, vehicle_slot, range_slot, target_slot, clock)
    bb = Blackbox(cfg.logging.directory)
    ex = Executive(cfg, link, spec, clock, range_slot, target_slot, bb, pipeline)
    return clock, link, array, pipeline, ex, vehicle_slot, range_slot, bb, camera


def run(args) -> int:
    cfg = load_config(args.config)
    clock, link, array, pipeline, ex, vehicle_slot, range_slot, bb, camera = build(args, cfg)
    wd = Watchdog(link, airborne_fn=lambda: ex.phase in AIRBORNE)

    def sense():
        range_slot.put(array.sample(clock.now()))

    def perceive():
        vehicle_slot.put(link.state())
        pipeline.step()

    def execute():
        ex.step()
        wd.kick()

    loops = [
        RateLoop("sensors", cfg.loop.sensor_hz, sense, clock),
        RateLoop("perception", cfg.loop.perception_fps, perceive, clock),
        RateLoop("executive", cfg.loop.executive_hz, execute, clock),
    ]
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    for lp in loops[:2]:
        lp.start()
    time.sleep(1.0)            # let sensors/perception fill their slots
    ex.start()
    loops[2].start()
    wd.start()
    last_report = 0.0
    try:
        while not stop["flag"] and not ex.finished:
            time.sleep(0.2)
            if time.monotonic() - last_report > 5.0:
                last_report = time.monotonic()
                s = loops[2].stats
                log.info("phase=%s perception=%.1f fps (%.0f ms) exec overruns=%d/%d",
                         ex.phase.value, pipeline.fps(), pipeline.mean_latency_ms(), s.overruns, s.iterations)
    finally:
        if stop["flag"] and ex.phase in AIRBORNE:
            log.warning("shutdown requested in flight -> RTL")
            link.rtl()
        for lp in loops:
            lp.stop()
        wd.stop()
        array.close()
        if camera:
            camera.close()
        bb.close()
        link.close()
    log.info("mission ended in %s (delivered=%s)", ex.phase.value, ex.delivered)
    return 0 if ex.delivered else 2


def check(args) -> int:
    """Bench self-test: sensors, camera + model FPS, FC link. Props OFF."""
    cfg = load_config(args.config)
    ok = True
    try:
        from atlas.sensors.array import build_hardware_array
        arr = build_hardware_array(cfg)
        arr.open()
        for _ in range(20):
            snap = arr.sample(time.monotonic())
            time.sleep(0.1)
        print("range sectors:", {k.value: round(v, 2) for k, v in snap.sectors.items()},
              "unhealthy:", sorted(s.value for s in snap.unhealthy))
        ok &= not snap.unhealthy
        arr.close()
    except Exception as exc:
        print("SENSORS FAIL:", exc)
        ok = False
    try:
        from atlas.perception.camera import Camera
        from atlas.perception.detector import TFLiteDropZoneDetector
        p = cfg.perception
        cam = Camera(*p.camera.resolution)
        cam.open()
        time.sleep(1.0)
        det = TFLiteDropZoneDetector(p.model_path, p.input_size, p.num_threads, p.conf_threshold)
        t0, n = time.perf_counter(), 60
        for _ in range(n):
            f, _ = cam.latest()
            d = det.detect(f, time.monotonic())
        fps = n / (time.perf_counter() - t0)
        print(f"perception: {fps:.1f} fps end-to-end, last latency {d.latency * 1000:.0f} ms")
        ok &= fps >= cfg.perception.get("min_fps", 15) * 0.95
        cam.close()
    except Exception as exc:
        print("PERCEPTION FAIL:", exc)
        ok = False
    try:
        from atlas.comms.mavlink_link import MavlinkLink
        link = MavlinkLink(cfg)
        link.connect(timeout=10)
        time.sleep(2.0)
        s = link.state()
        print(f"FC: mode={s.mode} armed={s.armed} gps_fix={s.gps_fix} sats={s.satellites} "
              f"batt={s.battery_v:.2f} V")
        link.close()
    except Exception as exc:
        print("MAVLINK FAIL:", exc)
        ok = False
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="atlas", description="ATLAS onboard software")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "sitl"):
        p = sub.add_parser(name)
        p.add_argument("--mission", required=True)
        p.add_argument("--config", default=None)
        p.add_argument("--connection", default=None, help="override MAVLink connection string")
        if name == "sitl":
            p.add_argument("--world", default="sim/worlds/demo.yaml")
    pc = sub.add_parser("check")
    pc.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "check":
        return check(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
