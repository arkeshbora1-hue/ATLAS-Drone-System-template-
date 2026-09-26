"""Run a simulated ATLAS delivery and plot it.

    python -m sim.run_sim                                  # nominal demo
    python -m sim.run_sim --fault low_battery              # inject a fault
    python -m sim.run_sim --plot out.png --log logs/

Faults: low_battery, gps_loss, front_sensor_fail, no_marker, wind
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas.comms.sim_link import SimParams  # noqa: E402
from atlas.core.config import load_config  # noqa: E402
from atlas.navigation.mission import MissionSpec  # noqa: E402
from sim.harness import Fault, run_mission  # noqa: E402
from sim.world import World  # noqa: E402


def _fail_front(ctx):
    for s in ctx.array.sensors:
        if s.sector.value == "FRONT":
            s.failed = True


FAULTS = {
    "low_battery": [Fault(60.0, lambda c: setattr(c.link, "battery", 29.0), "battery -> 29%")],
    "gps_loss": [Fault(45.0, lambda c: setattr(c.link, "gps_fix", 1), "GPS fix lost")],
    "front_sensor_fail": [Fault(30.0, _fail_front, "FRONT ultrasonic + ToF failed")],
    "no_marker": [],
    "wind": [],
}


def plot(result, world, spec, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    tr = result.trace
    t = [r[0] for r in tr]
    ns = [r[1] for r in tr]
    es = [r[2] for r in tr]
    alts = [r[3] for r in tr]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.1, 1]})
    for ob in world.obstacles:
        col = "#b94a48" if ob.h > cfg.vehicle.cruise_alt else "#d9a441"
        if hasattr(ob, "r"):
            ax.add_patch(Circle((ob.e, ob.n), ob.r, color=col, alpha=0.8))
        else:
            ax.add_patch(Rectangle((min(ob.e0, ob.e1), min(ob.n0, ob.n1)), abs(ob.e1 - ob.e0),
                                   abs(ob.n1 - ob.n0), color=col, alpha=0.8))
    ax.plot(es, ns, color="#1f5fa8", lw=1.6, label="flown path")
    ex = result.ctx.executive
    from atlas.core.geo import ll_to_ne
    plan = [(0, 0)] + [ll_to_ne(spec.home, w) for w in spec.waypoints] + [ll_to_ne(spec.home, spec.dropzone)]
    ax.plot([p[1] for p in plan], [p[0] for p in plan], "--", color="grey", lw=1, label="planned route")
    ax.plot(plan[-1][1], plan[-1][0], "x", color="grey", ms=10, label="mission drop-zone estimate")
    if world.dropzone:
        ax.plot(world.dropzone[1], world.dropzone[0], "s", color="#2e8b57", ms=9, label="true marker")
    for _, kind, d in ex.events:
        if kind == "replan":
            ax.plot(d["detour"][0][1], d["detour"][0][0], "^", color="#e67e22", ms=7)
        if kind == "payload_released":
            ax.plot(d["e"], d["n"], "*", color="#8e44ad", ms=14, label="release point")
    ax.plot([], [], "^", color="#e67e22", label="detour waypoint (replan)")
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title("ATLAS simulated delivery: top view")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ax2.plot(t, alts, color="#1f5fa8")
    ymax = max(alts) + 4
    for tt, kind, d in ex.events:
        if kind == "phase":
            ax2.axvline(tt, color="grey", lw=0.6, alpha=0.6)
            ax2.text(tt, ymax, d["to"], rotation=90, fontsize=7, va="top", ha="right")
    ax2.set_ylim(0, ymax + 1)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("altitude above home (m)")
    ax2.set_title("Altitude and mission phases")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mission", default=str(ROOT / "sim/missions/demo.yaml"))
    ap.add_argument("--world", default=str(ROOT / "sim/worlds/demo.yaml"))
    ap.add_argument("--fault", choices=sorted(FAULTS), default=None)
    ap.add_argument("--plot", default=None, help="save a trajectory plot (PNG)")
    ap.add_argument("--log", default=None, help="directory for blackbox CSV/JSONL")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    spec = MissionSpec.load(a.mission)
    world = World.load(a.world)
    params = SimParams(seed=a.seed)
    if a.fault == "no_marker":
        world.dropzone = None
    if a.fault == "wind":
        params.wind_n, params.wind_e = 4.0, -3.0
        params.gps_noise = 0.4
    res = run_mission(cfg, spec, world, FAULTS.get(a.fault, []), sim_params=params, log_dir=a.log, seed=a.seed)
    print(res.summary())
    print("\nevents:")
    for t, kind, d in res.events:
        if kind in ("phase", "replan", "avoid_climb", "failsafe", "payload_released", "fault_injected",
                    "target_lost", "dropzone_not_found", "preflight_fail"):
            print(f"  {t:7.1f}s  {kind:18s} {d}")
    if a.plot:
        plot(res, world, spec, cfg, a.plot)
        print(f"\nplot saved to {a.plot}")
    return 0 if not res.collided else 1


if __name__ == "__main__":
    sys.exit(main())
