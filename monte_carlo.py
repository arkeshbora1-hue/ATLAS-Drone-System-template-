"""Monte-Carlo robustness sweep.

Randomises obstacle layout along the route, wind, GPS noise, sensor noise
spikes/dropouts, detector miss rate and the true marker position, then runs
full missions and reports delivery rate, collisions, clearance and release
accuracy.

    python -m sim.monte_carlo --runs 50
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas.comms.sim_link import SimParams  # noqa: E402
from atlas.core.config import load_config  # noqa: E402
from atlas.core.geo import ll_to_ne  # noqa: E402
from atlas.navigation.mission import MissionSpec  # noqa: E402
from sim.harness import run_mission  # noqa: E402
from sim.world import Box, Cylinder, World  # noqa: E402


def random_world(spec: MissionSpec, rng: random.Random) -> World:
    pts = [(0.0, 0.0)] + [ll_to_ne(spec.home, w) for w in spec.waypoints] + [ll_to_ne(spec.home, spec.dropzone)]
    obs = []
    for (n0, e0), (n1, e1) in zip(pts[:-2], pts[1:-1]):        # transit legs only
        L = math.hypot(n1 - n0, e1 - e0)
        un, ue = (n1 - n0) / L, (e1 - e0) / L
        for _ in range(rng.randint(1, 3)):
            s = rng.uniform(15, L - 8)
            off = rng.uniform(-3, 3)
            cn, ce = n0 + un * s - ue * off, e0 + ue * s + un * off
            if rng.random() < 0.7:
                obs.append(Cylinder(cn, ce, rng.uniform(0.3, 2.0), rng.uniform(22, 40)))
            else:
                hw = rng.uniform(1.5, 4.0)
                obs.append(Box(cn - hw, ce - hw, cn + hw, ce + hw, rng.uniform(22, 40)))
    dn, de = pts[-1]
    a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, 8)
    return World(obs, (dn + r * math.cos(a), de + r * math.sin(a)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    spec = MissionSpec.load(ROOT / "sim/missions/demo.yaml")
    rng = random.Random(a.seed)
    results = []
    for i in range(a.runs):
        world = random_world(spec, rng)
        params = SimParams(wind_n=rng.uniform(-4, 4), wind_e=rng.uniform(-4, 4),
                           gps_noise=rng.uniform(0.0, 0.5), seed=a.seed + i)
        det = {"miss_rate": rng.uniform(0.05, 0.3), "fp_rate": rng.uniform(0.0, 0.03)}
        r = run_mission(cfg, spec, world, sim_params=params, seed=a.seed + i, detector_kw=det)
        fs = [d["action"] for _, k, d in r.events if k == "failsafe"]
        results.append({"run": i, "phase": r.phase, "delivered": r.delivered, "collided": r.collided,
                        "min_clearance": round(r.min_clearance, 2), "replans": r.replans,
                        "release_error": None if r.release_error is None else round(r.release_error, 2),
                        "failsafes": fs, "duration": round(r.duration, 1), "obstacles": len(world.obstacles)})
        print(f"run {i:3d}: {r.phase:8s} delivered={r.delivered!s:5s} collided={r.collided!s:5s} "
              f"clear={r.min_clearance:5.2f} replans={r.replans} err={results[-1]['release_error']} fs={fs}")
    n = len(results)
    delivered = [x for x in results if x["delivered"]]
    errs = sorted(x["release_error"] for x in delivered)
    clear = sorted(x["min_clearance"] for x in results if x["min_clearance"] != math.inf)
    summary = {
        "runs": n,
        "delivered_pct": 100 * len(delivered) / n,
        "collisions": sum(x["collided"] for x in results),
        "safe_aborts": sum(1 for x in results if not x["delivered"] and not x["collided"]),
        "release_error_median": statistics.median(errs) if errs else None,
        "release_error_p95": errs[min(len(errs) - 1, int(0.95 * len(errs)))] if errs else None,
        "min_clearance_min": clear[0] if clear else None,
        "min_clearance_median": statistics.median(clear) if clear else None,
        "mean_replans": statistics.mean(x["replans"] for x in results),
    }
    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"  {k:22s} {v:.2f}" if isinstance(v, float) else f"  {k:22s} {v}")
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": summary, "runs": results}, indent=1))


if __name__ == "__main__":
    main()
