"""Flight data recorder: one CSV row per executive tick plus an event log."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

FIELDS = [
    "t", "state", "mode", "armed", "n", "e", "alt", "yaw", "vn", "ve", "vd",
    "battery_pct", "sats", "hdop",
    "r_front", "r_front_left", "r_front_right", "r_left", "r_right", "r_down",
    "avoid", "tgt_n", "tgt_e", "tgt_confirmed", "cmd", "cmd_vn", "cmd_ve", "cmd_vd", "cmd_yaw",
    "safety",
]


class Blackbox:
    def __init__(self, directory: str | Path, run_name: str | None = None, enabled: bool = True):
        self.enabled = enabled
        self.rows = 0
        if not enabled:
            return
        run = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(directory) / run
        self.dir.mkdir(parents=True, exist_ok=True)
        self._csv_fh = open(self.dir / "flight.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_fh, fieldnames=FIELDS, extrasaction="ignore")
        self._csv.writeheader()
        self._ev = open(self.dir / "events.jsonl", "w", encoding="utf-8")

    def row(self, **kw) -> None:
        if not self.enabled:
            return
        out = {}
        for k, v in kw.items():
            out[k] = round(v, 3) if isinstance(v, float) else v
        self._csv.writerow(out)
        self.rows += 1
        if self.rows % 50 == 0:
            self._csv_fh.flush()

    def event(self, t: float, kind: str, **data) -> None:
        if not self.enabled:
            return
        self._ev.write(json.dumps({"t": round(t, 3), "event": kind, **data}, default=str) + "\n")
        self._ev.flush()

    def close(self) -> None:
        if self.enabled:
            self._csv_fh.close()
            self._ev.close()
