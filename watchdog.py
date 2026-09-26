"""Software watchdog + systemd notify.

Three independent layers keep the vehicle safe if the onboard software stalls:
  1. ArduPilot GUIDED timeout: no velocity setpoint for ~3 s -> vehicle stops.
  2. This watchdog: executive tick older than ``timeout`` -> command LOITER
     directly over MAVLink from its own thread (position hold, pilot can
     take over, FC failsafes still active).
  3. systemd WatchdogSec: if the whole process hangs, systemd kills and
     restarts it; the FC is already holding position from (1)/(2).
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time

log = logging.getLogger("atlas.watchdog")


def sd_notify(msg: str) -> None:
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg.encode())
    except OSError:
        pass


class Watchdog:
    def __init__(self, link, timeout: float = 1.5, airborne_fn=lambda: True):
        self.link = link
        self.timeout = timeout
        self.airborne_fn = airborne_fn
        self._kick = time.monotonic()
        self._tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)

    def kick(self) -> None:
        self._kick = time.monotonic()
        if self._tripped:
            log.warning("executive recovered; watchdog re-armed (vehicle stays in LOITER)")
            self._tripped = False
        sd_notify("WATCHDOG=1")

    def start(self) -> None:
        sd_notify("READY=1")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            stale = time.monotonic() - self._kick
            if stale > self.timeout and not self._tripped and self.airborne_fn():
                self._tripped = True
                log.error("executive stalled for %.1f s -> LOITER", stale)
                try:
                    self.link.set_mode("LOITER")
                except Exception as exc:  # pragma: no cover
                    log.error("watchdog could not command LOITER: %s", exc)
