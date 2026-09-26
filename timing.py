"""Clocks and fixed-rate loops.

Every time-dependent module takes a ``Clock`` so the full stack can run
faster than real time in the simulator and deterministically in tests.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class Clock:
    """Wall clock (monotonic)."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimClock(Clock):
    """Manually advanced clock for simulation and tests."""

    def __init__(self, start: float = 0.0):
        self._t = start
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def advance(self, dt: float) -> None:
        with self._lock:
            self._t += dt

    def sleep(self, seconds: float) -> None:  # pragma: no cover - sim is stepped explicitly
        self.advance(seconds)


@dataclass
class LoopStats:
    iterations: int = 0
    overruns: int = 0
    max_exec: float = 0.0
    total_exec: float = 0.0

    @property
    def mean_exec(self) -> float:
        return self.total_exec / self.iterations if self.iterations else 0.0


@dataclass
class RateLoop:
    """Runs ``fn`` at a fixed rate on a background thread and records overruns.

    Uses absolute deadlines (next += period) so jitter does not accumulate
    into drift; if an iteration overruns by more than a full period the
    schedule is re-anchored instead of trying to catch up in a burst.
    """

    name: str
    hz: float
    fn: callable
    clock: Clock = field(default_factory=Clock)
    stats: LoopStats = field(default_factory=LoopStats)

    def __post_init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def period(self) -> float:
        return 1.0 / self.hz

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def _run(self) -> None:
        nxt = self.clock.now()
        while not self._stop.is_set():
            t0 = self.clock.now()
            self.fn()
            dt = self.clock.now() - t0
            s = self.stats
            s.iterations += 1
            s.total_exec += dt
            s.max_exec = max(s.max_exec, dt)
            nxt += self.period
            slack = nxt - self.clock.now()
            if slack < 0:
                s.overruns += 1
                if slack < -self.period:
                    nxt = self.clock.now()
                continue
            self._stop.wait(slack)
