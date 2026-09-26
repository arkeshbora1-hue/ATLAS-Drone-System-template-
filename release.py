"""Payload release mechanism.

Default: a hobby servo on flight-controller output ``servo_channel``
(SERVOx_FUNCTION = 0, i.e. passthrough-for-commands) driven with
MAV_CMD_DO_SET_SERVO. Routing through the FC means the servo keeps a valid
PWM signal even if the Pi reboots. A direct-GPIO variant is included for
bench testing.
"""
from __future__ import annotations

import logging
from enum import Enum

log = logging.getLogger("atlas.payload")


class ReleaseState(str, Enum):
    LOCKED = "LOCKED"
    OPEN = "OPEN"
    DONE = "DONE"


class PayloadRelease:
    def __init__(self, cfg, link, clock):
        p = cfg.payload
        self.link, self.clock = link, clock
        self.method = p.method
        self.channel = p.servo_channel
        self.pwm_closed, self.pwm_open = p.pwm_closed, p.pwm_open
        self.open_time = p.open_time
        self.gpio = getattr(p, "gpio", None) if hasattr(p, "gpio") else None
        self.state = ReleaseState.LOCKED
        self._opened_at = 0.0
        self.released_at: float | None = None

    def _write(self, pwm: int) -> bool:
        if self.method == "fc_servo":
            return self.link.set_servo(self.channel, pwm)
        import pigpio  # pragma: no cover - bench only
        pi = pigpio.pi()
        pi.set_servo_pulsewidth(self.gpio, pwm)
        return True

    def arm_lock(self) -> bool:
        """Drive the latch closed before flight."""
        self.state = ReleaseState.LOCKED
        return self._write(self.pwm_closed)

    def release(self) -> bool:
        if self.state != ReleaseState.LOCKED:
            return False
        ok = self._write(self.pwm_open)
        if ok:
            self.state = ReleaseState.OPEN
            self._opened_at = self.released_at = self.clock.now()
            log.info("payload released")
        return ok

    def update(self) -> ReleaseState:
        """Close the latch again after ``open_time``; returns current state."""
        if self.state == ReleaseState.OPEN and self.clock.now() - self._opened_at >= self.open_time:
            self._write(self.pwm_closed)
            self.state = ReleaseState.DONE
        return self.state
