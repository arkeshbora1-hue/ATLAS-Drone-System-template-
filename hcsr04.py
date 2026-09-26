"""HC-SR04 ultrasonic ranger via pigpio.

Wiring: TRIG direct to a Pi GPIO; ECHO is a 5 V signal and MUST go through
a 1k/2k divider (or level shifter) to the Pi's 3.3 V input.

pigpio timestamps edges in the daemon with microsecond resolution, which
avoids the millisecond jitter of Python-side polling (1 ms ~ 17 cm).
Sensors are fired one at a time by ``SensorArray`` so their pings do not
cross-talk; each read blocks for at most ``timeout`` (≈ 2 x max range / c).
"""
from __future__ import annotations

import threading

from atlas.core.state import RangeReading
from atlas.sensors.base import RangeSensor

try:
    import pigpio  # type: ignore
except ImportError:  # pragma: no cover
    pigpio = None


def speed_of_sound(temp_c: float) -> float:
    return 331.3 + 0.606 * temp_c


class HCSR04(RangeSensor):
    kind = "ultrasonic"

    def __init__(self, name, sector, trig_gpio: int, echo_gpio: int,
                 max_range: float = 4.0, min_range: float = 0.03,
                 temperature_c: float = 25.0, pi=None):
        super().__init__(name, sector, max_range, min_range)
        self.trig, self.echo = trig_gpio, echo_gpio
        self.c = speed_of_sound(temperature_c)
        self.timeout = 2.0 * max_range / self.c + 0.002
        self._pi = pi
        self._rise = None
        self._width_us = None
        self._done = threading.Event()
        self._cb = None

    def open(self) -> None:
        if pigpio is None:
            raise RuntimeError("pigpio not installed (sudo apt install pigpio python3-pigpio; sudo systemctl enable --now pigpiod)")
        if self._pi is None:
            self._pi = pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError("pigpiod not running")
        self._pi.set_mode(self.trig, pigpio.OUTPUT)
        self._pi.set_mode(self.echo, pigpio.INPUT)
        self._pi.write(self.trig, 0)
        self._cb = self._pi.callback(self.echo, pigpio.EITHER_EDGE, self._edge)

    def close(self) -> None:
        if self._cb:
            self._cb.cancel()

    def _edge(self, gpio, level, tick):
        if level == 1:
            self._rise = tick
        elif level == 0 and self._rise is not None:
            self._width_us = pigpio.tickDiff(self._rise, tick)
            self._done.set()

    def read(self, now: float) -> RangeReading:
        try:
            self._done.clear()
            self._rise, self._width_us = None, None
            self._pi.gpio_trigger(self.trig, 10, 1)  # 10 us pulse
            if not self._done.wait(self.timeout):
                # no echo inside the window -> nothing within max_range
                return RangeReading(self.name, self.sector, float("inf"), True, now)
            d = (self._width_us * 1e-6) * self.c / 2.0
            return self._ok(now, d)
        except Exception:
            return self._fail(now)
