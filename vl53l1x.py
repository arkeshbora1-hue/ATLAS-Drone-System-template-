"""VL53L1X time-of-flight ranger (I2C).

All four VL53L1X boot at address 0x29, so each has its XSHUT pin wired to a
GPIO. At start-up every sensor is held in reset, then released one by one
and moved to its own address (0x30..0x33). Ranging runs continuously on the
sensor (long mode, 50 ms timing budget), so ``read`` just fetches the latest
result without blocking the 10 Hz sampling cycle.
"""
from __future__ import annotations

import time

from atlas.core.state import RangeReading
from atlas.sensors.base import RangeSensor

try:
    import VL53L1X as _vl  # pimoroni "vl53l1x" package
except ImportError:  # pragma: no cover
    _vl = None

try:
    import pigpio  # type: ignore
except ImportError:  # pragma: no cover
    pigpio = None

DEFAULT_ADDRESS = 0x29
LONG_RANGE = 3
TIMING_BUDGET_US = 50_000
INTER_MEASUREMENT_MS = 60


class VL53L1XSensor(RangeSensor):
    kind = "tof"

    def __init__(self, name, sector, xshut_gpio: int, address: int,
                 max_range: float = 4.0, min_range: float = 0.03, bus: int = 1):
        super().__init__(name, sector, max_range, min_range)
        self.xshut, self.address, self.bus = xshut_gpio, address, bus
        self._dev = None

    def open(self) -> None:
        if _vl is None:
            raise RuntimeError("VL53L1X driver not installed: pip install vl53l1x")
        dev = _vl.VL53L1X(i2c_bus=self.bus, i2c_address=self.address)
        dev.open()
        dev.set_timing(TIMING_BUDGET_US, INTER_MEASUREMENT_MS)
        dev.start_ranging(LONG_RANGE)
        self._dev = dev

    def close(self) -> None:
        if self._dev:
            try:
                self._dev.stop_ranging()
                self._dev.close()
            except Exception:
                pass

    def read(self, now: float) -> RangeReading:
        try:
            mm = self._dev.get_distance()
        except Exception:
            return self._fail(now)
        if mm <= 0:  # 0 = no valid target / signal fail (e.g. strong sunlight)
            return RangeReading(self.name, self.sector, float("inf"), True, now)
        return self._ok(now, mm / 1000.0)


def assign_addresses(sensors: list[VL53L1XSensor], pi=None) -> None:
    """XSHUT boot sequence: move each sensor from 0x29 to its configured address."""
    if pigpio is None or _vl is None:
        raise RuntimeError("pigpio and vl53l1x are required on the vehicle")
    pi = pi or pigpio.pi()
    for s in sensors:                        # hold every sensor in reset
        pi.set_mode(s.xshut, pigpio.OUTPUT)
        pi.write(s.xshut, 0)
    time.sleep(0.01)
    for s in sensors:                        # wake one at a time and re-address
        pi.write(s.xshut, 1)
        time.sleep(0.01)
        dev = _vl.VL53L1X(i2c_bus=s.bus, i2c_address=DEFAULT_ADDRESS)
        dev.open()
        dev.change_address(s.address)
        dev.close()
