"""Flight-controller link interface.

The executive only talks to this interface. ``MavlinkLink`` implements it
against ArduPilot on the SpeedyBee F405; ``SimLink`` implements it with a
kinematic multirotor model for the simulator and tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from atlas.core.geo import LatLon
from atlas.core.state import VehicleState


class FlightLink(ABC):
    @abstractmethod
    def connect(self, timeout: float = 30.0) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def state(self) -> Optional[VehicleState]: ...

    @abstractmethod
    def home(self) -> Optional[LatLon]: ...

    @abstractmethod
    def set_mode(self, mode: str) -> bool: ...

    @abstractmethod
    def arm(self) -> bool: ...

    @abstractmethod
    def disarm(self) -> bool: ...

    @abstractmethod
    def takeoff(self, alt: float) -> bool: ...

    @abstractmethod
    def send_velocity(self, vn: float, ve: float, vd: float, yaw: Optional[float] = None) -> None:
        """NED velocity setpoint (m/s). ``yaw`` is absolute heading in degrees."""

    @abstractmethod
    def set_servo(self, channel: int, pwm: int) -> bool: ...

    def authorized(self) -> bool:
        """Operator authorisation to start the mission (pilot switch)."""
        return True

    def hold(self) -> None:
        self.send_velocity(0.0, 0.0, 0.0)

    def land(self) -> bool:
        return self.set_mode("LAND")

    def rtl(self) -> bool:
        return self.set_mode("RTL")
