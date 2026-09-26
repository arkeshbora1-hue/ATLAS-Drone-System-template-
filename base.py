"""Range-sensor interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from atlas.core.state import RangeReading, Sector


class RangeSensor(ABC):
    kind: str = "range"

    def __init__(self, name: str, sector: Sector, max_range: float, min_range: float = 0.03):
        self.name = name
        self.sector = Sector(sector)
        self.max_range = max_range
        self.min_range = min_range

    def open(self) -> None:
        """Initialise hardware. Called once before sampling."""

    def close(self) -> None:
        """Release hardware."""

    @abstractmethod
    def read(self, now: float) -> RangeReading:
        """Take (or fetch) one measurement.

        Conventions:
          * no return within range   -> distance=inf, valid=True (path clear)
          * hardware/driver failure  -> valid=False
        """

    def _ok(self, now: float, d: float) -> RangeReading:
        if d < self.min_range:
            return RangeReading(self.name, self.sector, d, False, now)
        if d > self.max_range:
            d = float("inf")
        return RangeReading(self.name, self.sector, d, True, now)

    def _fail(self, now: float) -> RangeReading:
        return RangeReading(self.name, self.sector, float("nan"), False, now)
