"""IISPH policy, independent of the composed DFSPH/shifting solvers."""

from dataclasses import dataclass
import math


@dataclass
class IISPHConfig:
    minIterations: int = 2
    maxIterations: int = 100
    maxErrorPercent: float = 0.01
    relaxationFactor: float = 0.5
    # auto: analytic if a provider is present, otherwise sampled particles.
    boundaryRepresentation: str = "auto"
    # self: static boundary with no pressure unknown of its own.
    # contactShepard: interpolate fluid pressure at the closest solid contact.
    boundaryPressure: str = "self"
    contactGravity: bool = True

    def validate(self):
        if not (isinstance(self.minIterations, int) and isinstance(self.maxIterations, int)
                and 2 <= self.minIterations <= self.maxIterations):
            raise ValueError("IISPH requires 2 <= minIterations <= maxIterations")
        if not math.isfinite(self.maxErrorPercent) or self.maxErrorPercent < 0:
            raise ValueError("IISPH maxErrorPercent must be finite and nonnegative")
        if not math.isfinite(self.relaxationFactor) or not 0 < self.relaxationFactor <= 1:
            raise ValueError("IISPH relaxationFactor must be in (0, 1]")
        if self.boundaryRepresentation not in ("auto", "analytic", "particles"):
            raise ValueError("IISPH boundaryRepresentation must be auto, analytic or particles")
        if self.boundaryPressure not in ("self", "contactShepard"):
            raise ValueError("IISPH boundaryPressure must be self or contactShepard")
