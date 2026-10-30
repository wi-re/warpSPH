"""Known-answer body-force controls, separate from stochastic turbulence driving.

The oscillating transverse shear solves forced compressible Euler exactly:
constant rho/P/u, v_j=A sin(omega*t) sin(k*x_i), i != j. Both advection and
dilatation vanish. The only energy source is rho*v dot a. This is an analytic
verification control, not a viscous steady Kolmogorov-flow model.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class OscillatingShear:
    amplitude: float = .2
    period: float = 2.
    length: float = 1.
    mode: int = 1
    dim: int = 2
    coordinate: int = 0
    component: int = 1

    def __post_init__(self):
        if (not math.isfinite(self.amplitude) or self.amplitude < 0 or
                any(not math.isfinite(v) or v <= 0 for v in (self.period, self.length))):
            raise ValueError('need finite nonnegative amplitude and positive period/length')
        if (type(self.mode) is not int or self.mode < 1 or self.dim not in (2, 3) or
                self.coordinate not in range(self.dim) or self.component not in range(self.dim) or
                self.coordinate == self.component):
            raise ValueError('need a positive integer mode and distinct valid transverse axes in 2D/3D')

    @property
    def omega(self):
        return 2*math.pi/self.period

    def _shape(self, positions):
        if positions.ndim != 2 or positions.shape[1] != self.dim:
            raise ValueError('positions must be N x dim')
        return torch.sin((2*math.pi*self.mode/self.length)*positions[:, self.coordinate])

    def _vector(self, positions, coefficient):
        out = torch.zeros_like(positions)
        out[:, self.component] = coefficient*self._shape(positions)
        return out

    def __call__(self, positions, time):
        return self._vector(positions, self.amplitude*self.omega*math.cos(self.omega*float(time)))

    def velocity(self, positions, time):
        return self._vector(positions, self.amplitude*math.sin(self.omega*float(time)))

    def displacement(self, initialPositions, time):
        return self._vector(initialPositions, self.amplitude*(1-math.cos(self.omega*float(time)))/self.omega)

    def accelerationBound(self, time):
        return self.amplitude*self.omega

    def kineticEnergy(self, time, mass=1.):
        """Continuum/lattice integral for uniform sampling of a complete mode."""
        return mass*self.amplitude**2*math.sin(self.omega*float(time))**2/4

    def inputPower(self, time, mass=1.):
        return mass*self.amplitude**2*self.omega*math.sin(2*self.omega*float(time))/4
