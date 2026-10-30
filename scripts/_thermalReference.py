"""Independent constant-property heat-equation oracles, not solver operators.

The medium is stationary with fixed density: rho*cv*dT/dt = kappa*d2T/dx2.
Energies and cumulative heat are per unit transverse area. Flux is positive in
+x, so heat entering a slab is the time integral of q(0)-q(L). No projection,
source, mechanical-work or artificial-conductivity terms are hidden here.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class IdealGasReference:
    """Explicit caloric units for a single ideal gas; reference use only."""
    density: float
    cv: float
    gamma: float
    conductivity: float

    def __post_init__(self):
        values = (self.density, self.cv, self.gamma, self.conductivity)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Material values must be finite')
        if self.density <= 0 or self.cv <= 0 or self.gamma <= 1 or self.conductivity < 0:
            raise ValueError('Require rho>0, cv>0, gamma>1 and conductivity>=0')

    @property
    def diffusivity(self):
        """Isochoric chi for the stationary fixed-density heat equation."""
        return self.conductivity / (self.density * self.cv)

    @property
    def cp(self):
        return self.gamma * self.cv

    @property
    def isobaric_diffusivity(self):
        """Alpha=kappa/(rho*cp), commonly used for convection."""
        return self.conductivity / (self.density * self.cp)

    @property
    def gas_constant(self):
        return (self.gamma - 1) * self.cv


@dataclass(frozen=True)
class HeatProfile:
    temperature: np.ndarray
    specific_internal_energy: np.ndarray
    heat_flux: np.ndarray
    energy_per_area: float
    cumulative_boundary_heat: float


def _inputs(x, t, length, amplitude, minimum_temperature):
    if not all(math.isfinite(v) for v in (t, length, amplitude, minimum_temperature)):
        raise ValueError('Reference parameters must be finite')
    if t < 0 or length <= 0 or minimum_temperature <= abs(amplitude):
        raise ValueError('Require t>=0, L>0 and positive temperature throughout the mode')
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > length):
        raise ValueError('Coordinates must be finite and in [0,L]')
    return x


def periodic_mode(x, t, *, length, mean, amplitude, material):
    """One periodic sine mode: T=mean+A*exp(-chi*k^2*t)*sin(k*x)."""
    x = _inputs(x, t, length, amplitude, mean)
    k = 2 * math.pi / length
    a = amplitude * math.exp(-material.diffusivity * k*k * t)
    temperature = mean + a * np.sin(k*x)
    flux = -material.conductivity * a * k * np.cos(k*x)
    energy = material.density * material.cv * length * mean
    return HeatProfile(temperature, material.cv*temperature, flux, energy, 0.)


def insulated_mode(x, t, *, length, mean, amplitude, material):
    """One cosine mode with zero normal heat flux at both ends."""
    x = _inputs(x, t, length, amplitude, mean)
    k = math.pi / length
    a = amplitude * math.exp(-material.diffusivity * k*k * t)
    temperature = mean + a * np.cos(k*x)
    flux = material.conductivity * a * k * np.sin(k*x)
    energy = material.density * material.cv * length * mean
    return HeatProfile(temperature, material.cv*temperature, flux, energy, 0.)


def plate_mode(x, t, *, length, left, right, amplitude, material):
    """Fixed-temperature plates plus one decaying homogeneous sine mode.

    The stationary linear profile carries nonzero through-flux for unequal
    plates, but zero net heat input. The perturbation supplies the transient
    signed boundary exchange. Positive amplitude releases heat to the plates.
    """
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError('Plate temperatures must be finite')
    x = _inputs(x, t, length, amplitude, min(left, right))
    k = math.pi / length
    exponent = -material.diffusivity * k*k * t
    a = amplitude * math.exp(exponent)
    slope = (right-left)/length
    temperature = left + slope*x + a*np.sin(k*x)
    flux = -material.conductivity * (slope + a*k*np.cos(k*x))
    capacity = material.density * material.cv
    energy = capacity * (length*(left+right)/2 + 2*a/k)
    heat = capacity * 2*amplitude/k * math.expm1(exponent)
    return HeatProfile(temperature, material.cv*temperature, flux, energy, heat)
