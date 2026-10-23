"""Pressure-volume closure shared by pressure-energy and pressure-entropy SPH.

Hopkins (2013), Eqs. 18-22: the support constraint uses number density,
whereas the thermodynamic volume is w_i / y_i. Neither is the ordinary
mass-density sum. This module has no integration or dissipation policy.
"""

from dataclasses import dataclass

import torch

from ...enumTypes import PESPHVariant
from ..eos.idealGas import idealGasEOS


def requirePositive(value, name):
    """Reject invalid fractional-power/division inputs without a CUDA host read."""
    torch._assert_async(torch.all(torch.isfinite(value) & (value > 0)),
                        f'PESPH requires finite positive {name}')


def pressureWeights(state, variant, gamma):
    """Weights from the authoritative E or A, never a cached derived u."""
    requirePositive(state.masses, 'masses')
    if variant is PESPHVariant.PressureEntropy:
        requirePositive(state.entropies, 'entropic function A')
        return state.masses * state.entropies.pow(1.0 / gamma)
    if variant is PESPHVariant.PressureEnergy:
        u = state.totalEnergies / state.masses - 0.5 * state.velocities.square().sum(-1)
        requirePositive(u, 'specific internal energy')
        return (gamma - 1.0) * state.masses * u
    raise ValueError(f'Unknown PESPH variant {variant!r}')


@dataclass(frozen=True)
class PressureSums:
    numberDensity: torch.Tensor
    weightedSum: torch.Tensor
    numberSupportDerivative: torch.Tensor
    weightedSupportDerivative: torch.Tensor
    massDensity: torch.Tensor


@dataclass(frozen=True)
class PressureState:
    """O(N) stage-local data; gradH is b_i, with fH_ij = 1 - b_i/w_j."""
    weights: torch.Tensor
    sums: PressureSums
    pressures: torch.Tensor
    pressureFactors: torch.Tensor
    gradH: torch.Tensor
    thermodynamicDensity: torch.Tensor
    internalEnergies: torch.Tensor
    soundspeeds: torch.Tensor


def closePressure(state, sums, weights, variant, gamma, *, adaptive=True):
    """Derive thermodynamics and the number-density grad-h coefficient.

    Differentiate y=P for energy and y=P**(1/gamma) for entropy. The leading
    1 in fH is outside Omega's denominator. Mass density stays an independent
    output for diagnostics and operators that require that estimate.
    """
    n, y = sums.numberDensity, sums.weightedSum
    requirePositive(n, 'number density')
    requirePositive(y, 'weighted pressure sum')
    if variant is PESPHVariant.PressureEntropy:
        pressure = y.pow(gamma)
    elif variant is PESPHVariant.PressureEnergy:
        pressure = y
    else:
        raise ValueError(f'Unknown PESPH variant {variant!r}')
    requirePositive(pressure, 'pressure')
    volume = weights / y
    rho = state.masses / volume
    requirePositive(rho, 'thermodynamic density')
    # Reuse the existing EOS at the pressure-volume density. The explicit
    # positivity checks above prevent its abs(P) sound-speed fallback from
    # hiding invalid PESPH thermodynamics.
    _, u, _, sound = idealGasEOS(A=None, u=None, P=pressure, rho=rho, gamma=gamma)
    if adaptive:
        hFactor = state.supports / (state.positions.shape[1] * n)
        omega = 1.0 + hFactor * sums.numberSupportDerivative
        requirePositive(omega, 'number-density Omega')
        gradH = hFactor * sums.weightedSupportDerivative / omega
    else:
        gradH = torch.zeros_like(y)
    return PressureState(weights, sums, pressure, pressure / y.square(), gradH,
                         rho, u, sound)
