"""Shared support/thermodynamic refresh for RHS stages and accepted states."""

import torch

from ...enumTypes import PESPHVariant, AdaptiveSupportScheme
from ...utils.support import volumeToSupport
from warpSPHCore.util.latticeDensity import latticeDensityFactor
from ...systems.pesph import pesphStateType
from ..adaptiveSupport import solveStageSupport
from .operators import computePressureState
from .thermodynamics import requirePositive


def refreshPESPHState(system, config, schemeConfig, *, solveSupport=True):
    """Refresh derived fields without replacing the authoritative E or A."""
    schemeConfig.validate(config)
    state = system.state
    if not isinstance(state, pesphStateType(schemeConfig.variant)):
        raise ValueError('PESPH state type does not match the configured variant')
    expected = 'energy' if schemeConfig.variant is PESPHVariant.PressureEnergy else 'entropy'
    if state.thermodynamicVariable != expected:
        raise ValueError('PESPH checkpoint thermodynamic variable does not match the configured variant')
    requirePositive(state.supports, 'supports')
    torch._assert_async(torch.all(state.kinds == 0), 'PESPH solid walls are not implemented')
    if solveSupport:
        solveStageSupport(system, state, config, schemeConfig)
    pressure = computePressureState(state, config, schemeConfig, system.adjacency)
    if schemeConfig.adaptiveSupportScheme is AdaptiveSupportScheme.NumberDensity:
        # The shared support Newton solve uses the uncalibrated kernel. A
        # constant lattice normalization in pressure sums changes the target
        # n*h^d by that constant; it must not be mistaken for nonconvergence.
        n = pressure.sums.numberDensity
        if getattr(config, 'calibrateNormalization', False):
            n = n / latticeDensityFactor(config.kernel, float(config.n_h), int(config.dim))
        target = volumeToSupport(1/n, config.targetNeighbors, config.dim)
        residual = (state.supports-target).abs()/state.supports
        tolerance = max(schemeConfig.adaptiveSupportThreshold, 32*torch.finfo(n.dtype).eps)
        torch._assert_async(torch.all(torch.isfinite(residual) & (residual <= tolerance)),
            'PESPH number-density support constraint did not converge; increase the iteration budget or fix the neighbourhood')
    state.densities = pressure.sums.massDensity
    state.internalEnergies = pressure.internalEnergies
    state.pressures = pressure.pressures
    state.soundspeeds = pressure.soundspeeds
    if schemeConfig.variant is PESPHVariant.PressureEntropy:
        state.totalEnergies = state.masses * (pressure.internalEnergies + 0.5*state.velocities.square().sum(-1))
    else:
        state.entropies = pressure.pressures / pressure.thermodynamicDensity.pow(schemeConfig.gamma)
    return pressure
