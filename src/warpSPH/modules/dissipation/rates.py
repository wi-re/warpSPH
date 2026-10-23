"""Compose the existing compressible dissipation operators and switch hooks.

Returns acceleration and specific heat separately so a host can evolve u,
total E, or an independently validated thermodynamic variable. Density and
pressure are supplied by that host's closure; no EOS is recomputed here.
"""

from dataclasses import dataclass

import torch
from warpSPHCore import OperationProperties, SupportScheme

from .wp_conductivity import computeConductivityWarp
from .wp_diffusion import computeViscosityWarp
from .wp_dissipation import computeThermalDissipationWarp
from ..momentum import computeMomentumConsistent
from ..reconstruction import reconstructionInputs, stateGradientInputs
from ..shockCapturing import computeViscositySwitchTerms, updateViscositySwitch


@dataclass(frozen=True)
class DissipationRates:
    acceleration: torch.Tensor
    viscousHeating: torch.Tensor
    conduction: torch.Tensor
    switchHeating: torch.Tensor

    @property
    def heating(self):
        return self.viscousHeating + self.conduction + self.switchHeating


def computeDissipationRates(state, dt, config, schemeConfig, adjacency, pressureAcceleration):
    state.alphas, switch = computeViscositySwitchTerms(
        dt, state, config, schemeConfig, SupportScheme.SuperSymmetric, adjacency)
    params, jacobian, balsara = reconstructionInputs(state, config, schemeConfig.diffusionParams, adjacency)
    params, gradients = stateGradientInputs(state, config, params, adjacency)
    common = dict(operationProperties=OperationProperties(
        kernel=config.kernel, supportMode=SupportScheme.KernelMeanSymmetric,
        n_h=getattr(config, 'n_h', None),
        calibrateNormalization=getattr(config, 'calibrateNormalization', False)),
        domain=config.domain, adjacency=adjacency, queryAlphas=state.alphas)
    pair = dict(queryVelocityTensor=jacobian, queryBalsara=balsara, queryStateGradients=gradients)
    acceleration = computeViscosityWarp(state, viscosityParams=params, **common, **pair)
    viscous = computeThermalDissipationWarp(state, conductivityParams=params, **common, **pair)
    conduction = computeConductivityWarp(state, conductivityParams=params, **common)
    state.alpha0s, switch = updateViscositySwitch(
        switch, dt, pressureAcceleration+acceleration, state, config, schemeConfig,
        SupportScheme.SuperSymmetric, adjacency)
    drho = computeMomentumConsistent(state, config, schemeConfig=schemeConfig, adjacency=adjacency)
    state.divergence = -drho/state.densities
    extra = getattr(switch, 'dudt_diss', None)
    return DissipationRates(acceleration, viscous, conduction,
                            torch.zeros_like(viscous) if extra is None else extra)
