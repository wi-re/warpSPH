"""Artificial-viscosity dissipation power, split into its linear and quadratic parts.

AV_PLAN Part 0, Group B. The pairwise operator's momentum rate `a_i^AV` removes
kinetic energy at `P_AV = -sum_i m_i v_i . a_i^AV` (>= 0 for a dissipative
operator; the same amount is returned as heat, which is what makes the scheme
conserve total energy). For the `v_sig = C_l c - C_q mu` family (`Price2012_98`,
`Monaghan1992`, ...) `a^AV` is linear in `(C_l, C_q)` at fixed alpha, so the two
contributions are obtained by evaluating the operator twice -- once with
`C_q = 0`, once with `C_l = 0` -- and `avPowerTotal - linear - quadratic` is a
residual that flags a formulation (or `scaleBeta`, which multiplies the two
coefficients) for which the split is not additive.

This is a *diagnostic* path: it re-runs `computeViscosity` on the state and the
Verlet list left on the system by the last RHS evaluation, so it is off the hot
path and only costs anything when called.
"""

from __future__ import annotations

from typing import Dict

import torch

from warpSPHCore import OperationProperties, SupportScheme

from ...configurations import CRKSPHConfig
from ...configurations.moduleConfigurations.diffusionParameters import (
    dictToDiffusionParams, diffusionParamsToDict)
from .wp_diffusion import computeViscosityWarp

__all__ = ['computeAVPowerSplit']


def _power(state, config, params, adjacency, alphas=None) -> float:
    dvdt = computeViscosityWarp(
        state,
        operationProperties=OperationProperties(
            kernel=config.kernel, supportMode=SupportScheme.KernelMeanSymmetric),
        domain=config.domain,
        adjacency=adjacency,
        viscosityParams=params,
        queryAlphas=state.alphas if alphas is None else alphas,
    )
    return float(-(state.masses * torch.einsum('ij,ij->i', state.velocities, dvdt)).sum())


def computeAVPowerSplit(system, config, schemeConfig) -> Dict[str, float]:
    """`avPowerTotal/Linear/Quadratic` for `system`'s current state (Monaghan/CompSPH
    pair operator; CRKSPH's viscosity lives elsewhere and is not covered)."""
    if isinstance(schemeConfig, CRKSPHConfig):
        # CRKSPH's viscosity is its own reconstructed operator (modules/crk); this
        # function would evaluate a hypothetical pair operator, not what ran.
        nan = float('nan')
        return dict(avPowerTotal=nan, avPowerLinear=nan, avPowerQuadratic=nan,
                    avPowerSplitResidual=nan)
    state = system.state
    base = diffusionParamsToDict(schemeConfig.diffusionParams)
    out = {}
    for name, overrides in (('avPowerTotal', {}),
                            ('avPowerLinear', {'C_q': 0.0}),
                            ('avPowerQuadratic', {'C_l': 0.0})):
        params = dictToDiffusionParams({**base, **overrides})
        out[name] = _power(state, config, params, system.adjacency)
    out['avPowerSplitResidual'] = out['avPowerTotal'] - out['avPowerLinear'] - out['avPowerQuadratic']
    return out
