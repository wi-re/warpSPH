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

import math
from typing import Dict

import torch

from warpSPHCore import OperationProperties, SupportScheme

from ...configurations import CRKSPHConfig
from ...configurations.moduleConfigurations.diffusionParameters import (
    dictToDiffusionParams, diffusionParamsToDict)
from ..reconstruction import reconstructionInputs
from .wp_diffusion import computeViscosityWarp

__all__ = ['computeAVPowerSplit', 'computeChenNixonRatio']


def _power(state, config, params, adjacency, alphas=None, velocityTensor=None, balsara=None) -> float:
    dvdt = computeViscosityWarp(
        state,
        operationProperties=OperationProperties(
            kernel=config.kernel, supportMode=SupportScheme.KernelMeanSymmetric),
        domain=config.domain,
        adjacency=adjacency,
        viscosityParams=params,
        queryAlphas=state.alphas if alphas is None else alphas,
        queryVelocityTensor=velocityTensor,
        queryBalsara=balsara,
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
    # the pair velocity (raw or reconstructed, AV_PLAN Phases 3-4) as the scheme evaluates it
    params, velocityTensor, balsara = reconstructionInputs(state, config, schemeConfig.diffusionParams, system.adjacency)
    base = diffusionParamsToDict(params)
    out = {}
    for name, overrides in (('avPowerTotal', {}),
                            ('avPowerLinear', {'C_q': 0.0}),
                            ('avPowerQuadratic', {'C_l': 0.0})):
        params = dictToDiffusionParams({**base, **overrides})
        out[name] = _power(state, config, params, system.adjacency, velocityTensor=velocityTensor, balsara=balsara)
    out['avPowerSplitResidual'] = out['avPowerTotal'] - out['avPowerLinear'] - out['avPowerQuadratic']
    return out


def computeChenNixonRatio(system, config, schemeConfig) -> Dict[str, float]:
    """Chen & Nixon (2025) Eq. (6): the quadratic-to-linear AV ratio `135 beta h / (62 pi alpha H)` per particle,
    with `h` the smoothing length (`support / sphKernelScale`), `alpha`, `beta` the switched coefficients as the
    pair operator forms them (`BetaMode`) and -- there being no disc -- `H` the local density gradient length
    `rho / |grad rho|` (AV_PLAN Phase 5A). Reported as the median and the 90th percentile over particles with a
    nonzero gradient; above 1 the quadratic term dominates. NaN for CRKSPH (its own operator)."""
    from warpSPHCore import GradientScheme, WarpOperation, sphKernelScale, warpOperation
    from ...configurations.moduleConfigurations.diffusionParameters import BetaMode
    nan = float('nan')
    if isinstance(schemeConfig, CRKSPHConfig):
        return dict(chenNixonRatioMedian=nan, chenNixonRatioP90=nan)
    st = system.state
    params = schemeConfig.diffusionParams
    gradRho = warpOperation(
        st, OperationProperties(kernel=config.kernel, operation=WarpOperation.Gradient,
                                supportMode=SupportScheme.SuperSymmetric, gradientMode=GradientScheme.Difference),
        domain=config.domain, adjacency=system.adjacency, queryValues=st.densities)
    alpha_ = st.alphas if getattr(st, 'alphas', None) is not None else torch.ones_like(st.densities)
    alpha = alpha_ * float(params.C_l)
    if params.betaMode == BetaMode.Fixed.value:
        beta = torch.full_like(alpha, float(params.C_q))
    else:
        beta = alpha_ * float(params.C_q) * (float(params.C_l) * alpha_ if params.scaleBeta else 1.0)
    h = st.supports / float(sphKernelScale(config.kernel.value, config.dim))
    g = gradRho.norm(dim=-1)
    sel = (g > 0) & (alpha > 0)
    if not bool(sel.any()):
        return dict(chenNixonRatioMedian=nan, chenNixonRatioP90=nan)
    ratio = (135.0 * beta * h * g / (62.0 * math.pi * alpha * st.densities))[sel]
    return dict(chenNixonRatioMedian=float(ratio.median()), chenNixonRatioP90=float(torch.quantile(ratio.double(), 0.9)))
