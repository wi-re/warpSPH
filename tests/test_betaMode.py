"""`BetaMode` (AV_PLAN S2 step 3): the quadratic coefficient beta either follows the
switched alpha (`Coupled`, the historical default) or does not (`Fixed`)."""

from __future__ import annotations

import torch

from warpSPH.configurations.moduleConfigurations.diffusionParameters import (
    BetaMode, buildDefaultDiffusionParamsCompressibleSPH, dictToDiffusionParams,
    diffusionParamsToDict)
from warpSPH.modules.dissipation.avPower import _power
from warpSPH.runner import run


def test_default_is_coupled_and_dict_round_trips():
    params = buildDefaultDiffusionParamsCompressibleSPH()
    assert params.betaMode == BetaMode.Coupled.value
    d = diffusionParamsToDict(params)
    assert d['betaMode'] == 'Coupled'
    d['betaMode'] = 'Fixed'
    assert dictToDiffusionParams(d).betaMode == BetaMode.Fixed.value
    assert dictToDiffusionParams(diffusionParamsToDict(dictToDiffusionParams(d))).betaMode \
        == BetaMode.Fixed.value


def test_configs_stored_before_the_mode_existed_load_as_coupled():
    d = diffusionParamsToDict(buildDefaultDiffusionParamsCompressibleSPH())
    del d['betaMode']
    assert dictToDiffusionParams(d).betaMode == BetaMode.Coupled.value


def _quadraticPower(system, ctx, mode, alphaScale):
    """Quadratic-only AV dissipation power (C_l = 0) with alpha scaled by `alphaScale`."""
    params = dictToDiffusionParams({**diffusionParamsToDict(ctx.schemeConfig.diffusionParams),
                                    'C_l': 0.0, 'C_q': 2.0, 'betaMode': mode.name})
    alphas = torch.full_like(system.state.alphas, alphaScale)
    return _power(system.state, ctx.config, params, system.adjacency, alphas)


def test_fixed_beta_ignores_alpha_and_coupled_follows_it():
    from warpSPH.cases.sod import sodCase
    res = run(sodCase, scheme='Monaghan', nx=60, nSteps=40, progress=False, quiet=True,
              params=dict(viscositySwitch='NoneSwitch', right_rho=0.125, right_pressure=0.1))
    ctx, system = res.ctx, res.state
    p1 = {m: _quadraticPower(system, ctx, m, 1.0) for m in BetaMode}
    p05 = {m: _quadraticPower(system, ctx, m, 0.5) for m in BetaMode}
    assert p1[BetaMode.Coupled] > 0
    # at alpha = 1 the two modes are the same arithmetic
    assert p1[BetaMode.Fixed] == p1[BetaMode.Coupled]
    # Fixed: beta does not move with alpha; Coupled: the quadratic term is linear in it
    assert abs(p05[BetaMode.Fixed] - p1[BetaMode.Fixed]) <= 1e-5 * abs(p1[BetaMode.Fixed])
    assert abs(p05[BetaMode.Coupled] - 0.5 * p1[BetaMode.Coupled]) <= 1e-5 * abs(p1[BetaMode.Coupled])
