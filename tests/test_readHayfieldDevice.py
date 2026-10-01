"""Read & Hayfield (2012) pair loops run on the device with the minimum-image convention.

The host (torch) version differenced raw positions, so a pair straddling a periodic boundary had
|x_ij| ~ the domain size (wrong direction, wrong `K_ij` cut-off). Shifting every particle by the same
periodic offset moves which pairs straddle the boundary but changes no physical quantity, so the
signal velocity and the entropy dissipation must be unchanged; a raw difference changes them.
"""

from __future__ import annotations

import dataclasses
import copy

import pytest
import torch

from warpSPH.cases.greshoVortex import greshoVortexCase
from warpSPH.cases.sod import sodCase
from warpSPH.modules.shockCapturing.ReadHayfield2012 import computeReadHayfieldTerms
from warpSPH.runner import run
from warpSPHCore import SupportScheme

_CASES = {
    'sod1d': (sodCase, dict(nx=60, nSteps=20,
                            params=dict(viscositySwitch='ReadHayfield2012', right_rho=0.125, right_pressure=0.1,
                                        alpha_min=0.2, alpha_max=1.0))),
    'gresho2d': (greshoVortexCase, dict(nx=24, nSteps=15,
                                        params=dict(viscositySwitch='ReadHayfield2012'))),
}


@pytest.mark.parametrize('name', list(_CASES))
def test_pair_loops_are_invariant_under_a_periodic_shift(name):
    case, kw = _CASES[name]
    res = run(case, scheme='Monaghan', progress=False, quiet=True, **kw)
    st, config, scheme = res.state.state, res.ctx.config, res.ctx.schemeConfig
    adj = res.state.adjacency
    lo, hi = config.domain.min, config.domain.max
    L = hi - lo
    assert bool(config.domain.periodic.all()), 'the test needs a fully periodic case'

    def terms(state):
        _, sw = computeReadHayfieldTerms(1e-3, state, config, scheme, SupportScheme.KernelMeanSymmetric, adj)
        return sw.v_sig, sw.dudt_diss

    v0, d0 = terms(st)
    assert d0.abs().max() > 0, 'the entropy dissipation never engaged, so the test proves nothing'

    shifted = copy.copy(st)
    shift = 0.37 * L
    shifted.positions = lo + torch.remainder(st.positions + shift - lo, L)
    v1, d1 = terms(shifted)
    torch.testing.assert_close(v1, v0, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(d1, d0, rtol=1e-3, atol=1e-4 * d0.abs().max().item())


def test_a_raw_position_difference_would_have_failed_this():
    """Documents the failure mode the device kernels remove: across the boundary the raw difference is
    ~ the domain size, far outside the support, so such a pair is lost from the sum entirely."""
    case, kw = _CASES['sod1d']
    res = run(case, scheme='Monaghan', progress=False, quiet=True, **kw)
    st, adj, config = res.state.state, res.state.adjacency, res.ctx.config
    raw = (st.positions[adj.i] - st.positions[adj.j]).norm(dim=-1)
    assert (raw > 2.0 * st.supports.max()).any(), 'no pair straddles the periodic boundary in this state'
