"""OPEN_PROBLEMS §22: the M^-1-corrected velocity gradient of the Cullen-Dehnen / Cullen-Hopkins switches
(`modules/shockCapturing/common.py`, opt-in via `correctVelocityGradient` / `divergenceScheme='cullen'`).

Two bugs, both checked here: `computeM` summed `m_j` where the difference gradient sums `m_j / rho_j` (so `M ~ rho I`
and the corrected divergence came out as `div v / rho`: 4.0 for a unit divergence at rho = 0.25), and
`computeShearTensor` corrected as `M^-1 Vs` where `Vs = A M` needs `Vs M^-1` (wrong shear and rotation wherever
`M != I`, i.e. off a regular lattice).
"""

from __future__ import annotations

import pytest
import torch

from warpSPHCore import SupportScheme, buildVerletList
from warpSPH.modules.shockCapturing.common import computeDivergence, computeM, computeShearTensor
from warpSPH.runner import run

A = torch.tensor([[0.3, -0.7], [0.5, 0.2]])


def test_cullen_divergence_is_not_divided_by_density():
    from warpSPH.cases.sodND import sod2dCase
    res = run(sod2dCase, scheme='Monaghan', nSteps=1, progress=False, quiet=True,
              params=dict(viscositySwitch='CullenDehnen2010'))
    st, adj, config, sc = res.state.state, res.state.adjacency, res.ctx.config, res.ctx.schemeConfig
    sc.viscositySwitchParams.divergenceScheme = 'cullen'
    sc.viscositySwitchParams.correctVelocityGradient = True
    st.velocities = torch.stack([st.positions[:, 0], torch.zeros_like(st.positions[:, 0])], 1)   # div v = 1
    div = computeDivergence(st, config, sc, SupportScheme.SuperSymmetric, adj)
    rho = st.densities
    for lo, hi in ((0.9, 1.1), (0.2, 0.3)):
        sel = (rho > lo) & (rho < hi)
        assert int(sel.sum()) > 100
        assert div[sel].median().item() == pytest.approx(1.0, abs=1e-3), f'rho in ({lo}, {hi})'


def test_corrected_shear_tensor_exact_on_jittered_lattice():
    from warpSPH.cases.greshoVortex import greshoVortexCase
    res = run(greshoVortexCase, scheme='Monaghan', nx=32, nSteps=1, progress=False, quiet=True,
              params=dict(viscositySwitch='CullenDehnen2010'))
    st, config, sc = res.state.state, res.ctx.config, res.ctx.schemeConfig
    sc.viscositySwitchParams.correctVelocityGradient = True
    g = torch.Generator().manual_seed(1)
    st.positions = st.positions + (0.3 / 32 * (2 * torch.rand(st.positions.shape, generator=g) - 1)).to(st.positions)
    adj = buildVerletList(st, config.domain, verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)
    A_ = A.to(st.positions)
    st.velocities = st.positions @ A_.T
    M_inv = torch.linalg.pinv(computeM(st, config, sc, SupportScheme.SuperSymmetric, adj))
    trace, S, R = computeShearTensor(M_inv, st, config, sc, SupportScheme.SuperSymmetric, adj)
    inner = st.positions.abs().amax(dim=1) < 0.5 - 1.05 * st.supports
    sym = 0.5 * (A_ + A_.T) - torch.eye(2).to(A_) * torch.trace(A_) / 2
    rot = 0.5 * (A_ - A_.T)
    assert (trace[inner] - torch.trace(A_)).abs().max() < 1e-4
    assert (S[inner] - sym).abs().max() < 1e-4, 'shear wrong: the correction is applied on the wrong side'
    assert (R[inner] - rot).abs().max() < 1e-4
