"""The Monaghan host's RHS must conserve total energy piece by piece (OPEN_PROBLEMS §16, resolved
2026-09-30): the viscous heating used to lack the 1/2 of `du_i/dt = 1/2 sum m_j Pi_ij v_ij . gradW_ij`,
so it was exactly twice the kinetic energy the viscous force removes and the scheme gained one full
dissipation's worth of energy (Sedov 1.0 -> 2.33 in 1D)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

from probe_monaghanEnergy import pieces   # noqa: E402
from warpSPH.cases.sod import sodCase      # noqa: E402
from warpSPH.runner import run             # noqa: E402


@pytest.mark.parametrize('Cq', [0.0, 2.0])
def test_each_rhs_piece_injects_no_net_energy(Cq):
    def configure(ctx, _orig=sodCase.configureScheme):
        _orig(ctx)
        ctx.schemeConfig.diffusionParams.C_q = Cq

    res = run(dataclasses.replace(sodCase, configureScheme=configure), scheme='Monaghan', nx=100, nSteps=120,
              progress=False, quiet=True,
              params=dict(viscositySwitch='NoneSwitch', right_rho=0.125, right_pressure=0.1))
    st, out = pieces(res.state, res.ctx.config, res.ctx.schemeConfig)
    m, v = st.masses, st.velocities
    # one physical scale for every piece (a piece with no kinetic part, e.g. the conductivity, would
    # otherwise be normalised by its own round-off)
    dvdtP, _ = out['pressure force + work']
    scale = (m * torch.einsum('ij,ij->i', v, dvdtP)).abs().sum().item()
    for name, (dvdt, dudt) in out.items():
        kinetic = (m * torch.einsum('ij,ij->i', v, dvdt)).sum().item()
        heat = (m * dudt).sum().item()
        assert abs(kinetic + heat) / scale < 1e-5, f'{name}: kinetic {kinetic:+.4e} heat {heat:+.4e}'
    # the viscous piece must actually be doing something, or the test proves nothing
    dvdt, dudt = out['viscous force + heating']
    assert abs((m * dudt).sum().item()) > 1e-3
