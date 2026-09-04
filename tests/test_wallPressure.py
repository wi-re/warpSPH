"""`modules/incompressible/wallPressure.py`'s Adami hydrostatic correction.

The correction (Adami et al. 2012 Eq. 27, volume-weighted as De Courcy et al.
2024 Eq. 61) has an exact property worth testing directly rather than through a
case: for a pressure field that is *linear* in space, every neighbour's
extrapolated contribution `p_f + rho_f * g . (r_w - r_f)` equals the analytic
`p(r_w)` exactly, so the kernel-weighted average of them does too -- regardless
of how truncated or lopsided the wall particle's neighbourhood is. The plain
Shepard gather has no such property: it returns a weighted average of `p_f`
over a one-sided neighbourhood, which is biased by O(h) in the gradient
direction. That bias is what stops a hydrostatic column from holding its
gradient (ACSPH_PLAN.md Sec. 4.4), and it is what these tests measure.
"""

import math

import pytest
import torch

from warpSPHCore import ParticleState
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.modules.incompressible.wallPressure import (
    _MIN_WEIGHT, _shepardValue, wallPressureExtrapolation)
from warpSPH.utils.domain import buildDomainDescription

RHO0 = 1000.0
G = 9.81
HEIGHT = 0.5
DX = HEIGHT / 24
N_WALL_ROWS = 3


def buildColumn(device, dtype, periodic=False):
    """A 2D column of fluid on `y in (0, HEIGHT]` over `N_WALL_ROWS` rows of
    `kind == 1` wall particles, on one shared lattice. Wide enough in x that
    the probed wall rows are interior in x (side truncation would be a
    different -- and for a y-only pressure gradient, harmless -- effect)."""
    nx = 24
    xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) * DX
    fluidRows = int(round(HEIGHT / DX))
    ysFluid = (torch.arange(fluidRows, device=device, dtype=dtype) + 0.5) * DX
    ysWall = -(torch.arange(N_WALL_ROWS, device=device, dtype=dtype) + 0.5) * DX

    def lattice(ys):
        gx, gy = torch.meshgrid(xs, ys, indexing='ij')
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    fluid, wall = lattice(ysFluid), lattice(ysWall)
    positions = torch.cat([fluid, wall], dim=0).contiguous()
    n = positions.shape[0]
    kinds = torch.cat([
        torch.zeros(fluid.shape[0], dtype=torch.int32, device=device),
        torch.ones(wall.shape[0], dtype=torch.int32, device=device)]).contiguous()

    config = SimulationConfig(
        device=device, dtype=dtype, dim=2,
        domain=buildDomainDescription(l=4.0, dim=2, periodic=periodic,
                                      device=device, dtype=dtype))
    h = config.n_h * DX
    state = ParticleState(
        positions=positions,
        supports=torch.full((n,), h, device=device, dtype=dtype),
        masses=torch.full((n,), RHO0 * DX ** 2, device=device, dtype=dtype),
        densities=torch.full((n,), RHO0, device=device, dtype=dtype),
        kinds=kinds)
    state.pressures = torch.zeros(n, device=device, dtype=dtype)
    return state, config


def hydrostatic(positions):
    """`p = rho0 * g * (HEIGHT - y)`: zero at the free surface, linear below."""
    return RHO0 * G * (HEIGHT - positions[:, 1])


@pytest.fixture(scope='module')
def column(runtime):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return buildColumn(device, torch.float32)


def wallRows(state, config):
    """The `kind == 1` rows the extrapolation actually writes: boundary rows
    with enough fluid Shepard weight that `_shepardMirror` does not zero them."""
    _, den = _shepardValue(state, config, None, torch.zeros_like(state.masses))
    return (state.kinds == 1) & (den > _MIN_WEIGHT)


def test_bodyForceTermReproducesALinearPressureFieldExactly(column):
    """The defining property: with the correction on, a linear `p` is
    extrapolated to the wall exactly, because each neighbour's contribution is
    already the analytic wall value."""
    state, config = column
    p = hydrostatic(state.positions)
    fluid = state.kinds == 0
    gravity = torch.tensor([0.0, -G], device=p.device, dtype=p.dtype)

    corrected = wallPressureExtrapolation(
        state, config, None, p, fluid, mode='shepard', bodyForce=gravity)

    rows = wallRows(state, config)
    assert int(rows.sum()) > 0
    exact = hydrostatic(state.positions)
    err = (corrected[rows] - exact[rows]).abs().max() / (RHO0 * G * HEIGHT)
    assert err < 1e-5, f'relative error {err:.3g} is not machine precision'


def test_plainShepardIsBiasedByRoughlyTheNeighbourhoodOffset(column):
    """The control: without the correction the wall reads the depth-*average*
    of its fluid neighbours, i.e. it is short by `rho0 * g * <y_f - y_w>`, a
    fraction of `h`. This is the error the correction removes."""
    state, config = column
    p = hydrostatic(state.positions)
    fluid = state.kinds == 0

    plain = wallPressureExtrapolation(
        state, config, None, p, fluid, mode='shepard')

    rows = wallRows(state, config)
    exact = hydrostatic(state.positions)
    deficit = (exact[rows] - plain[rows])
    # Every probed wall row under-reads, by a length scale of order h.
    assert bool((deficit > 0).all())
    offset = float(deficit.max()) / (RHO0 * G)
    assert DX < offset < config.n_h * DX


def test_mirrorModeTakesTheSameCorrection(column):
    """'mirror' is a different closure (it reflects about the wall's own
    carried `p_wall`), but the correction enters it additively and identically
    -- so switching it on has to move it by exactly the same amount."""
    state, config = column
    p = hydrostatic(state.positions)
    fluid = state.kinds == 0
    gravity = torch.tensor([0.0, -G], device=p.device, dtype=p.dtype)

    plain = wallPressureExtrapolation(state, config, None, p, fluid, mode='mirror')
    corrected = wallPressureExtrapolation(
        state, config, None, p, fluid, mode='mirror', bodyForce=gravity)
    shepPlain = wallPressureExtrapolation(state, config, None, p, fluid, mode='shepard')
    shepCorrected = wallPressureExtrapolation(
        state, config, None, p, fluid, mode='shepard', bodyForce=gravity)

    rows = wallRows(state, config)
    # `mode='mirror'` clamps at 0 like 'shepard' does; compare only rows where
    # neither closure is clamped, so the shift is the raw additive one.
    unclamped = rows & (plain > 0)
    delta = (corrected - plain)[unclamped]
    reference = (shepCorrected - shepPlain)[unclamped]
    assert torch.allclose(delta, reference, rtol=1e-5, atol=1e-3 * RHO0 * G * HEIGHT)


def test_perParticleBodyForceMatchesTheUniformOne(column):
    """`(dim,)` and `(N, dim)` are the same field, so they must agree."""
    state, config = column
    p = hydrostatic(state.positions)
    fluid = state.kinds == 0
    gravity = torch.tensor([0.0, -G], device=p.device, dtype=p.dtype)
    perParticle = gravity.unsqueeze(0).expand_as(state.positions).contiguous()

    a = wallPressureExtrapolation(state, config, None, p, fluid,
                                  mode='shepard', bodyForce=gravity)
    b = wallPressureExtrapolation(state, config, None, p, fluid,
                                  mode='shepard', bodyForce=perParticle)
    assert torch.allclose(a, b)


def test_mlsRejectsABodyForce(column):
    """'mls' already carries the gradient in its linear fit; taking the
    correction too would double-count it, so it raises rather than ignoring."""
    state, config = column
    p = hydrostatic(state.positions)
    fluid = state.kinds == 0
    with pytest.raises(ValueError, match='bodyForce'):
        wallPressureExtrapolation(state, config, None, p, fluid, mode='mls',
                                  bodyForce=[0.0, -G])


def test_rejectsABodyForceAlongAPeriodicAxis(runtime):
    """The two-gather moment decomposition is not minimum-image safe, so a
    body force with a component along a periodic axis is refused rather than
    silently returning a wrong wall pressure (see the module docstring)."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    state, config = buildColumn(device, torch.float32,
                                periodic=torch.tensor([True, False], device=device))
    p = hydrostatic(state.positions)
    fluid = state.kinds == 0
    # Along y (non-periodic): fine.
    wallPressureExtrapolation(state, config, None, p, fluid, mode='shepard',
                              bodyForce=[0.0, -G])
    # Along x (periodic): refused.
    with pytest.raises(ValueError, match='periodic'):
        wallPressureExtrapolation(state, config, None, p, fluid, mode='shepard',
                                  bodyForce=[-G, 0.0])
