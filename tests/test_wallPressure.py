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


def buildWrappingColumn(device, periodicAxes):
    """A column whose lattice reaches both faces of the domain, so a
    minimum-image pair genuinely can wrap. `buildColumn`'s domain is four
    times the lattice, which is exactly the case the guard must *not* fire on."""
    dtype = torch.float32
    state, config = buildColumn(device, dtype, periodic=periodicAxes)
    positions = state.positions
    low = positions.min(dim=0).values - 0.25 * DX
    high = positions.max(dim=0).values + 0.25 * DX
    config.domain = type(config.domain)(
        min=low, max=high,
        periodic=torch.as_tensor(periodicAxes, device=device).expand(2).clone(),
        dim=2)
    return state, config


def momentByTwoGathers(state, config, adjacency=None):
    """The moment as it used to be computed: `r_w sum V rho W - sum V rho r W`.
    Exact in an unwrapped domain, wrong by `+-L_d` per wrapping pair otherwise
    -- which is what these tests are for."""
    from warpSPHCore import (OperationDirection, OperationProperties, SupportScheme,
                             WarpOperation, warpOperation)
    props = OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric,
        operationMode=OperationDirection.FluidToBoundary)
    rho = state.densities
    rhoW = warpOperation(state, props, domain=config.domain, referenceValues=rho,
                         adjacency=adjacency)
    rhoXW = warpOperation(state, props, domain=config.domain,
                          referenceValues=rho.unsqueeze(-1) * state.positions,
                          adjacency=adjacency)
    return state.positions * rhoW.unsqueeze(-1) - rhoXW


def test_theMomentKernelMatchesTheOldDecompositionWhereThatWasExact(column):
    """The two-gather decomposition is exact algebra in an unwrapped domain, so
    the kernel that replaced it has to agree with it there. This is the test
    that says the swap changed nothing except the periodic case."""
    from warpSPH.modules.incompressible.wp_wallMoment import computeWallMomentWarp
    from warpSPHCore import OperationDirection, OperationProperties, SupportScheme
    state, config = column

    kernelMoment = computeWallMomentWarp(
        state,
        OperationProperties(kernel=config.kernel,
                            supportMode=SupportScheme.SuperSymmetric,
                            operationMode=OperationDirection.FluidToBoundary),
        domain=config.domain)
    twoGather = momentByTwoGathers(state, config)

    boundary = state.kinds == 1
    scale = float(twoGather[boundary].abs().max())
    assert scale > 0
    assert float((kernelMoment - twoGather)[boundary].abs().max()) < 1e-4 * scale


def test_theMomentKernelIsMinimumImageWhereTheDecompositionIsNot(runtime):
    """On a periodic domain whose lattice reaches both faces, pairs genuinely
    wrap. The kernel takes `x_ij` from `computeDistanceVec` like every other
    operator, so it stays correct; the old decomposition is off by a domain
    length per wrapping pair. Graded against a brute-force `O(N^2)` sum with
    explicit minimum-image, which shares no code with either."""
    from warpSPH.modules.incompressible.wp_wallMoment import computeWallMomentWarp
    from warpSPHCore import OperationDirection, OperationProperties, SupportScheme
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    state, config = buildWrappingColumn(device, torch.tensor([True, True], device=device))

    kernelMoment = computeWallMomentWarp(
        state,
        OperationProperties(kernel=config.kernel,
                            supportMode=SupportScheme.SuperSymmetric,
                            operationMode=OperationDirection.FluidToBoundary),
        domain=config.domain)
    reference = bruteForceMoment(state, config)
    twoGather = momentByTwoGathers(state, config)

    boundary = state.kinds == 1
    scale = float(reference[boundary].abs().max())
    assert scale > 0

    assert float((kernelMoment - reference)[boundary].abs().max()) < 1e-4 * scale
    # ... and the old decomposition really was wrong here, so the test above is
    # discriminating rather than vacuous.
    assert float((twoGather - reference)[boundary].abs().max()) > 0.5 * scale


def bruteForceMoment(state, config):
    """`sum_f V_f rho_f (r_w - r_f) W_wf` over all pairs, with the minimum image
    taken explicitly and a hand-written Wendland C2. Independent of the kernel
    under test on purpose."""
    positions = state.positions
    span = (torch.as_tensor(config.domain.max) - torch.as_tensor(config.domain.min))
    periodic = torch.as_tensor(config.domain.periodic).to(positions.dtype)
    delta = positions.unsqueeze(1) - positions.unsqueeze(0)
    wrapped = delta - span * torch.round(delta / span) * periodic

    r = wrapped.norm(dim=-1)
    h = state.supports[0]
    q = (r / h).clamp(max=1.0)
    cD = 7.0 / math.pi
    w = torch.where(r <= h, (1 - q) ** 4 * (1 + 4 * q) * cD / h ** 2,
                    torch.zeros_like(r))

    volumes = state.masses / state.densities
    weight = (volumes * state.densities).unsqueeze(0) * w
    # FluidToBoundary: query rows are `kind == 1`, reference rows `kind == 0`.
    weight = weight * (state.kinds == 0).unsqueeze(0).to(weight.dtype)
    weight = weight * (state.kinds == 1).unsqueeze(1).to(weight.dtype)
    return torch.einsum('ij,ijd->id', weight, wrapped)


def test_aWrappingPeriodicDomainNoLongerNeedsAWorkaround(runtime):
    """The end-to-end statement: the wall pressure is computable on a periodic
    domain with gravity along a periodic axis and pairs that genuinely wrap.
    That configuration used to raise, which is why `hydrostaticColumn` had to
    be made non-periodic under ACSPH."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    state, config = buildWrappingColumn(device, torch.tensor([True, True], device=device))
    p = hydrostatic(state.positions)
    fluid = state.kinds == 0
    out = wallPressureExtrapolation(state, config, None, p, fluid, mode='shepard',
                                    bodyForce=[0.0, -G])
    assert torch.isfinite(out).all()
