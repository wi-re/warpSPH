"""`divergenceFree` (DFSPH) on analytic walls (EXPERIMENTAL, OPEN_PROBLEMS 31): the wall terms of its step (density, free-surface detector, velocity diffusion, the omni pressure solves, the analytic no-penetration shift,
the XSPH / friction filters) come together on a still tank and a short dam break.
"""
import os

import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase, run  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')

ROLL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples', 'sloshingTank', 'SPHERIC_TestCase10', 'data_files', 'lateral_water_1x.txt')


def tank(params=None, nSteps=60, nh=2.57):
    importAll()
    case = getCase('sloshingTank')
    spec = CaseSpec(caseName='dfsph', scheme='divergenceFree', params={**case.params, 'wallRepresentation': 'analytic', 'rollDataFile': ROLL, 'rollStartTime': 100.0, **(params or {})}).merged(**case.defaults).merged(
        scheme='divergenceFree', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=nh, nSteps=nSteps,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    with pytest.warns(UserWarning, match='EXPERIMENTAL'):
        return run(case, spec)


@pytest.mark.parametrize('nh', [2.57, 4.0])
def test_still_tank_stays_quiet(nh):
    res = tank(nh=nh)
    assert float(np.nanmax(res.series('maxVelocity'))) < 0.5
    st = res.state.state
    assert float(st.densities[st.kinds == 0].max()) < 1.02


def test_filters_and_mls_wall_pressure_run_with_divergence_free():
    res = tank({'xsphCoefficient': 1e-4, 'boundaryFriction': 5e-3, 'analyticWallPressure': 'mls'}, nSteps=40)
    assert not res.diverged
    assert float(np.nanmax(res.series('maxVelocity'))) < 1.5


def test_analytic_no_penetration_acts_on_a_wall_bound_particle():
    """the step folds `analyticNoPenShift` into `dvdt`: a particle sent into the floor does not pass it."""
    from warpSPH.modules.analyticBoundary import analyticNoPenShift
    res = tank(nSteps=1)
    st, ctx = res.state.state, res.ctx
    sc, cfg = ctx.schemeConfig, ctx.config
    fluid = st.kinds == 0
    i = int(torch.argmin(torch.where(fluid, st.positions[:, 1], torch.full_like(st.positions[:, 1], 9.0))))
    d, _, _ = sc.boundaryProvider.scene.signed_distance(st.positions.double(), supportMax=float(st.supports.max()), want_body=True)[:3]
    st.positions[i, 1] -= float(d[i]) - 0.1 * float(cfg.dx)              # 0.1 dx from the floor: inside the law's 0.25 dx
    st.velocities = torch.zeros_like(st.velocities)
    st.velocities[i, 1] = -1.0
    shift = analyticNoPenShift(sc.boundaryProvider, st, cfg, sc, float(cfg.dx))
    assert float(shift[i, 1]) > 0.0


# ---- stage 2 A5 / A6: moving walls and the body loads of the incompressible loops ---------------------------------------------------------------------------------------

def _tankWith(scheme, setup, nSteps, params=None, nh=2.57):
    importAll()
    case = getCase('sloshingTank')
    ic = case.initialConditions

    def ic2(ctx, system):
        ic(ctx, system)
        setup(ctx)
    case.initialConditions = ic2                                       # the case object is shared by the registry: restored below
    spec = CaseSpec(caseName='mv', scheme=scheme, params={**case.params, 'wallRepresentation': 'analytic', 'rollDataFile': ROLL, 'rollStartTime': 100.0, **(params or {})}).merged(**case.defaults).merged(
        scheme=scheme, integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=nh, nSteps=nSteps,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    try:
        with pytest.warns(UserWarning, match='EXPERIMENTAL'):
            return run(case, spec)
    finally:
        case.initialConditions = ic


@pytest.mark.parametrize('scheme', ['omniIncompressible', 'divergenceFree'])
@pytest.mark.parametrize('inDivergence', [False, True])
def test_a_translating_tank_carries_the_fluid_with_it(scheme, inDivergence):
    """the whole tank translates at U in x (a prescribed rigid body: the wall velocity enters the source through `wallDivergence(bodyVelocity=True)`): the fluid, at rest in the lab frame, is
    pushed by the trailing wall and follows it (mean vx past 0.4 U after 150 steps, bounded, no suction); the body moves with U."""
    U = 0.5

    def setup(ctx):
        ctx.schemeConfig.analyticWallInDivergence = inDivergence
        rb = ctx.schemeConfig.boundaryProvider.rigidBodies[0]
        rb.linearVelocity = torch.tensor([U, 0.0], dtype=rb.linearVelocity.dtype, device=rb.linearVelocity.device)
    res = _tankWith(scheme, setup, 150)
    st = res.state.state
    f = st.kinds == 0
    rb = res.ctx.schemeConfig.boundaryProvider.rigidBodies[0]
    assert float(rb.centerOfMass[0]) == pytest.approx(U * float(res.state.t), rel=0.02)
    assert float(st.velocities[f, 0].mean()) > 0.4 * U
    assert float(st.velocities[f].norm(dim=1).max()) < 2.0


@pytest.mark.parametrize('scheme', ['omniIncompressible', 'divergenceFree'])
def test_the_booked_pressure_load_of_a_still_tank_averages_to_the_weight_of_the_fluid(scheme):
    """the instantaneous load is as noisy as the density-solve pressure (0.5 - 1.2 of the weight from step to step); its mean over 40 steps is the weight (the momentum balance of the step)."""
    importAll()
    case = getCase('sloshingTank')
    ic, ps = case.initialConditions, case.postStep
    rec = []

    def ic2(ctx, system):
        ic(ctx, system)
        ctx.schemeConfig.analyticWallLoads = True

    def ps2(ctx, state, step):
        if ps is not None:
            ps(ctx, state, step)
        rec.append(float(ctx.schemeConfig.boundaryProvider.rigidBodies[0].load[0][1]))
    case.initialConditions, case.postStep = ic2, ps2
    spec = CaseSpec(caseName='ld', scheme=scheme, params={**case.params, 'wallRepresentation': 'analytic', 'rollDataFile': ROLL, 'rollStartTime': 100.0}).merged(**case.defaults).merged(
        scheme=scheme, integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=2.57, nSteps=100,
        plot=False, store=False, progress=False, video=False, show=False, quiet=True)
    try:
        with pytest.warns(UserWarning, match='EXPERIMENTAL'):
            res = run(case, spec)
    finally:
        case.initialConditions, case.postStep = ic, ps
    st = res.state.state
    f = st.kinds == 0
    weight = float(st.masses[f].sum()) * 9.81
    assert -np.mean(rec[-40:]) == pytest.approx(weight, rel=0.08)
    load = res.ctx.schemeConfig.boundaryProvider.rigidBodies[0].load[0]
    assert abs(float(load[0])) < 0.1 * weight


def test_a_free_body_is_refused_by_the_incompressible_loops():
    def setup(ctx):
        ctx.schemeConfig.boundaryProvider.rigidBodies[0].dynamic = True
    with pytest.raises(NotImplementedError, match='dynamic'):
        _tankWith('omniIncompressible', setup, 2)
