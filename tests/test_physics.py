"""Short runs of each converted case, asserting a physical invariant.

Deliberately tiny -- 20 steps at a coarse resolution -- so the suite stays
usable as a pre-commit check. Each assertion is a property the scheme is
supposed to have, not a golden number, so they survive refactors and fail on
real regressions.
"""

import io
import contextlib

import numpy as np
import pytest

from warpSPH.runner import run

STEPS = 20


def _run(case, **overrides):
    """Run quietly; the schemes print per-step solver diagnostics."""
    with contextlib.redirect_stdout(io.StringIO()):
        return run(case, progress=False, nSteps=STEPS, **overrides)


@pytest.fixture(scope='module')
def sodResult():
    from warpSPH.cases.sod import sodCase
    return _run(sodCase, nx=200)


@pytest.fixture(scope='module')
def tgvResult():
    from warpSPH.cases.tgv import tgvCase
    return _run(tgvCase, nx=32)


@pytest.fixture(scope='module')
def shearWaveResult():
    from warpSPH.cases.shearWave import shearWaveCase
    return _run(shearWaveCase, nx=32)


@pytest.fixture(scope='module')
def dambreakResult():
    from warpSPH.cases.dambreak import dambreakCase
    return _run(dambreakCase, nx=48)


@pytest.fixture(scope='module')
def kolmogorovIncompressibleResult():
    from warpSPH.cases.kolmogorovIncompressible import kolmogorovIncompressibleCase
    return _run(kolmogorovIncompressibleCase, nx=32)


@pytest.fixture(scope='module')
def randomFlowPeriodicResult():
    from warpSPH.cases.randomFlowIncompressible import randomFlowIncompressibleCase
    return _run(randomFlowIncompressibleCase, nx=48)


@pytest.fixture(scope='module')
def columnCollapseResult():
    from warpSPH.cases.columnCollapse import columnCollapseCase
    # Needs to run past the collapse impact (t ~ 0.4 at nx=32) for the wall
    # penetration watch to mean anything -- longer than the suite's usual 20,
    # so this bypasses `_run`'s fixed step count.
    with contextlib.redirect_stdout(io.StringIO()):
        return run(columnCollapseCase, nx=32, nSteps=260, progress=False)


@pytest.fixture(scope='module')
def randomFlowBoundedResult():
    from warpSPH.cases.randomFlowIncompressible import randomFlowIncompressibleCase
    # Historically detonated under the post-c637785 scheme -- at nx=96, t ~
    # 0.18 (~step 23 at this case's adaptive dt), past the suite's usual
    # STEPS=20 (t ~ 0.17), which is why Part 54's sampler-bug fix briefly
    # misread this case as passing. Part 56 landed the actual fix
    # (`modules/incompressible/incompressible.py`'s `_SHIFT_WALL_PRESSURE`
    # default), which holds every resolution in {32,48,56,64,96,128} for
    # 300-400 steps; 40 steps here (well past the old onset) is enough to
    # exercise it without slowing the suite down.
    with contextlib.redirect_stdout(io.StringIO()):
        return run(randomFlowIncompressibleCase, nx=96, params=dict(bounded=True),
                  nSteps=40, progress=False)


# --- Sod: compressible, energy conserving -----------------------------------

def test_sodConservesTotalEnergy(sodResult):
    """CompSPH is an energy-conserving discretisation: the kinetic/thermal
    exchange across the shock must not change the total."""
    energy = sodResult.series('totalEnergy')
    drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    assert drift < 1e-5, f'total energy drifted by {drift:.3e}'


def test_sodConvertsThermalEnergyIntoMotion(sodResult):
    """The tube starts at rest; the pressure jump has to set it moving."""
    kinetic = sodResult.series('kineticEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-12)
    assert kinetic[-1] > 0.0
    assert sodResult.series('thermalEnergy')[-1] < sodResult.series('thermalEnergy')[0]


def test_sodDoesNotDiverge(sodResult):
    assert not sodResult.diverged
    assert len(sodResult.trajectory) == STEPS + 1


# --- Sod in 2D and 3D: the same tube, sampled at equal mass ------------------

#: `nx` here is the dense side's count across its own half of the domain, so
#: these are ~500 (2D) and ~4000 (3D) particles -- the 3D one is the most
#: expensive test in the suite even at that size. `transverseSpacings` is left
#: at its default: it is a multiple of the particle spacing, so the minimum the
#: sampler will accept (~16 in 3D) does not fall with `nx`, and trimming it to
#: make the test cheaper only trips the periodic-image guard.
_SOD_ND = {'sod2d': dict(nx=20), 'sod3d': dict(nx=8)}


@pytest.fixture(scope='module', params=sorted(_SOD_ND))
def sodNDResult(request):
    from warpSPH.cases import sodND
    case = {'sod2d': sodND.sod2dCase, 'sod3d': sodND.sod3dCase}[request.param]
    return request.param, _run(case, **_SOD_ND[request.param])


def test_sodNDConservesTotalEnergyAndMoves(sodNDResult):
    """The 1D assertions, in the dimension the case actually runs in."""
    name, result = sodNDResult
    assert not result.diverged, f'{name} diverged'
    energy = result.series('totalEnergy')
    drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    assert drift < 1e-5, f'{name} total energy drifted by {drift:.3e}'
    kinetic = result.series('kineticEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-12)
    assert kinetic[-1] > 0.0, f'{name} never started moving'
    assert result.series('thermalEnergy')[-1] < result.series('thermalEnergy')[0]


def test_sodNDStaysUniformAcrossTheSlab(sodNDResult):
    """The transverse directions are periodic and the solution does not depend
    on them, so the spread of density at a given x measures how much the
    sampling and the periodic wrap are corrupting a 1D answer."""
    name, result = sodNDResult
    state = result.state.state
    x = state.positions[:, 0].detach().cpu().numpy()
    rho = state.densities.detach().cpu().numpy()
    # Away from the waves, where the answer is still exactly the initial state.
    quiet = np.abs(x) < 0.15
    assert quiet.sum() > 10
    spread = rho[quiet].std() / rho[quiet].mean()
    assert spread < 0.02, f'{name} density varies by {spread:.2%} across the slab'


@pytest.mark.parametrize('dim,nx', [(1, 100), (1, 800), (2, 20), (2, 100), (3, 10), (3, 40)])
def test_sodSamplingMatchesParticleMasses(dim, nx):
    """The point of `sodND`'s sampler. Equal *spacing* would leave the dense
    side's particles `rho_l/rho_r = 4` times heavier (a 75% mismatch); equal
    mass is what it is for, and 3D cannot be exact -- `4**(1/3)` is irrational
    -- so it is allowed a little slack, but not much."""
    from warpSPH.caseUtils import sodInitialState, sodSampling
    sampling = sodSampling(nx, dim, 2.0, 20 if dim > 1 else 1,
                           sodInitialState(p=1.0, rho=1.0, v=0.0),
                           sodInitialState(p=0.1795, rho=0.25, v=0.0))
    assert abs(sampling.massRatio - 1) < 0.05, (
        f'dim={dim} nx={nx}: masses differ by {abs(sampling.massRatio - 1):.2%}')
    assert sampling.anisotropy <= 1.25, (
        f'dim={dim} nx={nx}: cells stretched {sampling.anisotropy:.3f}:1')


def test_sodSamplingAgreesWithThe1DBuilder():
    """`sodND` at dim=1 must reproduce `buildSod1D`'s lattice, or the 2D/3D
    cases are not extruding the case the 1D one actually runs."""
    from warpSPH.caseUtils import sodInitialState, sodSampling
    left = sodInitialState(p=1.0, rho=1.0, v=0.0)
    right = sodInitialState(p=0.1795, rho=0.25, v=0.0)
    for nx in (100, 800):
        sampling = sodSampling(nx, 1, 2.0, 1, left, right)
        # buildSod1D: dx = (L/2)/nx on the left, samplingRatio=4 coarser on the
        # right, giving nx//4 particles over the same extent.
        assert sampling.dense[2] == pytest.approx(1.0 / nx)
        assert sampling.light[0] == nx // 4
        assert sampling.light[2] == pytest.approx(4.0 / nx)
        assert sampling.massRatio == pytest.approx(1.0)


def test_sodNDKeepsTheLightStateOutOfTheDenseBlock():
    """`buildSod1D` samples the light state as one block centred on the origin
    and pushes its halves outward, which leaves a particle at exactly x=0 --
    inside the *dense* state, carrying the light state's mass -- whenever the
    light count is odd (`nx=100` does it; the `nx=800` default does not).
    `sodND` lays the same state out as a wrapped periodic interval instead, so
    it cannot happen at any `nx`. Both are checked here: the 1D builder's
    behaviour is recorded rather than fixed, since the backprop notebook's
    numbers are tied to its exact output.
    """
    import io
    import contextlib

    import torch

    from warpSPH.caseUtils import buildSodND, sodInitialState
    from warpSPH.cases.sod import sodCase
    from warpSPH.runner import CaseSpec, buildContext

    def build(nx, dim):
        spec = CaseSpec(caseName=sodCase.name, scheme=sodCase.scheme,
                        params=dict(sodCase.params)).merged(**sodCase.defaults)
        spec = spec.merged(nx=nx, dim=dim, plot=False, store=False, quiet=True)
        ctx = buildContext(sodCase, spec)
        sodCase.configureScheme(ctx)
        with contextlib.redirect_stdout(io.StringIO()):
            oneD = sodCase.buildSystem(ctx).state
            nd = buildSodND(
                ctx.SimulationSystem, ctx.SimulationState,
                sodInitialState(p=1.0, rho=1.0, v=0.0),
                sodInitialState(p=0.1795, rho=0.25, v=0.0),
                ctx.param('gamma'), ctx.config, transverseSpacings=1, verbose=False).state
        return oneD, nd

    def strandedLightParticles(state):
        return int(((state.materials == 1) & (state.positions[:, 0].abs() < 0.5 - 1e-6)).sum())

    oneD, nd = build(nx=100, dim=1)
    assert strandedLightParticles(oneD) == 1, 'buildSod1D no longer strands one at x=0'
    assert strandedLightParticles(nd) == 0
    # Same particle count and the same set of masses, stray particle aside.
    assert nd.positions.shape[0] == oneD.positions.shape[0]
    assert torch.allclose(nd.masses.sort().values, oneD.masses.sort().values)


def test_adaptiveSupportDoesNotDependOnWhatRanBefore():
    """Owen's psi lookup table is sliced by dimension and cached, so building a
    2D case in between two 1D builds must not change the 1D answer.

    It used to: the cache was a single unkeyed global, so the first dimension a
    *process* touched won, and everything after it got supports relaxed against
    the wrong table -- silently, since a support radius has no obviously wrong
    value. Invisible until a case ran in two dimensions in one process, which
    nothing did before `sod3d`.
    """
    import io
    import contextlib

    from warpSPH.cases import sodND  # noqa: F401 - registers sod2d/sod3d
    from warpSPH.runner import CaseSpec, buildContext, getCase

    def supports(name, **overrides):
        case = getCase(name)
        spec = CaseSpec(caseName=case.name, scheme=case.scheme,
                        params=dict(case.params)).merged(**case.defaults)
        spec = spec.merged(plot=False, store=False, quiet=True, **overrides)
        ctx = buildContext(case, spec)
        case.configureScheme(ctx)
        with contextlib.redirect_stdout(io.StringIO()):
            return case.buildSystem(ctx).state.supports.max().item()

    before = supports('sod', nx=100)
    supports('sod2d', nx=20)
    after = supports('sod', nx=100)
    assert after == pytest.approx(before, rel=1e-9), (
        f'a 2D build in between changed the 1D supports: {before} -> {after}')


def test_sodNDRejectsASlabNarrowerThanItsKernel():
    """A slab under twice the support radius lets particles interact with their
    own periodic images, and nothing downstream would notice -- so the sampler
    has to refuse rather than return a quietly wrong initial condition."""
    from warpSPH.cases import sodND
    with pytest.raises(ValueError, match='periodic images'):
        _run(sodND.sod2dCase, nx=20, params=dict(transverseSpacings=4))


# --- the three compressible solvers, as comparison runs ----------------------

#: Total-energy drift each solver is allowed over 20 steps. CompSPH is
#: energy-conserving by construction and measures exactly 0; CRKSPH is
#: conservative to round-off. Monaghan is *not* an energy-conserving
#: discretisation -- its artificial viscosity and conductivity are dissipative
#: by design -- so it gets a loose bound that still catches a solver that has
#: stopped integrating anything.
_ENERGY_DRIFT = {'CompSPH': 1e-5, 'CRKSPH': 1e-4, 'Monaghan': 5e-3}


@pytest.fixture(scope='module', params=sorted(_ENERGY_DRIFT))
def compressibleSchemeResult(request):
    """The Sod tube under each solver `--scheme` can select.

    These exist because **Monaghan was broken and nothing noticed**: it called
    all three boundary-condition helpers with their pre-`t` argument list, and
    `computeMomentumConsistent` with a `supportScheme=` keyword the function no
    longer takes. Every compressible case defaults to CRKSPH or CompSPH, so the
    only way to reach it was `--scheme Monaghan`, which no test did. Any solver
    reachable from the command line is now exercised here.
    """
    from warpSPH.cases.sod import sodCase
    return request.param, _run(sodCase, nx=200, scheme=request.param)


def test_everyCompressibleSolverRuns(compressibleSchemeResult):
    scheme, result = compressibleSchemeResult
    assert not result.diverged, f'{scheme} diverged'
    assert len(result.trajectory) == STEPS + 1


def test_everyCompressibleSolverConvertsThermalEnergyIntoMotion(compressibleSchemeResult):
    """The shared physics: the tube starts at rest and the pressure jump moves
    it, whichever discretisation is doing the integrating."""
    scheme, result = compressibleSchemeResult
    kinetic = result.series('kineticEnergy')
    thermal = result.series('thermalEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-12)
    assert kinetic[-1] > 0.0, f'{scheme} never started moving'
    assert thermal[-1] < thermal[0], f'{scheme} did not convert thermal energy'


def test_everyCompressibleSolverKeepsEnergyDriftInBounds(compressibleSchemeResult):
    scheme, result = compressibleSchemeResult
    energy = result.series('totalEnergy')
    drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    assert drift < _ENERGY_DRIFT[scheme], (
        f'{scheme} total energy drifted by {drift:.3e}')


def test_theSolverIsSelectedByTheSchemeFlag():
    """`--scheme` has to reach `buildScheme`, or a comparison run would
    silently compare a scheme against itself."""
    from warpSPH.cases.sod import sodCase
    from warpSPH.runner import buildContext
    from warpSPH.runner.caseSpec import CaseSpec

    expected = {'CompSPH': 'compSPH_step', 'CRKSPH': 'crkSPH_step',
                'Monaghan': 'compressibleSPH_Monaghan'}
    for scheme, stepFunction in expected.items():
        spec = CaseSpec(caseName=sodCase.name, scheme=scheme,
                        params=dict(sodCase.params)).merged(**sodCase.defaults)
        ctx = buildContext(sodCase, spec)
        assert ctx.stepFunction.__name__ == stepFunction, scheme


# --- TGV: incompressible, viscously decaying --------------------------------

def test_tgvKineticEnergyDecaysAtRoughlyTheAnalyticRate(tgvResult):
    """`KE(t) = KE(0) exp(-4 nu k^2 t)` for the Taylor-Green vortex.

    The measured rate sits near 0.55-0.6x the analytic one and is *stable under
    refinement* (0.605 at nx=32/20 steps, 0.564 at nx=32/50, 0.550 at
    nx=64/200), so it is not discretisation error.

    It is the Monaghan switch in the diffusion operator: viscosity is
    deactivated for particle pairs that are separating, so only the approaching
    half of the pairs dissipates at any instant and the effective viscosity is
    roughly half the prescribed nu. This is expected SPH behaviour, not a bug.
    Disabling the switch does recover the analytic decay rate, at the cost of
    stability elsewhere in the simulation, so it stays on.

    The band is therefore wide on purpose: it catches viscosity being
    disconnected (rate -> 0) or mis-scaled, without pretending the ~0.55 factor
    is an error to be driven out.
    """
    from warpSPH.cases.tgv import analyticDecayRate

    kinetic = tgvResult.series('kineticEnergy')
    time = tgvResult.series('t')
    measured = -np.polyfit(time, np.log(kinetic), 1)[0]
    analytic = analyticDecayRate(tgvResult.ctx)

    assert measured / analytic == pytest.approx(0.6, rel=0.45)


def test_tgvKineticEnergyIsMonotoneDecreasing(tgvResult):
    kinetic = tgvResult.series('kineticEnergy')
    assert np.all(np.diff(kinetic) < 0)


def test_tgvDoesNotDiverge(tgvResult):
    assert not tgvResult.diverged


# --- Shear wave: an exact solution with a constant pressure -----------------

def test_shearWaveHoldsItsAmplitude(shearWaveResult):
    """At `nu = 0` the analytic answer is that nothing happens.

    `u_x = u0 sin(k_w y)`, `u_y = 0` makes both nonlinear terms vanish
    identically -- `(u . grad) u = u_x d_x u_x e_x = 0` since `u_x` depends only
    on `y` -- so the exact solution is stationary and its pressure is constant.
    Any loss of amplitude is therefore numerical dissipation and nothing else,
    which is the property the case exists to measure (`cases/shearWave.py`).

    The band is loose because this is 20 steps at nx=32: the point is to catch
    the amplitude *collapsing* or *growing*, i.e. the pressure solve feeding
    energy into or out of a flow it should not touch at all. Production
    resolutions hold it far tighter -- see `DFSPH_IMPROVEMENT_PLAN.md` Part 16.
    """
    amplitude = shearWaveResult.series('amplitudeRatio')
    assert amplitude[0] == pytest.approx(1.0, abs=1e-3)
    assert np.all(np.abs(np.array(amplitude) - 1.0) < 0.05)


def test_shearWaveKeepsTheExactSolutionsZeroTransverseVelocity(shearWaveResult):
    """`u_y = 0` exactly, for all time. What appears there is solver artifact."""
    transverse = shearWaveResult.series('transverseVelocity')
    assert transverse[0] == pytest.approx(0.0, abs=1e-12)
    assert max(transverse) < 0.1


def test_shearWaveStaysIncompressible(shearWaveResult):
    maxDensity = max(row['maxDensity'] for row in shearWaveResult.trajectory)
    minDensity = min(row['minDensity'] for row in shearWaveResult.trajectory)
    assert 0.99 < minDensity <= maxDensity < 1.01


def test_shearWaveDoesNotDiverge(shearWaveResult):
    assert not shearWaveResult.diverged


# --- Dam break: weakly compressible, gravity driven -------------------------

def test_dambreakStaysWeaklyCompressible(dambreakResult):
    """The defining property of the scheme: density stays within ~1% of rho0."""
    maxDensity = max(row['maxDensity'] for row in dambreakResult.trajectory)
    minDensity = min(row['minDensity'] for row in dambreakResult.trajectory)
    assert 0.99 < minDensity <= maxDensity < 1.01


def test_dambreakGravityDoesWorkOnTheFluid(dambreakResult):
    """The column starts at rest and is released; kinetic energy has to grow."""
    kinetic = dambreakResult.series('kineticEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(kinetic) > 0)


def test_dambreakDoesNotDiverge(dambreakResult):
    assert not dambreakResult.diverged


# --- Incompressible (DFSPH) regressions the suite did not cover -------------
#
# `kolmogorovIncompressible` and `randomFlowIncompressible` regressed in the
# `c637785` rewrite and were caught only by hand (`DFSPH_IMPROVEMENT_PLAN.md`
# "Immediate", Parts 46-48). They live here so that class of silent regression
# fails the pre-commit check next time.

def test_kolmogorovIncompressibleForcingDrivesTheFlow(kolmogorovIncompressibleResult):
    """The case starts from rest; its body force has to spin the flow up.

    `c637785` left `computeForcing(...) * 0` in `divergenceFree_step`, which made the
    forced case completely inert (`KE == 0` for all time). This asserts the
    forcing term is actually wired in -- kinetic energy has to climb well off
    zero over the short run.
    """
    kinetic = kolmogorovIncompressibleResult.series('kineticEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-12)
    assert kinetic[-1] > 0.1
    assert np.all(np.diff(kinetic) >= 0)


def test_kolmogorovIncompressibleStaysIncompressible(kolmogorovIncompressibleResult):
    maxDensity = max(row['maxDensity'] for row in kolmogorovIncompressibleResult.trajectory)
    minDensity = min(row['minDensity'] for row in kolmogorovIncompressibleResult.trajectory)
    assert 0.98 < minDensity <= maxDensity < 1.02


def test_randomFlowPeriodicDoesNotDiverge(randomFlowPeriodicResult):
    """The periodic decaying random flow holds -- KE decays, does not run away."""
    assert not randomFlowPeriodicResult.diverged
    kinetic = randomFlowPeriodicResult.series('kineticEnergy')
    assert np.all(np.isfinite(kinetic))
    assert kinetic[-1] < 1.1 * kinetic[0]


def test_randomFlowPeriodicStaysIncompressible(randomFlowPeriodicResult):
    maxDensity = max(row['maxDensity'] for row in randomFlowPeriodicResult.trajectory)
    minDensity = min(row['minDensity'] for row in randomFlowPeriodicResult.trajectory)
    assert 0.97 < minDensity <= maxDensity < 1.03


def test_columnCollapseWallHoldsOnImpact(columnCollapseResult):
    """The released column collapses into the far wall; the mDBC no-penetration
    shift (`divergenceFree.NOPEN_SHIFT`, restored Part 49) has to keep fluid out of the
    wall band. `c637785` had commented the shift's call out of `divergenceFree_step`,
    leaving the pressure projection alone -- not enough for the impact, which
    then put ~6 particles a full spacing past the wall. Asserts the run holds
    and almost nothing crosses.
    """
    assert not columnCollapseResult.diverged
    npen = np.array([row['nPenetrating'] for row in columnCollapseResult.trajectory])
    pdx = np.array([row['maxPenetrationDx'] for row in columnCollapseResult.trajectory])
    assert np.all(np.isfinite(columnCollapseResult.series('kineticEnergy')))
    assert npen.max() <= 2
    assert pdx.max() < 1.0


@pytest.fixture(scope='module')
def bandColumnResult():
    from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase
    # Long enough for the hydrostatic gradient to build and for the nx=128
    # spray failure (see test_band2018pbColumnStaysQuiescent) to have shown up.
    with contextlib.redirect_stdout(io.StringIO()):
        return run(hydrostaticColumnCase, nx=64, nSteps=200,
                   scheme='band2018pb', progress=False,
                   integrationScheme='semiImplicitEuler')


@pytest.fixture(scope='module')
def bandBoundedResult():
    from warpSPH.cases.randomFlowIncompressible import randomFlowIncompressibleCase
    with contextlib.redirect_stdout(io.StringIO()):
        return run(randomFlowIncompressibleCase, nx=64, nSteps=120,
                   scheme='band2018pb', params=dict(bounded=True),
                   progress=False, integrationScheme='semiImplicitEuler')


def test_band2018pbColumnStaysQuiescent(bandColumnResult):
    """`band2018pb` (extended PPE, boundary samples as pressure unknowns) has
    to hold a quiescent hydrostatic column: near rest, near rest density, and
    the correct hydrostatic gradient.

    Guards the `bandRelaxation` scaling fix. The paper's per-sample
    `omega_i = 0.5 V0_i/h^d` scales every row by one base constant; this
    codebase detunes that base to `OMEGA_FLUID` at `n_h = 4`, and applying the
    detune to the fluid only (leaving boundary rows at a hardcoded 0.5) gave
    them a Jacobi step ~100x the fluid's -- boundary pressure ran away, Eq. 8
    turned it into `|a_p| ~ 1e3` on the near-wall fluid and the column was
    kicked apart at nx=128 (`pressureSlopeRatio` 0.02, `embeddedMinDensity`
    0.14, 1695 spray particles).
    """
    assert not bandColumnResult.diverged
    kinetic = bandColumnResult.series('kineticEnergy')
    assert np.all(np.isfinite(kinetic))
    last = bandColumnResult.trajectory[-1]
    assert last['maxVelocity'] < 0.5, 'column is not quiescent'
    assert 0.95 < last['embeddedMinDensity'] <= last['maxDensity'] < 1.05
    assert 0.9 < last['pressureSlopeRatio'] < 1.1, 'wrong hydrostatic gradient'


def test_band2018pbBoundedRandomFlowDecays(bandBoundedResult):
    """The closed-box sheared case that defeats every other incompressible
    scheme here (`divergenceFree` NaNs by step ~20, `iisph` and
    `omniIncompressible` detonate -- see `test_randomFlowBoundedDoesNotDiverge`
    below) is held by `band2018pb`, and the flow *decays* as a viscous random
    flow should.

    Guards `CLOSED_DOMAIN_GAUGE`. Eq. 8 is a summation gradient, so a uniform
    `p` accelerates nothing where the kernel support is complete: in a fully
    enclosed domain the constant is a null mode of `A`, and the Eq. 18
    `max(., 0)` clamp -- which can only push a row up -- ratchets it away
    instead of pinning it (step 1 solved to `p in [1.6e3, 2.6e3]`, a pure
    offset, and KE reached 1e8 by step 200 with the density still at
    [0.992, 1.007]).
    """
    assert not bandBoundedResult.diverged
    kinetic = bandBoundedResult.series('kineticEnergy')
    assert np.all(np.isfinite(kinetic))
    assert kinetic[-1] < kinetic[0], 'closed-box random flow should decay'
    velocity = np.array(bandBoundedResult.series('maxVelocity'))
    assert velocity.max() < 5.0, f'velocity spike: {velocity.max():.3g}'
    last = bandBoundedResult.trajectory[-1]
    assert 0.97 < last['minDensity'] <= last['maxDensity'] < 1.03


def test_randomFlowBoundedDoesNotDiverge(randomFlowBoundedResult):
    """Was `xfail(strict)` through Parts 48/54/55 -- see the fixture's own
    comment. Part 56 found the actual mechanism and fixed it: the composed
    constant-density Jacobi (`modules/incompressible/incompressible.py`
    `solveIncompressible`, the VD+PS shift's own solve) never re-derived the
    `kind==1` wall pressure from the current fluid iterate, so the near-wall
    iteration matrix carried a wall term the operator itself did not -- the
    same non-contraction Part 41 diagnosed and fixed for `omniIncompressible`,
    just never previously pointed at this solve/case combination.
    `_SHIFT_WALL_PRESSURE = 'shepard'` (now the default) recomputes it every
    Jacobi sweep via `wallPressureExtrapolation`; measured to hold cleanly
    (finite KE, monotone-ish decay, no `_CLOSED_DOMAIN_GAUGE` needed) at every
    resolution in {32,48,56,64,96,128} for 300-400 steps each -- a real fix,
    not the ~20-140x *delay* the `_CLOSED_DOMAIN_GAUGE` mean-centering
    experiment (Part 55) produced on its own. `band2018pb`
    (`test_band2018pbBoundedRandomFlowDecays`) also holds this case,
    independently.
    """
    assert not randomFlowBoundedResult.diverged
    kinetic = randomFlowBoundedResult.series('kineticEnergy')
    assert np.all(np.isfinite(kinetic))
    assert kinetic[-1] < 1.1 * kinetic[0]
    maxDensity = max(row['maxDensity'] for row in randomFlowBoundedResult.trajectory)
    minDensity = min(row['minDensity'] for row in randomFlowBoundedResult.trajectory)
    assert 0.85 < minDensity <= maxDensity < 1.15


# --- Sedov-Taylor: point energy deposit, one case run at dim 1/2/3 ----------

#: `nx` is the per-dimension particle count (`nx**dim` total), so these keep
#: 2D/3D in the same rough budget as 1D despite the extra dimension(s). Odd,
#: to satisfy `'hat'`/`'singular'`'s "particle exactly at the origin" need
#: (`buildSedov` itself bumps an even `nx` up by one; picking odd here just
#: avoids the warning).
_SEDOV = {'sedov1d': dict(dim=1, nx=51), 'sedov2d': dict(dim=2, nx=21), 'sedov3d': dict(dim=3, nx=11)}


@pytest.fixture(scope='module', params=sorted(_SEDOV))
def sedovResult(request):
    from warpSPH.cases.sedov import sedovCase
    return request.param, _run(sedovCase, **_SEDOV[request.param])


def test_sedovConservesTotalEnergy(sedovResult):
    """CRKSPH is an energy-conserving discretisation: the point deposit
    converting to a blast wave must not change the total."""
    name, result = sedovResult
    energy = result.series('totalEnergy')
    drift = abs(energy[-1] - energy[0]) / abs(energy[0])
    assert drift < 1e-3, f'{name} total energy drifted by {drift:.3e}'


def test_sedovConvertsThermalEnergyIntoMotion(sedovResult):
    """The medium starts at rest; the point deposit has to set it moving."""
    name, result = sedovResult
    kinetic = result.series('kineticEnergy')
    assert kinetic[0] == pytest.approx(0.0, abs=1e-8)
    assert kinetic[-1] > 0.0, f'{name} never started moving'


def test_sedovDoesNotDiverge(sedovResult):
    name, result = sedovResult
    assert not result.diverged, f'{name} diverged'
    assert len(result.trajectory) == STEPS + 1


@pytest.mark.parametrize('initialization,dim,nx', [
    ('hat', 1, 51), ('hat', 2, 21), ('hat', 3, 11),
    ('singular', 1, 51), ('quadrant', 1, 50),
])
def test_sedovInitialConditionConservesE0(initialization, dim, nx):
    """Regression test for `'hat'`: it used to raise `NotImplementedError`
    (`buildSedov` called `warpKernelToDiffSPHKernel`/`diffSPHKernel`, names
    left over from the pre-warp stack that no longer exist). The fix deposits
    E0 on the particle nearest the origin -- same as `'singular'` -- then
    smooths it with one SPH interpolation pass over the finalized adaptive
    supports, renormalized because that pass is not an exact partition of
    unity next to so few neighbours. All three initializations must still
    conserve E0 exactly at t=0, regardless of dimension.
    """
    from warpSPH.cases.sedov import sedovCase
    from warpSPH.runner import CaseSpec, buildContext

    spec = CaseSpec(caseName=sedovCase.name, scheme=sedovCase.scheme,
                    params=dict(sedovCase.params)).merged(**sedovCase.defaults)
    spec = spec.merged(dim=dim, nx=nx, plot=False, store=False, quiet=True,
                       params=dict(initialization=initialization))
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = buildContext(sedovCase, spec)
        sedovCase.configureScheme(ctx)
        state = sedovCase.buildSystem(ctx).initializeNewState().state

    totalEnergy = (state.internalEnergies * state.masses).sum().item()
    assert totalEnergy == pytest.approx(spec.param('E0'), rel=1e-3)


def test_sedovHatSpreadsTheSpikeOverMoreThanOneParticle():
    """The point of the fix: `'hat'` must not still be a single-particle
    delta the way `'singular'` deliberately is."""
    from warpSPH.cases.sedov import sedovCase
    from warpSPH.runner import CaseSpec, buildContext

    def nonzeroParticles(initialization):
        spec = CaseSpec(caseName=sedovCase.name, scheme=sedovCase.scheme,
                        params=dict(sedovCase.params)).merged(**sedovCase.defaults)
        spec = spec.merged(dim=1, nx=101, plot=False, store=False, quiet=True,
                           params=dict(initialization=initialization))
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = buildContext(sedovCase, spec)
            sedovCase.configureScheme(ctx)
            state = sedovCase.buildSystem(ctx).initializeNewState().state
        return int((state.internalEnergies > 0).sum())

    assert nonzeroParticles('singular') == 1
    assert nonzeroParticles('hat') > 1


@pytest.mark.parametrize('dim,nx', [(1, 40), (2, 40), (3, 32)])
def test_uniformLatticeDensityMatchesBuiltDensity(dim, nx):
    """`sum_j m_j W_ij` over a uniform lattice of known density must return
    what it was built from -- `PORTING_EXAMPLES.md`'s recipe for catching a
    dimension-dependent kernel-normalisation bug, section 4.7.

    This is a real regression test, not a precaution: `buildSedov`'s B7
    density estimate came back at exactly 1/16 of `rho0` at dim=3 only (found
    while adding the 3D Sedov variant), tracing to a wrong `B7_C_d(3)`
    constant in the sibling `warpSPHCore` repo -- the same "16x too small"
    bug `PORTING_EXAMPLES.md` already documented as found and fixed via Sod's
    3D porting, evidently not on this code path (`sampleRegularParticles` +
    plain `WarpOperation.Density`, which only a dim=3 case exercises; Sod's
    own 3D IC uses a different, bespoke sampler). Fixed in `warpSPHCore`
    (uncommitted there -- a separate repo -- flag to whoever finds this
    failing again rather than re-deriving the diagnosis from scratch).
    """
    import torch

    from warpSPH.cases.sedov import sedovCase
    from warpSPH.runner import CaseSpec, buildContext
    from warpSPH.sample import sampleRegularParticles
    from warpSPHCore import (OperationProperties, WarpOperation, SupportScheme,
                             GradientScheme, warpOperation)

    spec = CaseSpec(caseName=sedovCase.name, scheme=sedovCase.scheme,
                    params=dict(sedovCase.params)).merged(**sedovCase.defaults)
    spec = spec.merged(dim=dim, nx=nx, L=2.0, plot=False, store=False, quiet=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = buildContext(sedovCase, spec)
        sedovCase.configureScheme(ctx)
        particles_ = sampleRegularParticles(nx, ctx.config.domain, ctx.config.targetNeighbors)
        particles = ctx.SimulationState(
            positions=particles_.positions, supports=particles_.supports,
            masses=particles_.masses, densities=particles_.densities,
            velocities=torch.zeros_like(particles_.positions),
            kinds=torch.zeros_like(particles_.positions[:, 0], dtype=torch.int32),
            materials=torch.zeros_like(particles_.positions[:, 0], dtype=torch.int32),
            UIDs=torch.arange(particles_.positions.shape[0], device=ctx.device, dtype=torch.int32),
            UIDcounter=particles_.positions.shape[0],
            internalEnergies=None, totalEnergies=None, entropies=None,
            pressures=None, soundspeeds=None,
            divergence=torch.zeros_like(particles_.densities),
            alpha0s=torch.ones_like(particles_.densities),
            alphas=torch.ones_like(particles_.densities),
        )
        densities = warpOperation(
            particles,
            OperationProperties(kernel=ctx.config.kernel, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, gradientMode=GradientScheme.Difference),
            domain=ctx.config.domain,
        )

    assert densities.mean().item() == pytest.approx(1.0, rel=1e-3), (
        f'dim={dim} nx={nx}: uniform-lattice B7 density estimate is '
        f'{densities.mean().item():.6g}, not 1.0 -- kernel normalisation regression')


# --- Wave equation: non-fluid demo scheme, absorbed at the boundary ---------

#: `nx` per dimension is chosen so the point/sphere source (`sourceRadius`
#: 0.15 by default) is resolved by more than a couple of particles --
#: important in 3D, where the particle count grows as `nx**3` and a too-coarse
#: `nx` can leave the source's support empty.
_WAVE = {'wave1d': dict(dim=1, nx=128), 'wave2d': dict(dim=2, nx=32), 'wave3d': dict(dim=3, nx=20)}

#: How much the discrete wave energy (kinetic + gradient-based potential, see
#: `cases/waveEquation.py`'s `diagnostics`) is allowed to grow over the run
#: relative to its initial value. Unlike the compressible schemes above, this
#: one is not energy-conserving by construction -- the border damping is
#: explicitly dissipative -- and the initial condition (a compactly-supported
#: kernel bump) is not smooth on the particle scale, so the first few steps
#: redistribute a real amount of energy into a form the diagnostic reads as
#: growth before damping starts winning. Measured growth over `STEPS` steps at
#: the resolutions above is ~1.2x (1D), ~2.9x (2D), ~11.5x (3D); the bound
#: below leaves headroom over that but still catches genuine blow-up -- an
#: unstable `cflFactor` drives this into the billions within a similar step
#: count.
_WAVE_ENERGY_DRIFT = 25.0


@pytest.fixture(scope='module', params=sorted(_WAVE))
def waveResult(request):
    from warpSPH.cases.waveEquation import waveEquationCase
    return request.param, _run(waveEquationCase, **_WAVE[request.param])


def test_waveEquationDoesNotDiverge(waveResult):
    name, result = waveResult
    assert not result.diverged, f'{name} diverged'
    assert len(result.trajectory) == STEPS + 1


def test_waveEquationEnergyStaysBounded(waveResult):
    name, result = waveResult
    energy = result.series('totalEnergy')
    assert np.all(np.isfinite(energy)), f'{name} produced a non-finite energy'
    drift = energy.max() / energy[0]
    assert drift < _WAVE_ENERGY_DRIFT, f'{name} total energy grew {drift:.2f}x'


def test_waveEquationRunsInEveryDimension(waveResult):
    """The payoff of generalizing the base pipeline to N-D
    (`docs/historic_plans/WAVE_EQUATION_PLAN.md` step 1): the same scheme code is exercised by the
    neighbour search and Laplacian operator in 1D, 2D and 3D, not just the
    2D shape-source path."""
    name, result = waveResult
    positions = result.state.state.positions
    assert positions.shape[1] == _WAVE[name]['dim']
