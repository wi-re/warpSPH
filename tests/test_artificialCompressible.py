"""The artificial-compressibility scheme (ACSPH_PLAN.md steps 4-5): the
family wiring, and the dual-time driver itself.

The driver's acceptance criterion is the one thing the scheme exists to do:
drive `div v -> 0` at time level n+1 by integrating a pressure equation in
pseudo-time. `test_theDualTimeLoopProjectsOutTheDivergence` measures exactly
that on a periodic box (no free surface, no walls, so nothing but the solve is
being graded), and `test_moreIterationsMeansLessDivergence` checks it is a
*convergent* iteration rather than a lucky single sweep.

The rest pins the parts that are easy to get quietly wrong: the exact-delta
hand-off (and the integrator that would silently break it), the BDF2
coefficients at unequal step sizes, and the `1/N` in the convergence metric.
"""

import pytest
import torch
from warpSPHIntegrators import getIntegrator
from warpSPHIntegrators.integration import IntegrationSchemeType

import warpSPH.schemes.artificialCompressible as acmod
from warpSPH.configurations import (ArtificialCompressibleSPHConfig,
                                    artificialCompressibleConfigToDict,
                                    dictToArtificialCompressibleConfig)
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.enumTypes import (ArtificialCompressibleSPHScheme,
                               PressureSmoothingScheme,
                               isArtificialCompressibleScheme)
from warpSPH.runner.caseSpec import schemeNames
from warpSPH.schemes import buildScheme
from warpSPH.enumTypes import PressureSmoothingScheme as _PSS
from warpSPH.systems.artificialCompressible import (ArtificialCompressibleState,
                                                    ArtificialCompressibleSystem,
                                                    bdfCoefficients)
from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         SupportScheme, WarpOperation, warpOperation)
from warpSPH.utils.domain import buildDomainDescription

N_PER_SIDE = 10


def buildSystem(device, dtype, integrationScheme=IntegrationSchemeType.forwardEuler):
    dx = 1.0 / N_PER_SIDE
    xs = (torch.arange(N_PER_SIDE, device=device, dtype=dtype) + 0.5) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing='ij')
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1).contiguous()
    n = positions.shape[0]

    config = SimulationConfig(
        device=device, dtype=dtype, dim=2, dx=dx, nx=N_PER_SIDE,
        integrationScheme=integrationScheme,
        domain=buildDomainDescription(l=4.0, dim=2, periodic=False,
                                      device=device, dtype=dtype))
    state = ArtificialCompressibleState(
        positions=positions,
        velocities=torch.zeros_like(positions),
        pressures=torch.zeros(n, device=device, dtype=dtype),
        supports=torch.full((n,), config.n_h * dx, device=device, dtype=dtype),
        masses=torch.full((n,), dx ** 2, device=device, dtype=dtype),
        densities=torch.ones(n, device=device, dtype=dtype),
        kinds=torch.zeros(n, dtype=torch.int32, device=device),
        materials=torch.zeros(n, dtype=torch.int32, device=device),
        UIDs=torch.arange(n, dtype=torch.int32, device=device),
        UIDcounter=n,
    )
    system = ArtificialCompressibleSystem(state=state, domain=config.domain)
    return system, config, ArtificialCompressibleSPHConfig()


@pytest.fixture
def built(runtime):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return buildSystem(device, torch.float32)


# --- registration ----------------------------------------------------------

def test_theSchemeIsRegisteredEverywhereTheOthersAre():
    bundle = buildScheme('artificialCompressible')
    assert bundle.SimulationState is ArtificialCompressibleState
    assert bundle.SimulationSystem is ArtificialCompressibleSystem
    assert bundle.stepFunction is acmod.artificialCompressible_step
    assert 'artificialCompressible' in schemeNames()
    assert isArtificialCompressibleScheme(
        ArtificialCompressibleSPHScheme.artificialCompressible)

    from warpSPH.io.importIO import schemeNameToSimulationScheme
    assert schemeNameToSimulationScheme('artificialCompressible') is \
        ArtificialCompressibleSPHScheme.artificialCompressible


def test_theConfigRoundTrips():
    config = ArtificialCompressibleSPHConfig()
    config.acParams.pressureSmoothing = PressureSmoothingScheme.jst
    config.acParams.uChar = 1.23
    config.acParams.rkStages = 4
    config.acParams.referenceSoundSpeedForViscosity = 20.0

    asDict = artificialCompressibleConfigToDict(config)
    back = dictToArtificialCompressibleConfig(asDict)

    assert back.acParams.pressureSmoothing is PressureSmoothingScheme.jst
    assert back.acParams.uChar == pytest.approx(1.23)
    assert back.acParams.rkStages == 4
    assert back.acParams.referenceSoundSpeedForViscosity == pytest.approx(20.0)
    assert artificialCompressibleConfigToDict(back) == asDict


def test_uCharAndTheReferenceSoundSpeedSurviveBeingNone():
    """Both are `Optional` for real reasons (see the config's docstring), so
    `None` has to make the round trip rather than becoming 0.0."""
    config = ArtificialCompressibleSPHConfig()
    assert config.acParams.uChar is None
    back = dictToArtificialCompressibleConfig(
        artificialCompressibleConfigToDict(config))
    assert back.acParams.uChar is None
    assert back.acParams.referenceSoundSpeedForViscosity is None


# --- BDF2 ------------------------------------------------------------------

def test_bdfCoefficientsReduceToTheFixedStepValues():
    dt = 0.01
    alpha, beta, gamma, order = bdfCoefficients(dt, dt)
    assert order == 2
    assert (alpha, beta, gamma) == pytest.approx((1.5 / dt, -2.0 / dt, 0.5 / dt))


def test_bdfCoefficientsFallBackToBdf1WithoutAHistory():
    dt = 0.01
    alpha, beta, gamma, order = bdfCoefficients(dt, None)
    assert order == 1
    assert (alpha, beta, gamma) == pytest.approx((1.0 / dt, -1.0 / dt, 0.0))


@pytest.mark.parametrize('dtPrev', [0.01, 0.004, 0.031])
def test_variableStepBdf2DifferentiatesAQuadraticExactly(dtPrev):
    """The defining property of Eq. (42): `alpha_t u^{n+1} + beta_t u^n +
    gamma_t u^{n-1}` must be the exact derivative at `t^{n+1}` for any
    polynomial up to degree 2, at any step ratio. This is the check that the
    variable-`dt` algebra is right -- the fixed-step limit alone would not
    catch a swapped `dt^n`/`dt^{n-1}`."""
    dt = 0.01
    alpha, beta, gamma, order = bdfCoefficients(dt, dtPrev)
    assert order == 2

    def u(t):
        return 3.0 - 2.0 * t + 5.0 * t ** 2

    def dudt(t):
        return -2.0 + 10.0 * t

    tNext = 0.7
    approx = alpha * u(tNext) + beta * u(tNext - dt) + gamma * u(tNext - dt - dtPrev)
    assert approx == pytest.approx(dudt(tNext), rel=1e-9)


# --- the integrator guard --------------------------------------------------

def test_aMultiStageIntegratorIsRefused(built):
    """The exact-delta contract only holds for a single-evaluation integrator;
    anything else would run the dual-time solve once per stage and blend the
    results. Refused at step entry, not warned about."""
    system, config, schemeConfig = built
    config.integrationScheme = IntegrationSchemeType.rungeKutta4
    with pytest.raises(ValueError, match='forwardEuler'):
        acmod.artificialCompressible_step(system, 1e-3, config, schemeConfig)


def test_forwardEulerIsAccepted(built):
    system, config, schemeConfig = built
    acmod.validateIntegrationScheme(config)  # does not raise


# --- the step itself -------------------------------------------------------

def test_theStepRunsThroughTheRealIntegrator(built):
    """One step through `getIntegrator(forwardEuler)` exactly as the runner
    drives it -- the point being that the state/update tag wiring
    (`pressure`/`pressure_derivative`) actually resolves."""
    system, config, schemeConfig = built
    integrator = getIntegrator(config.integrationScheme)
    result = integrator.function(state=system, f=acmod.artificialCompressible_step,
                                 dt=1e-3, config=config, verbose=False,
                                 schemeConfig=schemeConfig)
    assert torch.isfinite(result.state.state.positions).all()
    assert torch.isfinite(result.state.state.pressures).all()


def test_theStepRollsTheBdfHistory(built):
    """`finalize` must advance `u^n -> u^{n-1}`, so the second step is the
    first one that can use BDF2."""
    system, config, schemeConfig = built
    integrator = getIntegrator(config.integrationScheme)
    assert system.positionsPrev is None
    assert system.bdfCoefficients(1e-3)[3] == 1

    result = integrator.function(state=system, f=acmod.artificialCompressible_step,
                                 dt=1e-3, config=config, verbose=False,
                                 schemeConfig=schemeConfig)
    after = result.state
    assert after.positionsPrev is not None
    assert after.positionsPrev2 is None
    assert after.bdfCoefficients(1e-3)[3] == 1, 'one history entry is still BDF1'

    result = integrator.function(state=after, f=acmod.artificialCompressible_step,
                                 dt=1e-3, config=config, verbose=False,
                                 schemeConfig=schemeConfig)
    after2 = result.state
    assert after2.positionsPrev2 is not None
    assert after2.dtPrev == pytest.approx(1e-3)
    assert after2.bdfCoefficients(1e-3)[3] == 2


def test_theHistoryIsNotAliasedToTheLiveState(built):
    """`rollHistory` clones. If it did not, the "previous" positions would
    track the current ones and the BDF source would be identically zero --
    a failure mode that looks exactly like perfect convergence."""
    system, config, schemeConfig = built
    system.rollHistory(1e-3)
    system.state.positions += 1.0
    assert not torch.allclose(system.positionsPrev, system.state.positions)




# --- the dual-time driver --------------------------------------------------

PERIODIC_N = 24


def buildPeriodicBox(device, dtype, rho0=1.0):
    """A periodic lattice carrying a Taylor-Green field plus a deliberately
    *compressible* perturbation. Periodic on purpose: no free surface and no
    walls, so what these tests grade is the dual-time solve and nothing else."""
    L, n = 1.0, PERIODIC_N
    dx = L / n
    xs = (torch.arange(n, device=device, dtype=dtype) + 0.5) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing='ij')
    positions = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1).contiguous()
    N = positions.shape[0]

    config = SimulationConfig(
        device=device, dtype=dtype, dim=2, dx=dx, nx=n,
        integrationScheme=IntegrationSchemeType.forwardEuler,
        domain=buildDomainDescription(l=L, dim=2, periodic=True,
                                      device=device, dtype=dtype))
    k = 2.0 * torch.pi / L
    solenoidal = torch.stack([torch.sin(k * positions[:, 0]) * torch.cos(k * positions[:, 1]),
                              -torch.cos(k * positions[:, 0]) * torch.sin(k * positions[:, 1])], -1)
    compressive = 0.3 * torch.stack([torch.sin(k * positions[:, 0]),
                                     torch.sin(k * positions[:, 1])], -1)

    state = ArtificialCompressibleState(
        positions=positions, velocities=solenoidal + compressive,
        pressures=torch.zeros(N, device=device, dtype=dtype),
        supports=torch.full((N,), config.n_h * dx, device=device, dtype=dtype),
        masses=torch.full((N,), rho0 * dx ** 2, device=device, dtype=dtype),
        densities=torch.full((N,), rho0, device=device, dtype=dtype),
        kinds=torch.zeros(N, dtype=torch.int32, device=device),
        materials=torch.zeros(N, dtype=torch.int32, device=device),
        UIDs=torch.arange(N, dtype=torch.int32, device=device), UIDcounter=N)

    schemeConfig = ArtificialCompressibleSPHConfig()
    # No free surface here, so detection has nothing to find and only costs.
    schemeConfig.surfaceDetectionConfig.active = False
    schemeConfig.acParams.uChar = 1.0
    config.dt = 2e-3
    return ArtificialCompressibleSystem(state=state, domain=config.domain), \
        config, schemeConfig


def rmsDivergence(state, config):
    div = warpOperation(
        state,
        OperationProperties(kernel=config.kernel, operation=WarpOperation.Divergence,
                            supportMode=SupportScheme.SuperSymmetric,
                            operationMode=OperationDirection.AllToAll,
                            gradientMode=GradientScheme.Difference),
        queryValues=state.velocities, domain=config.domain)
    return float(div.pow(2).mean().sqrt())


@pytest.fixture(scope='module')
def periodicBox(runtime):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return buildPeriodicBox(device, torch.float32)


def runOneStep(system, config, schemeConfig, maxIterations):
    import copy
    local = ArtificialCompressibleSystem(state=copy.deepcopy(system.state),
                                         domain=config.domain)
    schemeConfig.acParams.maxPseudoIterations = maxIterations
    integrator = getIntegrator(config.integrationScheme)
    result = integrator.function(state=local, f=acmod.artificialCompressible_step,
                                 dt=config.dt, config=config, verbose=False,
                                 schemeConfig=schemeConfig)
    return result.state


def test_theDualTimeLoopProjectsOutTheDivergence(periodicBox):
    """The scheme's whole purpose: `div v -> 0` at time level n+1. Nothing else
    in the step can do this -- with the pressure equation not integrated, the
    velocity would be untouched."""
    system, config, schemeConfig = periodicBox
    before = rmsDivergence(system.state, config)
    assert before > 1.0, 'the initial field has to be genuinely compressible'

    after = rmsDivergence(runOneStep(system, config, schemeConfig, 200).state, config)
    assert after < before / 100.0, (
        f'rms(div v) only went {before:.3e} -> {after:.3e}')


def test_moreIterationsMeansLessDivergence(periodicBox):
    """A convergent iteration, not one lucky sweep: the residual has to fall
    monotonically with the iteration budget and then stop falling (converged),
    never rise."""
    system, config, schemeConfig = periodicBox
    residuals = [rmsDivergence(runOneStep(system, config, schemeConfig, n).state, config)
                 for n in (1, 10, 50, 200)]
    for earlier, later in zip(residuals, residuals[1:]):
        assert later <= earlier * 1.01, f'divergence rose: {residuals}'
    assert residuals[-1] < residuals[0] / 100.0


def test_thePressureIsWhatDoesTheProjection(periodicBox):
    """A sanity anchor on the mechanism: the loop must build a real pressure
    field. A step that left `p == 0` and still reported convergence would mean
    the metric, not the solve, is what is working."""
    system, config, schemeConfig = periodicBox
    after = runOneStep(system, config, schemeConfig, 200).state
    assert float(after.pressures.abs().max()) > 1.0
    assert torch.isfinite(after.pressures).all()


def test_theUpdateIsAnExactDelta(periodicBox):
    """The contract the `forwardEuler` guard exists to protect: applying
    `dt * dxdt` to `x^n` must reproduce the converged `x^{n+1}` exactly, since
    the step already did the whole advance."""
    system, config, schemeConfig = periodicBox
    schemeConfig.acParams.maxPseudoIterations = 25
    x0 = system.state.positions.clone()
    v0 = system.state.velocities.clone()
    p0 = system.state.pressures.clone()

    import copy
    local = ArtificialCompressibleSystem(state=copy.deepcopy(system.state),
                                         domain=config.domain)
    update, _, _ = acmod.artificialCompressible_step(local, config.dt, config,
                                                     schemeConfig)
    integrator = getIntegrator(config.integrationScheme)
    local2 = ArtificialCompressibleSystem(state=copy.deepcopy(system.state),
                                          domain=config.domain)
    stepped = integrator.function(state=local2, f=acmod.artificialCompressible_step,
                                  dt=config.dt, config=config, verbose=False,
                                  schemeConfig=schemeConfig).state.state

    assert torch.allclose(stepped.positions, x0 + config.dt * update.dxdt,
                          rtol=1e-5, atol=1e-6)
    assert torch.allclose(stepped.velocities, v0 + config.dt * update.dvdt,
                          rtol=1e-5, atol=1e-6)
    assert torch.allclose(stepped.pressures, p0 + config.dt * update.dpdt,
                          rtol=1e-5, atol=1e-4)


def test_higherRkOrderBuysNoAccuracy(periodicBox):
    """The paper's Sec. 4.3 finding, reproduced qualitatively: pseudo-time RK
    order does not improve the converged answer (the BDF2 sets the accuracy),
    while cost rises linearly in stages. Recorded as a test because it is the
    reason `rkStages` defaults to 2 -- if a future change made RK4 genuinely
    better here, that would be worth knowing about."""
    system, config, schemeConfig = periodicBox
    results = {}
    for stages in (2, 3, 4):
        schemeConfig.acParams.rkStages = stages
        results[stages] = rmsDivergence(
            runOneStep(system, config, schemeConfig, 60).state, config)
    schemeConfig.acParams.rkStages = 2
    assert results[4] > results[2] / 10.0, (
        f'RK4 gained more than an order over RK2, which contradicts Sec. 4.3: {results}')


# --- parameters and the convergence metric ---------------------------------

def test_acParametersFollowEquation24(periodicBox):
    """`dtau = dt/R`, `beta = CFL_tau h / dtau`, `k1 = beta^2`,
    `k2 = k2Factor h beta` -- with `h` the paper's smoothing length,
    `supports / xi`, not this repo's support radius."""
    from warpSPHCore import sphKernel_xi
    system, config, schemeConfig = periodicBox
    dt = 1e-3
    dtau, beta, k1, k2, nu = acmod.acParameters(system.state, config, schemeConfig, dt)

    assert dtau == pytest.approx(dt / schemeConfig.acParams.dtOverDtau)
    h = system.state.supports / sphKernel_xi(config.kernel.value, config.dim)
    assert torch.allclose(beta, schemeConfig.acParams.cflTau * h / dtau)
    assert torch.allclose(k1, beta ** 2)
    assert torch.allclose(k2, schemeConfig.acParams.k2Factor * h * beta)


def test_theConvergenceMetricUsesOneOverNNotOneOverSqrtN(periodicBox):
    """Eq. (48) as printed -- and it matters. With a per-particle residual `c`,
    `|tilde v|_2 = c sqrt(N)`, so `eps_v = log10(c / U_eps) - 0.5 log10(N)`:
    doubling the particle count at the *same* per-particle residual lowers
    `eps_v` by `0.5 log10(2)`, i.e. a fixed target is a stricter tolerance at
    higher resolution (about 0.6 of a decade across the paper's own
    `L/dx = 200 -> 800` sweep). An RMS -- `1/sqrt(N)` -- would make it
    resolution-independent instead, which is exactly the "cleanup" this test
    exists to block. Reproduced verbatim because it is what the paper's numbers
    mean (ACSPH_PLAN.md Sec. 1.6)."""
    system, config, schemeConfig = periodicBox
    device = system.state.positions.device

    def epsFor(n):
        tildeV = torch.full((n, 2), 1e-3, device=device)
        velocities = torch.zeros((n, 2), device=device)
        fluid = torch.ones(n, dtype=torch.bool, device=device)
        return acmod.convergenceMetric(tildeV, velocities, fluid, schemeConfig)

    import math
    drift = epsFor(2000) - epsFor(1000)
    assert drift == pytest.approx(-0.5 * math.log10(2.0), abs=1e-5)
    assert drift != pytest.approx(0.0, abs=1e-3), 'this would be an RMS metric'


def test_uCharCapsTheNormalisation(periodicBox):
    """`U_eps = max(min(|v|_max, U_char), eps_s)`: `U_char` is a cap on the
    measured velocity, not a replacement for it."""
    system, config, schemeConfig = periodicBox
    device = system.state.positions.device
    tildeV = torch.full((100, 2), 1e-3, device=device)
    fast = torch.full((100, 2), 10.0, device=device)
    fluid = torch.ones(100, dtype=torch.bool, device=device)

    schemeConfig.acParams.uChar = None
    uncapped = acmod.convergenceMetric(tildeV, fast, fluid, schemeConfig)
    schemeConfig.acParams.uChar = 1.0
    capped = acmod.convergenceMetric(tildeV, fast, fluid, schemeConfig)
    assert capped > uncapped, 'a smaller U_eps must give a LARGER (worse) eps_v'


# --- what is deliberately not implemented yet ------------------------------

@pytest.mark.parametrize('field,value', [
    ('useTildeVAdvection', True),
    ('shiftInsidePseudoLoop', True),
    ('k3', 1.0),
])
def test_theUnimplementedOptionsRaiseRatherThanNoOp(periodicBox, field, value):
    """Every one of these is a paper option the paper itself switched off. They
    exist as config fields so the choice is visible -- and they raise, because
    a flag that silently does nothing is worse than one that is absent."""
    system, config, schemeConfig = periodicBox
    original = getattr(schemeConfig.acParams, field)
    setattr(schemeConfig.acParams, field, value)
    try:
        with pytest.raises(NotImplementedError, match='ACSPH_PLAN'):
            acmod.artificialCompressible_step(system, config.dt, config, schemeConfig)
    finally:
        setattr(schemeConfig.acParams, field, original)


@pytest.mark.parametrize('scheme', [_PSS.biharmonic, _PSS.jst])
def test_theUnimplementedSmoothingOperatorsRaise(periodicBox, scheme):
    system, config, schemeConfig = periodicBox
    original = schemeConfig.acParams.pressureSmoothing
    schemeConfig.acParams.pressureSmoothing = scheme
    try:
        with pytest.raises(NotImplementedError, match='step 8'):
            acmod.artificialCompressible_step(system, config.dt, config, schemeConfig)
    finally:
        schemeConfig.acParams.pressureSmoothing = original


def test_ac2AndAc2lAreBothAvailableAndDifferent(periodicBox):
    """AC-2 (Eq. 32) and AC-2L (Eqs. 33-34) both run; they had better not be
    the same operator, since separating them is what Sec. 4.1.1 is for."""
    system, config, schemeConfig = periodicBox
    original = schemeConfig.acParams.pressureSmoothing
    try:
        schemeConfig.acParams.pressureSmoothing = _PSS.renormalizedBiLaplacian
        ac2l = runOneStep(system, config, schemeConfig, 40).state.pressures.clone()
        schemeConfig.acParams.pressureSmoothing = _PSS.laplacian
        ac2 = runOneStep(system, config, schemeConfig, 40).state.pressures.clone()
    finally:
        schemeConfig.acParams.pressureSmoothing = original
    assert torch.isfinite(ac2).all() and torch.isfinite(ac2l).all()
    assert not torch.allclose(ac2, ac2l, rtol=1e-3)


# --- the Eq. (46) timestep -------------------------------------------------

def test_theTimestepDispatcherRoutesAnAcsphSystem(periodicBox):
    """`modules/timestep/wrapper.py` used to send anything that was not a
    `WeaklyCompressibleSystem` to the *compressible* branch, which would build
    `dt` from a sound speed this scheme does not have."""
    from warpSPH.modules.timestep import computeTimestep
    system, config, schemeConfig = periodicBox
    dt = computeTimestep(system, config, schemeConfig, dt=1e-3)
    assert float(dt) > 0.0
    assert torch.isfinite(torch.as_tensor(dt))


def test_theTimestepIsAdvectiveNotAcoustic(periodicBox):
    """Halving the velocity field must roughly double the advective limit --
    `CFL_t h / |v|_max`. An acoustic constraint would not move at all."""
    from warpSPH.modules.timestep.artificialCompressible import computeTimestep
    import copy
    system, config, schemeConfig = periodicBox
    schemeConfig = copy.deepcopy(schemeConfig)
    schemeConfig.dt_viscosityConstraint = False   # isolate the advective term
    config = copy.copy(config)
    config.maxDt = 1.0

    fast = ArtificialCompressibleSystem(state=copy.deepcopy(system.state),
                                        domain=config.domain)
    slow = ArtificialCompressibleSystem(state=copy.deepcopy(system.state),
                                        domain=config.domain)
    slow.state.velocities = slow.state.velocities * 0.5

    dtFast = float(computeTimestep(fast, config, schemeConfig, dt=None))
    dtSlow = float(computeTimestep(slow, config, schemeConfig, dt=None))
    assert dtSlow == pytest.approx(2.0 * dtFast, rel=1e-4)


def test_theStepRatioIsClampedInBothDirections(periodicBox):
    """Eq. (46)'s `[0.8, 1.2] x dt^{n-1}`. The symmetric part is the point: it
    exists to protect BDF2 accuracy, so a *shrink* is bounded too -- which
    `weaklyCompressible.py`'s growth-only clamp does not do."""
    from warpSPH.modules.timestep.artificialCompressible import (
        STEP_RATIO_BOUNDS, computeTimestep)
    import copy
    system, config, schemeConfig = periodicBox
    config = copy.copy(config)
    config.minDt, config.maxDt = 1e-12, 1.0
    shrink, grow = STEP_RATIO_BOUNDS

    tiny = float(computeTimestep(system, config, schemeConfig, dt=1e-9))
    assert tiny == pytest.approx(grow * 1e-9, rel=1e-5), 'growth not clamped'

    huge = float(computeTimestep(system, config, schemeConfig, dt=1.0))
    assert huge == pytest.approx(shrink * 1.0, rel=1e-5), 'shrink not clamped'


def test_aFixedTimestepIsLeftAlone(periodicBox):
    from warpSPH.modules.timestep.artificialCompressible import computeTimestep
    import copy
    system, config, schemeConfig = periodicBox
    config = copy.copy(config)
    config.adaptiveDt = False
    assert computeTimestep(system, config, schemeConfig, dt=1.234e-3) == 1.234e-3


def test_aCflAboveTheCeilingWarnsRatherThanPassingSilently(periodicBox, capsys):
    """Tables 1-2 measure a sharp accuracy cliff above `CFL_t = 0.4`. It is a
    ceiling, not a guideline, so exceeding it has to be visible."""
    from warpSPH.modules.timestep import artificialCompressible as tsmod
    import copy
    system, config, schemeConfig = periodicBox
    schemeConfig = copy.deepcopy(schemeConfig)
    schemeConfig.acParams.cflT = 0.9
    tsmod._warnedCfl = False
    try:
        tsmod.computeTimestep(system, config, schemeConfig, dt=1e-3)
        assert 'CFL_t' in capsys.readouterr().out
    finally:
        tsmod._warnedCfl = False
