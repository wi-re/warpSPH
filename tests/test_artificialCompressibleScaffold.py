"""The artificial-compressibility scheme family, wired end to end but not yet
doing physics (ACSPH_PLAN.md step 4).

`schemes/artificialCompressible.py` is a scaffold: it builds the neighbourhood,
enforces the BCs, detects the free surface, and then returns a **zero update**.
These tests pin the parts that are finished -- the state/system/config triad,
the config round-trip, the registration surface, the BDF2 coefficients, and the
integrator guard -- so that step 5 drops the dual-time driver into a socket
whose shape is already tested. `PHYSICS_IMPLEMENTED` is the switch: when it
flips, `test_theStepIsStillAScaffold` is the test that should be replaced, not
silently left passing.
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
from warpSPH.systems.artificialCompressible import (ArtificialCompressibleState,
                                                    ArtificialCompressibleSystem,
                                                    bdfCoefficients)
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


def test_theStepIsStillAScaffold(built):
    """DELETE ME with plan step 5. Until the dual-time driver lands the step
    returns a zero update, and this test exists so that fact is asserted rather
    than assumed."""
    system, config, schemeConfig = built
    assert acmod.PHYSICS_IMPLEMENTED is False
    update, adjacency, state = acmod.artificialCompressible_step(
        system, 1e-3, config, schemeConfig)
    assert not update.dxdt.any()
    assert not update.dvdt.any()
    assert not update.dpdt.any()
    assert adjacency is not None
    assert state.surfaceIndicators is not None, 'surface detection still ran'
