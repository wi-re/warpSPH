"""Projection contracts: constraint sources, force choices, coupling and driver ownership."""

from types import SimpleNamespace as NS
from dataclasses import FrozenInstanceError
import json
import pytest
import torch
from warpSPH.modules.incompressible.sources import (
    omniSource,
    referenceSource,
    pressureBoundarySource,
)
from warpSPH.modules.incompressible.consistent import applyConsistentCoupling
from warpSPH.configurations import BoundaryPressureMode


@pytest.mark.parametrize("dt", [0.01, 0.1])
@pytest.mark.parametrize("rho0", [1.0, 1000.0])
def test_sourceUnitsFollowPredictedContinuityConstraint(dt, rho0):
    # Dv=+.3 expands a uniform element; continuity predicts rho*=rho*(1-.3dt).
    rho = torch.full((3,), rho0, dtype=torch.float64)
    divergence = torch.full_like(rho, 0.3)
    densityRate = -rho0 * divergence
    state = NS(densities=rho)
    cd = referenceSource(rho, densityRate, rho0=rho0, dt=dt, mode="density")
    df = referenceSource(rho, densityRate, rho0=rho0, dt=dt, mode="divergence")
    assert torch.allclose(cd, torch.full_like(rho, 0.3 * dt), atol=1e-15, rtol=0)
    assert torch.allclose(df, torch.full_like(rho, 0.3), atol=1e-15, rtol=0)
    for mode in ("density", "divergence"):
        omni = omniSource(state, None, None, rho0=rho0, dt=dt, divergence=divergence, mode=mode)
        assert torch.allclose(omni, 0.3 * dt * torch.ones_like(rho), atol=1e-15, rtol=0)
    volumeSource = pressureBoundarySource(
        torch.ones(3), torch.ones(3), dt=dt, divergence=divergence
    )
    assert torch.allclose(volumeSource, cd, atol=1e-15, rtol=0)


def test_surfaceSourcePoliciesDoNotMaskVelocityDivergence():
    state = NS(densities=torch.tensor([0.5, 1.1]), surfaceIndicators=torch.tensor([1, 0]))
    args = dict(rho0=1.0, dt=0.1, divergence=torch.tensor([2.0, 2.0]), mode="density")
    for policy in ("mask", "clamp"):
        source = omniSource(state, None, None, surfaceSource=policy, **args)
        assert torch.allclose(source, torch.tensor([0.2, 0.1]), atol=1e-7, rtol=0)
    with pytest.raises(ValueError, match="pressure mode"):
        referenceSource(state.densities, state.densities, rho0=1.0, dt=0.1, mode="typo")
    with pytest.raises(ValueError, match="surfaceSource"):
        omniSource(state, None, None, surfaceSource="typo", **args)


@pytest.mark.parametrize(
    "bodyForce,viscosity,total", [(False, False, 5.0), (True, False, 10.0), (True, True, 17.0)]
)
def test_sharedForceStageHonorsMethodChoices(monkeypatch, bodyForce, viscosity, total):
    from warpSPH.modules.incompressible import stages as S

    state = NS(masses=torch.tensor([2.0]), velocities=torch.zeros(1, 2), kinds=torch.tensor([0]))
    system = NS(state=state, t=0.0)
    events = []

    def value(name, amount):
        def f(*args, **kwargs):
            events.append(name)
            return torch.full((1, 2), amount)

        return f

    monkeypatch.setattr(S, "computeGravity", value("g", 2.0))
    monkeypatch.setattr(S, "computeBodyForce", value("body", 5.0))
    monkeypatch.setattr(S, "computeVelocityDiffusion", value("viscosity", 7.0))
    monkeypatch.setattr(S, "computeForcing", value("forcing", 6.0))
    acceleration = S.nonPressureAcceleration(
        system, 0.1, None, None, None, includeBodyForce=bodyForce, viscosity=viscosity
    )
    assert torch.equal(acceleration, torch.full((1, 2), total))
    assert events == ["g"] + (["body"] if bodyForce else []) + (
        ["viscosity"] if viscosity else []
    ) + ["forcing"]


def test_consistentCouplingRestoresTensorObjectsAndNeverOverwritesGhosts():
    state = NS(
        kinds=torch.tensor([0, 1, 2]),
        densities=torch.tensor([0.9, 0.8, 0.7]),
        masses=torch.tensor([1.0, 2.0, 3.0]),
    )
    originalRho, originalMass = state.densities, state.masses
    scheme = NS(
        fluid=NS(restDensity=1.0),
        solverConfig=NS(akinciBoundaryVolume=False, akinciBoundaryVolumeScale=2.0),
    )
    with pytest.raises(RuntimeError, match="solve failed"):
        with applyConsistentCoupling(state, None, scheme, None, BoundaryPressureMode.consistent):
            assert torch.equal(state.densities, torch.tensor([0.9, 1.0, 0.7]))
            assert torch.equal(state.masses, torch.tensor([1.0, 4.0, 3.0]))
            raise RuntimeError("solve failed")
    assert state.densities is originalRho
    assert state.masses is originalMass


def test_pressurePoliciesAreSnapshotsOfLegacySwitches(monkeypatch):
    from warpSPH.schemes.incompressible import omniIncompressible as O

    prior = O.pressurePolicy()
    monkeypatch.setattr(O, "OMEGA", 0.123)
    selected = O.pressurePolicy()
    assert selected.OMEGA == 0.123
    assert prior.OMEGA == 0.3
    with pytest.raises(FrozenInstanceError):
        selected.OMEGA = 0.7


def test_configRoundTripPreservesPressureSchedulesAndAllShiftingFields():
    from warpSPH.configurations.incompressible import (
        IncompressibleSPHConfig,
        incompressibleConfigToDict,
        dictToIncompressibleSPHConfig,
    )
    from warpSPH.configurations.region import ParticleRegion, RegionType

    config = IncompressibleSPHConfig()
    config.fluid.fixedSoundSpeed = 10.0
    config.solverConfig.pressureSolver.convergenceCheckSchedule = "every"
    config.solverConfig.divergenceFreeSolver.convergenceCheckSchedule = "adaptive"
    config.iisph.relaxationFactor = 0.25
    config.iisph.boundaryPressure = "contactShepard"
    config.shiftProperties.correctdrhodt = True
    config.shiftProperties.implicitSolverThreshold = 0.023
    config.regions = [
        ParticleRegion(
            sdf=lambda x: x[..., 0], type=RegionType.Fluid, particles=None, initialConditions={}
        )
    ]
    encoded = json.loads(json.dumps(incompressibleConfigToDict(config)))
    restored = dictToIncompressibleSPHConfig(encoded)
    assert restored.shiftProperties == config.shiftProperties
    assert restored.solverConfig == config.solverConfig
    assert restored.iisph == config.iisph
    assert len(restored.regions) == 1
    assert torch.equal(restored.regions[0].sdf(torch.tensor([[2.0, 3.0]])), torch.tensor([2.0]))
    for sub in ("pressureSolver", "divergenceFreeSolver"):
        del encoded["solverConfig"][sub]["convergenceCheckSchedule"]
    encoded.pop("regions")
    encoded.pop("rigidBodies")
    encoded.pop("iisph")
    legacy = dictToIncompressibleSPHConfig(encoded)
    assert legacy.solverConfig.pressureSolver.convergenceCheckSchedule == "auto"
    assert legacy.regions == legacy.rigidBodies == []
    from warpSPH.configurations import IISPHConfig
    assert legacy.iisph == IISPHConfig()


def _nativeIisphFixture(device, dim, contact, rho0=1000.0, wrappedWall=False):
    from warpSPHCore import DomainDescription, KernelFunctions, SupportScheme, buildVerletList
    from warpSPHCore.type_config import get_torch_precision
    from warpSPH.modules.iisph import IISPHOperator, IISPHBoundary

    dtype = get_torch_precision()
    x = torch.tensor([[-.31, .14], [-.06, .17], [.24, .12], [-.28, .42], [.01, .39], [.27, .43],
                      [-.25, -.13], [.14, -.16]], dtype=dtype, device=device)
    if dim == 3:
        x = torch.cat((x, torch.linspace(-.1, .1, len(x), dtype=dtype, device=device)[:, None]), 1)
    n = len(x)
    domain = DomainDescription(-2*torch.ones(dim, dtype=dtype, device=device),
        2*torch.ones(dim, dtype=dtype, device=device), torch.zeros(dim, dtype=torch.bool, device=device), dim)
    if wrappedWall:
        domain.periodic[1] = True
        x[-1, 1] += 4.
    config = NS(domain=domain, kernel=KernelFunctions.Wendland4, n_h=4., calibrateNormalization=False)
    state = NS(positions=x, supports=torch.full((n,), .8, dtype=dtype, device=device),
        masses=rho0*torch.linspace(.08, .12, n, dtype=dtype, device=device),
        kinds=torch.tensor([0]*6+[1]*2, dtype=torch.int32, device=device), velocities=torch.zeros_like(x))
    adj = buildVerletList(state, domain, supportMode=SupportScheme.Gather)
    boundary = None
    if contact:
        points = x.clone()
        points[:, 1] = 0
        g = torch.zeros_like(x)
        g[:, 1] = -9.81
        boundary = IISPHBoundary(torch.zeros_like(state.masses), torch.zeros_like(x), True, points, g)
    return IISPHOperator(state, config, adj, rho0, boundary=boundary)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA'))])
@pytest.mark.parametrize('dim,contact', [(2, False), (2, True), (3, False), (3, True)])
def test_nativeIisphForceContinuityAndExactDiagonal(device, dim, contact):
    # Independent small dense F/C oracle. Unequal volumes, unequal density,
    # disordered positions and static samples distinguish IISPH from DFSPH.
    op = _nativeIisphFixture(device, dim, contact)
    n = len(op.volume)
    F = torch.zeros((n*dim, n), dtype=op.volume.dtype, device=device)
    C = torch.zeros((n, n*dim), dtype=op.volume.dtype, device=device)
    T = torch.zeros((n, n), dtype=op.volume.dtype, device=device)
    if contact:
        cp = op.contact.pairs
        for i in range(n):
            if op.contact.denominator[i] > 0:
                for e in range(int(cp.starts[i]), int(cp.starts[i]+cp.counts[i])):
                    j = int(cp.j[e])
                    if op.fluid[j]:
                        T[i, j] += op.volume[j]*cp.weights[e]/op.contact.denominator[i]
    for i in range(n):
        if not op.fluid[i]:
            continue
        rows = slice(i*dim, (i+1)*dim)
        C[i, rows] += op.G[i]
        F[rows, i] -= op.G[i]/op.densityRatio[i]**2
        F[rows] -= op.G[i, :, None]*T[i, None, :]
        for e in range(int(op.pairs.starts[i]), int(op.pairs.starts[i]+op.pairs.counts[i])):
            j = int(op.pairs.j[e])
            if not op.fluid[j]:
                continue
            g = op.pairs.gradient[e]
            C[i, rows] += op.volume[j]*g
            C[i, j*dim:(j+1)*dim] -= op.volume[j]*g
            F[rows, i] -= op.volume[j]*g/op.densityRatio[i]**2
            F[rows, j] -= op.volume[j]*g/op.densityRatio[j]**2
    q = torch.linspace(.2, 1.4, n, dtype=op.volume.dtype, device=device)
    v = torch.sin(torch.arange(n*dim, dtype=q.dtype, device=device)).reshape(n, dim)
    torch.testing.assert_close(op.acceleration(q).flatten(), F@q, rtol=4e-6, atol=4e-6)
    torch.testing.assert_close(op.densityRate(v), C@v.flatten(), rtol=4e-6, atol=4e-6)
    torch.testing.assert_close(op.diagonal, (C@F).diagonal(), rtol=6e-6, atol=6e-6)

    from warpSPH.configurations import IISPHConfig
    policy = IISPHConfig(minIterations=6, maxIterations=6, relaxationFactor=.25, maxErrorPercent=0)
    previous = q*op.rho0
    dt = .01
    actual = op.solve(v, previous, dt, policy)
    b = 1-op.densityRatio-dt*(C@v.flatten())-dt**2*(C@op.constantAcceleration.flatten())
    A = dt**2*C@F
    diagonal = A.diagonal()
    old = .5*previous/op.rho0
    old[~op.fluid] = 0
    valid = op.fluid & (diagonal.abs() > 1.e-9)
    for _ in range(6):
        off = A@old-diagonal*old
        new = torch.where(valid, (.75*old+.25*(b-off)/torch.where(valid, diagonal, 1)).clamp_min(0), 0)
        error = torch.where(new != 0, op.rho0*(diagonal*new+off-b), 0)[op.fluid].mean()
        old = new
    torch.testing.assert_close(actual.pressure/op.rho0, old, rtol=1.e-5, atol=1.e-5)
    assert actual.iterations == 6
    assert actual.error == pytest.approx(float(error), rel=2.e-5, abs=.003)


def test_nativeIisphDensityUnitsAndDeviceParity():
    a = _nativeIisphFixture('cpu', 2, True, rho0=1.)
    b = _nativeIisphFixture('cpu', 2, True, rho0=1000.)
    torch.testing.assert_close(a.densityRatio, b.densityRatio)
    torch.testing.assert_close(a.diagonal, b.diagonal)
    from warpSPH.configurations import IISPHConfig
    policy = IISPHConfig(relaxationFactor=.25)
    outputs = [op.solve(op.state.velocities, torch.zeros_like(op.volume), .001, policy) for op in (a, b)]
    assert float(outputs[1].pressure.max()) > 0
    torch.testing.assert_close(outputs[0].pressure, outputs[1].pressure/1000)
    torch.testing.assert_close(outputs[0].acceleration, outputs[1].acceleration)
    if torch.cuda.is_available():
        gpu = _nativeIisphFixture('cuda', 2, True)
        result = gpu.solve(gpu.state.velocities, torch.zeros_like(gpu.volume), .001, policy)
        torch.testing.assert_close(result.pressure.cpu(), outputs[1].pressure, rtol=2e-5, atol=.005)
        torch.testing.assert_close(result.acceleration.cpu(), outputs[1].acceleration, rtol=2e-5, atol=2e-5)
        assert result.iterations == outputs[1].iterations


def test_nativeIisphViscosityIsFreeSlipAndDissipatesFluidMotion():
    from warpSPH.modules.iisph import iisphViscosity
    from warpSPH.configurations import IncompressibleSPHConfig
    op = _nativeIisphFixture('cpu', 2, False)
    state = op.state
    state.densities = op.densityRatio*op.rho0
    config = IncompressibleSPHConfig()
    config.fluid.restDensity = op.rho0
    config.fluid.fixedSoundSpeed = 10.
    config.diffusionParams.inviscid = False
    config.diffusionParams.viscidNu = .01
    # Stationary sampled walls exert no viscous drag on uniform fluid motion.
    state.velocities[op.fluid] = torch.tensor([1., 2.], dtype=state.positions.dtype)
    from warpSPHCore import buildVerletList
    adj = buildVerletList(state, op.config.domain)
    acceleration = iisphViscosity(state, op.config, config, adj)
    torch.testing.assert_close(acceleration, torch.zeros_like(acceleration))
    # Converging and separating pairs both contribute; net work is negative.
    state.velocities[op.fluid] = state.positions[op.fluid]
    acceleration = iisphViscosity(state, op.config, config, adj)
    assert float((state.masses[:, None]*state.velocities*acceleration).sum()) < 0
    assert torch.equal(acceleration[~op.fluid], torch.zeros_like(acceleration[~op.fluid]))


def test_nativeIisphSampledGravityUsesTheNearestPeriodicWallImage():
    base = _nativeIisphFixture('cpu', 2, True)
    wrapped = _nativeIisphFixture('cpu', 2, True, wrappedWall=True)
    torch.testing.assert_close(wrapped.densityRatio, base.densityRatio, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(wrapped.constantAcceleration, base.constantAcceleration, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(wrapped.diagonal, base.diagonal, rtol=3e-6, atol=3e-6)


def test_iisphPoliciesAndPressureCarryDoNotLeakBetweenSolves():
    from dataclasses import asdict
    from warpSPH.configurations import IISPHConfig
    first = _nativeIisphFixture('cpu', 2, True)
    other = _nativeIisphFixture('cpu', 2, False)
    short = IISPHConfig(minIterations=2, maxIterations=2, relaxationFactor=.25)
    longer = IISPHConfig(minIterations=6, maxIterations=6, relaxationFactor=.4)
    original = asdict(short), asdict(longer)
    previous = torch.full_like(first.volume, 1000.)
    carry = previous.clone()
    before = first.solve(first.state.velocities, previous, .001, short)
    between = other.solve(other.state.velocities, previous, .001, longer)
    after = first.solve(first.state.velocities, previous, .001, short)
    assert (before.iterations, between.iterations, after.iterations) == (2, 6, 2)
    assert (asdict(short), asdict(longer)) == original
    assert torch.equal(previous, carry)
    assert torch.equal(before.pressure, after.pressure)
    assert torch.equal(before.acceleration, after.acceleration)


def test_iisphProjectionRejectsNonfiniteResultsBeforeStateAdvance():
    from warpSPH.modules.iisph.stages import solveIISPHPressure
    state = NS(masses=torch.ones(3), pressures=None, positions=torch.zeros(3, 2))
    def failed(velocity, previous, dt, policy):
        assert torch.equal(previous, torch.zeros_like(state.masses))
        return NS(termination='nonfinite')
    with pytest.raises(FloatingPointError, match='Nonfinite IISPH'):
        solveIISPHPressure(state, .01, NS(solve=failed), torch.zeros(3, 2), NS())
    assert state.pressures is None
    assert torch.equal(state.positions, torch.zeros(3, 2))


@pytest.mark.parametrize("scheme", ["dfsphReference", "iisph", "omniIncompressible", "band2018pb"])
def test_fullMapSchemesSelectPackageDriverEvenWhenRKRequested(scheme):
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, prepareContext
    from warpSPH.runner.caseSpec import CaseSpec
    from warpSPHIntegrators.schemes import ProjectedStep

    importAll()
    ctx = prepareContext(getCase("tgv"), CaseSpec(scheme=scheme, integrationScheme="rungeKutta4"))
    assert isinstance(ctx.integrator, ProjectedStep)
    assert ctx.integrator.identifier.name == "projected_state_map_v1"
    assert len(tuple(ctx.bundle)) == 7


@pytest.mark.parametrize("name", ["iisph", "dfsphReference", "omniIncompressible", "band2018pb"])
def test_directFullMapCallRefusesMultistageWrapper(name):
    from importlib import import_module
    from warpSPHIntegrators.schemes import IntegrationSchemeType as I

    module = import_module("warpSPH.schemes.incompressible." + name)
    with pytest.raises(ValueError, match="full projected state map"):
        getattr(module, name + "_step")(NS(), 0.1, NS(integrationScheme=I.rungeKutta4), NS())


def test_projectedSystemKeepsLegacyClassAlias():
    from warpSPH.systems.incompressible import ProjectedSystem, DFSPHReferenceSystem

    assert DFSPHReferenceSystem is ProjectedSystem


def test_band_viscosity_flag_round_trips_and_defaults_off():
    from warpSPH.configurations.incompressible import IncompressibleSPHConfig, incompressibleConfigToDict, dictToIncompressibleSPHConfig
    c = IncompressibleSPHConfig()
    assert not c.bandViscosity
    c.bandViscosity = True
    assert dictToIncompressibleSPHConfig(incompressibleConfigToDict(c)).bandViscosity


@pytest.mark.parametrize('enabled', [False, True])
def test_band_viscosity_enters_predictor_and_does_not_move_wall_particles(monkeypatch, enabled):
    from warpSPH.schemes.incompressible import band2018pb as band
    from warpSPH.modules.incompressible import stages

    state = NS(kinds=torch.tensor([0, 1]), positions=torch.zeros(2, 2),
               velocities=torch.zeros(2, 2), masses=torch.ones(2),
               densities=torch.ones(2), pressures=torch.zeros(2))
    system = NS(state=state, t=0.)
    config = NS()
    scheme = NS(fluid=NS(restDensity=1.), bandViscosity=enabled)
    diffusion = torch.tensor([[2., 0.], [4., 0.]])
    monkeypatch.setattr(band, 'requireProjectedStep', lambda *a: None)
    monkeypatch.setattr(band, '_rebuildAdjacency', lambda *a: None)
    monkeypatch.setattr(band, 'computeDensities', lambda *a: torch.ones(2))
    monkeypatch.setattr(band, 'bandBoundaryUnknownMask', lambda *a: torch.zeros(2, dtype=torch.bool))
    monkeypatch.setattr(band, 'enforceDirichlet', lambda *a: None)
    monkeypatch.setattr(band, 'projectedUpdate', lambda *a: None)
    monkeypatch.setattr(stages, 'computeGravity', lambda *a: torch.zeros(2, 2))
    monkeypatch.setattr(stages, 'computeForcing', lambda *a, **kw: torch.zeros(2, 2))
    monkeypatch.setattr(stages, 'computeVelocityDiffusion', lambda *a, **kw: diffusion)
    entered = []

    def pressure(*a, **kw):
        entered.append(kw['vStar'].clone())
        return torch.zeros(2, 2), torch.zeros(2), 0, 0.

    monkeypatch.setattr(band, '_solve', pressure)
    band.band2018pb_step(system, .1, config, scheme)
    expected = torch.tensor([[.2 if enabled else 0., 0.], [0., 0.]])
    torch.testing.assert_close(entered[0], expected)
    torch.testing.assert_close(state.velocities, expected)
    torch.testing.assert_close(state.positions[1], torch.zeros(2))


@pytest.mark.parametrize('mode', ['density', 'divergence'])
def test_reference_pressure_correction_is_invariant_under_density_units(mode):
    """A physical-pressure preconditioner must scale with the force operator."""
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, prepareContext, CaseSpec
    from warpSPH.modules.incompressible.operators import rebuildAdjacency
    from warpSPH.modules.incompressible.reference import solveReference

    importAll()
    case = getCase('periodicChannel')
    results = []
    for rho0 in (1., 1000.):
        spec = CaseSpec(caseName=case.name, scheme='iisph', nx=16, L=1., dim=2,
                        n_h=4., dt=.001, device='cpu', kernel='Wendland2',
                        supportMode='SuperSymmetric', quiet=True, cudaGraph=False,
                        params={**case.params, 'shearWave': True})
        ctx = prepareContext(case, spec)
        ctx.schemeConfig.fluid.restDensity = rho0
        system = case.buildSystem(ctx)
        case.initialConditions(ctx, system)
        state = system.state
        fluid = state.kinds == 0
        state.densities[fluid] *= 1 + .02 * torch.sin(2 * torch.pi * state.positions[fluid, 0])
        velocity = torch.zeros_like(state.velocities)
        velocity[fluid, 0] = .1 * torch.sin(2 * torch.pi * state.positions[fluid, 0])
        solver = ctx.schemeConfig.solverConfig.pressureSolver
        solver.maxIterations = 12
        solver.tolerance = 0.
        adjacency = rebuildAdjacency(state, system, ctx.config)
        acceleration, pressure, iterations, _ = solveReference(
            state, ctx.config, ctx.schemeConfig, adjacency, vEnter=velocity,
            rho0=rho0, warmStart=torch.zeros_like(state.densities),
            opDt=.001**(2 if mode == 'density' else 1), solverCfg=solver,
            minIters=12, fluidMask=fluid, mode=mode)
        assert iterations == 12
        results.append((acceleration[fluid], pressure[fluid] / rho0))
    tolerance = 5e-5 if results[0][0].dtype == torch.float32 else 2e-10
    for actual, expected in zip(results[1], results[0]):
        assert torch.linalg.vector_norm(actual - expected) < tolerance * torch.linalg.vector_norm(expected)
