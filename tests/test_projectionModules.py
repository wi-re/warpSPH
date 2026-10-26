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
    config.iisph.clampPressure = False
    config.iisph.boundaryPressure = "contactShepard"
    config.dfsph.divergence.maxErrorPercent = .025
    config.dfsph.deficiencyGuard = False
    config.dfsph.warmStart = False
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
    assert restored.dfsph == config.dfsph
    assert len(restored.regions) == 1
    assert torch.equal(restored.regions[0].sdf(torch.tensor([[2.0, 3.0]])), torch.tensor([2.0]))
    for sub in ("pressureSolver", "divergenceFreeSolver"):
        del encoded["solverConfig"][sub]["convergenceCheckSchedule"]
    encoded.pop("regions")
    encoded.pop("rigidBodies")
    encoded.pop("iisph")
    encoded.pop("dfsph")
    legacy = dictToIncompressibleSPHConfig(encoded)
    assert legacy.solverConfig.pressureSolver.convergenceCheckSchedule == "auto"
    assert legacy.regions == legacy.rigidBodies == []
    from warpSPH.configurations import IISPHConfig
    assert legacy.iisph == IISPHConfig()
    from warpSPH.configurations import DFSPHConfig
    assert legacy.dfsph == DFSPHConfig()


def _nativeIisphFixture(device, dim, contact, rho0=1000.0, wrappedWall=False):
    from warpSPHCore import DomainDescription, KernelFunctions, SupportScheme, buildVerletList
    from warpSPHCore.type_config import get_torch_precision
    from warpSPH.modules.iisph import computeIISPHCoefficients, computeIISPHBoundaryPressureAcceleration
    from warpSPH.modules.density import computeDensities
    from warpSPH.configurations import IISPHConfig

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
        kinds=torch.tensor([0]*6+[1]*2, dtype=torch.int32, device=device), velocities=torch.zeros_like(x),
        densities=torch.full((n,), rho0, dtype=dtype, device=device))
    adj = buildVerletList(state, domain, supportMode=SupportScheme.Gather)
    points = None
    g = None
    if contact:
        points = x.clone()
        points[:, 1] = 0
        g = torch.zeros_like(x)
        g[:, 1] = -9.81
    scheme = NS(fluid=NS(restDensity=rho0), iisph=IISPHConfig(), rigidBodies=[])
    state.densities = computeDensities(state, config, scheme, adj, wall=False)
    (restVolumes, densityRatio, wallGradient, gradientSum, diagonal,
     contactAdjacency, contactDenominator, _) = computeIISPHCoefficients(state, config, scheme, adj, contactPoints=points)
    coefficients = NS(restVolumes=restVolumes, densityRatio=densityRatio, wallGradient=wallGradient,
        gradientSum=gradientSum, diagonal=diagonal, contactAdjacency=contactAdjacency,
        contactDenominator=contactDenominator)
    boundaryAcceleration = computeIISPHBoundaryPressureAcceleration(
        state, config, adj, restVolumes, wallGradient, None, points, g,
        contactGravity=scheme.iisph.contactGravity)
    return NS(state=state, config=config, scheme=scheme, coefficients=coefficients,
              boundaryAcceleration=boundaryAcceleration, adjacency=adj, contactPoints=points, gravity=g)


def test_iisphDirichletDensityReachesPressureSolve(monkeypatch):
    from warpSPH.schemes.incompressible import iisph
    from warpSPH.configurations.moduleConfigurations.boundaryConditions import BoundaryCondition, BoundaryConditionType

    fixture = _nativeIisphFixture('cpu', 2, False)
    state, config, scheme = fixture.state, fixture.config, fixture.scheme
    config.verletScale = 1.0
    scheme.rigidBodies = []
    sampledDensity = state.densities.clone()
    prescribed = 1.1 * scheme.fluid.restDensity

    def sdf(positions):
        distance = torch.ones_like(positions[:, 0])
        distance[0] = -1
        return distance, torch.zeros_like(positions)

    def densityBC(currentState, *args):
        torch.testing.assert_close(currentState.densities, sampledDensity)
        return torch.full_like(currentState.densities, prescribed)

    scheme.boundaryConditions = [BoundaryCondition(BoundaryConditionType.constant, sdf,
        dirichletFunctions={'densities': densityBC})]
    system = NS(state=state, adjacency=None, t=0., boundarySupport=.8,
                analyticBoundaryProvider=None, contactBoundaryProvider=None)
    state.densities = torch.zeros_like(state.masses)
    state.pressures = None
    expected = sampledDensity.clone()
    expected[0] = prescribed
    monkeypatch.setattr(iisph, 'computeGravity', lambda *args: torch.zeros_like(state.positions))
    monkeypatch.setattr(iisph, 'computeBodyForce', lambda *args: torch.zeros_like(state.positions))
    monkeypatch.setattr(iisph, 'computeVelocityDiffusion', lambda *args, **kwargs: torch.zeros_like(state.positions))

    class ReachedPressureSolve(Exception):
        pass

    def pressure(currentState, config, schemeConfig, velocity, dt, adjacency, **kwargs):
        from warpSPH.modules.iisph import computeIISPHCoefficients
        _, densityRatio, *_ = computeIISPHCoefficients(currentState, config, schemeConfig, adjacency,
            contactPoints=kwargs['contactPoints'])
        torch.testing.assert_close(currentState.densities, expected)
        torch.testing.assert_close(densityRatio, expected / scheme.fluid.restDensity)
        raise ReachedPressureSolve

    monkeypatch.setattr(iisph, 'solveIISPH', pressure)
    with pytest.raises(ReachedPressureSolve):
        iisph.iisph_step(system, .001, config, scheme)


def test_iisphDensityInitializationDoesNotBuildPressureCoefficients(monkeypatch):
    from warpSPH.modules import iisph
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPH.enumTypes import IncompressibleSPHScheme

    fixture = _nativeIisphFixture('cpu', 2, False)
    state, config, scheme = fixture.state, fixture.config, fixture.scheme
    config.verletScale = 1.0
    scheme.rigidBodies = []
    expected = state.densities.clone()
    state.densities = torch.zeros_like(state.masses)
    ctx = NS(scheme=IncompressibleSPHScheme.iisph, config=config, schemeConfig=scheme,
             spec=NS(n_h=config.n_h), scratch={})

    def unexpected(*args, **kwargs):
        pytest.fail('Density initialization must not build pressure coefficients')

    monkeypatch.setattr(iisph, 'computeIISPHCoefficients', unexpected)
    monkeypatch.setattr(iisph, 'computeIISPHBoundaryIntegrals', unexpected)
    initializeProjectionDensity(ctx, NS(state=state, adjacency=None))
    torch.testing.assert_close(state.densities, expected)


@pytest.mark.parametrize('kernelName', ['Wendland2', 'Wendland4'])
@pytest.mark.parametrize('n_h', [2., 4.])
def test_iisphInitializationCalibratesMassesOnceAndLeavesOrdinaryKernels(monkeypatch, kernelName, n_h):
    from warpSPHCore import DomainDescription, KernelFunctions, buildVerletList
    from warpSPHCore.type_config import get_torch_precision
    from warpSPHCore.util.latticeDensity import latticeDensityFactor
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPH.enumTypes import IncompressibleSPHScheme
    from warpSPH.configurations import IISPHConfig
    from warpSPH.modules.iisph import computeIISPHCoefficients

    dtype, rho0, dx = get_torch_precision(), 1000., .125
    axis = dx * torch.arange(-4, 5, dtype=dtype)
    positions = torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)
    kinds = torch.zeros(len(positions), dtype=torch.int32)
    kinds[:9] = 1  # Sampled wall masses receive the same calibration as fluid masses.
    domain = DomainDescription(-torch.ones(2, dtype=dtype), torch.ones(2, dtype=dtype),
                               torch.zeros(2, dtype=torch.bool), 2)
    config = NS(domain=domain, kernel=KernelFunctions[kernelName], n_h=n_h,
                calibrateNormalization=True)
    scheme = NS(fluid=NS(restDensity=rho0), iisph=IISPHConfig(boundaryRepresentation='particles'),
                rigidBodies=[], analyticWallMass=1.)
    state = NS(positions=positions, supports=torch.full((len(positions),), n_h*dx, dtype=dtype),
        masses=torch.full((len(positions),), rho0*dx**2, dtype=dtype), kinds=kinds,
        densities=torch.zeros(len(positions), dtype=dtype), velocities=torch.zeros_like(positions))
    system = NS(state=state, adjacency=buildVerletList(state, domain))
    ctx = NS(scheme=IncompressibleSPHScheme.iisph, config=config, schemeConfig=scheme,
             spec=NS(n_h=n_h), scratch={})
    originalMasses = state.masses.clone()
    factor = latticeDensityFactor(config.kernel, n_h, 2)
    initializeProjectionDensity(ctx, system)
    assert not config.calibrateNormalization
    assert scheme.analyticWallMass == factor
    torch.testing.assert_close(state.masses, originalMasses*factor, rtol=0, atol=0)
    assert float(state.densities[40]) == pytest.approx(rho0, rel=2e-6)
    calibratedMasses = state.masses.clone()

    def unexpected(*args, **kwargs):
        pytest.fail('Lattice calibration must not run after initialization')

    # Import the module rather than the util package's same-named function.
    from importlib import import_module
    monkeypatch.setattr(import_module('warpSPHCore.util.latticeDensity'), 'latticeDensityFactor', unexpected)
    initializeProjectionDensity(ctx, system)
    integrals = None
    computeIISPHCoefficients(state, config, scheme, system.adjacency, boundaryIntegrals=integrals)
    assert torch.equal(state.masses, calibratedMasses)
    assert scheme.analyticWallMass == factor


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA'))])
def test_iisphLatticeRelaxationUsesNativePressureAndCalibratedMasses(monkeypatch, device):
    from warpSPH.runner.runner import prepareContext
    from warpSPH.runner.caseSpec import CaseSpec
    from warpSPH.cases.periodic.tgv import tgvCase
    from warpSPH.caseUtils.incompressible import relaxLattice
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPHCore.type_config import get_torch_precision
    from warpSPHCore.util.latticeDensity import latticeDensityFactor
    import warpSPH.modules as modules
    from warpSPH.modules import iisph

    spec = CaseSpec(caseName='tgv', scheme='iisph', params=dict(tgvCase.params)).merged(**tgvCase.defaults).merged(
        nx=16, kernel='Wendland4', n_h=4., device=device, precision=str(get_torch_precision()).split('.')[-1],
        plot=False, store=False, show=False, quiet=True, progress=False)
    ctx = prepareContext(tgvCase, spec)
    # This checks initializer ownership, not the reference omega=.5. At this
    # support .5 is outside the native Jacobi window and makes the jittered
    # fixture fail nondeterministically (also reproduced on the baseline).
    ctx.schemeConfig.iisph.relaxationFactor = .3
    system = tgvCase.buildSystem(ctx)
    expectedMasses = system.state.masses * latticeDensityFactor(ctx.config.kernel, 4., 2)
    originalPositions = system.state.positions.clone()
    originalSolve = iisph.solveIISPH
    calls = []

    def unexpected(*args, **kwargs):
        pytest.fail('Native IISPH initialization must not use the legacy shifting solve')

    def observed(state, config, scheme, acceleration, dt, adjacency, **kwargs):
        assert not config.calibrateNormalization
        assert not scheme.iisph.clampPressure
        torch.testing.assert_close(state.masses, expectedMasses, rtol=0, atol=0)
        assert not bool(state.velocities.any())
        calls.append(1)
        return originalSolve(state, config, scheme, acceleration, dt, adjacency, **kwargs)

    monkeypatch.setattr(modules, 'solveIncompressible', unexpected)
    monkeypatch.setattr(iisph, 'solveIISPH', observed)
    relaxLattice(ctx, system, 2, .001, .01)
    assert len(calls) == 2
    assert not torch.equal(system.state.positions, originalPositions)
    assert not bool(system.state.velocities.any())
    initializeProjectionDensity(ctx, system)
    torch.testing.assert_close(system.state.masses, expectedMasses, rtol=0, atol=0)


@pytest.mark.parametrize('representation,providerPresent,sampled', [
    ('auto', False, True), ('auto', True, False), ('particles', True, True),
])
def test_iisphBoundarySetupResolvesProvidersAndHostSupport(representation, providerPresent, sampled):
    from warpSPH.modules.iisph import initializeIISPHBoundaries, computeIISPHBoundaryIntegrals
    from warpSPH.systems.incompressible import IISPHSystem

    fixture = _nativeIisphFixture('cpu', 2, False)
    state, scheme = fixture.state, fixture.scheme
    if not sampled:
        state.kinds.zero_()
    aggregate = object()
    calls = []

    def evaluate(current, support, kernel, *, lean):
        calls.append((current, support, kernel, lean))
        return aggregate

    provider = NS(rigidBodies=[], aggregate=evaluate) if providerPresent else None
    scheme.boundaryProvider = provider
    scheme.iisph.boundaryRepresentation = representation
    scheme.iisph.boundaryPressure = 'contactShepard' if providerPresent else 'self'
    support, analytic, contact = initializeIISPHBoundaries(state, scheme)
    assert isinstance(support, float) and support == float(state.supports[0])
    assert scheme.iisph.boundaryRepresentation == ('analytic' if representation == 'auto' and providerPresent else 'particles')
    assert analytic is (provider if scheme.iisph.boundaryRepresentation == 'analytic' else None)
    assert contact is provider
    if analytic is not None:
        assert computeIISPHBoundaryIntegrals(state, analytic, support, fixture.config.kernel) is aggregate
        assert calls == [(state, support, fixture.config.kernel, True)]
    system = IISPHSystem(state=state, boundarySupport=support,
        analyticBoundaryProvider=analytic, contactBoundaryProvider=contact)
    # Stage creation must carry setup metadata without inspecting particle data.
    from warpSPH.systems.incompressible import IncompressibleState
    kinds = state.kinds
    system.state = IncompressibleState(**vars(state), materials=torch.zeros_like(kinds),
        UIDs=torch.arange(len(kinds), dtype=torch.int32), UIDcounter=len(kinds))
    stage = system.initializeNewState()
    assert stage.boundarySupport == support
    assert stage.analyticBoundaryProvider is analytic and stage.contactBoundaryProvider is contact


@pytest.mark.parametrize('covered,supported,message', [
    (False, True, 'did not cover'), (True, False, 'no fluid support'), (False, False, 'did not cover'),
])
def test_iisphGeometryFailuresShareOneCheckpoint(monkeypatch, covered, supported, message):
    from importlib import import_module
    from warpSPH.modules.iisph import computeIISPHContactPoints, solveIISPH, solver
    weightsModule = import_module('warpSPH.modules.iisph.contactWeights')
    fixture = _nativeIisphFixture('cpu', 2, True)
    state = fixture.state

    def distance(positions):
        hit = torch.ones_like(state.kinds, dtype=torch.bool)
        hit[0] = covered
        normal = torch.zeros_like(positions)
        normal[:, 1] = 1
        return positions[:, 1], normal, hit

    points, coverage = computeIISPHContactPoints(state, NS(signed_distance=distance))
    if not supported:
        monkeypatch.setattr(weightsModule, 'warpOperation', lambda *a, **kw: torch.zeros_like(state.masses))

    def unexpected(*args, **kwargs):
        pytest.fail('Invalid geometry must be reported before pressure iteration')

    monkeypatch.setattr(solver, 'runStationary', unexpected)
    reads = []
    tensorCpu = torch.Tensor.cpu

    def read(tensor, *args, **kwargs):
        if tensor.dtype == torch.bool and tensor.shape == (2,):
            reads.append(tensor.clone())
        return tensorCpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', read)
    with pytest.raises(ValueError, match=message):
        solveIISPH(state, fixture.config, fixture.scheme, torch.zeros_like(state.velocities),
            .001, fixture.adjacency, contactPoints=points, contactCoverageFailure=coverage)
    assert len(reads) == 1
    assert reads[0].tolist() == [not covered, not supported]


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA'))])
def test_iisphIterationDoesNotReadScalarValuesOrCompactMasks(monkeypatch, device):
    from warpSPH.modules.iisph import solveIISPH, solver
    fixture = _nativeIisphFixture(device, 2, False)
    fixture.scheme.iisph.minIterations = fixture.scheme.iisph.maxIterations = 6
    driver = solver.runStationary

    def inspect(step, initial, **kwargs):
        step(initial, 0)  # Compile operators before profiling the iteration itself.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            step(initial, 0)
        operations = {event.key for event in trace.key_averages()}
        assert not operations.intersection({'aten::nonzero', 'aten::item', 'aten::_local_scalar_dense'})
        return driver(step, initial, **kwargs)

    monkeypatch.setattr(solver, 'runStationary', inspect)
    result = solveIISPH(fixture.state, fixture.config, fixture.scheme,
        torch.zeros_like(fixture.state.velocities), .001, fixture.adjacency)
    assert result.iterations == 6


def _denseIisphKernels(state, config, points=None):
    """Independent dense Wendland C4 values/gradients for the IISPH fixture."""
    query = state.positions if points is None else points
    distance = query[:, None] - state.positions[None, :]
    length = config.domain.max - config.domain.min
    distance = distance - torch.where(config.domain.periodic,
        torch.round(distance / length) * length, 0)
    dim = distance.shape[-1]
    support = state.supports[:, None]
    radius = torch.linalg.vector_norm(distance, dim=-1) / support
    cutoff = (1 - radius).clamp_min(0)
    normalization = (9.0 if dim == 2 else 495.0 / 32.0) / torch.pi
    weights = normalization / support**dim * cutoff**6 * (1 + 6*radius + 35/3*radius**2)
    gradient = (-56/3 * normalization / support**(dim+2) * (5*radius+1) * cutoff**5)[..., None] * distance
    return weights, gradient


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA'))])
@pytest.mark.parametrize('dim,contact', [(2, False), (2, True), (3, False), (3, True)])
@pytest.mark.parametrize('clampPressure', [True, False])
def test_nativeIisphForceContinuityAndExactDiagonal(device, dim, contact, clampPressure):
    from warpSPH.modules.iisph import computeIISPHPressureAcceleration, computeIISPHDensityRate, solveIISPH
    # Independent small dense F/C oracle. Unequal volumes, unequal density,
    # disordered positions and static samples distinguish IISPH from DFSPH.
    fixture = _nativeIisphFixture(device, dim, contact)
    state, coefficients = fixture.state, fixture.coefficients
    volume = coefficients.restVolumes
    _, gradients = _denseIisphKernels(state, fixture.config)
    fluid, rho0 = state.kinds == 0, fixture.scheme.fluid.restDensity
    n = len(volume)
    F = torch.zeros((n*dim, n), dtype=volume.dtype, device=device)
    C = torch.zeros((n, n*dim), dtype=volume.dtype, device=device)
    T = torch.zeros((n, n), dtype=volume.dtype, device=device)
    expectedWallGradient = (gradients[:, ~fluid] * volume[~fluid][None, :, None]).sum(1)
    torch.testing.assert_close(coefficients.wallGradient, expectedWallGradient, rtol=4e-6, atol=4e-6)
    expectedGradientSum = (gradients * volume[None, :, None]).sum(1)
    torch.testing.assert_close(coefficients.gradientSum, expectedGradientSum, rtol=4e-6, atol=4e-6)
    if contact:
        contactWeights, _ = _denseIisphKernels(state, fixture.config, fixture.contactPoints)
        denominator = contactWeights[:, fluid] @ volume[fluid]
        denominator = torch.where(fluid & (expectedWallGradient.square().sum(-1) > 1.e-24), denominator, 0)
        torch.testing.assert_close(coefficients.contactDenominator, denominator, rtol=4e-6, atol=4e-6)
        for i in range(n):
            if coefficients.contactDenominator[i] > 0:
                for j in range(n):
                    if fluid[j]:
                        T[i, j] = volume[j]*contactWeights[i, j]/coefficients.contactDenominator[i]
    for i in range(n):
        if not fluid[i]:
            continue
        rows = slice(i*dim, (i+1)*dim)
        C[i, rows] += coefficients.wallGradient[i]
        F[rows, i] -= coefficients.wallGradient[i]/coefficients.densityRatio[i]**2
        F[rows] -= coefficients.wallGradient[i, :, None]*T[i, None, :]
        for j in range(n):
            if not fluid[j]:
                continue
            g = gradients[i, j]
            C[i, rows] += volume[j]*g
            C[i, j*dim:(j+1)*dim] -= volume[j]*g
            F[rows, i] -= volume[j]*g/coefficients.densityRatio[i]**2
            F[rows, j] -= volume[j]*g/coefficients.densityRatio[j]**2
    q = torch.linspace(.2, 1.4, n, dtype=volume.dtype, device=device)
    v = torch.sin(torch.arange(n*dim, dtype=q.dtype, device=device)).reshape(n, dim)
    acceleration = computeIISPHPressureAcceleration(state, fixture.config, fixture.adjacency, q,
        volume, coefficients.densityRatio, coefficients.gradientSum, coefficients.wallGradient,
        contactPoints=fixture.contactPoints, contactAdjacency=coefficients.contactAdjacency,
        contactDenominator=coefficients.contactDenominator)
    torch.testing.assert_close(acceleration.flatten(), F@q, rtol=4e-6, atol=4e-6)
    torch.testing.assert_close(computeIISPHDensityRate(state, fixture.config, fixture.adjacency,
        v, volume, coefficients.wallGradient), C@v.flatten(), rtol=4e-6, atol=4e-6)
    torch.testing.assert_close(coefficients.diagonal, (C@F).diagonal(), rtol=6e-6, atol=6e-6)

    from warpSPH.configurations import IISPHConfig
    policy = IISPHConfig(minIterations=6, maxIterations=6, relaxationFactor=.25, maxErrorPercent=0,
        clampPressure=clampPressure)
    previous = q*rho0
    dt = .01
    fixture.scheme.iisph = policy
    state.velocities = torch.where(fluid[:, None], v, 0)
    actual = solveIISPH(state, fixture.config, fixture.scheme, torch.zeros_like(v), dt, fixture.adjacency, previous,
                        contactPoints=fixture.contactPoints, gravity=fixture.gravity)
    torch.testing.assert_close(state.velocities, torch.where(fluid[:, None], v, 0), rtol=0, atol=0)
    b = 1-coefficients.densityRatio-dt*(C@v.flatten())-dt**2*(C@fixture.boundaryAcceleration.flatten())
    torch.testing.assert_close(actual.source, torch.where(fluid, b, 0), rtol=2.e-6, atol=2.e-6)
    A = dt**2*C@F
    diagonal = A.diagonal()
    old = .5*previous/rho0
    old[~fluid] = 0
    valid = fluid & (diagonal.abs() > 1.e-9)
    for _ in range(6):
        off = A@old-diagonal*old
        new = .75*old+.25*(b-off)/torch.where(valid, diagonal, 1)
        if clampPressure:
            new = torch.where(valid, new.clamp_min(0), 0)
            error = torch.where(new != 0, rho0*(diagonal*new+off-b), 0)[fluid].mean()
        else:
            new = torch.where(valid, new, 0)
            error = (rho0*(A@old-b))[fluid].abs().mean()
        old = new
    torch.testing.assert_close(actual.pressure/rho0, old, rtol=1.e-5, atol=1.e-5)
    torch.testing.assert_close(actual.acceleration.flatten(),
        F@old + fixture.boundaryAcceleration.flatten(), rtol=1.e-5, atol=1.e-5)
    assert actual.iterations == 6
    assert actual.error == pytest.approx(float(error), rel=2.e-5, abs=.003)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA'))])
def test_nativeIisphOperatorGradients(device):
    from warpSPH.modules.iisph import (
        computeIISPHCoefficients, computeIISPHBoundaryPressureAcceleration,
        computeIISPHPressureAcceleration, computeIISPHDensityRate,
    )
    fixture = _nativeIisphFixture(device, 2, True)
    state = fixture.state
    if state.positions.dtype != torch.float64:
        pytest.skip('Run with warpSPHCore_PRECISION=float64 for numerical gradient checks')
    q = torch.linspace(.2, 1.4, len(state.masses), dtype=state.positions.dtype, device=device)
    velocity = torch.sin(torch.arange(state.positions.numel(), dtype=q.dtype, device=device)).reshape_as(state.positions)
    inputs = tuple(value.detach().clone().requires_grad_() for value in (
        state.positions, state.supports, state.masses, state.densities,
        fixture.contactPoints, fixture.gravity, q, velocity,
    ))

    def evaluate(positions, supports, masses, densities, points, gravity, pressure, field):
        current = NS(**{**vars(state), 'positions': positions, 'supports': supports,
                        'masses': masses, 'densities': densities})
        (restVolumes, densityRatio, wallGradient, gradientSum, diagonal,
         contactAdjacency, contactDenominator, _) = computeIISPHCoefficients(
            current, fixture.config, fixture.scheme, fixture.adjacency, contactPoints=points)
        return (
            wallGradient, gradientSum, diagonal, contactDenominator,
            computeIISPHPressureAcceleration(current, fixture.config, fixture.adjacency, pressure,
                restVolumes, densityRatio, gradientSum, wallGradient,
                contactPoints=points, contactAdjacency=contactAdjacency, contactDenominator=contactDenominator),
            computeIISPHDensityRate(current, fixture.config, fixture.adjacency, field, restVolumes, wallGradient),
            computeIISPHBoundaryPressureAcceleration(current, fixture.config,
                fixture.adjacency, restVolumes, wallGradient, None, points, gravity,
                contactGravity=fixture.scheme.iisph.contactGravity),
        )

    # CUDA adjoints accumulate neighbor contributions with floating-point atomics.
    assert torch.autograd.gradcheck(evaluate, inputs, eps=1.e-6, atol=5.e-5, rtol=5.e-4,
        nondet_tol=1.e-10 if device == 'cuda' else 0, fast_mode=True)


def test_nativeIisphDensityUnitsAndDeviceParity():
    a = _nativeIisphFixture('cpu', 2, True, rho0=1.)
    b = _nativeIisphFixture('cpu', 2, True, rho0=1000.)
    torch.testing.assert_close(a.coefficients.densityRatio, b.coefficients.densityRatio)
    torch.testing.assert_close(a.coefficients.diagonal, b.coefficients.diagonal)
    from warpSPH.configurations import IISPHConfig
    policy = IISPHConfig(relaxationFactor=.25)
    from warpSPH.modules.iisph import solveIISPH
    a.scheme.iisph = b.scheme.iisph = policy
    outputs = [solveIISPH(op.state, op.config, op.scheme, torch.zeros_like(op.state.velocities), .001,
                          op.adjacency, contactPoints=op.contactPoints, gravity=op.gravity) for op in (a, b)]
    assert float(outputs[1].pressure.max()) > 0
    torch.testing.assert_close(outputs[0].pressure, outputs[1].pressure/1000)
    torch.testing.assert_close(outputs[0].acceleration, outputs[1].acceleration)
    if torch.cuda.is_available():
        gpu = _nativeIisphFixture('cuda', 2, True)
        gpu.scheme.iisph = policy
        result = solveIISPH(gpu.state, gpu.config, gpu.scheme, torch.zeros_like(gpu.state.velocities), .001, gpu.adjacency,
                            contactPoints=gpu.contactPoints, gravity=gpu.gravity)
        torch.testing.assert_close(result.pressure.cpu(), outputs[1].pressure, rtol=2e-5, atol=.005)
        torch.testing.assert_close(result.acceleration.cpu(), outputs[1].acceleration, rtol=2e-5, atol=2e-5)
        assert result.iterations == outputs[1].iterations


def test_sharedViscosityDissipatesFluidMotionWithoutWallInteractions():
    from warpSPH.modules.deltaSPH import computeVelocityDiffusion
    from warpSPH.configurations import IncompressibleSPHConfig
    op = _nativeIisphFixture('cpu', 2, False)
    state = op.state
    state.densities = op.coefficients.densityRatio*op.scheme.fluid.restDensity
    config = IncompressibleSPHConfig()
    config.fluid.restDensity = op.scheme.fluid.restDensity
    config.fluid.fixedSoundSpeed = 10.
    config.diffusionParams.inviscid = False
    config.diffusionParams.viscidNu = .01
    # Explicit wall exclusion leaves uniform fluid motion unchanged.
    state.velocities[(op.state.kinds == 0)] = torch.tensor([1., 2.], dtype=state.positions.dtype)
    from warpSPHCore import buildVerletList, OperationDirection
    adj = buildVerletList(state, op.config.domain)
    acceleration = computeVelocityDiffusion(state, op.config, config, adj, wall=False,
        operationMode=OperationDirection.FluidToFluid)
    torch.testing.assert_close(acceleration, torch.zeros_like(acceleration))
    # Converging and separating pairs both contribute; net work is negative.
    state.velocities[(op.state.kinds == 0)] = state.positions[(op.state.kinds == 0)]
    acceleration = computeVelocityDiffusion(state, op.config, config, adj, wall=False,
        operationMode=OperationDirection.FluidToFluid)
    assert float((state.masses[:, None]*state.velocities*acceleration).sum()) < 0
    assert torch.equal(acceleration[~(op.state.kinds == 0)], torch.zeros_like(acceleration[~(op.state.kinds == 0)]))


def test_iisphAnalyticGravityExtensionMatchesSolidQuadrature():
    from warpSPH.modules.iisph import computeIISPHBoundaryPressureAcceleration
    from warpSPH.configurations import IISPHConfig
    # Two bodies with different volumes and a nonzero contact offset. Build
    # the reference directly from solid sample pressures, independently of
    # the provider's gradient-moment contraction.
    x = torch.tensor([[.2, .3], [.4, .1], [.1, -.1]], dtype=torch.float64)
    points = torch.tensor([[.2, 0.], [0., .1], [.1, 0.]], dtype=x.dtype)
    state = NS(positions=x, kinds=torch.tensor([0, 0, 1]))
    gravity = torch.tensor([[1., -9.81]]*len(x), dtype=x.dtype)
    samples = torch.tensor([[[-.2, -.3], [.1, -.2]],
                            [[-.1, -.4], [.3, -.1]]], dtype=x.dtype)
    volumes = torch.tensor([[.1, .2], [.3, .15]], dtype=x.dtype)
    gradients = torch.arange(1, 25, dtype=x.dtype).reshape(2, 3, 2, 2)/10
    bodyGradient = (volumes[:, None, :, None]*gradients).sum(2)
    offset = samples[:, None, :, :] - x[None, :, None, :]
    moment = (volumes[:, None, :, None, None] *
              offset[..., :, None] * gradients[..., None, :]).sum(2)
    scheme = NS(iisph=IISPHConfig(), analyticWallMass=.8)
    integrals = NS(out={'G': bodyGradient, 'Cov': moment})
    wallGradient = .8*bodyGradient.sum(0)
    expected = torch.zeros_like(x)
    for body in range(2):
        for sample in range(2):
            q = ((samples[body, sample]-points)*gravity).sum(-1)
            expected -= .8*volumes[body, sample]*q[:, None]*gradients[body, :, sample]
    expected[state.kinds != 0] = 0
    actual = computeIISPHBoundaryPressureAcceleration(
        state, None, None, None, wallGradient, integrals, points, gravity,
        contactGravity=scheme.iisph.contactGravity, massFactor=scheme.analyticWallMass)
    torch.testing.assert_close(actual, expected)
    # Gravity-off contact pressure needs no first moment at all.
    scheme.iisph.contactGravity = False
    del integrals.out['Cov']
    actual = computeIISPHBoundaryPressureAcceleration(
        state, None, None, None, wallGradient, integrals, points, gravity,
        contactGravity=scheme.iisph.contactGravity, massFactor=scheme.analyticWallMass)
    torch.testing.assert_close(actual, torch.zeros_like(x))


def test_nativeIisphSampledGravityUsesTheNearestPeriodicWallImage():
    base = _nativeIisphFixture('cpu', 2, True)
    wrapped = _nativeIisphFixture('cpu', 2, True, wrappedWall=True)
    torch.testing.assert_close(wrapped.coefficients.densityRatio, base.coefficients.densityRatio, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(wrapped.boundaryAcceleration, base.boundaryAcceleration, rtol=3e-6, atol=3e-6)
    torch.testing.assert_close(wrapped.coefficients.diagonal, base.coefficients.diagonal, rtol=3e-6, atol=3e-6)


def test_iisphPoliciesAndPressureCarryDoNotLeakBetweenSolves():
    from dataclasses import asdict
    from warpSPH.modules.iisph import solveIISPH
    from warpSPH.configurations import IISPHConfig
    first = _nativeIisphFixture('cpu', 2, True)
    other = _nativeIisphFixture('cpu', 2, False)
    short = IISPHConfig(minIterations=2, maxIterations=2, relaxationFactor=.25)
    longer = IISPHConfig(minIterations=6, maxIterations=6, relaxationFactor=.4)
    original = asdict(short), asdict(longer)
    previous = torch.full_like(first.coefficients.restVolumes, 1000.)
    carry = previous.clone()
    first.scheme.iisph, other.scheme.iisph = short, longer
    before = solveIISPH(first.state, first.config, first.scheme, torch.zeros_like(first.state.velocities),
                        .001, first.adjacency, previous,
                        contactPoints=first.contactPoints, gravity=first.gravity)
    between = solveIISPH(other.state, other.config, other.scheme, torch.zeros_like(other.state.velocities),
                         .001, other.adjacency, previous,
                         contactPoints=other.contactPoints, gravity=other.gravity)
    after = solveIISPH(first.state, first.config, first.scheme, torch.zeros_like(first.state.velocities),
                       .001, first.adjacency, previous,
                        contactPoints=first.contactPoints, gravity=first.gravity)
    assert (before.iterations, between.iterations, after.iterations) == (2, 6, 2)
    assert (asdict(short), asdict(longer)) == original
    assert torch.equal(previous, carry)
    assert torch.equal(before.pressure, after.pressure)
    assert torch.equal(before.acceleration, after.acceleration)


def test_iisphProjectionRejectsNonfiniteResultsBeforeStateAdvance(monkeypatch):
    from warpSPH.modules.iisph import solveIISPH, solver
    from warpSPH.configurations import IISPHConfig
    fixture = _nativeIisphFixture('cpu', 2, True)
    fixture.scheme.iisph = IISPHConfig()
    state = fixture.state
    before = state.positions.clone(), state.velocities.clone()
    monkeypatch.setattr(solver, 'runStationary', lambda *a, **kw: NS(termination='nonfinite'))
    with pytest.raises(FloatingPointError, match='Nonfinite IISPH'):
        solveIISPH(state, fixture.config, fixture.scheme, torch.zeros_like(state.velocities), .01, fixture.adjacency,
                   contactPoints=fixture.contactPoints, gravity=fixture.gravity)
    assert torch.equal(state.positions, before[0])
    assert torch.equal(state.velocities, before[1])



@pytest.mark.parametrize('requested', ['semiImplicitEuler', 'rungeKutta4'])
def test_iisphSelectsStandardSemiImplicitEuler(requested):
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, prepareContext, CaseSpec
    from warpSPH.systems.incompressible import IISPHSystem
    from warpSPHIntegrators.schemes import IntegrationSchemeType

    importAll()
    ctx = prepareContext(getCase('tgv'), CaseSpec(scheme='iisph', integrationScheme=requested))
    assert ctx.bundle.SimulationSystem is IISPHSystem
    assert ctx.integrator.identifier == IntegrationSchemeType.semiImplicitEuler
    assert ctx.config.integrationScheme == IntegrationSchemeType.semiImplicitEuler
    assert len(tuple(ctx.bundle)) == 7


@pytest.mark.parametrize('override', [False, True])
def test_iisphRhsAndIntegratorOwnSeparateUpdates(monkeypatch, override):
    from warpSPH.schemes.incompressible import iisph
    from warpSPH.systems.incompressible import IISPHSystem, IncompressibleState
    from warpSPH.modules.iisph import IISPHResult
    from warpSPH.configurations.moduleConfigurations.boundaryConditions import BoundaryCondition, BoundaryConditionType
    from warpSPHIntegrators.schemes import get_scheme, IntegrationSchemeType

    fixture = _nativeIisphFixture('cpu', 2, False)
    fields = vars(fixture.state)
    kinds = fields['kinds']
    state = IncompressibleState(**fields, materials=torch.zeros_like(kinds),
        UIDs=torch.arange(len(kinds), dtype=torch.int32), UIDcounter=len(kinds), pressures=None)
    fluid = kinds == 0
    state.velocities[fluid] = .2
    system = IISPHSystem(state=state, domain=fixture.config.domain, t=.5, boundarySupport=.8)
    config, scheme = fixture.config, fixture.scheme
    config.verletScale = 1.
    scheme.rigidBodies, scheme.boundaryConditions = [], []
    acceleration = torch.full_like(state.velocities, .7)
    pressureAcceleration = torch.full_like(state.velocities, -.3)
    monkeypatch.setattr(iisph, 'computeGravity', lambda *a: acceleration)
    monkeypatch.setattr(iisph, 'computeBodyForce', lambda *a: torch.zeros_like(acceleration))
    monkeypatch.setattr(iisph, 'computeVelocityDiffusion', lambda *a, **kwargs: torch.zeros_like(acceleration))
    previousPressures = []

    def pressure(current, config, scheme, nonpressure, dt, adjacency, **kwargs):
        torch.testing.assert_close(nonpressure, acceleration)
        value = kwargs['previousPressure']
        previousPressures.append(None if value is None else value.clone())
        return IISPHResult(torch.full_like(current.masses, 200.), pressureAcceleration,
                           2, 0., 'tolerance', torch.zeros_like(current.masses))

    monkeypatch.setattr(iisph, 'solveIISPH', pressure)
    if override:
        def sdf(x):
            distance = torch.ones_like(x[:, 0])
            distance[0] = distance[-1] = -1
            return distance, torch.zeros_like(x)
        scheme.boundaryConditions = [BoundaryCondition(BoundaryConditionType.constant, sdf,
            updateFunctions={'dvdt': lambda *a: torch.full_like(state.velocities, 3.)})]
    expectedAcceleration = torch.where(fluid[:, None], acceleration+pressureAcceleration, 0)
    if override:
        expectedAcceleration[0] = 3.
    before = {key: value.clone() for key, value in fields.items() if isinstance(value, torch.Tensor)}
    dt = .01
    stage = system.initializeNewState()
    update, adjacency, evaluated, _ = iisph.iisph_step(stage, dt, config, scheme)
    torch.testing.assert_close(stage.state.positions, before['positions'], rtol=0, atol=0)
    torch.testing.assert_close(stage.state.velocities, before['velocities'], rtol=0, atol=0)
    torch.testing.assert_close(update.dvdt, expectedAcceleration)
    torch.testing.assert_close(update.drhodt, torch.zeros_like(state.densities))
    density = evaluated.densities.clone()
    previousPressures.clear()

    method = get_scheme(IntegrationSchemeType.semiImplicitEuler)
    result = method.function(system, dt, iisph.iisph_step, config=config, schemeConfig=scheme)
    expectedVelocity = before['velocities'] + dt*expectedAcceleration
    expectedPosition = before['positions'] + dt*expectedVelocity
    torch.testing.assert_close(result.state.state.velocities, expectedVelocity, rtol=0, atol=0)
    torch.testing.assert_close(result.state.state.positions, expectedPosition, rtol=0, atol=0)
    torch.testing.assert_close(result.state.state.densities, density, rtol=0, atol=0)
    torch.testing.assert_close(result.state.state.pressures, torch.full_like(state.masses, 200.))
    assert result.state.t == .5+dt and len(result.stages) == 1
    assert len(previousPressures) == 1 and previousPressures[0] is None
    # Initial input ownership and fixed wall particles survive finalization.
    for key, value in before.items():
        torch.testing.assert_close(getattr(system.state, key), value, rtol=0, atol=0)
    assert system.state.pressures is None
    torch.testing.assert_close(result.state.state.positions[~fluid], before['positions'][~fluid])
    result2 = method.function(result.state, dt, iisph.iisph_step, config=config, schemeConfig=scheme)
    assert len(previousPressures) == 2 and result2.state.t == .5+2*dt
    torch.testing.assert_close(previousPressures[-1], torch.full_like(state.masses, 200.))

@pytest.mark.parametrize("scheme", ["dfsph", "dfsphReference", "omniIncompressible", "band2018pb"])
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


@pytest.mark.parametrize("name", ["dfsph", "dfsphReference", "omniIncompressible", "band2018pb"])
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


def _denseDfsphMatrices(fixture):
    """Independent dense C/F with static walls and optional contact pressure."""
    state, coefficients = fixture.state, fixture.coefficients
    volume, ratio = coefficients.restVolumes, coefficients.densityRatio
    _, gradient = _denseIisphKernels(state, fixture.config)
    n, dim = state.positions.shape
    fluid = state.kinds == 0
    wall = torch.einsum('j,ijd->id', volume[~fluid], gradient[:, ~fluid])
    C = volume.new_zeros((n, n*dim))
    F = volume.new_zeros((n*dim, n))
    T = volume.new_zeros((n, n))
    if fixture.contactPoints is not None:
        weights, _ = _denseIisphKernels(state, fixture.config, fixture.contactPoints)
        for i in range(n):
            if fluid[i] and wall[i].square().sum() > 1.e-24:
                T[i, fluid] = weights[i, fluid]*volume[fluid]/(weights[i, fluid]@volume[fluid])
    for i in range(n):
        if not fluid[i]:
            continue
        rows = slice(i*dim, (i+1)*dim)
        C[i, rows] += wall[i]
        F[rows, i] -= wall[i]/ratio[i]**2
        F[rows] -= wall[i, :, None]*T[i, None, :]
        for j in range(n):
            if fluid[j]:
                C[i, rows] += volume[j]*gradient[i, j]
                C[i, j*dim:(j+1)*dim] -= volume[j]*gradient[i, j]
                F[rows, i] -= volume[j]*gradient[i, j]/ratio[i]**2
                F[rows, j] -= volume[j]*gradient[i, j]/ratio[j]**2
    return C, F


@pytest.mark.parametrize('mode', ['density', 'divergence'])
@pytest.mark.parametrize('dim,contact', [(2, False), (2, True), (3, False), (3, True)])
@pytest.mark.parametrize('clamp', [False, True])
def test_dfsphProjectionsMatchIndependentDenseJacobi(mode, dim, contact, clamp):
    from warpSPH.configurations import DFSPHConfig
    from warpSPH.modules.dfsph import solveDFSPHProjection

    fixture = _nativeIisphFixture('cpu', dim, contact)
    state, c, sc = fixture.state, fixture.coefficients, fixture.scheme
    sc.dfsph = DFSPHConfig(minDivergenceDensityRatio=0.)
    for policy in (sc.dfsph.density, sc.dfsph.divergence):
        policy.minIterations = policy.maxIterations = 6
        policy.maxErrorPercent, policy.relaxationFactor, policy.clampPressure = 0, .25, clamp
    C, F = _denseDfsphMatrices(fixture)
    fluid, rho0 = state.kinds == 0, sc.fluid.restDensity
    velocity = torch.sin(torch.arange(state.positions.numel(), dtype=state.masses.dtype)).reshape_as(state.positions)
    velocity[~fluid] = 0
    dt, oldDt = .02, .01
    scale = dt**(2 if mode == 'density' else 1)
    previous = rho0*torch.linspace(.2, 1.4, len(fluid), dtype=state.masses.dtype)
    impulse = previous*oldDt**(2 if mode == 'density' else 1)
    deficient = fluid & (torch.arange(len(fluid)) < 2)
    before = {name: value.clone() for name, value in vars(state).items() if torch.is_tensor(value)}
    result = solveDFSPHProjection(state, fixture.config, sc, dt, fixture.adjacency,
        c.restVolumes, c.densityRatio, c.wallGradient, c.gradientSum, c.diagonal,
        mode=mode, velocity=velocity, pressureImpulse=impulse,
        boundaryAcceleration=fixture.boundaryAcceleration, deficient=deficient,
        contactPoints=fixture.contactPoints, contactAdjacency=c.contactAdjacency,
        contactDenominator=c.contactDenominator)
    known = fixture.boundaryAcceleration if mode == 'density' else torch.zeros_like(velocity)
    b = -C@velocity.flatten()
    if mode == 'density':
        b = 1-c.densityRatio+dt*b-dt**2*(C@known.flatten())
    b = torch.where(fluid, b, 0)
    torch.testing.assert_close(result.source, b, rtol=3.e-6, atol=3.e-6)
    A = scale*(C@F)
    diagonal = A.diagonal()
    valid = fluid & (diagonal != 0)
    excluded = deficient if clamp and mode == 'divergence' else torch.zeros_like(fluid)
    old = torch.where(fluid, .5*impulse/(rho0*scale), 0)
    if clamp:
        old = torch.where((b < 0) & ~excluded, old.clamp_min(0), 0)
    for _ in range(6):
        residual = b-A@old
        new = old+.25*residual/torch.where(valid, diagonal, 1)
        error = (-residual).clamp_min(0) if clamp else residual.abs()
        error = rho0*torch.where(fluid & ~excluded, error, 0).sum()/fluid.sum()
        old = torch.where(valid, new.clamp_min(0) if clamp else new, 0)
    torch.testing.assert_close(result.pressure/rho0, old, rtol=1.e-5, atol=2.e-5)
    torch.testing.assert_close(result.acceleration.flatten(), F@old+known.flatten(), rtol=2.e-5, atol=2.e-5)
    assert result.iterations == 6
    assert result.error == pytest.approx(float(error), rel=3.e-5, abs=.005)
    for name, value in before.items():
        torch.testing.assert_close(getattr(state, name), value, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['density', 'divergence'])
def test_dfsphCorrectionAndWarmHistoryAreInvariantUnderDensityUnitsAndDt(mode):
    from warpSPH.configurations import DFSPHConfig
    from warpSPH.modules.dfsph import solveDFSPHProjection

    corrections, histories = [], []
    for rho0, dt in ((1., .01), (1000., .01), (1000., .000001)):
        f = _nativeIisphFixture('cpu', 2, False, rho0=rho0)
        state, c, sc = f.state, f.coefficients, f.scheme
        sc.dfsph = DFSPHConfig()
        for policy in (sc.dfsph.density, sc.dfsph.divergence):
            policy.minIterations = policy.maxIterations = 8
            policy.maxErrorPercent, policy.relaxationFactor, policy.clampPressure = 0, .25, False
        # With fixed density defect (CD) or velocity (DF), the displacement /
        # velocity correction and its scaled pressure history are dt invariant.
        velocity = torch.zeros_like(state.velocities) if mode == 'density' else -state.positions
        scale = dt**(2 if mode == 'density' else 1)
        history = torch.linspace(.001, .008, len(state.masses), dtype=state.masses.dtype)*rho0
        result = solveDFSPHProjection(state, f.config, sc, dt, f.adjacency,
            c.restVolumes, c.densityRatio, c.wallGradient, c.gradientSum, c.diagonal,
            mode=mode, velocity=velocity, pressureImpulse=history)
        corrections.append(scale*result.acceleration)
        histories.append(scale*result.pressure/rho0)
    for sequence in (corrections, histories):
        for value in sequence[1:]:
            torch.testing.assert_close(value, sequence[0], rtol=3.e-5, atol=3.e-6)


@pytest.mark.parametrize('rho0,dt', [(1., .001), (1., .002), (1000., .001)])
def test_dfsphDivergenceAloneRemovesCompressionAndPreservesPeriodicVortex(rho0, dt):
    """A velocity projection must recover a solenoidal flow without density correction."""
    import math
    from warpSPHCore import DomainDescription, KernelFunctions, SupportScheme, buildVerletList
    from warpSPHCore.type_config import get_torch_precision
    from warpSPH.configurations import IISPHConfig, DFSPHConfig
    from warpSPH.modules.iisph import computeIISPHCoefficients, computeIISPHDensityRate
    from warpSPH.modules.dfsph import solveDFSPHProjection

    dtype = get_torch_precision()
    dx = 2 * math.pi / 16
    line = (torch.arange(16, dtype=dtype) + .5) * dx
    x = torch.stack(torch.meshgrid(line, line, indexing='ij'), -1).reshape(-1, 2).contiguous()
    n = len(x)
    state = NS(positions=x, supports=torch.full((n,), 4*dx, dtype=dtype),
        masses=torch.full((n,), rho0*dx*dx, dtype=dtype),
        densities=torch.full((n,), rho0, dtype=dtype),
        kinds=torch.zeros(n, dtype=torch.int32), velocities=torch.zeros_like(x))
    config = NS(domain=DomainDescription(torch.zeros(2, dtype=dtype),
        torch.full((2,), 2*math.pi, dtype=dtype), torch.ones(2, dtype=torch.bool), 2),
        kernel=KernelFunctions.Wendland4, n_h=4., calibrateNormalization=False)
    scheme = NS(fluid=NS(restDensity=rho0), rigidBodies=[],
        iisph=IISPHConfig(), dfsph=DFSPHConfig())
    scheme.dfsph.warmStart = False
    policy = scheme.dfsph.divergence
    policy.clampPressure = False
    policy.maxErrorPercent = 1.e-7
    policy.maxIterations = 100
    policy.relaxationFactor = .25
    adjacency = buildVerletList(state, config.domain, supportMode=SupportScheme.SuperSymmetric)
    geometry = computeIISPHCoefficients(state, config, scheme, adjacency)

    # The symmetric periodic lattice makes the analytic Taylor-Green velocity
    # solenoidal for C. Add a longitudinal Fourier mode with nonzero divergence.
    vortex = torch.stack((-torch.sin(x[:, 0])*torch.cos(x[:, 1]),
                           torch.cos(x[:, 0])*torch.sin(x[:, 1])), 1)
    velocity = vortex + .1*torch.sin(2*x)
    result = solveDFSPHProjection(state, config, scheme, dt, adjacency,
        *geometry[:5], mode='divergence', velocity=velocity)
    projected = velocity + dt*result.acceleration
    rate = computeIISPHDensityRate(state, config, adjacency, projected, geometry[0], geometry[2])

    assert result.termination == 'converged'
    assert float(rate.square().mean().sqrt()) < 2.e-5
    torch.testing.assert_close(projected, vortex, atol=2.e-5, rtol=2.e-5)
    # Internal pressure forces must not change total momentum.
    momentumChange = (state.masses[:, None]*(projected-velocity)).sum(0)
    assert float(momentumChange.norm()/state.masses.sum()) < 2.e-6


def test_dfsphDeficientRowsRetainRawSourceAndUpdateButAreExcludedFromStatistic():
    from warpSPH.configurations import DFSPHConfig
    from warpSPH.modules.dfsph import solveDFSPHProjection

    f = _nativeIisphFixture('cpu', 2, False)
    state, c, sc = f.state, f.coefficients, f.scheme
    sc.dfsph = DFSPHConfig(minDivergenceDensityRatio=0.)
    sc.dfsph.divergence.maxIterations = 1
    fluid = state.kinds == 0
    velocity = -state.positions
    velocity[~fluid] = 0
    result = solveDFSPHProjection(state, f.config, sc, .01, f.adjacency,
        c.restVolumes, c.densityRatio, c.wallGradient, c.gradientSum, c.diagonal,
        mode='divergence', velocity=velocity, deficient=fluid,
        pressureImpulse=torch.ones_like(state.masses))
    assert bool((result.source[fluid] < 0).any())
    assert bool((result.pressure[fluid] > 0).any())
    assert result.error == 0 and result.iterations == 1


def test_dfsphStagesShareGeometryProjectBeforeViscosityAndAdvanceOnce(monkeypatch, tmp_path):
    from warpSPH.schemes.incompressible import dfsph
    from warpSPH.systems.incompressible import DFSPHState, DFSPHSystem
    from warpSPH.configurations import DFSPHConfig
    from warpSPH.modules.iisph import IISPHResult
    from warpSPHIntegrators.schemes import ProjectedStep
    from warpSPH.io.hdf5 import dumpState, loadState
    import h5py

    f = _nativeIisphFixture('cpu', 2, False)
    fields, sc, config = vars(f.state), f.scheme, f.config
    kinds = fields['kinds']
    state = DFSPHState(**fields, materials=torch.zeros_like(kinds),
        UIDs=torch.arange(len(kinds), dtype=torch.int32), UIDcounter=len(kinds),
        soundspeeds=torch.full_like(f.state.masses, 7.))
    fluid = kinds == 0
    state.velocities[fluid] = .2
    sc.dfsph, sc.boundaryConditions = DFSPHConfig(), []
    config.verletScale = 1.
    system = DFSPHSystem(state=state, domain=config.domain, t=.5, boundarySupport=.8)
    events, histories = [], []
    originalGeometry = dfsph.computeIISPHProjectionGeometry

    def geometry(*args, **kwargs):
        events.append('geometry')
        return originalGeometry(*args, **kwargs)

    monkeypatch.setattr(dfsph, 'computeIISPHProjectionGeometry', geometry)
    monkeypatch.setattr(dfsph, 'computeGravity', lambda *a: torch.full_like(state.positions, .7))
    monkeypatch.setattr(dfsph, 'computeBodyForce', lambda *a: torch.zeros_like(state.positions))

    def viscosity(current, *args, **kwargs):
        events.append('viscosity')
        torch.testing.assert_close(current.velocities[fluid], expectedProjected)
        return torch.zeros_like(current.velocities)

    monkeypatch.setattr(dfsph, 'computeVelocityDiffusion', viscosity)

    def projection(current, config, scheme, dt, adjacency, *args, mode, velocity, **kwargs):
        events.append(mode)
        if mode == 'density':
            torch.testing.assert_close(velocity[fluid], expectedProjected+dt*.7)
        histories.append((mode, kwargs['pressureImpulse']))
        value = .2 if mode == 'divergence' else -.3
        acceleration = torch.where(fluid[:, None], torch.full_like(current.velocities, value), 0)
        pressure = torch.where(fluid, torch.full_like(current.masses, 100. if mode == 'divergence' else 200.), 0)
        return IISPHResult(pressure, acceleration, 2, 0., 'converged', torch.zeros_like(pressure))

    monkeypatch.setattr(dfsph, 'solveDFSPHProjection', projection)
    snapshot = {name: value.clone() for name, value in vars(state).items() if torch.is_tensor(value)}
    dt = .01
    expectedProjected = state.velocities[fluid]+dt*.2
    first = ProjectedStep()(system, dt, dfsph.dfsph_step, config=config, schemeConfig=sc)
    result = first.state
    assert events == ['geometry', 'divergence', 'viscosity', 'density']
    assert result.t == .5+dt and len(first.stages) == 1
    expectedVelocity = snapshot['velocities'].clone()
    expectedVelocity[fluid] = expectedProjected+dt*.4
    expectedPosition = snapshot['positions']+dt*expectedVelocity
    torch.testing.assert_close(result.state.velocities, expectedVelocity)
    torch.testing.assert_close(result.state.positions, expectedPosition)
    torch.testing.assert_close(result.state.soundspeeds, snapshot['soundspeeds'], rtol=0, atol=0)
    torch.testing.assert_close(result.state.densityPressureImpulse, result.state.pressures*dt**2)
    torch.testing.assert_close(result.state.divergencePressureImpulse, result.state.divergencePressures*dt)
    assert histories == [('divergence', None), ('density', None)]
    for name, value in snapshot.items():
        torch.testing.assert_close(getattr(system.state, name), value, rtol=0, atol=0)
    # The dedicated fields survive a checkpoint and a new dt without leaking
    # into soundspeeds, state pressure units, or another system's histories.
    with h5py.File(tmp_path/'dfsph.h5', 'w') as output:
        dumpState(result.state, output.create_group('state'))
    with h5py.File(tmp_path/'dfsph.h5', 'r') as source:
        restored = loadState(source['state'], 'cpu', DFSPHState)
    for name in ('pressures', 'divergencePressures', 'densityPressureImpulse', 'divergencePressureImpulse'):
        torch.testing.assert_close(getattr(restored, name), getattr(result.state, name), rtol=0, atol=0)
    result.state = restored
    dt2 = .02
    expectedProjected = restored.velocities[fluid]+dt2*.2
    second = ProjectedStep()(result, dt2, dfsph.dfsph_step, config=config, schemeConfig=sc)
    assert second.state.t == .5+dt+dt2
    torch.testing.assert_close(histories[2][1], restored.divergencePressureImpulse)
    torch.testing.assert_close(histories[3][1], restored.densityPressureImpulse)
    assert second.state.boundarySupport == .8


def test_dfsphRejectsNonfiniteSolveBeforeMutatingInputs(monkeypatch):
    from warpSPH.configurations import DFSPHConfig
    from warpSPH.modules.dfsph import solver

    f = _nativeIisphFixture('cpu', 2, False)
    f.scheme.dfsph = DFSPHConfig()
    state, c = f.state, f.coefficients
    before = state.velocities.clone()
    monkeypatch.setattr(solver, 'computeIISPHPressureOperator',
                        lambda *a, **kw: torch.full_like(state.densities, float('nan')))
    with pytest.raises(FloatingPointError, match='Nonfinite DFSPH divergence'):
        solver.solveDFSPHProjection(state, f.config, f.scheme, .01, f.adjacency,
            c.restVolumes, c.densityRatio, c.wallGradient, c.gradientSum, c.diagonal,
            mode='divergence', velocity=state.velocities)
    torch.testing.assert_close(state.velocities, before, rtol=0, atol=0)


@pytest.mark.parametrize('kernel', ['Wendland2', 'Wendland4'])
def test_dfsphDefaultRelaxationDampsTheWideSupportPressureMode(kernel):
    """A matched exact diagonal does not make reference omega=.5 stable at n_h=4."""
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, prepareContext, CaseSpec
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPH.modules.iisph import computeIISPHProjectionGeometry
    from warpSPH.modules.dfsph import solveDFSPHProjection
    from warpSPHCore import buildVerletList, SupportScheme

    importAll()
    case = getCase('tgv')
    ctx = prepareContext(case, CaseSpec(scheme='dfsph', device='cpu', nx=16,
        n_h=4., kernel=kernel, calibrateNormalization=True, L=2*torch.pi,
        params={**case.params, 'relaxSteps': 0, 'jitter': 0.}, quiet=True))
    system = case.buildSystem(ctx)
    case.initialConditions(ctx, system)
    initializeProjectionDensity(ctx, system)
    state, cfg, sc = system.state, ctx.config, ctx.schemeConfig
    adj = buildVerletList(state, cfg.domain, supportMode=SupportScheme.SuperSymmetric)
    V, s, G, S, diagonal, *_ = computeIISPHProjectionGeometry(state, cfg, sc, adj)
    # Independent periodic C and its highest Jacobi mode. Equal masses make
    # F_chi=-C^T, so H=C C^T and the diagonal is spatially constant here.
    x = state.positions.double()
    distance = x[:, None]-x[None, :]
    length = (cfg.domain.max-cfg.domain.min).double()
    distance -= torch.round(distance/length)*length
    h = state.supports.double()[:, None]
    radius = distance.norm(dim=-1)/h
    cutoff = (1-radius).clamp_min(0)
    derivative = (-20*(7/torch.pi)/h**4*cutoff**3 if kernel == 'Wendland2'
                  else -56/3*(9/torch.pi)/h**4*(5*radius+1)*cutoff**5)
    gradient = derivative[..., None]*distance
    C = -V.double()[None, :, None]*gradient
    index = torch.arange(len(x))
    C[index, index] += torch.einsum('j,ijd->id', V.double(), gradient)
    C = C.reshape(len(x), -1)
    H = C@C.T
    D = H.diagonal().sqrt()
    eigenvalues, eigenvectors = torch.linalg.eigh(H/D[:, None]/D[None, :])
    maximum = float(eigenvalues[-1])
    policy = sc.dfsph.divergence
    assert policy.relaxationFactor < 2/maximum < .5
    error = eigenvectors[:, -1].to(state.masses)
    dt = .01
    policy.maxIterations = 1
    result = solveDFSPHProjection(state, cfg, sc, dt, adj, V, s, G, S, diagonal,
        mode='divergence', velocity=torch.zeros_like(state.velocities),
        pressureImpulse=error*sc.fluid.restDensity*dt)
    assert (result.pressure/sc.fluid.restDensity).norm() < .9*(.5*error).norm()


@pytest.mark.parametrize('name,value', [('maxErrorPercent', -1.), ('maxErrorPercent', float('nan')),
                                        ('relaxationFactor', 0.), ('maxIterations', 0)])
def test_dfsphPoliciesRejectInvalidValues(name, value):
    from warpSPH.configurations import DFSPHConfig
    config = DFSPHConfig()
    setattr(config.divergence, name, value)
    with pytest.raises(ValueError, match='DFSPH'):
        config.validate()


def test_dfsphPartialPolicyDecoderPreservesIndependentDensityDefaults():
    from warpSPH.configurations.moduleConfigurations.dfsph import dictToDFSPHConfig
    config = dictToDFSPHConfig({'density': {'maxErrorPercent': .02}, 'warmStart': False})
    assert config.density.minIterations == 2 and config.density.maxErrorPercent == .02
    assert config.divergence.minIterations == 1 and config.divergence.maxErrorPercent == .1
    assert config.density is not config.divergence and not config.warmStart


def test_nativeContactGeometryKeepsParticlePrecisionWhenProviderUsesDouble():
    from warpSPH.modules.iisph import computeIISPHContactPoints
    state = NS(positions=torch.tensor([[.2, .1], [.3, .2]], dtype=torch.float32),
               kinds=torch.zeros(2, dtype=torch.int32))

    def distance(positions):
        normal = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64)
        return positions[:, 1].double(), normal, torch.tensor([True, False])

    points, failed = computeIISPHContactPoints(state, NS(signed_distance=distance))
    assert points.dtype == state.positions.dtype and points.device == state.positions.device
    torch.testing.assert_close(points, torch.tensor([[.2, 0.], [.3, .2]], dtype=torch.float32))
    assert bool(failed)


def test_dfsphLatticeInitializationUsesItsNativeDensityPolicy(monkeypatch):
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, prepareContext, CaseSpec
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPH.modules import dfsph, iisph
    import warpSPH.modules as modules

    importAll()
    case = getCase('tgv')
    ctx = prepareContext(case, CaseSpec(scheme='dfsph', device='cpu', nx=16,
        n_h=4., kernel='Wendland4', calibrateNormalization=True, L=2*torch.pi,
        params={**case.params, 'relaxSteps': 2, 'jitter': .001}, quiet=True))
    system = case.buildSystem(ctx)
    calls = []
    original = dfsph.solveDFSPHProjection

    def observe(state, config, sc, *args, **kwargs):
        assert not config.calibrateNormalization
        assert not sc.dfsph.density.clampPressure
        assert sc.dfsph.density.relaxationFactor == .3
        assert kwargs['mode'] == 'density'
        assert not bool(state.velocities.any())
        calls.append(1)
        return original(state, config, sc, *args, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail('Native DFSPH initialization must use its own density projection')

    monkeypatch.setattr(dfsph, 'solveDFSPHProjection', observe)
    monkeypatch.setattr(iisph, 'solveIISPH', unexpected)
    monkeypatch.setattr(modules, 'solveIncompressible', unexpected)
    case.initialConditions(ctx, system)
    assert calls == [1, 1]
    masses = system.state.masses.clone()
    initializeProjectionDensity(ctx, system)
    torch.testing.assert_close(system.state.masses, masses, rtol=0, atol=0)


def test_periodic_density_source_policy_is_explicit_and_preserves_raw_density():
    from warpSPH.modules.iisph.solver import applyDensitySourcePolicy
    state=NS(kinds=torch.zeros(3,dtype=torch.int32),densities=torch.tensor([1.01,1.02,.99]))
    cfg=NS(domain=NS(periodic=torch.tensor([True,True])))
    source=torch.tensor([.001,.002,-.001])
    before=state.densities.clone()
    assert applyDensitySourcePolicy(source,state,cfg,'absolute') is source
    centered=applyDensitySourcePolicy(source,state,cfg,'periodicMeanFree')
    assert centered.sum().abs()<1e-9
    torch.testing.assert_close(centered[1]-centered[0],source[1]-source[0])
    assert torch.equal(state.densities,before)
    cfg.domain.periodic[0]=False
    with pytest.raises(ValueError,match='fully periodic'):
        applyDensitySourcePolicy(source,state,cfg,'periodicMeanFree')


def test_new_projection_policies_default_to_existing_method_and_roundtrip():
    from warpSPH.configurations.moduleConfigurations.dfsph import DFSPHConfig,dictToDFSPHConfig
    from warpSPH.configurations.moduleConfigurations.iisph import IISPHConfig
    assert DFSPHConfig().ordering=='df_np_cd'
    assert IISPHConfig().densitySourcePolicy=='absolute'
    assert dictToDFSPHConfig({'ordering':'np_df_shift'}).ordering=='np_df_shift'
    p=IISPHConfig(densitySourcePolicy='periodicMeanFree')
    with pytest.raises(ValueError,match='signed'):p.validate()
    p.clampPressure=False;p.validate()


@pytest.mark.parametrize('order',['np_df_cd','np_df_shift'])
def test_composed_projection_preserves_uniform_periodic_transport(order):
    import math
    from warpSPH.runner import CaseSpec,prepareContext
    from warpSPH.cases.periodic.tgv import tgvCase
    from warpSPH.caseUtils.projection import initializeProjectionDensity
    from warpSPH.modules.dfsph.composition import advanceDFSPH
    ctx=prepareContext(tgvCase,CaseSpec(scheme='dfsph',device='cpu',nx=12,
        L=2*math.pi,periodic=True,n_h=4.,kernel='Wendland4',
        params={**tgvCase.params,'relaxSteps':0}))
    ctx.config.calibrateNormalization=False
    sc=ctx.schemeConfig
    sc.iisph.densitySourcePolicy='periodicMeanFree'
    sc.diffusionParams.viscidNu=0.
    sc.dfsph.warmStart=False
    system=tgvCase.buildSystem(ctx)
    initializeProjectionDensity(ctx,system)
    st=system.state
    st.velocities[:]=torch.tensor([.7,-.2])
    st.pressures=torch.zeros_like(st.masses)
    st.divergencePressureImpulse=None;st.densityPressureImpulse=None
    x,mass=st.positions.clone(),st.masses.clone()
    system._projectedStepActive=True
    update,_,after,_=advanceDFSPH(system,.01,ctx.config,sc,order)
    expected=x+.01*torch.tensor([.7,-.2])
    expected=torch.remainder(expected,2*math.pi)
    torch.testing.assert_close(after.velocities,torch.tensor([.7,-.2]).expand_as(after.velocities),atol=2e-5,rtol=0)
    torch.testing.assert_close(after.positions,expected,atol=2e-6,rtol=0)
    assert torch.equal(after.masses,mass)
    if order=='np_df_shift':
        events=update.orderAudit['events']
        assert events.index('drift')<events.index('rebuild_after_drift')<events.index('density_position')<events.index('velocity_resample')
