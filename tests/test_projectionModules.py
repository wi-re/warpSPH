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
    scheme = NS(fluid=NS(restDensity=rho0), iisph=IISPHConfig())
    state.densities = computeDensities(state, config, scheme, adj, wall=False)
    coefficients = computeIISPHCoefficients(state, config, scheme, adj, contactPoints=points)
    boundaryAcceleration = computeIISPHBoundaryPressureAcceleration(
        state, config, scheme, coefficients, None, points, g)
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
    system = NS(state=state, adjacency=None, t=0.)
    state.densities = torch.zeros_like(state.masses)
    state.pressures = None
    expected = sampledDensity.clone()
    expected[0] = prescribed
    monkeypatch.setattr(iisph, 'computeGravity', lambda *args: torch.zeros_like(state.positions))
    monkeypatch.setattr(iisph, 'computeBodyForce', lambda *args: torch.zeros_like(state.positions))
    monkeypatch.setattr(iisph, 'computeIISPHViscosity', lambda *args: torch.zeros_like(state.positions))

    class ReachedPressureSolve(Exception):
        pass

    def pressure(currentState, config, schemeConfig, velocity, dt, adjacency, **kwargs):
        from warpSPH.modules.iisph import computeIISPHCoefficients
        coefficients = computeIISPHCoefficients(currentState, config, schemeConfig, adjacency,
            boundaryIntegrals=kwargs['boundaryIntegrals'], contactPoints=kwargs['contactPoints'])
        torch.testing.assert_close(currentState.densities, expected)
        torch.testing.assert_close(coefficients.densityRatio, expected / scheme.fluid.restDensity)
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
    from warpSPH.modules.iisph import computeIISPHBoundaryIntegrals, computeIISPHCoefficients

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
    integrals = computeIISPHBoundaryIntegrals(state, config, scheme)
    computeIISPHCoefficients(state, config, scheme, system.adjacency, boundaryIntegrals=integrals)
    assert torch.equal(state.masses, calibratedMasses)
    assert scheme.analyticWallMass == factor


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA'))])
@pytest.mark.parametrize('dim,contact', [(2, False), (2, True), (3, False), (3, True)])
def test_nativeIisphForceContinuityAndExactDiagonal(device, dim, contact):
    from warpSPH.modules.iisph import computeIISPHPressureAcceleration, computeIISPHDensityRate, solveIISPH
    # Independent small dense F/C oracle. Unequal volumes, unequal density,
    # disordered positions and static samples distinguish IISPH from DFSPH.
    fixture = _nativeIisphFixture(device, dim, contact)
    state, coefficients = fixture.state, fixture.coefficients
    volume, pairs = coefficients.restVolumes, coefficients.pairs
    fluid, rho0 = state.kinds == 0, fixture.scheme.fluid.restDensity
    n = len(volume)
    F = torch.zeros((n*dim, n), dtype=volume.dtype, device=device)
    C = torch.zeros((n, n*dim), dtype=volume.dtype, device=device)
    T = torch.zeros((n, n), dtype=volume.dtype, device=device)
    if contact:
        cp = coefficients.contactWeights.pairs
        for i in range(n):
            if coefficients.contactWeights.denominator[i] > 0:
                for e in range(int(cp.offsets[i]), int(cp.offsets[i]+cp.counts[i])):
                    j = int(cp.indices[e])
                    if fluid[j]:
                        T[i, j] += volume[j]*cp.weights[e]/coefficients.contactWeights.denominator[i]
    for i in range(n):
        if not fluid[i]:
            continue
        rows = slice(i*dim, (i+1)*dim)
        C[i, rows] += coefficients.wallGradient[i]
        F[rows, i] -= coefficients.wallGradient[i]/coefficients.densityRatio[i]**2
        F[rows] -= coefficients.wallGradient[i, :, None]*T[i, None, :]
        for e in range(int(pairs.offsets[i]), int(pairs.offsets[i]+pairs.counts[i])):
            j = int(pairs.indices[e])
            if not fluid[j]:
                continue
            g = pairs.gradient[e]
            C[i, rows] += volume[j]*g
            C[i, j*dim:(j+1)*dim] -= volume[j]*g
            F[rows, i] -= volume[j]*g/coefficients.densityRatio[i]**2
            F[rows, j] -= volume[j]*g/coefficients.densityRatio[j]**2
    q = torch.linspace(.2, 1.4, n, dtype=volume.dtype, device=device)
    v = torch.sin(torch.arange(n*dim, dtype=q.dtype, device=device)).reshape(n, dim)
    torch.testing.assert_close(computeIISPHPressureAcceleration(state, q, coefficients).flatten(), F@q, rtol=4e-6, atol=4e-6)
    torch.testing.assert_close(computeIISPHDensityRate(state, v, coefficients), C@v.flatten(), rtol=4e-6, atol=4e-6)
    torch.testing.assert_close(coefficients.diagonal, (C@F).diagonal(), rtol=6e-6, atol=6e-6)

    from warpSPH.configurations import IISPHConfig
    policy = IISPHConfig(minIterations=6, maxIterations=6, relaxationFactor=.25, maxErrorPercent=0)
    previous = q*rho0
    dt = .01
    fixture.scheme.iisph = policy
    state.velocities = torch.where(fluid[:, None], v, 0)
    actual = solveIISPH(state, fixture.config, fixture.scheme, torch.zeros_like(v), dt, fixture.adjacency, previous,
                        contactPoints=fixture.contactPoints, gravity=fixture.gravity)
    torch.testing.assert_close(state.velocities, torch.where(fluid[:, None], v, 0), rtol=0, atol=0)
    b = 1-coefficients.densityRatio-dt*(C@v.flatten())-dt**2*(C@fixture.boundaryAcceleration.flatten())
    A = dt**2*C@F
    diagonal = A.diagonal()
    old = .5*previous/rho0
    old[~fluid] = 0
    valid = fluid & (diagonal.abs() > 1.e-9)
    for _ in range(6):
        off = A@old-diagonal*old
        new = torch.where(valid, (.75*old+.25*(b-off)/torch.where(valid, diagonal, 1)).clamp_min(0), 0)
        error = torch.where(new != 0, rho0*(diagonal*new+off-b), 0)[fluid].mean()
        old = new
    torch.testing.assert_close(actual.pressure/rho0, old, rtol=1.e-5, atol=1.e-5)
    torch.testing.assert_close(actual.acceleration.flatten(),
        F@old + fixture.boundaryAcceleration.flatten(), rtol=1.e-5, atol=1.e-5)
    assert actual.iterations == 6
    assert actual.error == pytest.approx(float(error), rel=2.e-5, abs=.003)


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


def test_nativeIisphViscosityIsFreeSlipAndDissipatesFluidMotion():
    from warpSPH.modules.iisph import computeIISPHViscosity
    from warpSPH.configurations import IncompressibleSPHConfig
    op = _nativeIisphFixture('cpu', 2, False)
    state = op.state
    state.densities = op.coefficients.densityRatio*op.scheme.fluid.restDensity
    config = IncompressibleSPHConfig()
    config.fluid.restDensity = op.scheme.fluid.restDensity
    config.fluid.fixedSoundSpeed = 10.
    config.diffusionParams.inviscid = False
    config.diffusionParams.viscidNu = .01
    # Stationary sampled walls exert no viscous drag on uniform fluid motion.
    state.velocities[(op.state.kinds == 0)] = torch.tensor([1., 2.], dtype=state.positions.dtype)
    from warpSPHCore import buildVerletList
    adj = buildVerletList(state, op.config.domain)
    acceleration = computeIISPHViscosity(state, op.config, config, adj)
    torch.testing.assert_close(acceleration, torch.zeros_like(acceleration))
    # Converging and separating pairs both contribute; net work is negative.
    state.velocities[(op.state.kinds == 0)] = state.positions[(op.state.kinds == 0)]
    acceleration = computeIISPHViscosity(state, op.config, config, adj)
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
    coefficients = NS(wallGradient=.8*bodyGradient.sum(0))
    expected = torch.zeros_like(x)
    for body in range(2):
        for sample in range(2):
            q = ((samples[body, sample]-points)*gravity).sum(-1)
            expected -= .8*volumes[body, sample]*q[:, None]*gradients[body, :, sample]
    expected[state.kinds != 0] = 0
    actual = computeIISPHBoundaryPressureAcceleration(
        state, None, scheme, coefficients, integrals, points, gravity)
    torch.testing.assert_close(actual, expected)
    # Gravity-off contact pressure needs no first moment at all.
    scheme.iisph.contactGravity = False
    del integrals.out['Cov']
    actual = computeIISPHBoundaryPressureAcceleration(
        state, None, scheme, coefficients, integrals, points, gravity)
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
    system = IISPHSystem(state=state, domain=fixture.config.domain, t=.5)
    config, scheme = fixture.config, fixture.scheme
    config.verletScale = 1.
    scheme.rigidBodies, scheme.boundaryConditions = [], []
    acceleration = torch.full_like(state.velocities, .7)
    pressureAcceleration = torch.full_like(state.velocities, -.3)
    monkeypatch.setattr(iisph, 'computeGravity', lambda *a: acceleration)
    monkeypatch.setattr(iisph, 'computeBodyForce', lambda *a: torch.zeros_like(acceleration))
    monkeypatch.setattr(iisph, 'computeIISPHViscosity', lambda *a: torch.zeros_like(acceleration))
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

@pytest.mark.parametrize("scheme", ["dfsphReference", "omniIncompressible", "band2018pb"])
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


@pytest.mark.parametrize("name", ["dfsphReference", "omniIncompressible", "band2018pb"])
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
