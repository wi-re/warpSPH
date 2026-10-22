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
    assert len(restored.regions) == 1
    assert torch.equal(restored.regions[0].sdf(torch.tensor([[2.0, 3.0]])), torch.tensor([2.0]))
    for sub in ("pressureSolver", "divergenceFreeSolver"):
        del encoded["solverConfig"][sub]["convergenceCheckSchedule"]
    encoded.pop("regions")
    encoded.pop("rigidBodies")
    legacy = dictToIncompressibleSPHConfig(encoded)
    assert legacy.solverConfig.pressureSolver.convergenceCheckSchedule == "auto"
    assert legacy.regions == legacy.rigidBodies == []


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
