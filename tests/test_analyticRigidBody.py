"""`buildAnalyticRigidBody`: a RigidBody whose boundary is analytic (no boundary particles).

The existing rigid-body machinery must run on it unchanged: `integrateRigidBody` advances the pose, and
`updateBodyParticlesWCSPH` is an identity on a state that has no rows of that body (empty gather / scatter).
Mass, centre of mass and inertia come from the representation (`warpSPHBoundaries`' `Body.massProperties`).
"""
import math

import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from types import SimpleNamespace  # noqa: E402

from warpSPH.configurations.region import BCType  # noqa: E402
from warpSPH.rigidBody import buildAnalyticRigidBody, integrateRigidBody, updateBodyParticlesWCSPH  # noqa: E402
from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState  # noqa: E402
from warpSPHBoundaries.scene import Body  # noqa: E402
from warpSPHBoundaries.scene.implicitBodies import DiskBody  # noqa: E402
from warpSPHBoundaries.scene.scene import BoxRep, ImplicitRep, Scene  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def fluidState(n=50):
    x = torch.rand((n, 2), device=DEV, dtype=torch.float64)
    z = torch.zeros(n, device=DEV, dtype=torch.float64)
    return WeaklyCompressibleState(
        positions=x, velocities=torch.zeros_like(x), supports=z + 0.1, masses=z + 1.0, densities=z + 1.0,
        kinds=torch.zeros(n, dtype=torch.int32, device=DEV), materials=torch.zeros(n, dtype=torch.int32, device=DEV),
        UIDs=torch.arange(n, device=DEV), UIDcounter=n, pressures=z.clone(), soundspeeds=z + 10.0,
        ghostIndices=-torch.ones(n, dtype=torch.int32, device=DEV), ghostOffsets=x.clone())


def region(body, kind=BCType.noSlip):
    Scene([body], DEV)
    return SimpleNamespace(sdf=lambda p: (torch.zeros(len(p), device=p.device), None), kind=kind, representation=body)


def test_finite_body_mass_properties_and_pose():
    R = 0.25
    body = Body(bodyId=0, center=(0.4, 0.6), angle=0.3, linearVelocity=(0.1, 0.0), angularVelocity=2.0,
                reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=R))])
    rb = buildAnalyticRigidBody(region(body), 0, fluidState(), restDensity=2.0)
    assert math.isclose(float(rb.mass), 2.0 * math.pi * R * R, rel_tol=1e-12)
    assert math.isclose(float(rb.inertia), 0.5 * float(rb.mass) * R * R, rel_tol=1e-12)
    assert torch.allclose(rb.centerOfMass, torch.tensor([0.4, 0.6], dtype=torch.float64, device=DEV))
    assert math.isclose(float(rb.orientation), 0.3) and math.isclose(float(rb.angularVelocity), 2.0)
    assert rb.particlePositions.shape == (0, 2) and not bool(rb.particleIndices.any()) and rb.representation is body


def test_unbounded_wall_has_infinite_mass():
    body = Body(bodyId=1, reps=[BoxRep((0.0, 0.0), (1.0, 1.0), solid='outside')])
    rb = buildAnalyticRigidBody(region(body), 1, fluidState())
    assert math.isinf(float(rb.mass)) and math.isinf(float(rb.inertia))


def test_body_origin_away_from_the_centre_of_mass_is_refused():
    body = Body(bodyId=2, center=(0.0, 0.0), reps=[ImplicitRep(DiskBody(center=(0.3, 0.0), radius=0.1))])
    with pytest.raises(ValueError):
        buildAnalyticRigidBody(region(body), 2, fluidState())


def test_existing_machinery_runs_on_an_analytic_body():
    body = Body(bodyId=0, center=(0.4, 0.6), linearVelocity=(0.5, -0.25), angularVelocity=1.5,
                reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=0.2))])
    rb = buildAnalyticRigidBody(region(body), 0, fluidState())
    state = fluidState()
    before = state.positions.clone(), state.velocities.clone()
    dt = 0.01
    for _ in range(10):
        rb = integrateRigidBody(rb, 0, 0, dt)
        state = updateBodyParticlesWCSPH(state, rb)
    assert torch.equal(state.positions, before[0]) and torch.equal(state.velocities, before[1])        # no rows of this body: the fluid is untouched
    assert torch.allclose(rb.centerOfMass, torch.tensor([0.4 + 0.05, 0.6 - 0.025], dtype=torch.float64, device=DEV))
    assert math.isclose(float(rb.orientation), 1.5 * 0.1, rel_tol=1e-12)
