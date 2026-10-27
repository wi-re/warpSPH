"""Analytic boundary regions (`buildRegion(representation=...)`, the initializer, the dam break's `wallRepresentation`).

An analytic region is clipped against like any boundary (`filterRegion` reads its `sdf`), is not sampled
(no boundary particles, no ghost layer), and gets a RigidBody from its representation. The fluid sampled
for the particle tank and for the analytic tank is the same set of particles.
"""
import numpy as np
import pytest
import torch

from warpSPH.caseUtils.attachedBoundary import floorAttachedPolygon, polygonSolidSDF

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import buildContext, getCase  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402


def build(representation, nx=24):
    importAll()
    case = getCase('dambreak')
    spec = CaseSpec(caseName=case.name, scheme=case.scheme, params={**case.params, 'wallRepresentation': representation})
    spec = spec.merged(**case.defaults).merged(nx=nx)
    ctx = buildContext(case, spec)
    case.configureScheme(ctx)
    return ctx, case.buildSystem(ctx)


def test_analytic_tank_has_no_boundary_particles_and_one_analytic_body():
    ctx, system = build('analytic')
    kinds = system.state.kinds
    assert int((kinds == 1).sum()) == 0 and int((kinds == 2).sum()) == 0 and int((kinds == 0).sum()) > 100
    bodies = ctx.config.rigidBodies
    assert len(bodies) == 1 and bodies[0].representation is not None
    assert torch.isinf(bodies[0].mass) and bodies[0].particlePositions.shape[0] == 0


def test_fluid_is_the_same_set_of_particles_as_with_boundary_particles():
    _, wall = build('particles')
    _, ana = build('analytic')
    fp, fa = wall.state.positions[wall.state.kinds == 0], ana.state.positions[ana.state.kinds == 0]
    assert fp.shape == fa.shape
    key = lambda p: p[torch.argsort(p[:, 0] * 1e6 + p[:, 1])]
    assert torch.allclose(key(fp), key(fa))
    assert int((wall.state.kinds == 1).sum()) > 0                                    # the particle tank does have wall particles




@pytest.mark.parametrize('floor', [-.595, -.6, -.605])
@pytest.mark.parametrize('reverse', [False, True])
def test_floor_union_removes_buried_edges_without_moving_wedge(floor, reverse):
    v = np.array([[-.29, -.614], [.29, -.614], [0, -.363]])
    p = floorAttachedPolygon([-1.2, floor], [1.2, .6], v[::-1] if reverse else v)
    assert p is not None and len(p) == 7
    assert p[:, 1].min() == floor
    assert np.any(np.all(p == v[2], axis=1))  # physical apex is unchanged
    sdf = polygonSolidSDF(p)
    x = torch.tensor([[0, -.61], [.35, -.58], [0, -.35]], dtype=torch.float64)
    d, n = sdf(x)
    # A point inside both solids is far from any exposed surface: no buried
    # triangle base or floor under the wedge remains in the distance field.
    assert float(d[0]) < -.1
    assert bool((d[1:] > 0).all())
    assert bool(torch.isfinite(n).all())
    assert torch.allclose(n.norm(dim=1), torch.ones(3, dtype=n.dtype))


def test_floating_and_side_crossing_obstacles_are_not_floor_unions():
    assert floorAttachedPolygon([-1, 0], [1, 1], [[-.2,.1],[.2,.1],[0,.4]]) is None
    assert floorAttachedPolygon([-1, 0], [1, 1], [[-.2,-.1],[1.1,-.1],[0,.4]]) is None


def test_analytic_tank_and_attached_obstacle_use_one_exposed_polygon():
    from types import SimpleNamespace as NS
    from warpSPH.caseUtils.weaklyCompressible import analyticTankBody

    domain = NS(min=torch.tensor([-1.2, -.6]), max=torch.tensor([1.2, .6]))
    vertices = np.array([[-.29, -.614], [.29, -.614], [0, -.363]])
    body = analyticTankBody(domain, obstacleVertices=vertices)
    assert len(body.reps) == 1
    assert len(body.reps[0].vertices) == 7
    assert body.reps[0].background == 1
    np.testing.assert_array_equal(body._attachedObstacleVertices, vertices)
