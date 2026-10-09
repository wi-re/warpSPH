"""Analytic boundary regions (`buildRegion(representation=...)`, the initializer, the dam break's `wallRepresentation`).

An analytic region is clipped against like any boundary (`filterRegion` reads its `sdf`), is not sampled
(no boundary particles, no ghost layer), and gets a RigidBody from its representation. The fluid sampled
for the particle tank and for the analytic tank is the same set of particles.
"""
import pytest
import torch

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
