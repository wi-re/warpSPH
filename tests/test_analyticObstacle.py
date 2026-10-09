"""The analytic obstacles of the dam break (`analyticObstacleBody`) against the case's own obstacle SDFs, and the dam break with an obstacle: analytic walls against boundary particles."""
import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.caseUtils.weaklyCompressible import analyticObstacleBody, buildObstacleSDF, buildPresetObstacles  # noqa: E402
from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import getCase  # noqa: E402

from warpSPH.runner import run  # noqa: E402
from test_analyticDambreak import check, final, spec_of  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')

PRESETS = ['circleBottom', 'circleMiddle', 'equilateralMiddle', 'triangleMiddle', 'triangleBottom', 'squareMiddle', 'wallMiddle']


@pytest.mark.parametrize('name', PRESETS)
@pytest.mark.parametrize('aoa', [0.0, 25.0])
def test_analytic_obstacle_is_the_sdf_solid(name, aoa):
    L = 2.0
    presets = buildPresetObstacles(0.25, 0.3, L, 0.4, aoa)
    if name not in presets:
        pytest.skip(f'no preset {name}')
    ob = presets[name]
    body = analyticObstacleBody(ob)
    sdf = buildObstacleSDF(ob['obstacleType'], ob['offsetX'], ob['offsetY'], ob['maxExtent'], ob['aspectRatio'], ob['aoa'], None, None, L)
    rng = np.random.default_rng(1)
    ext = 2.5 * ob['maxExtent'] * max(1.0, ob['aspectRatio'])
    pts = torch.as_tensor(np.array([ob['offsetX'], ob['offsetY']]) + rng.uniform(-ext, ext, (4000, 2)), dtype=torch.float64)
    inside_sdf = (sdf(pts) <= 0).numpy()
    d, n, hit = body.signedDistance(pts.cuda()) if hasattr(body, 'signedDistance') else (None, None, None)
    if d is None:
        from warpSPHBoundaries.scene import Scene
        d, n, hit = Scene([body], 'cuda:0').signed_distance(pts.cuda(), body=0)
    inside_body = (d <= 0).cpu().numpy()
    assert inside_sdf.sum() > 20
    # points within a hair of the boundary may differ by round-off
    near = np.abs(sdf(pts).numpy()) < 1e-6
    assert (inside_sdf[~near] == inside_body[~near]).all(), int((inside_sdf != inside_body)[~near].sum())


OBSTACLE = dict(obstacleActive=True, maxExtent=0.5, offsetX=-0.4)


@pytest.mark.parametrize('name', ['circleBottom', 'squareBottom'])
def test_dam_break_with_an_analytic_obstacle_reproduces_the_particle_obstacle(name):
    """The dam break with an obstacle (radius 6 dx / half-width 12 dx on the floor in the column's way), tank and obstacle analytic against both sampled by boundary particles. Measured at 0.3 s: fastest
    particle 11 % (circle) and 6 % (square) lower, kinetic energy 3.2 % and 3.8 % lower; the boundary-particle obstacle has the wider density spread (down to 0.948), so the density bounds are compared at 6 %.
    The fastest particle is a single spray particle off the obstacle: 15 % stated."""
    importAll()
    case = getCase('dambreak')
    par = dict(OBSTACLE, obstacleType=name)
    check(final(case, 'analytic', 600, par), final(case, 'particles', 600, par), vtol=0.15, ketol=0.08, rtol=0.06)


def test_mixed_scene_analytic_tank_with_a_particle_obstacle_runs():
    """A mixed scene: the tank analytic, the obstacle boundary particles (mDBC), each with its own wall treatment in the same step. Stable, no penetration, the kinetic energy of the fully particle scene to 8 %
    (measured 1.1 %); the fastest particle (a spray particle off the obstacle) is not compared."""
    importAll()
    case = getCase('dambreak')
    par = dict(OBSTACLE, obstacleType='circleBottom')
    mixed = final(case, 'analytic', 600, dict(par, obstacleRepresentation='particles'))
    ref = final(case, 'particles', 600, par)
    assert mixed['n'] == ref['n'] and np.isfinite(mixed['vmax'])
    assert abs(mixed['ke'] / ref['ke'] - 1.0) < 0.08, (mixed['ke'], ref['ke'])
    assert mixed['rmin'] > 0.95 and mixed['rmax'] < 1.06
    assert mixed['row'].series('nPenetrating').max() == 0
