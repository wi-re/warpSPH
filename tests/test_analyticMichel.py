"""Michel et al. 2022 shifting with analytic walls (`computeMichelShift`): the wall continuum's share of grad C-tilde (Eq. 2-3, volume weighted) and of the characteristic velocity U_char (Eq. 20, the
maximum over the wall) against warpSPH's own boundary-particle path on the dam break's initial state, and the closed form of the flat-wall U_char.

* grad C-tilde: with the wall particles of the particle flavour (one lattice of the fluid's packing) and the analytic continuum on the same fluid particles the sums agree to the lattice quadrature error;
* U_char: the maximum over a lattice of wall particles is a lower bound of the supremum over the continuum, so the analytic value is never below the particle one and equals it where the lattice has a point
  on the line of the relative velocity; the pinned wall (u_g = 0) and a uniform fluid velocity make the wall the only source (fluid-only U_char = 0).
"""
import math

import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPHCore import OperationProperties, SupportScheme, WarpOperation, buildVerletList  # noqa: E402

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.configurations.region import BCType  # noqa: E402
from warpSPH.modules.analyticBoundary import evaluateWall, wallShiftRaw, wallUChar  # noqa: E402
from warpSPH.modules.gravity import computeGravity  # noqa: E402
from warpSPH.modules.shifting.michel import computeMichelShift  # noqa: E402
from warpSPH.modules.shifting.wp_michelUChar import computeUCharWarp  # noqa: E402
from warpSPH.runner import buildContext, getCase  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402
from warpSPH.sample.wp_deltaShift import computeDeltaShiftWarp  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def build(rep, nx=32):
    importAll()
    case = getCase('dambreak')
    params = {**case.params}
    if rep == 'analytic':
        params['wallRepresentation'] = 'analytic'
    spec = CaseSpec(caseName=case.name, scheme=case.scheme, params=params).merged(**case.defaults).merged(nx=nx)
    ctx = buildContext(case, spec)
    case.configureScheme(ctx)
    return ctx, case.buildSystem(ctx)


def pieces(rep, velocity):
    """grad C-tilde and U_char of the fluid particles (sorted by position) with the wall of `rep`: the boundary particles (summed by warpSPH) or the analytic continuum (pinned wall velocity)."""
    ctx, system = build(rep)
    st, config, sc = system.state, ctx.config, ctx.schemeConfig
    n = st.positions.shape[0]
    fl = (st.kinds == 0)
    st.velocities = torch.where(fl[:, None], velocity(n).to(st.positions.device, st.positions.dtype), torch.zeros_like(st.velocities))
    sc.fluid.fixedSoundSpeed = 20.0
    adj = buildVerletList(st, config.domain, verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric, verbose=False)
    op = OperationProperties(operation=WarpOperation.Density, kernel=config.kernel, supportMode=SupportScheme.Gather)
    g = computeDeltaShiftWarp(st, operationProperties=op, referenceParticles=st, domain=config.domain, adjacency=adj, CFL=0.0, computeMach=False, c_max=0.0,
                              rho0=sc.fluid.restDensity, dx=float(config.dx), R=0.2, n=4, volumeWeighted=True)
    U = computeUCharWarp(st, operationProperties=op, referenceParticles=st, domain=config.domain, adjacency=adj)
    if rep == 'analytic':
        for rb in sc.boundaryProvider.rigidBodies:
            rb.kind = BCType.zeros
        wall = evaluateWall(sc.boundaryProvider, st, config, sc, computeGravity(st, config, sc, adj))
        g = g + wallShiftRaw(wall, st, config, sc, R=0.2, volumeWeighted=True).to(g.dtype)
        U = torch.maximum(U, wallUChar(wall, st).to(U.dtype))
    pos = st.positions.double().cpu().numpy()
    m = (st.kinds == 0).cpu().numpy()
    key = np.lexsort((pos[m][:, 1].round(9), pos[m][:, 0].round(9)))
    f = lambda a: a.double().cpu().numpy()[m][key]
    return pos[m][key], f(g), f(U), float(config.dx), float(st.supports.max())


def test_wall_share_of_the_michel_gradient_equals_the_wall_particle_sum():
    rng = np.random.default_rng(5)
    vel = lambda n: torch.as_tensor(rng.normal(scale=0.5, size=(n, 2)))
    pa, ga, _, dx, H = pieces('particles', vel)
    pb, gb, _, _, _ = pieces('analytic', vel)
    assert np.abs(pa - pb).max() < 1e-5
    scale = np.linalg.norm(ga, axis=1).max()
    err = np.linalg.norm(ga - gb, axis=1) / scale
    assert scale > 1.0
    print(f'grad C-tilde scale {scale:.3f}, max error {err.max():.3e}, p90 {np.percentile(err, 90):.3e}')
    assert err.max() < 0.05 and np.percentile(err, 90) < 0.03                                      # measured 3.0 % and 2.3 % (lattice quadrature of the wall sums, H = 4 dx)
    assert np.percentile(err, 50) < 1e-4                                                          # the interior, where both vanish, exactly


def test_wall_u_char_bounds_the_wall_particle_maximum_and_equals_the_closed_form_for_a_flat_wall():
    vel = lambda n: torch.tensor([[1.0, 0.3]]).repeat(n, 1)
    pa, _, Ua, dx, H = pieces('particles', vel)
    pb, _, Ub, _, _ = pieces('analytic', vel)
    w = Ua > 0
    assert w.sum() > 50
    assert (Ub[w] >= Ua[w] - 1e-5).all()                                                           # the lattice maximum is a lower bound
    assert np.median(Ub[w] / Ua[w]) < 1.001                                                        # and equals it where the lattice has a point on the line of u
    # closed form at the first layers of the floor (y = y0 + d, away from the corners): u = (1, 0.3), the floor y0 below (solid for y < y0), support H
    u = np.array([1.0, 0.3]); nu = np.linalg.norm(u)
    floor = pb[:, 0] > pb[:, 0].min() + 1.01 * H                                                   # clear of the left wall (the column stands in the left corner; to its right the floor is flat)
    y0 = float(build('analytic')[0].scratch['interiorDomain'].min[1])                                # the floor (the interior domain of the tank)
    d = pb[:, 1] - y0
    sel = floor & (d < H) & (d > 0) & (pb[:, 1] < y0 + H)
    assert sel.sum() >= 4
    def f(dd):
        if abs(u[1]) / nu >= dd / H:                                                               # the line of u meets the floor within the support
            return 1.0
        L = math.sqrt(H * H - dd * dd)
        return max(abs(u @ np.array([s * L, -dd])) / (nu * H) for s in (1.0, -1.0))
    expect = np.array([nu * f(dd) for dd in d[sel]])
    assert np.abs(Ub[sel] - expect).max() < 2e-5 * nu, (Ub[sel], expect)


def test_computeMichelShift_runs_with_analytic_walls():
    ctx, system = build('analytic')
    st, config, sc = system.state, ctx.config, ctx.schemeConfig
    n = st.positions.shape[0]
    st.velocities = torch.as_tensor(np.random.default_rng(5).normal(scale=0.5, size=(n, 2)), device=st.positions.device, dtype=st.positions.dtype)
    adj = buildVerletList(st, config.domain, verletScale=config.verletScale, supportMode=SupportScheme.SuperSymmetric, verbose=False)
    beta = (st.supports / torch.pow(st.masses / sc.fluid.restDensity, 0.5)) ** 3.0
    d, _ = computeMichelShift(st, config, sc, config.domain, adj, beta=beta, dt=1e-3, iters=1)
    assert torch.isfinite(d).all() and float(d.norm(dim=1).max()) > 0
    assert float(d.norm(dim=1).max()) < 0.5 * float(config.dx)                                    # a fraction of the particle spacing
