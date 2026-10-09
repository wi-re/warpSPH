"""The dam break with analytic tank walls against the same case with boundary particles (warpSPH's own mDBC walls).

Same fluid particles, same scheme settings (time-centred continuity, Antuono switch, fourtakas2019, Michel / delta+ shifting as the case
configures it). After 0.15 s and 0.3 s the analytic run must agree with the particle run in the fastest particle (5 %: two different wall models), the kinetic energy
(8 %) and the density bounds, with no penetration of the walls. Regression for the wall flux in the system's time-centred continuity
closure (`drift_rates`): without it the density at the walls is advanced without the wall's compression and the run blows up within 0.06 s.
Also: `analyticWallPressure='normal'` leaves a fluid in free fall (uniform density, the case's start) without the tangential hydrostatic
force that the default imposes on the first fluid column.
"""
import numpy as np
import pytest
import torch

pytest.importorskip('warpSPHBoundaries')

from warpSPH.cases import importAll  # noqa: E402
from warpSPH.runner import buildContext, getCase, run  # noqa: E402
from warpSPH.runner.caseSpec import CaseSpec  # noqa: E402
from warpSPH.schemes.deltaSPH import deltaSPH_step  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')


def spec_of(case, rep, **kw):
    params = {**case.params, 'wallRepresentation': rep, **kw.pop('params', {})}
    return CaseSpec(caseName=case.name, scheme=case.scheme, params=params).merged(**case.defaults).merged(nx=48, **kw)


def final(case, rep, nSteps, params=None):
    res = run(case, spec_of(case, rep, nSteps=nSteps, params=params or {}, plot=False, store=False, progress=False, video=False, show=False, quiet=True))
    s = res.state.state
    f = s.kinds == 0
    v, rho = s.velocities[f].double(), s.densities[f].double()
    return dict(vmax=float(v.norm(dim=1).max()), ke=float((v * v).sum()), rmin=float(rho.min()), rmax=float(rho.max()), n=int(f.sum()), row=res)


@pytest.mark.parametrize('nSteps', [300, 600])
def test_analytic_tank_reproduces_the_particle_tank(nSteps):
    importAll()
    case = getCase('dambreak')
    a, p = final(case, 'analytic', nSteps), final(case, 'particles', nSteps)
    assert a['n'] == p['n']
    assert abs(a['vmax'] / p['vmax'] - 1.0) < 0.05, (a['vmax'], p['vmax'])
    assert abs(a['ke'] / p['ke'] - 1.0) < 0.08, (a['ke'], p['ke'])
    assert abs(a['rmax'] - p['rmax']) < 0.01 and abs(a['rmin'] - p['rmin']) < 0.01
    assert a['row'].series('nPenetrating').max() == 0


def check(a, p, vtol=0.05, ketol=0.08, rtol=0.01):
    assert a['n'] == p['n']
    assert abs(a['vmax'] / p['vmax'] - 1.0) < vtol, (a['vmax'], p['vmax'])
    assert abs(a['ke'] / p['ke'] - 1.0) < ketol, (a['ke'], p['ke'])
    assert abs(a['rmax'] - p['rmax']) < rtol and abs(a['rmin'] - p['rmin']) < rtol
    assert a['row'].series('nPenetrating').max() == 0


def test_analytic_tank_reproduces_the_particle_tank_with_michel_shifting():
    """Michel 2022 (grad C-tilde and U_char with the wall continuum) in place of delta+ against the same case with boundary particles: measured 1.9 % in the fastest particle and 4.8 % in the kinetic energy at 0.3 s."""
    importAll()
    case = getCase('dambreak')
    shift = dict(shiftScheme='michel2022', shiftProjection='michel2022')
    check(final(case, 'analytic', 600, shift), final(case, 'particles', 600, shift))


def test_analytic_tank_with_implicit_shifting_follows_the_analytic_delta_plus_run():
    """Implicit shifting (grad C with the wall continuum) against the analytic delta+ run (measured 1.1 % in the fastest particle, 0.04 % in the kinetic energy at 0.3 s). Not against the boundary-particle
    path: there `computeImplicitShift` sums the mDBC ghost nodes into grad C (they are not wall mass) and the run diverges (fastest particle ~900 at 0.3 s)."""
    importAll()
    case = getCase('dambreak')
    check(final(case, 'analytic', 600, dict(shiftScheme='implicit', shiftProjection='surfaceNormal')), final(case, 'analytic', 600), vtol=0.03, ketol=0.02)


def test_normal_wall_pressure_leaves_free_fall_alone():
    importAll()
    case = getCase('dambreak')
    first = {}
    for mode in ('hydrostatic', 'normal'):
        ctx = buildContext(case, spec_of(case, 'analytic', params={'shifting': False, 'analyticWallPressure': mode}))
        case.configureScheme(ctx)
        ctx.schemeConfig.cudaGraph = False
        system = case.buildSystem(ctx)
        case.initialConditions(ctx, system)
        upd = deltaSPH_step(system, 5e-4, ctx.config, ctx.schemeConfig)
        upd = upd[0] if isinstance(upd, tuple) else upd
        st = system.state
        n = int((st.kinds == 0).sum())
        x, a = st.positions[:n].double().cpu().numpy(), upd.dvdt[:n].double().cpu().numpy()
        col = np.abs(x[:, 0] - x[:, 0].min()) < 1e-6
        first[mode] = np.median(a[col, 1])
    assert abs(first['normal'] + 9.81) < 0.05                    # free fall along the wall
    assert first['hydrostatic'] > -8.0                           # the default pushes the wall column up (it assumes hydrostatic balance)
