"""Per-step diagnostic for the `hydrostaticColumn` divergence.

`hydrostaticColumn` (baseline case 3, `DFSPH_IMPROVEMENT_PLAN.md` "what is
left" item 2) starts at the exact at-rest hydrostatic state -- the pressure
fit's slope ratio is 1.0 at step -1 -- and diverges within five steps at
nx=32: the fitted slope runs +1.58, -12.4, -23.2, -28.3 against the analytic
-9.81, velocities grow 0.04 -> 0.42 -> 0.75, then the run explodes. This probe
runs the case and wraps the two pressure solves plus the per-step force terms
to show, step by step:

- the divergence-free solve: iterations run, final error, incoming vs outgoing
  pressure (min/max/mean and the fitted dp/dy slope), max |a_p|;
- the constant-density solve (`finalize`): whether the free-surface guard
  downgraded the gauge to `nonNegativeClamp`, the source-term stats,
  iterations, error, and the resulting shift magnitude `|dx|max = dt**2 |a_p|`;
- the step's acceleration channels (no-penetration shift, viscosity, pressure)
  so a spurious channel shows up as its own line.

The probe monkey-patches the per-step functions in the scheme/system modules
(the same pattern `probe_dambreakEnergyBudget.py` uses) and restores them on
exit; the scheme itself runs untouched.

Usage:
    python scripts/probe_hydrostaticColumn.py [--nx 32] [--steps 6]
"""
import argparse

import torch

from warpSPHCore import SupportScheme
from warpSPH.modules.incompressible.drift import computePressureShiftIISPH
from warpSPH.modules.momentum.incompressible import computeMomentumIncompressible
from warpSPH.modules.pressure.iisph import computePressureAccelIISPH

from warpSPH.schemes import divergenceFree as dfsph_mod
from warpSPH.systems import incompressible as sysmod
from warpSPH.configurations import ShiftPressureGauge
from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase as case
from warpSPH.runner import caseMain

args = argparse.ArgumentParser()
args.add_argument('--nx', type=int, default=32)
args.add_argument('--steps', type=int, default=6)
args.add_argument('--zeroIC', action='store_true',
                 help='zero the stored pressure after initialConditions (A/B against the analytic IC)')
args.add_argument('--verboseDF', action='store_true',
                 help='pass verbose=True to the first divergence-free solve (per-iteration error)')
args.add_argument('--shiftApplication', default=None,
                 help='override solverConfig.shiftApplication (positionShift/positionAndVelocity/inStepVelocity)')
args.add_argument('--forceGauge', action='store_true',
                 help='set solverConfig.forceShiftPressureGauge=True (keep minShift on the free surface)')
args.add_argument('--boundaryPressureMode', default=None,
                 help='override solverConfig.boundaryPressureMode (plain/mdbcDensity/mdbcMlsPressure/consistent)')
args = args.parse_args()

VERBOSE_DF = args.verboseDF

_origs = {
    'solveDF': dfsph_mod.solveDivergenceFree,
    'solveIC': sysmod.solveIncompressible,
    'nopen': dfsph_mod.computeMdbcNoPenShift,
    'visc': dfsph_mod.computeVelocityDiffusion,
    'finalize': sysmod.IncompressibleSystem.finalize,
    'timestep': case.timestep,
}
captured = {}
step_no = [-1]


def slope_of(p, pos, mask):
    """Fitted dp/dy over the masked rows, the same fit the case's diagnostics
    use (no margins here: the point is the field's overall tilt)."""
    y = pos[mask, 1]
    v = p[mask]
    yBar, pBar = y.mean(), v.mean()
    denom = ((y - yBar) ** 2).sum()
    return float(((y - yBar) * (v - pBar)).sum() / denom) if denom > 0 else float('nan')


def stats(p, pos, mask):
    return (float(p[mask].min()), float(p[mask].max()), float(p[mask].mean()),
            slope_of(p, pos, mask))


def _op(particles, config, schemeConfig, adjacency, pressureValues, dt):
    """The solver's operator: pressure field -> predicted displacement, the
    exact `dt * computePressureShiftIISPH(computePressureAccelIISPH(...))` the
    Jacobi loop iterates."""
    a = computePressureAccelIISPH(
        state=particles, pressureValues=pressureValues, config=config,
        supportScheme=SupportScheme.Scatter, adjacency=adjacency)
    return dt * computePressureShiftIISPH(
        state=particles, config=config, pressureAccels=a,
        supportScheme=SupportScheme.Scatter, adjacency=adjacency)


def _solveDF(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
    fluid = particles.kinds == 0
    p_in = particles.pressures
    st_in = stats(p_in, particles.positions, fluid) if p_in is not None else None
    first = 'df' not in captured
    out = _origs['solveDF'](particles, config, schemeConfig, adjacency, dvdt, dt,
                            verbose or (VERBOSE_DF and first))
    a_p, pressure, errors, pressures = out
    captured['aDF'] = a_p.clone()
    captured['df'] = (len(errors), errors[-1], st_in,
                      stats(pressure, particles.positions, fluid),
                      float(a_p.norm(dim=-1).max()))
    # The projection's actual job: v' = v + dt*(dvdt + a_p) should be
    # divergence-free. `computeMomentumIncompressible` is the solver's own
    # divergence (its source term is its negation), so measure both the
    # predicted field v* and the corrected field v' with it.
    v_star = particles.velocities + dt * dvdt
    v_prime = v_star + dt * a_p
    div_star = computeMomentumIncompressible(
        particles, config, schemeConfig, adjacency, advectionVelocities=v_star)
    div_prime = computeMomentumIncompressible(
        particles, config, schemeConfig, adjacency, advectionVelocities=v_prime)
    ms = float(div_star[fluid].abs().mean())
    mp = float(div_prime[fluid].abs().mean())
    captured['div'] = (ms, mp)
    if not captured.get('didOpCheck'):
        # Step 0 only: is the exact hydrostatic profile (the incoming
        # pressure, which is the analytic IC) a solution of the solver's
        # equation `op(p) = -div(v*)`?
        b = -div_star
        r_in = b - _op(particles, config, schemeConfig, adjacency, p_in, dt)
        r_out = b - _op(particles, config, schemeConfig, adjacency, pressure, dt)
        captured['didOpCheck'] = True
        captured['opcheck'] = (
            float(b[fluid].abs().mean()),
            float(r_in[fluid].abs().mean()),
            float(r_out[fluid].abs().mean()))
    return out


def _solveIC(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
    fluid = particles.kinds == 0
    surface = getattr(particles, 'surfaceIndicators', None)
    guardFired = (getattr(schemeConfig.solverConfig, 'shiftPressureGauge',
                          ShiftPressureGauge.minShift) is ShiftPressureGauge.minShift
                  and surface is not None and bool((surface > 0.5).any()))
    rho = particles.densities
    out = _origs['solveIC'](particles, config, schemeConfig, adjacency, dvdt, dt, verbose)
    a_p, pressure, errors, pressures = out
    captured['ic'] = (guardFired,
                      (float(rho[fluid].min()), float(rho[fluid].max())),
                      len(errors), errors[-1],
                      (float(pressure[fluid].min()), float(pressure[fluid].max()),
                       float(pressure[fluid].mean())),
                      float(a_p.norm(dim=-1).max()),
                      float((dt * dt * a_p).norm(dim=-1).max()))
    return out


def _nopen(currentState, config, schemeConfig, adjacency):
    v = _origs['nopen'](currentState, config, schemeConfig, adjacency)
    captured['nopen'] = v.clone()
    return v


def _visc(currentState, config, schemeConfig, adjacency):
    v = _origs['visc'](currentState, config, schemeConfig, adjacency)
    captured['visc'] = v.clone()
    return v


def printStep(v_max):
    df = captured.get('df')
    ic = captured.get('ic')
    if df is None:
        captured.clear()
        return
    n_it, err, st_in, st_out, aMax = df
    print(f'--- step {step_no[0]}: vMax={v_max:.4g} ---')
    if st_in is not None:
        print(f'  DF in : p[min={st_in[0]:+.4g} max={st_in[1]:+.4g} mean={st_in[2]:+.4g} '
              f'slope={st_in[3]:+.4g}]')
    print(f'  DF out: p[min={st_out[0]:+.4g} max={st_out[1]:+.4g} mean={st_out[2]:+.4g} '
          f'slope={st_out[3]:+.4g}]  iters={n_it} err={err:.4g}  |a_p|max={aMax:.4g}')
    if 'nopen' in captured:
        print(f'  nopen : max|shift/dt|={float(captured["nopen"].norm(dim=-1).max()):.4g}')
    if 'visc' in captured:
        print(f'  visc  : max|a|={float(captured["visc"].norm(dim=-1).max()):.4g}')
    if 'div' in captured:
        ms, mp = captured['div']
        print(f'  div   : |div v*|={ms:.4g}  |div v\'|={mp:.4g}  ratio={mp / ms if ms else float("nan"):.4g}')
    if 'opcheck' in captured:
        mb, r_in, r_out = captured['opcheck']
        print(f'  opchk : |b|={mb:.4g}  |b-op(p_hydro)|={r_in:.4g}  |b-op(p_out)|={r_out:.4g}')
    if ic is not None:
        guard, rho, n_it, err, pst, aMax, dxMax = ic
        print(f'  IC    : guardClamp={guard} rho[min={rho[0]:.4g} max={rho[1]:.4g}] '
              f'iters={n_it} err={err:.4g} p[min={pst[0]:+.4g} max={pst[1]:+.4g} '
              f'mean={pst[2]:+.4g}] |a_p|max={aMax:.4g} |dx|max={dxMax:.4g}')
    captured.clear()


def patched_finalize(self, initialState, dt, returnValues, updateValues, weights=...,
                     *fargs, **fkwargs):
    result = _origs['finalize'](self, initialState, dt, returnValues, updateValues,
                                weights=weights, *fargs, **fkwargs)
    step_no[0] += 1
    if step_no[0] <= args.steps:
        v_max = float(self.state.velocities[self.state.kinds == 0]
                      .norm(dim=-1).max())
        printStep(v_max)
    else:
        captured.clear()
    return result


dfsph_mod.solveDivergenceFree = _solveDF
sysmod.solveIncompressible = _solveIC
dfsph_mod.computeMdbcNoPenShift = _nopen
dfsph_mod.computeVelocityDiffusion = _visc
sysmod.IncompressibleSystem.finalize = patched_finalize

if args.zeroIC:
    _orig_ic = case.initialConditions

    def _zeroedIC(ctx, system):
        _orig_ic(ctx, system)
        system.state.pressures = torch.zeros_like(system.state.densities)

    case.initialConditions = _zeroedIC
    _origs['ic'] = _orig_ic

if args.shiftApplication or args.forceGauge or args.boundaryPressureMode:
    from warpSPH.configurations import ShiftApplication, BoundaryPressureMode
    _orig_cfg = case.configureScheme

    def _cfgOverride(ctx):
        _orig_cfg(ctx)
        sc = ctx.schemeConfig.solverConfig
        if args.shiftApplication:
            sc.shiftApplication = ShiftApplication[args.shiftApplication]
        if args.forceGauge:
            sc.forceShiftPressureGauge = True
        if args.boundaryPressureMode:
            sc.boundaryPressureMode = BoundaryPressureMode[args.boundaryPressureMode]

    case.configureScheme = _cfgOverride
    _origs['cfg'] = _orig_cfg

# Pin dt so the table is readable; the case's own hook caps at 1e-2 anyway.
case.timestep = lambda ctx, state: 1e-2

_argv = ['--nx', str(args.nx), '--nSteps', str(args.steps + 3),
         '--scheme', 'divergenceFree', '--quiet', '--no-store', '--no-plot',
         '--integrationScheme', 'semiImplicitEuler']

try:
    r = caseMain(case, argv=_argv)
finally:
    dfsph_mod.solveDivergenceFree = _origs['solveDF']
    sysmod.solveIncompressible = _origs['solveIC']
    dfsph_mod.computeMdbcNoPenShift = _origs['nopen']
    dfsph_mod.computeVelocityDiffusion = _origs['visc']
    sysmod.IncompressibleSystem.finalize = _origs['finalize']
    case.timestep = _origs['timestep']
    if 'ic' in _origs:
        case.initialConditions = _origs['ic']
    if 'cfg' in _origs:
        case.configureScheme = _origs['cfg']

print(f'diverged={getattr(r, "diverged", None)}')
