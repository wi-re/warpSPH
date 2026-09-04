"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` item 1, 2026-08-29): the per-step
kinetic-energy budget of `dambreak --scheme divergenceFree` around the impact.

Part 19 found the scheme's only working free surface dissipates 88% of the
flow's kinetic energy between t=0.5 and t=0.8 -- exactly when the falling
column should be turning into horizontal run-out -- and Part 21 ruled out the
free-surface clamp (forcing it off NaNs the run in 4 steps). The plan's
"What is next" names the instrument: a per-particle or per-region energy
budget around t=0.4-0.5, following the document's own method note (§1.11:
measure, don't derive).

The velocity update under semiImplicitEuler + ShiftApplication.positionShift
is, per fluid particle of mass m:

    v*     = v + dt (g + F/m + nopen/dt + a_visc + a_DF)
    v_next = v* + c,   c = grad v* . (dt^2 dvdt_incomp)   (Eq. 17 resample)

so the kinetic-energy change closes exactly as

    KE(v_next) - KE(v)
      = dt m v . g                    gravity
      + m v . F                       external forcing
      + m v . nopen                   mDBC no-penetration shift
      + dt m v . a_visc               viscosity (Monaghan, or physical)
      + dt m v . a_DF                 divergence-free pressure projection
      + 0.5 m |dt (g + F/m + nopen/dt + a_visc + a_DF)|^2
      + m v* . c + 0.5 m |c|^2        the shift resample

The W form is what the per-bin tables report (it localizes each channel's
work), but at impact it has a seesaw problem: the projection's first-order
work and the quadratic/resample remainders are large with opposite signs, so
`W_DF` alone does not say how much KE the projection actually costs. For the
channel question the probe therefore also reports a sequential decomposition:
the exact KE change of adding each force to the running velocity, in the
order the step applies them (gravity, no-penetration, viscosity, the
projection last, then the resample). Those five values telescope to `dKE` by
construction and are unambiguous -- `d_DF` is the KE cost of the projection
against the fully-predicted velocity, `d_resample` the KE change of the
resample. A per-step `chain` check (`|u4 - v*|`) verifies that the captured
forces reproduce the integrator's real update. Both views are reported per
step and per --interval window; the per-x-bin rows carry the local closure
too, which is not expected to be small there (particles cross bin edges
during a window; its size bounds that flux).

One setup fact the budget is built to make visible: unlike
`randomFlowIncompressible` (which sets `inviscid=False, viscidNu=nu`),
`dambreak` never touches `diffusionParams`, so it runs with the defaults
`inviscid=True, inviscidAlpha=0.01` -- Monaghan artificial viscosity at the
*weakly-compressible* sound speed `fixedSoundSpeed` that
`setupWeaklyCompressibleTimestep` calibrated against `targetDt`, an acoustic
regime this scheme does not have. Whether that channel accounts for the
impact loss is precisely what the `d_visc` column answers; the `deltaSPH`
control of Part 19 carries the same viscosity, so the cross-scheme gap is the
part this probe does not explain by itself.

The probe monkey-patches the per-step force functions in
`warpSPH.schemes.divergenceFree`, the `solveIncompressible` binding in
`warpSPH.systems.incompressible`, and wraps `IncompressibleSystem.finalize`;
the scheme runs untouched and every patch is restored on exit.

Usage:
  python scripts/probe_dambreakEnergyBudget.py
  python scripts/probe_dambreakEnergyBudget.py --tLimit 1.0 --interval 0.1
  python scripts/probe_dambreakEnergyBudget.py --fixedDt 0.0005 --tLimit 1.0
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--tLimit', type=float, default=1.0)
parser.add_argument('--cflFactor', type=float, default=0.2,
                    help='Part 20: 0.2 is the case-safe value; the published 0.4 NaNs by step 30')
parser.add_argument('--fixedDt', type=float, default=None,
                    help='pin dt (overrides the adaptive timestep hook); the Part 19 baseline is 5e-4')
parser.add_argument('--interval', type=float, default=0.1,
                    help='window size for the per-region tables')
parser.add_argument('--bins', type=int, default=20,
                    help='x-bins per window; edges are the first step\'s fluid x-extent')
parser.add_argument('--samples', type=int, default=40,
                    help='rows in the downsampled per-step table')
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch

from warpSPH.runner.cli import caseMain
from warpSPH.cases.dambreak import dambreakCase as case

import warpSPH.schemes.divergenceFree as dfsph_mod
import warpSPH.systems.incompressible as sysmod

# ------------------------------------------------------------------
# capture machinery: the scheme's per-step force terms, plus the two
# velocity states that bracket the shift resample.
# ------------------------------------------------------------------
TERMS = ['grav', 'forc', 'nopen', 'visc', 'DF', 'quad', 'shift']

captured = {}
records = []        # per step: [t, dt, KE, dKE, W_grav, W_forc, W_nopen, W_visc,
                    #            W_DF, W_quad, W_shift, closure, gap, vMax, maxResample]
interval_rows = []  # per window: [t0, t1, dKE] + 7 term sums + [sumW, residual]
bin_rows = []       # per window: (t0, t1, (nbins, 8) matrix: dKE + 7 terms)

_origs = {
    'gravity': dfsph_mod.computeGravity,
    'forcing': dfsph_mod.computeForcing,
    'nopen': dfsph_mod.computeMdbcNoPenShift,
    'visc': dfsph_mod.computeVelocityDiffusion,
    'solveDF': dfsph_mod.solveDivergenceFree,
    'solveIC': sysmod.solveIncompressible,
    'finalize': sysmod.IncompressibleSystem.finalize,
    'timestep': case.timestep,
}


def _wrap(attr, name):
    fn = _origs[name]

    def wrapper(*a, **kw):
        out = fn(*a, **kw)
        captured[name] = out
        return out

    setattr(dfsph_mod, attr, wrapper)


_wrap('computeGravity', 'gravity')
_wrap('computeForcing', 'forcing')
_wrap('computeMdbcNoPenShift', 'nopen')
_wrap('computeVelocityDiffusion', 'visc')


def _solveDF(*a, **kw):
    out = _origs['solveDF'](*a, **kw)
    # `particles.positions` is still the step-start x here: the integrator
    # updates a different state, and nothing in `divergenceFree_step` touches positions.
    captured['x_n'] = kw['particles'].positions
    captured['a_DF'] = out[0]
    return out


dfsph_mod.solveDivergenceFree = _solveDF


def _solveIC(*a, **kw):
    out = _origs['solveIC'](*a, **kw)
    captured['dvdt_incomp'] = out[0]
    return out


sysmod.solveIncompressible = _solveIC

device = None
nbins = args.bins
x_edges = None
n_nonshift = len(TERMS) - 1
bin_acc = None      # (nbins, 6) float64 work accumulator for the open window
bin_acc_shift = None  # (nbins,) float64; shift's work is only known after finalize
bin_ke_prev = None  # (nbins,) float64 per-bin KE at the window's start
interval_start = 0.0
next_flush = args.interval
last_step = None    # (v_next, m1, bin_idx, t_next) for the final partial window


def _flush(t0, t1, v_next, m1, bin_idx, step_idx):
    """`step_idx` is the index of the last step accumulated into this window;
    the window's steps are records[prev_step_idx+1 .. step_idx], and the
    per-interval table below reuses exactly that range so the two agree."""
    global bin_ke_prev
    ke = (0.5 * m1 * (v_next ** 2).sum(-1)).double()
    bin_ke_now = torch.zeros(nbins, dtype=torch.float64, device=ke.device)
    bin_ke_now.index_add_(0, bin_idx, ke)
    matrix = torch.zeros(nbins, len(TERMS) + 1, dtype=torch.float64, device=ke.device)
    matrix[:, 0] = bin_ke_now - bin_ke_prev
    matrix[:, 1:1 + n_nonshift] = bin_acc
    matrix[:, 1 + n_nonshift] = bin_acc_shift
    bin_rows.append((t0, t1, matrix.cpu(), step_idx))
    bin_ke_prev = bin_ke_now
    bin_acc.zero_()
    bin_acc_shift.zero_()


def patched_finalize(self, initialState, dt, returnValues, updateValues, weights=..., *fargs, **fkwargs):
    global device, x_edges, bin_acc, bin_acc_shift, bin_ke_prev, next_flush, interval_start, last_step
    st = self.state
    fluid = st.kinds == 0
    m1 = st.masses[fluid].view(-1)
    v_star = st.velocities[fluid]           # copy (boolean index); v before the resample
    a_total = updateValues[0].dvdt[fluid]   # copy; the integrator's acceleration
    v_n = v_star - dt * a_total             # the velocity at the step's start

    g = captured['gravity'][fluid].view(-1, 2)
    F = captured['forcing'][fluid].view(-1, 2)
    nopen = captured.get('nopen')
    nopen = torch.zeros_like(g) if nopen is None else nopen[fluid].view(-1, 2)
    avisc = captured['visc'][fluid].view(-1, 2)
    aDF = captured['a_DF'][fluid].view(-1, 2)
    x_n = captured['x_n'][fluid]

    # Per-particle work vectors, in the scheme's own precision; the reductions
    # below go to float64 before summing so the closure resolves round-off.
    w = {
        'grav': dt * (m1 * (v_n * g).sum(-1)),
        'forc': (v_n * F).sum(-1),
        'nopen': m1 * (v_n * nopen).sum(-1),
        'visc': dt * (m1 * (v_n * avisc).sum(-1)),
        'DF': dt * (m1 * (v_n * aDF).sum(-1)),
        'quad': 0.5 * m1 * ((dt * a_total) ** 2).sum(-1),
    }

    # Completeness check on the patching: `a_total` must be exactly the sum
    # of the five captured terms. A gap here means a term is missing from the
    # decomposition, not a physics term.
    gap = float((a_total - (g + F / m1.view(-1, 1) + nopen / dt + avisc + aDF))
                .norm(dim=-1).max().item())

    def _ke(u):
        return float((0.5 * m1 * (u ** 2).sum(-1)).double().sum().item())

    # Sequential channel decomposition: the exact KE change of adding each
    # force to the running velocity, in the order the step applies them
    # (gravity, no-penetration, viscosity, then the projection last, against
    # the fully-predicted velocity). This is unambiguous -- the per-step
    # `W_` form above splits the same dKE as a first-order work sum plus a
    # quadratic remainder, and at impact the projection's first-order work
    # and that remainder are large with opposite signs, so the W form alone
    # does not say which channel dissipates.
    u1 = v_n + dt * g
    u2 = u1 + nopen
    u3 = u2 + dt * avisc
    u4 = u3 + dt * aDF
    # The chain check: u4 must equal v_star, the integrator's actual output.
    # A nonzero `chain` means the captured forces do not reproduce the
    # step's real update, i.e. the decomposition is wrong.
    chain = float((u4 - v_star).norm(dim=-1).max().item())
    KE_n = _ke(v_n)
    K1, K2, K3, K4 = _ke(u1), _ke(u2), _ke(u3), _ke(u4)
    d_grav, d_nopen, d_visc, d_DF = K1 - KE_n, K2 - K1, K3 - K2, K4 - K3

    if x_edges is None:
        # Bin over the whole domain, not the first step's fluid extent: the
        # dam break starts with all fluid in the column, and the run-out that
        # follows would all land in one bin otherwise.
        dom = getattr(self, 'domain', None)
        try:
            lo, hi = float(dom.min[0]), float(dom.max[0])
        except Exception:
            lo, hi = float(x_n[:, 0].min()), float(x_n[:, 0].max())
        x_edges = torch.linspace(lo, hi, nbins + 1, device=x_n.device, dtype=torch.float64)
        bin_acc = torch.zeros(nbins, n_nonshift, dtype=torch.float64, device=x_n.device)
        bin_acc_shift = torch.zeros(nbins, dtype=torch.float64, device=x_n.device)
        device = x_n.device
    bin_idx = torch.bucketize(x_n[:, 0].double(), x_edges[1:-1])
    if bin_ke_prev is None:
        # The first window starts at t=0, whose per-bin KE is v_n of this step
        # (zero here, since the dam break starts at rest -- but do it generally).
        bin_ke_prev = torch.zeros(nbins, dtype=torch.float64, device=x_n.device)
        bin_ke_prev.index_add_(0, bin_idx, (0.5 * m1 * (v_n ** 2).sum(-1)).double())
    bin_acc.index_add_(0, bin_idx,
                       torch.stack([w[t] for t in TERMS if t != 'shift'], dim=1).double())

    result = _origs['finalize'](self, initialState, dt, returnValues, updateValues,
                                weights=weights, *fargs, **fkwargs)

    v_next = st.velocities[fluid]
    c = v_next - v_star                        # the Eq. 17 resample, exactly
    w['shift'] = m1 * (v_star * c).sum(-1) + 0.5 * m1 * (c ** 2).sum(-1)
    bin_acc_shift.index_add_(0, bin_idx, w['shift'].double())

    KE_next = _ke(v_next)
    dKE = KE_next - KE_n
    vals = {t: float(w[t].double().sum().item()) for t in TERMS}
    closure = dKE - sum(vals.values())
    # The last channel is the resample's exact KE change, KE(v_next) - KE(u4);
    # the five d_ values telescope to dKE by construction, so the W-form
    # `closure` above (dKE vs the first-order works) is the independent check.
    d_resample = KE_next - K4
    vMax = float(v_next.norm(dim=-1).max().item())
    maxResample = float(c.norm(dim=-1).max().item())

    # [t, dt, KE, dKE, d_grav, d_nopen, d_visc, d_DF, d_resample, chain,
    #  closure, gap, vMax, maxResample]
    records.append([self.t - dt, dt, KE_n, dKE, d_grav, d_nopen, d_visc, d_DF,
                    d_resample, chain, closure, gap, vMax, maxResample])
    last_step = (v_next, m1, bin_idx, self.t)

    t_next = self.t
    if t_next >= next_flush - 1e-9:
        _flush(interval_start, next_flush, v_next, m1, bin_idx, len(records) - 1)
        interval_start = next_flush
        next_flush += args.interval

    captured.clear()
    return result


sysmod.IncompressibleSystem.finalize = patched_finalize

if args.fixedDt is not None:
    # The `dambreakTimestep` hook re-picks dt every step under divergenceFree,
    # so a `--dt` flag alone would be ignored; pin the hook itself.
    case.timestep = lambda ctx, state: args.fixedDt

argv = ['--nx', str(args.nx), '--tLimit', str(args.tLimit),
        '--scheme', 'divergenceFree', '--quiet', '--no-store', '--no-plot',
        '--integrationScheme', 'semiImplicitEuler',
        '--cflFactor', str(args.cflFactor)]

try:
    r = caseMain(case, argv=argv)
finally:
    dfsph_mod.computeGravity = _origs['gravity']
    dfsph_mod.computeForcing = _origs['forcing']
    dfsph_mod.computeMdbcNoPenShift = _origs['nopen']
    dfsph_mod.computeVelocityDiffusion = _origs['visc']
    dfsph_mod.solveDivergenceFree = _origs['solveDF']
    sysmod.solveIncompressible = _origs['solveIC']
    sysmod.IncompressibleSystem.finalize = _origs['finalize']
    case.timestep = _origs['timestep']

# The final partial window -- only if steps remain after the last flush.
if (bin_acc is not None and last_step is not None
        and len(records) - 1 > (bin_rows[-1][3] if bin_rows else -1)):
    _flush(interval_start, last_step[3], last_step[0], last_step[1], last_step[2],
           len(records) - 1)

sc = r.ctx.schemeConfig
dp = sc.diffusionParams
dt_desc = f'fixed dt={args.fixedDt:g}' if args.fixedDt is not None \
    else f'adaptive CFL {args.cflFactor:g}'
print(f"\n=== dambreak energy budget --scheme divergenceFree, nx={args.nx} "
      f"({len(records)} steps, t={records[-1][0] + records[-1][1]:.3f}, "
      f"{r.wallTime:.0f}s, diverged={r.diverged}) ===")
print(f"dt: {dt_desc}")
print(f"viscosity setup: inviscid={dp.inviscid} inviscidAlpha={dp.inviscidAlpha} "
      f"viscidNu={dp.viscidNu} fixedSoundSpeed={sc.fluid.fixedSoundSpeed:.4g}")
print(f"nFluid={int((r.state.state.kinds == 0).sum())}")
print()

# ------------------------------------------------------------------
# per-step table
# ------------------------------------------------------------------
hdr = (f"{'t':>7} {'dt':>9} {'KE':>9} {'dKE':>9} {'d_grav':>9} {'d_nopen':>9} "
       f"{'d_visc':>9} {'d_DF':>9} {'d_resamp':>9} {'chain':>10}")
print('--- per step (sequential channels: d_X = KE change of adding X to the')
print('running velocity, in step order; the five d_ columns telescope to dKE) ---')
print(hdr)
print('-' * len(hdr))
step = max(1, len(records) // args.samples)
for row in records[::step] + [records[-1]]:
    t, dt, KE, dKE, dg, dn, dv, dF, dr, ch, cl, gap, vMax, mr = row
    print(f"{t:7.4f} {dt:9.2e} {KE:9.4f} {dKE:9.2e} {dg:9.2e} {dn:9.2e} "
          f"{dv:9.2e} {dF:9.2e} {dr:9.2e} {ch:10.2e}", flush=True)
maxClosure = max(abs(row[10]) for row in records)
maxGap = max(row[11] for row in records)
maxChain = max(row[9] for row in records)
print(f"\nchain (|u4 - v*|, captured forces vs the integrator's real update): "
      f"max over run = {maxChain:.3e}")
print(f"W-form closure (dKE - sum of first-order works): max |.| = {maxClosure:.3e}; "
      f"decomposition gap (a_total vs captured sum): max = {maxGap:.3e}")
print('(chain at round-off and gap ~0 confirm the decomposition is complete;')
print('a large gap would mean a term is missing, not that physics is lost.)')
print()

# ------------------------------------------------------------------
# per-interval table
# ------------------------------------------------------------------
print(f'--- per interval (windows of {args.interval:g}) ---')
print('Each window covers exactly the steps its per-bin matrix accumulated.')
print('sumD telescopes to dKE by construction; it is the window closure.')
hdr = (f"{'t0':>7} {'t1':>7} {'steps':>6} {'dKE':>9} {'d_grav':>9} {'d_nopen':>9} "
       f"{'d_visc':>9} {'d_DF':>9} {'d_resamp':>9} {'sumD':>9} {'dKE-sumD':>10}")
print(hdr)
print('-' * len(hdr))
prev_idx = -1
for (t0, t1, matrix, idx) in bin_rows:
    win = records[prev_idx + 1: idx + 1]
    dKE = sum(r[3] for r in win)
    sums = [sum(r[4 + i] for r in win) for i in range(5)]
    sumD = sum(sums)
    print(f"{t0:7.3f} {t1:7.3f} {len(win):6d} {dKE:9.4f} {sums[0]:9.2e} {sums[1]:9.2e} "
          f"{sums[2]:9.2e} {sums[3]:9.2e} {sums[4]:9.2e} "
          f"{sumD:9.4f} {dKE - sumD:10.2e}", flush=True)
    prev_idx = idx
print()

# ------------------------------------------------------------------
# per-interval, per-x-bin tables
# ------------------------------------------------------------------
print(f'--- per x-bin, per interval ({nbins} bins over '
      f'x in [{x_edges[0]:.3f}, {x_edges[-1]:.3f}]) ---')
print('dKE is the bin-local KE change between window edges; the work columns are')
print('attributed to the step-start bin, so particles crossing a bin edge during')
print('the window make the last column (dKE - sumW) nonzero; its size bounds that flux.')
for (t0, t1, matrix, _idx) in bin_rows:
    print(f"\ninterval [{t0:.3f}, {t1:.3f}]:")
    hdr = (f"{'x':>13} {'dKE':>9} {'grav':>9} {'forc':>9} {'nopen':>9} "
           f"{'visc':>9} {'DF':>9} {'quad':>9} {'shift':>9} {'dKE-sumW':>10}")
    print(hdr)
    print('-' * len(hdr))
    for b in range(nbins):
        row = matrix[b]
        xlo, xhi = float(x_edges[b]), float(x_edges[b + 1])
        print(f"[{xlo:6.3f},{xhi:6.3f}) {row[0]:9.4f} {row[1]:9.2e} {row[2]:9.2e} "
              f"{row[3]:9.2e} {row[4]:9.2e} {row[5]:9.2e} {row[6]:9.2e} {row[7]:9.2e} "
              f"{row[0] - row[1:].sum():10.2e}", flush=True)
print()

# ------------------------------------------------------------------
# the dissipation window, from the KE peak
# ------------------------------------------------------------------
peak = max(records, key=lambda r: r[2] + r[3])
t_peak = peak[0] + peak[1]
KE_peak = peak[2] + peak[3]
# Part 19's loss window is t=0.5-0.8; run it from the measured KE peak to the
# first step that reaches 0.8 (or to the end if the peak lands past it).
t_end = next((r[0] + r[1] for r in records if r[0] + r[1] >= 0.8),
             records[-1][0] + records[-1][1])
if t_peak >= t_end:
    t_end = records[-1][0] + records[-1][1]
win = [r for r in records
       if r[0] >= t_peak - 1e-9 and r[0] + r[1] <= t_end + 1e-9]
dKE_win = sum(r[3] for r in win)
sums_win = [sum(r[4 + i] for r in win) for i in range(5)]
print(f'--- dissipation window, t_peak={t_peak:.3f} (KE={KE_peak:.3f}) to t={t_end:.3f} ---')
print(f"KE change: {dKE_win:.4f}  ({dKE_win / KE_peak * 100:.1f}% of the peak)")
for name, val in zip(['grav', 'nopen', 'visc', 'DF', 'resample'], sums_win):
    share = (val / dKE_win * 100) if dKE_win < 0 else 0.0
    print(f"  d_{name:<8} {val:10.4f}   ({share:5.1f}% of the KE loss)")
