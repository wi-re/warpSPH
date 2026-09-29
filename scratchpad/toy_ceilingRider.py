"""CEILING_STICKING_PLAN.md §7 toy: one fluid particle sliding under a flat
ceiling at a controlled speed -- the lone rider of Marrone 3.1 seed 2 (uid 80)
in isolation.

Everything scheme-side is the Marrone probe's own nx67 setup (dx, c0, dt,
sun2017DeltaSPH + PST, fourtakas2019, english2025, symplecticEuler; the drift
fix on with --timeCentredContinuity), built through `_runOne`; only the
geometry and the initial condition change: `semiPeriodic` (x-periodic, walls
top and bottom), one fluid particle placed `s0` spacings under the ceiling
wall surface with velocity (vx, 0), gravity down.

Exact answer: the equations have no energy source (static wall, artificial
viscosity dissipative, a restitution impulse can only remove kinetic energy),
so per unit mass

    E = 1/2 |v|^2 + c0^2 (rho - rho0)^2 / (2 rho0^2) + g y

never rises above E(0), at any vx. A pump shows as E growth and |P| ringing up
at the resonant vx ~ omega_contact dx / 2 pi (~7 m/s for uid 80).

  python scratchpad/toy_ceilingRider.py --vx 7 --noPen finalize --tLimit 0.1
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

ap = argparse.ArgumentParser()
ap.add_argument('--vx', type=float, nargs='+', default=[7.0])
ap.add_argument('--s0', type=float, default=0.25, help='start depth below the ceiling wall surface [dx]')
ap.add_argument('--vy0', type=float, default=0.0)
ap.add_argument('--noPen', default='finalize', choices=('finalize', 'impulse', 'derivative', 'off'))
ap.add_argument('--timeCentredContinuity', action=argparse.BooleanOptionalAction, default=True)
ap.add_argument('--tLimit', type=float, default=0.1)
ap.add_argument('--out', default=os.path.join(ROOT, 'scripts', 'out_ceiling', 'toy_rider'))
ap.add_argument('--video', action=argparse.BooleanOptionalAction, default=True)
ap.add_argument('--plotInterval', type=int, default=10)
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import numpy as np
import torch
import warpSPH.runner as R
from warpSPH.cases.dambreak import dambreakCase
import probe_deltaSPHMarrone as P

os.makedirs(args.out, exist_ok=True)
NX = 67
dx = P.TANK_L / NX
_run = R.run
_ic0 = dambreakCase.initialConditions
_diag0 = dambreakCase.diagnostics
_cfg0 = dambreakCase.configureScheme
summary = []


def runOne(vx):
    rec = {k: [] for k in ('t', 'x', 'y', 'vx', 'vy', 'rho', 'P', 'nW')}
    geo = {}

    def run(*a, **kw):
        # x-periodic channel, one lattice cell of fluid (moved into place below)
        kw['params'] = dict(kw['params'], semiPeriodic=True,
                            fillRatio=1.0 * dx / P.TANK_L, fluidWidth=1.0 * dx / P.TANK_W)
        return _run(*a, **kw)

    def ic(ctx, system):
        _ic0(ctx, system)
        st = system.state
        fl = torch.nonzero(st.kinds == 0).squeeze(-1)
        b = st.positions[st.kinds == 1]
        yTop = float(b[b[:, 1] > 0][:, 1].min()) - 0.5 * dx   # wall surface
        geo.update(yTop=yTop, n=int(fl.numel()))
        # the rider; any extra lattice point of the one-cell block goes to the
        # floor half a period away, where it cannot reach the ceiling
        st.positions[fl[0]] = torch.tensor([0.0, yTop - args.s0 * dx], device=st.positions.device, dtype=st.positions.dtype)
        st.velocities[fl[0]] = torch.tensor([vx, args.vy0], device=st.positions.device, dtype=st.positions.dtype)
        for k, i in enumerate(fl[1:]):
            st.positions[i, 0] = 0.5 * P.TANK_W * 0.98 - k * dx
            st.velocities[i] = 0.0
        st.densities[fl] = ctx.schemeConfig.fluid.restDensity
        ctx.velocityScale = max(abs(vx), 1.0)
        print(f'[toy] {int(fl.numel())} fluid rows; rider at y = yTop - {args.s0} dx, v = ({vx}, {args.vy0})', flush=True)
        geo['uid'] = int(st.UIDs[fl[0]])

    def diag(ctx, system):
        row = {}
        st = system.state
        i = torch.nonzero(st.UIDs == geo['uid']).squeeze(-1)[0]
        v = st.velocities[i]
        rec['t'].append(float(system.t)); rec['x'].append(float(st.positions[i, 0])); rec['y'].append(float(st.positions[i, 1]))
        rec['vx'].append(float(v[0])); rec['vy'].append(float(v[1])); rec['rho'].append(float(st.densities[i]))
        rec['P'].append(float(st.pressures[i]) if st.pressures is not None else np.nan)
        row.update(riderS=(geo['yTop'] - rec['y'][-1]) / dx, riderVy=rec['vy'][-1], riderP=rec['P'][-1])
        return row

    def cfg(ctx):
        _cfg0(ctx)
        ctx.schemeConfig.mdbcNoPenShiftMode = args.noPen

    R.run = run
    dambreakCase.initialConditions = ic
    dambreakCase.diagnostics = diag
    dambreakCase.configureScheme = cfg
    tag = f'vx{vx:g}_s{args.s0:g}_{args.noPen}' + ('_tcc' if args.timeCentredContinuity else '')
    try:
        P._runOne(NX, 40.0, args.tLimit, os.path.join(args.out, tag), args.video, args.plotInterval,
                  scheme='sun2017DeltaSPH', shifting='default', densityDiffusionTerm='fourtakas2019',
                  show=False, timeCentredContinuity=args.timeCentredContinuity,
                  watch=dict(velocityAlarmPlotInterval=1, stallProgress=1e-3))
    finally:
        R.run = _run
        dambreakCase.initialConditions = _ic0
        dambreakCase.diagnostics = _diag0
        dambreakCase.configureScheme = _cfg0
    r = {k: np.array(v) for k, v in rec.items()}
    np.savez(os.path.join(args.out, tag + '_rider.npz'), yTop=geo['yTop'], dx=dx, **r)
    c2 = (P.U_MAX / (1.95 / 40.0)) ** 2
    E = 0.5 * (r['vx'] ** 2 + r['vy'] ** 2) + c2 * (r['rho'] - 1.0) ** 2 / 2 + P.G * r['y']
    n = len(E)
    h = n // 2
    s = (geo['yTop'] - r['y']) / dx
    res = dict(vx=vx, steps=n, E0=E[0], dEmax=float((E - E[0]).max()), dEend=float(E[-1] - E[0]),
               Pmax1=float(np.nanmax(np.abs(r['P'][:h]))), Pmax2=float(np.nanmax(np.abs(r['P'][h:]))),
               vyMax=float(np.abs(r['vy']).max()), vMax=float(np.hypot(r['vx'], r['vy']).max()),
               sMin=float(s.min()), sEnd=float(s[-1]))
    print('[toy] ' + ' '.join(f'{k}={v:.4g}' for k, v in res.items()), flush=True)
    return res


for vx in args.vx:
    summary.append(runOne(vx))
print(f'\n[toy] summary  noPen={args.noPen}  s0={args.s0}  tcc={args.timeCentredContinuity}')
print('   vx   steps  dE_max    dE_end   |P|max 1st/2nd half   |vy|max  |v|max  s_min  s_end')
for r in summary:
    print(f"{r['vx']:6.2f} {r['steps']:6d} {r['dEmax']:+8.3g} {r['dEend']:+8.3g}  {r['Pmax1']:9.3g} {r['Pmax2']:9.3g}  "
          f"{r['vyMax']:7.3g} {r['vMax']:7.3g} {r['sMin']:6.2f} {r['sEnd']:6.2f}")
