"""CEILING_STICKING_PLAN.md §3: resume Marrone 3.1 (δ⁺ nx67 fourtakas2019,
seed 3) from a checkpoint and decompose, at every RHS evaluation, what acts on
each fluid row near the ceiling (or on the UIDs given):

  pressure force (Antuono, surface branch  a_i = -(1/rho_i) sum_j V_j (P_i+P_j) gradW_ij),
  split by linearity in the pressure array (surface mask held fixed):
    aWallP   = -(1/rho_i) sum_{j wall}  V_j P_j gradW       (the wall's own pressure)
    aSelfW   = -(P_i/rho_i) sum_{j wall} V_j gradW          (i's pressure x wall truncation)
    aFluid   = the rest (fluid-fluid pairs, both terms)
  continuity  drho_i = -rho_i sum_j V_j (v_j - v_i).gradW, split wall / fluid pairs,
  DDT (fluid-fluid), artificial viscosity, gravity;
  and for each wall row within the support: english2025's nNbFluid, alpha, P_g,
  hydrostatic increment, P_b, and the freeSlip wall velocity it was given.

Eager (no CUDA graph) so the hooks see every stage.

  python scratchpad/ceiling_forensics.py --resumeFrom <state_N.h5> --offset N --nSteps 400 --tag ev1 [--uids ...]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

ap = argparse.ArgumentParser()
ap.add_argument('--resumeFrom', required=True)
ap.add_argument('--offset', type=int, required=True)
ap.add_argument('--nSteps', type=int, default=400)
ap.add_argument('--uids', type=int, nargs='*', default=None)
ap.add_argument('--band', type=float, default=2.5, help='watch fluid rows within this many dx of the ceiling')
ap.add_argument('--out', default=os.path.join(ROOT, 'scripts', 'out_ceiling', 'forensics'))
ap.add_argument('--tag', default='ev')
ap.add_argument('--video', action=argparse.BooleanOptionalAction, default=True)
ap.add_argument('--cflFactor', type=float, default=None)
ap.add_argument('--integrationScheme', default=None)
ap.add_argument('--patch', default=None, help='python file exec()d after the hooks (candidate-fix A/B)')
ap.add_argument('--seed', type=int, default=3, help='jitter seed of the run the checkpoint came from')
ap.add_argument('--region', default='ceiling', choices=('ceiling', 'rightWall'),
                help="watched band: fluid within --band dx of the ceiling, or of the right wall "
                     "above --yMin (the run-up film, §7.2); 'rightWall' also records finalize for every watched row")
ap.add_argument('--yMin', type=float, default=-0.2, help='rightWall: only rows above this y [m]')
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import numpy as np
import torch
from warpSPHCore import OperationProperties, WarpOperation, SupportScheme, OperationDirection, GradientScheme, warpOperation
import warpSPH.schemes.deltaSPH as S
from warpSPH.enumTypes import PressureForceScheme
import warpSPH.modules.mdbc.english2025 as E
from warpSPH.modules.pressure.wp_surfaceAware import computePressureSurfaceAwareWarp
from warpSPH.cases.dambreak import dambreakCase
import probe_deltaSPHMarrone as P

os.makedirs(args.out, exist_ok=True)
REC = []          # one dict per watched row per RHS evaluation
WALL = []         # english2025 rows next to watched rows
CTX = {'call': 0, 't': 0.0, 'geo': None, 'cfg': None, 'dt': None}
CUR = {}          # per-call scratch: watched indices, pieces

_orig = dict(pf=S.computePressureForceSurfaceAware, mom=S.computeMomentum,
             ddt=S.computeDensityDiffusion, visc=S.computeVelocityDiffusion,
             eng=S.computeMdbcDensityEnglish2025, bvel=S.computeBoundaryVelocities)


def geo(state, config):
    if CTX['geo'] is None:
        dx = float(config.dx)
        pos, k = state.positions, state.kinds
        b = pos[k == 1]
        top = b[(b[:, 1] > 0) & (b[:, 0].abs() < 0.95 * float(b[:, 0].abs().max()) - 6 * dx)]
        CTX['geo'] = (float(top[:, 1].min()) - 0.5 * dx, dx)
    return CTX['geo']


def wallX(state, config):
    # right-wall surface: half a spacing inside its first boundary layer
    if 'xR' not in CTX:
        dx = float(config.dx)
        b = state.positions[state.kinds == 1]
        side = b[(b[:, 0] > 0) & (b[:, 1].abs() < 0.3)]
        CTX['xR'] = float(side[:, 0].min()) - 0.5 * dx
    return CTX['xR']


def watched(state, config):
    yC, dx = geo(state, config)
    fl = state.kinds == 0
    if args.region == 'rightWall':
        sel = fl & (state.positions[:, 0] > wallX(state, config) - args.band * dx) & (state.positions[:, 1] > args.yMin)
    else:
        sel = fl & (state.positions[:, 1] > yC - args.band * dx)
    if args.uids:
        sel = sel | (fl & torch.isin(state.UIDs, torch.tensor(args.uids, device=state.UIDs.device)))
    return torch.nonzero(sel).squeeze(-1)


def pOp(state, config, adjacency, pressures, mask):
    return computePressureSurfaceAwareWarp(
        state, operationProperties=OperationProperties(kernel=config.kernel, supportMode=SupportScheme.SuperSymmetric),
        domain=config.domain, adjacency=adjacency, queryPressures=pressures,
        pressureTerm=PressureForceScheme.nonConservative,
        querySurfaceMask=mask, renormalizationState=None)


def hook_eng(state, config, schemeConfig, adjacency):
    E._CAPTURE = []
    out = _orig['eng'](state, config, schemeConfig, adjacency)
    cap = E._CAPTURE[-1]
    E._CAPTURE = None
    CUR['eng'] = cap
    return out


def hook_bvel(state, config, schemeConfig, adjacency):
    v = _orig['bvel'](state, config, schemeConfig, adjacency)
    CUR['bvel'] = v.detach().clone()
    return v


def hook_ddt(state, config, schemeConfig, adjacency, *a, **kw):
    r = _orig['ddt'](state, config, schemeConfig, adjacency, *a, **kw)
    CUR['ddt'] = r.detach().clone()
    return r


def hook_visc(state, config, schemeConfig, adjacency, *a, **kw):
    r = _orig['visc'](state, config, schemeConfig, adjacency, *a, **kw)
    CUR['visc'] = r.detach().clone()
    return r


def hook_mom(state, config, schemeConfig, adjacency):
    r = _orig['mom'](state, config, schemeConfig, adjacency)
    # wall-pair share: the same divergence with only wall rows as neighbours
    divW = warpOperation(state, OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Divergence, supportMode=SupportScheme.SuperSymmetric,
        operationMode=OperationDirection.BoundaryToFluid, gradientMode=GradientScheme.Difference),
        queryValues=state.velocities, domain=config.domain, adjacency=adjacency, consistentDivergence=False)
    # ... and with the wall rows at rest (v_j = 0): the continuity term whose
    # energy adjoint is exactly the force's self term -(P_i/rho_i) sum_wall V_j gradW
    # (CEILING_STICKING_PLAN.md §7)
    vRest = torch.where((state.kinds == 0).unsqueeze(-1), state.velocities, torch.zeros_like(state.velocities))
    divW0 = warpOperation(state, OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Divergence, supportMode=SupportScheme.SuperSymmetric,
        operationMode=OperationDirection.BoundaryToFluid, gradientMode=GradientScheme.Difference),
        queryValues=vRest, domain=config.domain, adjacency=adjacency, consistentDivergence=False)
    CUR['mom'] = r.detach().clone()
    CUR['momW'] = (-state.densities * divW).detach().clone()
    CUR['momW0'] = (-state.densities * divW0).detach().clone()
    return r


def hook_pf(state, config, schemeConfig, adjacency, renormalizationState=None):
    a = _orig['pf'](state, config, schemeConfig, adjacency, renormalizationState=renormalizationState)
    idx = watched(state, config)
    CTX['call'] += 1
    if idx.numel() == 0:
        return a
    yC, dx = CTX['geo']
    k = state.kinds
    Pw = torch.where(k == 1, state.pressures, torch.zeros_like(state.pressures))
    onesW = torch.where(k == 1, torch.ones_like(state.pressures), torch.zeros_like(state.pressures))
    ones = torch.ones_like(state.surfaceIndicators)
    # nonConservative = P_i + P_j for every row (the surface branch); for a row
    # in the bulk branch with P_i < 0 the split below is labelled by 'sw'.
    aWallP = pOp(state, config, adjacency, Pw, ones)
    aW1 = pOp(state, config, adjacency, onesW, ones)
    P_i = state.pressures
    aSelfW = P_i.view(-1, 1) * aW1
    # neighbour counts (fluid / wall) from the adjacency
    ii, jj = adjacency.i, adjacency.j
    fl_i = (k[ii] == 0)
    nF = torch.zeros_like(P_i).index_add_(0, ii[fl_i & (k[jj] == 0) & (ii != jj)], torch.ones_like(ii[fl_i & (k[jj] == 0) & (ii != jj)], dtype=P_i.dtype))
    nW = torch.zeros_like(P_i).index_add_(0, ii[fl_i & (k[jj] == 1)], torch.ones_like(ii[fl_i & (k[jj] == 1)], dtype=P_i.dtype))
    sw = (P_i >= 0) | (state.surfaceIndicators == 1)
    g = lambda x: x[idx].detach().cpu().numpy()
    rec = dict(call=np.full(idx.numel(), CTX['call']), t=np.full(idx.numel(), CTX['t']),
               uid=g(state.UIDs), x=g(state.positions[:, 0]), y=g(state.positions[:, 1]),
               v=g(state.velocities), rho=g(state.densities), P=g(P_i), fs=g(state.surfaceIndicators),
               sw=g(sw), nF=g(nF), nW=g(nW), a=g(a), aWallP=g(aWallP), aSelfW=g(aSelfW),
               aW1=g(aW1), lam=g(state.surfaceLambdas),
               drho=g(CUR.get('mom', torch.zeros_like(P_i))), drhoW=g(CUR.get('momW', torch.zeros_like(P_i))),
               drhoW0=g(CUR.get('momW0', torch.zeros_like(P_i))), m=g(state.masses),
               ddt=g(CUR.get('ddt', torch.zeros_like(P_i))), visc=g(CUR.get('visc', torch.zeros_like(state.velocities))))
    REC.append(rec)
    # wall rows within the support of a watched row: english2025 + freeSlip velocity
    cap = CUR.get('eng')
    if cap is not None and cap['boundaryUID'] is not None:
        wm = torch.zeros(k.shape[0], dtype=torch.bool, device=k.device)
        near = torch.isin(ii, idx) & (k[jj] == 1)
        wm[jj[near]] = True
        wuid = set(state.UIDs[wm].cpu().numpy().tolist())
        sel = np.array([u in wuid for u in cap['boundaryUID']])
        if sel.any():
            bv = CUR.get('bvel')
            uidToRow = {int(u): r for r, u in enumerate(state.UIDs.cpu().numpy())} if bv is not None else None
            rows = [uidToRow[int(u)] for u in cap['boundaryUID'][sel]] if bv is not None else None
            WALL.append(dict(call=CTX['call'], t=CTX['t'], buid=cap['boundaryUID'][sel], bpos=cap['boundaryPos'][sel],
                             nNb=cap['nNbFluid'][sel], alpha=cap['alpha'][sel], Pg=cap['P_g'][sel],
                             Pb=cap['P_b'][sel], rho_b=cap['rho_b'][sel],
                             hydro=(cap['P_b'][sel] - cap['P_g'][sel]),
                             vb=bv[rows].cpu().numpy() if bv is not None else None))
    return a


# per step, around WeaklyCompressibleSystem.finalize (CEILING_STICKING_PLAN.md §7):
# the watched rows after the integrator (pre), the PST displacement, the noPen
# correction, and after finalize (post) -- so a step's energy change can be split
# into integrator / shifting / noPen replacement
import warpSPH.systems.weaklyCompressible as WS
FIN = []
_origFin = dict(fin=WS.WeaklyCompressibleSystem.finalize, shift=WS.solveShifting, nopen=WS.computeMdbcNoPenShift)


def hook_shift(*a, **kw):
    r = _origFin['shift'](*a, **kw)
    CUR['shiftDx'] = r.detach().clone()
    return r


def hook_nopen(state, *a, **kw):
    r = _origFin['nopen'](state, *a, **kw)
    CUR['nopen'] = r.detach().clone()
    return r


def hook_fin(self, initialState, dt, *a, **kw):
    CUR.pop('shiftDx', None); CUR.pop('nopen', None)
    st = self.state
    if args.region == 'rightWall':
        idx = watched(st, kw.get('config'))
    else:
        idx = torch.nonzero((st.kinds == 0) & torch.isin(st.UIDs, torch.tensor(args.uids or [-1], device=st.UIDs.device))).squeeze(-1)
    g = lambda x: x[idx].detach().cpu().numpy().copy()
    pre = dict(uid=g(st.UIDs), x0=g(initialState.state.positions), v0=g(initialState.state.velocities),
               rho0=g(initialState.state.densities), xPre=g(st.positions), vPre=g(st.velocities), rhoPre=g(st.densities))
    r = _origFin['fin'](self, initialState, dt, *a, **kw)
    st = self.state
    z = torch.zeros_like(st.positions)
    pre.update(t=CTX['t'], m=g(st.masses), dt=float(dt), xPost=g(st.positions), vPost=g(st.velocities), rhoPost=g(st.densities),
               shiftDx=g(CUR.get('shiftDx', z)), nopen=g(CUR.get('nopen', z)))
    FIN.append(pre)
    return r


WS.solveShifting = hook_shift
WS.computeMdbcNoPenShift = hook_nopen
WS.WeaklyCompressibleSystem.finalize = hook_fin

S.computePressureForceSurfaceAware = hook_pf
S.computeMomentum = hook_mom
S.computeDensityDiffusion = hook_ddt
S.computeVelocityDiffusion = hook_visc
S.computeMdbcDensityEnglish2025 = hook_eng
S.computeBoundaryVelocities = hook_bvel

_diag = dambreakCase.diagnostics


def diag(ctx, system, _d=_diag):
    CTX['t'] = float(system.t)
    return dict(_d(ctx, system)) if _d is not None else {}


dambreakCase.diagnostics = diag
if args.patch:
    exec(open(args.patch).read())

import h5py
with h5py.File(args.resumeFrom, 'r') as f:
    t0 = float(f.attrs['time'])
tLimit = t0 + 1.0  # bounded by nSteps below
watch = dict(resumeFrom=args.resumeFrom, resumeStepOffset=args.offset, nSteps=args.nSteps,
             velocityAlarmPlotInterval=1, stallProgress=1e-3)
try:
    P._runOne(67, 40.0, tLimit, os.path.join(args.out, args.tag), args.video, 5,
              scheme='sun2017DeltaSPH', shifting='default', densityDiffusionTerm='fourtakas2019',
              jitter=1e-3, seed=args.seed, show=False, watch=watch, cudaGraph=False, pipeline=False,
              cflFactor=args.cflFactor, integrationScheme=args.integrationScheme)
finally:
    cat = {k: np.concatenate([r[k] for r in REC]) for k in REC[0]} if REC else {}
    np.savez(os.path.join(args.out, f'{args.tag}_rows.npz'), yCeil=CTX['geo'][0], dx=CTX['geo'][1], **cat)
    import pickle
    with open(os.path.join(args.out, f'{args.tag}_wall.pkl'), 'wb') as f:
        pickle.dump(WALL, f)
    if FIN and args.region == 'rightWall':
        with open(os.path.join(args.out, f'{args.tag}_fin.pkl'), 'wb') as f:
            pickle.dump(FIN, f)
    elif FIN:
        np.savez(os.path.join(args.out, f'{args.tag}_fin.npz'),
                 **{k: np.stack([f_[k] for f_ in FIN]) if np.ndim(FIN[0][k]) else np.array([f_[k] for f_ in FIN]) for k in FIN[0]})
    print(f'[forensics] {len(REC)} calls, {sum(len(r["uid"]) for r in REC)} rows -> {args.out}/{args.tag}_rows.npz', flush=True)
