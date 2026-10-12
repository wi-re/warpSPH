#!/usr/bin/env python
"""Yee vortex mechanism probe (COMPRESSIBLE_SPH_PLAN E2): at t = 0, no time stepping.

The Yee vortex (zero background flow) is an exact steady solution, so the exact dv/dt of every particle is 0.
On the frontend's own MFM geometry, per closure arm:
  (a) IC consistency: rho_geo = m / V_geo against the analytic rho(r)  (an imbalance here is not the closure's)
  (b) |a| of the vortex state (rates at dt ~ 0): the force truncation error at equilibrium
  (c) |a| of a uniform state (rho = 1, v = 0, P = 1): exactly the closure's job, must be 0 for a closed cell
  (d) the raw face-area residual |sum_j A_ij| relative to the face scale
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch

torch.set_grad_enabled(False)

from warpSPH.runner.runner import buildContext
from warpSPH.runner.caseSpec import CaseSpec
from warpSPH.cases.compressible.yeeVortex import yeeVortexCase as case
from warpSPHCore.mfm import MeshlessWarp, primitives

NR = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 64

spec = CaseSpec(caseName=case.name, scheme='MFM', params=dict(case.params)).merged(**case.defaults).merged(
    nx=NX, scheme='MFM', precision='float32', params=dict(case.params, nr=NR))
ctx = buildContext(case, spec)
case.configureScheme(ctx)
system = case.buildSystem(ctx)
st, sc, cfg, dom = system.state, ctx.schemeConfig, ctx.config, system.domain
gamma = float(sc.gamma)
pos = st.positions
N = pos.shape[0]
r = pos.norm(dim=-1)
beta = ctx.param('beta')
T = 1.0 - (gamma - 1) * beta ** 2 / (8 * gamma * math.pi ** 2) * torch.exp(1 - r ** 2)
rhoExact = T ** (1 / (gamma - 1))
print(f'# Yee nr={NR} nx={NX} N={N} L={float((dom.max - dom.min)[0]):.3g} nngb={sc.nngb:.1f}', flush=True)

ARMS = [('project', dict(closure='project', faceGuard=0)), ('none', dict(closure='none', faceGuard=0)),
        ('fg1', dict(closure='project', faceGuard=1)), ('fg2', dict(closure='project', faceGuard=2)),
        ('cap0.5', dict(closure='project', faceGuard=3)), ('cap0.25', dict(closure='project', faceGuard=4)),
        ('pow0', dict(closure='project', faceGuard=0, power=0.0)), ('pow0.5', dict(closure='project', faceGuard=0, power=0.5)),
        ('pow2', dict(closure='project', faceGuard=0, power=2.0))]


def rms(x):
    return float(x.double().pow(2).mean().sqrt())


for name, a in ARMS:
    g = MeshlessWarp.build(
        pos, st.supports, dom, cfg.kernel, cond_max=sc.condMax, closure=a['closure'],
        closure_power=a.get('power', sc.closurePower), rtol=None, cg_tol=sc.cgTol, cg_iters=sc.cgIters, lam0=None, guards=True,
        face_cond_max=sc.faceCondMax, centred_weights=True, area_cap=False, closure_solver=sc.closureSolver,
        mg_min_particles=sc.mgMinParticles, mg_max_coarse=sc.mgMaxCoarse)
    rho, vel, P = primitives(st.Q, g.volume, gamma)
    kw = dict(dt=1e-9, mode='MFM', order=sc.order, beta_min=sc.betaMin, beta_max=sc.betaMax, cond_crit=sc.condCrit,
              psi1=sc.psi1, psi2=sc.psi2, timeCentredFrame=sc.timeCentredFrame, retry=sc.retry,
              massFluxLimit=sc.massFluxLimit, faceGuard=a['faceGuard'])
    # (b) the IC state as the step sees it
    dQ = g.rates(rho, vel, P, gamma, **kw)
    m = st.Q[:, 0]
    acc = dQ[:, 1:-1] / m[:, None]
    # (c) a uniform state
    one = torch.ones_like(rho)
    dQu = g.rates(one, torch.zeros_like(vel), one, gamma, **kw)
    accU = dQu[:, 1:-1] / m[:, None]
    # (d) the raw residual
    res, fnorm = g.residual()
    faceScale = fnorm / math.sqrt(N)
    aV = rms(acc)
    # the centripetal acceleration the vortex's pressure gradient must balance (the force scale)
    vmag = vel.norm(dim=-1)
    aC = rms(vmag ** 2 / r.clamp_min(0.2))
    inner = r < 3.0
    print(f'{name:8s} rho_geo/rho_exact-1: rms {rms(rho / rhoExact - 1):.2e} | '
          f'|a| vortex rms {aV:.3e} (inner r<3: {rms(acc[inner]):.3e}; centripetal scale {aC:.3e}) | '
          f'|a| uniform rms {rms(accU):.3e} max {float(accU.norm(dim=-1).max()):.3e} | '
          f'raw |sum A|/face rms {rms(res.norm(dim=-1)) / float(faceScale):.3e} | '
          f'closure resid {float(g.closureResidualRatio):.2e} its {int(g.closureIterations)}', flush=True)

    bins = [(0, 2), (2, 3.5), (3.5, 5.0), (5.0, 6.0), (6.0, 8.0)]
    rr = pos.norm(dim=-1)
    resn = res.norm(dim=-1) / float(faceScale)
    out = []
    for lo, hi in bins:
        mk = (rr >= lo) & (rr < hi)
        if int(mk.sum()) == 0:
            continue
        out.append(f'r[{lo},{hi}) n={int(mk.sum())}: aU {rms(accU[mk]):.2e} aV {rms(acc[mk]):.2e} rawRes {rms(resn[mk]):.2f}')
    print('         ' + ' | '.join(out), flush=True)
