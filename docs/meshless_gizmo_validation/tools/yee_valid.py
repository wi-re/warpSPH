#!/usr/bin/env python
"""Valid Yee vortex for the meshless schemes (COMPRESSIBLE_SPH_PLAN E2): a (jittered) lattice that FILLS the
periodic box with the analytic far field -- no vacuum rim -- optionally with a background flow, so the vortex
translates and the exact solution is known at any t:

    rho, v, P (x - bg t wrapped) -- exact; with bg = 0 it is steady.

Compares closure 'project' vs 'none' (and faceGuard arms) with accuracy metrics against the exact solution:
KE ratio, L1 density error, L1 velocity error. Video on (each run is seconds).

    python .tmp/yee_valid.py --nx 64 --jitter 0.1 --bg 1 --schemes MFM,MFV --arms project,none
"""
import argparse
import dataclasses
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch

torch.set_grad_enabled(False)

from warpSPH.cases.compressible.yeeVortex import yeeVortexCase as base
from warpSPH.runner import run
from warpSPH.sample.compressible import setupBasicCompressibleInitialState
from warpSPH.caseUtils.meshless import toMeshlessSystem
from warpSPH.configurations import CompressibleSPHConfig
from warpSPH.systems.compressibleMonaghan import CompressibleState, CompressibleSystem

BETA, GAMMA = 5.0, 1.4
OPT = {}


def exact(x, t, domain, bg):
    """(rho, v, P) of the Yee vortex at particle positions x at time t (periodic images, background bg)."""
    L = domain.max - domain.min
    shift = torch.tensor([bg, bg], dtype=x.dtype, device=x.device) * t
    d = x - shift
    d = d - L * torch.round(d / L)              # nearest periodic image of the centre
    r2 = (d ** 2).sum(-1)
    w = BETA / (2 * math.pi) * torch.exp((1 - r2) / 2)
    v = torch.stack([-w * d[:, 1], w * d[:, 0]], dim=-1) + torch.tensor([bg, bg], dtype=x.dtype, device=x.device)
    dT = -(GAMMA - 1) * BETA ** 2 / (8 * GAMMA * math.pi ** 2) * torch.exp(1 - r2)
    T = 1.0 + dT
    rho = T ** (1 / (GAMMA - 1))
    return rho, v, rho * T


def buildSystem(ctx):
    sc = ctx.schemeConfig
    comp = setupBasicCompressibleInitialState(
        ctx.spec.nx, ctx.config, CompressibleSPHConfig(gamma=GAMMA, rho0=1.0), CompressibleState,
        CompressibleSystem, jitter=OPT['jitter'])
    st = comp.state
    vol = st.masses / st.densities
    rho, v, P = exact(st.positions, 0.0, ctx.config.domain, OPT['bg'])
    st.densities = rho.to(st.densities.dtype)
    st.masses = (rho * vol).to(st.masses.dtype)
    st.velocities = v.to(st.positions.dtype)
    st.pressures = P.to(st.densities.dtype)
    sc.gamma = GAMMA
    sc.nngb = ctx.config.targetNeighbors
    return toMeshlessSystem(comp, ctx.config, sc)


ARMS = {'project': dict(closure='project', faceGuard=0), 'none': dict(closure='none', faceGuard=0),
        'fg2': dict(closure='project', faceGuard=2), 'fg3': dict(closure='project', faceGuard=3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=64)
    ap.add_argument('--tLimit', type=float, default=2.0)
    ap.add_argument('--jitter', type=float, default=0.0)
    ap.add_argument('--bg', type=float, default=0.0)
    ap.add_argument('--cfl', type=float, default=0.1)
    ap.add_argument('--schemes', default='MFM,MFV')
    ap.add_argument('--arms', default='project,none')
    ap.add_argument('--no-video', dest='video', action='store_false')
    ap.add_argument('--out', default=str(REPO / 'results' / 'e2_yee_valid_2026-10-12'))
    a = ap.parse_args()
    OPT.update(jitter=a.jitter, bg=a.bg)
    for scheme in a.schemes.split(','):
        for arm in a.arms.split(','):
            cfg = dict(ARMS[arm], cfl=a.cfl)

            def configureScheme(ctx, cfg=cfg):
                base.configureScheme(ctx)
                for k, v in cfg.items():
                    setattr(ctx.schemeConfig, k, v)

            case = dataclasses.replace(base, buildSystem=buildSystem, configureScheme=configureScheme)
            tag = f'nx{a.nx}_j{a.jitter}_bg{a.bg}_{scheme}_{arm}'
            kw = dict(scheme=scheme, nx=a.nx, tLimit=a.tLimit, precision='float32', quiet=True, progress=True,
                      velocityAlarmPlotInterval=1, stallProgress=1e-3, exportRoot=os.path.join(a.out, tag))
            if a.video:
                kw.update(plot=True, video=True, plotInterval=10, plotBackend='vispy')
            t0 = time.perf_counter()
            try:
                r = run(case, **kw)
            except Exception as e:
                print(f'[yee] {tag} | FAILED {type(e).__name__}: {e}', flush=True)
                continue
            wall = time.perf_counter() - t0
            p = r.state.state
            rows = r.trajectory
            t = float(rows[-1]['t'])
            rho_e, v_e, _ = exact(p.positions, t, r.ctx.config.domain, a.bg)
            rho = p.densities
            m = p.Q[:, 0]
            vel = p.velocities
            ke = lambda vv: float((0.5 * m * (vv ** 2).sum(-1)).sum())
            vortexV = vel - a.bg
            vortexVe = v_e - a.bg
            ke0 = rows[0]['kineticEnergy']
            # velocity relative to the background: the vortex KE ratio is the clean measure
            l1rho = float((rho - rho_e).abs().mean())
            l1v = float((vel - v_e).norm(dim=-1).mean())
            keV = ke(vortexV) / ke(vortexVe)
            print(f'[yee] {tag} | t={t:.2f} steps={len(rows) - 1} {1e3 * wall / (len(rows) - 1):.1f}ms/step(incl video) '
                  f'vortexKE/exact={keV:.4f} L1rho={l1rho:.4e} L1v={l1v:.4e} '
                  f'maxV={max(x["maxVelocity"] for x in rows) if "maxVelocity" in rows[0] else float("nan"):.3f} '
                  f'drift={abs(rows[-1]["totalEnergy"] - rows[0]["totalEnergy"]) / abs(rows[0]["totalEnergy"]):.2e}',
                  flush=True)


if __name__ == '__main__':
    main()
