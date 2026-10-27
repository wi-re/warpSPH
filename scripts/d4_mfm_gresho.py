#!/usr/bin/env python
"""D4 probe, part 2 (COMPRESSIBLE_SPH_PLAN E.3 item 2): the three candidate time
schemes on the Gresho-Chan vortex, against the core driver (scheme 'hancock' is
the core driver's single stage verbatim, so its N = 4096 cfl 0.2 row must
reproduce the G1 numbers, L1(v) 0.0689 MFM / 0.0569 MFV).

The IC (hexagonal lattice, rho = 1, gamma = 5/3, periodic unit box, the standard
piecewise profile, 4 support pre-refine iterations, the closure warm start
dropped before the run) and the scoring (volume-weighted L1(v) against the
analytic profile, KE/KE0, conservation) are the G1 Gresho probe's
(`scripts/g1_mfm_gresho.py`). Candidates, as in scripts/d4_mfm_wave.py:
  hancock -- (i) the core driver's single stage (prediction uses the full dt)
  ssprk   -- (ii) SSPRK2, rates with dt = 0 (two stages, no internal prediction)
  kdk     -- (iii) kick half, drift with the half-kicked velocity, kick half

    python scripts/d4_mfm_gresho.py --n 4096 --modes MFM MFV --schemes hancock ssprk kdk --cfls 0.2
"""
import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import warp as wp

from pathlib import Path
import warpSPHCore

_HARNESS = Path(os.environ.get(
    "WARPSPHCORE_HARNESS",
    str(Path(warpSPHCore.__file__).resolve().parents[2] / "higherOrderSPH" / "harness")))
import sys
sys.path.insert(0, str(_HARNESS))

from particle_sets import build_case
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.mfm import MeshlessWarp, primitives, conserved

G = 5.0 / 3.0
P_OUTER = 3 + 4 * math.log(2)
DIM = 2
BALL2 = math.pi


@dataclass
class St:
    pos: torch.Tensor
    Q: torch.Tensor
    h: torch.Tensor
    lam: Optional[object] = None
    t: float = 0.0


def supports_from_volume(V, nngb):
    return (nngb * V / BALL2) ** 0.5


def wrap(pos, dom):
    L = dom.max - dom.min
    per = dom.periodic
    return torch.where(per, dom.min + torch.remainder(pos - dom.min, L), pos)


def build_geom_at(pos, h, dom, kernel, lam0=None, closure='project'):
    return MeshlessWarp.build(
        pos, h, dom, kernel,
        cond_max=1e3, closure=closure, closure_power=1.0, rtol=None,
        cg_tol=None, cg_iters=200, lam0=lam0, guards=True, face_cond_max=None,
        centred_weights=True, area_cap=False, closure_solver='auto',
        mg_min_particles=2048, mg_max_coarse=512)


def build_geom(st, dom, kernel):
    g = build_geom_at(st.pos, st.h, dom, kernel, lam0=st.lam)
    st.lam = g.lam
    return g


def profile(r):
    vphi = torch.where(r < 0.2, 5 * r, torch.where(r < 0.4, 2 - 5 * r, torch.zeros_like(r)))
    rs = r.clamp(min=1e-30)
    P = torch.where(r < 0.2, 12.5 * r ** 2 + 5,
                    torch.where(r < 0.4, 12.5 * r ** 2 - 20 * r + 4 * torch.log(5 * rs) + 9,
                                torch.full_like(r, P_OUTER)))
    return vphi, P


def vortex_velocity(x, centre, box):
    d = x - centre
    d = d - box * torch.round(d / box)
    r = torch.linalg.norm(d, dim=-1)
    vphi, _ = profile(r)
    safe = torch.where(r > 0, r, torch.ones_like(r))
    return torch.stack([-vphi * d[:, 1] / safe, vphi * d[:, 0] / safe], -1), r


def ke_of(Q):
    return float((0.5 * Q[:, 1:-1].pow(2).sum(-1) / Q[:, 0:1]).sum())


def make_ic(n, nngb, device, kernel):
    """The reference IC: hexagonal lattice, rho = 1, the piecewise vortex profile,
    the 4 support pre-refine iterations, the warm start dropped (the core's
    init_state builds a fresh state). Returns (state, dom, box, centre, Q0sum, ke0)."""
    case = build_case(n, DIM, int(nngb), jitter=0.0, seed=42, periodic=True,
                      device=device, kernel=kernel)
    dt_ = torch.float64
    x = case.particles.positions.to(dt_)
    dom = case.domain
    box = torch.as_tensor(case.box, dtype=dt_, device=device)
    centre = 0.5 * box
    N = x.shape[0]
    V0 = torch.full((N,), float(case.cell_vol), dtype=dt_, device=device)
    h = supports_from_volume(V0, nngb)
    st0 = St(pos=x, Q=torch.zeros(N, 2 + DIM, dtype=dt_, device=device), h=h)
    g0 = None
    for _ in range(4):
        g0 = build_geom_at(x, st0.h, dom, kernel, lam0=st0.lam)
        st0.lam = g0.lam
        st0.h = supports_from_volume(g0.volume, nngb)
    v, r = vortex_velocity(x, centre, box)
    _, P = profile(r)
    rho = torch.full((N,), 1.0, dtype=dt_, device=device)
    Q = conserved(rho * g0.volume, rho, v, P, G)
    st = St(pos=x, Q=Q, h=st0.h, lam=None)
    return st, dom, box, centre, Q.sum(0), ke_of(Q)


def step_hancock(st, dom, kernel, nngb, cfl, mode, order, tmax):
    """(i) the core driver's single stage, verbatim."""
    g = build_geom(st, dom, kernel)
    rho, vel, P = primitives(st.Q, g.volume, G)
    dt = float(g.timestep(rho, vel, P, G, cfl).min())
    dt = min(dt, tmax - st.t)
    rates = g.rates(rho, vel, P, G, dt=dt, mode=mode, order=order)
    Qn = st.Q + dt * rates
    vel_n = Qn[:, 1:-1] / Qn[:, 0:1]
    st.t += dt
    st.pos = wrap(st.pos + 0.5 * dt * (vel + vel_n), dom)
    st.Q = Qn
    st.h = supports_from_volume(g.volume, nngb)
    return dt


def step_ssprk(st, dom, kernel, nngb, cfl, mode, order, tmax):
    """(ii) SSPRK2, rates with dt = 0 (no internal prediction)."""
    g1 = build_geom(st, dom, kernel)
    rho, vel, P = primitives(st.Q, g1.volume, G)
    dt = float(g1.timestep(rho, vel, P, G, cfl).min())
    dt = min(dt, tmax - st.t)
    h = supports_from_volume(g1.volume, nngb)
    f1 = g1.rates(rho, vel, P, G, dt=0.0, mode=mode, order=order)
    Qs = st.Q + dt * f1
    vels = Qs[:, 1:-1] / Qs[:, 0:1]
    poss = wrap(st.pos + 0.5 * dt * (vel + vels), dom)
    s2 = St(pos=poss, Q=Qs, h=h, lam=g1.lam)
    g2 = build_geom(s2, dom, kernel)
    rho2, vel2, P2 = primitives(Qs, g2.volume, G)
    f2 = g2.rates(rho2, vel2, P2, G, dt=0.0, mode=mode, order=order)
    Qn = 0.5 * Qs + 0.5 * (st.Q + dt * f2)
    vel_n = Qn[:, 1:-1] / Qn[:, 0:1]
    st.t += dt
    st.pos = wrap(st.pos + 0.5 * dt * (vel + vel_n), dom)
    st.Q = Qn
    st.h = h
    st.lam = g2.lam
    return dt


def step_kdk(st, dom, kernel, nngb, cfl, mode, order, tmax):
    """(iii) KDK: kick half, drift with the half-kicked velocity, kick half."""
    g1 = build_geom(st, dom, kernel)
    rho, vel, P = primitives(st.Q, g1.volume, G)
    dt = float(g1.timestep(rho, vel, P, G, cfl).min())
    dt = min(dt, tmax - st.t)
    h = supports_from_volume(g1.volume, nngb)
    f1 = g1.rates(rho, vel, P, G, dt=0.0, mode=mode, order=order)
    Qh = st.Q + 0.5 * dt * f1
    velh = Qh[:, 1:-1] / Qh[:, 0:1]
    poss = wrap(st.pos + dt * velh, dom)
    s2 = St(pos=poss, Q=Qh, h=h, lam=g1.lam)
    g2 = build_geom(s2, dom, kernel)
    rho2, vel2, P2 = primitives(Qh, g2.volume, G)
    f2 = g2.rates(rho2, vel2, P2, G, dt=0.0, mode=mode, order=order)
    st.t += dt
    st.pos = poss
    st.Q = Qh + 0.5 * dt * f2
    st.h = h
    st.lam = g2.lam
    return dt


STEPS = {'hancock': step_hancock, 'ssprk': step_ssprk, 'kdk': step_kdk}


def run(scheme, mode, cfl, n, nngb, tend, device, kernel, order, log=False):
    st, dom, box, centre, Q0sum, ke0 = make_ic(n, nngb, device, kernel)
    step = STEPS[scheme]
    t0 = time.time()
    steps = 0
    kes = []
    dts = []
    while st.t < tend - 1e-12:
        dt = step(st, dom, kernel, nngb, cfl, mode, order, tend)
        dts.append(dt)
        steps += 1
        if steps % 20 == 0:
            kes.append(ke_of(st.Q) / ke0)
        if log and steps % 200 == 0:
            print(f"   t {st.t:.2f} steps {steps} ({time.time()-t0:.0f}s)", flush=True)
    g = build_geom_at(st.pos, st.h, dom, kernel, lam0=st.lam)
    r, v, p = primitives(st.Q, g.volume, G)
    vex, _ = vortex_velocity(st.pos, centre, box)
    w = g.volume
    l1 = float((w * (v - vex).norm(dim=-1)).sum() / w.sum())
    ke1 = ke_of(st.Q) / ke0
    kemax = max(kes) if kes else ke1
    cons = (st.Q.sum(0) - Q0sum).abs().max().item()
    return l1, ke1, kemax, cons, steps, sum(dts) / max(len(dts), 1), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[4096])
    ap.add_argument("--modes", nargs="+", default=["MFM", "MFV"])
    ap.add_argument("--schemes", nargs="+", default=["hancock", "ssprk", "kdk"],
                    choices=sorted(STEPS))
    ap.add_argument("--orders", type=int, nargs="+", default=[2])
    ap.add_argument("--cfls", type=float, nargs="+", default=[0.2])
    ap.add_argument("--nngb", type=float, default=30.0)
    ap.add_argument("--kernel", default="Wendland4")
    ap.add_argument("--tend", type=float, default=3.0)
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args(argv)
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kernel = KernelFunctions[a.kernel]
    for order in a.orders:
        for scheme in a.schemes:
            for mode in a.modes:
                for cfl in a.cfls:
                    for n in a.n:
                        l1, ke1, kemax, cons, steps, dtm, wall = run(
                            scheme, mode, cfl, n, a.nngb, a.tend, device, kernel, order, a.log)
                        print(f"{scheme:8s} {mode} order {order} N={n:5d} cfl {cfl:.2f}: L1(v) {l1:.4f}  "
                              f"KE/KE0 final {ke1:.4f} max {kemax:.4f}  |dQ| {cons:.1e}  steps {steps}  "
                              f"dt {dtm:.3e}  ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
