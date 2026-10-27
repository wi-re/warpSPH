#!/usr/bin/env python
"""D4 probe (COMPRESSIBLE_SPH_PLAN E.3 item 2): the temporal order of the three
candidate time schemes on the 1-D sound wave, at fixed resolution (dt swept via
cfl; dt = 2 cfl h / v_sig, so halving cfl halves dt).

Candidates, all on the MeshlessWarp backend (float64, closure 'project', the same
calls the frontend's meshlessStep makes):
  (i)   Hancock + Euler -- the core driver's single stage (the internal prediction
                           uses the full dt); this is the current frontend default
  (ii)  SSPRK2, dt = 0  -- two stages, no internal prediction (first order in time
                           per stage, second order combined)
  (iii) KDK, dt = 0     -- kick half, drift with the half-kicked velocity, kick half

The IC and the scoring mirror the core harness's run_mfm_smooth.py exactly: a
linear right-going wave (eps = 1e-4, gamma = 1.4, c = sqrt(gamma)) that returns
after one period T = L / c; the relative L2 error of the density perturbation
and of v_x, scored at the final particle positions (mean offset removed).

The spatial error floor at fixed N is ~10x the temporal error at cfl = 0.2, so
the temporal part is read off as E(cfl) - E(floor) from the cfl sweep. The
driver's one-step-lagged h adds its own O(dt) term (core finding: the closure
residual is first-order sensitive to a stale h); --h-iters 0 keeps the driver's
lag (what the E1 thin scheme ships) and --h-iters 2 refreshes h to the h<->V
fixed point at the moved particles (the core's H_ITERS) to isolate the stepper.

    python scripts/d4_mfm_wave.py --n 200 --cfls 0.4 0.2 0.1 0.05 --modes MFM MFV --schemes hancock ssprk kdk
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

from warpSPHCore import DomainDescription
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.mfm import MeshlessWarp, primitives, conserved

BALL1 = 2.0
G = 1.4
EPS = 1.0e-4
C_SND = math.sqrt(G)


@dataclass
class St:
    pos: torch.Tensor
    Q: torch.Tensor
    h: torch.Tensor
    lam: Optional[object] = None
    t: float = 0.0


def supports_from_volume(V, nngb):
    return (nngb * V / BALL1) ** 1.0


def wrap(pos, dom):
    L = dom.max - dom.min
    return dom.min + torch.remainder(pos - dom.min, L)


def build_geom_at(pos, h, dom, kernel, lam0=None, closure='project'):
    return MeshlessWarp.build(
        pos, h, dom, kernel,
        cond_max=1e3, closure=closure, closure_power=1.0, rtol=None,
        cg_tol=None, cg_iters=200, lam0=lam0, guards=True, face_cond_max=None,
        centred_weights=True, area_cap=False, closure_solver='auto',
        mg_min_particles=2048, mg_max_coarse=512)


def build_geom(st, dom, kernel, closure='project'):
    g = build_geom_at(st.pos, st.h, dom, kernel, lam0=st.lam, closure=closure)
    st.lam = g.lam
    return g


def refresh_h(st, dom, kernel, nngb, iters):
    """The core driver's H_ITERS: re-derive h from the volumes at the moved
    particles (closure 'none' -- the volume does not depend on the closure)."""
    for _ in range(iters):
        gh = build_geom_at(st.pos, st.h, dom, kernel, lam0=None, closure='none')
        st.h = supports_from_volume(gh.volume, nngb)


def make_ic(n, nngb, device):
    """Mirrors run_mfm_smooth.setup + the 4 pre-refine h<->V iterations.
    Returns (run_state, domain, Q0_sum, kernel) with the run state's lam
    dropped (the core's init_state builds a fresh state, so step 1 cold-starts
    the closure)."""
    dt_ = torch.float64
    x = ((torch.arange(n, dtype=dt_, device=device) + 0.5) / n)[:, None]
    L = torch.ones(1, dtype=dt_, device=device)
    dom = DomainDescription(torch.zeros(1, dtype=dt_, device=device), L,
                            torch.ones(1, dtype=torch.bool, device=device), 1)
    V0 = torch.full((n,), 1.0 / n, dtype=dt_, device=device)
    h = supports_from_volume(V0, nngb)
    kernel = KernelFunctions.Wendland2
    st0 = St(pos=x, Q=torch.zeros(n, 3, dtype=dt_, device=device), h=h)
    g0 = None
    for _ in range(4):
        g0 = build_geom_at(x, st0.h, dom, kernel, lam0=st0.lam)
        st0.lam = g0.lam
        st0.h = supports_from_volume(g0.volume, nngb)
    k = 2.0 * math.pi / float(L[0])
    ph = torch.sin(k * x[:, 0])
    rho = 1.0 + EPS * ph
    mass = rho * g0.volume
    vel = torch.zeros(n, 1, dtype=dt_, device=device)
    vel[:, 0] = EPS * C_SND * ph
    P = 1.0 + G * EPS * ph
    Q = conserved(mass, rho, vel, P, G)
    st = St(pos=x, Q=Q, h=st0.h, lam=None)
    return st, dom, Q.sum(0), kernel


def step_hancock(st, dom, kernel, nngb, cfl, mode, order, tmax, h_iters):
    """(i) the core driver's single stage, verbatim (plus the optional H_ITERS)."""
    g = build_geom(st, dom, kernel)
    rho, vel, P = primitives(st.Q, g.volume, G)
    dt = float(g.timestep(rho, vel, P, G, cfl).min())
    dt = min(dt, tmax - st.t)
    rates = g.rates(rho, vel, P, G, dt=dt, mode=mode, order=order)
    Qn = st.Q + dt * rates
    vel_n = Qn[:, 1:-1] / Qn[:, 0:1]
    pos = wrap(st.pos + 0.5 * dt * (vel + vel_n), dom)
    st.t += dt
    st.pos = pos
    st.Q = Qn
    st.h = supports_from_volume(g.volume, nngb)
    refresh_h(st, dom, kernel, nngb, h_iters)
    return dt


def step_ssprk(st, dom, kernel, nngb, cfl, mode, order, tmax, h_iters):
    """(ii) SSPRK2: f1 at the start of the step (rates with dt = 0, no internal
    prediction), a stage state at the midpoint drift, f2 at the stage state,
    Q_{n+1} = 1/2 (Q* + Q + dt f2); the final drift is the time-centred one."""
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
    refresh_h(st, dom, kernel, nngb, h_iters)
    return dt


def step_kdk(st, dom, kernel, nngb, cfl, mode, order, tmax, h_iters):
    """(iii) KDK: kick half (0.5 dt f1 into the conserved Q), drift the full dt
    with the half-kicked velocity, kick half again with f2 at the drifted state."""
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
    refresh_h(st, dom, kernel, nngb, h_iters)
    return dt


STEPS = {'hancock': step_hancock, 'ssprk': step_ssprk, 'kdk': step_kdk}


def run(n, cfl, mode, scheme, nngb, device, h_iters, log=False):
    st, dom, Q0sum, kernel = make_ic(n, nngb, device)
    T = float(dom.max[0] - dom.min[0]) / C_SND
    step = STEPS[scheme]
    t0 = time.time()
    steps = 0
    dts = []
    while st.t < T - 1e-12:
        dt = step(st, dom, kernel, nngb, cfl, mode, 2, T, h_iters)
        dts.append(dt)
        steps += 1
        if log and steps % 500 == 0:
            print(f"   t {st.t:.3f} steps {steps} ({time.time()-t0:.0f}s)", flush=True)
    g = build_geom_at(st.pos, st.h, dom, kernel, lam0=st.lam)
    st.lam = g.lam
    rho, vel, P = primitives(st.Q, g.volume, G)
    k = 2.0 * math.pi / float(dom.max[0] - dom.min[0])
    ph1 = torch.sin(k * st.pos[:, 0])
    dr = rho - 1.0 - EPS * ph1
    er = float(((dr - dr.mean()) ** 2).mean().sqrt() / (EPS * (ph1 ** 2).mean().sqrt()))
    ev = float(((vel[:, 0] - EPS * C_SND * ph1) ** 2).mean().sqrt()
               / (EPS * C_SND * (ph1 ** 2).mean().sqrt()))
    cons = (st.Q.sum(0) - Q0sum).abs().max().item()
    return er, ev, cons, steps, sum(dts) / max(len(dts), 1), time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[200])
    ap.add_argument("--cfls", type=float, nargs="+", default=[0.4, 0.2, 0.1, 0.05])
    ap.add_argument("--modes", nargs="+", default=["MFM", "MFV"])
    ap.add_argument("--schemes", nargs="+", default=["hancock", "ssprk", "kdk"],
                    choices=sorted(STEPS))
    ap.add_argument("--nngb", type=float, default=7.0)
    ap.add_argument("--h-iters", type=int, default=0,
                    help="h<->V fixed-point iterations at the moved particles (0: the driver's one-step lag)")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args(argv)
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for n in a.n:
        for mode in a.modes:
            for scheme in a.schemes:
                prev = None
                for cfl in a.cfls:
                    er, ev, cons, steps, dtm, wall = run(n, cfl, mode, scheme, a.nngb, device, a.h_iters, a.log)
                    rate = ""
                    if prev is not None:
                        p_er, p_ev, p_dtm = prev
                        o_r = math.log(p_er / er) / math.log(p_dtm / dtm)
                        o_v = math.log(p_ev / ev) / math.log(p_dtm / dtm)
                        rate = f"  order rho {o_r:.2f} v {o_v:.2f}"
                    prev = (er, ev, dtm)
                    print(f"{scheme:8s} {mode} n={n:4d} h{a.h_iters} cfl {cfl:.2f}: err rho {er:.3e} v {ev:.3e}"
                          f"  |dQ| {cons:.1e}  steps {steps}  dt {dtm:.3e}  mean{rate}  ({wall:.0f}s)",
                          flush=True)


if __name__ == "__main__":
    main()
