#!/usr/bin/env python
"""G1 gate probe (COMPRESSIBLE_SPH_PLAN E1): the frontend MFM / MFV scheme
reproduces the core harness (`run_mfm_sod.py`) on a periodic double Sod tube.

Uses the frontend's `MeshlessState` (the conserved-Q state) and the core
`MeshlessWarp` backend -- the same calls the E1 step function makes -- and mirrors
`mfm_driver.step` exactly: one global signal timestep, the single-stage MUSCL-Hancock
update, Euler on the conserved Q, a midpoint drift on the positions, and the supports
re-derived from the pre-move volumes. Scores L1(rho) / L1(v) / L1(P) against the exact
Riemann solution and checks conservation to round-off. A separate check confirms the
E1 step function (`meshlessStep`) returns the same rates / drift as the direct backend
calls (bitwise, same geometry).

    python scripts/g1_mfm_sod.py --n 200 400 --modes MFM MFV --orders 1 2

Oracle (core harness run_mfm_sod.py, float64, Wendland2, nngb 7, tend 0.12,
tag mfm-oracle-2026-10-11) -- MFM order 2:
    L1 rho  1.4652e-2 (n=200)  /  7.2854e-3 (n=400)
    conservation |dQ|  ~ 1e-15
(The plan's earlier 1.4188e-2 / 7.9946e-3 predate the oracle tag.)
"""
import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import time

import numpy as np
import torch
import warp as wp

import sys
from pathlib import Path
import warpSPHCore

# the core evidence harness (mfm_driver, exact_riemann): shipped with warpSPHCore;
# override with WARPSPHCORE_HARNESS for a separate checkout
_HARNESS = Path(os.environ.get(
    "WARPSPHCORE_HARNESS",
    str(Path(warpSPHCore.__file__).resolve().parents[2] / "higherOrderSPH" / "harness")))
sys.path.insert(0, str(_HARNESS))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # warpSPH

import exact_riemann as ex
from warpSPHCore import DomainDescription
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.mfm import MeshlessWarp, primitives, conserved

from warpSPHIntegrators.integration import IntegrationSchemeType
from warpSPH.systems.meshless import MeshlessState, MeshlessSystem
from warpSPH.configurations.meshless import MeshlessConfig
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.schemes.meshless import meshlessStep, supports_from_volume

G = 1.4
CFL = 0.2


def wrap(pos, domain):
    L = domain.max - domain.min
    per = domain.periodic
    return torch.where(per, domain.min + torch.remainder(pos - domain.min, L), pos)


def make_state(n, nngb, device):
    dt_ = torch.float64
    x = (torch.arange(n, dtype=dt_, device=device) + 0.5) / n
    high = (x >= 0.25) & (x < 0.75)
    rho = torch.where(high, 1.0, 0.125).to(dt_)
    P = torch.where(high, 1.0, 0.1).to(dt_)
    vel = torch.zeros(n, 1, dtype=dt_, device=device)
    mass = rho / n
    h = torch.full((n,), nngb / (2.0 * n), dtype=dt_, device=device)
    Q = conserved(mass, rho, vel, P, G)
    state = MeshlessState(
        positions=x[:, None], Q=Q, supports=h,
        kinds=torch.zeros(n, dtype=torch.int32, device=device),
        materials=torch.zeros(n, dtype=torch.int32, device=device),
        UIDs=torch.arange(n, dtype=torch.int32, device=device),
        UIDcounter=n)
    return state, Q


def make_context(n, nngb, device, mode, order):
    dom = DomainDescription(torch.zeros(1, dtype=torch.float64, device=device),
                            torch.ones(1, dtype=torch.float64, device=device),
                            torch.ones(1, dtype=torch.bool, device=device), 1)
    state, Q = make_state(n, nngb, device)
    system = MeshlessSystem(state=state, domain=dom)
    config = SimulationConfig(device=device, dtype=torch.float64, domain=dom, dim=1,
                              kernel=KernelFunctions.Wendland2)
    # the step owns its full time step (D4), so it must run under a single-evaluation
    # integrator; and G1 is the port-fidelity gate, so pin the core driver's temporal scheme
    config.integrationScheme = IntegrationSchemeType.forwardEuler
    schemeConfig = MeshlessConfig(closure='project', cflLength='support', mode=mode, order=order, nngb=nngb, cfl=CFL, gamma=G,
                                  temporalScheme='hancock')
    return system, config, schemeConfig, dom, Q


def build_geom(state, dom, config, schemeConfig):
    return MeshlessWarp.build(
        state.positions, state.supports, dom, config.kernel,
        cond_max=schemeConfig.condMax, closure=schemeConfig.closure,
        closure_power=schemeConfig.closurePower, rtol=None, cg_tol=schemeConfig.cgTol,
        cg_iters=schemeConfig.cgIters, lam0=state.lam, guards=True,
        face_cond_max=schemeConfig.faceCondMax, centred_weights=True, area_cap=False,
        closure_solver=schemeConfig.closureSolver,
        mg_min_particles=schemeConfig.mgMinParticles, mg_max_coarse=schemeConfig.mgMaxCoarse)


def score(state, prim, tend):
    r, v, p = [t.cpu().numpy() for t in prim]
    x = state.positions[:, 0].cpu().numpy()
    xr = np.where(x >= 0.5, x, 1.0 - x)
    s = ex.sample((xr - 0.75) / tend, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, G)
    sign = np.where(x >= 0.5, 1.0, -1.0)
    exr, exv, exp_ = s[:, 0], sign * s[:, 1], s[:, 2]
    return (np.abs(r - exr).mean(), np.abs(v[:, 0] - exv).mean(), np.abs(p - exp_).mean())


def run(n, mode, order, nngb, tend, device):
    system, config, schemeConfig, dom, Q0 = make_context(n, nngb, device, mode, order)
    state = system.state
    Q0sum = Q0.sum(0)
    t = 0.0
    steps = 0
    while t < tend - 1e-12:
        g = build_geom(state, dom, config, schemeConfig)
        r, v, p = primitives(state.Q, g.volume, G)
        dt = float(g.timestep(r, v, p, G, CFL).min())
        dt = min(dt, tend - t)
        rates = g.rates(r, v, p, G, dt=dt, mode=mode, order=order)
        Qn = state.Q + dt * rates
        vel_n = Qn[:, 1:-1] / Qn[:, 0:1]
        pos_n = wrap(state.positions + 0.5 * dt * (v + vel_n), dom)
        state.Q = Qn
        state.positions = pos_n
        state.supports = supports_from_volume(g.volume, nngb, 1)
        state.lam = g.lam
        t += dt
        steps += 1
    g = build_geom(state, dom, config, schemeConfig)
    r, v, p = primitives(state.Q, g.volume, G)
    cons = (state.Q.sum(0) - Q0sum).abs().max().item()
    return score(state, (r, v, p), tend), cons, steps


def check_step_function(n, mode, order, nngb, device):
    """Confirm meshlessStep (the hancock exact-delta step) returns the same rates / drift as
    the direct backend calls (bitwise, same geometry): dQdt == rates, dxdt == the midpoint
    drift 0.5 * (v + v_n)."""
    system, config, schemeConfig, dom, _ = make_context(n, nngb, device, mode, order)
    state = system.state
    g = build_geom(state, dom, config, schemeConfig)
    r, v, p = primitives(state.Q, g.volume, G)
    dt = float(g.timestep(r, v, p, G, CFL).min())
    rates = g.rates(r, v, p, G, dt=dt, mode=mode, order=order)
    Qn = state.Q + dt * rates
    vel_n = Qn[:, 1:-1] / Qn[:, 0:1]
    update, _adj, _cs = meshlessStep(system, dt, config, schemeConfig)
    dq_ok = bool(torch.allclose(update.dQdt, rates, rtol=0, atol=0))
    dx_ok = bool(torch.allclose(update.dxdt, 0.5 * (v + vel_n), rtol=0, atol=0))
    return dq_ok, dx_ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[200, 400])
    ap.add_argument("--tend", type=float, default=0.12)
    ap.add_argument("--nngb", type=float, default=7.0)
    ap.add_argument("--modes", nargs="+", default=["MFM", "MFV"])
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args(argv)
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mode in a.modes:
        for order in a.orders:
            dq_ok, dx_ok = check_step_function(64, mode, order, a.nngb, device)
            for n in a.n:
                t0 = time.time()
                (l1rho, l1v, l1p), cons, steps = run(n, mode, order, a.nngb, a.tend, device)
                print(f"{mode} order {order} n={n:5d}: L1 rho {l1rho:.4e} v {l1v:.4e} P {l1p:.4e}  "
                      f"|dQ| {cons:.1e}  steps {steps}  stepfn(dQ,dx)={dq_ok},{dx_ok}  "
                      f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
