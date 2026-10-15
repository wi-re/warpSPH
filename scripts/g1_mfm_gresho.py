#!/usr/bin/env python
"""G1 gate probe, part 2 (COMPRESSIBLE_SPH_PLAN E1): the frontend MFM / MFV scheme
reproduces the core harness (`run_mfm_gresho.py`) on the Gresho-Chan vortex.

The vortex is a steady state, so the volume-weighted L1 error |v - v_exact| after
t = 3 is pure scheme error; the KE growth next to it is the Phase-1 anomaly the
Riemann-flux meshless scheme must not show (the CRKSPH pair pressure force used to
inject kinetic energy). Uses the frontend's `MeshlessState` (the conserved-Q state)
and the core `MeshlessWarp` backend -- the same calls the E1 step function makes --
mirroring `mfm_driver.step` exactly, with the reference IC (hexagonal lattice,
rho = 1, gamma = 5/3, periodic unit box, the standard piecewise profile, 4 support
pre-refine iterations). A separate check confirms the E1 step function
(`meshlessStep`) returns the same rates / drift as the direct backend calls
(bitwise, same geometry).

    python scripts/g1_mfm_gresho.py --n 4096 --modes MFM MFV --orders 2

Oracle (core harness run_mfm_gresho.py, float64, Wendland4, nngb 30, closure
project, order 2, tend 3.0, N = 4096, tag mfm-oracle-2026-10-11):
    L1(v)  0.0689 (MFM) / 0.0569 (MFV); conservation |dQ| ~ 1e-15
    KE growth small (no rebound). (The plan's earlier 0.066/0.051 predate the tag.)
"""
import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import math
import sys
import time

import torch
import warp as wp

from pathlib import Path
import warpSPHCore

# the core evidence harness (mfm_driver, particle_sets): shipped with warpSPHCore;
# override with WARPSPHCORE_HARNESS for a separate checkout
_HARNESS = Path(os.environ.get(
    "WARPSPHCORE_HARNESS",
    str(Path(warpSPHCore.__file__).resolve().parents[2] / "higherOrderSPH" / "harness")))
sys.path.insert(0, str(_HARNESS))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # warpSPH

from particle_sets import build_case
from warpSPHCore.enumTypes import KernelFunctions
from warpSPHCore.mfm import MeshlessWarp, primitives, conserved

from warpSPHIntegrators.schemes import IntegrationSchemeType
from warpSPH.systems.meshless import MeshlessState, MeshlessSystem
from warpSPH.configurations.meshless import MeshlessConfig
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.schemes.compressible.meshless import meshlessStep, supports_from_volume

G = 5.0 / 3.0
P_OUTER = 3 + 4 * math.log(2)
CFL = 0.2
DIM = 2


def wrap(pos, domain):
    L = domain.max - domain.min
    per = domain.periodic
    return torch.where(per, domain.min + torch.remainder(pos - domain.min, L), pos)


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


def make_context(n, nngb, device, mode, order, kernel, jitter, seed):
    case = build_case(n, DIM, int(nngb), jitter=jitter, seed=seed, periodic=True,
                      device=device, kernel=kernel)
    dt_ = torch.float64
    x = case.particles.positions.to(dt_)
    dom = case.domain
    box = torch.as_tensor(case.box, dtype=dt_, device=device)
    centre = 0.5 * box
    N = x.shape[0]
    V0 = torch.full((N,), float(case.cell_vol), dtype=dt_, device=device)
    state = MeshlessState(
        positions=x, Q=torch.zeros(N, 2 + DIM, dtype=dt_, device=device),
        supports=supports_from_volume(V0, nngb, DIM),
        kinds=torch.zeros(N, dtype=torch.int32, device=device),
        materials=torch.zeros(N, dtype=torch.int32, device=device),
        UIDs=torch.arange(N, dtype=torch.int32, device=device),
        UIDcounter=N)
    config = SimulationConfig(device=device, dtype=dt_, domain=dom, dim=DIM, kernel=kernel)
    # the step owns its full time step (D4), so it must run under a single-evaluation
    # integrator; and G1 is the port-fidelity gate, so pin the core driver's temporal scheme
    config.integrationScheme = IntegrationSchemeType.forwardEuler
    schemeConfig = MeshlessConfig(closure='project', cflLength='support', mode=mode, order=order, nngb=nngb, cfl=CFL, gamma=G,
                                  temporalScheme='hancock')
    system = MeshlessSystem(state=state, domain=dom)
    return system, config, schemeConfig, dom, box, centre


def build_geom(state, dom, config, schemeConfig):
    return MeshlessWarp.build(
        state.positions, state.supports, dom, config.kernel,
        cond_max=schemeConfig.condMax, closure=schemeConfig.closure,
        closure_power=schemeConfig.closurePower, rtol=None, cg_tol=schemeConfig.cgTol,
        cg_iters=schemeConfig.cgIters, lam0=state.lam, guards=True,
        face_cond_max=schemeConfig.faceCondMax, centred_weights=True, area_cap=False,
        closure_solver=schemeConfig.closureSolver,
        mg_min_particles=schemeConfig.mgMinParticles, mg_max_coarse=schemeConfig.mgMaxCoarse)


def init_vortex(state, dom, box, centre, config, schemeConfig, nngb):
    """The reference IC: rho = 1, the piecewise vortex profile, and the 4 support
    pre-refine iterations of run_mfm_gresho. Returns the initial total Q sum."""
    x = state.positions
    v, r = vortex_velocity(x, centre, box)
    _, P = profile(r)
    g0 = None
    for _ in range(4):
        g0 = build_geom(state, dom, config, schemeConfig)
        state.supports = supports_from_volume(g0.volume, nngb, DIM)
        state.lam = g0.lam
    # match the core harness: `drv.init_state` builds a fresh state without the
    # pre-refine closure potential, so the main loop cold-starts the first build
    state.lam = None
    rho = torch.full((x.shape[0],), 1.0, dtype=torch.float64, device=x.device)
    state.Q = conserved(rho * g0.volume, rho, v, P, G)
    return state.Q.sum(0)


def run(n, mode, order, nngb, tend, device, kernel, jitter, seed, log):
    system, config, schemeConfig, dom, box, centre = make_context(
        n, nngb, device, mode, order, kernel, jitter, seed)
    state = system.state
    Q0sum = init_vortex(state, dom, box, centre, config, schemeConfig, nngb)
    ke0 = ke_of(state.Q)
    t = 0.0
    steps = 0
    kes = []
    t0 = time.time()
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
        state.supports = supports_from_volume(g.volume, nngb, DIM)
        state.lam = g.lam
        t += dt
        steps += 1
        if steps % 20 == 0:
            kes.append(ke_of(state.Q) / ke0)
        if log and steps % 200 == 0:
            print(f"   t {t:.2f} steps {steps} ({time.time()-t0:.0f}s)", flush=True)
    g = build_geom(state, dom, config, schemeConfig)
    r, v, p = primitives(state.Q, g.volume, G)
    vex, _ = vortex_velocity(state.positions, centre, box)
    w = g.volume
    l1 = float((w * (v - vex).norm(dim=-1)).sum() / w.sum())
    ke1 = ke_of(state.Q) / ke0
    kemax = max(kes) if kes else ke1
    cons = (state.Q.sum(0) - Q0sum).abs().max().item()
    return l1, ke1, kemax, cons, steps


def check_step_function(n, mode, order, nngb, device, kernel, jitter, seed):
    """Confirm meshlessStep (the hancock exact-delta step) returns the same rates / drift as
    the direct backend calls (bitwise, same geometry): dQdt == rates, dxdt == the midpoint
    drift 0.5 * (v + v_n)."""
    system, config, schemeConfig, dom, box, centre = make_context(
        n, nngb, device, mode, order, kernel, jitter, seed)
    state = system.state
    init_vortex(state, dom, box, centre, config, schemeConfig, nngb)
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
    ap.add_argument("--n", type=int, nargs="+", default=[4096])
    ap.add_argument("--modes", nargs="+", default=["MFM", "MFV"])
    ap.add_argument("--orders", type=int, nargs="+", default=[2])
    ap.add_argument("--nngb", type=float, default=30.0)
    ap.add_argument("--kernel", default="Wendland4")
    ap.add_argument("--tend", type=float, default=3.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-n", type=int, default=64)
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args(argv)
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kernel = KernelFunctions[a.kernel]
    for mode in a.modes:
        for order in a.orders:
            dq_ok, dx_ok = check_step_function(
                a.check_n, mode, order, a.nngb, device, kernel, a.jitter, a.seed)
            for n in a.n:
                t0 = time.time()
                l1, ke1, kemax, cons, steps = run(
                    n, mode, order, a.nngb, a.tend, device, kernel, a.jitter, a.seed, a.log)
                print(f"{mode} order {order} N={n:5d}: L1(v) {l1:.4f}  KE/KE0 final {ke1:.4f} "
                      f"max {kemax:.4f}  |dQ| {cons:.1e}  steps {steps}  "
                      f"stepfn(dQ,dx)={dq_ok},{dx_ok}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
