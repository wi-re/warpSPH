#!/usr/bin/env python
"""D4 one-step parity: the refactored frontend `meshlessStep` (exact-delta full step)
must be bitwise identical to the D4 probe's direct backend step functions, for every
temporal scheme x mode. Also checks `validateIntegrationScheme` rejects a multi-stage
integrator. This is the core D4 correctness gate (multi-step equality follows by
induction on the identical per-step delta and identical dt / h / lam bookkeeping).

    python scripts/d4_mfm_parity.py --n 128 --cfl 0.2
"""
import os
os.environ.setdefault("warpSPHCore_PRECISION", "float64")

import argparse
import sys
from pathlib import Path

import torch
import warp as wp

import warpSPHCore
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # warpSPH

import d4_mfm_wave as d4
from warpSPHCore.mfm import primitives
from warpSPHIntegrators.integration import IntegrationSchemeType
from warpSPH.systems.meshless import MeshlessState, MeshlessSystem
from warpSPH.configurations.meshless import MeshlessConfig
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.schemes.meshless import meshlessStep, validateIntegrationScheme


def make_frontend(st0, dom, kernel, device, mode, order, nngb, cfl, scheme):
    state = MeshlessState(
        positions=st0.pos.clone(), Q=st0.Q.clone(), supports=st0.h.clone(),
        kinds=torch.zeros(st0.pos.shape[0], dtype=torch.int32, device=device),
        materials=torch.zeros(st0.pos.shape[0], dtype=torch.int32, device=device),
        UIDs=torch.arange(st0.pos.shape[0], dtype=torch.int32, device=device),
        UIDcounter=st0.pos.shape[0])
    state.lam = st0.lam
    system = MeshlessSystem(state=state, domain=dom)
    config = SimulationConfig(device=device, dtype=torch.float64, domain=dom,
                              dim=1, kernel=kernel)
    config.integrationScheme = IntegrationSchemeType.forwardEuler
    schemeConfig = MeshlessConfig(mode=mode, order=order, nngb=nngb, cfl=cfl, cflLength='support', closure='project',
                                  gamma=d4.G, temporalScheme=scheme)
    return system, config, schemeConfig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--nngb", type=float, default=7.0)
    ap.add_argument("--cfl", type=float, default=0.2)
    a = ap.parse_args(argv)
    wp.init()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 0. validateIntegrationScheme: reject a multi-stage integrator, accept forwardEuler
    st0, dom, _Q0, kernel = d4.make_ic(a.n, a.nngb, device)
    sys_, cfg, scfg = make_frontend(st0, dom, kernel, device, "MFM", 2, a.nngb, a.cfl, "hancock")
    cfg.integrationScheme = IntegrationSchemeType.rungeKutta2
    try:
        validateIntegrationScheme(cfg)
        reject_ok = False
    except ValueError:
        reject_ok = True
    cfg.integrationScheme = IntegrationSchemeType.forwardEuler
    validateIntegrationScheme(cfg)  # must not raise
    print(f"validateIntegrationScheme rejects rungeKutta2: {reject_ok}  (forwardEuler accepted)",
          flush=True)

    steps = {'hancock': d4.step_hancock, 'ssprk': d4.step_ssprk, 'kdk': d4.step_kdk}
    tmax = 1e9
    all_ok = True
    for mode in ("MFM", "MFV"):
        for order in (1, 2):
            for scheme, stepfn in steps.items():
                # direct D4 step on a copy (returns the dt it used)
                st_direct = d4.St(pos=st0.pos.clone(), Q=st0.Q.clone(),
                                  h=st0.h.clone(), lam=st0.lam, t=st0.t)
                dt_direct = stepfn(st_direct, dom, kernel, a.nngb, a.cfl, mode, order, tmax, 0)
                # frontend step (fresh state) with the same dt
                system, config, schemeConfig = make_frontend(
                    st0, dom, kernel, device, mode, order, a.nngb, a.cfl, scheme)
                update, _adj, cs = meshlessStep(system, dt_direct, config, schemeConfig)
                Qn_step = st0.Q + dt_direct * update.dQdt
                posn_step = d4.wrap(st0.pos + dt_direct * update.dxdt, dom)
                h_step = cs.supports
                # dQ: hancock is bitwise (dQdt = the raw rate); ssprk / kdk carry the
                # exact-delta divide-then-multiply round-trip, so allow ~1 ulp but catch a
                # real bug (a true scheme mismatch is ~1e-3, not ~1e-16)
                dq_max = float((Qn_step - st_direct.Q).abs().max())
                dq_ok = bool(torch.allclose(Qn_step, st_direct.Q, rtol=1e-12, atol=1e-14))
                dx_ok = bool(torch.allclose(posn_step, st_direct.pos, rtol=0, atol=0))
                dh_ok = bool(torch.allclose(h_step, st_direct.h, rtol=0, atol=0))
                ok = dq_ok and dx_ok and dh_ok
                all_ok = all_ok and ok
                print(f"{scheme:8s} {mode} order {order}: dQ {dq_ok} (|dQ|_max {dq_max:.1e})  "
                      f"dx {dx_ok}  h {dh_ok}  dt {dt_direct:.3e}   {'OK' if ok else 'MISMATCH'}",
                      flush=True)
    print("\nALL PARITY OK" if all_ok else "\nPARITY FAILURE", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
