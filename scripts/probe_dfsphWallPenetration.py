"""Boundary-penetration A/B for the post-`c637785` `divergenceFree` scheme.

FINDINGS 1.6: the `c637785` rewrite dropped the mDBC no-penetration velocity
shift (`computeMdbcNoPenShift`) -- the call is commented out in `divergenceFree_step`,
so the shipped scheme relies on the pressure projection alone. FINDINGS 2 had
measured (on the *pre*-rewrite VD+PS scheme) that removing it is strictly worse
on the wall-crossing metrics. This re-grades that on the rewritten step.

`schemes/divergenceFree.NOPEN_SHIFT` gates the restored call: 'config' (obey the solver
config, default True), True, or False. This probe runs each wall-bounded /
free-surface case with the shift OFF and ON and reports the per-run wall
penetration (`nPenetrating` peak, `maxPenetrationDx` peak) plus the usual
stability FOMs, so the default flip can be made on a measurement.

    python scripts/probe_dfsphWallPenetration.py [case ...]

cases: columnCollapse dambreak hydrostaticColumn randomFlowBounded  (default: all)
"""
import os
import sys

import numpy as np

from warpSPH.runner import run
import warpSPH.schemes.divergenceFree as dfsph


def _series(r, key, default=np.nan):
    return np.array([x.get(key, default) for x in r.trajectory
                     if x.get("step", -1) >= 0], dtype=float)


def _grade(r):
    ke = _series(r, "kineticEnergy")
    vm = _series(r, "maxVelocity")
    rho = _series(r, "maxDensity")
    rlo = _series(r, "minDensity")
    npen = _series(r, "nPenetrating", 0.0)
    pdx = _series(r, "maxPenetrationDx", 0.0)
    finite = np.all(np.isfinite(ke)) and np.all(np.isfinite(vm))
    return dict(
        steps=len(ke),
        stable=(not r.diverged) and finite,
        vmax=float(np.nanmax(vm)) if len(vm) else np.nan,
        ke_last=float(ke[-1]) if len(ke) else np.nan,
        rho_hi=float(np.nanmax(rho)) if len(rho) else np.nan,
        rho_lo=float(np.nanmin(rlo)) if len(rlo) else np.nan,
        npen_peak=int(np.nanmax(npen)) if len(npen) else 0,
        npen_final=int(npen[-1]) if len(npen) else 0,
        pdx_peak=float(np.nanmax(pdx)) if len(pdx) else 0.0,
    )


def _load(case):
    if case == "columnCollapse":
        from warpSPH.cases.columnCollapse import columnCollapseCase as c
        return c, dict(nx=64, tLimit=2.0), {}
    if case == "dambreak":
        from warpSPH.cases.dambreak import dambreakCase as c
        return c, dict(nx=64, scheme="divergenceFree", tLimit=1.5), {}
    if case == "hydrostaticColumn":
        from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase as c
        return c, dict(nx=64, tLimit=1.5), {}
    if case == "randomFlowBounded":
        from warpSPH.cases.randomFlowIncompressible import randomFlowIncompressibleCase as c
        return c, dict(nx=64, tLimit=1.5), dict(bounded=True)
    raise SystemExit(f"unknown case {case!r}")


CASES = sys.argv[1:] or ["columnCollapse", "dambreak", "hydrostaticColumn",
                         "randomFlowBounded"]

RESULTS = os.path.join(os.path.dirname(__file__),
                       "probe_dfsphWallPenetration.results.txt")


def emit(line):
    print(line, flush=True)
    with open(RESULTS, "a") as fh:
        fh.write(line + "\n")


open(RESULTS, "w").close()
for case in CASES:
    c, kw, params = _load(case)
    emit(f"\n=== {case}  {kw} {params} ===")
    for label, mode in (("shift OFF", False), ("shift ON ", True)):
        divergenceFree.NOPEN_SHIFT = mode
        r = run(c, params=dict(params), quiet=True, plot=False, store=False,
                progress=False, **kw)
        g = _grade(r)
        tag = "OK  " if g["stable"] else "BLOW"
        emit(f"  {label}  [{tag}] {g['steps']:4d} steps  "
             f"|v|max {g['vmax']:8.3g}  KE_end {g['ke_last']:9.3g}  "
             f"rho [{g['rho_lo']:.3f},{g['rho_hi']:.3f}]  "
             f"pen peak {g['npen_peak']:4d} (final {g['npen_final']:4d})  "
             f"deepest {g['pdx_peak']:.2f} dx")

divergenceFree.NOPEN_SHIFT = "config"
