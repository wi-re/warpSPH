"""Video A/B for the tgv energy injection (DFSPH_FINDINGS.md 1.16 / the
`_RESTORE_PS_SHIFT` experiment).

Two renders of `tgv --scheme divergenceFree`, nx=64, fixed dt=1e-3:

  * baseline   -- as shipped: no VD+PS particle shift, the constant-density
                  `_solve` folded into the velocity in-step. The jittered
                  lattice re-snaps in the first ~15 steps and the fluid KE
                  grows ~6-8% before it turns over.
  * ps-restore -- `IncompressibleSystem._RESTORE_PS_SHIFT = True` +
                  `dfsph.INSTEP_CD = False`: the true pre-tmp VD+PS
                  (divergence projection in-step, constant density only as the
                  finalize position shift). KE x1.000, monotone, no snap.

Also renders `hydrostaticColumn` nx=64 under the baseline (holds) for
reference -- ps-restore NaNs it (Part 23), so that arm is skipped.

    python scripts/make_video_dfsphPsCompare.py [nsteps]
"""
import os
import shutil
import sys

import numpy as np

import warpSPH.systems.incompressible as I
from warpSPH.cases import hydrostaticColumn, tgv
from warpSPH.runner import run
from warpSPH.schemes import dfsph as D

OUT = "/home/lu26029/dev/warpSPH/scripts/videos/dfsph_ps_compare"
os.makedirs(OUT, exist_ok=True)

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PLOT_INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 2

# (tag, case, per-run overrides, {flags})
ARMS = [
    ("tgv_nx64_baseline", tgv.tgvCase, dict(nx=64),
     dict(ps=False, instep_cd=True)),
    ("tgv_nx64_psRestore", tgv.tgvCase, dict(nx=64),
     dict(ps=True, instep_cd=False)),
    ("hydrostaticColumn_nx64_baseline", hydrostaticColumn.hydrostaticColumnCase,
     dict(nx=64, params=dict(calibrateRestDensity=True)),
     dict(ps=False, instep_cd=True)),
]

for tag, case, kw, flags in ARMS:
    I._RESTORE_PS_SHIFT = flags["ps"]
    D.INSTEP_CD = flags["instep_cd"]
    r = run(case, nSteps=NS, scheme="divergenceFree", kernel="Wendland2",
            quiet=True, plot=True, video=True, store=False, progress=False,
            plotBackend="matplotlib", plotInterval=PLOT_INTERVAL,
            integrationScheme="semiImplicitEuler",
            dt=1e-3, minDt=1e-3, maxDt=1e-3, adaptiveDt=False, **kw)
    rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
    ke = [x.get("kineticEnergy") for x in rows if x.get("kineticEnergy") is not None]
    vm = [x.get("maxVelocity") for x in rows if x.get("maxVelocity") is not None]
    dest = os.path.join(OUT, tag)
    copied = []
    if r.videoPath and os.path.exists(r.videoPath):
        os.makedirs(dest, exist_ok=True)
        for f in ("output.mp4", "out.gif"):
            src = os.path.join(os.path.dirname(r.videoPath), f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dest, f))
                copied.append(f)
    ke0, ke1 = (ke[0], ke[-1]) if ke else (float("nan"), float("nan"))
    kepk = max(ke) / ke0 if ke else float("nan")
    mono = bool(np.all(np.diff(ke) < 0)) if len(ke) > 1 else None
    print(f"{tag}\n  ran {len(rows)}/{NS}  diverged={r.diverged}  "
          f"KE {ke0:.4g} -> {ke1:.4g}  (x{ke1/ke0:.4f}, peak x{kepk:.4f}, mono={mono})  "
          f"|v|max last {vm[-1] if vm else float('nan'):.3g}\n"
          f"  -> {dest}/  ({', '.join(copied) or 'NO VIDEO'})", flush=True)

I._RESTORE_PS_SHIFT = False
D.INSTEP_CD = True
