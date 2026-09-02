"""Render `hydrostaticColumn` under `IncompressibleSPHScheme.band2018pb`
(Band et al. 2018 pressure boundaries -- DFSPH_IMPROVEMENT_PLAN.md Part 45) at
nx=64 (holds, quiescent) and nx=128 (bounded but sprays at the free surface).

    python scripts/make_video_band2018pb.py [nsteps]        # both
    python scripts/make_video_band2018pb.py [nsteps] 64|128  # one
"""
import os
import shutil
import sys

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run

OUT = "/home/lu26029/dev/warpSPH/scripts/videos_band2018pb"
os.makedirs(OUT, exist_ok=True)

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
nxs = [int(sys.argv[2])] if len(sys.argv) > 2 else [64, 128]

for nx in nxs:
    tag = f"hydrostaticColumn_nx{nx}_band2018pb"
    r = run(hydrostaticColumn.hydrostaticColumnCase, nx=nx, nSteps=NS,
            scheme="band2018pb", kernel="Wendland2",
            quiet=True, plot=True, video=True, store=False, progress=False,
            plotBackend="matplotlib", plotInterval=4,
            integrationScheme="semiImplicitEuler")
    rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
    vm = [x.get("maxVelocity", float("nan")) for x in rows]
    vm = [v for v in vm if v == v]
    dest = os.path.join(OUT, tag)
    if r.videoPath and os.path.exists(r.videoPath):
        os.makedirs(dest, exist_ok=True)
        for f in ("output.mp4", "out.gif"):
            src = os.path.join(os.path.dirname(r.videoPath), f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dest, f))
    print(f"nx={nx:3d}  ran {len(rows)}/{NS}  diverged={r.diverged}  "
          f"|v|max peak {max(vm) if vm else float('nan'):.4g} "
          f"last {vm[-1] if vm else float('nan'):.3g}  -> {dest}/output.mp4",
          flush=True)
