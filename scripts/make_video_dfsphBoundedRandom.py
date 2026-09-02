"""Video A/B on `randomFlowIncompressible --bounded` (closed box, walls, no
free surface, decaying) -- the case the DFSPH_FINDINGS.md 1.16 experiment
turns out to *also* fix.

  * baseline    -- as shipped (omni divergence solve, in-step CD velocity,
                   no VD+PS shift). At fixed dt=1e-3 nx=64 it DETONATES:
                   KE x60, |v| -> 10.7 in a unit-speed flow.
  * ps-restore  -- IncompressibleSystem._RESTORE_PS_SHIFT + dfsph.INSTEP_CD=False
                   (the same setting that fixes tgv). KE x0.994, |v| ~ 1.0,
                   density band 8.5e-3.

    python scripts/make_video_dfsphBoundedRandom.py [nsteps] [plotInterval]
"""
import os, shutil, sys
import numpy as np
import warpSPH.systems.incompressible as I
from warpSPH.cases import randomFlowIncompressible as rf
from warpSPH.runner import run
from warpSPH.schemes import dfsph as D

OUT = "/home/lu26029/dev/warpSPH/scripts/videos/dfsph_bounded_random"
os.makedirs(OUT, exist_ok=True)
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
PI = int(sys.argv[2]) if len(sys.argv) > 2 else 4

ARMS = [
    ("randFlowBounded_nx64_baseline",   dict(ps=False, cd=True)),
    ("randFlowBounded_nx64_psRestore",  dict(ps=True,  cd=False)),
]
for tag, fl in ARMS:
    I._RESTORE_PS_SHIFT = fl["ps"]; D.INSTEP_CD = fl["cd"]
    D.DIVERGENCE_SOLVER = "omni"; D.XSPH_SCALE = 0.0; D.SOLVE_ORDER = "div_then_cd"
    I._PS_SHIFT_MODE = "cd"; I._PS_POSITION_SHIFT = True
    I._PS_VELOCITY_RESAMPLE = True; I._PS_SHIFT_AS_VELOCITY = False
    r = run(rf.randomFlowIncompressibleCase, nx=64, nSteps=NS, scheme="divergenceFree",
            kernel="Wendland2", quiet=True, plot=True, video=True, store=False,
            progress=False, plotBackend="matplotlib", plotInterval=PI,
            integrationScheme="semiImplicitEuler", params={"bounded": True},
            dt=1e-3, minDt=1e-3, maxDt=1e-3, adaptiveDt=False)
    rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
    ke = [x["kineticEnergy"] for x in rows]
    vm = [x["maxVelocity"] for x in rows]
    dest = os.path.join(OUT, tag); copied = []
    if r.videoPath and os.path.exists(r.videoPath):
        os.makedirs(dest, exist_ok=True)
        for f in ("output.mp4", "out.gif"):
            src = os.path.join(os.path.dirname(r.videoPath), f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dest, f)); copied.append(f)
    print(f"{tag}\n  ran {len(rows)}/{NS} div={r.diverged}  "
          f"KE {ke[0]:.4g}->{ke[-1]:.4g} (x{ke[-1]/ke[0]:.4f}, peak x{max(ke)/ke[0]:.3f})  "
          f"|v| {vm[0]:.2f}->{vm[-1]:.2f} (peak {max(vm):.2f})\n  -> {dest}/  ({', '.join(copied) or 'NO VIDEO'})",
          flush=True)
I._RESTORE_PS_SHIFT = False; D.INSTEP_CD = True
