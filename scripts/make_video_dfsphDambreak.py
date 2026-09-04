"""Dam break under the incompressible `divergenceFree` scheme, rendered to
video -- the free-surface counterpart to the weakly-compressible
`examples/weaklyCompressible/12-dambreak.py`.

`dambreak` has gravity, so the Part 47 gate routes it through the in-step
constant-density velocity impulse (`divergenceFree.INSTEP_CD` auto-on), same as
`hydrostaticColumn`. Per `DFSPH_IMPROVEMENT_PLAN.md` Part 20 this scheme needs
`semiImplicitEuler` + `cflFactor = 0.2` (it diverges at the published 0.4/0.3),
and holds at `nx = 64` but NOT the case default `nx = 128` (NaN ~step 88).

    python scripts/make_video_dfsphDambreak.py [tLimit] [nx]
    python scripts/make_video_dfsphDambreak.py 3.0 64

Videos land in scripts/videos_dfsph_dambreak/<tag>/{output.mp4,out.gif}.
"""
import os
import shutil
import sys

import numpy as np

from warpSPH.cases.dambreak import dambreakCase
from warpSPH.runner import run

OUT = os.path.join(os.path.dirname(__file__), "videos_dfsph_dambreak")
os.makedirs(OUT, exist_ok=True)

TLIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 64

tag = f"dambreak_nx{NX}_divergenceFree"
r = run(dambreakCase, nx=NX, scheme="divergenceFree", kernel="Wendland4",
        integrationScheme="semiImplicitEuler", adaptiveDt=True, cflFactor=0.2,
        tLimit=TLIMIT, quiet=True, plot=True, video=True, store=False,
        progress=False, plotBackend="matplotlib", plotInterval=8)

rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
ke = np.array([x.get("kineticEnergy", np.nan) for x in rows])
vm = np.array([x.get("maxVelocity", np.nan) for x in rows])
rho = np.array([x.get("maxDensity", np.nan) for x in rows])
t = np.array([x.get("t", np.nan) for x in rows])
stable = (not r.diverged) and np.all(np.isfinite(ke)) and np.all(np.isfinite(vm))

dest = os.path.join(OUT, tag)
copied = []
if r.videoPath and os.path.exists(r.videoPath):
    os.makedirs(dest, exist_ok=True)
    for f in ("output.mp4", "out.gif"):
        src = os.path.join(os.path.dirname(r.videoPath), f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dest, f))
            copied.append(f)

print(f"{tag}")
print(f"  ran {len(rows)} steps to t={t[-1]:.3f}/{TLIMIT}  diverged={r.diverged}  "
      f"STABLE={stable}")
print(f"  |v|max peak {np.nanmax(vm):.3g}  KE {ke[0]:.3g} -> {ke[-1]:.3g}  "
      f"maxRho peak {np.nanmax(rho):.4f}")
print(f"  -> {dest}/  ({', '.join(copied) or 'NO VIDEO'})", flush=True)
sys.exit(0 if stable else 1)
