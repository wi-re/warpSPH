"""Water-column collapse in a closed tank under the incompressible
`divergenceFree` scheme, rendered to video -- the free-surface companion to
`make_video_dfsphDambreak.py`.

`columnCollapse` (a released water column oscillating in a closed tank) has
gravity, so it takes the same Part 47 in-step constant-density path as
`dambreak` / `hydrostaticColumn`, and like `dambreak` uses `semiImplicitEuler`
+ a small `cflFactor` (case default 0.2) for the collapse impact.

NOTE: a few fluid particles penetrate the wall band during the impact -- see
the case module docstring / DFSPH_FINDINGS.md; the run reports `nPenetrating`
and `maxPenetrationDx` each step.

NOT a sloshing tank in the usual sense (wide shallow basin under an
oscillating gravity) -- that is a separate TODO case.

    python scripts/make_video_dfsphColumnCollapse.py [tLimit] [nx] [tiltDeg]
    python scripts/make_video_dfsphColumnCollapse.py 3.0 64 0

Videos land in scripts/videos_dfsph_columnCollapse/<tag>/{output.mp4,out.gif}.
"""
import os
import shutil
import sys

import numpy as np

from warpSPH.cases.columnCollapse import columnCollapseCase
from warpSPH.runner import run

OUT = os.path.join(os.path.dirname(__file__), "videos_dfsph_columnCollapse")
os.makedirs(OUT, exist_ok=True)

TLIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 64
TILT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

tag = f"columnCollapse_nx{NX}_tilt{TILT:g}_divergenceFree"
r = run(columnCollapseCase, nx=NX, scheme="divergenceFree",
        integrationScheme="semiImplicitEuler", tLimit=TLIMIT,
        params=dict(tiltDeg=TILT),
        quiet=True, plot=True, video=True, store=False, progress=False,
        plotBackend="matplotlib", plotInterval=8)

rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
ke = np.array([x.get("kineticEnergy", np.nan) for x in rows])
vm = np.array([x.get("maxVelocity", np.nan) for x in rows])
rho = np.array([x.get("maxDensity", np.nan) for x in rows])
lh = np.array([x.get("leftWallHeight", np.nan) for x in rows])
rh = np.array([x.get("rightWallHeight", np.nan) for x in rows])
npen = np.array([x.get("nPenetrating", 0) for x in rows])
pdx = np.array([x.get("maxPenetrationDx", np.nan) for x in rows])
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
print(f"  |v|max peak {np.nanmax(vm):.3g}  KE peak {np.nanmax(ke):.4g} -> {ke[-1]:.4g}  "
      f"maxRho peak {np.nanmax(rho):.4f}")
print(f"  wall height L {np.nanmin(lh):.3f}..{np.nanmax(lh):.3f}  "
      f"R {np.nanmin(rh):.3f}..{np.nanmax(rh):.3f}")
print(f"  wall penetration: max {int(npen.max())} particles, "
      f"deepest {np.nanmax(pdx):.2f} dx")
print(f"  -> {dest}/  ({', '.join(copied) or 'NO VIDEO'})", flush=True)
sys.exit(0 if stable else 1)
