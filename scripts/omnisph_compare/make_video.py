"""Render the hydrostatic column under `omniIncompressible` at nx=128 -- the
case Part 41's per-iterate MLS wall pressure fixes. Two arms so the fix is
visible: wall pressure OFF (diverges at the bottom corners) and the shipped
default 'mls' (holds at the exact hydrostatic gradient).

    python scripts/omnisph_compare/make_video.py [nsteps]   # both arms
    python scripts/omnisph_compare/make_video.py [nsteps] mls|off
"""
import os
import shutil
import sys

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run
import warpSPH.schemes.omniIncompressible as omod

OUT = "/home/lu26029/dev/warpSPH/scripts/omnisph_compare/videos"
os.makedirs(OUT, exist_ok=True)

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
arms = [sys.argv[2]] if len(sys.argv) > 2 else ["mls", "off"]

for arm in arms:
    omod.WALL_PRESSURE_MODE = None if arm == "off" else arm
    tag = f"hydrostaticColumn_nx128_omniIncompressible_wallP-{arm}"
    r = run(hydrostaticColumn.hydrostaticColumnCase, nx=128, nSteps=NS,
            scheme="omniIncompressible", kernel="Wendland2",
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
    print(f"{arm:4s}  ran {len(rows)}/{NS}  diverged={r.diverged}  "
          f"|v|max peak {max(vm) if vm else float('nan'):.4g} last {vm[-1] if vm else float('nan'):.3g}  "
          f"-> {dest}/output.mp4", flush=True)

omod.WALL_PRESSURE_MODE = "mls"
