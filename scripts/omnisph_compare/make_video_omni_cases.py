"""Render `omniIncompressible` (current defaults: WALL_PRESSURE_MODE='shepard',
CD_SOURCE_PROJECT='auto') on the two cases Part 42 closed:

  * randomFlowIncompressible --bounded  -- the closed box. With the CD-source
    compatibility projection the constant-density Jacobi now converges; the
    random field just decays.
  * tgv                                 -- periodic, no walls. The scheme's
    3-iter divergence pass holds it (where `iisph` injects energy).

    python scripts/omnisph_compare/make_video_omni_cases.py [nsteps]

Videos land in scripts/omnisph_compare/videos/<tag>/{output.mp4,out.gif}
(the videos/ tree is git-ignored).
"""
import os
import shutil
import sys

from warpSPH.cases import randomFlowIncompressible, tgv
from warpSPH.runner import run

OUT = os.path.join(os.path.dirname(__file__), "videos")
os.makedirs(OUT, exist_ok=True)

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 300

arms = [
    ("randFlowBounded_nx64_omniIncompressible",
     randomFlowIncompressible.randomFlowIncompressibleCase,
     dict(nx=64, params=dict(bounded=True))),
    ("tgv_nx128_omniIncompressible",
     tgv.tgvCase,
     dict(nx=128)),
]

for tag, case, kw in arms:
    r = run(case, nSteps=NS, scheme="omniIncompressible", kernel="Wendland2",
            quiet=True, plot=True, video=True, store=False, progress=False,
            plotBackend="matplotlib", plotInterval=4,
            integrationScheme="semiImplicitEuler", **kw)
    rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
    vm = [x.get("maxVelocity") for x in rows if x.get("maxVelocity") is not None]
    ke = [x.get("kineticEnergy") for x in rows if x.get("kineticEnergy") is not None]
    dest = os.path.join(OUT, tag)
    copied = []
    if r.videoPath and os.path.exists(r.videoPath):
        os.makedirs(dest, exist_ok=True)
        for f in ("output.mp4", "out.gif"):
            src = os.path.join(os.path.dirname(r.videoPath), f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dest, f))
                copied.append(f)
    print(f"{tag}\n  ran {len(rows)}/{NS}  diverged={r.diverged}  "
          f"|v|max peak {max(vm) if vm else float('nan'):.4g} last "
          f"{vm[-1] if vm else float('nan'):.3g}  "
          f"KE {ke[0]:.3e} -> {ke[-1]:.3e}\n"
          f"  -> {dest}/  ({', '.join(copied) or 'NO VIDEO'})", flush=True)
