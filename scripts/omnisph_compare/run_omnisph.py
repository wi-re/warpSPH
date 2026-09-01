"""Run an omniSPH incompressible case through its Python bindings and report
whether it holds -- the reference the warpSPH `omniIncompressible` port was
transcribed from.

    python scripts/omnisph_compare/run_omnisph.py <nsteps> [config.yaml]

Default config: scripts/omnisph_compare/column.yaml (a hydrostatic column
matched to warpSPH's `hydrostaticColumn`: 1x1 box, fluid fills the bottom
half, gravity down, DFSPH incompressible). Needs the omnySPH _core built for
this env (scripts/omnisph_compare/build_omnysph.sh).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dev/omniSPH/omnySPH/src"))
import omnySPH  # noqa: E402

nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
cfg_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(__file__), "column.yaml")
sim = omnySPH.SPHSimulation(open(cfg_path).read())

n = sim.getInteger("props.numPtcls")
print(f"omniSPH {os.path.basename(cfg_path)}: numPtcls={n}  "
      f"minDt={sim.getScalar('sim.minDt')}  "
      f"h/dx~{np.sqrt(20)*0.399200743:.2f}")


def stats():
    P = np.asarray(sim.fluidPosition)[:n]
    V = np.asarray(sim.fluidVelocity)[:n]
    D = np.asarray(sim.fluidDensity)[:n]
    sp = np.linalg.norm(V, axis=1)
    # embedded density: fluid rows > 1 "dx" below the 95th-pct surface
    ys = P[:, 1]
    surf95 = np.quantile(ys, 0.95)
    dx = 1.0 / 128
    emb = D[ys < surf95 - dx]
    return dict(vmax=sp.max(), vmean=sp.mean(),
               rmin=D.min(), rmax=D.max(), rmean=D.mean(),
               rp05=np.quantile(D, 0.05),
               embmin=(emb.min() if emb.size else float('nan')),
               ymax=ys.max(), ke=float(0.5 * (sp ** 2).sum() / n))


t0 = time.time()
hdr = f'{"step":>6} {"t":>7} {"vmax":>7} {"vmean":>8} {"KE/n":>9} ' \
      f'{"rmin":>7} {"rmax":>7} {"rp05":>7} {"embmin":>7} {"surfY":>7} {"it d/i":>8}'
print(hdr)
print("-" * len(hdr))
for k in range(0, nsteps + 1):
    if k:
        sim.timestep()
    if k % max(1, nsteps // 20) == 0 or k <= 3:
        s = stats()
        t = sim.getScalar("sim.time")
        di = sim.getInteger("dfsph.divergenceIterations")
        dd = sim.getInteger("dfsph.densityIterations")
        print(f'{k:6d} {t:7.4f} {s["vmax"]:7.3f} {s["vmean"]:8.5f} '
              f'{s["ke"]:9.2e} {s["rmin"]:7.4f} {s["rmax"]:7.4f} '
              f'{s["rp05"]:7.4f} {s["embmin"]:7.4f} {s["ymax"]:7.4f} '
              f'{di:3d}/{dd:<3d}')
        if not np.isfinite(s["vmax"]):
            print("  DIVERGED")
            break
print(f"({time.time() - t0:.1f}s)")
