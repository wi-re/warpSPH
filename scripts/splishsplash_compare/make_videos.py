"""Render every scheme on the hydrostatic column to mp4 -- native warpSPH IC and
the imported SPlisHSPlasH fluid state -- so the failure modes are visible.

Outputs land in scripts/splishsplash_compare/videos/<tag>/output.mp4 (+ .gif).
"""
import os, sys, json, shutil, dataclasses, traceback
import numpy as np
import torch

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run

OUT = "/home/lu26029/dev/warpSPH/scripts/splishsplash_compare/videos"
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "index.txt"), "a", buffering=1)

SCHEMES = ["divergenceFree", "iisph", "dfsphReference", "omniIncompressible"]

# ---- imported-SP initial condition (see import_and_run.py) -------------------
_sp = np.load("/home/lu26029/dev/warpSPH/scripts/splishsplash_compare/splish_fluid8001.npz",
              allow_pickle=True)
_spi = json.loads(str(_sp["info"]))
_sp_pos = torch.tensor(_sp["pos"], dtype=torch.float64)
_sp_mass = float(_spi["massMean"])
_sp_h = float(_spi["supportRadius"])
_orig_ic = hydrostaticColumn.hydrostaticColumnCase.initialConditions

def _patched_ic(ctx, system):
    _orig_ic(ctx, system)
    st = system.state
    fluid = st.kinds == 0
    if int(fluid.sum()) != _sp_pos.shape[0]:
        raise SystemExit(f"count mismatch {_int if False else int(fluid.sum())} vs {_sp_pos.shape[0]}")
    dev, dt = st.positions.device, st.positions.dtype
    P = st.positions.clone()
    P[fluid, 0] = _sp_pos[:, 0].to(dev, dt)
    P[fluid, 1] = _sp_pos[:, 1].to(dev, dt)
    if P.shape[1] > 2:
        P[fluid, 2] = 0.0
    st.positions = P
    st.velocities = torch.zeros_like(st.velocities)
    M = st.masses.clone(); M[fluid] = _sp_mass; st.masses = M
    st.supports = torch.full_like(st.supports, _sp_h)
    if st.pressures is not None:
        st.pressures = torch.zeros_like(st.pressures)
    ctx.scratch["initialPositions"] = st.positions.clone()

_imported_case = dataclasses.replace(hydrostaticColumn.hydrostaticColumnCase,
                                     initialConditions=_patched_ic)

# ---- the run matrix --------------------------------------------------------
JOBS = []
for s in SCHEMES:
    JOBS.append(dict(tag=f"native_{s}", case=hydrostaticColumn.hydrostaticColumnCase,
                     scheme=s, nx=64, nsteps=500, kernel="Wendland2", every=4))
for s in SCHEMES:
    JOBS.append(dict(tag=f"importSP_{s}", case=_imported_case, scheme=s,
                     nx=128, nsteps=(650 if s == "iisph" else 400),
                     kernel="CubicSpline", every=5))

only = sys.argv[1:] if len(sys.argv) > 1 else None
for job in JOBS:
    if only and job["tag"] not in only:
        continue
    dest = os.path.join(OUT, job["tag"])
    try:
        kw = dict(nx=job["nx"], nSteps=job["nsteps"], scheme=job["scheme"],
                  kernel=job["kernel"], quiet=True, plot=True, video=True, store=False,
                  progress=False, plotBackend="matplotlib", plotInterval=job["every"],
                  integrationScheme="semiImplicitEuler")
        r = run(job["case"], **kw)
        rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
        vm = [x.get("maxVelocity", float("nan")) for x in rows]
        vm = [v for v in vm if v == v]
        if r.videoPath and os.path.exists(r.videoPath):
            os.makedirs(dest, exist_ok=True)
            for f in ("output.mp4", "out.gif"):
                src = os.path.join(os.path.dirname(r.videoPath), f)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(dest, f))
        msg = (f"{job['tag']:28s} scheme={job['scheme']:16s} nx={job['nx']} "
               f"ran {len(rows)}/{job['nsteps']} diverged={r.diverged} "
               f"|v|max peak {max(vm):.4g} -> {dest}/output.mp4")
    except Exception as e:
        msg = f"{job['tag']:28s} FAILED: {e!r}\n{traceback.format_exc()}"
    print(msg, flush=True)
    LOG.write(msg + "\n")
