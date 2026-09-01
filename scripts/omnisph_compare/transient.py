"""Transient A/B for ranked-queue item 0 (Part 41+): step warpSPH
`omniIncompressible` and omniSPH on the hydrostatic column, logging per-step /
per-solve diagnostics, to locate the step and the solve where warpSPH's
composed Jacobi loses it -- and whether it is an interior or a wall failure.

Three arms:
  * warp  walled       -- the failing nx=128 four-walled column
  * warp  semiperiodic -- x wraps, floor-only (Part 38): isolates the interior
  * omniSPH            -- the reference (analytic triangle walls)

    python scripts/omnisph_compare/transient.py [nsteps] [nx]
"""
import contextlib
import dataclasses
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 128


@contextlib.contextmanager
def muffle():
    sys.stdout.flush()
    saved = os.dup(1)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.close(dn)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


import warpSPH.schemes.omniIncompressible as omod          # noqa: E402
from warpSPHCore import DomainDescription                  # noqa: E402
from warpSPH.cases import hydrostaticColumn                 # noqa: E402
from warpSPH.cases.weaklyCompressible import (buildRegionSystem, fluidRegion,  # noqa: E402
                                              boundaryRegion, shapeSdf)
from warpSPH.runner import run                              # noqa: E402

_orig_solve = omod._solve
_orig_step = omod.omniIncompressible_step
_LOG = {"solve": [], "step": [], "i": 0}


def _solve_spy(*a, **k):
    a_p, p, nit, err = _orig_solve(*a, **k)
    mode = k.get("mode")
    if mode is None and len(a) > 8:
        mode = a[8]
    _LOG["solve"].append((_LOG["i"], str(mode), nit, float(err),
                          float(p.abs().max()), float(a_p.abs().max())))
    return a_p, p, nit, err


def _step_spy(system, dt, config, schemeConfig, verbose=False):
    _LOG["i"] += 1
    out = _orig_step(system, dt, config, schemeConfig, verbose=False)
    st = system.state
    fl = st.kinds == 0
    v = st.velocities[fl].norm(dim=-1)
    fp = st.positions[fl]
    p = st.pressures[fl] if st.pressures is not None else torch.zeros_like(v)
    vi, pi = int(v.argmax()), int(p.abs().argmax())
    rho = st.densities[fl]
    _LOG["step"].append(dict(
        step=_LOG["i"], vmax=float(v.max()), vy=float(fp[vi, 1]), vx=float(fp[vi, 0]),
        pmax=float(p.abs().max()), py=float(fp[pi, 1]), px=float(fp[pi, 0]),
        rmin=float(rho.min()), rmax=float(rho.max())))
    return out


omod._solve = _solve_spy
omod.omniIncompressible_step = _step_spy
import warpSPH.schemes.builder as _b                        # noqa: E402
if hasattr(_b, "omniIncompressible_step"):
    _b.omniIncompressible_step = _step_spy


# ---- semi-periodic case (from splishsplash_compare/semiperiodic.py) --------
BAND = 5


def _sp_configure(ctx):
    hydrostaticColumn.hydrostaticColumnCase.configureScheme(ctx)
    dx, L = ctx.config.dx, ctx.spec.L
    dev, dt = ctx.device, ctx.config.domain.min.dtype
    xh = 0.5 * L
    dom = DomainDescription(
        torch.tensor([-xh, -0.5 * L - BAND * dx], device=dev, dtype=dt),
        torch.tensor([xh, 1.0 * L], device=dev, dtype=dt),
        torch.tensor([True, False], device=dev), 2)
    interior = DomainDescription(
        torch.tensor([-xh, -0.5 * L], device=dev, dtype=dt),
        torch.tensor([xh, 1.0 * L + 10.0], device=dev, dtype=dt),
        torch.tensor([True, False], device=dev), 2)
    ctx.config.domain = dom
    ctx.scratch['spInterior'] = interior


def _sp_build(ctx):
    from warpSPH.regions import sampleDomainSDF
    fill = ctx.param('fillRatio')
    hh = 0.5 * fill * ctx.spec.L
    csdf = shapeSdf('box', args=[[0.5 * ctx.spec.L, hh]], offset=[0.0, -0.5 * ctx.spec.L + hh])
    fsdf = lambda x: sampleDomainSDF(x, ctx.scratch['spInterior'], invert=False)
    return buildRegionSystem(ctx, [fluidRegion(ctx, csdf), boundaryRegion(ctx, fsdf)])


_sp_case = dataclasses.replace(hydrostaticColumn.hydrostaticColumnCase,
                               name='hydrostaticColumnSemiPeriodic',
                               configureScheme=_sp_configure, buildSystem=_sp_build)


# ---- run one warp arm ----------------------------------------------------
def warp_arm(name, case, floor_y, wallP=None):
    omod.WALL_PRESSURE_MODE = wallP
    name = name + (f"  [+wallP:{wallP}]" if wallP else "")
    _LOG["solve"].clear(); _LOG["step"].clear(); _LOG["i"] = 0
    surf_y = floor_y + 0.5      # ~ fillRatio*L above the floor
    print(f"\n=== warpSPH omniIncompressible  {name}  nx={NX}  {NSTEPS} steps ===")
    r = run(case, nx=NX, nSteps=NSTEPS, scheme="omniIncompressible",
            kernel="Wendland2", quiet=True, plot=False, store=False,
            progress=False, integrationScheme="semiImplicitEuler")

    def loc(y):
        if y < floor_y + 0.04:
            return "FLOOR"
        if y > surf_y - 0.06:
            return "surf"
        return "bulk"

    byv = {}
    for (s, mode, nit, err, mp, mA) in _LOG["solve"]:
        byv.setdefault(s, {})[mode] = (nit, err, mp, mA)
    h = f'{"step":>5} {"nRho":>5} {"errRho":>10} {"nDiv":>4} {"errDiv":>10} ' \
        f'{"maxP":>10} {"maxAp":>10} {"vmax":>9} {"vloc":>6} {"rmin":>6} {"rmax":>6}'
    print(h)
    print("-" * len(h))
    for row in _LOG["step"]:
        s = row["step"]
        d = byv.get(s, {})
        rh = d.get("density", (0, np.nan, np.nan, np.nan))
        dv = d.get("divergence", (0, np.nan, np.nan, np.nan))
        show = s <= 12 or s % max(1, NSTEPS // 25) == 0
        if show or not np.isfinite(row["vmax"]) or row["vmax"] > 1e3:
            print(f'{s:5d} {rh[0]:5d} {rh[1]:10.3e} {dv[0]:4d} {dv[1]:10.3e} '
                  f'{rh[2]:10.3e} {rh[3]:10.3e} {row["vmax"]:9.3g} '
                  f'{loc(row["vy"]):>6} {row["rmin"]:6.3f} {row["rmax"]:6.3f}')
        if not np.isfinite(row["vmax"]) or row["vmax"] > 1e4:
            print(f'  -> DIVERGED step {s}: |v|max ptcl (x={row["vx"]:.3f},'
                  f'y={row["vy"]:.3f})[{loc(row["vy"])}]  maxP ptcl '
                  f'(x={row["px"]:.3f},y={row["py"]:.3f})[{loc(row["py"])}]')
            break
    fin = _LOG["step"][-1]
    nrho = [d.get("density", (0,))[0] for d in byv.values()]
    print(f'  ran {len(_LOG["step"])}/{NSTEPS}  diverged={r.diverged}  '
          f'nRho: max {max(nrho) if nrho else 0}  '
          f'mean {np.mean(nrho) if nrho else 0:.1f}  (cap {omod.DENSITY_MAX_ITERATIONS})')
    return r


warp_arm("walled", hydrostaticColumn.hydrostaticColumnCase, floor_y=-0.5, wallP=None)
warp_arm("walled", hydrostaticColumn.hydrostaticColumnCase, floor_y=-0.5, wallP="shepard")
warp_arm("walled", hydrostaticColumn.hydrostaticColumnCase, floor_y=-0.5, wallP="mls")

# ============================================================ omniSPH
sys.path.insert(0, os.path.expanduser("~/dev/omniSPH/omnySPH/src"))
import omnySPH  # noqa: E402

print(f"\n=== omniSPH  column.yaml  nx~128  {NSTEPS} steps ===")
with muffle():
    sim = omnySPH.SPHSimulation(open(os.path.join(HERE, "column.yaml")).read())
n = sim.getInteger("props.numPtcls")
h = f'{"step":>5} {"nDiv":>5} {"nRho":>5} {"errDiv":>10} {"errRho":>10} {"vmax":>9} {"rmin":>7} {"rmax":>7}'
print(h)
print("-" * len(h))
for k in range(NSTEPS + 1):
    if k:
        with muffle():
            sim.timestep()
    if k <= 12 or k % max(1, NSTEPS // 25) == 0:
        V = np.asarray(sim.fluidVelocity)[:n]
        D = np.asarray(sim.fluidDensity)[:n]
        sp = np.linalg.norm(V, axis=1)
        print(f'{k:5d} {sim.getInteger("dfsph.divergenceIterations"):5d} '
              f'{sim.getInteger("dfsph.densityIterations"):5d} '
              f'{sim.getScalar("dfsph.divergenceError"):10.3e} '
              f'{sim.getScalar("dfsph.densityError"):10.3e} {sp.max():9.3g} '
              f'{D.min():7.4f} {D.max():7.4f}')
        if not np.isfinite(sp.max()):
            print("  omniSPH DIVERGED")
            break
print("\ndone.")
