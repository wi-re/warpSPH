"""Semi-periodic hydrostatic column: x wraps (no side walls -> no tangential
fluid<->boundary interaction), floor wall band in y, free surface on top.
Isolates the *vertical* stability of the fluid from any wall-slip question.

Usage: semiperiodic.py <scheme> <nsteps> [nx] [n_h] [kernel] [video]
"""
import sys, os, json, shutil, dataclasses
import numpy as np, torch

from warpSPHCore import DomainDescription
from warpSPH.cases import hydrostaticColumn
from warpSPH.cases.weaklyCompressible import buildRegionSystem, fluidRegion, boundaryRegion
from warpSPH.runner import run

SCHEME = sys.argv[1] if len(sys.argv) > 1 else "omniIncompressible"
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
NX     = int(sys.argv[3]) if len(sys.argv) > 3 else 64
N_H    = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
KERNEL = sys.argv[5] if len(sys.argv) > 5 else "Wendland2"
VIDEO  = len(sys.argv) > 6 and sys.argv[6] == "video"

BAND = 5

def columnSdf(ctx):
    # full width (x wraps), bottom-anchored, fillRatio deep
    from warpSPH.cases.weaklyCompressible import shapeSdf
    L = ctx.spec.L
    fill = ctx.param('fillRatio')
    hh = 0.5 * fill * L
    return shapeSdf('box', args=[[0.5 * L, hh]], offset=[0.0, -0.5 * L + hh])

def floorSdf(ctx):
    """Wall = everything below the interior floor y = -L/2 (x periodic, no ceiling)."""
    interior = ctx.scratch['spInterior']
    from warpSPH.regions import sampleDomainSDF
    return lambda x: sampleDomainSDF(x, interior, invert=False)

def configureScheme(ctx):
    hydrostaticColumn.hydrostaticColumnCase.configureScheme(ctx)   # gravity, solver, band, surf-det
    dx = ctx.config.dx
    L = ctx.spec.L
    dev, dt = ctx.device, ctx.config.domain.min.dtype
    xh = 0.5 * L
    yLo = -0.5 * L - BAND * dx          # floor band bottom
    yHi = 0.5 * L + 0.5 * L             # generous headroom for splash, no ceiling wall
    dom = DomainDescription(
        torch.tensor([-xh, yLo], device=dev, dtype=dt),
        torch.tensor([ xh, yHi], device=dev, dtype=dt),
        torch.tensor([True, False], device=dev), 2)
    interior = DomainDescription(                       # walls cut from OUTSIDE this
        torch.tensor([-xh, -0.5 * L], device=dev, dtype=dt),
        torch.tensor([ xh,  yHi + 10.0], device=dev, dtype=dt),   # ceiling at +inf
        torch.tensor([True, False], device=dev), 2)
    ctx.config.domain = dom
    ctx.scratch['spInterior'] = interior

def buildSystem(ctx):
    return buildRegionSystem(ctx, [fluidRegion(ctx, columnSdf(ctx)),
                                   boundaryRegion(ctx, floorSdf(ctx))])

case = dataclasses.replace(hydrostaticColumn.hydrostaticColumnCase,
                           name='hydrostaticColumnSemiPeriodic',
                           configureScheme=configureScheme,
                           buildSystem=buildSystem)

kw = dict(nx=NX, nSteps=NSTEPS, scheme=SCHEME, n_h=N_H, kernel=KERNEL,
          quiet=True, plot=VIDEO, video=VIDEO, store=False, progress=False,
          integrationScheme="semiImplicitEuler")
if VIDEO:
    kw.update(plotBackend="matplotlib", plotInterval=4)

r = run(case, **kw)
rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
n = len(rows)
def g(row, k): return row.get(k, float("nan"))
def mn(key, a, b):
    v = [g(x, key) for x in rows[a:b]]; v = [k for k in v if k == k]
    return sum(v) / len(v) if v else float("nan")
vm = [g(x, "maxVelocity") for x in rows]; vm = [v for v in vm if v == v]

print(f"\nsemiPeriodic {SCHEME} nx={NX} n_h={N_H} {KERNEL}  ran {n}/{NSTEPS} div={r.diverged}")
hdr = f"  {'step':>5} {'t':>7} {'|v|max':>9} {'KE':>9} {'yTop':>7} {'colH':>7} {'embMin':>7} {'maxRho':>7} {'slope':>7}"
print(hdr)
# column top / height from stored positions is not in trajectory; use dispMax + surface density proxies
for i in sorted(set(min(n-1, round(k*(n-1)/12)) for k in range(13))):
    row = rows[i]
    print(f"  {g(row,'step'):>5.0f} {g(row,'t'):>7.3f} {g(row,'maxVelocity'):>9.4g} "
          f"{g(row,'kineticEnergy'):>9.4g} {'-':>7} {'-':>7} "
          f"{g(row,'embeddedMinDensity'):>7.3f} {g(row,'maxDensity'):>7.3f} "
          f"{g(row,'pressureSlopeRatio'):>7.3f}")
print(f"  |v|max peak {max(vm):.4g} final {vm[-1]:.4g}   late KE {mn('kineticEnergy',3*n//4,n):.4f}   "
      f"late embMin {mn('embeddedMinDensity',3*n//4,n):.3f}   late slope {mn('pressureSlopeRatio',3*n//4,n):.3f}")

if VIDEO and r.videoPath:
    dest = f"/home/lu26029/dev/warpSPH/scripts/splishsplash_compare/videos/semiPeriodic_{SCHEME}_{KERNEL}_nh{N_H:g}"
    os.makedirs(dest, exist_ok=True)
    for f in ("output.mp4", "out.gif"):
        s = os.path.join(os.path.dirname(r.videoPath), f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(dest, f))
    print("video ->", dest)
