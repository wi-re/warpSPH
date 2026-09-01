"""Import SPlisHSPlasH's EXACT fluid initial state into warpSPH's
hydrostaticColumn and run -- the controlled 'match the physics + init' test.

Overwrites the fluid rows with SPlisHSPlasH's particle positions and mass,
sets every support radius to SPlisHSPlasH's (h = 2*dx), the kernel to a cubic
spline, zeroes velocity and the IC pressure seed. The boundary stays warpSPH's
5-layer band (SPlisHSPlasH used a volume-map / Akinci wall with no importable
particles) but is given the same h. Then runs the requested scheme.

Usage: import_and_run.py <scheme> <nsteps> [kernel] [n_h_fallback]
"""
import sys, json, dataclasses
import numpy as np
import torch

from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run

SCHEME = sys.argv[1] if len(sys.argv) > 1 else "dfsphReference"
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
KERNEL = sys.argv[3] if len(sys.argv) > 3 else "CubicSpline"

sp = np.load("/home/lu26029/dev/warpSPH/scripts/splishsplash_compare/splish_fluid8001.npz",
             allow_pickle=True)
spi = json.loads(str(sp["info"]))
sp_pos = torch.tensor(sp["pos"], dtype=torch.float64)         # (N,2)
sp_mass = float(spi["massMean"])
sp_h = float(spi["supportRadius"])
print(f"SP import: {sp_pos.shape[0]} fluid, mass {sp_mass:.4e}, h {sp_h:.6f} "
      f"(= {sp_h/(1/128):.2f} dx), kernel cubic, rho0 {spi['density0']}")

_orig_ic = hydrostaticColumn.hydrostaticColumnCase.initialConditions

def patched_ic(ctx, system):
    _orig_ic(ctx, system)
    st = system.state
    fluid = (st.kinds == 0)
    nF = int(fluid.sum())
    if nF != sp_pos.shape[0]:
        raise SystemExit(f"count mismatch: warpSPH {nF} fluid vs SP {sp_pos.shape[0]}")
    dev, dt = st.positions.device, st.positions.dtype
    P = st.positions.clone()
    P[fluid, 0] = sp_pos[:, 0].to(dev, dt)
    P[fluid, 1] = sp_pos[:, 1].to(dev, dt)
    if P.shape[1] > 2:
        P[fluid, 2] = 0.0
    st.positions = P
    st.velocities = torch.zeros_like(st.velocities)
    M = st.masses.clone(); M[fluid] = sp_mass; st.masses = M
    st.supports = torch.full_like(st.supports, sp_h)          # fluid AND wall
    if st.pressures is not None:
        st.pressures = torch.zeros_like(st.pressures)
    ctx.scratch['initialPositions'] = st.positions.clone()

case = dataclasses.replace(hydrostaticColumn.hydrostaticColumnCase,
                           initialConditions=patched_ic)

r = run(case, nx=128, nSteps=NSTEPS, scheme=SCHEME, kernel=KERNEL,
        quiet=True, plot=False, store=False, progress=False,
        integrationScheme="semiImplicitEuler")

rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
n = len(rows)
def g(row, k): return row.get(k, float("nan"))
def mn(key, a, b):
    v = [g(x, key) for x in rows[a:b]]; v = [k for k in v if k == k]
    return sum(v) / len(v) if v else float("nan")
vm = [g(x, "maxVelocity") for x in rows]; vm = [v for v in vm if v == v]
print(f"\n{SCHEME} nx=128 kernel={KERNEL}  IMPORTED SP fluid state  "
      f"ran {n}/{NSTEPS}  diverged={r.diverged}")
step_idx = sorted(set(min(n - 1, round(k * (n - 1) / 12)) for k in range(13))) if n else []
print(f"  {'step':>5} {'t':>7} {'|v|max':>10} {'KE':>10} {'embMin':>8} {'maxRho':>8} {'slope':>8}")
for i in step_idx:
    row = rows[i]
    print(f"  {g(row,'step'):>5.0f} {g(row,'t'):>7.3f} {g(row,'maxVelocity'):>10.4g} "
          f"{g(row,'kineticEnergy'):>10.4g} {g(row,'embeddedMinDensity'):>8.3f} "
          f"{g(row,'maxDensity'):>8.3f} {g(row,'pressureSlopeRatio'):>8.3f}")
if vm:
    print(f"  |v|max peak {max(vm):.4g}  final {vm[-1]:.4g}   "
          f"late KE {mn('kineticEnergy',3*n//4,n):.4f}  late embMin {mn('embeddedMinDensity',3*n//4,n):.3f}")
