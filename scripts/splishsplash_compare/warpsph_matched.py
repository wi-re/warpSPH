"""Run warpSPH hydrostaticColumn with the SPlisHSPlasH discretisation
(support h = 2*dx, cubic spline, lattice calibrated to rho0) and report the
|v|max / density trajectory to compare against the SPlisHSPlasH DFSPH run
(peak |v|max ~2.4 at t~0.11, decaying to ~0.66 by t~0.3, column intact)."""
import sys, numpy as np
from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run

SCHEME = sys.argv[1] if len(sys.argv) > 1 else "iisph"
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 128
NSTEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
N_H = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
KERNEL = sys.argv[5] if len(sys.argv) > 5 else "CubicSpline"
CALIB = (sys.argv[6] if len(sys.argv) > 6 else "1") == "1"

r = run(hydrostaticColumn.hydrostaticColumnCase, nx=NX, nSteps=NSTEPS,
        scheme=SCHEME, n_h=N_H, kernel=KERNEL,
        params={"calibrateRestDensity": CALIB},
        quiet=True, plot=False, store=False, progress=False,
        integrationScheme="semiImplicitEuler")

rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
n = len(rows)
def g(row, k): return row.get(k, float("nan"))
print(f"scheme={SCHEME} nx={NX} n_h={N_H} kernel={KERNEL} calibrate={CALIB}  "
      f"ran {n}/{NSTEPS}  diverged={r.diverged}")
print(f"  {'step':>5} {'t':>7} {'|v|max':>9} {'KE':>10} {'minRho':>8} {'rhoP05':>8} "
      f"{'embMin':>8} {'maxRho':>8} {'slope':>8} {'dispMax':>8}")
idx = sorted(set(min(n - 1, round(k * (n - 1) / 15)) for k in range(16))) if n else []
for i in idx:
    row = rows[i]
    print(f"  {row.get('step', i+1):>5} {g(row,'t'):>7.3f} {g(row,'maxVelocity'):>9.4g} "
          f"{g(row,'kineticEnergy'):>10.4g} {g(row,'minDensity'):>8.4f} "
          f"{g(row,'densityP05'):>8.4f} {g(row,'embeddedMinDensity'):>8.4f} "
          f"{g(row,'maxDensity'):>8.4f} {g(row,'pressureSlopeRatio'):>8.3f} "
          f"{g(row,'dispMax'):>8.3f}")
vmax = [g(x, "maxVelocity") for x in rows]
vmax = [v for v in vmax if v == v]
if vmax:
    print(f"  |v|max: peak {max(vmax):.4g}  final {vmax[-1]:.4g}")
