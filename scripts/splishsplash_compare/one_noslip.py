import sys
import warpSPH.schemes.omniIncompressible as m
m.XSPH_FLUID = float(sys.argv[1])
m.XSPH_BOUNDARY = float(sys.argv[2])
NX = int(sys.argv[3]) if len(sys.argv) > 3 else 64
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run

r = run(hydrostaticColumn.hydrostaticColumnCase, nx=NX, nSteps=NS,
        scheme="omniIncompressible", quiet=True, plot=False, store=False,
        progress=False, integrationScheme="semiImplicitEuler")
rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
n = len(rows)
def g(rw, k): return rw.get(k, float("nan"))
def mn(k, a, b):
    v = [g(x, k) for x in rows[a:b]]; v = [q for q in v if q == q]
    return sum(v) / len(v) if v else float("nan")
vm = [g(x, "maxVelocity") for x in rows]; vm = [v for v in vm if v == v]
print(f"XSPH fluid={sys.argv[1]} bdy={sys.argv[2]}  nx={NX} ran {n}/{NS} div={r.diverged}  "
      f"|v|peak {max(vm):.4g} final {vm[-1]:.3g}  lateKE {mn('kineticEnergy',3*n//4,n):.4f}  "
      f"lateEmbMin {mn('embeddedMinDensity',3*n//4,n):.3f}  "
      f"lateSlope {mn('pressureSlopeRatio',3*n//4,n):.3f}  lateDisp {mn('dispMax',3*n//4,n):.3f}")
