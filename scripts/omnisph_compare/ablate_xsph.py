"""omniSPH column A/B: does its clean hydrostatic hold survive turning off
XSPH (`ptcl.viscosityConstant`) and the wall no-slip BXSPH
(`ptcl.boundaryViscosity`)?

omniSPH's timestep ALWAYS runs XSPH() + BXSPH(); initializeParameters sets
viscosityConstant = 0.01 and boundaryViscosity = 0.50 -- so the shipped
"DFSPH incompressible" loop is NOT inviscid. This isolates how load-bearing
those two filters are.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dev/omniSPH/omnySPH/src"))
import omnySPH  # noqa: E402

CFG = os.path.join(os.path.dirname(__file__), "column.yaml")
NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1200

ARMS = [
    ("bv=0.01 (xsph=0.01)", 0.01, 0.01),
    ("bv=0.05 (xsph=0.01)", 0.01, 0.05),
    ("default (xsph=0.01, bxsph=0.50)", 0.01, 0.50),
    ("no wall no-slip (bxsph=0)",       0.01, 0.0),
    ("no XSPH (xsph=0)",                0.0,  0.50),
    ("fully inviscid (both 0)",         0.0,  0.0),
]

for name, vc, bv in ARMS:
    sim = omnySPH.SPHSimulation(open(CFG).read())
    sim.setScalar("ptcl.viscosityConstant", vc)
    sim.setScalar("ptcl.boundaryViscosity", bv)
    n = sim.getInteger("props.numPtcls")
    kes, vmaxs, vmeans, rmaxs = [], [], [], []
    for k in range(NSTEPS):
        sim.timestep()
        V = np.asarray(sim.fluidVelocity)[:n]
        D = np.asarray(sim.fluidDensity)[:n]
        sp = np.linalg.norm(V, axis=1)
        kes.append(0.5 * (sp ** 2).sum() / n)
        vmaxs.append(sp.max())
        vmeans.append(sp.mean())
        rmaxs.append(D.max())
        if not np.isfinite(sp.max()):
            break
    q = max(1, len(kes) // 4)
    print(f"{name:34s}  steps={len(kes):4d}  "
          f"KE/n q1={np.mean(kes[:q]):.2e}->q4={np.mean(kes[-q:]):.2e}  "
          f"vmax last~{np.mean(vmaxs[-q:]):.3f}  "
          f"vmean last~{np.mean(vmeans[-q:]):.4f}  "
          f"rmax last~{np.mean(rmaxs[-q:]):.4f}")
