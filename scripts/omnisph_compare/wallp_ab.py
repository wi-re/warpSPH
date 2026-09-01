"""A/B the per-iterate wall-pressure closure (Part 41) on the schemes that
share `dfsphReference`'s step body -- `iisph` (landed baseline, holds
`hydrostaticColumn` nx=128 already but with a creeping slosh + startup dip,
Part 34) and `dfsphReference` (the two-solve DFSPH, does NOT hold it -- Part
37, diverges ~step 940).

    python scripts/omnisph_compare/wallp_ab.py [nsteps] [nx]

Arms: <scheme>:<wallPressureMode>[:df]  (df = also on the divergence solve).
"""
import sys

import numpy as np

import warpSPH.schemes.dfsphReference as ref
from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 128

ARMS = [
    ("iisph", None, False),
    ("iisph", "mls", False),
    ("dfsphReference", None, False),
    ("dfsphReference", "mls", False),
    # ("dfsphReference", "mls", True),  # diverges step ~110 -- wall pressure on
    # the DF (divergence) Jacobi detonates; omniSPH's divergenceSolve has none.
]


def g(row, k):
    return row.get(k, float("nan"))


def mean_tail(rows, k, frac=0.25):
    v = [g(x, k) for x in rows[int(len(rows) * (1 - frac)):]]
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float("nan")


hdr = (f'{"scheme":>16} {"wallP":>6} {"df":>3} {"ran":>9} {"div":>5} '
       f'{"|v|pk":>7} {"|v|end":>7} {"KEend":>9} {"embMin":>7} {"p05":>6} '
       f'{"slope":>7} {"presid":>7}')
print(hdr)
print("-" * len(hdr))
for scheme, wallP, onDiv in ARMS:
    ref.WALL_PRESSURE_MODE = wallP
    ref.WALL_PRESSURE_ON_DIVERGENCE = onDiv
    r = run(hydrostaticColumn.hydrostaticColumnCase, nx=NX, nSteps=NS,
            scheme=scheme, kernel="Wendland2", quiet=True, plot=False,
            store=False, progress=False, integrationScheme="semiImplicitEuler")
    rows = [x for x in r.trajectory if x.get("step", -1) >= 0]
    vm = [g(x, "maxVelocity") for x in rows if g(x, "maxVelocity") == g(x, "maxVelocity")]
    print(f'{scheme:>16} {str(wallP):>6} {str(onDiv):>3} {len(rows):>4d}/{NS:<4d} '
          f'{str(r.diverged):>5} {max(vm) if vm else float("nan"):>7.3g} '
          f'{vm[-1] if vm else float("nan"):>7.3g} '
          f'{mean_tail(rows, "kineticEnergy"):>9.3g} '
          f'{mean_tail(rows, "embeddedMinDensity"):>7.3f} '
          f'{mean_tail(rows, "densityP05"):>6.3f} '
          f'{mean_tail(rows, "pressureSlopeRatio"):>7.3f} '
          f'{mean_tail(rows, "pressureResidual"):>7.3f}')

ref.WALL_PRESSURE_MODE = None
ref.WALL_PRESSURE_ON_DIVERGENCE = False
