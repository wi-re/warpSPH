#!/usr/bin/env python
"""Where does the relaxed-Jacobi limit sit? A still analytic-wall tank (`sloshingTank`, zero roll) under `omniIncompressible` at relaxation factors OMEGA in {0.3, 0.4, 0.5} and supports n_h in {2.57, 4},
with the divergence stage as omniSPH's 3 Jacobi sweeps and as the converged compact projection: the divergence stage's share of the instability is what the compact projection removes, the rest is the
constant-density solve (clamped, still Jacobi). Prints peak and late max |v| over 150 steps.
"""
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from warpSPHBootstrap import bootstrap  # noqa: E402

ROLL = os.path.join(REPO, 'examples', 'sloshingTank', 'SPHERIC_TestCase10', 'data_files', 'lateral_water_1x.txt')


def main():
    bootstrap(precision='float32')
    warnings.simplefilter('ignore')
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.caseSpec import CaseSpec
    import warpSPH.schemes.omniIncompressible as O
    importAll()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    default = O.OMEGA
    for nh in (2.57, 4.0):
        for proj in ('jacobi', 'compact'):
            for omega in (0.3, 0.4, 0.5):
                O.OMEGA = omega
                case = getCase('sloshingTank')
                spec = CaseSpec(caseName='omega', scheme='omniIncompressible', params={**case.params, 'wallRepresentation': 'analytic', 'rollDataFile': ROLL, 'rollStartTime': 100.0, 'projection': proj}).merged(**case.defaults).merged(
                    scheme='omniIncompressible', integrationScheme='semiImplicitEuler', kernel='Wendland2', supportMode='SuperSymmetric', cflFactor=0.2, dt=1e-3, maxDt=2e-3, nx=100, n_h=nh, nSteps=n,
                    plot=False, store=False, progress=False, video=False, show=False, quiet=True)
                try:
                    res = run(case, spec)
                    v = np.asarray(res.series('maxVelocity'))
                    print(f'n_h {nh}  divergence stage {proj:8s} omega {omega}: peak |v| {np.nanmax(v):7.3f}   last 30 steps {np.nanmax(v[-30:]):7.3f}   diverged {res.diverged}', flush=True)
                except Exception as e:           # noqa: BLE001
                    print(f'n_h {nh}  divergence stage {proj:8s} omega {omega}: FAILED {type(e).__name__}: {e}', flush=True)
    O.OMEGA = default


if __name__ == '__main__':
    main()
