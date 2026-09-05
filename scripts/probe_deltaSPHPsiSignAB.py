#!/usr/bin/env python3
"""Re-run a case with the PRE-2026-09-05 delta-SPH `psi` sign, for a clean A/B.

`modules/deltaSPH/wp_densityDelta.py` computed `psi = +grad_ij - rho_ij` until
the sign fix; it now computes `-grad_ij - rho_ij` (Marrone et al. 2011 Eq. 6).
Negating the gradient field handed to the kernel reproduces the old operator
**bit for bit**, with no kernel edit and no rebuild -- so this measures the sign
alone, on today's code, with every other change held fixed. Compare its
`<scheme>_series.npz` against a normal run's.

Recorded result (`sloshingTank --scheme wcsph`, nx=100, t <= 3.6 s), see
ACSPH_PLAN.md Part 3 "The sign error":

                        pre-fix psi     post-fix psi
    diverged                 no              no
    min density            0.396          0.532
    peak |p| (t > 2s)     39.2 kPa       32.5 kPa      (measured band 2.2-13.1)

    python scripts/probe_deltaSPHPsiSignAB.py --out /tmp/psiSignAB
    python scripts/probe_deltaSPHPsiSignAB.py --tLimit 1.0 --out /tmp/psiSignAB
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, 'examples', 'sloshingTank', 'run_sloshingTank.py')

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--scheme', default='wcsph')
parser.add_argument('--tLimit', type=float, default=3.6)
parser.add_argument('--nx', type=int, default=None)
parser.add_argument('--out', default=None, help='output directory (default: the runner\'s)')
args = parser.parse_args()

sys.argv = [RUNNER, '--scheme', args.scheme, '--tLimit', str(args.tLimit)]
if args.nx is not None:
    sys.argv += ['--nx', str(args.nx)]
if args.out is not None:
    os.makedirs(args.out, exist_ok=True)
    sys.argv += ['--out', args.out]

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

import warpSPH.schemes.deltaSPH as deltaSPHScheme

_realGradRhoL = deltaSPHScheme.computeGradRhoL


def _negatedGradRhoL(*args, **kwargs):
    gradient = _realGradRhoL(*args, **kwargs)
    return None if gradient is None else -gradient


deltaSPHScheme.computeGradRhoL = _negatedGradRhoL
print("[A/B] psi gradient term NEGATED -- reproducing the pre-2026-09-05 operator")

namespace = {'__name__': '__main__', '__file__': RUNNER}
exec(compile(open(RUNNER).read(), RUNNER, 'exec'), namespace)
