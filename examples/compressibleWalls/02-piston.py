#!/usr/bin/env python
"""A piston driving into gas at rest (1D) -- COMPRESSIBLE_WALLS_PLAN.md case 2.

    python examples/compressibleWalls/02-piston.py                      # shock, u_p = 1
    python examples/compressibleWalls/02-piston.py --pistonSpeed -1     # withdrawing: rarefaction
    python examples/compressibleWalls/02-piston.py --scheme CRKSPH --supportMode KernelMeanSymmetric
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.piston import pistonCase   # noqa: E402
from warpSPH.runner import caseMain           # noqa: E402

PRESET = ['--plot', '--video']

if __name__ == '__main__':
    caseMain(pistonCase, PRESET + sys.argv[1:])
