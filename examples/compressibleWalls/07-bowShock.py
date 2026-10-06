#!/usr/bin/env python
"""A cylinder at Mach 3 through gas at rest, bow shock standoff (2D) -- COMPRESSIBLE_WALLS_PLAN.md case 7.

    python examples/compressibleWalls/07-bowShock.py
    python examples/compressibleWalls/07-bowShock.py --scheme CRKSPH --supportMode KernelMeanSymmetric
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.bowShock import bowShockCase   # noqa: E402
from warpSPH.runner import caseMain                         # noqa: E402

PRESET = ['--plot', '--video', '--progress']

if __name__ == '__main__':
    caseMain(bowShockCase, PRESET + sys.argv[1:])
