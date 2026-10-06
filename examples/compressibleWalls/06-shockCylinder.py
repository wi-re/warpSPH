#!/usr/bin/env python
"""A Mach-2 shock hitting a fixed cylinder in a channel (2D) -- COMPRESSIBLE_WALLS_PLAN.md case 6.

    python examples/compressibleWalls/06-shockCylinder.py
    python examples/compressibleWalls/06-shockCylinder.py --scheme CRKSPH --supportMode KernelMeanSymmetric
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.shockCylinder import shockCylinderCase   # noqa: E402
from warpSPH.runner import caseMain                         # noqa: E402

PRESET = ['--plot', '--video', '--progress']

if __name__ == '__main__':
    caseMain(shockCylinderCase, PRESET + sys.argv[1:])
