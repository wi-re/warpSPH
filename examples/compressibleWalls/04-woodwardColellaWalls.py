#!/usr/bin/env python
"""Woodward-Colella blast waves between solid walls (1D) -- COMPRESSIBLE_WALLS_PLAN.md case 4.

    python examples/compressibleWalls/04-woodwardColellaWalls.py
    python examples/compressibleWalls/04-woodwardColellaWalls.py --scheme Monaghan --supportMode Gather

Compare with the mirror-symmetric periodic reference: scripts/compare_wallMirror.py.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.woodwardColellaWalls import woodwardColellaWallsCase   # noqa: E402
from warpSPH.runner import caseMain                                       # noqa: E402

PRESET = ['--plot', '--video']

if __name__ == '__main__':
    caseMain(woodwardColellaWallsCase, PRESET + sys.argv[1:])
