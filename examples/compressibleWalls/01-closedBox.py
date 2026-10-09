#!/usr/bin/env python
"""Closed box of quiescent gas between solid walls (1D) -- COMPRESSIBLE_WALLS_PLAN.md case 1.

    python examples/compressibleWalls/01-closedBox.py
    python examples/compressibleWalls/01-closedBox.py --scheme CRKSPH
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.closedBox import closedBoxCase   # noqa: E402
from warpSPH.runner import caseMain                 # noqa: E402

PRESET = ['--plot', '--video']

if __name__ == '__main__':
    caseMain(closedBoxCase, PRESET + sys.argv[1:])
