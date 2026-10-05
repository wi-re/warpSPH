#!/usr/bin/env python
"""A Mach-2 shock reflecting off a solid wall (1D) -- COMPRESSIBLE_WALLS_PLAN.md case 3.

    python examples/compressibleWalls/03-shockReflection.py
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.shockReflection import shockReflectionCase   # noqa: E402
from warpSPH.runner import caseMain                             # noqa: E402

PRESET = ['--plot', '--video']

if __name__ == '__main__':
    caseMain(shockReflectionCase, PRESET + sys.argv[1:])
