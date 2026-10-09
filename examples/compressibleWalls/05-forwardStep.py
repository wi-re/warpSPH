#!/usr/bin/env python
"""Mach 3 forward-facing step, in the gas frame (2D) -- COMPRESSIBLE_WALLS_PLAN.md case 5.

    python examples/compressibleWalls/05-forwardStep.py
    python examples/compressibleWalls/05-forwardStep.py --scheme CRKSPH --supportMode KernelMeanSymmetric
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.forwardStep import forwardStepCase   # noqa: E402
from warpSPH.runner import caseMain                         # noqa: E402

PRESET = ['--plot', '--video', '--progress']

if __name__ == '__main__':
    caseMain(forwardStepCase, PRESET + sys.argv[1:])
