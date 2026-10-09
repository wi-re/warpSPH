#!/usr/bin/env python
"""Sedov-Taylor blast in a box with solid walls (2D by default).

    python examples/compressibleWalls/sedovWalls.py                    # CRKSPH, nx 101, until r_free = 1.4
    python examples/compressibleWalls/sedovWalls.py --scheme Monaghan --supportMode Gather
    python examples/compressibleWalls/sedovWalls.py --dim 1 --nx 801

Compare with the periodic (mirror-symmetric) reference: scripts/compare_wallMirror.py --case sedov.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.sedovWalls import sedovWallsCase   # noqa: E402
from warpSPH.runner import caseMain                   # noqa: E402

PRESET = ['--plot', '--video', '--progress']

if __name__ == '__main__':
    caseMain(sedovWallsCase, PRESET + sys.argv[1:])
