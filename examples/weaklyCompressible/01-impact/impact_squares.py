#!/usr/bin/env python
"""Impact of two fluid squares (2D) -- thin wrapper around the shared runner.

The same case as `impact_spheres.py` with the bodies changed: 2.0 x 1.0
rectangles colliding along y, starting one particle spacing apart rather than
at a fixed separation (`--touching --gap 1`, which measures the shape instead
of hard-coding `H/2 + dx` the way the original notebook did). Equivalent
invocations::

    python examples/weaklyCompressible/01-impact/impact_squares.py
    warpsph-run impact --shape box --size 1.0 --aspectRatio 0.5 \\
        --impactAxis 1 --touching --nx 128 --tLimit 1.5

See `impact_spheres.py` for the rest of the family this case covers, and
`impact_squares.ipynb` for what every flag does.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.freeSurface.impact import impactCase    # noqa: E402
from warpSPH.runner import caseMain            # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
PRESET = ['--plot', '--shape', 'box', '--size', '1.0', '--aspectRatio', '0.5',
          '--impactAxis', '1', '--touching', '--gap', '1.0',
          '--nx', '128', '--tLimit', '1.5', '--caseName', '01-impactSquares']

if __name__ == '__main__':
    caseMain(impactCase, PRESET + sys.argv[1:])
