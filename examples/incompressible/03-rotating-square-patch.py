#!/usr/bin/env python
"""03 rotating square patch -- thin wrapper around the shared runner.

The notebook this came from is `03-rotating-square-patch.ipynb`; the case itself is
`warpSPH.cases.freeSurface.rotatingSquarePatch`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/incompressible/03-rotating-square-patch.py --scheme divergenceFree
    warpsph-run squarePatch --scheme divergenceFree            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.freeSurface.rotatingSquarePatch import rotatingSquarePatchCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
PRESET = ['--plot', '--scheme', 'divergenceFree']

if __name__ == '__main__':
    caseMain(rotatingSquarePatchCase, PRESET + sys.argv[1:])
