#!/usr/bin/env python
"""Periodic random flow -- thin wrapper around the shared runner.

The notebook this came from is `periodic-random-flow.ipynb`; the case itself is
`warpSPH.cases.periodic.randomFlow`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/incompressible/periodic-random-flow.py --scheme divergenceFree --L 6.283185307179586
    warpsph-run randomFlow --scheme divergenceFree --L 6.283185307179586            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.periodic.randomFlow import randomFlowCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
PRESET = ['--plot', '--scheme', 'divergenceFree', '--L', '6.283185307179586']

if __name__ == '__main__':
    caseMain(randomFlowCase, PRESET + sys.argv[1:])
