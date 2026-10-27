#!/usr/bin/env python
"""Taylor-Green vortex (2D, incompressible) -- thin wrapper around the runner.

The case lives in `warpSPH.cases.periodic.tgv`; the relaxation prepass, the TGV velocity
field and the diagnostics are its hooks, everything else is `warpSPH.runner`.
Equivalent invocations::

    python examples/incompressible/01-taylor-green-vortex.py --nx 256 --tLimit 2.0
    python -m warpSPH.cases.periodic.tgv --nx 256 --tLimit 2.0
    warpsph-run tgv --nx 256 --tLimit 2.0
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.periodic.tgv import tgvCase          # noqa: E402
from warpSPH.runner import caseMain            # noqa: E402

#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
PRESET = ['--plot']

if __name__ == '__main__':
    caseMain(tgvCase, PRESET + sys.argv[1:])
