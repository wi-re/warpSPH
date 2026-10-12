#!/usr/bin/env python
"""Decaying random flow in a walled box (2D) -- thin wrapper around the runner.

The case itself is `warpSPH.cases.periodic.randomFlow`, and everything generic (config,
step loop, export, plotting, ffmpeg) is `warpSPH.runner`. Equivalent
invocations::

    python examples/weaklyCompressible/06-randomFlow/randomFlow_bounded.py
    warpsph-run randomFlow --bounded --nx 256      # also honours --precision

The other shipped point of this case is `randomFlow_periodic.py`: the same
divergence-free noise field, periodic on every side and with a rigid body in
it. Here the box has free-slip walls instead, so the decay includes whatever
the boundary contributes -- which is the comparison the pair exists for.

`--bounded` widens the simulated domain by `--band` particle layers and cuts the
walls out of the interior; the case supplies `band=5` when it is left at the
periodic default of 0, because a zero-width wall region samples no boundary
particles at all. `randomFlow_bounded.ipynb` documents every knob.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.periodic.randomFlow import randomFlowCase    # noqa: E402
from warpSPH.runner import caseMain                    # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
#:
#: `--no-inviscid --nu 0.005` is the viscosity the notebook this came from ran
#: with -- half the periodic variant's, at twice its resolution.
PRESET = ['--plot', '--bounded', '--nx', '256',
          '--no-inviscid', '--nu', '0.005',
          '--caseName', '06-randomFlowBounded']

if __name__ == '__main__':
    caseMain(randomFlowCase, PRESET + sys.argv[1:])
