#!/usr/bin/env python
"""Decaying random flow, periodic box with an obstacle (2D) -- thin wrapper.

The case itself is `warpSPH.cases.periodic.randomFlow`, and everything generic (config,
step loop, export, plotting, ffmpeg) is `warpSPH.runner`. Equivalent
invocations::

    python examples/weaklyCompressible/06-randomFlow/randomFlow_periodic.py
    warpsph-run randomFlow --no-bounded --obstacle --nx 128   # also honours --precision

`randomFlow` is one case with two shipped points: this one is periodic on every
side with a rigid body in the middle, `randomFlow_bounded.py` is the same noise
field in a walled box. The initial field is divergence-free by construction, so
the run measures how the scheme *decays* a valid incompressible field rather
than how it recovers from a bad start; `--octaves`, `--baseFrequency`,
`--persistence`, `--lacunarity` and `--seed` are the field itself, and the five
`--obstacle*` flags are the body. `randomFlow_periodic.ipynb` documents every
knob.
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
#: `--no-inviscid --nu 0.01` is the viscosity the notebook this came from ran
#: with: a decaying flow with no physical viscosity decays at whatever rate the
#: scheme's own dissipation happens to give, which is not a controlled
#: experiment. See `05-taylor-green-vortex.ipynb` for how that rate is measured.
PRESET = ['--plot', '--no-bounded', '--obstacle', '--nx', '128',
          '--no-inviscid', '--nu', '0.01',
          '--caseName', '06-randomFlowPeriodic']

if __name__ == '__main__':
    caseMain(randomFlowCase, PRESET + sys.argv[1:])
