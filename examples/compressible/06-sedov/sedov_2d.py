#!/usr/bin/env python
"""Sedov-Taylor blast wave (2D) -- thin wrapper around the shared runner.

The same case as `sedov_1d.py`, one dimension up: `warpSPH.cases.compressible.sedov`'s
sampler and stopping rule are dimension-generic, so this only pins `--dim 2`
and the resolution the old `07-Sedov_Taylor_Blastwave_2D.ipynb` used.
Equivalent invocations::

    python examples/compressible/06-sedov/sedov_2d.py --plot --store
    warpsph-run sedov --plot --store --dim 2      # also honours --precision

See `sedov_1d.py`'s docstring for what `initialization` does; `sedov_2d.ipynb`
shows the two shock-radius reference targets (the full self-similar solve and
the closed-form beta-fit estimate) the profile panels overlay.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.compressible.sedov import sedovCase       # noqa: E402
from warpSPH.runner import caseMain             # noqa: E402

#: As in `sedov_1d.py`: an example is meant to be watched, so plotting is on --
#: pass `--no-plot` for a headless run, `--no-show` to keep the frames without
#: a window, `--no-video` to skip the ffmpeg encode.
PRESET = ['--plot', '--video', '--dim', '2', '--nx', '200',
          '--caseName', '06-sedovTaylorBlastwave2D']

if __name__ == '__main__':
    caseMain(sedovCase, PRESET + sys.argv[1:])
