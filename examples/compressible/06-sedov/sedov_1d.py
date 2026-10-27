#!/usr/bin/env python
"""Sedov-Taylor blast wave (1D) -- thin wrapper around the shared runner.

`sedovCase` is one `Case` covering all three dimensionalities (see
`warpSPH.cases.compressible.sedov`'s module docstring); this script just pins `--dim 1` and
the resolution the old `06-Sedov_Taylor_Blastwave_1D.ipynb` used. Equivalent
invocations::

    python examples/compressible/06-sedov/sedov_1d.py --plot --store
    python -m warpSPH.cases.compressible.sedov --plot --store --dim 1
    warpsph-run sedov --plot --store --dim 1   # also honours --precision

Localized energy deposited into a near-rest medium drives a self-similar
blast wave; the run stops once the shock reaches `goalRadius`, a time derived
from the analytic self-similar solution (`warpSPH.caseUtils.SedovSolution`)
rather than a number chosen by hand. `initialization='hat'` (the default) is
the smoothed initial condition: it deposits all of `E0` on the single
particle nearest the origin, exactly like `initialization='singular'`, then
runs one SPH interpolation pass over that field to spread the spike across
one smoothing scale -- see `sedov_1d.ipynb` for the picture. `--initialization
singular` reverts to the raw, unsmoothed spike; `--initialization quadrant`
spreads `E0` evenly over the `2**dim` innermost particles instead.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.compressible.sedov import sedovCase       # noqa: E402
from warpSPH.runner import caseMain             # noqa: E402

#: `--plot`/`--video` are on because an example is meant to be watched -- pass
#: `--no-plot` for a headless run, `--no-show` to keep the frames without a
#: window, or `--no-video` to skip the ffmpeg encode. `--caseName` keeps this
#: dimensionality's exports apart from `sedov_2d.py`/`sedov_3d.py`'s.
PRESET = ['--plot', '--video', '--dim', '1', '--nx', '800',
          '--caseName', '06-sedovTaylorBlastwave1D']

if __name__ == '__main__':
    caseMain(sedovCase, PRESET + sys.argv[1:])
