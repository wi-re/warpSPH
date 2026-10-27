#!/usr/bin/env python
"""Triple-point interaction (equal-spacing sampling) -- thin wrapper around the
shared runner.

The notebook this came from is `triplePoint_equalSpacing.ipynb`; the case
itself is `warpSPH.cases.compressible.triplePoint` (one `Case` covers both sampling
strategies, selected by `--equalMass`/`--no-equalMass` the way `sedovCase`
selects dimension by `--dim`), and everything generic (config, step loop,
export, plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/compressible/14-triplePoint/triplePoint_equalSpacing.py
    warpsph-run triplePoint --no-equalMass --nx 256   # also honours --precision

See `triplePoint_equalMass.py` for the equal-mass variant.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.compressible.triplePoint import triplePointCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
#: `--caseName` keeps this variant's exports apart from
#: `triplePoint_equalMass.py`'s.
PRESET = ['--plot', '--no-equalMass', '--nx', '256',
          '--caseName', '14-triplePointEqualSpacing']

if __name__ == '__main__':
    caseMain(triplePointCase, PRESET + sys.argv[1:])
