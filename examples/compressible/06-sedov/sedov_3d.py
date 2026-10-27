#!/usr/bin/env python
"""Sedov-Taylor blast wave (3D) -- thin wrapper around the shared runner.

`sedov_2d.py` one dimension further. There was no 3D Sedov notebook before
this directory -- `warpSPH.cases.compressible.sedov`'s sampler, stopping rule and plotting
were already dimension-generic, so getting a 3D run only needed the `hat`
initialization fix (see `sedov_1d.py`'s docstring) and this wrapper.
Equivalent invocations::

    python examples/compressible/06-sedov/sedov_3d.py --plot --store
    warpsph-run sedov --plot --store --dim 3      # also honours --precision

`nx=32` is ~33k particles (`nx**dim`), a comparable budget to the 2D case's
40k for one dimension up.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.compressible.sedov import sedovCase       # noqa: E402
from warpSPH.runner import caseMain             # noqa: E402

PRESET = ['--plot', '--video', '--dim', '3', '--nx', '32',
          '--caseName', '06-sedovTaylorBlastwave3D']

if __name__ == '__main__':
    caseMain(sedovCase, PRESET + sys.argv[1:])
