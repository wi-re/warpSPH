#!/usr/bin/env python
"""Sod shock tube (2D slab) -- thin wrapper around the shared runner.

The 1D problem extruded into a periodic slab, and the reason the case exists:
sampling both states at equal particle *mass* rather than equal spacing. In 1D
that fell out of `samplingRatio=4` being the density ratio; in 2D the light
side has to be coarsened by `sqrt(rho_l/rho_r)` in both directions instead.
See `warpSPH.caseUtils.compressible.sod.sodND` for how the counts are picked,
and `sod_3d.py` for the same case one dimension up. Equivalent invocations::

    python examples/compressible/01-sod/sod_2d.py --plot --store
    python -m warpSPH.cases.compressible.sodND --plot --store
    warpsph-run sod2d --plot --store          # also honours --precision

Pass `--no-equalMass` to sample both states on the same lattice instead, which
leaves the dense side's particles 4x heavier across the contact discontinuity
-- the comparison this case is built to make easy.

The six panels are the 1D case's, as a scatter: the solution is uniform along
the slab, so every particle at a given x should land on the same point of the
analytic profile, and the visible spread is the error.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.compressible.sodND import sod2dCase        # noqa: E402
from warpSPH.runner import caseMain              # noqa: E402

#: As in `sod_1d.py`: an example is meant to be watched, so plotting is on --
#: pass `--no-plot` for a headless run, `--no-show` to keep the frames without
#: a window, `--no-video` to skip the ffmpeg encode.
PRESET = ['--plot', '--video', '--store', '--storeMode', 'trajectory', '--exportInterval', '0.005']

if __name__ == '__main__':
    caseMain(sod2dCase, PRESET + sys.argv[1:])
