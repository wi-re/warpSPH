#!/usr/bin/env python
"""Sod shock tube (1D) -- thin wrapper around the shared runner.

The 264 lines that used to live here (boilerplate cell, argparse block, step
loop, export, ffmpeg) are now `warpSPH.runner`, and the case itself is
`warpSPH.cases.compressible.sod`. Equivalent invocations::

    python examples/compressible/01-sod/sod_1d.py --plot --store
    python -m warpSPH.cases.compressible.sod --plot --store
    warpsph-run sod --plot --store            # also honours --precision

This example demos this case's export as a single growing `trajectory.h5`
(see `sod_resume.py` and `warpSPH.io.loadTrajectory`) rather than the
one-file-per-stored-step layout `--storeMode states` (the default for
`sod`, unaffected here, still used by `scripts/run_sweep.py` and the tests)
writes.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.compressible.sod import sodCase          # noqa: E402
from warpSPH.runner import caseMain            # noqa: E402

#: `--plot`/`--video` are on because an example is meant to be watched -- pass
#: `--no-plot` for a headless run, `--no-show` to keep the frames without a
#: window, or `--no-video` to skip the ffmpeg encode.
#: `--storeMode trajectory` demos the single-growing-file export this
#: directory's `sod_resume.py` reads back from; `--exportInterval` is a
#: *simulated-time* interval (unlike the states-mode `--storeInterval`, which
#: counts steps), picked here for roughly 30 stored frames over the default
#: `tLimit=0.15`.
PRESET = ['--plot', '--video', '--store', '--storeMode', 'trajectory', '--exportInterval', '0.005']

if __name__ == '__main__':
    caseMain(sodCase, PRESET + sys.argv[1:])
