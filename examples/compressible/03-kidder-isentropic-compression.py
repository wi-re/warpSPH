#!/usr/bin/env python
"""03 kidder isentropic compression -- thin wrapper around the shared runner.

The notebook this came from is `03-kidder-isentropic-compression.ipynb`; the case itself is
`warpSPH.cases.compressible.kidder`, and everything generic (config, step loop, export,
plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/compressible/03-kidder-isentropic-compression.py
    warpsph-run kidder            # also honours --precision
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.compressible.kidder import kidderCase    # noqa: E402
from warpSPH.runner import caseMain                # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
PRESET = ['--plot']

if __name__ == '__main__':
    caseMain(kidderCase, PRESET + sys.argv[1:])
