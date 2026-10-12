#!/usr/bin/env python
"""Impact of two fluid spheres (2D) -- thin wrapper around the shared runner.

The case itself is `warpSPH.cases.freeSurface.impact`, and everything generic (config, step
loop, export, plotting, ffmpeg) is `warpSPH.runner`. Equivalent invocations::

    python examples/weaklyCompressible/01-impact/impact_spheres.py
    warpsph-run impact --shape circle --nx 256      # also honours --precision

`impact` is the *family*, not just this pair of circles: `--shape` takes any of
`SHAPE_PRESETS` (circle, box, hexagon, star5, ...), `--arrangement ring
--nBodies 5` swaps the pair for a closing ring, and `--impactAngle`,
`--lateralOffset` and `--spin` turn the head-on collision into a glancing or
spinning one. `impact_squares.py` is the other shipped point of that family;
`impact_spheres.ipynb` documents every knob and draws the shapes.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.freeSurface.impact import impactCase    # noqa: E402
from warpSPH.runner import caseMain            # noqa: E402

#: What this example fixes relative to the case defaults; still overridable,
#: because these are prepended to the command line rather than appended.
#: `--plot` is on because an example is meant to be watched -- pass `--no-plot`
#: for a headless run, or `--no-show` to keep the frames without a window.
PRESET = ['--plot', '--shape', 'circle', '--size', '0.5', '--separation', '0.75',
          '--impactAxis', '0', '--nx', '256', '--tLimit', '10.0',
          '--caseName', '01-impactSpheres']

if __name__ == '__main__':
    caseMain(impactCase, PRESET + sys.argv[1:])
