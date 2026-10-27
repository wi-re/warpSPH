#!/usr/bin/env python
"""Sod shock tube (3D slab) -- thin wrapper around the shared runner.

`sod_2d.py` one dimension further: a periodic slab in both y and z, with the
light state coarsened by `(rho_l/rho_r)**(1/3)` in every direction to keep the
particle masses equal. That cube root is irrational, so unlike 1D and 2D the
match cannot be exact here -- the sampler gets within ~1% and prints what it
managed. Equivalent invocations::

    python examples/compressible/01-sod/sod_3d.py --plot --store
    warpsph-run sod3d --plot --store          # also honours --precision

The default `nx=40` is ~20k particles: 3D pays the transverse count twice, so
the resolution along the tube is lower than the 2D case's for a comparable
budget. `--nx` scales it; the slab width follows automatically, being measured
in particle spacings.

This was the first case in the repo to run any 3D physics, and it immediately
found a real bug: `warpSPHCore`'s B7 kernel had a 3D normalisation constant
16x too small (1D and 2D were right), so every density came out at 1/16 of the
mass it was built from. Fixed in warpSPHCore; if a 3D run here ever comes back
with uniformly wrong densities and correct-looking wave positions, suspect the
kernel normalisation first.
"""

import sys

from warpSPHBootstrap import bootstrap

bootstrap(precision='float32')

from warpSPH.cases.compressible.sodND import sod3dCase        # noqa: E402
from warpSPH.runner import caseMain              # noqa: E402

PRESET = ['--plot', '--video', '--store', '--storeMode', 'trajectory', '--exportInterval', '0.005']

if __name__ == '__main__':
    caseMain(sod3dCase, PRESET + sys.argv[1:])
