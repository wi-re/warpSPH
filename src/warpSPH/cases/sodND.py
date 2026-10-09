"""Sod shock tube in 2D and 3D, compressible.

The same Riemann problem `sod.py` runs, extruded into a periodic slab: the
shock still travels along x, and y (and z) are periodic directions the solution
is uniform along. The x arrangement is the 1D case's unchanged -- dense state
in the middle half, light state wrapped around the outer quarters, so the
mirror symmetry at x=0 and x=±L/2 does the work a reflecting wall would.

What is genuinely new here is the sampling: `buildSodND` gives both states the
same particle *mass* rather than the same spacing, which in 1D fell out of the
`samplingRatio=4` default but in 2D/3D needs the light side coarsened by
`(rho_l/rho_r)**(1/dim)` in every direction. See
`caseUtils/compressible/sod/sodND.py` for how the two integer counts that come
out of that are chosen.

Everything else -- the Riemann states, the scheme configuration, the
diagnostics, the six profile panels -- is imported from `sod.py` rather than
restated, so the three cases cannot drift apart.
"""

from __future__ import annotations

import functools

from ..caseUtils import buildSodND
from ..runner import Case, RunContext, caseMain, registerCase
from .compressible import paramExtraData
from .sod import configureScheme, diagnostics, setupPlot, states, updatePlot

__all__ = ['sod2dCase', 'sod3dCase']


def buildSystem(ctx: RunContext):
    left, right = states(ctx)
    return buildSodND(
        ctx.SimulationSystem, ctx.SimulationState,
        left, right,
        ctx.param('gamma'), ctx.config,
        transverseSpacings=ctx.param('transverseSpacings'),
        equalMass=ctx.param('equalMass'),
        adaptiveSupportScheme=ctx.schemeConfig.adaptiveSupportScheme,
        verbose=not ctx.spec.quiet,
    )


def _sodNDCase(name: str, dim: int, nx: int, description: str) -> Case:
    return registerCase(Case(
        name=name,
        scheme='CompSPH',
        description=description,
        buildSystem=buildSystem,
        configureScheme=configureScheme,
        diagnostics=diagnostics,
        # Many particles share an x once there is a transverse direction, so
        # the profile panels are a scatter from the first frame on.
        setupPlot=functools.partial(setupPlot, scatter=True),
        updatePlot=updatePlot,
        extraData=paramExtraData,
        extraFields=('internalEnergies', 'supports'),
        defaults=dict(
            caseName=f'01-sodShockTube{dim}D',
            dim=dim,
            #: The dense state's particle count across its own half of the
            #: domain, i.e. the same meaning `nx` has in the 1D case.
            nx=nx,
            L=2.0,
            n_h=4.0,
            periodic=True,
            kernel='B7',
            integrationScheme='rungeKutta2',
            supportMode='Gather',
            gradientMode='Difference',
            laplacianMode='Brookshaw',
            samplingScheme='regular',
            tLimit=0.15,
            dt=1e-3,
            adaptiveDt=True,
            cflFactor=0.3,
            plotInterval=10,
            storeInterval=50,
        ),
        params=dict(
            gamma=5 / 3,
            #: Width of the periodic slab in each transverse direction, in
            #: dense-side particle spacings -- so the transverse particle count
            #: is this number regardless of `nx`, and the "wider than twice the
            #: support radius" constraint holds at every resolution (the
            #: sampler raises if it does not). 20 clears it by ~1.5x on both
            #: sides while keeping the particle count example-sized.
            transverseSpacings=20,
            #: False samples both states on the same lattice instead, leaving
            #: the dense side's particles `rho_l/rho_r` times heavier -- the
            #: comparison this case exists to make easy.
            equalMass=True,
            left_rho=1.0,
            left_pressure=1.0,
            left_velocity=0.0,
            right_rho=0.25,
            right_pressure=0.1795,
            right_velocity=0.0,
            viscositySwitch=None,      # the scheme's default (cases/compressible.py COMPRESSIBLE_PARAMS)
            adaptiveSupportScheme='Owen',
            adaptiveSupportCorrections=False,
        ),
    ))


#: ~3100 particles at the default nx=100 (100x25 dense, 52x12 light).
sod2dCase = _sodNDCase('sod2d', dim=2, nx=100,
                       description='Sod shock tube (2D slab), compressible SPH.')

#: ~5000 particles at the default nx=40 (40x10x10 dense, 14x6x6 light). 3D
#: costs a transverse count in every direction, so the default resolution is
#: lower than 2D's for a comparable particle budget.
sod3dCase = _sodNDCase('sod3d', dim=3, nx=40,
                       description='Sod shock tube (3D slab), compressible SPH.')


if __name__ == '__main__':
    caseMain(sod2dCase)
