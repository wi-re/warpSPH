"""Channel flow past a fixed obstacle (2D), weakly compressible.

The script form of this case was `examples/weaklyCompressible/13-open-flow.ipynb`.

It is the same construction the dam break already uses -- a `W x L` box, a
boundary region cut from an obstacle SDF, and a freestream forcing band at the
inflow -- assembled by the very same `caseUtils.weaklyCompressible` helpers. So
rather than a second copy of that machinery, this is the dam break's hooks
under different defaults: no gravity, the box filled with fluid, and the
freestream switched on.

The notebook reached the same place by hand, writing out the obstacle SDF, the
Dirichlet band and the inflow ramp inline; it had already started importing
`buildObstacleSDF` from the shared helpers.

`drivenSquare` (case 11, `examples/weaklyCompressible/11-driven-square.py`) used
to live here too, built the same way with `band`/`semiPeriodic` numbers picked
to approximate a driven square -- but a fixed obstacle in a driven channel is
not what "driven square" describes, and it moved to its own module
(`warpSPH.cases.wallBounded.drivenSquare`) rather than staying a `channelCase` under a
misleading name.
"""

from __future__ import annotations

from ..freeSurface.dambreak import (buildSystem, configureScheme, diagnostics, extraData,
                       initialConditions, setupPlot, updatePlot)
from ...runner import Case, caseMain, registerCase
from ..freeSurface.dambreak import dambreakCase

__all__ = ['openFlowCase', 'channelCase']


def channelCase(name: str, description: str, defaults, params) -> Case:
    """A dam-break-hooked case with channel defaults."""
    return registerCase(Case(
        name=name,
        scheme='deltaSPH',
        description=description,
        buildSystem=buildSystem,
        configureScheme=configureScheme,
        initialConditions=initialConditions,
        diagnostics=diagnostics,
        setupPlot=setupPlot,
        updatePlot=updatePlot,
        extraData=extraData,
        defaults=dict(dambreakCase.defaults, **defaults),
        params=dict(dambreakCase.params, **params),
    ))


openFlowCase = channelCase(
    'openFlow',
    'Open channel flow past an obstacle (2D), weakly compressible deltaSPH.',
    defaults=dict(
        caseName='13-openFlow',
        nx=128,
        L=2.0,
        tLimit=5.0,
    ),
    params=dict(
        W=4.0,
        band=5,
        # A partially filled, semi-periodic channel with a free surface: this
        # is what separates "open flow" from the closed driven square below.
        fillRatio=0.25,
        fluidWidth=1.0,
        semiPeriodic=True,
        disableGravity=False,
        enableFreestream=True,
        freeStreamVelocity=1.0,
        forcingWidth=2.0 / 16.0,
        obstacleActive=True,
        obstacleType='circleMiddle',
        maxExtent=0.125,
        offsetX=0.0,
    ),
)

if __name__ == '__main__':
    caseMain(openFlowCase)
