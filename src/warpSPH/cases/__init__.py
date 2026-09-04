"""Runnable cases.

Each module here declares one :class:`~warpSPH.runner.case.Case` and nothing
else -- the boilerplate, the config plumbing and the step loop all live in
:mod:`warpSPH.runner`. Import a case module to register it; ``warpSPHRun``
imports them all so it can dispatch by name.

The three ``*.py`` modules that are *not* cases -- :mod:`compressible`,
:mod:`weaklyCompressible` and :mod:`plotting` -- hold what the cases of each
family share: the scheme settings, the domain construction, and the plot hooks.
"""

from __future__ import annotations

#: Every module declaring a case, in the order the examples are numbered.
CASE_MODULES = (
    # compressible -- examples/compressible/*.ipynb
    'sod',
    'sodND',
    'linearWave',
    'kidder',
    'noh',
    'woodwardColella',
    'sedov',
    'hydrostatic',
    'greshoVortex',
    'yeeVortex',
    'shearingNoh',
    'kelvinHelmholtz',
    'rayleighTaylor',
    'triplePoint',
    # weakly compressible -- examples/weaklyCompressible/*.ipynb
    'impact',
    'rotatingSquarePatch',
    'oscillatingDroplet',
    'tgvWeaklyCompressible',
    'randomFlow',
    'kolmogorov',
    'lidDrivenCavity',
    'movingObstacle',
    'drivenSquare',
    'dambreak',
    'sloshingTank',
    'channelFlow',
    # incompressible -- examples/incompressible/*.ipynb
    'tgv',
    'kolmogorovIncompressible',
    'randomFlowIncompressible',
    'shearWave',
    'staticBlob',
    'hydrostaticColumn',
    'columnCollapse',
    # non-fluid demo -- see docs/historic_plans/WAVE_EQUATION_PLAN.md
    'waveEquation',
)


def importAll():
    """Import every case module so the registry is fully populated."""
    import importlib
    for name in CASE_MODULES:
        importlib.import_module(f'{__name__}.{name}')


__all__ = ['CASE_MODULES', 'importAll']
