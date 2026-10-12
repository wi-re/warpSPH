"""Runnable cases, grouped by physical regime.

Each case module declares one :class:`~warpSPH.runner.case.Case` and nothing
else -- the boilerplate, the config plumbing and the step loop all live in
:mod:`warpSPH.runner`. Import a case module to register it; ``warpSPHRun``
imports them all so it can dispatch by name.

The families are subpackages (the same grouping as the pages of the
``warpSPHCases`` repo, which spells them with hyphens):

========================  ==============================================================
:mod:`.compressible`      shock tubes, blasts, vortices, instabilities (no walls)
:mod:`.compressibleWalls` compressible gas with solid walls
:mod:`.freeSurface`       dam break, sloshing, droplet, impact, column collapse
:mod:`.periodic`          periodic-box flows (Taylor-Green, Kolmogorov, random flow, ...)
:mod:`.wallBounded`       lid-driven cavity, moving obstacle, driven square
:mod:`.boundaries`        boundary-treatment cases (channel flows, Taylor-Couette, ...)
:mod:`.demos`             non-fluid demos
========================  ==============================================================

What several cases share lives next to them: ``compressible.common`` and
``compressibleWalls.common`` (scheme settings, wall construction), and
:mod:`.common` for what spans families (``plotting``, ``weaklyCompressible``).
Registry names (``warpsph-run sod``, ``caseSpec.json``) do not depend on the
module layout.

The pre-2026-10 flat paths (``warpSPH.cases.sod``, ``warpSPH.cases.plotting``,
...) still import, with a :class:`DeprecationWarning`, through
:data:`LEGACY_MODULES`; ``python -m`` needs the new path.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

#: Every module declaring a case, as a path relative to this package.
CASE_MODULES = (
    # compressible -- examples/compressible/*.ipynb
    'compressible.sod',
    'compressible.sodND',
    'compressible.linearWave',
    'compressible.kidder',
    'compressible.noh',
    'compressible.woodwardColella',
    'compressible.sedov',
    'compressible.hydrostatic',
    'compressible.greshoVortex',
    'compressible.yeeVortex',
    'compressible.shearingNoh',
    'compressible.shearBox',
    'compressible.kelvinHelmholtz',
    'compressible.rayleighTaylor',
    'compressible.triplePoint',
    # compressible with solid walls
    'compressibleWalls.closedBox',
    'compressibleWalls.piston',
    'compressibleWalls.shockReflection',
    'compressibleWalls.woodwardColellaWalls',
    'compressibleWalls.sedovWalls',
    'compressibleWalls.shockCylinder',
    'compressibleWalls.bowShock',
    'compressibleWalls.forwardStep',
    # free surface
    'freeSurface.impact',
    'freeSurface.rotatingSquarePatch',
    'freeSurface.oscillatingDroplet',
    'freeSurface.dambreak',          # also declares `openFlow`
    'freeSurface.sloshingTank',
    'freeSurface.columnCollapse',
    # periodic
    'periodic.tgvWeaklyCompressible',
    'periodic.tgv',
    'periodic.randomFlow',
    'periodic.randomFlowIncompressible',
    'periodic.kolmogorov',
    'periodic.kolmogorovIncompressible',
    'periodic.shearWave',
    'periodic.staticBlob',
    # wall-bounded
    'wallBounded.lidDrivenCavity',
    'wallBounded.movingObstacle',
    'wallBounded.drivenSquare',
    # boundary treatments
    'boundaries.channelFlow',
    'boundaries.periodicChannel',
    'boundaries.stokesArray',
    'boundaries.taylorCouette',
    'boundaries.cylinderWake',
    'boundaries.hydrostaticColumn',
    # non-fluid demo -- see docs/historic_plans/WAVE_EQUATION_PLAN.md
    'demos.waveEquation',
)

#: Old flat module name -> new path relative to this package.
LEGACY_MODULES = {
    'plotting': 'common.plotting',
    'weaklyCompressible': 'common.weaklyCompressible',
    'channelFlow': 'boundaries.channelFlow',
    'randomFlowIncompressible': 'periodic.randomFlowIncompressible',
    **{name.rsplit('.', 1)[1]: name for name in CASE_MODULES},
}


def importAll():
    """Import every case module so the registry is fully populated."""
    for name in CASE_MODULES:
        importlib.import_module(f'{__name__}.{name}')


class _LegacyModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Make ``warpSPH.cases.<oldFlatName>`` resolve to the regrouped module.

    The old name becomes an alias of the very same module object (so classes
    and pickled references keep one identity). ``compressible`` /
    ``compressibleWalls`` are real packages now and forward the old helper
    names to their ``common`` module instead (see their ``__getattr__``).
    """

    _prefix = __name__ + '.'

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._prefix):
            return None
        if fullname[len(self._prefix):] not in LEGACY_MODULES:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        old = spec.name[len(self._prefix):]
        new = LEGACY_MODULES[old]
        warnings.warn(f'{spec.name} moved to {self._prefix}{new}; the flat path is deprecated',
                      DeprecationWarning, stacklevel=2)
        return importlib.import_module(f'{self._prefix}{new}')

    def exec_module(self, module):
        pass


if not any(isinstance(f, _LegacyModuleFinder) for f in sys.meta_path):
    sys.meta_path.append(_LegacyModuleFinder())

__all__ = ['CASE_MODULES', 'LEGACY_MODULES', 'importAll']
