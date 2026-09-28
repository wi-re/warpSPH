"""A case, reduced to the parts that actually differ between cases.

Everything a script used to spell out -- config construction, `buildScheme`
unpacking, the step loop, export, plotting, ffmpeg -- is generic and lives in
:mod:`warpSPH.runner.runner`. What is left per case is the triple this module
describes: how the geometry is built, what the initial conditions are, and what
to measure while it runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

__all__ = ['Case', 'RunContext', 'registerCase', 'getCase', 'listCases']


@dataclass
class RunContext:
    """Everything the hooks of a case are handed.

    Built once by :func:`warpSPH.runner.runner.run` and passed to every hook, so
    a case never re-derives the config or re-unpacks ``buildScheme``.
    """

    spec: Any                      # CaseSpec
    case: 'Case'
    config: Any                    # SimulationConfig
    integrator: Any
    schemeConfig: Any              # the scheme's own config dataclass
    scheme: Any                    # enum or str, as handed to buildScheme
    device: Any
    dtype: Any

    #: The scheme's :class:`~warpSPH.schemes.builder.SchemeBundle`, built once.
    #: The seven properties below read through to it rather than copying it, so
    #: there is one source of truth for what a scheme is made of.
    bundle: Any = None

    exportPath: Optional[str] = None
    imagePath: Optional[str] = None
    #: The velocity this run is expected to stay near, for the runner's
    #: velocity alarm (`runner/velocityAlarm.py`). A case sets it during setup
    #: when it knows its own scale (`dambreak`: `U_max`); `None` lets the
    #: runner estimate one. `velocityScaleSource` says where it came from.
    velocityScale: Optional[float] = None
    velocityScaleSource: Optional[str] = None
    # Free-form slot for a case to stash state between hooks (a plotter handle,
    # a reference solution, the initial energy) without globals.
    scratch: Dict[str, Any] = field(default_factory=dict)

    # Read-through accessors for the bundle's members, kept because cases and
    # notebooks address them by these names.
    @property
    def SimulationSystem(self):
        return self.bundle.SimulationSystem

    @property
    def SimulationState(self):
        return self.bundle.SimulationState

    @property
    def SimulationConfig(self):
        return self.bundle.SimulationConfig

    @property
    def SimulationUpdate(self):
        return self.bundle.SimulationUpdate

    @property
    def stepFunction(self):
        return self.bundle.stepFunction

    @property
    def exportFunction(self):
        return self.bundle.exportFunction

    @property
    def importFunction(self):
        return self.bundle.importFunction

    def param(self, name: str, default: Any = None) -> Any:
        return self.spec.params.get(name, default)


@dataclass
class Case:
    """A named case: the hooks, plus the defaults it wants a `CaseSpec` to have.

    Only ``buildSystem`` is required. The rest are optional hooks, each called
    with the :class:`RunContext`.
    """

    name: str
    scheme: str

    #: (ctx) -> SimulationSystem. Geometry and particle sampling.
    buildSystem: Callable[[RunContext], Any]

    #: (ctx) -> None. Mutate ``ctx.schemeConfig`` before the system is built.
    configureScheme: Optional[Callable[[RunContext], None]] = None

    #: (ctx, system) -> None. Stamp velocities/energies onto the sampled state.
    initialConditions: Optional[Callable[[RunContext, Any], None]] = None

    #: (ctx, state) -> dict of scalars, recorded every step and shown on the
    #: progress bar. This is what the tests assert against.
    diagnostics: Optional[Callable[[RunContext, Any], Dict[str, float]]] = None

    #: (ctx, state) -> figure-ish handle, stored on ``ctx.scratch['plot']``.
    setupPlot: Optional[Callable[[RunContext, Any], Any]] = None
    #: (ctx, state, handle, step) -> None.
    updatePlot: Optional[Callable[[RunContext, Any, Any, int], None]] = None

    #: (ctx, state) -> dict merged into every exported frame's extra data.
    extraData: Optional[Callable[[RunContext, Any], Dict[str, Any]]] = None

    #: (ctx, state, step) -> None, called after every integrator step. This is
    #: where a case re-imposes something the step function does not know about
    #: -- Kidder drives its boundary bands from the analytic solution this way.
    postStep: Optional[Callable[[RunContext, Any, int], None]] = None

    #: (ctx, state) -> float, called after every step to pick the next `dt`.
    #: Cases that leave it unset run at the fixed `dt` the setup produced;
    #: `warpSPH.cases.compressible.compressibleTimestep` is the CFL one.
    timestep: Optional[Callable[[RunContext, Any], float]] = None

    #: CaseSpec field overrides that make sense for this case.
    defaults: Dict[str, Any] = field(default_factory=dict)
    #: Case-specific knobs and their defaults; become ``--flags`` and land in
    #: ``CaseSpec.params``.
    params: Dict[str, Any] = field(default_factory=dict)

    #: State field names written every frame under ``storeMode='trajectory'``,
    #: in addition to the always-written positions/velocities/densities --
    #: e.g. a compressible case's ``('internalEnergies', 'supports')``.
    #: Ignored under ``storeMode='states'``.
    extraFields: tuple = ()

    description: str = ''


_REGISTRY: Dict[str, Case] = {}


def registerCase(case: Case) -> Case:
    """Register a case under its name so a CLI can look it up."""
    _REGISTRY[case.name] = case
    return case


def getCase(name: str) -> Case:
    if name not in _REGISTRY:
        raise KeyError(f'Unknown case {name!r}. Known cases: {sorted(_REGISTRY)}')
    return _REGISTRY[name]


def listCases():
    return sorted(_REGISTRY)
