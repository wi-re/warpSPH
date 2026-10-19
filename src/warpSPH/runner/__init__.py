"""Case running: one bootstrap, one spec, one step loop.

Deliberately *not* imported from ``warpSPH/__init__.py`` -- import it explicitly
so it stays out of the flat star-import namespace::

    from warpSPH.runner import CaseSpec, Case, run
"""

from .case import Case, RunContext, getCase, listCases, registerCase
from .caseSpec import CaseSpec, buildArgumentParser, specFromArgs
from .cli import caseMain
from .display import closeWindow, figureOf, holdWindow, openWindow, pumpEvents
from .media import encodeFrames
from .runner import RunResult, buildContext, prepareContext, resolveEnum, run
from .parameters import Parameter
from .configuration import RunConfiguration, configurationSchema, parameterReference, resolvedConfiguration, configurationCatalogue

__all__ = [
    'Case', 'RunContext', 'registerCase', 'getCase', 'listCases',
    'CaseSpec', 'buildArgumentParser', 'specFromArgs', 'caseMain',
    'RunResult', 'run', 'buildContext', 'prepareContext', 'resolveEnum',
    'Parameter', 'RunConfiguration', 'configurationSchema', 'parameterReference', 'resolvedConfiguration', 'configurationCatalogue',
    'encodeFrames',
    'openWindow', 'pumpEvents', 'holdWindow', 'closeWindow', 'figureOf',
]
