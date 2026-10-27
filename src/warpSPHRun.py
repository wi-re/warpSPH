"""CLI entry point: ``warpsph-run <case> [options]``.

Lives at top level, alongside :mod:`warpSPHBootstrap`, so that it can bootstrap
the precision *before* importing ``warpSPH``. ``python -m warpSPH.cases.compressible.sod``
works too, but by then ``warpSPHCore`` is already imported and ``--precision``
can no longer take effect.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from warpSPHBootstrap import PRECISIONS, bootstrap

__all__ = ['main']


def _buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='warpsph-run',
        description='Run a warpSPH case. Run without a case name to list them.',
        add_help=False,
        usage='%(prog)s [case] [options]',
    )
    # Deliberately no `choices`: knowing the case names means importing the
    # case modules, and that would pull in warpSPHCore and lock the precision
    # before --precision has been read. Unknown names are caught by `getCase`.
    parser.add_argument('--configurationCatalogue', metavar='JSON', help='Write case contract coverage and exit.')
    parser.add_argument('--config', help='Legacy CaseSpec or versioned JSON/YAML run document.')
    parser.add_argument('--precision', choices=PRECISIONS, default=None,
                        help='scalar precision; must be chosen before import (default: float32)')
    parser.add_argument('--dim', default=None,
                        help="fixed dimension for warp types, or 'Any' (default: Any)")
    parser.add_argument('-h', '--help', action='store_true', dest='help')
    return parser


def _listCases() -> int:
    """Print the registry. Only reached on the help path, so the import is free."""
    from warpSPH.cases import importAll
    importAll()
    from warpSPH.runner import getCase, listCases

    print('\ncases:')
    for name in listCases():
        print(f'  {name:<18} {getCase(name).description}')
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _buildParser()
    known, rest = parser.parse_known_args(argv)
    # Unknown case flags may have positional values (e.g. --nx 80). A generic
    # positional argparse action would mistake that value for the case name.
    known.case = rest.pop(0) if rest and not rest[0].startswith('-') else None

    raw = {}
    if known.config:
        from warpSPHConfig import readMapping
        try:
            raw = readMapping(known.config)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    versioned = 'schemaVersion' in raw
    simulation = raw.get('simulation', {}) if versioned else raw
    if not isinstance(simulation, dict):
        parser.error('configuration.simulation must be a mapping')
    precision = known.precision or simulation.get('precision', 'float32')
    if precision not in PRECISIONS:
        parser.error(f'invalid configuration precision: {precision!r}')
    if versioned:
        fileCase = raw.get('case')
        if not isinstance(fileCase, str) or not fileCase:
            parser.error('configuration.case must be a registered case name')
        if known.case is not None and known.case != fileCase:
            parser.error('positional case must agree with configuration.case')
        known.case = fileCase
    dim = known.dim
    if dim is None or str(dim).lower() in ('any', 'dynamic'):
        from typing import Any
        dim = Any
    else:
        dim = int(dim)

    bootstrap(precision=precision, dim=dim)

    if known.configurationCatalogue:
        from warpSPH.runner.configuration import configurationCatalogue, _write
        _write(known.configurationCatalogue, configurationCatalogue())
        return 0

    if known.case is None:
        parser.print_help()
        _listCases()
        return 0 if known.help else 2

    from warpSPH.cases import importAll
    importAll()

    from warpSPH.runner import caseMain, getCase
    case = getCase(known.case)
    if known.help:
        rest = rest + ['--help']
    # Keep the precision the case sees consistent with what was bootstrapped.
    if known.config:
        rest += ['--config', known.config]
    if known.dim is not None and str(known.dim).lower() not in ('any', 'dynamic'):
        rest += ['--dim', str(known.dim)]
    result = caseMain(case, rest + ['--precision', precision])
    return 1 if result is not None and result.diverged else 0


if __name__ == '__main__':
    raise SystemExit(main())
