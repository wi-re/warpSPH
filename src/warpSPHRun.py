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
    )
    # Deliberately no `choices`: knowing the case names means importing the
    # case modules, and that would pull in warpSPHCore and lock the precision
    # before --precision has been read. Unknown names are caught by `getCase`.
    parser.add_argument('case', nargs='?', help='which case to run')
    parser.add_argument('--precision', choices=PRECISIONS, default='float32',
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

    dim = known.dim
    if dim is None or str(dim).lower() in ('any', 'dynamic'):
        from typing import Any
        dim = Any
    else:
        dim = int(dim)

    bootstrap(precision=known.precision, dim=dim)

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
    result = caseMain(case, rest + ['--precision', known.precision])
    return 1 if result.diverged else 0


if __name__ == '__main__':
    raise SystemExit(main())
