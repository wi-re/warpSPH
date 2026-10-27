"""Export or audit warpSPHCases recipes through warpSPH's configuration contract.

Requires the optional editable warpSPHCases package. No run, notebook, figure,
trajectory, catalogue file or website in that checkout is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='warpSPHCases data checkout')
    parser.add_argument('--page', help='page slug for one exported recipe')
    parser.add_argument('--run', help='run id for one exported recipe')
    parser.add_argument('--audit', action='store_true', help='validate all public recipes with migrated case contracts')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.audit and (not args.page or not args.run):
        parser.error('export needs --page and --run')
    try:
        from warpSPHCases.spec import allPages, loadPage
        from warpSPHCases.runs import _substitute
    except ModuleNotFoundError:
        parser.error('install the optional warpSPHCases package in this environment (pip install -e ../warpSPHCases)')
    entry = page = None
    if not args.audit:
        page = loadPage(args.page, root=args.root)
        entry = page.run(args.run)
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=entry.spec.get('precision', 'float32') if entry else 'float32')
    from warpSPH.cases import importAll
    from warpSPH.runner import RunConfiguration, getCase
    importAll()

    def configuration(recipe, pageDir):
        return RunConfiguration.fromOverrides(getCase(recipe.case),
            _substitute(recipe.caseSpecOverrides(), pageDir))

    if args.audit:
        result = {'passed': [], 'skipped': [], 'failed': []}
        for page in allPages(args.root):
            for entry in page.publicRuns:
                row = {'page': page.slug, 'run': entry.id, 'case': entry.case, 'scheme': entry.scheme}
                try:
                    case = getCase(entry.case)
                    if not case.parameterDefinitions:
                        result['skipped'].append({**row, 'reason': 'case contract not migrated'})
                        continue
                    configuration(entry, page.dir)
                    result['passed'].append(row)
                except (ValueError, TypeError, KeyError) as error:
                    result['failed'].append({**row, 'error': str(error)})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(' / '.join(f'{len(result[k])} {k}' for k in ('passed', 'skipped', 'failed')))
        return int(bool(result['failed']))
    try:
        document = configuration(entry, page.dir)
        document.save(args.output)
    except (ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(f'Exported {page.slug}/{entry.id} to {args.output}; recipe label {entry.variant!r} retained as a recipe label, not a physical variant.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
