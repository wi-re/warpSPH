"""Check or smoke-run migrated case contracts in separate sequential processes.

Example: python examples/configuration/check_catalogue.py --steps 5 --nx 64
         --outputRoot /tmp/new-configured-case-smoke
The output directory must be new; generated configurations and logs stay outside Git.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputRoot', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=0, help='zero checks setup only; positive values step each case')
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args(argv)
    if args.steps < 0 or args.nx < 1 or args.timeout <= 0:
        parser.error('steps must be nonnegative; nx and timeout must be positive')
    if args.outputRoot.exists():
        parser.error('choose a new outputRoot to preserve previous evidence')
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    from warpSPH.runner import RunConfiguration, configurationCatalogue
    ready = {c['id'] for c in configurationCatalogue()['cases'] if c['status'] == 'ready'}
    names = sorted(args.cases or ready)
    if set(names) - ready:
        parser.error(f'cases without a ready contract: {sorted(set(names) - ready)}')
    args.outputRoot.mkdir(parents=True)
    rows = []
    for name in names:
        directory = args.outputRoot / name
        directory.mkdir()
        simulation = {'nx': args.nx, 'plot': False, 'store': False, 'quiet': True, 'progress': False}
        if args.steps:
            simulation['nSteps'] = args.steps
        document = RunConfiguration(name, simulation=simulation)
        path = directory / 'configuration.yaml'
        document.save(path)
        command = [sys.executable, '-m', 'warpSPHRun', '--config', str(path),
                   '--saveResolvedConfig', str(directory / 'resolved.json')]
        if not args.steps:
            command.append('--checkConfig')
        try:
            with (directory / 'run.log').open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            status = 'passed' if completed.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            status = 'timeout'
        rows.append({'case': name, 'status': status})
        print(f'{name}: {status}', flush=True)
    (args.outputRoot / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    return int(any(r['status'] != 'passed' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
