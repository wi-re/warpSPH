"""One recorded native projection control on an existing shared matrix fixture.

Preserves the source dataset and uses current numerical sources, snapshotted by
the matrix worker. Numerical overrides are explicit and recorded; no retries.
For example, select hydrostatic__dfsph__particles and --pressure-boundary
band2018 --max-iterations 1000, or a TGV row and --df-error-percent .001.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--row', required=True)
    parser.add_argument('--t-limit', type=float)
    parser.add_argument('--pressure-boundary', choices=['native', 'band2018'])
    parser.add_argument('--max-iterations', type=int)
    parser.add_argument('--df-error-percent', type=float)
    parser.add_argument('--iisph-relaxation', type=float)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest.get('profile') != 'prepared-incompressible':
        parser.error('Use a prepared incompressible matrix source')
    row = copy.deepcopy(next(r for r in manifest['rows'] if r['id'] == args.row))
    if row.get('solver') not in ('dfsph', 'iisph') or row.get('skip'):
        parser.error('Select a supported native DFSPH or IISPH row')
    if row['solver'] != 'dfsph' and (args.pressure_boundary is not None or args.df_error_percent is not None):
        parser.error('DFSPH overrides require a DFSPH row')
    if out.exists():
        parser.error('Use a fresh output directory')
    if args.t_limit is not None:
        row['spec']['tLimit'] = args.t_limit
    fixture = Path('fixtures')/f"{row['test']}__{row['boundary']}"
    if not (source/fixture/'state.pt').is_file():
        parser.error('The source shared fixture must already exist')
    out.mkdir(parents=True)
    shutil.copytree(source/fixture, out/fixture)
    overrides = {k:v for k,v in vars(args).items()
                 if k not in ('source','out','row') and v is not None}
    manifest.update(rows=[row], schemes=[row['scheme']],
                    experiment=dict(source=str(source), overrides=overrides,
                                    policy='One explicit control; current sources; saved physical fixture unchanged.'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=row['precision'])
    import prepared_fluidMatrix as prepared
    from run_fluidMatrix import worker
    original = prepared.prepared_case

    def adapt(case, root, selected):
        case = original(case, root, selected)
        configure = case.configureScheme

        def config(ctx):
            configure(ctx)
            sc = ctx.schemeConfig
            if args.pressure_boundary is not None:
                sc.dfsph.pressureBoundary = args.pressure_boundary
            if args.max_iterations is not None:
                for policy in (sc.iisph, sc.dfsph.density, sc.dfsph.divergence):
                    policy.maxIterations = args.max_iterations
            if args.df_error_percent is not None:
                sc.dfsph.divergence.maxErrorPercent = args.df_error_percent
            if args.iisph_relaxation is not None:
                sc.iisph.relaxationFactor = args.iisph_relaxation
            sc.iisph.validate()
            sc.dfsph.validate()
        return replace(case, configureScheme=config)

    prepared.prepared_case = adapt
    try:
        worker(out, row)
    finally:
        prepared.prepared_case = original


if __name__ == '__main__':
    main()
