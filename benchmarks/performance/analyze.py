#!/usr/bin/env python3
"""Write CSV/Markdown comparisons, pairing only equivalent benchmark requests.

Usage: python benchmarks/performance/analyze.py results/perf/3090 results/perf/gh200 --output results/perf/comparison
Speedup is first-run wall time / comparison-run wall time (greater than one is faster).
"""
import argparse
import csv
import json
from pathlib import Path


def key(request, *, allow_backend_change=False):
    # Runtime sweep controls do not define numerical work. Device deliberately
    # differs across comparisons. Warmup age, CPU threads, index widths, timestep
    # and the full input configuration remain part of the identity.
    ignored = {'device', 'repeats', 'profile', 'memory_mib', 'steps', 'calls', 'warmup',
               'particle_order', 'verlet_scale', 'configuration', 'neighbors', 'kernel',
               'lanes', 'traversal', 'dim', 'nx'}
    kept = {k: v for k, v in request.items() if k not in ignored}
    kept['operator_backend'] = request.get('operator_backend', request['backend'])
    if allow_backend_change:
        kept.pop('backend', None)
        kept.pop('operator_backend', None)
    if request['kind'] == 'memory':
        kept.update(memory_mib=request['memory_mib'], calls=request['calls'], warmup=request['warmup'])
    else:
        kept.update(nx=request['nx'], dim=request['dim'], kernel=request['kernel'],
                    neighbors=request['neighbors'], warmup=request['warmup'])
        if request['kind'] in ('steps', 'rhs'):
            kept.update(steps=request['steps'], particle_order=request['particle_order'],
                        verlet_scale=request['verlet_scale'], configuration=request['configuration'], lanes=request['lanes'])
        else:
            kept.update(calls=request['calls'], lanes=request['lanes'], traversal=request['traversal'], particle_order=request.get('particle_order', 'input'))
    return json.dumps(kept, sort_keys=True, separators=(',', ':'))


def load(path):
    path = path/'results.json' if path.is_dir() else path
    document = json.loads(path.read_text())
    if document.get('schema_version') != 1 or not isinstance(document.get('results'), list):
        raise ValueError(f'{path}: expected schema_version=1 aggregate results')
    return document['results']


def flatten(label, result):
    r, timing = result['request'], result.get('timing', {})
    machine = result.get('machine', {})
    wall = timing.get('wall_ms') or {}
    stream = timing.get('stream_ms') or {}
    rhs_host = timing.get('rhs_host_ms') or {}
    rhs_stream = timing.get('rhs_stream_ms') or {}
    samples = timing.get('samples', [])
    work = result.get('workload', {})
    peaks = [row['peak_allocated_bytes'] for row in timing.get('peak_memory_by_repetition', [])
             if row.get('peak_allocated_bytes') is not None]
    return dict(run=label, status=result['status'], kind=r['kind'], operation=r['operation'],
                precision=r['precision'], device=r['device'], threads=r['threads'], nx=r['nx'], dim=r['dim'],
                backend=machine.get('backend', r['backend']), lanes=r['lanes'], traversal=r['traversal'],
                operator_backend=machine.get('operator_backend', r.get('operator_backend', r['backend'])),
                particle_order=r.get('particle_order', 'input'), execution=r.get('execution', 'eager'),
                neighborhood=r.get('neighborhood', 'csr'), iteration_checks=r.get('iteration_checks'),
                effective_compute=json.dumps(work.get('effective_compute', {}), sort_keys=True),
                wall_median_ms=wall.get('median'), wall_min_ms=wall.get('min'), wall_max_ms=wall.get('max'),
                stream_median_ms=stream.get('median'), rhs_host_median_ms=rhs_host.get('median'),
                rhs_stream_median_ms=rhs_stream.get('median'), repetitions=len(samples),
                directed_edges=work.get('directed_edges_including_self', work.get('final_directed_edges_including_self')), particles=work.get('particles'),
                logical_GBps_wall=result.get('logical_GBps_wall'), logical_GBps_stream=result.get('logical_GBps_stream'),
                peak_allocated_bytes=timing.get('peak_allocated_bytes', max(peaks) if peaks else None),
                host=machine.get('host'), hardware=(machine.get('gpu') or {}).get('name', machine.get('cpu_model')),
                error=result.get('error'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs='+')
    p.add_argument('--labels', nargs='+', help='one unique name per input run, in the same order')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-backend-change', action='store_true', help='explicitly compare different backends with identical requests')
    args = p.parse_args(argv)
    labels = args.labels or [path.stem for path in args.runs]
    if len(labels) != len(args.runs) or len(set(labels)) != len(labels):
        p.error('--labels must provide a unique name per run')
    datasets = [load(path) for path in args.runs]
    rows = [flatten(label, row) for label, data in zip(labels, datasets) for row in data]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'measurements.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['run', 'status'])
        writer.writeheader(); writer.writerows(rows)
    indexed = []
    for label, dataset in zip(labels, datasets):
        index = {}
        for row in dataset:
            k = key(row['request'], allow_backend_change=args.allow_backend_change)
            if k in index:
                raise ValueError(f'{label}: duplicate comparable request; do not combine multiple runs into one aggregate')
            index[k] = row
        indexed.append(index)
    lines = ['# Single-device performance', '',
             'Wall time is completed work per call/accepted step. CUDA stream intervals include launch gaps; '
             'they are not pure kernel time. Memory bandwidth is logical traffic, not measured DRAM bandwidth. '
             'RHS rows describe separately instrumented trajectories.', '',
             '| Run | Successful | Failed | Unsupported | Capacity skipped |', '|---|---:|---:|---:|---:|']
    for label, data in zip(labels, datasets):
        counts = [sum(row['status'] == status for row in data) for status in ('ok', 'failed', 'unsupported', 'skipped_capacity')]
        lines.append('| '+label+' | '+' | '.join(map(str, counts))+' |')
    lines += ['', '| Run | Workload | Storage / execution / topology | Precision / threads | nx | Wall median [min, max] ms | Stream median ms |',
              '|---|---|---|---|---:|---:|---:|']
    def fmt(value):
        return f'{value:.5g}' if value is not None else '—'
    for row in rows:
        if row['status'] == 'ok':
            lines.append(f'| {row["run"]} | {row["kind"]}/{row["operation"]}/{row["traversal"]}/L{row["lanes"]} | '
                         f'{row["particle_order"]} / {row["execution"]} / {row["neighborhood"]} | '
                         f'{row["precision"]} / {row["threads"]} | {row["nx"] or "—"} | '
                         f'{fmt(row["wall_median_ms"])} [{fmt(row["wall_min_ms"])}, {fmt(row["wall_max_ms"])}] | '
                         f'{fmt(row["stream_median_ms"])} |')
    comparisons = []
    if any(row['status'] == 'ok' and row['kind'] == 'rhs' for row in rows):
        lines += ['', '## Instrumented per-RHS intervals', '',
                  '| Run | Solver / nx | Host median ms | Stream median ms |', '|---|---|---:|---:|']
        for row in rows:
            if row['status'] == 'ok' and row['kind'] == 'rhs':
                lines.append(f'| {row["run"]} | {row["operation"]} / {row["nx"]} | '
                             f'{fmt(row["rhs_host_median_ms"])} | {fmt(row["rhs_stream_median_ms"])} |')
    if len(datasets) > 1:
        lines += ['', f'Ratios below use **{labels[0]} / comparison**. Greater than one means the comparison run is faster. '
                  'Only identical numerical requests and timing windows are paired; CPU thread counts must match. '
                  'Check the source revisions, library versions, hardware and clocks in the raw JSON before attributing a difference.', '',
                  '| Comparison | Workload | Storage / execution / topology | nx | Precision / threads | Wall speedup | Stream speedup |',
                  '|---|---|---|---:|---|---:|---:|']
        for label, index in zip(labels[1:], indexed[1:]):
            for k in sorted(indexed[0].keys() & index.keys()):
                a, b = indexed[0][k], index[k]
                if a['status'] != 'ok' or b['status'] != 'ok':
                    continue
                at, bt = a['timing'], b['timing']
                wall_ratio = at['wall_ms']['median'] / bt['wall_ms']['median']
                stream_ratio = (at['stream_ms']['median'] / bt['stream_ms']['median']) if at.get('stream_ms') and bt.get('stream_ms') else None
                comparison = dict(baseline=labels[0], comparison=label, request=json.loads(k),
                                  wall_speedup=wall_ratio, stream_speedup=stream_ratio)
                if a['request']['kind'] == 'rhs':
                    comparison.update(rhs_host_speedup=at['rhs_host_ms']['median']/bt['rhs_host_ms']['median'],
                                      rhs_stream_speedup=at['rhs_stream_ms']['median']/bt['rhs_stream_ms']['median']
                                      if at.get('rhs_stream_ms') and bt.get('rhs_stream_ms') else None)
                comparisons.append(comparison)
                r = a['request']
                lines.append(f'| {label} | {r["kind"]}/{r["operation"]}/{r["traversal"]}/L{r["lanes"]} | '
                             f'{r.get("particle_order", "input")} / {r.get("execution", "eager")} / {r.get("neighborhood", "csr")} | {r["nx"] or "—"} | '
                             f'{r["precision"]} / {r["threads"]} | {fmt(wall_ratio)} | {fmt(stream_ratio)} |')
    tuning = []
    tune_keys = ('particle_order', 'execution', 'neighborhood', 'lanes', 'iteration_checks',
                 'krylov_check_every', 'neighbor_omit_i', 'collection', 'query_order', 'traversal', 'edge_budget', 'max_rows')
    lines += ['', '## Compute flag comparisons within each run', '',
              'Same numerical fixture, first successful input/eager/CSR/lane-1 variant when present as baseline. '
              'Ratios are baseline / variant; an unsupported baseline is never replaced silently across machines.', '',
              '| Run | Workload / nx | Baseline flags | Variant flags | Wall speedup |', '|---|---|---|---|---:|']
    for label, dataset in zip(labels, datasets):
        groups = {}
        for row in dataset:
            if row['status'] != 'ok':
                continue
            base_key = json.loads(key(row['request']))
            for name in tune_keys:
                base_key.pop(name, None)
            groups.setdefault(json.dumps(base_key, sort_keys=True), []).append(row)
        for group in groups.values():
            def priority(row):
                r = row['request']
                return (r.get('particle_order') != 'input', r.get('execution') != 'eager',
                        r.get('neighborhood') != 'csr', r['lanes'], json.dumps(r, sort_keys=True))
            group.sort(key=priority)
            baseline = group[0]
            def flags(row):
                return {name: row['request'][name] for name in tune_keys if name in row['request']}
            base_flags = flags(baseline)
            for row in group[1:]:
                different = {k: v for k, v in flags(row).items() if v != base_flags[k]}
                before = {k: base_flags[k] for k in different}
                ratio = baseline['timing']['wall_ms']['median'] / row['timing']['wall_ms']['median']
                r = row['request']
                tuning.append(dict(run=label, baseline=base_flags, variant=flags(row), wall_speedup=ratio,
                                   request=json.loads(key(r))))
                lines.append(f'| {label} | {r["kind"]}/{r["operation"]} / {r["nx"]} | {json.dumps(before, sort_keys=True)} | '
                             f'{json.dumps(different, sort_keys=True)} | {fmt(ratio)} |')
    failures = [r for r in rows if r['status'] != 'ok']
    if failures:
        lines += ['', '## Unmeasured cases', '']
        for r in failures:
            lines.append(f'- {r["run"]} {r["kind"]}/{r["operation"]} nx={r["nx"]}: {r["status"]}: {r["error"]}')
    (args.output/'comparison.json').write_text(json.dumps(comparisons, indent=2, allow_nan=False) + '\n')
    (args.output/'compute_flags.json').write_text(json.dumps(tuning, indent=2, allow_nan=False) + '\n')
    (args.output/'report.md').write_text('\n'.join(lines) + '\n')
    print(args.output/'report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
