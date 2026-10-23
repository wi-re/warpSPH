#!/usr/bin/env python3
"""Plan, selectively execute and report an append-only compressible matrix.

No simulation runs at import, plan or report time. Each selected cell runs in
its own subprocess, sequentially. See docs/compressible_matrix/README.md.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from compressible_matrix import (CASES, METHODS, DEFAULT_ROOT, REPO, clean, comparison,
    complete, digest, identifier, make_plan, now, read_json, select_rows, snap_endpoint, write_json)


def git_info(root):
    def git(*args):
        run = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return run.stdout.strip() if run.returncode == 0 else None
    commit = git('rev-parse', 'HEAD')
    remote = git('config', '--get', 'remote.origin.url') or ''
    match = re.search(r'github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?$', remote)
    url = f'https://github.com/{match[1]}/commit/{commit}' if match and commit else None
    return dict(commit=commit, commitURL=url, dirty=bool(git('status', '--porcelain')),
                branch=git('branch', '--show-current'))


def source_paths():
    roots = {'warpSPH': REPO}
    for name in ('warpSPHCore', 'warpSPHIntegrators', 'warpSPHPlotting'):
        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            package = Path(spec.origin).resolve().parent
            roots[name] = package.parent.parent if package.parent.name == 'src' else package.parent
    return roots


def source_inventory(roots):
    files = {}
    for name, root in roots.items():
        base = root/'src' if (root/'src').exists() else root/name
        paths = list(base.rglob('*.py'))
        if name == 'warpSPH':
            paths.extend((root/'scripts').glob('*.py'))
        for path in sorted(paths):
            if path.is_file():
                files[name+'/'+str(path.relative_to(root))] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    return files


def capture_sources(directory, roots):
    inventory = source_inventory(roots)
    write_json(directory/'sourceHashes.json', {k: v[1] for k, v in inventory.items()})
    # Numeric owner fields avoid potentially network-backed pwd/grp lookups.
    with tarfile.open(directory/'sources.tar.gz', 'w:gz') as archive:
        for name, (path, _) in inventory.items():
            info = path.stat()
            entry = tarfile.TarInfo(name)
            entry.size, entry.mtime, entry.mode = info.st_size, info.st_mtime, info.st_mode & 0o777
            entry.uid, entry.gid = info.st_uid, info.st_gid
            with path.open('rb') as handle:
                archive.addfile(entry, handle)
    for name, root in roots.items():
        diff = subprocess.run(['git', '-C', str(root), 'diff', 'HEAD', '--binary'], capture_output=True)
        if diff.returncode == 0:
            (directory/(name+'.patch')).write_bytes(diff.stdout)
    return {k: v[1] for k, v in inventory.items()}


def worker(root, row, run_id):
    """Run one cell. Preserve partial history and a terminal result on failure."""
    directory = root/'runs'/run_id
    input_path = directory/'input.json'
    if not input_path.exists() or read_json(input_path) != row or (directory/'result.json').exists():
        raise ValueError('Worker requires a fresh controller-created input record')
    record = dict(schemaVersion=1, runId=run_id, rowId=row['id'], caseId=row['caseId'],
        method=row['method'], inputSHA256=row['inputSHA256'], created=now(), status='running',
        targetTime=None, actualTime=0., nSteps=0, metrics={}, configuration=row['configuration'])
    write_json(directory/'result.json', record)
    started = time.perf_counter()
    roots, before = {}, {}
    history, result, initial, latest = [], None, None, {}
    plan = read_json(root/'manifest.json')
    history_file = (directory/'history.jsonl').open('w', buffering=1)
    try:
        sys.path.insert(0, str(REPO/'src'))
        from warpSPHBootstrap import bootstrap
        bootstrap(precision=plan['precision'])
        import numpy as np
        import torch
        import warp
        torch.manual_seed(plan['seed'])
        np.random.seed(plan['seed'])
        from warpSPH.cases import importAll
        from warpSPH.runner import RunConfiguration, getCase, resolvedConfiguration
        from warpSPH.modules.timestep import defaultTimestep
        from evaluate_pesph import arrays, diagnostics, metrics
        importAll()
        device = row['configuration']['simulation']['device']
        cuda = torch.device(device).type == 'cuda'
        if cuda and not torch.cuda.is_available():
            raise RuntimeError(f'Requested {device}, but CUDA is unavailable; no CPU fallback was used')
        if cuda:
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        roots = source_paths()
        before = capture_sources(directory, roots)
        provenance = dict(created=now(), python=sys.version, platform=platform.platform(),
            processor=platform.processor(), torch=torch.__version__, warp=warp.__version__,
            device=device, deviceName=torch.cuda.get_device_name(device) if cuda else platform.machine(),
            cuda=torch.version.cuda, precision=plan['precision'],
            torchThreads=torch.get_num_threads(), cpuCount=os.cpu_count(),
            repositories={name: dict(path=str(path), **git_info(path)) for name, path in roots.items()},
            sourceSHA256=digest(before), inputSHA256=row['inputSHA256'],
            timingScope=plan['protocol']['timing'])
        write_json(directory/'provenance.json', provenance)
        configuration = RunConfiguration.fromDict(row['configuration'])
        base = getCase(row['configuration']['case'])
        build = base.buildSystem

        def build_case(ctx):
            if base.name != 'linearWave':
                return build(ctx)
            # Use the same physical travelling-wave masses for every method,
            # rather than the legacy sampler's iterative mass adjustment.
            from warpSPH.caseUtils import sampleLinearWave
            from warpSPH.configurations.meshless import MeshlessConfig
            if isinstance(ctx.schemeConfig, MeshlessConfig):
                from warpSPH.configurations import CompressibleSPHConfig
                from warpSPH.systems.compressibleMonaghan import CompressibleState, CompressibleSystem
                from warpSPH.caseUtils.meshless import toMeshlessSystem
                sc = CompressibleSPHConfig(gamma=ctx.param('gamma'))
                system = sampleLinearWave(ctx.spec.nx, ctx.config, sc, CompressibleState, CompressibleSystem,
                    *(ctx.param(k) for k in ('A', 'lamda', 'c_s', 'rho0', 'gamma')),
                    supportScheme=sc.adaptiveSupportScheme, analyticMass=True)
                return toMeshlessSystem(system, ctx.config, ctx.schemeConfig)
            return sampleLinearWave(ctx.spec.nx, ctx.config, ctx.schemeConfig, ctx.SimulationState, ctx.SimulationSystem,
                *(ctx.param(k) for k in ('A', 'lamda', 'c_s', 'rho0', 'gamma')),
                supportScheme=ctx.schemeConfig.adaptiveSupportScheme, analyticMass=True)

        def timestep(ctx, system):
            # Native scheme dispatch: MFM/MFV have their own signal/CFL policy.
            step = (base.timestep or defaultTimestep)(ctx, system)
            at_end = snap_endpoint(system, float(ctx.spec.tLimit), plan['precision'])
            remaining = 0. if at_end else max(0., float(ctx.spec.tLimit)-float(system.t))
            return torch.clamp(step, max=remaining) if isinstance(step, torch.Tensor) else min(float(step), remaining)

        def warmup_timestep(ctx, system):
            # Warmup has a fixed step count, not a physical endpoint; do not
            # leave it taking zero-size steps after the measurement endpoint.
            return (base.timestep or defaultTimestep)(ctx, system)

        def record_diagnostics(ctx, system):
            nonlocal initial
            latest.update(ctx=ctx, system=system)
            if initial is None:
                ctx.scratch['evalPhase'] = (((system.state.positions[:, 1] > .25) &
                    (system.state.positions[:, 1] < .75)).to(system.state.masses.dtype)
                    if base.name == 'kelvinHelmholtz' else torch.zeros_like(system.state.masses))
                initial = {k: np.array(v, copy=True) for k, v in arrays(ctx, system).items()}
                np.savez_compressed(directory/'initial.npz', **initial)
                ctx.config.dt = timestep(ctx, system)  # physical cap on the first step too
                write_json(directory/'resolved.json', resolvedConfiguration(ctx))
                record.update(targetTime=float(ctx.spec.tLimit), particles=len(system.state.masses))
                if cuda:
                    torch.cuda.synchronize(device)
                latest['loopStart'] = time.perf_counter()
            values = diagnostics(base.diagnostics, ctx, system)
            values.update(t=float(system.t), dt=float(ctx.config.dt))
            history.append(values)
            history_file.write(json.dumps(clean(values), allow_nan=False)+'\n')
            record.update(actualTime=float(system.t), nSteps=len(history)-1)
            # A heartbeat records physical progress even if a later subprocess
            # crashes or times out. No per-step source scans or state exports.
            if len(history) == 1 or len(history) % 100 == 0:
                write_json(directory/'result.json', record)
            if cuda:
                torch.cuda.synchronize(device)
            latest['loopEnd'] = time.perf_counter()
            if len(history) == 1:
                # First diagnostics can import/compile diagnostic helpers;
                # keep those outside the actual time-stepping wall window.
                latest['loopStart'] = latest['loopEnd']
            return values

        wrapped = replace(base, buildSystem=build_case, diagnostics=record_diagnostics, timestep=timestep)
        with torch.no_grad():
            if not plan['smoke']:
                # Warm an independent run, then rebuild the measured initial
                # state. Never advance the measured particles as a warmup.
                warm = RunConfiguration.fromDict(deepcopy(row['configuration']))
                warm.simulation.update(nSteps=plan['warmupSteps'], progress=False, quiet=True)
                warm_started = time.perf_counter()
                try:
                    warm_result = warm.run(replace(base, buildSystem=build_case, timestep=warmup_timestep,
                        diagnostics=None, setupPlot=None, updatePlot=None))
                    record['warmup'] = dict(steps=warm_result.nSteps, diverged=warm_result.diverged,
                        stopReason=warm_result.stopReason)
                    del warm_result
                except Exception:
                    # Preserve a failed warmup, then evaluate the declared
                    # physical run independently, without parameter changes.
                    record['warmup'] = dict(error=traceback.format_exc())
                if cuda:
                    torch.cuda.synchronize(device)
                record['warmup']['wallSeconds'] = time.perf_counter()-warm_started
                import gc
                gc.collect()
                torch.manual_seed(plan['seed'])
                np.random.seed(plan['seed'])
                if cuda:
                    torch.cuda.reset_peak_memory_stats(device)
            result = configuration.run(wrapped)
        record.update(actualTime=float(result.state.t), targetTime=float(result.ctx.spec.tLimit),
            nSteps=result.nSteps, stopReason=result.stopReason, velocityAlarmEvents=len(result.velocityAlarms))
        positive = all(math_value > 0 for math_value in (
            min(h['minDensity'] for h in history), min(h['minEnergy'] for h in history),
            min(h['minSupport'] for h in history)))
        finite_state = all(math.isfinite(h[k]) for h in history for k in
            ('minDensity', 'minEnergy', 'minPressure', 'minSupport', 'maxVelocity', 'physicalEnergy'))
        final_arrays = arrays(result.ctx, result.state)
        finite_state = finite_state and all(np.isfinite(final_arrays[k]).all() for k in
            ('positions', 'velocities', 'masses', 'supports', 'densities', 'pressures', 'internalEnergies'))
        positive = positive and bool(np.all(final_arrays['masses'] > 0))
        finished = not result.diverged and not result.stopReason and positive and finite_state
        record['admissibility'] = dict(positive=positive, finite=finite_state)
        record['status'] = ('smoke' if plan['smoke'] and finished else 'completed'
            if complete('finished' if finished else 'stopped', record['actualTime'], record['targetTime'])
            else 'stopped')
        times = result.series('stepTime_ms')[1:]
        steady = times[plan['warmupSteps']:]
        # A failed reference/accuracy calculation must not discard separately
        # measured execution costs or turn a completed state into a failure.
        record['metrics'].update(integratorSeconds=float(times.sum()/1000), warmupSteps=plan['warmupSteps'],
            steadyMeanStepMs=float(np.mean(steady)) if len(steady) else None,
            steadyMedianStepMs=float(np.median(steady)) if len(steady) else None,
            peakTorchAllocatedBytes=torch.cuda.max_memory_allocated(device) if cuda else None,
            peakTorchReservedBytes=torch.cuda.max_memory_reserved(device) if cuda else None,
            loopWallSeconds=latest['loopEnd']-latest['loopStart'], totalRunWallSeconds=result.wallTime)
        try:
            record['metrics'].update(metrics(result.ctx, result, initial))
            m = record['metrics']
            if m.get('postShockDensityRelativeError') is not None:
                m['postShockDensityAbsoluteRelativeError'] = abs(m['postShockDensityRelativeError'])
            if base.name == 'kelvinHelmholtz':
                record['diagnosticNotes'] = 'Grid mixing uses initial element labels; MFV has no evolved conservative tracer here.'
        except Exception:
            record['metricsError'] = traceback.format_exc()
        write_json(directory/'history.json', result.trajectory)
    except KeyboardInterrupt:
        record.update(status='interrupted', error='Interrupted by user')
    except Exception:
        record.update(status='error', error=traceback.format_exc())
        print(record['error'], flush=True)
    finally:
        history_file.close()
        if latest:
            try:
                import numpy as np
                from evaluate_pesph import arrays
                np.savez_compressed(directory/'final.npz', **arrays(latest['ctx'], latest['system']))
                write_json(directory/'resolved.json', resolvedConfiguration(latest['ctx']))
            except Exception:
                record['snapshotError'] = traceback.format_exc()
        if before:
            try:
                after = {k: v[1] for k, v in source_inventory(roots).items()}
                record['sourcesStable'] = before == after
                if before != after:
                    write_json(directory/'sourceHashesAfter.json', after)
            except Exception:
                record['sourcesStable'] = False
        record.update(finished=now(), workerWallSeconds=time.perf_counter()-started)
        if result is None and latest:
            record['partialLoopWallSeconds'] = latest.get('loopEnd', time.perf_counter())-latest['loopStart']
        write_json(directory/'result.json', record)
    return 0 if record['status'] in ('completed', 'smoke') else 2


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def execute(root, plan, rows, rerun=False, timeout=14400):
    lock = root/'.run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f'{lock} exists; another controller may be active. Inspect it before removing a stale lock.')
    failed = False
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(f'pid={os.getpid()} host={platform.node()} started={now()}\n')
        selection = read_json(root/'selection.json') if (root/'selection.json').exists() else {}
        for row in rows:
            if row['id'] in selection and not rerun:
                print(f"SKIP {row['id']}: retained attempt {selection[row['id']]} (use --rerun)", flush=True)
                continue
            run_id = identifier()
            directory = root/'runs'/run_id
            directory.mkdir(parents=True)
            write_json(directory/'input.json', row)
            print(f"RUN {row['id']} -> {directory}", flush=True)
            interrupted = False
            with (directory/'worker.log').open('w') as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                    '--root', str(root), '--row', row['id'], '--run-id', run_id],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                terminal = None
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    terminate(process)
                    terminal = 'timeout'
                except KeyboardInterrupt:
                    terminate(process)
                    terminal, interrupted = 'interrupted', True
            path = directory/'result.json'
            record = read_json(path) if path.exists() else dict(runId=run_id, rowId=row['id'],
                caseId=row['caseId'], method=row['method'], inputSHA256=row['inputSHA256'], metrics={})
            if terminal or record.get('status') in (None, 'running'):
                record.update(status=terminal or 'error', returnCode=process.returncode, finished=now())
                write_json(path, record)
            selection[row['id']] = run_id
            write_json(root/'selection.json', selection)
            print(f"{record['status'].upper()} {row['id']} (log: {directory/'worker.log'})", flush=True)
            failed |= record['status'] not in ('completed', 'smoke')
            if interrupted:
                return 130
    finally:
        lock.unlink()
    return 2 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('list', 'plan', 'check', 'run', 'report', 'select', 'worker'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--cases', nargs='+', help='Case IDs; omitted = all cases in the saved plan.')
    parser.add_argument('--methods', '--schemes', nargs='+', help='Preset IDs; omitted = all methods in the saved plan.')
    parser.add_argument('--presets', type=Path, help='Plan only: JSON object of named presets with base/solver/simulation/params overrides.')
    parser.add_argument('--precision', choices=('float32', 'float64'), default='float32', help='Plan only.')
    parser.add_argument('--device', default='cuda:0', help='Plan only; unavailable CUDA never falls back to CPU.')
    parser.add_argument('--smoke', action='store_true', help='Plan only: three coarse steps, excluded from endpoint rankings.')
    parser.add_argument('--rerun', action='store_true', help='Run: append new attempts even for already selected cells.')
    parser.add_argument('--timeout', type=float, default=14400, help='Run: maximum seconds per cell (default 4 hours).')
    parser.add_argument('--include', type=Path, nargs='+', default=[], help='Report: additional roots with distinct trial preset IDs.')
    parser.add_argument('--comparison', type=Path, help='Report: regenerate an explicit saved comparison instead of current selection.')
    parser.add_argument('--out', type=Path, help='Report output directory (must be new).')
    parser.add_argument('--row', help=argparse.SUPPRESS)
    parser.add_argument('--run-id', help='Select: retained run ID. Worker: internal attempt ID.')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.action == 'list':
            for label, catalog in (('Cases', CASES), ('Methods', METHODS)):
                print(label+':')
                for key, item in catalog.items():
                    print(f"  {key:20} {item['label'].replace(chr(10), ' ')}")
            return 0
        if args.action == 'plan':
            if (root/'manifest.json').exists():
                raise ValueError('Plan already exists. Reuse it with run, or choose a new root for changed inputs.')
            plan = make_plan(args.cases, args.methods, read_json(args.presets) if args.presets else None,
                             args.precision, args.device, args.smoke)
            write_json(root/'manifest.json', plan)
            write_json(root/'selection.json', {})
            print(f"Planned {len(plan['rows'])} cells: {root/'manifest.json'}; no simulations started.")
            return 0
        if args.action == 'report':
            if args.comparison and (args.cases or args.methods or args.include):
                raise ValueError('--comparison already specifies the cells; omit filters and --include')
            view = read_json(args.comparison) if args.comparison else comparison(root, args.cases, args.methods, args.include)
            from plot_compressibleMatrix import report
            destination = args.out or root/'comparisons'/view['id']
            report(view, destination)
            print(f'Report: {destination.resolve()}/index.html')
            return 0
        plan = read_json(root/'manifest.json')
        rows = select_rows(plan, args.cases, args.methods)
        if args.action == 'worker':
            row = next(r for r in plan['rows'] if r['id'] == args.row)
            return worker(root, row, args.run_id)
        if args.action == 'run':
            if args.presets or args.smoke or args.device != 'cuda:0' or args.precision != 'float32':
                raise ValueError('Execution uses the saved plan; device, precision, smoke and presets belong to plan.')
            if not math.isfinite(args.timeout) or args.timeout <= 0:
                raise ValueError('--timeout must be positive')
            return execute(root, plan, rows, args.rerun, args.timeout)
        if args.action == 'select':
            if len(rows) != 1 or not args.run_id or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.run_id):
                raise ValueError('Select exactly one --case and --method, and provide a valid --run-id')
            record = read_json(root/'runs'/args.run_id/'result.json')
            if record['inputSHA256'] != rows[0]['inputSHA256']:
                raise ValueError('Run does not belong to the selected planned input')
            selected = read_json(root/'selection.json')
            selected[rows[0]['id']] = args.run_id
            write_json(root/'selection.json', selected)
            return 0
        if args.action == 'check':
            sys.path.insert(0, str(REPO/'src'))
            from warpSPHBootstrap import bootstrap
            bootstrap(precision=plan['precision'])
            from warpSPH.cases import importAll
            from warpSPH.runner import RunConfiguration
            importAll()
            failed = False
            for row in rows:
                try:
                    # Use CPU for metadata preparation: this does not sample or step.
                    config = RunConfiguration.fromDict(row['configuration'])
                    config.simulation['device'] = 'cpu'
                    config.prepare()
                    print('VALID '+row['id'])
                except Exception as exc:
                    failed = True
                    print(f"INVALID {row['id']}: {exc}")
            return 2 if failed else 0
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
