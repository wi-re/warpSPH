#!/usr/bin/env python3
"""Exact-history Lobovsky checkpoint capture and bounded precision branches.

The unchanged frozen comparison fixture owns the physical step, video and
watchdogs. This wrapper saves/restores all ACSPH system fields, adjacency,
domain and non-callable rigid-body fields. FP64 promotes the captured FP32
inputs; it cannot undo error accumulated before the checkpoint.
"""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--end-step', type=int, required=True)
    ap.add_argument('--precision', choices=['float32', 'float64'], required=True)
    ap.add_argument('--restart', type=Path)
    ap.add_argument('--checkpoint-steps', type=int, nargs='*', default=[])
    ap.add_argument('--plot-interval', type=int, default=10)
    args = ap.parse_args()
    if args.end_step < 1:
        ap.error('--end-step must be positive')
    runtime = args.runtime.resolve()
    sys.path[:0] = [str(runtime/'src'), str(runtime/'scripts/acsph_reference')]
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=args.precision)
    import torch
    from diagnostic_state import map_tensors
    import warpSPH.runner as runner
    original_run = runner.run
    baseline = [json.loads(s) for s in (args.baseline/'progress.jsonl').read_text().splitlines()]
    saved = torch.load(args.restart, map_location='cpu', weights_only=False) if args.restart else None
    if saved:
        assert saved['schema'] == 1
        assert saved['sourceControls'] == dict(wallVelocity='source-consistent',
            ghostSpacing='volume', wallBounceRamp=.5)
    start_step = saved['step'] if saved else 0
    if args.end_step <= start_step or args.end_step >= len(baseline):
        ap.error('The bounded endpoint must follow the restart and fit the baseline')
    for k in args.checkpoint_steps:
        if not start_step < k <= args.end_step:
            ap.error('Checkpoint step must lie inside the requested interval')
    reference_rows = json.loads((args.reference/'result.json').read_text())['steps']
    dt = [x['dt'] for x in reference_rows]
    audit = dict(precision=args.precision, startStep=start_step, endStep=args.end_step,
                 restart=str(args.restart.resolve()) if args.restart else None,
                 scope='Precision branch from promoted FP32 state/history' if saved else 'Exact FP32 prefix',
                 baseline=str(args.baseline.resolve()), checkpoints=[])
    checkpoint_dir = args.out/'checkpoints'
    parity_rows = []
    system_fields = ('state', 'adjacency', 'domain', 't', 'positionsPrev', 'velocitiesPrev',
                     'positionsPrev2', 'velocitiesPrev2', 'dtPrev', 'positionStep', 'positionLow')

    def capture(ctx, system, step):
        checkpoint_dir.mkdir(exist_ok=True)
        bodies = [{f.name: getattr(body, f.name) for f in fields(body)
                   if not callable(getattr(body, f.name))} for body in ctx.schemeConfig.rigidBodies]
        payload = dict(schema=1, step=step,
            system={name: getattr(system, name) for name in system_fields},
            configDomain=ctx.config.domain, bodies=bodies, precision=args.precision,
            runtime=str(runtime), sourceControls=dict(wallVelocity='source-consistent',
                ghostSpacing='volume', wallBounceRamp=.5))
        path = checkpoint_dir/f'step-{step:06d}.pt'
        torch.save(map_tensors(payload, device='cpu'), path)
        audit['checkpoints'].append(dict(step=step, path=str(path.resolve()), bytes=path.stat().st_size))
        print('CHECKPOINT', step, float(system.t), path, flush=True)

    def physical_snapshot(system, index):
        import numpy as np
        import struct
        st = system.state
        keep = st.kinds != 2
        arrays = {name: getattr(st, name)[keep].detach().cpu().numpy() for name in
                  ('positions', 'velocities', 'pressures', 'densities', 'masses', 'kinds')}
        n = len(arrays['positions'])
        path = args.out/'snapshots'/f'{index:06d}.bin'
        with path.open('wb') as out:
            out.write(struct.pack('<id', n, float(system.t)))
            for name in ('positions', 'pressures', 'densities', 'velocities'):
                out.write(arrays[name].astype('<f8').tobytes())
            out.write(np.where(arrays['kinds']==0, 1, 3).astype('<i4').tobytes())
            for array in (np.zeros(n), arrays['masses']/arrays['densities'], np.zeros(n)):
                out.write(array.astype('<f8').tobytes())

    def run(case, **kwargs):
        original_initial = case.initialConditions
        original_post = case.postStep
        original_diagnostics = case.diagnostics
        count = [start_step]

        def initialize(ctx, system):
            original_initial(ctx, system)
            if saved:
                dtype = getattr(torch, args.precision)
                restored = map_tensors(saved, device=ctx.device, dtype=dtype)
                for name in system_fields:
                    setattr(system, name, restored['system'][name])
                ctx.config.domain = restored['configDomain']
                assert len(restored['bodies']) == len(ctx.schemeConfig.rigidBodies)
                for body, record in zip(ctx.schemeConfig.rigidBodies, restored['bodies']):
                    for name, value in record.items():
                        setattr(body, name, value)
                ctx.scratch['completed'] = start_step
                ctx.config.dt = dt[start_step]
                # Keep construction qualification separate from restart data.
                initial_path = args.out/'snapshots/000000.bin'
                initial_path.rename(args.out/'construction-initial.bin')
                physical_snapshot(system, start_step)
                audit['restartTime'] = float(system.t)
                audit['restartBDF'] = list(system.bdfCoefficients(dt[start_step]))
                audit['promotedInputFieldsExact'] = all(
                    torch.equal(getattr(system, name).cpu(), saved['system'][name].to(dtype))
                    for name in system_fields if isinstance(saved['system'][name], torch.Tensor))
                audit['promotedParticleFieldsExact'] = all(
                    torch.equal(getattr(system.state, f.name).cpu(),
                        getattr(saved['system']['state'], f.name).to(dtype if getattr(system.state, f.name).is_floating_point() else getattr(system.state, f.name).dtype))
                    for f in fields(system.state) if isinstance(getattr(system.state, f.name), torch.Tensor))
                assert audit['promotedInputFieldsExact'] and audit['promotedParticleFieldsExact']
                print('RESTORED', start_step, float(system.t), args.precision, flush=True)
            (args.out/'section-audit.json').write_text(json.dumps(audit, indent=2)+'\n')

        def post(ctx, system, step):
            original_post(ctx, system, step)
            absolute = step+1
            count[0] = absolute
            if absolute in args.checkpoint_steps:
                capture(ctx, system, absolute)
            if absolute == args.end_step:
                physical_snapshot(system, absolute)

        def diagnostics(ctx, system):
            row = original_diagnostics(ctx, system)
            absolute = count[0]
            if args.precision == 'float32':
                expected = baseline[absolute]
                check = {k: row[k] == expected[k] for k in row}
                if absolute == start_step:
                    # At a restart initial diagnostics have no step update yet.
                    check.pop('pseudoIterations', None)
                    check.pop('epsilonV', None)
                check['t'] = float(system.t) == expected['t']
                record = dict(step=absolute, matches=all(check.values()), fields=check)
                parity_rows.append(record)
                with (args.out/'baseline-parity.jsonl').open('a') as log:
                    log.write(json.dumps(record)+'\n')
                if not record['matches']:
                    raise AssertionError('FP32 baseline/restart mismatch: '+json.dumps(record))
            return row

        kwargs.update(nSteps=args.end_step-start_step, dt=dt[start_step],
                      tLimit=reference_rows[args.end_step-1]['t'], resumeStepOffset=start_step)
        result = original_run(replace(case, initialConditions=initialize, postStep=post,
                                      diagnostics=diagnostics), **kwargs)
        # The fixture's post hook uses absolute step numbers. Prevent its final
        # fallback from adding a duplicate snapshot under the local step count.
        result.ctx.scratch['last_snapshot_step'] = result.nSteps
        return result

    runner.run = run
    script = runtime/'scripts/acsph_reference/compare_lobovsky.py'
    sys.argv = [str(script), '--reference', str(args.reference.resolve()), '--out', str(args.out.resolve()),
        '--steps', str(args.end_step), '--long-run', '--plot-interval', str(args.plot_interval),
        '--precision', args.precision, '--position-form', 'increment', '--fused-residual',
        '--coefficients', 'mapped', '--history', 'source', '--wall-diffusion', 'source',
        '--acoustic-damper', 'source', '--rk-treatment', 'source', '--wall-velocity', 'source-consistent',
        '--wall-bounce-ramp', '0.5', '--ghost-spacing', 'volume']
    started = time.monotonic()
    runpy.run_path(str(script), run_name='__main__')
    result = json.loads((args.out/'result.json').read_text())
    assert result['nSteps'] == args.end_step-start_step and not result['diverged'] and not result['stop_reason']
    progress = [json.loads(s) for s in (args.out/'progress.jsonl').read_text().splitlines()]
    assert abs(progress[-1]['t']-baseline[args.end_step]['t']) < 1e-12
    timing = [json.loads(s) for s in (args.out/'solve-timing.jsonl').read_text().splitlines()]
    assert len(timing) == result['nSteps'] and all(r['captured'] and not r['disabled'] for r in timing)
    audit.update(status='passed', wallSeconds=time.monotonic()-started,
                 physicalSteps=result['nSteps'], graphSteps=len(timing), endpointTime=progress[-1]['t'],
                 msPerStep=1000*result['wall_seconds']/result['nSteps'],
                 baselineParityRows=len(parity_rows),
                 baselineParityAll=all(r['matches'] for r in parity_rows) if parity_rows else None)
    (args.out/'section-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('SECTION COMPLETE', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
