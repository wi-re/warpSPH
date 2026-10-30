#!/usr/bin/env python3
"""Reproduce one early cap, then audit same-input position-bounce controls.

The physical prefix returns the original result. Extra target-step solves use
the exact same solver-entry tensors/history and are explicitly diagnostic.
No full trajectory or production-default change is made.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import fields, replace
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--step', type=int, default=56)
    ap.add_argument('--additional-steps', type=int, nargs='*', default=[], help='Audit more real steps in one unchanged physical prefix')
    ap.add_argument('--prefix-steps', type=int, help='Optional bounded physical prefix length, at least --step')
    ap.add_argument('--harness', type=Path, help='Explicit corrected harness; numerical runtime still comes from the frozen campaign')
    ap.add_argument('--reference', type=Path, help='Explicit retained native run when separate from the runtime campaign')
    ap.add_argument('--capture-complete', action='store_true', help='Save full fixed solve topology/configuration and initial tensors for fresh-process precision controls')
    ap.add_argument('--wall-bounce-ramp', type=float, default=0., help='Physical-prefix D5 ramp; retains hard0 by default')
    args = ap.parse_args()
    targets=set([args.step,*args.additional_steps])
    if min(targets)<1:ap.error('diagnostic steps must be positive')
    if args.prefix_steps is not None and args.prefix_steps<max(targets):ap.error('--prefix-steps must reach every diagnostic step')
    campaign = args.campaign.resolve(); out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = campaign/'source'
    sys.path.insert(0, str(frozen/'src'))
    sys.path.insert(0, str(frozen/'scripts/acsph_reference'))
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch
    from warpSPH.modules.artificialCompressible import dualTime
    import warpSPH.schemes.incompressible.artificialCompressible as acscheme
    import performance
    # Our own bounded observations replace retained profile/graph receipts.
    performance.install_light = lambda *a, **kw: None
    original_solve = acscheme.solveDualTime
    count = 0
    reports = {}

    def tensor_cpu(value):
        return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value

    def inspect_iteration(problem, result, index):
        q, v, p, eps = result
        x = dualTime.positionCoordinates(problem, q)
        rate = dualTime.bdfSource(problem, q, v)[0]
        bounced = dualTime.bounceVelocities(problem.bounce, x, v)
        fluid = problem.fluidIndex
        residual = (bounced-rate).index_select(0, fluid)
        raw = (v-rate).index_select(0, fluid)
        e2 = residual.square().sum(-1)
        total = e2.sum()
        raw_eps = dualTime.convergenceMetricDevice(v-rate, v, fluid, problem.nFluid, problem.schemeConfig)
        values, order = torch.topk(e2, min(10, len(e2)))
        top = fluid[order]
        denom = total.clamp_min(1e-30)
        row = {'iteration': index, 'epsilonV': float(eps), 'epsilon_without_metric_bounce': float(raw_eps),
            'residual_l2': float(total.sqrt()), 'raw_position_residual_l2': float(raw.square().sum().sqrt()),
            'top1_fraction': float(values[0]/denom), 'top10_fraction': float(values.sum()/denom),
            'metric_bounced_fluid_rows': int(((bounced-v).square().sum(-1).index_select(0, fluid)>0).sum()),
            'top_rows': top.cpu().tolist(), 'top_uids': problem.state.UIDs[top].cpu().tolist(),
            'top_positions': x[top].cpu().tolist(), 'top_velocities': v[top].cpu().tolist(),
            'top_bdf_rates': rate[top].cpu().tolist(), 'top_residuals': (bounced-rate)[top].cpu().tolist()}
        row['top_pressures'] = p[top].cpu().tolist()
        row['top_neighbor_counts_verlet'] = problem.adjacency.numNeighbors[top].cpu().tolist()
        if problem.bounce is not None:
            bounce = problem.bounce
            selected = torch.isin(bounce.pi, top)
            pi, pj = bounce.pi[selected], bounce.pj[selected]
            displacement = x[pi]-x[pj]
            distance = (displacement*bounce.normals[pj]).sum(-1).abs()
            near = displacement.norm(dim=-1)<1.155*bounce.dx
            wall_rows = []
            for particle in top:
                mask = (pi==particle)&near
                best = distance[mask].min() if bool(mask.any()) else x.new_tensor(float('inf'))
                normal = bounce.normals[pj[(pi==particle)&near&(distance<=best)]].sum(0)
                normal = normal/normal.norm().clamp_min(1e-30)
                wall_rows.append(dict(distance_dx=float(best/bounce.dx),
                    threshold_margin_dx=float(best/bounce.dx-bounce.distance),
                    normal=normal.cpu().tolist(),normal_velocity=float((v[particle]*normal).sum()),
                    metric_velocity=bounced[particle].cpu().tolist()))
            row['top_wall_contact'] = wall_rows
        if index == 199:
            ordered = e2.sort().values
            ncut = round(.95*len(e2))
            vmax = v.index_select(0, fluid).norm(dim=-1).max()
            ueps = vmax.clamp(min=problem.schemeConfig.acParams.epsilonS,
                max=problem.schemeConfig.acParams.uChar)
            row['trimmed95_epsilonV'] = float(torch.log10(ordered[:ncut].sum().sqrt()/(ncut*ueps)))
        return row

    def solve(problem, x0, v0, p0, verbose=False):
        nonlocal count
        count += 1
        start = time.monotonic()
        if count not in targets:
            result = original_solve(problem, x0, v0, p0, verbose)
        else:
            target_out=out if not args.additional_steps else out/f'step-{count}'
            if args.additional_steps:target_out.mkdir()
            reports={}
            state = copy.copy(problem.state)
            for f in fields(state):
                setattr(state, f.name, tensor_cpu(getattr(state, f.name)))
            captured = {'state': state, 'history': tuple(map(tensor_cpu, problem.history)),
                'bdf': problem.bdf, 'dtau': problem.dtau,
                'positionLow': tensor_cpu(problem.positionLow),
                'previousPositionStep': tensor_cpu(problem.previousPositionStep),
                'bodyForce': tensor_cpu(problem.bodyForce), 'wallBodyForce': tensor_cpu(problem.wallBodyForce),
                'k1': tensor_cpu(problem.k1), 'k2': tensor_cpu(problem.k2), 'nu': tensor_cpu(problem.nu)}
            torch.save(captured, target_out/'solver-entry.pt')
            if args.capture_complete:
                from diagnostic_state import map_tensors, solve_payload
                payload = solve_payload(problem, x0, v0, p0)
                torch.save(map_tensors(payload, device='cpu'), target_out/'complete-solver-entry.pt')
                print('COMPLETE SOLVER ENTRY saved, including original neighbor ordering/history', flush=True)
            result = original_solve(problem, x0, v0, p0, verbose)
            reports['original_graph'] = {'iterations': result.iterations, 'epsilonV': result.epsilonV}
            torch.save({name: tensor_cpu(getattr(result, name)) for name in
                ('positions', 'velocities', 'pressures', 'positionStep', 'positionLow')},target_out/'original-graph-result.pt')
            print('TARGET ORIGINAL', count, json.dumps(reports['original_graph']), flush=True)
            original_iterate = dualTime.iterate  # includes the explicit source-RK diagnostic
            active = {'label': '', 'iteration': 0}
            def observe(pr, *inputs):
                values = original_iterate(pr, *inputs)
                index = active['iteration']; active['iteration'] += 1
                row = inspect_iteration(pr, values, index)
                with (target_out/(active['label']+'.jsonl')).open('a') as log:
                    log.write(json.dumps(row)+'\n')
                if index % 25 == 0:
                    print('RESIDUAL', active['label'], index, row['epsilonV'],
                        'top10 fraction', row['top10_fraction'], flush=True)
                return values
            dualTime.iterate = observe
            try:
                controls = ([('hard-eager', problem.bounce), ('bounce-off', None),
                             ('smooth-half', replace(problem.bounce, ramp=.5))]
                            if args.wall_bounce_ramp == 0 else
                            [('base-eager', problem.bounce), ('bounce-off', None),
                             ('hard-switch', replace(problem.bounce, ramp=0.)),
                             ('smooth-full', replace(problem.bounce, ramp=1.))])
                for label, bounce in controls:
                    sc = copy.copy(problem.schemeConfig)
                    sc.acParams = copy.copy(problem.schemeConfig.acParams)
                    sc._acsphGraphDisabled = 'target-step per-particle residual observations'
                    trial = replace(problem, schemeConfig=sc, bounce=bounce)
                    active.update(label=label, iteration=0)
                    tick = time.monotonic()
                    candidate = original_solve(trial, x0.clone(), v0.clone(), p0.clone(), False)
                    report = {'iterations': candidate.iterations, 'epsilonV': candidate.epsilonV,
                        'wall_seconds': time.monotonic()-tick,
                        'position_max_difference': float((candidate.positions-result.positions).abs().max()),
                        'velocity_max_difference': float((candidate.velocities-result.velocities).abs().max()),
                        'pressure_rms_difference': float((candidate.pressures-result.pressures).square().mean().sqrt())}
                    reports[label] = report
                    print('TARGET TRIAL', count, label, json.dumps(report), flush=True)
                    torch.save({name: tensor_cpu(getattr(candidate, name)) for name in
                        ('positions', 'velocities', 'pressures', 'positionStep', 'positionLow')},target_out/(label+'-result.pt'))
            finally:
                dualTime.iterate = original_iterate
            (target_out/'target-summary.json').write_text(json.dumps(reports, indent=2)+'\n')
        row = {'step': count, 'iterations': result.iterations, 'epsilonV': result.epsilonV,
            'wall_seconds': time.monotonic()-start}
        with (out/'prefix-step-receipts.jsonl').open('a') as log:
            log.write(json.dumps(row)+'\n')
        return result
    acscheme.solveDualTime = solve
    script = args.harness.resolve() if args.harness else frozen/'scripts/acsph_reference/compare_lobovsky.py'
    if args.harness:
        sys.path.insert(0, str(script.parent))
    reference = args.reference.resolve() if args.reference else campaign/'native-f64'
    sys.argv = [str(script), '--reference', str(reference), '--out', str(out/'prefix'),
        '--steps', str(args.prefix_steps or max(targets)), '--long-run', '--plot-interval', '10', '--precision', 'float32',
        '--position-form', 'increment', '--fused-residual', '--coefficients', 'mapped', '--history', 'source',
        '--wall-diffusion', 'source', '--acoustic-damper', 'source', '--rk-treatment', 'source', '--wall-velocity', 'source']
    if args.wall_bounce_ramp != 0:
        sys.argv += ['--wall-bounce-ramp', str(args.wall_bounce_ramp)]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
