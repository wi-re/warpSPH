#!/usr/bin/env python3
"""Isolated production-runner accepted-step and instrumented RHS benchmarks."""
import argparse
import gc
from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import wraps
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Source checkout is explicitly recorded by the orchestrator, not an alternate import of core.
sys.path.insert(0, str(Path(os.environ['WARPSPH_BENCH_CORE_ROOT']) / 'scripts/benchmarks'))
from _single_device import configure, metadata, profile_summary, summarize, synchronize, write_result


def configuration(request):
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase
    from warpSPH.cases.periodic.tgv import tgvCase
    from warpSPH.cases.periodic.tgvWeaklyCompressible import tgvWeaklyCompressibleCase
    scheme = request['operation']
    if scheme in ('MFM', 'MFV') and request['particle_order'] == 'morton':
        raise NotImplementedError('MFM/MFV do not yet have an audited single-device Morton layout adapter')
    if request.get('configuration'):
        conf = RunConfiguration.fromDict(request['configuration'])
        from warpSPH.runner.configuration import _registeredCase
        case = _registeredCase(conf.case)
    elif scheme in ('deltaSPH', 'sun2017DeltaSPH', 'artificialCompressible'):
        case = tgvWeaklyCompressibleCase
        conf = RunConfiguration(case=case.name, params=dict(shuffleIters=0, jitter=0., shuffleSeed=173, shifting=False))
    elif scheme in ('iisph', 'dfsph', 'dfsphPredictor', 'dfsphShift'):
        case = tgvCase
        conf = RunConfiguration(case=case.name, params=dict(relaxSteps=0, jitter=0., shuffleSeed=173, shifting=False))
    else:
        case = drivenTurbulenceCase
        conf = RunConfiguration(case=case.name)
    if case.name in ('tgv', 'tgv-wc') and request['dim'] != 2:
        raise NotImplementedError(f'{case.name} fixture is 2D; use --dim 2 or a supported --configuration')
    if case.name == 'drivenTurbulence' and request.get('execution', 'eager') != 'eager':
        raise NotImplementedError('drivenTurbulence supports eager execution only; use another --configuration for graph tests')
    if case.name == 'drivenTurbulence' and request['particle_order'] == 'morton' and request.get('neighborhood', 'csr') == 'csr':
        raise NotImplementedError('drivenTurbulence Morton/source adapter requires --neighborhood chunked on this revision')
    simulation = dict(conf.simulation)
    if simulation.get('resumeFrom') or simulation.get('resumeStepOffset', 0):
        raise NotImplementedError('benchmark fixtures currently require fresh-start configurations')
    simulation.update(nx=request['nx'], dim=request['dim'], device=request['device'],
                      precision=request['precision'], nSteps=request['warmup']+request['steps'],
                      plot=False, video=False, store=False, show=False, holdPlot=False, progress=False, quiet=True,
                      execution=request.get('execution', 'eager'), cudaGraph=None, pipelineOutputs=False,
                      iterationChecks=request.get('iteration_checks', 'adaptive'),
                      krylovCheckEvery=request.get('krylov_check_every'), neighborOmitI=request.get('neighbor_omit_i'),
                      velocityAlarmFactor=None, particleOrder=request['particle_order'], kernel=request['kernel'])
    if request['verlet_scale'] is not None:
        simulation['verletScale'] = request['verlet_scale']
    solver = dict(conf.solver)
    solver['scheme'] = scheme
    if scheme in ('MFM', 'MFV') and not request.get('configuration'):
        solver['temporalScheme'] = 'ssprk'
    conf = replace(conf, simulation=simulation, solver=solver)
    spec = conf.resolve(case)
    return case, spec


def trajectory(request, device, *, instrument_rhs=False, profile_path=None):
    import torch
    from warpSPH.runner import run
    case, spec = configuration(request)
    if instrument_rhs and (request['particle_order'] != 'input' or request.get('execution', 'eager') != 'eager'):
        raise NotImplementedError('per-RHS callable instrumentation requires input order and eager execution; use full steps/profile for Morton/graph execution')
    from warpSPHCore import setNeighborLanes
    setNeighborLanes(request['lanes'])
    samples, rhs_samples, counts, final_info = [], [], [], {}
    cuda = device.type == 'cuda'
    profiler = None
    if profile_path:
        import warpSPHCore.profiling as hooks
        hooks._torchRecordFunction = torch.profiler.record_function
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities)
    # Same initial field and warmup age for each repetition, never time repeated
    # calls against a progressively mutated frozen state.
    for rep in range(request['repeats']):
        torch.manual_seed(173)
        active = False
        rhs_count = 0
        rhs_start = 0
        event_pairs = []
        window_start = window_end = None
        peak_memory = {}
        start_event = torch.cuda.Event(enable_timing=True) if cuda else None
        end_event = torch.cuda.Event(enable_timing=True) if cuda else None
        if cuda:
            start_event.record(); end_event.record()
            synchronize(device)
        native_build = case.buildSystem

        def build(ctx, **kwargs):
            if hasattr(ctx.schemeConfig, 'stepMode'):
                ctx.schemeConfig.stepMode = request.get('execution', 'eager')
            ctx.config.compute.neighborCollection = request.get('collection', 'auto')
            ctx.config.compute.neighborQueryOrder = request.get('query_order', 'auto')
            if request.get('neighborhood') == 'chunked':
                import inspect
                if 'neighborhoodFactory' not in inspect.signature(native_build).parameters:
                    raise NotImplementedError(f'{case.name} has no chunked construction hook')
                from warpSPHCore.radiusSearch import RowChunkPolicy, buildChunkedVerletList
                def factory(state, config):
                    dim = state.positions.shape[1] if request['operation'] in ('CompSPH', 'CRKSPH') else 1
                    policy = RowChunkPolicy(edgeBudget=request['edge_budget'], maxRows=request['max_rows'],
                        bytesPerEdge=(8 if case.name in ('tgv', 'tgv-wc') else 4) + dim*state.positions.element_size(),
                        maxElementsPerEdge=dim)
                    return buildChunkedVerletList(state, config.domain, policy=policy,
                        supportMode=config.supportMode, verletScale=config.verletScale,
                        storeI=not bool(config.compute.neighborOmitI))
                kwargs['neighborhoodFactory'] = factory
            system = native_build(ctx, **kwargs)
            if not instrument_rhs:
                return system
            native_rhs = ctx.bundle.stepFunction
            @wraps(native_rhs)
            def rhs(*args, **kw):
                nonlocal rhs_count
                if not active:
                    return native_rhs(*args, **kw)
                rhs_count += 1
                a, b = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if cuda else (None, None)
                t0 = time.perf_counter()
                if cuda:
                    a.record()
                with torch.profiler.record_function('benchmark_rhs') if request.get('profile') else nullcontext():
                    value = native_rhs(*args, **kw)
                if cuda:
                    b.record()
                rhs_samples.append(dict(repetition=rep, host_ms=(time.perf_counter()-t0)*1000))
                event_pairs.append((a, b))
                return value
            ctx.bundle = replace(ctx.bundle, stepFunction=rhs)
            return system

        original_accepted = case.acceptedStep
        def accepted(ctx, system, completed):
            nonlocal active, window_start, window_end, rhs_start
            if original_accepted:
                original_accepted(ctx, system, completed)
            if completed == request['warmup']:
                synchronize(device)
                if profiler:
                    profiler.start()
                if cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                rhs_start = len(rhs_samples)
                window_start = time.perf_counter()
                if cuda:
                    start_event.record()
                active = True
            elif completed == request['warmup'] + request['steps']:
                if cuda:
                    end_event.record()
                synchronize(device)
                window_end = time.perf_counter()
                active = False
                peak_memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None)
                if profiler:
                    profiler.stop()

        timed_case = replace(case, buildSystem=build, acceptedStep=accepted, diagnostics=None,
                             extraData=None, trajectoryData=None, setupPlot=None, updatePlot=None)
        result = run(timed_case, spec)
        if result.diverged or result.stopReason or result.nSteps != request['warmup'] + request['steps'] or window_end is None:
            raise RuntimeError(f'trajectory incomplete: {result.nSteps} steps, diverged={result.diverged}, reason={result.stopReason}')
        # Validation outside all windows; validate physical state rather than
        # arbitrary caches with sentinel NaNs in unused entries.
        state = result.state.state
        for name in ('positions', 'velocities', 'densities', 'masses', 'supports', 'internalEnergies'):
            value = getattr(state, name, None)
            if isinstance(value, torch.Tensor):
                if not bool(torch.isfinite(value).all()):
                    raise AssertionError(f'nonfinite final {name}')
                if name in ('densities', 'masses', 'supports', 'internalEnergies') and not bool((value > 0).all()):
                    raise AssertionError(f'nonpositive final {name}')
        if instrument_rhs:
            for row, (a, b) in zip(rhs_samples[rhs_start:], event_pairs):
                row['stream_ms'] = a.elapsed_time(b) if cuda else None
        rows = [row for row in result.trajectory if row['step'] >= request['warmup']]
        t_start = next(row['t'] for row in result.trajectory if row['step'] == request['warmup']-1)
        samples.append(dict(wall_ms=(window_end-window_start)*1000/request['steps'],
                            stream_ms=start_event.elapsed_time(end_event)/request['steps'] if cuda else None,
                            runner_step_ms=[row['stepTime_ms'] for row in rows],
                            simulated_time_start=float(t_start), simulated_time_end=float(rows[-1]['t']),
                            accepted_times=[row['t'] for row in rows], rhs_calls=rhs_count,
                            rhs_per_step=rhs_count/request['steps']))
        counts.append(peak_memory)
        adjacency = getattr(result.state, 'adjacency', None)
        edges = int(adjacency.numNeighbors.sum()) if getattr(adjacency, 'numNeighbors', None) is not None else None
        if edges is None and hasattr(adjacency, 'chunks'):
            edges = sum(int(chunk.adjacency.numNeighbors.sum()) for chunk in adjacency.chunks)
        effective = dict(compute=asdict(result.ctx.config.compute),
                         execution=result.ctx.scratch.get('execution'),
                         compSPH_step_mode=result.ctx.scratch.get('compSPHStepMode'),
                         whole_step_graph=result.ctx.scratch.get('stepGraph') is not None,
                         solver_cuda_graph=getattr(result.ctx.schemeConfig, 'cudaGraph', None),
                         neighbor_lanes=request['lanes'], particle_order=result.ctx.spec.particleOrder,
                         topology_type=type(adjacency).__name__,
                         layout_sorts=[row.get('layoutSorts') for row in rows],
                         coverage_renewals=[row.get('coverageRenewals') for row in rows])
        if request.get('execution') == 'graph' and not (effective['whole_step_graph'] or effective['solver_cuda_graph']):
            raise NotImplementedError('explicit graph request did not produce a graph route; refusing to label eager work as graph')
        final_info = dict(particles=int(state.positions.shape[0]), final_directed_edges_including_self=edges,
                          effective_compute=effective,
                          case=case.name, specification=spec.toDict(),
                          timing_scope='accepted-boundary to accepted-boundary; includes ordinary runner observation, hooks, neighbour maintenance, timestep and integrator work; output/diagnostics disabled',
                          rhs_note='Instrumented trajectory: host RHS submission and CUDA stream intervals; not pure kernel time' if instrument_rhs else 'No RHS instrumentation in full-step timing; counts intentionally unavailable')
        if not instrument_rhs:
            samples[-1].update(rhs_calls=None, rhs_per_step=None)
        del result, state, adjacency, value, timed_case
        synchronize(device)
        gc.collect()  # Release run-owned cycles before constructing the next fresh trajectory.
    timing = dict(samples=samples, wall_ms=summarize([r['wall_ms'] for r in samples]),
                  stream_ms=summarize([r['stream_ms'] for r in samples]) if cuda else None,
                  peak_memory_by_repetition=counts)
    if instrument_rhs:
        timing.update(rhs_samples=rhs_samples, rhs_host_ms=summarize([r['host_ms'] for r in rhs_samples]),
                      rhs_stream_ms=summarize([r['stream_ms'] for r in rhs_samples]) if cuda else None)
    answer = dict(workload=final_info, timing=timing)
    if profiler:
        answer['profile'] = profile_summary(profiler, profile_path)
    return answer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    result = dict(schema_version=1, request=request, status='failed')
    started = time.perf_counter()
    try:
        device = configure(request)
        result['machine'] = metadata(device)
        if request['backend'] != 'warp':
            raise NotImplementedError('the current frontend benchmark qualifies Warp only; native frontend integration remains separate')
        base_request = dict(request, profile=False)
        result.update(trajectory(base_request, device, instrument_rhs=request['kind'] == 'rhs'))
        if request.get('profile'):
            profiled = trajectory(dict(request, repeats=1), device,
                                  instrument_rhs=request['particle_order'] == 'input' and request.get('execution', 'eager') == 'eager',
                                  profile_path=args.output.with_suffix('.trace.json'))
            result['profile'] = profiled['profile']
        result['status'] = 'ok'
    except NotImplementedError as exc:
        result.update(status='unsupported', error=str(exc))
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    result['process_wall_s'] = time.perf_counter()-started
    write_result(args.output, result)
    print(f'{result["status"]}: {args.output}')
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
