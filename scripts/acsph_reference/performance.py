"""Bounded performance observations for the exact-IC ACSPH comparison."""
import json
from pathlib import Path
import time


def install_light(path, scheme):
    """Per-step receipts without retaining graphs, events or iteration histories."""
    from warpSPH.modules.artificialCompressible import dualTime
    import warpSPH.schemes.incompressible.artificialCompressible as acscheme
    original_graph, original_solve = dualTime.graphedIteration, acscheme.solveDualTime
    current = {}
    path = Path(path)

    def graph(problem):
        value = original_graph(problem)
        current['graph'] = value
        return value

    def solve(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = original_solve(*args, **kwargs)
            value = current.get('graph')
            row = {'step': current.get('step', 0) + 1,
                'solve_wall_ms': (time.perf_counter()-start)*1000,
                'iterations': result.iterations, 'epsilonV': result.epsilonV,
                'captured': value is not None and value.graph is not None,
                'replays': value.replays if value is not None else 0,
                'disabled': getattr(scheme, '_acsphGraphDisabled', None)}
            reference = getattr(scheme, '_dyngpusphSolveDiagnostics', None)
            if scheme.acParams.referenceModel == 'dyngpusph-static-2d' and reference is not None:
                row.update({key: value for key, value in reference.items() if key != 'history'})
                row.update(captured=reference['graphCaptured'], replays=reference['graphReplays'])
            current['step'] = row['step']
            with path.open('a') as log:
                log.write(json.dumps(row) + '\n')
            return result
        finally:
            current.pop('graph', None)

    dualTime.graphedIteration = graph
    acscheme.solveDualTime = solve


def install(root, system, scheme, mode):
    import torch
    from warpSPH.modules.artificialCompressible import dualTime

    root = Path(root); root.mkdir()
    if mode == 'profile':
        scheme._acsphGraphDisabled = 'eager performance baseline'
    elif mode == 'graph':
        # Exact specialization of the shared function's current first two
        # guards for this immutable sampled-wall + stored-ghost fixture.
        # No shared boundary source is edited and no velocity is changed.
        assert bool((system.state.kinds == 1).any())
        assert bool((system.state.kinds == 2).any())
        from warpSPH.modules.mdbc import viscosityVelocity
        original = viscosityVelocity.computeViscosityBoundaryVelocities
        def known_ghosts(state, config, sc, *, adjacency=None, boundaryVelocitiesPrepared=True):
            if boundaryVelocitiesPrepared:
                return state.velocities
            return original(state, config, sc, adjacency=adjacency,
                            boundaryVelocitiesPrepared=boundaryVelocitiesPrepared)
        viscosityVelocity.computeViscosityBoundaryVelocities = known_ghosts

    original_iterate = dualTime.iterate
    original_solve = dualTime.solveDualTime
    original_graph = dualTime.graphedIteration
    rows=[]; graphs=[]; profiles=[]; solve_rows=[]; graph_rows=[]
    profiler=None

    def graph(problem):
        value=original_graph(problem)
        if value is not None:graphs.append(value)
        if value is None:return None
        class TimedGraph:
            @property
            def disabled(self):return value.disabled
            def __call__(self,*inputs):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();t=time.perf_counter();result=value(*inputs)
                cpu_ms=(time.perf_counter()-t)*1000;end.record()
                graph_rows.append((start,end,cpu_ms))
                return result
        return TimedGraph()

    def iterate(*args, **kwargs):
        nonlocal profiler
        # Captured device events would measure the wrong calls on replay.
        if torch.cuda.is_current_stream_capturing():
            return original_iterate(*args, **kwargs)
        m=len(rows)
        if mode == 'profile' and m == 4:
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
            profiler.__enter__()
        if mode == 'profile' and m == 9:
            from warpSPHCore import callAudit
            callAudit.reset();callAudit.enable()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();t=time.perf_counter()
        result=original_iterate(*args, **kwargs)
        cpu_ms=(time.perf_counter()-t)*1000;end.record()
        rows.append((start,end,cpu_ms))
        if mode == 'profile' and m == 6:
            profiler.__exit__(None,None,None)
            profiler.export_chrome_trace(str(root/'trace.json'))
            (root/'profiler-cpu.txt').write_text(profiler.key_averages().table(sort_by='self_cpu_time_total',row_limit=35))
            (root/'profiler-cuda.txt').write_text(profiler.key_averages().table(sort_by='self_device_time_total',row_limit=35))
            profiles.extend({'name':e.name,'device':str(e.device_type),
                'cpu_us':e.self_cpu_time_total,'device_us':e.self_device_time_total,
                'start_us':e.time_range.start,'end_us':e.time_range.end} for e in profiler.events())
            print('PERFORMANCE profiler saved (three iterations)',flush=True)
        if mode == 'profile' and m == 9:
            from warpSPHCore import callAudit
            (root/'call-audit.txt').write_text(callAudit.report(top=35,detail=8))
            callAudit.disable()
            print('PERFORMANCE one-iteration call audit saved',flush=True)
        return result

    def solve(*args, **kwargs):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();t=time.perf_counter();result=original_solve(*args, **kwargs)
        cpu_ms=(time.perf_counter()-t)*1000;end.record();end.synchronize()
        solve_rows.append({'cpu_wall_ms':cpu_ms,'device_span_ms':start.elapsed_time(end),
                           'pseudo_iterations':result.iterations})
        return result

    dualTime.iterate=iterate;dualTime.graphedIteration=graph
    # The scheme imports the solve entry point, so patch that alias too.
    import warpSPH.schemes.incompressible.artificialCompressible as acscheme
    acscheme.solveDualTime=solve

    def finish():
        torch.cuda.synchronize()
        data={'mode':mode,'solve':solve_rows,
            'iterations':[{'index':i,'cpu_wall_ms':cpu,'device_span_ms':a.elapsed_time(b)}
                          for i,(a,b,cpu) in enumerate(rows)],
            'graphs':[{'replays':g.replays,'disabled':g.disabled,'captured':g.graph is not None} for g in graphs],
            'graph_calls':[{'index':i,'cpu_wall_ms':cpu,'device_span_ms':a.elapsed_time(b)}
                           for i,(a,b,cpu) in enumerate(graph_rows)],
            'graph_validated':getattr(scheme,'_acsphGraphValidated',False)}
        (root/'timings.json').write_text(json.dumps(data,indent=2)+'\n')
        if profiles:(root/'events.json').write_text(json.dumps(profiles)+'\n')
        print('PERFORMANCE',json.dumps({k:v for k,v in data.items() if k not in ('iterations','graph_calls')}),flush=True)
    return finish
