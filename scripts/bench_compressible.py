"""Short, reproducible compressible-solver throughput and CUDA profiling window.

Run each configuration in a separate process, sequentially. No rendering in
this tight performance loop. Use --ncu with ncu --profile-from-start off.
"""
import argparse
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scheme', choices=['CRKSPH', 'CompSPH', 'MFM', 'MFV'], default='CRKSPH')
    p.add_argument('--case', default='blob')
    p.add_argument('--nx', type=int, default=512)
    p.add_argument('--precision', choices=['float32', 'float64'], default='float64')
    p.add_argument('--mode', choices=['eager', 'graph', 'auto'], default='eager')
    p.add_argument('--lanes', type=int, choices=[1, 2, 4, 8, 16, 32])
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--profile', type=int, default=0)
    p.add_argument('--ncu', action='store_true')
    p.add_argument('--separate', action='store_true', help='reference CRK force and heating in separate traversals')
    p.add_argument('--matching', type=Path, help='paper blob run directory containing input.npz and settings.json')
    p.add_argument('--video', action='store_true')
    p.add_argument('--end', type=float, help='run until this physical time, instead of a fixed step count')
    p.add_argument('--plot-interval', type=int, default=50)
    p.add_argument('--daemon', action='store_true', help='register live run, measurements and trajectory with the existing companion daemon')
    p.add_argument('--initial-state', type=Path, help='saved meshless positions/supports/Q for a developed-flow benchmark')
    p.add_argument('--scheme-set', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.warmup < 1 or a.steps < 1 or a.profile < 0:
        p.error('warmup and steps must be positive; profile must be nonnegative')
    if a.end is not None and (a.end <= 0 or a.profile or a.ncu):
        p.error('--end must be positive and cannot be combined with profiling')
    if a.ncu and a.profile < 1:
        p.error('--ncu needs --profile >= 1 to delimit its capture window')
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=a.precision)
    import torch
    from warpSPHCore.autograd.lanes import setNeighborLanes
    from warpSPH.cases import importAll
    from warpSPH.runner import run
    from warpSPH.runner.case import getCase
    importAll()
    if a.separate:
        import warpSPH.modules.crk.operators as operators
        def separate(state, config, schemeConfig, adjacency, stage, gradH=None):
            return operators.computeCrkSPHAccelWarp(**operators._common(state, config, schemeConfig, adjacency, stage, gradH))
        import warpSPH.schemes.compressible.crkSPH as crk_scheme
        crk_scheme.computeCRKAccel = separate
        import warpSPH.schemes.compressible.compSPH as comp_scheme
        from warpSPH.modules.compSPH.operators import computeCompSPHdudt
        original_accel = comp_scheme.computeCompSPHAccel
        def separate_comp(state, config, schemeConfig, adjacency, gradH=None, *, returnDudt=False):
            result = original_accel(state, config, schemeConfig, adjacency, gradH)
            return (*result, computeCompSPHdudt(state, config, schemeConfig, adjacency, gradH)) if returnDudt else result
        comp_scheme.computeCompSPHAccel = separate_comp
        import warpSPH.modules.compSPH.pairWork as pair_work
        from warpSPH.math.scatter import segment_sum
        def reference_work(adjacency, velocity, work):
            i, j = adjacency.paddedI, adjacency.paddedJ
            return segment_sum(torch.einsum('ij,ij->i', velocity[j] - velocity[i], work), adjacency.numNeighbors)
        pair_work.reducePairWork = reference_work
    setNeighborLanes(a.lanes)
    torch.manual_seed(1234)
    a.out.mkdir(parents=True, exist_ok=True)
    case = getCase(a.case)
    if case.supportedSchemes and a.scheme not in case.supportedSchemes:
        p.error(f'{a.case} does not support {a.scheme}; use sod2d --nx 2800 for a 70k-particle CompSPH case')
    sources = [Path(__file__), Path('scripts/crkIdealH.py'), Path('scripts/study_spheralMatching.py'),
        Path('src/warpSPH/modules/compSPH/workReduction.py'), Path('src/warpSPH/modules/compSPH/accel.py'),
        Path('src/warpSPH/modules/crk/accel.py'), Path('src/warpSPH/schemes/compressible/crkSPH.py'),
        Path('src/warpSPH/schemes/compressible/compSPH.py')]
    if a.scheme in ('MFM', 'MFV'):
        import warpSPH, warpSPHCore, warpSPHIntegrators
        frontend = Path(warpSPH.__file__).parent
        core = Path(warpSPHCore.__file__).parent
        sources = [Path(__file__)]
        for relative in ('modules/meshless', 'modules/reconstruction', 'modules/riemann', 'modules/eos', 'schemes/compressible/meshless'):
            sources += list((frontend / relative).glob('*.py'))
        sources += [frontend / p for p in ('systems/meshless.py', 'modules/timestep/meshless.py', 'configurations/meshless.py', 'runner/runner.py')]
        sources += [core / p for p in ('mfm/warpBackend.py', 'mfm/wp_mfm.py', 'autograd/prepared.py')]
        sources += [Path(warpSPHIntegrators.__file__).parent / p for p in (
            'drivers/mechanics/conserved_lagrangian.py', 'schemes/families/conserved_lagrangian.py')]
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (a.out / 'sources.json').write_text(json.dumps(provenance, indent=2) + '\n')
    overrides = {}
    if a.matching:
        if a.case != 'blob' or a.scheme != 'CRKSPH':
            p.error('--matching requires CRKSPH blob')
        import numpy as np
        from study_idealH import install_support
        from study_spheralMatching import reference_rate
        from warpSPH.modules.eos import refreshEOS
        from warpSPH.cases.common.plotting import Field, particlePlot
        import warpSPH.modules.crk.energyControl as energy
        energy.entropyRate = reference_rate
        if a.separate:
            from study_spheralMatching import reference_fraction
            from warpSPH.math.scatter import segment_sum
            def separate_entropy(adjacency, acceleration, entropy, gamma, velocity, dvdt, dt):
                half = velocity + .5 * dt * dvdt
                i, j = adjacency.paddedI, adjacency.paddedJ
                work = torch.einsum('ij,ij->i', half[j] - half[i], acceleration)
                return segment_sum(reference_fraction(entropy[i], entropy[j], work) * work, adjacency.numNeighbors)
            energy.entropyRate = separate_entropy
        settings = json.loads((a.matching / 'settings.json').read_text())
        if a.nx != settings['nx']:
            p.error('--nx must match the saved native initialization')
        install_support(dict(settings, steps=a.warmup + a.steps + a.profile))
        initial = dict(np.load(a.matching / 'input.npz'))
        original_build = case.buildSystem
        def build(ctx):
            system = original_build(ctx)
            for key, value in initial.items():
                dtype = ctx.dtype if value.dtype.kind == 'f' else None
                setattr(system.state, key, torch.as_tensor(value, device=ctx.device, dtype=dtype).clone())
            state = system.state
            state.crkMassDensities = state.densities.clone()
            state.crkVolumes = None
            state.totalEnergies = state.masses * (state.internalEnergies + .5 * state.velocities.square().sum(-1))
            system.adjacency = None
            refreshEOS(state, ctx.schemeConfig)
            return system
        setup, update = particlePlot([Field('densities', 'Density', colorMap='RdBu',
            colorMapKind='diverging', flip=True, vMin=.75, vMax=10., midPoint=5.375)],
            figsize=(15, 4), dpi=120)
        case = replace(case, buildSystem=build, setupPlot=setup, updatePlot=update)
        overrides = dict(kernel='B8', n_h=4., dt=settings['dt'], maxDt=settings['maxDt'], cflFactor=.25,
            schemeOverrides=dict(crkEvolution='paper', crkDensityPolicy='massReplacement',
                crkEnergyPartition='entropy', crkEnergyPredictor='raw', crkViscosityGradient='current',
                crkViscosityParams=dict(oneSidedWork=False, viscosityKernelScale=4.), diffusionParams=dict(C_l=2., C_q=1.)))
    if a.initial_state:
        if a.scheme not in ('MFM', 'MFV'):
            p.error('--initial-state requires MFM or MFV')
        initial = torch.load(a.initial_state, map_location='cpu')
        original_build = case.buildSystem
        def build_saved(ctx):
            system = original_build(ctx)
            state = system.state
            for key in ('positions', 'supports', 'Q'):
                value = initial[key].to(device=ctx.device, dtype=ctx.dtype)
                if value.shape != getattr(state, key).shape:
                    raise ValueError(f'{key} shape does not match the selected case/resolution')
                setattr(state, key, value.clone())
            state.masses = state.Q[:, 0].clone()
            state.velocities = state.Q[:, 1:-1] / state.Q[:, 0:1]
            state.internalEnergies = state.Q[:, -1] / state.Q[:, 0] - .5 * state.velocities.square().sum(-1)
            state.totalEnergies = state.Q[:, -1].clone()
            return system
        case = replace(case, buildSystem=build_saved)
    initial_conserved = [None]
    initial_mass = [None]
    if a.scheme in ('MFM', 'MFV'):
        untracked_build = case.buildSystem
        def build_tracked(ctx):
            system = untracked_build(ctx)
            initial_conserved[0] = system.state.Q.sum(0, dtype=torch.float64)
            initial_mass[0] = system.state.Q[:, 0].clone()
            return system
        case = replace(case, buildSystem=build_tracked)
    original_configure, original_post = case.configureScheme, case.postStep
    gaps, last, profiler = [], [None], [None]
    checkpoint_targets, saved_checkpoints = [], set()
    if a.end and a.matching:
        quarter = .25 * settings['tKH']
        checkpoint_targets = [float(torch.tensor(q * quarter, dtype=getattr(torch, a.precision)))
                              for q in range(1, int(round(a.end / quarter)) + 1)]
        (a.out / 'checkpoints').mkdir(exist_ok=True)
    def configure(ctx):
        if original_configure:
            original_configure(ctx)
        if a.scheme in ('CRKSPH', 'CompSPH'):
            ctx.schemeConfig.stepMode = a.mode
        for item in a.scheme_set:
            key, value = item.split('=', 1)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            setattr(ctx.schemeConfig, key, value)
    end = a.warmup + a.steps
    def post(ctx, state, step):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if last[0] is not None and a.warmup <= step < end:
            gaps.append((now - last[0]) * 1000)
        last[0] = now
        if original_post:
            original_post(ctx, state, step)
        if checkpoint_targets:
            current_time = float(state.t)
            for index, target in enumerate(checkpoint_targets):
                if index not in saved_checkpoints and current_time >= target - 1.e-10:
                    fields = ('positions', 'velocities', 'masses', 'materials', 'densities', 'internalEnergies', 'supports', 'pressures')
                    torch.save({key: getattr(state.state, key).detach().cpu() for key in fields},
                               a.out / 'checkpoints' / f'KH{(index + 1) / 4:g}.pt')
                    saved_checkpoints.add(index)
        if step % (100 if a.end else 10) == 0:
            print(f'BENCH step={step} t={float(state.t):.8g}', flush=True)
        if step == end - 1:
            if a.profile and not a.ncu:
                profiler[0] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                profiler[0].start()
            if a.ncu:
                torch.cuda.cudart().cudaProfilerStart()
        if step == end + a.profile - 1:
            if profiler[0] is not None:
                profiler[0].stop()
                profiler[0].export_chrome_trace(str(a.out / 'trace.json'))
            if a.ncu:
                torch.cuda.cudart().cudaProfilerStop()
    case = replace(case, configureScheme=configure, postStep=post)
    if a.end:
        overrides['tLimit'] = a.end
        # Match the requested endpoint in both precisions. No other dt change.
        from warpSPH.modules.timestep import defaultTimestep
        original_timestep = case.timestep or defaultTimestep
        def timestep(ctx, state):
            candidate = original_timestep(ctx, state)
            current_time = float(state.t)
            target = next((target for target in checkpoint_targets if target > current_time + 1.e-10), a.end)
            return min(float(candidate), max(0., target - current_time))
        case = replace(case, timestep=timestep)
    publisher = None
    if a.daemon:
        sys.path.insert(0, '/home/lu26029/dev/warpSPHRemote/python')
        from warpsph_remote.publisher import SimulationPublisher
        name = f'Paper blob nx512 | {a.precision} | performance comparison' if a.matching else f'{a.scheme} {a.case} | {a.precision} | performance'
        publisher = SimulationPublisher(name, measurements=True, export_every_steps=100, autostart=False)
        (a.out / 'daemon.json').write_text(json.dumps(dict(id=publisher.id, name=name), indent=2) + '\n')
        print(f'DAEMON run={publisher.id} name={name}', flush=True)
        case = publisher.bind_case(case)
    try:
        result = run(case, scheme=a.scheme, nx=a.nx, nSteps=None if a.end else end + a.profile,
                     quiet=True, progress=True, plot=a.video, video=a.video, show=False,
                     plotBackend='vispy', exportRoot=str(a.out / 'video'), plotInterval=a.plot_interval,
                     store=a.daemon, storeMode='trajectory', stallProgress=1.e-3, stallDtSteps=100, **overrides)
        if publisher:
            publisher.recording(result.ctx)
            publisher.finish(result.ctx, result.state, result.nSteps)
            publisher.update(step=result.nSteps, time=float(result.state.t),
                             dt=float(result.ctx.config.dt), particles=len(result.state.state.positions))
            publisher.close('failed' if result.diverged else 'completed', result.stopReason or '')
    except BaseException as error:
        if publisher:
            publisher.close('failed', str(error))
        raise
    fields = ['positions', 'velocities', 'densities', 'internalEnergies', 'pressures', 'supports', 'masses', 'materials', 'Q']
    torch.save({k: getattr(result.state.state, k).detach().cpu() for k in fields
                if getattr(result.state.state, k, None) is not None}, a.out / 'state.pt')
    rows = [r['stepTime_ms'] for r in result.trajectory if a.warmup < r['step'] <= end]
    data = dict(vars(a), initial_state=str(a.initial_state) if a.initial_state else None, out=str(a.out), matching=str(a.matching) if a.matching else None, particles=len(result.state.state.positions),
                wall_ms_mean=statistics.mean(gaps), wall_ms_median=statistics.median(gaps),
                integrator_ms_mean=statistics.mean(rows) if rows else None, diverged=bool(result.diverged),
                run_wall_seconds=float(result.wallTime),
                run_wall_ms_per_step=1000. * float(result.wallTime) / result.nSteps,
                completed_steps=result.nSteps, simulated_time=float(result.state.t),
                initial=result.trajectory[0], final=result.trajectory[-1], gpu=torch.cuda.get_device_name())
    if initial_conserved[0] is not None:
        Q = result.state.state.Q
        initial_total = initial_conserved[0].cpu().tolist()
        final_total = Q.sum(0, dtype=torch.float64).cpu().tolist()
        raw_u = Q[:, -1] / Q[:, 0] - .5 * (Q[:, 1:-1] / Q[:, 0:1]).square().sum(-1)
        data['conservation'] = dict(initial=initial_total, final=final_total,
            mass_relative_drift=(final_total[0] - initial_total[0]) / abs(initial_total[0]),
            energy_relative_drift=(final_total[-1] - initial_total[-1]) / abs(initial_total[-1]),
            min_mass=float(Q[:, 0].min()), max_particle_mass_change=float((Q[:, 0] - initial_mass[0]).abs().max()), min_raw_internal_energy=float(raw_u.min()))
    data['export_path'] = str(result.ctx.exportPath) if a.video or a.daemon else None
    data['daemon_run_id'] = publisher.id if publisher else None
    if a.matching:
        state = result.state.state
        cold = (state.densities > 6.4) & (state.internalEnergies < 1.35)
        data['paper_cold_fraction'] = float(torch.where(cold, state.masses, 0.).sum() / state.masses[state.materials == 1].sum())
        data['relative_energy_change'] = ((1 + data['final']['relativeEnergyDrift']) / (1 + data['initial']['relativeEnergyDrift']) - 1)
    if a.end:
        (a.out / 'trajectory.json').write_text(json.dumps(result.trajectory) + '\n')
    (a.out / 'result.json').write_text(json.dumps(data, indent=2) + '\n')
    print('RESULT ' + json.dumps({k: v for k, v in data.items() if k not in ('initial', 'final')}), flush=True)


if __name__ == '__main__':
    main()
