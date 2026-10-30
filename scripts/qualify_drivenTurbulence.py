"""Qualify one driven-flow realization, observing every RHS and accepted state.

Run jobs sequentially; the predeclared cohort uses nx=64, seeds1234/5678/9012
and nx=128, seed1234, at each acceleration0.3/6, to time40. This records
numerical evidence, not automatic certification of stationary training data.
Use analyze_drivenTurbulence.py afterward for time-weighted statistics.
"""
from pathlib import Path
from dataclasses import replace
import argparse, hashlib, json, math, shutil, sys, time, traceback
repo = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(repo / 'src'), str(repo / 'scripts')]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scheme', choices=['MFM','MFV','CRKSPH','CompSPH','Monaghan','GSPH','InutsukaGSPH'], default='MFM')
p.add_argument('--max-dt', type=float, default=.01)
p.add_argument('--plot-interval', type=int, default=1000)
p.add_argument('--progress-interval', type=int, default=100)
p.add_argument('--nx', type=int, default=64)
p.add_argument('--dim', type=int, choices=(2, 3), default=2)
p.add_argument('--slice-grid', type=int, default=96, help='3D video slice resolution; no effect on hydro')
p.add_argument('--final-checkpoint', action='store_true', help='Store native initial/final restart states; intermediate particle exports use sample-interval')
p.add_argument('--time', type=float, default=40.0)
p.add_argument('--seed', type=int, default=1234)
p.add_argument('--acceleration', type=float, default=0.3)
p.add_argument('--k-min', type=int, default=1)
p.add_argument('--k-max', type=int, default=2)
p.add_argument('--sample-interval', type=float, default=.1, help='Requested physical sampling interval; observe accepted states without changing dt')
p.add_argument('--instantaneous-power', action='store_true', help='Record accepted-state source power without changing the physical timestep')
p.add_argument('--root', type=Path, required=True)
p.add_argument('--device', default='cuda:0')
args = p.parse_args()
if not math.isfinite(args.sample_interval) or args.sample_interval <= 0:
    p.error('--sample-interval must be finite and positive')
if args.progress_interval < 1:
    p.error('--progress-interval must be positive')
if args.slice_grid < 4:
    p.error('--slice-grid must be at least4')
if not 1 <= args.k_min <= args.k_max <= 16:
    p.error('need 1 <= --k-min <= --k-max <= 16')
root = args.root.resolve()
root.mkdir(parents=True, exist_ok=False)
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import numpy as np
import torch
from warpSPH.runner import RunConfiguration, resolvedConfiguration
from warpSPH.utils.turbulenceBudget import AcceptedSampleClock
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from run_compressibleMatrix import capture_sources, source_inventory, source_paths, git_info
from compressible_matrix import write_json, clean
configuration = dict(
    schemaVersion=1, case='drivenTurbulence',
    simulation=dict(
        device=args.device, precision='float32', dim=args.dim, L=1., periodic=True,
        kernel='B7', n_h=4., nx=args.nx, nSteps=None, tLimit=args.time,
        plot=True, video=True, store=False, show=False, quiet=True, progress=True,
        plotBackend='vispy', plotBackendOptions={'app_backend':'egl', 'jupyter_backend':'image'},
        plotInterval=args.plot_interval, maxDt=args.max_dt, exportRoot=str(root/'exports'),
        cudaGraph=False, execution='eager', velocityAlarmStopRatio=20.,
        stallProgress=1e-3),
    params=dict(
        gamma=5/3, rho0=1., soundSpeed=1., seed=args.seed,
        accelerationRMS=args.acceleration, correlationTime=.5,
        forcingInterval=.05, kMin=args.k_min, kMax=args.k_max, solenoidalWeight=1.,
        removeMeanAcceleration=True, thermalModel='relaxation', coolingTime=.2),
    solver=(dict(scheme=args.scheme, temporalScheme='ssprk', cfl=.1)
            if args.scheme in ('MFM','MFV') else dict(scheme=args.scheme)))
if args.final_checkpoint:
    configuration['simulation'].update(store=True, storeMode='states', storeInterval=1000000000)
if args.dim == 3:
    from warpSPH.cases.common.plotting import Field, particlePlot
    setup, update = particlePlot([
        Field('densities', 'Density: z=0 slice', gridResolution=args.slice_grid, sliceAxis=2, slicePosition=0.),
        Field('velocities', 'Speed: z=0 slice', mapping='L2Norm', colorMap='inferno',
              gridResolution=args.slice_grid, sliceAxis=2, slicePosition=0.)], figsize=(10, 5))
    case = replace(case, setupPlot=setup, updatePlot=update)
write_json(root / 'recording.json', dict(dim=args.dim, sampleInterval=args.sample_interval,
    video='z=0 interpolated slice' if args.dim == 3 else 'native 2D particles',
    sliceGrid=args.slice_grid if args.dim == 3 else None, plotInterval=args.plot_interval))
write_json(root / 'input.json', configuration)
shutil.copy2(__file__, root / 'driver.py')
roots = source_paths()
before = capture_sources(root, roots)
write_json(root / 'provenance.json', dict(repositories={k: git_info(v) for k, v in roots.items()}, sourceSHA256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(), device=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else args.device, torch=torch.__version__, driverSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
# Independent ten-step warmup: omit video, then rebuild the fluid and OU process.
warm = RunConfiguration.fromDict(configuration)
warm.simulation.update(nSteps=10, plot=False, video=False, progress=False, store=False)
w = warm.run()
assert not w.diverged
del w
if args.device.startswith('cuda'):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
stats = dict(count=0, minEnergy=float('inf'), minPressure=float('inf'), bad=0)
stepTimings = []
lastDiagnostic = [None]
recordingCost = dict(samples=0, seconds=0.)
setupStarted = time.perf_counter()
latest = {}
history = []
sampleClock = [0]
sampleSchedule = AcceptedSampleClock(args.sample_interval)
sampleTimes = []
samples = root / 'samples'
samples.mkdir()
hf = (root / 'history.jsonl').open('w', buffering=1)

def arrays(system):
    s = system.state
    return {k: getattr(s, k).detach().cpu().numpy() for k in ['positions', 'Q', 'supports', 'masses', 'densities', 'velocities', 'pressures', 'internalEnergies', 'sourceEnergyBudget']}

def save_sample(path, system, t):
    start = time.perf_counter()
    np.savez_compressed(path, t=np.array(t), **arrays(system))
    recordingCost['samples'] += 1
    recordingCost['seconds'] += time.perf_counter()-start

def build(ctx):
    system = case.buildSystem(ctx)
    np.savez_compressed(root / 'initial.npz', **arrays(system))
    write_json(root / 'resolved.json', resolvedConfiguration(ctx))
    latest['ctx'] = ctx
    rhs = ctx.stepFunction

    def observed(s, dt, stageIndex=None, **kw):
        if stageIndex is not None:
            kw['stageIndex'] = stageIndex
        update, *rest = rhs(s, dt, **kw)
        state = s.state
        stage = getattr(s, 'meshlessContext', None)
        u, pressure, density = ((stage.u, stage.P, stage.rho) if stage is not None else
            (state.internalEnergies, state.pressures, state.densities))
        rates = ([update.dQdt] if stage is not None else [update.dvdt, update.dudt, update.dEdt])
        probes = [u.min(), pressure.min()] + [torch.isfinite(v).all().to(u) for v in [state.Q, state.positions, u, pressure, density, *rates]]
        values = torch.stack(probes).detach().cpu().tolist()
        u, pressure = values[:2]
        finite = all(values[2:])
        stats['count'] += 1
        stats['minEnergy'] = min(stats['minEnergy'], u)
        stats['minPressure'] = min(stats['minPressure'], pressure)
        stats['bad'] += int(not finite or u <= 0 or pressure <= 0)
        return (update, *rest)
    ctx.bundle = replace(ctx.bundle, stepFunction=observed)
    latest['setupWallSeconds'] = time.perf_counter()-setupStarted
    latest['start'] = time.perf_counter()
    return system

def diagnostics(ctx, system):
    stamp = time.perf_counter()
    if lastDiagnostic[0] is not None:
        stepTimings.append(stamp-lastDiagnostic[0])
    lastDiagnostic[0] = stamp
    row = case.diagnostics(ctx, system)
    row.update(t=float(system.t))
    if args.instantaneous_power:
        state = system.state
        momentum, work, heat = system.sourceTerms.fields(state.masses, state.positions,
            state.velocities, state.internalEnergies, system.t)
        powers = torch.stack((work.double().sum(), heat.double().sum(),
                              (momentum.double()*state.velocities.double()).sum()))
        row.update(zip(('instantaneousSourcePower', 'instantaneousBathPower',
                        'instantaneousSourcePowerDoubleDot'), powers.detach().cpu().tolist()))
    history.append(row)
    hf.write(json.dumps(clean(row)) + '\n')
    if sampleSchedule.due(row['t']):
        save_sample(samples / f'{sampleClock[0]:05d}.npz', system, row['t'])
        latest['sampleTime'] = row['t']
        sampleTimes.append(row['t'])
        sampleClock[0] += 1
    if len(history) % args.progress_interval == 0:
        write_json(root / 'progress.json', dict(row, steps=len(history) - 1, stages=stats))
        print(f"step={len(history) - 1} t={row['t']:.9g} dt={float(ctx.config.dt):.9g} badRHS={stats['bad']}", flush=True)
    return row
record = dict(status='running')
try:
    result = RunConfiguration.fromDict(configuration).run(replace(case, buildSystem=build, diagnostics=diagnostics))
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - latest['start']
    np.savez_compressed(root / 'final.npz', **arrays(result.state))
    if latest.get('sampleTime') != float(result.state.t):
        save_sample(samples / f'{sampleClock[0]:05d}.npz', result.state, float(result.state.t))
        sampleTimes.append(float(result.state.t))
    torch.save(result.state.sourceTerms.acceleration.state_dict(), root / 'forcing.pt')
    final = history[-1]
    stepArray = np.asarray(stepTimings)
    wallClock = dict(loopWallSeconds=elapsed, setupWallSeconds=latest['setupWallSeconds'],
        acceptedIntervals=len(stepTimings), wallStepQuantilesSeconds=np.quantile(stepArray, [0., .5, .9, .99, 1.]).tolist() if len(stepArray) else [],
        secondsPerSimulatedUnit=elapsed/float(result.state.t),
        dtQuantiles=np.quantile(np.diff([r['t'] for r in history]), [0., .5, .9, 1.]).tolist(),
        snapshotTransferCompressionWriteSeconds=recordingCost['seconds'],
        snapshots=sum(1 for _ in samples.glob('*.npz')),
        compressedSnapshotBytes=sum(p.stat().st_size for p in samples.glob('*.npz')),
        particleFrameBytes=sum(a.nbytes for a in arrays(result.state).values()))
    write_json(root/'performance.json', wallClock)
    positive = all((r['minEnergy'] > 0 and r['minDensity'] > 0 and (r['minPressure'] > 0) and all((np.isfinite(r[k]) for k in ['totalEnergy', 'rmsMachMass', 'maxVelocity', 'relativeEnergyBudgetResidual'])) for r in history))
    after = {k: v[1] for k, v in source_inventory(roots).items()}
    good = not result.diverged and (not result.stopReason) and positive and (stats['bad'] == 0) and (float(result.state.t) == args.time) and (before == after)
    record = dict(status='completed' if good else 'failed', extraRHS=stats['count']-2*result.nSteps, scheme=args.scheme, actualTime=float(result.state.t), targetTime=args.time, nSteps=result.nSteps, particles=len(result.state.state.Q), stageChecks=stats, acceptedPositiveFinite=positive, sourcesStable=before == after, final=final, maxAbsoluteEnergyBudgetResidual=max((abs(r['relativeEnergyBudgetResidual']) for r in history)), maxMach=max((r['rmsMachMass'] for r in history)), loopWallSeconds=elapsed, peakTorchAllocatedBytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None, peakTorchReservedBytes=torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else None, stopReason=result.stopReason)
    assert stats['count'] >= 2 * result.nSteps
    if args.scheme in ('MFM','MFV'):
        assert stats['count'] == 2 * result.nSteps
except Exception:
    record.update(status='error', error=traceback.format_exc())
    raise
finally:
    hf.close()
    write_json(root / 'sampling.json', dict(requestedInterval=args.sample_interval, times=sampleTimes, writtenSamples=len(sampleTimes), scheduledSamples=sampleSchedule.count, missedTargets=sampleSchedule.missedTargets, policy='Actual accepted states; no timestep restriction, interpolation or duplicate states. Endpoint saved explicitly.'))
    write_json(root / 'result.json', record)
    print(json.dumps(clean(record)), flush=True)
if record['status'] != 'completed':
    raise SystemExit(2)
