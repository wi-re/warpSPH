"""Qualify one driven-flow realization, observing every RHS and accepted state.

Run jobs sequentially; the predeclared cohort uses nx=64, seeds1234/5678/9012
and nx=128, seed1234, at each acceleration0.3/6, to time40. This records
numerical evidence, not automatic certification of stationary training data.
Use analyze_drivenTurbulence.py afterward for time-weighted statistics.
"""
from pathlib import Path
from dataclasses import replace
import argparse, hashlib, json, shutil, sys, time, traceback
repo = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(repo / 'src'), str(repo / 'scripts')]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--nx', type=int, default=64)
p.add_argument('--time', type=float, default=40.0)
p.add_argument('--seed', type=int, default=1234)
p.add_argument('--acceleration', type=float, default=0.3)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--device', default='cuda:0')
args = p.parse_args()
root = args.root.resolve()
root.mkdir(parents=True, exist_ok=False)
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import numpy as np
import torch
from warpSPH.runner import RunConfiguration, resolvedConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from run_compressibleMatrix import capture_sources, source_inventory, source_paths, git_info
from compressible_matrix import write_json, clean
configuration = dict(
    schemaVersion=1, case='drivenTurbulence',
    simulation=dict(
        device=args.device, precision='float32', dim=2, L=1., periodic=True,
        kernel='B7', n_h=4., nx=args.nx, nSteps=None, tLimit=args.time,
        plot=True, video=True, store=False, show=False, quiet=True, progress=True,
        plotBackend='vispy', plotBackendOptions={'app_backend':'egl', 'jupyter_backend':'image'},
        plotInterval=1000, exportRoot=str(root/'exports'),
        cudaGraph=False, execution='eager', velocityAlarmStopRatio=20.,
        stallProgress=1e-3),
    params=dict(
        gamma=5/3, rho0=1., soundSpeed=1., seed=args.seed,
        accelerationRMS=args.acceleration, correlationTime=.5,
        forcingInterval=.05, kMin=1, kMax=2, solenoidalWeight=1.,
        removeMeanAcceleration=True, thermalModel='relaxation', coolingTime=.2),
    solver=dict(scheme='MFM', temporalScheme='ssprk', cfl=.1))
write_json(root / 'input.json', configuration)
shutil.copy2(__file__, root / 'driver.py')
roots = source_paths()
before = capture_sources(root, roots)
write_json(root / 'provenance.json', dict(repositories={k: git_info(v) for k, v in roots.items()}, sourceSHA256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(), device=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else args.device, torch=torch.__version__, driverSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
# Independent ten-step warmup: omit video, then rebuild the fluid and OU process.
warm = RunConfiguration.fromDict(configuration)
warm.simulation.update(nSteps=10, plot=False, video=False, progress=False)
w = warm.run()
assert not w.diverged
del w
if args.device.startswith('cuda'):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
stats = dict(count=0, minEnergy=float('inf'), minPressure=float('inf'), bad=0)
latest = {}
history = []
sampleClock = [0]
samples = root / 'samples'
samples.mkdir()
hf = (root / 'history.jsonl').open('w', buffering=1)

def arrays(system):
    s = system.state
    return {k: getattr(s, k).detach().cpu().numpy() for k in ['positions', 'Q', 'supports', 'masses', 'densities', 'velocities', 'pressures', 'internalEnergies', 'sourceEnergyBudget']}

def build(ctx):
    system = case.buildSystem(ctx)
    np.savez_compressed(root / 'initial.npz', **arrays(system))
    write_json(root / 'resolved.json', resolvedConfiguration(ctx))
    latest['ctx'] = ctx
    rhs = ctx.stepFunction

    def observed(s, dt, **kw):
        update, *rest = rhs(s, dt, **kw)
        state = s.state
        stage = s.meshlessContext
        probes = [stage.u.min(), stage.P.min()] + [torch.isfinite(v).all().to(stage.u) for v in [state.Q, state.positions, stage.u, stage.P, stage.rho, update.dQdt]]
        values = torch.stack(probes).detach().cpu().tolist()
        u, pressure = values[:2]
        finite = all(values[2:])
        stats['count'] += 1
        stats['minEnergy'] = min(stats['minEnergy'], u)
        stats['minPressure'] = min(stats['minPressure'], pressure)
        stats['bad'] += int(not finite or u <= 0 or pressure <= 0)
        return (update, *rest)
    ctx.bundle = replace(ctx.bundle, stepFunction=observed)
    latest['start'] = time.perf_counter()
    return system

def diagnostics(ctx, system):
    row = case.diagnostics(ctx, system)
    row.update(t=float(system.t))
    history.append(row)
    hf.write(json.dumps(clean(row)) + '\n')
    if row['t'] + 1e-10 >= sampleClock[0] * 0.1:
        np.savez_compressed(samples / f'{sampleClock[0]:05d}.npz', t=np.array(row['t']), **arrays(system))
        latest['sampleTime'] = row['t']
        sampleClock[0] += 1
    if len(history) % 100 == 0:
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
        np.savez_compressed(samples / f'{sampleClock[0]:05d}.npz',
                            t=np.array(float(result.state.t)), **arrays(result.state))
    torch.save(result.state.sourceTerms.acceleration.state_dict(), root / 'forcing.pt')
    final = history[-1]
    positive = all((r['minEnergy'] > 0 and r['minDensity'] > 0 and (r['minPressure'] > 0) and all((np.isfinite(r[k]) for k in ['totalEnergy', 'rmsMachMass', 'maxVelocity', 'relativeEnergyBudgetResidual'])) for r in history))
    after = {k: v[1] for k, v in source_inventory(roots).items()}
    good = not result.diverged and (not result.stopReason) and positive and (stats['bad'] == 0) and (float(result.state.t) == args.time) and (before == after)
    record = dict(status='completed' if good else 'failed', actualTime=float(result.state.t), targetTime=args.time, nSteps=result.nSteps, particles=len(result.state.state.Q), stageChecks=stats, acceptedPositiveFinite=positive, sourcesStable=before == after, final=final, maxAbsoluteEnergyBudgetResidual=max((abs(r['relativeEnergyBudgetResidual']) for r in history)), maxMach=max((r['rmsMachMass'] for r in history)), loopWallSeconds=elapsed, peakTorchAllocatedBytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None, peakTorchReservedBytes=torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else None, stopReason=result.stopReason)
    assert stats['count'] == 2 * result.nSteps
except Exception:
    record.update(status='error', error=traceback.format_exc())
    raise
finally:
    hf.close()
    write_json(root / 'result.json', record)
    print(json.dumps(clean(record)), flush=True)
if record['status'] != 'completed':
    raise SystemExit(2)
