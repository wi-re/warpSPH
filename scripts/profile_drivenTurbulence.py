"""Short, sequential MFM/MFV profiling windows; timings are not long-flow qualification.

Example: PYTHONPATH=src python scripts/profile_drivenTurbulence.py --scheme MFV
--nx 64 --snapshot .tmp/driven-qualification-20261027/strong64-s1234/samples/00250.npz
--acceleration 6 --out .tmp/performance/example --profile

Video omitted for this tight A/B timing loop. Physics, CFL and fresh SSPRK
stages remain fixed. Snapshot mode is a prescribed MFM-generated initial state,
not a native checkpoint or a qualified late MFV trajectory. Compile/setup and
warmup are excluded. Progress and final physical diagnostics remain enabled.
"""
import argparse
import cProfile
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'src'))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scheme', choices=['MFM', 'MFV'], default='MFM')
p.add_argument('--skin', type=float, default=1.)
p.add_argument('--nx', type=int, default=64)
p.add_argument('--acceleration', type=float, default=6.)
p.add_argument('--device', default='cuda:0')
p.add_argument('--snapshot', type=Path)
p.add_argument('--steps', type=int, default=100)
p.add_argument('--warmup', type=int, default=20)
p.add_argument('--diagnostics', choices=['minimal', 'full', 'rhs'], default='minimal')
p.add_argument('--profile', action='store_true')
p.add_argument('--trace', action='store_true')
p.add_argument('--audit', action='store_true')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.steps < 2 or a.warmup < 1:
    p.error('need steps >= 2 and warmup >= 1')
a.out.mkdir(parents=True, exist_ok=False)
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import numpy as np
import torch
from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base
from warpSPHCore import callAudit

configuration = dict(schemaVersion=1, case='drivenTurbulence', simulation=dict(
    device=a.device, precision='float32', nx=a.nx, dim=2, L=1., periodic=True,
    kernel='B7', n_h=4., nSteps=a.warmup+a.steps, tLimit=100.,
    plot=False, video=False, store=False, show=False, quiet=True, progress=False,
    cudaGraph=False, execution='eager'), params=dict(gamma=5/3, rho0=1., soundSpeed=1.,
    seed=1234, accelerationRMS=a.acceleration, correlationTime=.5, forcingInterval=.05,
    kMin=1, kMax=2, solenoidalWeight=1., removeMeanAcceleration=True,
    thermalModel='relaxation', coolingTime=.2), solver=dict(
    scheme=a.scheme, temporalScheme='ssprk', cfl=.1, neighborSkin=a.skin))
(a.out/'input.json').write_text(json.dumps(configuration, indent=2)+'\n')
prof = cProfile.Profile() if a.profile else None
trace = None
if a.trace:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if a.device.startswith('cuda'):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace = torch.profiler.profile(activities=activities, record_shapes=True)
if a.audit:
    callAudit.enable(warmupSteps=a.warmup)
record = dict(times=[], rhsChecks=0, rhsBad=0, rhsMinU=float('inf'))

def sync():
    if a.device.startswith('cuda'):
        torch.cuda.synchronize()

def build(ctx):
    system = base.buildSystem(ctx)
    if a.snapshot:
        from warpSPH.modules.meshless.geometry import buildGeometry, stashGeometry
        from warpSPH.modules.meshless.primitives import refreshPrimitives
        with np.load(a.snapshot) as saved:
            for name in ['positions', 'supports', 'Q', 'sourceEnergyBudget']:
                old = getattr(system.state, name)
                value = torch.as_tensor(saved[name], device=old.device, dtype=old.dtype).clone()
                if value.shape != old.shape:
                    raise ValueError(f'snapshot {name} shape mismatch')
                setattr(system.state, name, value)
            system.t = float(saved['t'])
        g = buildGeometry(system.state.positions, system.state.supports, None, ctx.config, ctx.schemeConfig, system.domain)
        system.state.lam = g.lam
        refreshPrimitives(system.state, g, ctx.schemeConfig.gamma, ctx.schemeConfig.nngb, 2, updateSupports=False)
        stashGeometry(system.state.positions, system.state.supports, system.domain, ctx.config.kernel, ctx.schemeConfig, g, owner=system.state)
        ctx.config.dt = base.timestep(ctx, system)
    if a.diagnostics == 'rhs':
        rhs = ctx.stepFunction
        def observed(s, dt, **kw):
            update, *rest = rhs(s, dt, **kw)
            c = s.meshlessContext
            probes = [c.u.min(), c.P.min()] + [torch.isfinite(v).all().to(c.u) for v in [s.state.Q, s.state.positions, c.u, c.P, c.rho, update.dQdt]]
            values = torch.stack(probes).detach().cpu().tolist()
            record['rhsChecks'] += 1
            record['rhsMinU'] = min(record['rhsMinU'], values[0])
            record['rhsBad'] += int(values[0] <= 0 or values[1] <= 0 or not all(values[2:]))
            return (update, *rest)
        ctx.bundle = replace(ctx.bundle, stepFunction=observed)
    record['ctx'] = ctx
    record['initialT'] = float(system.t)
    return system

def post(ctx, system, step):
    base.postStep(ctx, system, step)



def diagnostics(ctx, system):
    step = ctx.scratch.get('completedSteps', 0)
    row = base.diagnostics(ctx, system) if a.diagnostics != 'minimal' else {}
    if step == a.warmup:
        sync()
        record['startT'] = float(system.t)
        record['tick'] = time.perf_counter()
        record['start'] = record['tick']
        if prof: prof.enable()
        if trace: trace.start()
    if step > a.warmup:
        # Native runner's time/max-speed transfer already synchronizes each step.
        now = time.perf_counter()
        record['times'].append(1000*(now-record['tick']))
        record['tick'] = now
    if step == a.warmup+a.steps:
        sync()
        record['elapsed'] = time.perf_counter()-record['start']
        if prof: prof.disable()
        if trace: trace.stop()
    if step % 25 == 0:
        print(f'PROFILE step={step} t={float(system.t):.9g}', flush=True)
    return row

result = RunConfiguration.fromDict(configuration).run(replace(base, buildSystem=build, postStep=post, diagnostics=diagnostics))
sync()
final = base.diagnostics(record['ctx'], result.state)
np.savez_compressed(a.out/'final.npz', **{k: getattr(result.state.state, k).detach().cpu().numpy() for k in ['positions', 'supports', 'Q', 'sourceEnergyBudget']})
if prof:
    prof.dump_stats(str(a.out/'profile.pstats'))
    with (a.out/'profile.txt').open('w') as f:
        pstats.Stats(prof, stream=f).strip_dirs().sort_stats('cumulative').print_stats(70)
if trace:
    trace.export_chrome_trace(str(a.out/'trace.json'))
    (a.out/'operators.txt').write_text(trace.key_averages().table(sort_by='self_cpu_time_total', row_limit=50)+'\n')
    (a.out/'gpu-operators.txt').write_text(trace.key_averages().table(sort_by='self_cuda_time_total', row_limit=50)+'\n')
if a.audit:
    callAudit.writeReport(str(a.out/'audit.txt'))
summary = dict(scheme=a.scheme, neighborSkin=a.skin, nx=a.nx, particles=len(result.state.state.Q),
    device=a.device, gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
    torch=torch.__version__, steps=a.steps, warmup=a.warmup, diagnostics=a.diagnostics,
    instrumented=a.profile or a.trace or a.audit,
    initialTime=record['initialT'], startTime=record['startT'], finalTime=float(result.state.t),
    elapsedSeconds=record['elapsed'], meanStepMS=record['elapsed']*1000/a.steps,
    medianStepMS=statistics.median(record['times']), stepMS=record['times'],
    integratorMS=[r['stepTime_ms'] for r in result.trajectory[-a.steps:]],
    final=final, rhsChecks=record['rhsChecks'], rhsBad=record['rhsBad'],
    rhsMinU=record['rhsMinU'] if record['rhsChecks'] else None,
    diverged=result.diverged, stopReason=result.stopReason,
    snapshot=str(a.snapshot) if a.snapshot else None,
    snapshotSHA256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest() if a.snapshot else None,
    head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    driverSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(a.out/'result.json').write_text(json.dumps(summary, indent=2)+'\n')
print('RESULT', json.dumps({k:v for k,v in summary.items() if k not in ['stepMS','integratorMS']}), flush=True)
assert not result.diverged and not result.stopReason and not record['rhsBad']
assert all(np.isfinite(final[k]) for k in ['totalEnergy','minEnergy','minDensity','minPressure'])
assert min(final[k] for k in ['minEnergy','minDensity','minPressure']) > 0
