"""Recorded classic-fluid comparison: resumable, sequential, failure-preserving.

plan writes the immutable inputs; run executes each supported row in a fresh
process; report rebuilds the comparison from saved data without simulations.
No solver settings are adapted in response to an outcome.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
from dataclasses import replace
import hashlib
import html
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

PROJECTION = ('dfsph', 'dfsphPredictor', 'dfsphShift', 'iisph')
SCHEMES = (*PROJECTION, 'deltaSPH', 'sun2017DeltaSPH', 'artificialCompressible')
ANALYTIC = (*PROJECTION, 'deltaSPH')
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / '.tmp/fluid-matrix'


def snapEndpoint(state, end):
    """Snap only a rounding-sized endpoint remainder; never advance physics."""
    if abs(end - float(state.t)) <= 1e-10 * max(1.0, end):
        state.t = end
        return True
    return False


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def make_plan(schemes=SCHEMES):
    rows = []
    # Periodic decay controls and the extra bounded-vortex wall check.
    definitions = [
        ('tgv', 256, 5.0, ('periodic',)),
        ('decayingVortices', 192, 5.0, ('periodic',)),
        ('shearDecay', 192, 2.0, ('periodic',)),
        ('boundedVortices', 128, 3.0, ('particles', 'analytic')),
        ('marrone31', 100, 2.45, ('particles', 'analytic')),
        ('sloshing', 200, 7.0, ('particles', 'analytic')),
        ('hydrostatic', 128, 4.0, ('particles', 'analytic')),
        ('englishWedge', 120, 4.0, ('particles', 'analytic')),
        ('taylorCouette', 96, 25.0, ('particles', 'analytic')),
    ]
    for test, nx, end, reps in definitions:
        for scheme in schemes:
            inc, ac = scheme in (*PROJECTION, 'dfsph'), scheme == 'artificialCompressible'
            for rep in reps:
                skip = None
                if rep == 'analytic' and scheme not in (*ANALYTIC, 'dfsph'):
                    skip = 'Existing runner analytic-wall allowlist excludes this scheme.'
                if test == 'boundedVortices' and rep == 'analytic':
                    skip = 'The existing random-flow case has no analytic-wall geometry adapter.'
                if test == 'taylorCouette':
                    if rep == 'particles':
                        skip = 'Existing Couette case is analytic-only; no prescribed spinning particle-wall adapter.'
                    elif scheme != 'deltaSPH':
                        skip = 'Prescribed rotating analytic bodies / viscous Couette closure unsupported by this scheme.'
                spec = dict(nx=nx, tLimit=end, nSteps=None, kernel='Wendland2', n_h=4.0)
                params = {}
                settings = {}
                if inc:
                    spec.update(integrationScheme='semiImplicitEuler', supportMode='SuperSymmetric',
                                dt=1e-3, maxDt=2e-3, cflFactor=0.2, adaptiveDt=True,
                                calibrateNormalization=True)
                if ac:
                    spec['integrationScheme'] = 'forwardEuler'
                if test == 'tgv':
                    case = 'tgv' if inc else 'tgv-wc'
                    params.update(nu=0.01, k=2, uMag=1.0)
                    if inc:
                        spec.update(maxDt=5e-3, cflFactor=0.3)
                    else:
                        params.update(inviscid=False, initialPressure=True, machTarget=0.1,
                                      referenceVelocity=1.0, shuffleEq7=False)
                elif test in ('decayingVortices', 'boundedVortices'):
                    case = 'randomFlowIncompressible' if inc else 'randomFlow'
                    params.update(bounded=test == 'boundedVortices', nu=0.01, seed=45906734)
                    if inc:
                        spec.update(maxDt=5e-3, cflFactor=0.4)
                    else:
                        params.update(inviscid=False, targetDt=0.00025)
                elif test == 'shearDecay':
                    # Same periodic viscous-mode case used to calibrate Couette.
                    case = 'periodicChannel'
                    params.update(shearWave=True, fluidViscosity='morris', wallViscosityClosure='noslipMoment',
                                  nuPhysical=0.0185, shifting=False)
                    if not inc and not ac:
                        # configureFlow converts alpha*c0*h/(8*xi) to the
                        # realized long-wave Morris viscosity. Match the
                        # projection target at this resolution, not alpha.
                        params['alpha'] = 0.0185 * 8 * 2.8213846683502197 / (10.0 * 4.0 / nx)
                    if ac:
                        skip = 'The existing periodic-channel viscous configuration has no ACSPH branch.'
                elif test == 'sloshing':
                    case = 'sloshingTank'
                    params.update(wallRepresentation=rep)
                    if inc:
                        params['hydrostaticInit'] = True
                    if not inc and not ac:
                        spec['kernel'] = 'Wendland4'
                elif test == 'taylorCouette':
                    case = 'taylorCouette'
                    params.update(closedPreset=inc, fluidViscosity='morris',
                                  wallViscosityClosure='noslipMoment', shifting=not inc)
                    if not inc and not ac:
                        params['alpha'] = 0.0185 * 8 * 2.8213846683502197 / (10.0 * 4.0 / nx)
                        params['nuReference'] = 0.0185
                    if inc:
                        spec.update(dt=0.001, adaptiveDt=True)
                elif test == 'hydrostatic' and rep == 'particles':
                    case = 'hydrostaticColumn'
                    params.update(calibrateRestDensity=True)
                    if not inc and not ac:
                        # Native column hook is projection-specific; use the
                        # established full-width still-pool dambreak adapter.
                        case = 'dambreak'
                        spec['L'] = 1.0
                        params = dict(W=1.0, fillRatio=0.5, fluidWidth=1.0,
                                      hydrostaticInit=True, gravityMagnitude=9.81,
                                      referenceVelocity=math.sqrt(9.81 * 0.5), machTarget=0.05,
                                      pressureProbeHeights=[], wallRepresentation=rep)
                else:
                    case = 'dambreak'
                    spec['L'] = 1.0
                    params.update(wallRepresentation=rep, gravityMagnitude=9.81, wallBC='freeSlip')
                    if test == 'marrone31':
                        params.update(W=3.2196, fillRatio=0.6, fluidWidth=1.2 / 3.2196,
                                      referenceVelocity=1.95 * math.sqrt(9.81 * 0.6), machTarget=0.1,
                                      pressureProbeHeights=[0.160, 0.584, 1.0],
                                      pressureProbeInset=0.0, pressureProbeDiscRadius=0.045)
                    elif test == 'englishWedge':
                        spec['L'] = 1.2
                        params.update(W=2.4, fillRatio=0.5 / 1.2, fluidWidth=1.0,
                                      hydrostaticInit=True, referenceVelocity=math.sqrt(9.81 * 0.5),
                                      machTarget=0.05, pressureProbeHeights=[], obstacleActive=True,
                                      obstacleType='equilateralBottom', offsetX=0.0, aoa=0.0,
                                      maxExtent=0.30 * (0.24 / 0.248), obstacleRepresentation=rep)
                    elif test == 'hydrostatic':
                        params.update(W=1.0, fillRatio=0.5, fluidWidth=1.0,
                                      hydrostaticInit=True, referenceVelocity=math.sqrt(9.81 * 0.5),
                                      machTarget=0.05, pressureProbeHeights=[])
                if scheme in ('deltaSPH', 'sun2017DeltaSPH'):
                    # Explicit plain/PST distinction. Sun's named scheme also
                    # retains its own frozen-diffusion default.
                    params['shifting'] = scheme == 'sun2017DeltaSPH'
                if test == 'shearDecay':
                    params['shifting'] = False  # Viscosity control, independent of PST.
                if test in ('hydrostatic', 'englishWedge'):
                    params['shifting'] = False if case != 'dambreak' else 'off'
                if test == 'englishWedge':
                    params['unifyAttachedObstacles'] = True
                rows.append(dict(id=f'{test}__{scheme}__{rep}', test=test, case=case,
                                 scheme=scheme, boundary=rep, precision='float32',
                                 spec=spec, params=params, scheme_settings=settings,
                                 skip=skip, timeout_seconds=7200,
                                 notes=''))
    return dict(version=5, created=datetime.now(timezone.utc).date().isoformat(), schemes=list(schemes), rows=rows,
                initialization='Normalized summation density, physical mass, scheme-aware hydrostatic seeds, unified static floor-wedge contour; continuum analytic wall density; first/every-step physical limits.',
                timestep='Fixed-case adaptive recovery uses initialized nominal ceiling and existing dtGrowthFactor; physical limits cap every step.',
                policy='One GPU child at a time. No retries or outcome-driven tuning. Preserve failures.',
                reference_notes='See docs/fluid_matrix/README.md for historical presets and comparison limits.')


def addSourceFile(archive, path, relative):
    """Archive file bytes without blocking on network user/group lookups."""
    import tarfile
    info = path.stat()
    entry = tarfile.TarInfo(str(relative))
    entry.size, entry.mtime = info.st_size, info.st_mtime
    entry.uid, entry.gid = info.st_uid, info.st_gid
    entry.mode = info.st_mode & 0o777
    with path.open('rb') as content:
        archive.addfile(entry, content)


def snapshot(root):
    import importlib.util
    roots = [REPO / 'src', REPO / 'scripts' / 'run_fluidMatrix.py',
             REPO / 'scripts' / 'analyze_fluidMatrix.py',
             REPO / 'scripts' / 'prepared_fluidMatrix.py',
             root / 'manifest.json',
             REPO / 'examples/sloshingTank/SPHERIC_TestCase10/data_files/lateral_water_1x.txt']
    for name in ('warpSPHCore', 'warpSPHIntegrators', 'warpSPHBoundaries', 'warpSPHPlotting'):
        package = importlib.util.find_spec(name)
        if package is not None and package.origin:
            roots.append(Path(package.origin).parent)
    hashes = {}
    for base in roots:
        for path in ([base] if base.is_file() else base.rglob('*.py')):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root / 'sources.json', hashes)
    # Source content, not only fingerprints, makes the long run reproducible.
    import tarfile
    with tarfile.open(root / 'sources.tar.gz', 'w:gz') as archive:
        for path in hashes:
            try:
                relative = Path(path).relative_to(REPO.parent)
            except ValueError:
                relative = Path('external') / str(path).lstrip('/')
            addSourceFile(archive, Path(path), relative)


def worker(root, row):
    inc = row.get('solver', row['scheme']) in (*PROJECTION, 'dfsph')
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=row['precision'])
    import numpy as np
    import torch
    from warpSPH.cases import importAll
    from warpSPH.runner import run
    from warpSPH.runner.case import getCase
    sys.path.insert(0, str(REPO.parent / 'warpSPHRemote' / 'python'))
    from warpsph_remote.publisher import SimulationPublisher
    importAll()
    torch.manual_seed(1234)
    np.random.seed(1234)
    out = root / row['id']
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'input.json', row)
    (out / 'harness.py').write_bytes(Path(__file__).read_bytes())
    analysis_script = REPO / 'scripts/analyze_fluidMatrix.py'
    if analysis_script.exists():
        (out / 'analysis_harness.py').write_bytes(analysis_script.read_bytes())
    snapshot(out)
    write_json(out / 'environment.json', {k: os.environ.get(k) for k in
               ('WARPSPH_COMPILE_WALLS', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS')})
    case = getCase(row['case'])
    if row.get('initialization'):
        from prepared_fluidMatrix import prepared_case
        case = prepared_case(case, root, row)
    original_config, original_diag = case.configureScheme, case.diagnostics
    original_post, original_initial = case.postStep, case.initialConditions
    # Preserve case choices; the runner caps every projection step with its
    # physical advection/viscosity limits, including formerly fixed cases.
    original_dt = case.timestep
    ctx_box, initial, last = [], {}, [0.0]
    diagnostic_time = [0.0]
    end = row['spec']['tLimit']
    publisher = SimulationPublisher('Fluid matrix | ' + row['id'], measurements=True,
                                    export_every_steps=100, autostart=False)
    write_json(out / 'daemon.json', dict(id=publisher.id))
    stream = (out / 'diagnostics.jsonl').open('w', buffering=1)

    def configure(ctx):
        ctx_box.append(ctx)
        if original_config:
            original_config(ctx)
        for key, value in row.get('scheme_settings', {}).items():
            setattr(ctx.schemeConfig, key, value)

    def initialize(ctx, system):
        if original_initial:
            original_initial(ctx, system)
        if inc:
            from warpSPH.caseUtils.projection import initializeProjectionDensity
            write_json(out / 'initial_calibration.json', initializeProjectionDensity(ctx, system))
        st = system.state
        initial.update(particles=len(st.positions), fluid_particles=int((st.kinds == 0).sum()),
                       mass=float(st.masses[st.kinds == 0].double().sum()),
                       nominal_dt=float(ctx.config.dt),
                       density_source_policy=getattr(ctx.schemeConfig.iisph,'densitySourcePolicy','absolute') if row.get('initialization') else None,
                       dfsph_ordering=getattr(ctx.schemeConfig.dfsph,'ordering',None) if row.get('initialization') else None,
                       actual_step=ctx.stepFunction.__module__+'.'+ctx.stepFunction.__qualname__,
                       pressure_seed=ctx.scratch.get('initialPressureCalibration'),
                       band_viscosity=bool(getattr(ctx.schemeConfig, 'bandViscosity', False)),
                       actual_viscosity=float(ctx.schemeConfig.diffusionParams.viscidNu),
                       actual_inviscid=bool(ctx.schemeConfig.diffusionParams.inviscid),
                       actual_artificial_viscosity_alpha=float(ctx.schemeConfig.diffusionParams.inviscidAlpha),
                       actual_reference_sound_speed=float(ctx.schemeConfig.fluid.fixedSoundSpeed),
                       actual_viscosity_term=ctx.schemeConfig.diffusionParams.viscousTerm.name,
                       viscosity_calibration=ctx.scratch.get('matrixViscosityCalibration'),
                       numerical_preset=row.get('numerical_preset'),
                       analytic_bodies=len(ctx.schemeConfig.boundaryProvider.scene.bodies)
                       if getattr(ctx.schemeConfig, 'boundaryProvider', None) is not None else 0)
        torch.save({k: v.detach().cpu() for k, v in vars(st).items()
                    if isinstance(v, torch.Tensor) and not k.startswith('_')}, out / 'initial_state.pt')
        write_json(out / 'effective_spec.json', ctx.spec.toDict())
        write_json(out / 'initial.json', initial)

    def diagnostics(ctx, state):
        st = state.state
        update = ctx.scratch.get('lastStageUpdate')
        if inc:
            from warpSPH.caseUtils.projection import physicalProjectionPressure
            pressure = physicalProjectionPressure(st, ctx.schemeConfig)
        else:
            pressure = st.pressures
        result = dict(original_diag(ctx, state) if original_diag else {})
        f = st.kinds == 0
        # Common scalar quantities use fluid particles only, including when
        # the case's own diagnostic includes wall samples.
        mass = st.masses[f].double()
        vel = st.velocities[f].double()
        result.update(maxVelocity=float(vel.norm(dim=-1).max()), fluidKineticEnergy=float((0.5 * mass * vel.square().sum(-1)).sum()),
                      fluidMass=float(mass.sum()), fluidDensityMin=float(st.densities[f].min()),
                      fluidDensityMax=float(st.densities[f].max()))
        if inc:
            from warpSPH.modules.timestep.projection import projectionTimestep
            projectionTimestep(ctx, state, float(ctx.config.dt))
            limits = ctx.scratch['projectionTimestepLimits']
            result['projectionStepLimit'] = limits['maximum']
        result['nextDt'] = float(ctx.config.dt)
        result['takenDt'] = float(state.t) - diagnostic_time[0]
        diagnostic_time[0] = float(state.t)
        if 'matrixTakenStepLimit' in ctx.scratch:
            result['stepLimitAtStart'] = ctx.scratch['matrixTakenStepLimit']
            if result['takenDt'] > result['stepLimitAtStart'] * (1.0 + 1e-7) + 1e-10:
                raise RuntimeError('Comparison step exceeded its physical timestep limit.')
        if row.get('initialization'):
            rho0 = float(ctx.schemeConfig.fluid.restDensity)
            result.update(fluidRmsSpeed=float(vel.square().sum(-1).mean().sqrt()),
                fluidDensityMean=float(st.densities[f].mean()/rho0),
                fluidDensityStd=float(st.densities[f].std()/rho0),
                fluidRestVolume=float((st.masses[f]/rho0).double().sum()),
                pressureMean=float(pressure[f].mean()) if pressure is not None else 0.,
                fluidHeight=float(st.positions[f,1].max()-st.positions[f,1].min()),
                fluidCenterY=float((mass*st.positions[f,1].double()).sum()/mass.sum()))
            if row['test']=='tgv':
                from prepared_fluidMatrix import sample_field
                exact=sample_field(st.positions[f], 'tgv',2*math.pi)*math.exp(-.02*float(state.t))
                result['tgvVelocityRelativeL2']=float((st.velocities[f]-exact).norm()/exact.norm())
            if row['test']=='shearDecay':
                mode=torch.sin(2*math.pi*st.positions[f,1]/float(ctx.spec.L))
                result['modeAmplitude']=float((st.velocities[f,0]*mode).sum()/mode.square().sum()/.05)
            if update is not None:
                for prefix, name in (('DF','divergenceResult'),('CD','densityResult'),('IISPH','iisphResult')):
                    res=getattr(update,name,None)
                    if res is not None:
                        result[prefix+'Iterations']=res.iterations
                        result[prefix+'Cap']=int(res.termination=='budget')
                        result[prefix+'Error']=res.error
                        result[prefix+'PressureMean']=float(res.pressure[f].mean())
            assert torch.equal(st.masses,ctx.scratch['matrixInitialMasses']), 'Particle mass changed during evolution'
        t = float(state.t)
        if row['test'] == 'tgv':
            result['analyticEnergyRatio'] = math.exp(-4.0 * 0.01 * (2.0 / 2.0) ** 2 * t)
        if row['test'] in ('hydrostatic', 'englishWedge') and pressure is not None:
            x = st.positions[f].double()
            y = x[:, 1]
            # Gauge-independent pressure slope over the interior water depth.
            dx = float(ctx.config.dx)
            mask = (y > y.min() + 4 * dx) & (y < y.max() - 4 * dx)
            if row['test'] == 'englishWedge':
                mask &= x[:, 0].abs() > 0.4
            if int(mask.sum()) > 8:
                yy, pp = y[mask], pressure[f][mask].double()
                yc, pc = yy - yy.mean(), pp - pp.mean()
                slope = (yc * pc).sum() / yc.square().sum()
                rho0 = float(ctx.schemeConfig.fluid.restDensity)
                result['matrixPressureSlopeRatio'] = float(-slope / (rho0 * 9.81))
                result['matrixPressureFitRms'] = float((pc - slope * yc).square().mean().sqrt())
        payload = dict(t=t, **result)
        stream.write(json.dumps(payload, default=str) + '\n')
        if time.monotonic() - last[0] > 15:
            last[0] = time.monotonic()
            write_json(out / 'progress.json', payload)
            print('MATRIX_PROGRESS ' + json.dumps(payload), flush=True)
        return result

    def timestep(ctx, state):
        if original_dt:
            dt = original_dt(ctx, state)
        elif inc:
            from warpSPH.modules.timestep.projection import fixedProjectionRequest
            dt = fixedProjectionRequest(ctx)
        else:
            dt = ctx.config.dt
        if not inc:
            # Case hooks and fixed initial choices are ceilings too: evaluate
            # the WCSPH / ACSPH physical constraints even without a native hook.
            from warpSPH.modules.timestep import computeTimestep
            dt = min(float(dt), float(computeTimestep(
                state, ctx.config, ctx.schemeConfig, dt=dt,
                systemUpdate=ctx.scratch.get('lastStageUpdate'))))
        if (row.get('initialization') and row.get('solver')=='dfsph'
                and row['params'].get('hydrostaticInit',False) and float(state.t)==0.
                and not ctx.scratch.get('matrixPressureSeedEncoded')):
            from warpSPH.modules.timestep.projection import projectionTimestep
            from prepared_fluidMatrix import seed_projection_pressure
            # Mirror the outer first-step limiter before converting pressure
            # to timestep-scaled histories; the outer cap is idempotent.
            dt=projectionTimestep(ctx,state,min(float(dt),float(ctx.config.dt)))
            seed_projection_pressure(state.state,row['ordering'],dt)
            ctx.scratch['matrixPressureSeedEncoded']=True
            write_json(out/'pressure_seed.json',dict(actualFirstDt=dt,
                physicalStage='divergence' if row['ordering']=='np_df_shift' else 'density',
                auxiliaryShiftSeedZero=row['ordering']=='np_df_shift',
                source='shared hydrostatic physical pressure; no preparation history'))
            torch.save({k:v.detach().cpu() for k,v in vars(state.state).items()
                if isinstance(v,torch.Tensor) and not k.startswith('_')},out/'first_step_state.pt')
        return min(float(dt), max(0.0, end - float(state.t)))

    def post(ctx, state, step):
        if inc:
            ctx.scratch['matrixTakenStepLimit'] = ctx.scratch['projectionTimestepLimits']['maximum']
        if original_post:
            original_post(ctx, state, step)
        # Do not evaluate pressure maps at a rounding-sized residual dt.
        # This changes only the time label at an already reached endpoint.
        snapEndpoint(state, end)
        if row.get('initialization'):
            from prepared_fluidMatrix import wrap
            state.state.positions=wrap(state.state.positions,ctx.config.domain)

    case = replace(case, configureScheme=configure, initialConditions=initialize,
                   diagnostics=diagnostics, postStep=post, timestep=timestep)
    case = publisher.bind_case(case)
    interval = 50 if inc else 200
    params = dict(row['params'])
    spec = dict(row['spec'])
    scale = 1.0
    if row['test'] in ('marrone31', 'hydrostatic', 'englishWedge'):
        scale = math.sqrt(9.81 * (0.6 if row['test'] == 'marrone31' else 0.5))
    elif row['test'] == 'sloshing':
        scale = math.sqrt(9.81 * 0.093)
    elif row['test'] == 'taylorCouette':
        scale = 0.2
    result = None
    started = time.time()
    def terminate(signum, frame):
        raise TimeoutError('Matrix wall budget / termination signal; preserve partial evidence.')
    signal.signal(signal.SIGTERM, terminate)
    try:
        result = run(case, scheme=row.get('solver', row['scheme']), params=params, **spec,
                     plot=True, video=True, plotBackend='vispy', show=False,
                     progress=True, quiet=True, exportRoot=str(out / 'recording'),
                     store=True, storeMode='trajectory', exportInterval=end / 150,
                     plotInterval=interval, cudaGraph=not inc,
                     velocityAlarmPlotInterval=1, velocityScale=scale,
                     velocityAlarmFactor=20.0, velocityAlarmStopRatio=20.0,
                     stallProgress=1e-3, stallDtSteps=100)
        st = result.state.state
        final_fields={k:v.detach().cpu() for k,v in vars(st).items() if isinstance(v,torch.Tensor) and not k.startswith('_')}
        u=result.ctx.scratch.get('lastStageUpdate')
        if row.get('solver')=='dfsph' and u is not None:
            final_fields['physicalPressures']=(u.divergenceResult.pressure+(0 if row.get('ordering')=='np_df_shift' else u.densityResult.pressure)).detach().cpu()

        torch.save(final_fields,out/'final_state.pt')
        trajectory = result.trajectory
        write_json(out / 'trajectory.json', trajectory)
        keys = sorted(set().union(*(r.keys() for r in trajectory)))
        with (out / 'diagnostics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, keys)
            writer.writeheader()
            writer.writerows(trajectory)
        np.savez_compressed(out / 'series.npz', **{
            k: np.asarray([r.get(k, float('nan')) for r in trajectory]) for k in keys})
        times = [r['stepTime_ms'] for r in trajectory[11:] if 'stepTime_ms' in r]
        graph = result.ctx.scratch.get('stepGraph')
        complete = not result.diverged and float(result.state.t) >= end-1e-10*max(1.0, end)
        data = dict(id=row['id'], status='completed' if complete else 'failed',
                    diverged=bool(result.diverged), stop_reason=result.stopReason,
                    time=float(result.state.t), target_time=end, steps=result.nSteps,
                    wall_seconds=result.wallTime,
                    preparation_seconds_this_run=result.ctx.scratch.get('matrixPreparationSecondsThisRun',0.),
                    evolution_and_output_seconds=result.wallTime-result.ctx.scratch.get('matrixPreparationSecondsThisRun',0.),
                    wall_ms_per_step=1000 * result.wallTime / max(result.nSteps, 1),
                    integrator_ms_mean=float(np.mean(times)) if times else None,
                    integrator=result.ctx.integrator.identifier.name,
                    initial=initial, final=trajectory[-1] if trajectory else {},
                    velocity_alarms=result.velocityAlarms, video_path=result.videoPath,
                    export_path=result.ctx.exportPath, daemon_run_id=publisher.id,
                    graph_replays=getattr(graph, 'replays', None),
                    gpu=torch.cuda.get_device_name(), elapsed_seconds=time.time() - started)
        write_json(out / 'effective_spec_after_run.json', result.ctx.spec.toDict())
        publisher.recording(result.ctx)
        publisher.finish(result.ctx, result.state, result.nSteps)
        publisher.update(step=result.nSteps, time=float(result.state.t),
                         dt=float(result.ctx.config.dt), particles=len(st.positions))
        publisher.close(data['status'], result.stopReason or '')
        write_json(out / 'result.json', data)
        print('MATRIX_RESULT ' + json.dumps(data, default=str), flush=True)
    except BaseException as error:
        traceback.print_exc()
        publisher.close('failed', str(error))
        data = dict(id=row['id'], status='error', error=repr(error),
                    elapsed_seconds=time.time() - started, daemon_run_id=publisher.id)
        if ctx_box:
            ctx = ctx_box[-1]
            data['export_path'] = ctx.exportPath
            if ctx.imagePath and Path(ctx.imagePath).exists():
                try:
                    from warpSPH.runner.media import encodeFrames
                    data['video_path'] = encodeFrames(ctx.imagePath, ctx.exportPath)
                except Exception:
                    traceback.print_exc()
        write_json(out / 'result.json', data)
        raise
    finally:
        stream.close()
        if analysis_script.exists():
            subprocess.run([sys.executable, str(analysis_script), '--root', str(root)], check=False)


def report(root, plan):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results, table = [], []
    for row in plan['rows']:
        path = root / row['id'] / 'result.json'
        data = json.loads(path.read_text()) if path.exists() else dict(status='unsupported' if row['skip'] else 'pending')
        entry = dict(id=row['id'], test=row['test'], scheme=row['scheme'], boundary=row['boundary'],
                     status=data['status'], time=data.get('time'), target_time=row['spec']['tLimit'],
                     particles=data.get('initial', {}).get('particles'),
                     steps=data.get('steps'), wall_seconds=data.get('wall_seconds'),
                     wall_ms_per_step=data.get('wall_ms_per_step'),
                     integrator_ms_mean=data.get('integrator_ms_mean'),
                     reason=row['skip'] or data.get('stop_reason') or data.get('error'),
                     notes=row['notes'])
        results.append(entry)
        video = next((root / row['id']).rglob('output.mp4'), None) if (root / row['id']).exists() else None
        # The default encoder uses out.mp4 on some versions.
        if video is None and (root / row['id']).exists():
            video = next((root / row['id']).rglob('*.mp4'), None)
        links = []
        for path, label in ((video, 'video'), (root / row['id'] / 'diagnostics.csv', 'CSV'),
                            (root / row['id'] / 'series.npz', 'NPZ'),
                            (root / row['id'] / 'run.log', 'log')):
            if path is not None and path.exists():
                links.append(f'<a href="{html.escape(str(path.relative_to(root)))}">{label}</a>')
        table.append('<tr>' + ''.join('<td>' + html.escape(str(entry.get(k) or '')) + '</td>'
                                     for k in ('test', 'scheme', 'boundary', 'status', 'time', 'target_time', 'particles', 'wall_ms_per_step', 'reason'))
                     + '<td>' + ' · '.join(links) + '</td></tr>')
    write_json(root / 'summary.json', results)
    with (root / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plots = []
    metrics = [('tgv', 'fluidKineticEnergy'), ('decayingVortices', 'fluidKineticEnergy'),
               ('boundedVortices', 'fluidKineticEnergy'), ('shearDecay', 'modeAmplitude'),
               ('sloshing', 'sensorPressure'), ('hydrostatic', 'matrixPressureSlopeRatio'),
               ('englishWedge', 'matrixPressureSlopeRatio'), ('taylorCouette', 'profileAmplitude')]
    for test, metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        found = False
        for row in plan['rows']:
            if row['test'] != test:
                continue
            file = root / row['id'] / 'series.npz'
            if not file.exists():
                continue
            with np.load(file) as arr:
                key = metric
                if key not in arr:
                    if test == 'sloshing':
                        key = next((k for k in ('sensorPressurePa', 'pSensor1Pa', 'pSensor1', 'sensor1Pressure', 'pressureSensor1') if k in arr), '')
                    if key not in arr:
                        continue
                t, y = arr['t'], arr[key]
                if metric == 'fluidKineticEnergy' and len(y) and y[0] > 0:
                    y = y / y[0]
                ax.plot(t, y, label=row['scheme'] + '/' + row['boundary'],
                        linestyle='--' if row['boundary'] == 'analytic' else '-')
                found = True
        if found:
            if test == 'tgv':
                t = np.linspace(0, 5, 200)
                ax.plot(t, np.exp(-0.04 * t), 'k:', label='analytic ν=0.01, kphysical=1')
            ax.set(xlabel='time [s]', ylabel=('E / E(0)' if metric == 'fluidKineticEnergy' else metric), title=test)
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            file = root / (test + '.png')
            fig.savefig(file, dpi=150)
            plots.append(f'<a href="{file.name}"><img width="600" src="{file.name}"></a>')
        plt.close(fig)
    counts = {status: sum(r['status'] == status for r in results)
              for status in sorted({r['status'] for r in results})}
    title = 'Classic fluid matrix — ' + plan['created']
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>' + title + '</title>'
        + '<style>body{font:14px system-ui;margin:28px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}img{max-width:100%}</style>'
        + f'<h1>{title}</h1><p>Updated {time.strftime("%Y-%m-%d %H:%M:%S %Z")}. {html.escape(str(counts))}</p>'
        + '<p>FP32; sequential GPU jobs; no outcome-driven tuning. Completion is not an accuracy verdict. '
          'Unsupported combinations are retained. Presets and initialization differ where required by the existing cases. '
          'Band periodic controls include the configured physical viscosity when bandViscosity is enabled; other rows retain their declared settings.</p>'
        + ''.join(plots) + '<table><tr>' + ''.join('<th>' + k + '</th>' for k in
            ('test', 'scheme', 'boundary', 'status', 't', 'target', 'particles', 'wall ms/step', 'reason', 'data'))
        + '</tr>' + ''.join(table) + '</table>')
    return counts


def execute(root, plan):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (root / 'sources.json').exists():
        snapshot(root)
    write_json(root / 'manifest.json', plan)
    for i, row in enumerate(plan['rows']):
        out = root / row['id']
        if row['skip'] or (out / 'result.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        write_json(root / 'queue_status.json', dict(pid=os.getpid(), state='running',
                   current=row['id'], index=i, total=len(plan['rows']), started=time.time()))
        print(f'MATRIX_START {i + 1}/{len(plan["rows"])} {row["id"]}', flush=True)
        with (out / 'run.log').open('w') as log:
            subprocess.run(['nvidia-smi', 'pmon', '-c', '1'], stdout=log, stderr=log)
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                '--root', str(root), '--manifest', str(root / 'manifest.json'), '--row', row['id']],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = child.wait(timeout=row['timeout_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                rc = -signal.SIGTERM
                write_json(out / 'result.json', dict(id=row['id'], status='timeout',
                    error=f'Exceeded declared {row["timeout_seconds"]} s wall budget; no retry.'))
            if not (out / 'result.json').exists():
                write_json(out / 'result.json', dict(id=row['id'], status='error', error=f'Child exit {rc}; see run.log'))
        counts = report(root, plan)
        print('MATRIX_COUNTS ' + json.dumps(counts), flush=True)
    counts = report(root, plan)
    write_json(root / 'queue_status.json', dict(pid=os.getpid(), state='finished', counts=counts,
                                               finished=time.time()))
    print('MATRIX_FINISHED ' + json.dumps(counts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run', 'worker', 'report'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--manifest', type=Path, help='Defaults to ROOT/manifest.json.')
    parser.add_argument('--row')
    parser.add_argument('--profile', choices=('classic','prepared'), default='prepared')
    parser.add_argument('--scale', choices=('full','screen'), default='full', help='Screen uses nx<=64; endpoints unchanged, never pooled with full scale.')
    parser.add_argument('--prepared-revision',choices=('original','stability-v1','consistency-v1','sampling-recovery-v1'),default='sampling-recovery-v1',
                        help='Prepared preset revision; existing saved manifests retain their own explicit overrides.')
    parser.add_argument('--schemes', nargs='+', default=None,
                        help='Schemes included when generating a new immutable plan.')
    args = parser.parse_args()
    args.manifest = args.manifest or args.root / 'manifest.json'
    if args.action == 'plan':
        if args.profile == 'prepared':
            from prepared_fluidMatrix import make_prepared_plan, METHODS
            plan = make_prepared_plan(make_plan, args.schemes or METHODS, args.scale, args.prepared_revision)
        else:
            plan = make_plan(args.schemes or SCHEMES)
        write_json(args.manifest, plan)
        print(json.dumps(dict(slots=len(plan['rows']), supported=sum(not r['skip'] for r in plan['rows']))))
        return
    plan = json.loads(args.manifest.read_text())
    if args.action == 'run':
        from warpSPH.schemes.builder import rejectRetiredScheme
        for row in plan['rows']:
            rejectRetiredScheme(row.get('solver', row['scheme']))
        execute(args.root, plan)
    elif args.action == 'worker':
        worker(args.root, next(r for r in plan['rows'] if r['id'] == args.row))
    else:
        print(report(args.root, plan))


if __name__ == '__main__':
    main()
