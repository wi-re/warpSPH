"""Recorded compiled-Spheral/warpSPH BoxTension and gas/gas Sod controls.

Explicit harness, not execution of the bundled drivers. Preserves their states
with matched quintic kernels and identical saved inputs. Native support and AV/
compatible-energy differences are recorded, not assumed to be identical.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

PIN = '116c71f2ea559d569456ba694194e5a6de3c75f7'
SOURCE = Path(__file__).resolve().parents[1] / 'results/materialEOS/spheral-build/source'
FIELDS = ('positions', 'velocities', 'densities', 'pressures', 'internalEnergies',
          'supports', 'masses', 'materials', 'UIDs', 'kinds')


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def prepare(args):
    box = args.case == 'box'
    n = args.n or (100 if box else 400)
    if n % 4 or n < 32:
        raise ValueError('Resolution must be >=32 and divisible by four')
    gamma = ([1.5, 1.5] if box else [5/3, 5/3]) if args.gamma == 'equal' else [1.4, 5/3]
    volume = getattr(args, 'volume', None) or ('massOverDensity' if box else 'kernelSum')
    if box:
        xx, yy = np.meshgrid((np.arange(n)+.5)/n-.5, (np.arange(n)+.5)/n-.5, indexing='ij')
        x = np.stack((xx.ravel(), yy.ravel()), axis=1)
        ids = np.all(abs(x) < .25, axis=1).astype(np.int32)
        order = np.argsort(ids, kind='stable'); x, ids = x[order], ids[order]
        V = np.full(len(x), 1/n**2); rho = np.where(ids == 0, 1., 4.)
        pressure = np.ones(len(x)); domain = [-.5, .5]
    else:
        # Same dx as the driver's nx1/nx2=400/100 on [-.5,.5].
        # Pad each reservoir to length 1 and observe |x|<.5 before seam arrival.
        nl, nr = 2*n, n//2
        x = np.r_[-1+(np.arange(nl)+.5)/nl, (np.arange(nr)+.5)/nr][:, None]
        ids = (x[:, 0] >= 0).astype(np.int32)
        V = np.r_[np.full(nl, 1/nl), np.full(nr, 1/nr)]
        rho = np.where(ids == 0, 1., .25); pressure = np.where(ids == 0, 1., .1795)
        domain = [-1., 1.]
    # Input rounding is shared, not independently resampled in either solver.
    x = x.astype(np.float32)
    a = dict(positions=x, velocities=np.zeros_like(x), densities=rho.astype(np.float32),
             pressures=pressure.astype(np.float32),
             internalEnergies=(pressure/((np.array(gamma)[ids]-1)*rho)).astype(np.float32),
             supports=(4.05*V**(1/x.shape[1])).astype(np.float32),
             masses=(rho*V).astype(np.float32), materials=ids,
             UIDs=np.arange(len(x), dtype=np.int64), kinds=np.zeros(len(x), dtype=np.int32))
    end = args.time if args.time is not None else (7. if box else .15)
    dt = args.dt or (.001 if box else .0001)
    # 2.5 bounds the initial/star characteristics of both prescribed gas/gas
    # states. Reserve one full coarse support beyond their propagation distance.
    if not box and (end > .15 or end*2.5+float(a['supports'].max()) >= .5):
        raise ValueError('Sod interval exceeds this harness padded-seam qualification')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'input.npz', **a)
    save(args.output/'settings.json', dict(case=args.case, gammaVariant=args.gamma,
        gamma=gamma, nx=n, dim=x.shape[1], end=end, dt=dt, support=args.support, volumePolicy=volume,
        frameEvery=args.frame_every, referencePin=PIN, sourceRoot=str(SOURCE),
        domain=domain, kernel='QuinticSpline', spheralKernel='NBSplineKernel(5)/TableKernel(10000)',
        nPerh=1.35, extent=3., n_h=4.05, precision=dict(warp='float32', spheral='float64'),
        provenance='Explicit compiled-Spheral harness using bundled BoxTension/Sod states, not execution of the bundled drivers. Box coordinates translated by -.5. Sod reservoirs periodically padded. Same saved input, synchronous midpoint/fixed dt, initial H iteration off. Native support/AV/compatible energy differ.'))
    print('Prepared', args.output, len(x), 'particles', flush=True)


class Measurements:
    def __init__(self, settings, initial):
        self.settings, self.initial = settings, initial
        mass, u, velocity = [np.asarray(initial[k], dtype=np.float64) for k in ('masses', 'internalEnergies', 'velocities')]
        self.energy = float(np.sum(mass*(u+.5*np.sum(velocity**2, axis=1))))
        self.material_mass = [float(initial['masses'][initial['materials']==k].sum(dtype=np.float64)) for k in (0, 1)]
        self.exact = None
        if settings['case'] == 'sod':
            path = Path(settings['sourceRoot'])/'tests/functional/Hydro/Sod/SodAnalyticSolution.py'
            spec = importlib.util.spec_from_file_location('spheral_sod_exact', path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            self.exact = module.SodSolutionGasGas(gamma1=settings['gamma'][0], gamma2=settings['gamma'][1])

    def __call__(self, a, t):
        # Widen reductions before arithmetic to avoid float32 accumulation noise.
        mass, velocity, rho, u = [np.asarray(a[k], dtype=np.float64) for k in ('masses', 'velocities', 'densities', 'internalEnergies')]
        energy = float(np.sum(mass*(u+.5*np.sum(velocity**2, axis=1))))
        row = dict(time=float(t), relativeEnergyDrift=energy/self.energy-1,
            minDensity=float(rho.min()), minEnergy=float(u.min()), minPressure=float(a['pressures'].min()),
            maxVelocity=float(np.linalg.norm(velocity, axis=1).max()),
            materialMassError=float(max(abs(mass[a['materials']==k].sum()/self.material_mass[k]-1) for k in (0, 1))))
        if self.exact is None:
            d = np.asarray(a['positions'], dtype=np.float64)-self.initial['positions']; d -= np.round(d)
            row.update(pressureError=float(abs(a['pressures']-1).max()),
                densityRelativeL1=float(np.mean(abs(rho/self.initial['densities']-1))),
                shapeDisplacement=float(np.linalg.norm(d, axis=1).max()))
        else:
            mask = abs(a['positions'][:, 0]) < .5
            x = a['positions'][mask, 0].astype(float)
            _, v, _, exactrho, p, _, _ = self.exact.solution(float(t), x=x)
            weights = mass[mask]/rho[mask]
            for name, value, exact in [('density', rho[mask], exactrho), ('pressure', a['pressures'][mask], p), ('velocity', velocity[mask, 0], v)]:
                row[name+'L1'] = float(np.dot(weights, abs(value-np.array(exact)))/weights.sum())
        return row


def summarize(root, solver, settings, history, initial, final, wall, steps, remote, extra):
    maxima = {k: max(abs(row[k]) for row in history) for k in history[0] if k != 'time'}
    # Clock roundoff can leave the nominal endpoint just below tLimit, causing
    # the fixed-step runner to take one more step. Permit that final step.
    gates = dict(completed=abs(history[-1]['time']-settings['end']) <= settings['dt']*1.01,
        finitePositive=all(np.isfinite(v) for row in history for v in row.values()) and
            min(row['minDensity'] for row in history)>0 and min(row['minEnergy'] for row in history)>0 and
            min(row['minPressure'] for row in history)>0,
        energy=maxima['relativeEnergyDrift']<1.e-4, materialMass=maxima['materialMassError']<1.e-6,
        identities=all(np.array_equal(initial[k], final[k]) for k in ('UIDs', 'materials', 'masses')))
    if settings['case'] == 'box':
        gates.update(pressure=maxima['pressureError']<.01, quiescent=maxima['maxVelocity']<.01,
                     shape=maxima['shapeDisplacement']<.01, sharpDensity=maxima['densityRelativeL1']<1.e-3)
    else:
        gates.update({k: history[-1][k]<.04 for k in ('densityL1', 'pressureL1', 'velocityL1')})
    gates = {k: bool(v) for k, v in gates.items()}
    row = dict(solver=solver, settings=settings, remoteRunID=remote, time=history[-1]['time'],
        steps=steps, wallSeconds=wall, meanStepMilliseconds=1000*wall/max(steps, 1),
        timingScope='Reference integration/diagnostics/publishing; excludes startup and post-rendered video.' if solver=='Spheral' else 'Full runner wall including initialization, plotting, storage and video encoding; step-only timing reported separately.',
        maxima=maxima, final=history[-1], gates=gates, passed=all(gates.values()), **extra)
    if solver == 'Spheral': row['integrationWallSeconds'] = wall
    np.savez_compressed(root/'initialState.npz', **initial)
    np.savez_compressed(root/'finalState.npz', **final)
    save(root/'diagnosticHistory.json', history); save(root/'comparisonRun.json', row)
    print(json.dumps(row, indent=2), flush=True)
    return row


def warp_context(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.hydrostatic import hydrostaticCase
    from warpSPH.cases.compressible.twoGasSod import twoGasSodCase
    from study_materialCases import materials
    box = settings['case'] == 'box'; case = hydrostaticCase if box else twoGasSodCase
    params = dict(viscositySwitch='NoneSwitch', adaptiveSupportScheme='NoScheme' if settings['support']=='fixed' else 'Owen')
    if box: params.update(rho_high=4., gamma=settings['gamma'][0])
    # The bundled Sod driver explicitly selects RKSumVolume; BoxTension leaves
    # the controller's mass/rho default. Existing saved controls without this
    # key retain their original mass/rho schedule.
    kernel_sum = settings.get('volumePolicy', 'massOverDensity') == 'kernelSum'
    solver = dict(materials=materials(*settings['gamma']), crkDensityPolicy='sameMaterial',
                  crkEvolution='summation' if kernel_sum else 'spheral')
    simulation = dict(nx=settings['nx'], tLimit=settings['end'], kernel='QuinticSpline', n_h=4.05,
                      dt=settings['dt'], adaptiveDt=False, execution='eager', cflFactor=.15)
    ctx = RunConfiguration(case.name, simulation=simulation, params=params, solver=solver).prepare(case)
    return case, ctx, simulation, params, solver


def host(state):
    return {k: getattr(state, k).detach().cpu().numpy().copy() for k in FIELDS}


def retain_failure(root, error, remote, history, initial, final, solver,
                   settings, summary_factory, wall, steps, extra):
    """Keep the actual stop state separately from periodic recording frames."""
    np.savez_compressed(root/'initialState.npz', **initial)
    np.savez_compressed(root/'failedState.npz', **final)
    failure = dict(error=str(error), remoteRunID=remote, time=history[-1]['time'] if history else None,
                   steps=int(steps), wallSeconds=wall, stateProvenance=(
                       'actual state at exception' if history else 'initial input fallback; no diagnostic state captured'))
    save(root/'failure.json', failure)
    # A finite negative-energy stop still has fully usable diagnostics and
    # geometry. Never turn that partial trajectory into a completed baseline.
    if history and all(np.isfinite(v) for row in history for v in row.values()):
        row = summary_factory(root, solver, settings, history, initial, final, wall, steps, remote,
                              dict(**extra, termination='failed', failure=str(error)))
        row['gates']['execution'] = False; row['passed'] = False
        save(root/'comparisonRun.json', row)
    else:
        clean = [{k: v if np.isfinite(v) else None for k, v in row.items()} for row in history]
        save(root/'diagnosticHistory.json', clean)


def warp_run(args, settings, context_factory=warp_context, measurement_factory=Measurements,
             summary_factory=summarize, name_prefix='Material baseline', build_factory=None,
             input_file='input.npz', initialize_factory=None):
    from dataclasses import replace
    case, ctx, simulation, params, solver = context_factory(settings)
    import torch
    from warpSPH.runner import run
    from warpSPH.modules.eos import refreshEOS
    from warpsph_remote.publisher import SimulationPublisher, SimulationStopped
    root = args.output/'warp'; root.mkdir(exist_ok=False)
    a = dict(np.load(args.output/input_file)); measure = measurement_factory(settings, a)
    def build(c):
        c.config.adaptiveDt = True
        system = case.buildSystem(c) if build_factory is None else build_factory(c, a, settings)
        for k, value in a.items():
            setattr(system.state, k, torch.as_tensor(value, device=c.device).clone())
        s = system.state; s.crkMassDensities = s.densities.clone(); s.crkVolumes = s.masses/s.densities
        s.totalEnergies = s.masses*(s.internalEnergies+.5*s.velocities.square().sum(-1))
        refreshEOS(s, c.schemeConfig)
        if initialize_factory is not None:initialize_factory(c, system, a, settings)
        c.config.dt = settings['dt']; c.config.adaptiveDt = False
        # Discard sampler topology built for its independently ordered lattice.
        system.adjacency = None
        return system
    history = []; latest = {}; start = time.perf_counter()
    def diagnostics(c, system):
        frame = host(system.state); row = measure(frame, float(system.t)); history.append(row)
        latest.update(frame=frame, ctx=c, steps=len(history)-1)
        if not all(np.isfinite(v).all() for v in frame.values()) or row['minDensity']<=0 or row['minEnergy']<=0 or row['maxVelocity']>100:
            raise RuntimeError('warpSPH finite-state/velocity watchdog')
        return {k: v for k, v in row.items() if k != 'time'}
    publisher = SimulationPublisher(name_prefix+' warp '+settings['case']+' '+settings['gammaVariant'], measurements=True)
    print('Remote run registered:', publisher.id, flush=True)
    save(root/'registration.json', dict(remoteRunID=publisher.id, pid=os.getpid()))
    try:
        result = run(publisher.bind_case(replace(case, buildSystem=build, diagnostics=diagnostics)),
            **simulation, params=params, schemeOverrides=solver,
            plot=True, video=True, plotBackend='vispy', progress=True, plotInterval=settings['frameEvery'],
            store=True, storeMode='trajectory', storeInterval=settings['frameEvery'],
            exportInterval=settings['frameEvery']*settings['dt'],
            holdPlot=False, show=False, exportRoot=str(root), stallProgress=1.e-3, velocityAlarmPlotInterval=1)
        publisher.recording(result.ctx); publisher.finish(result.ctx, result.state, result.nSteps)
        row = summary_factory(root, 'warpSPH', settings, history, a, host(result.state.state), result.wallTime,
            result.nSteps, publisher.id, dict(exportPath=str(result.ctx.exportPath),
                runnerMeanStepMilliseconds=float(result.series('stepTime_ms').mean()),
                viscosity={k: float(getattr(result.ctx.schemeConfig.diffusionParams, k)) for k in ('C_l', 'C_q')},
                meanVolumeWeights=bool(result.ctx.schemeConfig.crkViscosityParams.meanVolumeWeights),
                policy=settings.get('comparisonPolicy','sameMaterial; '+('native geometric volumes per RHS' if settings.get('volumePolicy')=='kernelSum' else 'spheral mass/rho volume/stage schedule')+'; native Owen supports/CRK viscosity and compatible energy')))
        publisher.close('completed' if row['passed'] else 'failed', '' if row['passed'] else 'Baseline qualification failed; comparisonRun.json')
    except BaseException as error:
        retain_failure(root, error, publisher.id, history, a, latest.get('frame', a), 'warpSPH',
            settings, summary_factory, time.perf_counter()-start, latest.get('steps', 0),
            dict(exportPath=str(latest['ctx'].exportPath)) if 'ctx' in latest else {})
        if 'ctx' in latest: publisher.recording(latest['ctx'])
        publisher.close('stopped' if isinstance(error, SimulationStopped) else 'failed', str(error)); raise
    return 0 if row['passed'] else 1


def spheral_run(args, settings, measurement_factory=Measurements, summary_factory=summarize,
                name_prefix='Material baseline', integrator_factory=None):
    sys.path.insert(0, str(args.publisher_path))
    import h5py
    import importlib
    s = importlib.import_module('Spheral'+str(settings['dim'])+'d')
    from warpsph_remote.publisher import SimulationPublisher, SimulationStopped
    root = args.output/'spheral'; root.mkdir(exist_ok=False)
    a = dict(np.load(args.output/'input.npz')); dim = settings['dim']; dt = settings['dt']
    extent = settings.get('extent', 3.)
    W = s.TableKernel(s.BSplineKernel() if settings.get('kernel')=='CubicSpline' else s.NBSplineKernel(settings.get('spheralKernelOrder',5)), 10000)
    db = s.DataBase(); nodes = []; eos = []; indices = []
    for material, gamma in enumerate(settings['gamma']):
        eos.append(s.GammaLawGasMKS(gamma, 1.))
        n = s.makeFluidNodeList('gas'+str(material), eos[-1], nPerh=settings.get('nPerh', 1.35), hmin=1.e-8, hmax=1., kernelExtent=extent)
        idx = np.flatnonzero(a['materials']==material); indices.append(idx); n.numInternalNodes = len(idx)
        for j, i in enumerate(idx):
            n.positions()[j] = s.Vector(*map(float, a['positions'][i]))
            n.velocity()[j] = s.Vector(*map(float, a['velocities'][i]))
            n.mass()[j] = float(a['masses'][i]); n.massDensity()[j] = float(a['densities'][i])
            n.specificThermalEnergy()[j] = float(a['internalEnergies'][i])
            h = extent/float(a['supports'][i])
            n.Hfield()[j] = s.SymTensor(h) if dim == 1 else s.SymTensor(h, 0., 0., h)
        nodes.append(n); db.appendNodeList(n)
    smoothing = s.FixedSmoothingScale() if settings['support']=='fixed' else s.SPHSmoothingScale(s.IdealH, W)
    hydro = s.CRKSPH(dataBase=db, W=W, order=s.LinearOrder, cfl=.15, XSPH=False,
        compatibleEnergyEvolution=settings.get('compatibleEnergy', True), evolveTotalEnergy=False,
        densityUpdate=s.RigorousSumDensity, smoothingScaleMethod=smoothing)
    if 'referenceViscosity' in settings:
        for key, value in settings['referenceViscosity'].items(): setattr(hydro.Q, key, value)
    boundaries = []  # Spheral Physics stores boundary references, not ownership.
    axes = np.array(settings['domainAxes'] if 'domainAxes' in settings else [settings['domain']]*dim)
    for axis in range(dim):
        lo = [0.]*dim; hi = [0.]*dim; normal = [0.]*dim
        lo[axis], hi[axis] = axes[axis]; normal[axis] = 1.
        planes = [s.Plane(s.Vector(*lo), s.Vector(*normal)), s.Plane(s.Vector(*hi), s.Vector(*[-v for v in normal]))]
        bcs = [s.ReflectingBoundary(plane) for plane in planes] if settings.get('boundary')=='reflecting' else [s.PeriodicBoundary(*planes)]
        for bc in bcs: boundaries.append(bc); hydro.appendBoundary(bc)
    integrator = (s.SynchronousRK2Integrator(db) if integrator_factory is None else
                  integrator_factory(s, db, hydro, nodes, settings, root))
    integrator.appendPhysicsPackage(hydro)
    integrator.lastDt = dt; integrator.dtMin = dt; integrator.dtMax = dt; integrator.dtGrowth = 1.
    volume_options = {'volumeType': s.RKSumVolume} if settings.get('volumePolicy')=='kernelSum' else {}
    control = s.SpheralController(integrator, kernel=W, iterateInitialH=False, **volume_options,
        statsStep=100, printStep=100, restartStep=None, restartBaseName=str(root/'restart'),
        restoreCycle=-1, skipInitialPeriodicWork=True)
    def arrays():
        out = {k: [] for k in FIELDS}
        for material, n in enumerate(nodes):
            p, v = n.positions().internalValues(), n.velocity().internalValues()
            rho = np.asarray(n.massDensity().internalValues(), dtype=np.float64)
            u = np.asarray(n.specificThermalEnergy().internalValues(), dtype=np.float64)
            count = n.numInternalNodes
            vals = dict(positions=np.column_stack([[point[d] for point in p] for d in range(dim)]),
                velocities=np.column_stack([[point[d] for point in v] for d in range(dim)]), densities=rho,
                pressures=(settings['gamma'][material]-1)*rho*u, internalEnergies=u,
                supports=np.array([extent/h.xx for h in n.Hfield().internalValues()]),
                masses=np.asarray(n.mass().internalValues(), dtype=np.float64),
                materials=np.full(count, material, dtype=np.int64),
                UIDs=a['UIDs'][indices[material]], kinds=np.zeros(count, dtype=np.int64))
            for k in FIELDS: out[k].append(vals[k])
        return {k: np.concatenate(v) for k, v in out.items()}
    initial = arrays()
    for k in FIELDS:
        np.testing.assert_allclose(initial[k], a[k], rtol=2.e-7 if k=='pressures' else 1.e-14, atol=2.e-7 if k=='pressures' else 1.e-14, err_msg=k)
    measure = measurement_factory(settings, initial)
    publisher = SimulationPublisher(name_prefix+' Spheral '+settings['case']+' '+settings['gammaVariant'], measurements=True)
    print('Remote run registered:', publisher.id, flush=True)
    save(root/'registration.json', dict(remoteRunID=publisher.id, pid=os.getpid(), threads=os.environ.get('OMP_NUM_THREADS')))
    file = h5py.File(root/'trajectory.h5', 'w'); history = []; last_frame = -1
    ctx = NS(spec=NS(store=True, storeMode='trajectory', tLimit=settings['end']),
        scratch={'trajectoryFile': file}, exportPath=str(root.resolve()), scheme='Spheral CRKSPH',
        config=NS(dt=dt, domain=NS(min=axes[:, 0], max=axes[:, 1], periodic=np.full(dim, settings.get('boundary')!='reflecting', dtype=bool)), kernel=settings.get('kernel', 'QuinticSpline'), integrationScheme='SynchronousRK2'))
    last_progress = [0., time.monotonic()]
    def record(cycle, t, current_dt):
        nonlocal last_frame
        frame = arrays(); row = measure(frame, float(t)); history.append(row)
        if not all(np.isfinite(v).all() for v in frame.values()) or row['minDensity']<=0 or row['minEnergy']<=0 or row['maxVelocity']>100:
            raise RuntimeError('Spheral finite-state/velocity watchdog')
        if t > last_progress[0]: last_progress[:] = [t, time.monotonic()]
        elif time.monotonic()-last_progress[1]>60: raise RuntimeError('Spheral time-progress watchdog')
        if cycle % settings['frameEvery']==0 or t>=settings['end']-1.e-12:
            if cycle != last_frame:
                g = file.require_group('states').create_group(f'frame_{cycle:07d}'); g.attrs['time'] = float(t)
                for k, v in frame.items(): g.create_dataset(k, data=v)
                file.flush(); last_frame = cycle
            print(f'cycle={cycle} | t={t:.7f} | dt={current_dt:.3g} | speed={row["maxVelocity"]:.6g} | energy={row["relativeEnergyDrift"]:.3g} | wall={time.perf_counter()-start:.2f}s', flush=True)
        ctx.config.dt = current_dt; publisher.post_step(ctx, NS(state=NS(**frame), t=float(t)), cycle)
    start = time.perf_counter()
    try:
        record(0, 0., dt); control.appendPeriodicWork(record, 1)
        if settings.get('timeGrid'):
            grid = np.asarray(settings['timeGrid'], dtype=float)
            if (np.any(grid<=0) or not np.isfinite(grid).all() or
                len(grid)!=settings.get('exactSteps') or
                not np.isclose(grid.sum(), settings['end'], rtol=0, atol=1.e-12) or
                not np.isclose(grid[0], dt, rtol=0, atol=1.e-15)):
                raise ValueError('Invalid shared timestep grid')
            def next_dt(cycle, t, previous_dt):
                if cycle < len(grid):
                    value = float(grid[cycle])
                    integrator.lastDt = value
                    integrator.dtMin = value; integrator.dtMax = value
            control.appendPeriodicWork(next_dt, 1)
        if settings.get('exactSteps') is not None:
            control.step(int(settings['exactSteps']))
        else:
            control.advance(settings['end'], maxSteps=int(np.ceil(settings['end']/dt))+3)
        wall = time.perf_counter()-start; final = arrays()
        publisher.finish(ctx, NS(state=NS(**final), t=control.time()), control.totalSteps)
        row = summary_factory(root, 'Spheral', settings, history, initial, final, wall, control.totalSteps, publisher.id,
            dict(threads=os.environ.get('OMP_NUM_THREADS'), viscosity={k: float(getattr(hydro.Q, k)) for k in ('Cl', 'Cq', 'etaCritFrac', 'etaFoldFrac')},
                 policy='RigorousSumDensity; controller '+('RKSumVolume' if settings.get('volumePolicy')=='kernelSum' else 'MassOverDensity')+'; native IdealH/limited viscosity/'+('compatible energy' if settings.get('compatibleEnergy', True) else 'standalone thermal du/dt')))
        file.close(); publisher.close('completed' if row['passed'] else 'failed', '' if row['passed'] else 'Baseline qualification failed; comparisonRun.json')
    except BaseException as error:
        file.close()
        retain_failure(root, error, publisher.id, history, initial, arrays(), 'Spheral', settings,
            summary_factory, time.perf_counter()-start, control.totalSteps,
            dict(threads=os.environ.get('OMP_NUM_THREADS')))
        publisher.close('stopped' if isinstance(error, SimulationStopped) else 'failed', str(error)); raise
    return 0 if row['passed'] else 1


def render(args, settings, context_factory=warp_context):
    case, ctx, _, _, _ = context_factory(settings)
    import h5py
    import torch
    from warpSPH.runner.media import encodeFrames
    from warpSPH.runner.display import closeWindow
    root = args.output/'spheral'; ctx.imagePath = str(root/'images'); Path(ctx.imagePath).mkdir(exist_ok=True)
    ctx.exportPath = str(root); ctx.scratch['_titleProvenance'] = 'Spheral '+PIN[:7]+' | '+settings.get('kernel', 'QuinticSpline')+' | '+settings['support']+' | '+settings['gammaVariant']
    plotter = None
    with h5py.File(root/'trajectory.h5', 'r') as f:
        for number, key in enumerate(sorted(f['states'])):
            group = f['states'][key]; a = {k: torch.as_tensor(v[()], device=ctx.device) for k, v in group.items()}
            system = NS(state=NS(**a), t=float(group.attrs['time']))
            if plotter is None: plotter = case.setupPlot(ctx, system)
            else: case.updatePlot(ctx, system, plotter, number)
    if plotter:
        if hasattr(plotter, 'drain'): plotter.drain()
        closeWindow(plotter)
    print(encodeFrames(ctx.imagePath, ctx.exportPath), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'warp', 'spheral', 'render'))
    parser.add_argument('--case', choices=('box', 'sod'), default='box')
    parser.add_argument('--gamma', choices=('equal', 'unequal'), default='equal')
    parser.add_argument('--n', type=int); parser.add_argument('--time', type=float); parser.add_argument('--dt', type=float)
    parser.add_argument('--support', choices=('fixed', 'adaptive'), default='adaptive')
    parser.add_argument('--volume', choices=('massOverDensity', 'kernelSum'), help='Default: published driver selection (mass/rho box, kernel sum Sod).')
    parser.add_argument('--frame-every', type=int, default=100)
    parser.add_argument('--publisher-path', type=Path, default=Path('/home/lu26029/dev/warpSPHRemote/python'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare': return prepare(args)
    settings = json.loads((args.output/'settings.json').read_text())
    return {'warp': warp_run, 'spheral': spheral_run, 'render': render}[args.mode](args, settings)


if __name__ == '__main__':
    raise SystemExit(main())
