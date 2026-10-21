"""Matched, daemon-visible Spheral/warpSPH stationary material contacts.

Use the warp environment for prepare/warp/render/report and the isolated
Spheral launcher for spheral. Each run is explicit: failed equilibrium gates
are recorded as failures and never silently launch another simulation.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np


FIELDS = ('positions', 'velocities', 'densities', 'pressures', 'internalEnergies',
          'supports', 'masses', 'materials', 'UIDs', 'kinds')
DT = 4.e-5
END = .02


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def measures(a, initial_energy):
    mask = np.abs(a['positions'][:, 0]) < .5
    energy = np.sum(a['masses']*(a['internalEnergies']+.5*(a['velocities']**2).sum(1)))
    return dict(pressureError=float(np.max(np.abs(a['pressures'][mask]-1))),
                maxPeculiarVelocity=float(np.max(np.abs(a['velocities'][mask]))),
                relativeEnergyDrift=float(energy/initial_energy-1),
                minDensity=float(a['densities'].min()), minEnergy=float(a['internalEnergies'].min()))


def finish(root, solver, sampling, support, history, initial, wall, steps, remote, acceleration):
    rows = dict(solver=solver, sampling=sampling, support=support, steps=steps,
        time=history[-1]['time'], wallSeconds=wall, meanStepMilliseconds=wall/steps*1000,
        remoteRunID=remote, initialMaxAcceleration=acceleration, final=history[-1],
        maxPressureError=max(r['pressureError'] for r in history),
        maxPeculiarVelocity=max(r['maxPeculiarVelocity'] for r in history),
        maxEnergyDrift=max(abs(r['relativeEnergyDrift']) for r in history))
    rows['gates'] = dict(completed=abs(rows['time']-END) < DT*1.1,
        finitePositive=all(np.isfinite(v) for r in history for v in r.values()) and
            min(r['minDensity'] for r in history)>0 and min(r['minEnergy'] for r in history)>0,
        pressure=rows['maxPressureError']<.01, quiescent=rows['maxPeculiarVelocity']<.01,
        energy=rows['maxEnergyDrift']<1.e-4)
    rows['passed'] = all(rows['gates'].values())
    np.savez_compressed(root/'initialState.npz', **initial)
    write_json(root/'diagnosticHistory.json', history)
    write_json(root/'comparisonRun.json', rows)
    print(json.dumps(rows, indent=2), flush=True)
    return rows


def warp_context(args):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.twoGasSod import twoGasSodCase
    from study_materialCases import materials
    settings = dict(simulation=dict(nx=200, tLimit=END, execution='eager', kernel='CubicSpline',
                                   n_h=4., dt=DT, adaptiveDt=False),
        params=dict(pressureRight=1., equalMass=args.sampling=='equalMass',
                    adaptiveSupportScheme='NoScheme' if args.support=='fixed' else 'Owen'),
        solver=dict(materials=materials(), crkDensityPolicy='sameMaterial', crkEvolution='spheral'))
    return twoGasSodCase, settings, RunConfiguration('twoGasSod', **settings).prepare(twoGasSodCase)


def host(state):
    return {k: getattr(state, k).detach().cpu().numpy().copy() for k in FIELDS}


def prepare(args):
    case, settings, ctx = warp_context(args)
    ctx.config.adaptiveDt = True  # Sampler computes its own initial CFL step.
    system = case.buildSystem(ctx)
    a = host(system.state)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'input.npz', **a)
    write_json(args.output/'settings.json', dict(**settings, sampling=args.sampling, support=args.support,
        referencePin='116c71f2ea559d569456ba694194e5a6de3c75f7',
        nPerh=2., extent=2., precision=dict(warp='float32', spheral='float64')))
    print('Prepared identical particle input:', args.output/'input.npz', flush=True)


def warp_run(args):
    from dataclasses import replace
    case, settings, ctx = warp_context(args)
    import torch
    from warpSPH.runner import run
    from warpSPH.schemes.compressible.crkSPH import crkSPH_step
    from warpsph_remote.publisher import SimulationPublisher, SimulationStopped
    root = args.output/'warp'; root.mkdir(exist_ok=False)
    input_state = dict(np.load(args.output/'input.npz'))
    def build(c):
        c.config.adaptiveDt = True
        system = case.buildSystem(c)
        for k, v in input_state.items():
            setattr(system.state, k, torch.as_tensor(v, device=c.device).clone())
        system.state.crkVolumes = system.state.masses/system.state.densities
        system.state.crkMassDensities = system.state.densities.clone()
        c.config.dt = DT
        c.config.adaptiveDt = False
        return system
    # A separate initial RHS probe never advances the production trajectory.
    probe = build(ctx)
    probe.initialize(DT, config=ctx.config, schemeConfig=ctx.schemeConfig, stageWeights=(0., 1.))
    update, _, evaluated = crkSPH_step(probe, DT, ctx.config, ctx.schemeConfig, stageIndex=0)
    mask = evaluated.positions[:, 0].abs()<.5
    acceleration = float(update.dvdt[mask].abs().max())
    np.savez_compressed(root/'initialRHS.npz', **host(evaluated), acceleration=update.dvdt.detach().cpu().numpy())
    publisher = SimulationPublisher('Spheral comparison warp '+args.sampling+' '+args.support,
                                    measurements=True, diagnostic_interval=.001)
    print('Remote run registered:', publisher.id, flush=True)
    energy = float(np.sum(input_state['masses']*input_state['internalEnergies']))
    history = []
    def diagnostics(c, system):
        row = dict(time=float(system.t), **measures(host(system.state), energy))
        history.append(row)
        return {k:v for k,v in row.items() if k!='time'}
    try:
        result = run(publisher.bind_case(replace(case, buildSystem=build, diagnostics=diagnostics)),
            nx=200, tLimit=END, nSteps=500, dt=DT, adaptiveDt=False, n_h=4., kernel='CubicSpline',
            execution='eager', params=settings['params'], schemeOverrides=settings['solver'],
            plot=True, video=True, plotBackend='vispy', progress=True, plotInterval=25,
            store=True, storeMode='trajectory', storeInterval=25, holdPlot=False, show=False,
            exportRoot=str(root), stallProgress=1.e-3, velocityAlarmPlotInterval=1)
        publisher.recording(result.ctx)
        publisher.finish(result.ctx, result.state, result.nSteps)
        arrays = host(result.state.state)
        np.savez_compressed(root/'finalState.npz', **arrays)
        row = finish(root, 'warpSPH', args.sampling, args.support, history, input_state,
                     result.wallTime, result.nSteps, publisher.id, acceleration)
        row['runnerMeanStepMilliseconds'] = float(result.series('stepTime_ms').mean())
        row['exportPath'] = str(result.ctx.exportPath)
        write_json(root/'comparisonRun.json', row)
        publisher.close('completed' if row['passed'] else 'failed', '' if row['passed'] else 'Equilibrium qualification failed; comparisonRun.json')
    except BaseException as e:
        publisher.close('stopped' if isinstance(e, SimulationStopped) else 'failed', str(e))
        raise
    return 0 if row['passed'] else 1


def spheral_run(args):
    # Only the pure NumPy companion adapter is shared with the reference.
    sys.path.insert(0, str(args.publisher_path))
    import h5py
    import Spheral1d as s
    from warpsph_remote.publisher import SimulationPublisher, SimulationStopped
    root=args.output/'spheral'; root.mkdir(exist_ok=True)
    if (root/'trajectory.h5').exists():
        raise FileExistsError('Preserve the existing reference trajectory; choose a fresh output')
    a=dict(np.load(args.output/'input.npz'))
    W=s.TableKernel(s.BSplineKernel(), 10000)
    db=s.DataBase()
    nodes=[]; eos=[]
    for material, gamma in enumerate((1.4, 5./3.)):
        eos.append(s.GammaLawGasMKS(gamma, 1.))
        n=s.makeFluidNodeList('gas'+str(material), eos[-1], nPerh=2., hmin=1.e-8, hmax=1.)
        idx=np.flatnonzero(a['materials']==material)
        n.numInternalNodes=len(idx)
        for j, i in enumerate(idx):
            n.positions()[j]=s.Vector(float(a['positions'][i,0]))
            n.mass()[j]=float(a['masses'][i])
            n.massDensity()[j]=float(a['densities'][i])
            n.velocity()[j]=s.Vector(0.)
            n.specificThermalEnergy()[j]=float(a['internalEnergies'][i])
            n.Hfield()[j]=s.SymTensor(2./float(a['supports'][i]))
        nodes.append(n); db.appendNodeList(n)
    smoothing = s.FixedSmoothingScale() if args.support=='fixed' else s.SPHSmoothingScale(s.IdealH, W)
    hydro=s.CRKSPH(dataBase=db, W=W, order=s.LinearOrder, cfl=.15, XSPH=False,
        compatibleEnergyEvolution=True, densityUpdate=s.RigorousSumDensity,
        smoothingScaleMethod=smoothing)
    bc=s.PeriodicBoundary(s.Plane(s.Vector(-1.),s.Vector(1.)), s.Plane(s.Vector(1.),s.Vector(-1.)))
    hydro.appendBoundary(bc)
    integrator=s.SynchronousRK2Integrator(db)
    integrator.appendPhysicsPackage(hydro)
    integrator.lastDt=DT; integrator.dtMin=DT; integrator.dtMax=DT; integrator.dtGrowth=1.
    control=s.SpheralController(integrator, kernel=W, iterateInitialH=False,
        statsStep=100, printStep=100, restartStep=None, restartBaseName=str(root/'restart'),
        restoreCycle=-1, skipInitialPeriodicWork=True)
    def arrays():
        out={k:[] for k in FIELDS}
        for material,n in enumerate(nodes):
            for j in range(n.numInternalNodes):
                rho=n.massDensity()[j]; u=n.specificThermalEnergy()[j]
                vals=dict(positions=[n.positions()[j].x], velocities=[n.velocity()[j].x],
                    densities=rho, pressures=(eos[material].gamma-1)*rho*u,
                    internalEnergies=u, supports=2./n.Hfield()[j].xx, masses=n.mass()[j],
                    materials=material, UIDs=len(out['UIDs']), kinds=0)
                for k in FIELDS: out[k].append(vals[k])
        return {k:np.asarray(v, dtype=np.int64 if k in ('UIDs','materials','kinds') else np.float64) for k,v in out.items()}
    initial=arrays()
    for k in FIELDS:
        if not np.array_equal(initial[k], a[k]):
            # EOS reconstruction and reciprocal H roundoff can differ by double epsilon.
            tolerance = 2.e-7 if k=='pressures' else 1.e-14
            np.testing.assert_allclose(initial[k], a[k], rtol=tolerance, atol=tolerance, err_msg=k)
    packages=integrator.physicsPackages()
    state=s.State(db,packages); derivs=s.StateDerivatives(db,packages)
    integrator.preStepInitialize(state,derivs)
    integrator.initializeDerivatives(0.,DT*.5,state,derivs)
    derivs.Zero()
    integrator.evaluateDerivatives(0.,DT*.5,db,state,derivs)
    integrator.finalizeDerivatives(0.,DT*.5,db,state,derivs)
    rhs=arrays()
    acc=np.asarray([[hydro.DvDt[k][j].x] for k,n in enumerate(nodes) for j in range(n.numInternalNodes)])
    acceleration=float(np.max(np.abs(acc[np.abs(rhs['positions'][:,0])<.5])))
    np.savez_compressed(root/'initialRHS.npz', **rhs, acceleration=acc)
    # Recreate state for the regular step; the initial audit has not advanced time.
    publisher=SimulationPublisher('Spheral comparison reference '+args.sampling+' '+args.support,
        measurements=True, diagnostic_interval=.001)
    print('Remote run registered:',publisher.id,flush=True)
    energy=float(np.sum(initial['masses']*initial['internalEnergies']))
    history=[]
    file=h5py.File(root/'trajectory.h5','w')
    ctx=NS(spec=NS(store=True,storeMode='trajectory',tLimit=END), scratch={'trajectoryFile':file},
        exportPath=str(root.resolve()), scheme='Spheral CRKSPH',
        config=NS(dt=DT,domain=NS(min=np.array([-1.]),max=np.array([1.]),periodic=np.array([True])),
                  kernel='CubicSpline',integrationScheme='SynchronousRK2'))
    last_frame=-1
    start=time.perf_counter()
    def record(cycle,t,dt):
        nonlocal last_frame
        frame=arrays(); t=float(t)
        row=dict(time=t, **measures(frame, energy)); history.append(row)
        if cycle%25==0 or t>=END-1.e-12:
            if cycle!=last_frame:
                g=file.require_group('states').create_group(f'frame_{cycle:05d}'); g.attrs['time']=t
                for k,v in frame.items(): g.create_dataset(k,data=v)
                file.flush(); last_frame=cycle
        ctx.config.dt=dt
        publisher.post_step(ctx,NS(state=NS(**frame),t=t),cycle)
        if cycle%25==0:
            print(f'cycle={cycle} | t={t:.8f} | pressureError={row["pressureError"]:.6g} | maxPeculiarVelocity={row["maxPeculiarVelocity"]:.6g}',flush=True)
        if not all(np.isfinite(v).all() for v in frame.values()) or row['minDensity']<=0 or row['minEnergy']<=0:
            raise RuntimeError('Nonfinite or nonpositive Spheral state')
    try:
        record(0,0.,DT)
        control.appendPeriodicWork(record,1)
        control.advance(END,maxSteps=501)
        wall=time.perf_counter()-start
        final=arrays()
        publisher.finish(ctx,NS(state=NS(**final),t=control.time()),control.totalSteps)
        np.savez_compressed(root/'finalState.npz',**final)
        row=finish(root,'Spheral',args.sampling,args.support,history,initial,wall,
                   control.totalSteps,publisher.id,acceleration)
        write_json(root/'settings.json',dict(kernel='BSplineKernel/TableKernel(10000)',nPerh=2.,
            HUpdate='FixedH' if args.support=='fixed' else 'IdealH',volume='controller default MassOverDensity',
            dt=DT,gamma=[1.4,5./3.],viscosity=dict(Cl=hydro.Q.Cl,Cq=hydro.Q.Cq),precision='float64'))
        file.close()
        publisher.close('completed' if row['passed'] else 'failed','' if row['passed'] else 'Equilibrium qualification failed; comparisonRun.json')
    except BaseException as e:
        file.close(); publisher.close('stopped' if isinstance(e,SimulationStopped) else 'failed',str(e)); raise
    return 0 if row['passed'] else 1


def render(args):
    case, _, ctx=warp_context(args)
    import torch
    import h5py
    from warpSPH.runner.media import encodeFrames
    from warpSPH.runner.display import closeWindow
    root=args.output/'spheral'
    ctx.imagePath=str(root/'images'); Path(ctx.imagePath).mkdir(exist_ok=True)
    ctx.exportPath=str(root)
    ctx.scratch['_titleProvenance']='Spheral 116c71f | cubic spline | midpoint | '+args.support
    plotter=None
    with h5py.File(root/'trajectory.h5','r') as f:
        for i,key in enumerate(sorted(f['states'])):
            g=f['states'][key]
            a={k:torch.as_tensor(v[()],device=ctx.device) for k,v in g.items()}
            system=NS(state=NS(**a),t=float(g.attrs['time']))
            if plotter is None: plotter=case.setupPlot(ctx,system)
            else: case.updatePlot(ctx,system,plotter,i)
    if plotter:
        plotter.drain() if hasattr(plotter,'drain') else None
        closeWindow(plotter)
    print(encodeFrames(ctx.imagePath,ctx.exportPath),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','warp','spheral','render'))
    p.add_argument('--sampling',choices=('equalSpacing','equalMass'),required=True)
    p.add_argument('--support',choices=('fixed','adaptive'),default='fixed')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--publisher-path',type=Path)
    args=p.parse_args()
    return {'prepare':prepare,'warp':warp_run,'spheral':spheral_run,'render':render}[args.mode](args)


if __name__=='__main__':
    raise SystemExit(main())
