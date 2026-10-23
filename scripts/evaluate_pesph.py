"""Sequential paper/general-case comparison: PESPH-E, PESPH-A and CRKSPH.

Each row runs in a fresh process (a failed CUDA assertion cannot poison later
rows), streams progress, records Vispy videos, quantitative histories and
physical-time snapshots. `run --wait-pattern run_fluidMatrix.py` queues behind
the existing incompressible matrix. `report` regenerates visual/CSV/HTML output.
The entropy variant explicitly opts into SWIFT's local heat conversion; its
nonlocal thermal-energy defect is measured, not silently repaired.
"""

import argparse
from dataclasses import replace
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCHEMES = ('CRKSPH', 'PESPH-E', 'PESPH-A')


def cases():
    def row(name, case=None, nx=64, end=1., params=None, **simulation):
        return dict(name=name, case=case or name, simulation=dict(nx=nx, tLimit=float(end), **simulation), params=params or {})
    return [
        row('linearWave', nx=128, end=1., params={'A': 1e-3}),
        row('sod', nx=400, end=.15, params={'gamma': 1.4, 'right_rho': .125, 'right_pressure': .1, 'samplingRatio': 8}),
        row('sedov2d', 'sedov', nx=96, dim=2, params={'uAmbient': 1e-8}),
        row('sedov3d', 'sedov', nx=40, dim=3, params={'uAmbient': 1e-8}),
        row('kelvinHelmholtz', nx=128, end=2.5),
        row('khSmooth', 'kelvinHelmholtz', nx=128, end=2.5, params={'smoothDensity': 1}),
        row('hydrostatic', nx=100, end=7., params={'rho_high': 4., 'gamma': 1.5}),
        row('kidder', nx=200, params={'tauFraction': .95}),
        row('gresho', nx=64, end=3.),
        row('yee', nx=64, end=2., params={'nr': 24, 'bufferRings': 10}),
        row('noh', nx=256, end=.6, params={'uAmbient': 1e-8}),
        row('sod2d', nx=160, end=.15),
        row('sod3d', nx=80, end=.15),
        row('shearBox', nx=64, end=2.),
        row('shearingNoh', nx=64, end=.6, params={'uAmbient': 1e-8}),
        row('woodwardColella', nx=400, end=.038),
        row('rayleighTaylor', nx=64, end=4.),
        row('triplePoint', nx=64, end=2.),
        row('blob', nx=128, end=8*1.6*2*np.sqrt(10)/(2.7*np.sqrt(5/3)), params={'initialDensity': 'prescribed'}),
    ]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True)+'\n')


def make_plan(root, selected=None, quick=False, device='cuda:0'):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cases():
        if selected and c['name'] not in selected:
            continue
        for scheme in SCHEMES:
            spec = dict(c['simulation'], device=device, precision='float64', integrationScheme='rungeKutta2',
                        supportMode='KernelMeanSymmetric',
                        adaptiveDt=True, cflFactor=.15, minDt=1e-10, calibrateNormalization=True,
                        execution='eager', cudaGraph=False, plot=True, video=True, show=False,
                        progress=True, quiet=True, plotBackend='vispy', plotInterval=1,
                        store=False, velocityAlarmPlotInterval=1, stallProgress=1e-3)
            if quick:
                spec.update(nx=24 if spec.get('dim') != 3 else 10, nSteps=3)
                if c['case'] == 'kidder':
                    spec['nx'] = 48
                if c['case'] == 'blob':
                    spec['nx'] = 64
            solver = dict(scheme='CRKSPH' if scheme == 'CRKSPH' else 'PESPH')
            if scheme != 'CRKSPH':
                solver.update(variant='PressureEnergy' if scheme == 'PESPH-E' else 'PressureEntropy',
                              dissipation=True, entropyHeating='SwiftLocal' if scheme == 'PESPH-A' else 'Guarded')
            rows.append(dict(id=c['name']+'__'+scheme, name=c['name'], case=c['case'], scheme=scheme,
                             configuration=dict(schemaVersion=1, case=c['case'], simulation=spec,
                                                params=c['params'], solver=solver)))
    source = {}
    for folder in ('src/warpSPH/modules/pesph', 'src/warpSPH/modules/dissipation',
                   'src/warpSPH/schemes/compressible', 'src/warpSPH/systems'):
        for p in (REPO/folder).glob('*.py'):
            source[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    source['scripts/evaluate_pesph.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), quick=quick, rows=rows, sources=source,
        comparison='Same case IC parameters/kernel/neighbor scale, kernel-mean symmetric support mode and fresh-stage RK2. CRKSPH uses its existing acoustic-CFL hook every step; its force/energy/dissipation implementation is unchanged. Both PESPH variants use their number-density supports and approach-signal CFL with the existing native dissipation stack. A uses explicit SwiftLocal thermal source.',
        limitations=['This battery implements the repository paper cases; it is not exact reproduction of every paper setting.',
                    'SwiftLocal entropy heating does not enforce exact conservation of the nonlocal U(A).',
                    'Driven Kidder/RT/Yee bands exchange energy; report energy changes rather than closed-system drift.',
                    'Mixed-gamma/material-EOS cases are outside the current PESPH contract.',
                    'Linear wave uses identical physical travelling-wave ICs for all schemes, bypassing legacy mass relaxation.',
                    'Cold Sedov/Noh cases add the same explicit uAmbient=1e-8 to every solver: pressure-based fractional weights require positive temperature. Initial energy is reported.',
                    'KH mode and grid mixing are diagnostics; no McNally reference value is used for these differing ICs.'])
    dump(root/'manifest.json', manifest)
    return manifest


def arrays(ctx, system):
    s = system.state
    a = {k: getattr(s, k).detach().cpu().numpy() for k in
         ('positions', 'velocities', 'densities', 'pressures', 'masses', 'internalEnergies', 'entropies', 'supports')}
    a['time'] = np.array(float(system.t))
    a['phase'] = ctx.scratch['evalPhase'].detach().cpu().numpy()
    for name in ('min', 'max', 'periodic'):
        a['domain'+name.title()] = getattr(ctx.config.domain, name).detach().cpu().numpy()
    return a


def display_positions(a, domain):
    """Wrap only plotted coordinates; retain unwrapped Lagrangian trajectories."""
    p = a['positions'].copy()
    def host(value):
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    lo, hi, periodic = (host(getattr(domain, name)) for name in ('min', 'max', 'periodic'))
    return np.where(periodic, lo+np.remainder(p-lo, hi-lo), p)


def profiles(ctx, a, *, at_particles=False):
    x = a['positions'][:, 0]
    if ctx.case.name == 'sedov':
        x = np.linalg.norm(a['positions'], axis=1)
        y = (a['densities'], a['pressures'], np.linalg.norm(a['velocities'], axis=1))
        r = x.copy() if at_particles else np.linspace(0, float(np.max(x)), 500)
        if float(a['time']) > 0:
            r, v, u, rho, p, *_ = ctx.scratch['solution'].solution(float(a['time']), r)
            return x, y, (np.array(r), (np.array(rho), np.array(p), np.array(v)))
        return x, y, None
    y = (a['densities'], a['pressures'], a['velocities'][:, 0])
    if ctx.case.name == 'linearWave':
        amp, cs, rho0 = [ctx.param(k) for k in ('A', 'c_s', 'rho0')]
        r = x.copy() if at_particles else np.linspace(x.min(), x.max(), 500)
        dp = amp*np.sin(2*np.pi*(r-cs*float(a['time']))/ctx.param('lamda'))
        return x, y, (r, (rho0+dp/cs**2, rho0*cs**2/ctx.param('gamma')+dp, dp/(rho0*cs)))
    if ctx.case.name.startswith('sod'):
        from warpSPH.caseUtils.compressible.sod.sodSolution import solve
        left = tuple(ctx.param('left_'+k) for k in ('pressure', 'rho', 'velocity'))
        right = tuple(ctx.param('right_'+k) for k in ('pressure', 'rho', 'velocity'))
        _, _, ref = solve(left, right, (0., 1., .5), max(float(a['time']), 1e-12), gamma=ctx.param('gamma'), npts=100000 if at_particles else 1000)
        r = np.r_[-ref['x'][::-1], ref['x']]
        values = tuple(np.r_[ref[k][::-1]*sgn, ref[k]] for k, sgn in (('rho', 1), ('p', 1), ('u', -1)))
        return x, y, (x, tuple(np.interp(x, r, value) for value in values)) if at_particles else (r, values)
    if ctx.case.name == 'kidder':
        sol = ctx.scratch['solution']
        r = x.copy() if at_particles else np.linspace(sol.rInner(float(a['time'])), sol.rOuter(float(a['time'])), 500)
        return x, y, (r, tuple(np.asarray(getattr(sol, key)(float(a['time']), r)) for key in ('rho', 'P', 'vr')))
    if ctx.case.name == 'noh':
        from warpSPH.cases.compressible.noh import shockState
        rho, pressure = shockState(ctx)
        r = x.copy() if at_particles else np.linspace(x.min(), x.max(), 1000)
        inside = abs(r) < ctx.param('v_s')*float(a['time'])
        return x, y, (r, (np.where(inside, rho, 1.), np.where(inside, pressure, 0.), np.where(inside, 0., -np.sign(r))))
    return x, y, None


class Recording:
    def __init__(self, ctx, system, rowdir, label):
        from vispy import scene
        self.ctx, self.root, self.label = ctx, rowdir, label
        self.index = 0
        self.nextTime = 0.
        self.interval = max(float(ctx.spec.tLimit)/80., 1e-12)
        self.profile = ctx.spec.dim == 1 or ctx.case.name.startswith('sod') or ctx.case.name == 'sedov'
        self.canvas = scene.SceneCanvas(size=(1200, 480), bgcolor='white', show=False)
        grid = self.canvas.central_widget.add_grid(margin=12, spacing=12)
        self.title = scene.Label('', color='black', font_size=13)
        self.title.height_max = 35
        grid.add_widget(self.title, row=0, col=0, col_span=3)
        self.views, self.points, self.refs, self.headers = [], [], [], []
        titles = ('Mass density', 'Pressure', 'Velocity') if self.profile else ('Mass density', 'Pressure', 'Original phase / speed')
        for i, title in enumerate(titles):
            header = scene.Label(title, color='black', font_size=11)
            header.height_max = 28
            self.headers.append(header)
            grid.add_widget(header, row=1, col=i)
            v = grid.add_view(row=2, col=i, border_color='#bbb')
            v.camera = 'panzoom'
            self.views.append(v)
            self.points.append(scene.visuals.Markers(parent=v.scene))
            self.refs.append(scene.visuals.Line(color='#cf412e', parent=v.scene))
        self.ranges = None
        if system is not None:
            self.record(ctx, system, force=True)

    def record(self, ctx, system, force=False):
        t = float(system.t)
        if not force and t+1e-12 < self.nextTime:
            return
        a = arrays(ctx, system)
        self.record_arrays(ctx, a, save_snapshot=True)

    def record_arrays(self, ctx, a, *, save_snapshot=False):
        from PIL import Image
        from vispy.color import get_colormap
        t = float(a['time'])
        profileData = profiles(ctx, a) if self.profile else None
        if profileData is not None and profileData[2] is not None:
            rx, ry = profileData[2]
            a.update(referenceX=rx, referenceDensities=ry[0], referencePressures=ry[1], referenceVelocities=ry[2])
        if save_snapshot:
            np.savez_compressed(self.root/'snapshots'/f'{self.index:04d}.npz', **a)
        self.title.text = f'{ctx.case.name} / {self.label} / N={len(a["masses"])} / t={t:.6g}'
        if self.profile:
            x, ys, reference = profileData
            for i, (v, points, line, y) in enumerate(zip(self.views, self.points, self.refs, ys)):
                points.set_data(np.c_[x, y].astype(np.float32), face_color='#1269a3', edge_width=0, size=3)
                allY = y
                if reference is not None:
                    rx, ry = reference[0], reference[1][i]
                    line.set_data(np.c_[rx, ry].astype(np.float32))
                    allY = np.r_[y, ry]
                finite = allY[np.isfinite(allY)]
                lo, hi = finite.min(), finite.max()
                pad = max((hi-lo)*.08, abs(hi)*1e-5, 1e-9)
                v.camera.set_range(x=(x.min(), x.max()), y=(lo-pad, hi+pad), margin=0)
        else:
            pos = display_positions(a, ctx.config.domain)[:, :2]
            phase = a['phase']
            values = (a['densities'], a['pressures'], phase if np.ptp(phase) else np.linalg.norm(a['velocities'], axis=1))
            if self.ranges is None:
                self.ranges = [(float(y.min())*.8, max(float(y.max())*1.2, float(y.min())+1e-9)) for y in values]
                if np.ptp(phase):
                    self.ranges[2] = (0., 1.)
            for i, (v, points, y, (lo, hi)) in enumerate(zip(self.views, self.points, values, self.ranges)):
                title = ('Mass density', 'Pressure', 'Original phase' if np.ptp(phase) else 'Speed')[i]
                self.headers[i].text = f'{title} [{lo:.7g}, {hi:.7g}]'
                colors = get_colormap('viridis').map(np.clip((y-lo)/(hi-lo), 0, 1)).astype(np.float32)
                points.set_data(pos.astype(np.float32), face_color=colors, edge_width=0,
                                size=max(1.3, min(5., 420./ctx.spec.nx)))
                v.camera.set_range(x=(float(ctx.config.domain.min[0]), float(ctx.config.domain.max[0])),
                                   y=(float(ctx.config.domain.min[1]), float(ctx.config.domain.max[1])), margin=0)
                v.camera.aspect = 1
        Image.fromarray(self.canvas.render()).save(Path(ctx.imagePath)/f'frame_{self.index:07d}.png', compress_level=1)
        self.index += 1
        self.nextTime = self.index*self.interval

    def close(self):
        self.canvas.close()


def diagnostics(base, ctx, system):
    import torch
    from warpSPH.configurations.pesph import PESPHConfig
    from warpSPH.cases.compressible.common import compressibleDiagnostics
    row = dict(base(ctx, system) if base else {})
    row.update(compressibleDiagnostics(ctx, system))
    s = system.state
    gamma = ctx.schemeConfig.gamma
    entropy = (s.entropies if isinstance(ctx.schemeConfig, PESPHConfig) else
               (gamma-1)*s.internalEnergies*s.densities.pow(1-gamma))
    row.update(physicalEnergy=float((s.masses*(s.internalEnergies+.5*s.velocities.square().sum(-1))).sum()),
               mass=float(s.masses.sum()), minPressure=float(s.pressures.min()), minEnergy=float(s.internalEnergies.min()),
               minDensity=float(s.densities.min()), minSupport=float(s.supports.min()),
               maxVelocity=float(s.velocities.norm(dim=-1).max()),
               physicalEntropy=float((s.masses*torch.log(entropy)/(gamma-1)).sum()),
               momentumX=float((s.masses*s.velocities[:, 0]).sum()))
    if ctx.case.name == 'kelvinHelmholtz':
        from av_report import _khMode
        row.update(_khMode(ctx, system))
    return row


def metrics(ctx, result, initial):
    a = arrays(ctx, result.state)
    first, last = result.trajectory[0], result.trajectory[-1]
    times = result.series('stepTime_ms')[1:]
    energy = result.series('physicalEnergy')
    m = dict(tFinal=float(result.state.t), targetTime=ctx.spec.tLimit, nSteps=result.nSteps,
             particles=len(a['masses']), wallSeconds=result.wallTime,
             medianStepMs=float(np.median(times)), meanStepMs=float(np.mean(times)),
             initialEnergy=float(energy[0]), relativeEnergyChange=float(energy[-1]/energy[0]-1),
             maxRelativeEnergyChange=float(np.max(abs(energy/energy[0]-1))),
             entropyChange=last['physicalEntropy']-first['physicalEntropy'],
             massChange=last['mass']-first['mass'], momentumXChange=last['momentumX']-first['momentumX'],
             kineticEnergyRetention=last['kineticEnergy']/first['kineticEnergy'] if first['kineticEnergy'] else None,
             minPressure=min(r['minPressure'] for r in result.trajectory),
             minEnergy=min(r['minEnergy'] for r in result.trajectory),
             minDensity=min(r['minDensity'] for r in result.trajectory),
             maxDensity=float(a['densities'].max()), maxPressure=float(a['pressures'].max()),
             maxVelocity=max(r['maxVelocity'] for r in result.trajectory),
             velocityAlarmEvents=len(result.velocityAlarms),
             closedSystem=ctx.case.name not in ('kidder', 'rayleighTaylor', 'yee'))
    m['initialMass'] = float(initial['masses'].sum())
    for key in ('positions', 'masses', 'velocities'):
        m['initial'+key.title()+'SHA256'] = hashlib.sha256(initial[key].tobytes()).hexdigest()
    x, ys, ref = profiles(ctx, a, at_particles=True)
    if ref is not None:
        mask = np.ones(len(x), dtype=bool)
        if ctx.case.name == 'kidder':
            mask[:ctx.param('band')] = False
            mask[-ctx.param('band'):] = False
        if ctx.case.name == 'sedov':
            mask &= x < ctx.scratch['solution'].shockState(float(a['time']))[1]*1.15
        for name, y, yr in zip(('density', 'pressure', 'velocity'), ys, ref[1]):
            error = abs(y[mask]-yr[mask])
            m[name+'L1'] = float(np.mean(error))
            volume = a['masses'][mask]/a['densities'][mask]
            m[name+'VolumeWeightedL1'] = float(np.sum(volume*error)/volume.sum())
        if ctx.case.name == 'linearWave':
            m['normalizedVelocityL1'] = m['velocityL1']/(ctx.param('A')/(ctx.param('rho0')*ctx.param('c_s')))
        if ctx.case.name == 'sedov':
            rShock = ctx.scratch['solution'].shockState(float(a['time']))[1]
            bins = np.linspace(0., rShock*1.5, 81)
            k = np.digitize(x, bins)-1
            means = np.array([np.mean(a['densities'][k == j]) if np.any(k == j) else -np.inf for j in range(80)])
            measured = .5*(bins[np.argmax(means)]+bins[np.argmax(means)+1])
            m.update(shockRadius=measured, exactShockRadius=rShock, shockRadiusRelativeError=measured/rShock-1,
                     shockRadiusBinWidthRelative=(bins[1]-bins[0])/rShock,
                     injectedEnergyRecovery=energy[0]/ctx.param('E0')-1)
    if ctx.case.name == 'gresho':
        from av_report import _greshoExact
        r = np.linalg.norm(a['positions'], axis=1)
        exact = _greshoExact(r)
        m['speedL1'] = float(np.mean(abs(np.linalg.norm(a['velocities'], axis=1)-exact)))
        vphi = (a['positions'][:, 0]*a['velocities'][:, 1]-a['positions'][:, 1]*a['velocities'][:, 0])/np.maximum(r, 1e-12)
        m['tangentialVelocityL1'] = float(np.mean(abs(vphi[r <= .5]-exact[r <= .5])))
    if ctx.case.name in ('gresho', 'yee'):
        m['angularMomentumLoss'] = float(1-last['angularMomentum']/first['angularMomentum'])
    if ctx.case.name in ('noh', 'shearingNoh'):
        # Same plateau/front estimators as av_report, separated from the
        # all-particle profile errors and their wall-heating contribution.
        x = np.abs(a['positions'][:, 0])
        speed = ctx.param('v_s') if ctx.case.name == 'noh' else (ctx.param('gamma')-1)/2
        exactRho = (ctx.param('gamma')+1)/(ctx.param('gamma')-1)
        mask = (x > .2*speed*float(a['time'])) & (x < .8*speed*float(a['time']))
        if mask.any():
            m['postShockDensityRelativeError'] = float(a['densities'][mask].mean()/exactRho-1)
        shocked = a['densities'] > .5*(1+exactRho)
        if shocked.any():
            front = float(np.quantile(x[shocked], .95))
            m.update(shockFront=front, exactShockFront=speed*float(a['time']),
                     shockFrontRelativeError=front/(speed*float(a['time']))-1)
    if ctx.case.name == 'yee':
        r = np.linalg.norm(a['positions']-np.array([ctx.param('xc'), ctx.param('yc')]), axis=1)
        exact = ctx.param('beta')/(2*np.pi)*np.exp((1-r*r)/2)*r
        mask = r<3
        m['speedL1'] = float(np.mean(abs(np.linalg.norm(a['velocities'][mask], axis=1)-exact[mask])))
    for key in ('khAmplitude', 'cloudMassFraction', 'cloudDenseFraction', 'cloudMeanEnergy', 'modeAmplitude',
                'shapeDisplacement', 'pressureError', 'maxPeculiarVelocity', 'mixingWidth',
                'peculiarKineticEnergy', 'densityRelativeL1'):
        if key in last:
            m[key] = last[key]
            if key.startswith('cloud'):
                m[key+'Initial'] = first[key]
    if first.get('cloudMassFraction', 0.) > 0:
        m['cloudColdRetentionRelativeToInitial'] = last['cloudMassFraction']/first['cloudMassFraction']
    if 'khAmplitude' in last:
        m['khAmplitudeInitial'] = first['khAmplitude']
        m['khAmplitudeMax'] = max(h['khAmplitude'] for h in result.trajectory)
        nearest = min(result.trajectory, key=lambda h: abs(h['t']-1.5))
        if abs(nearest['t']-1.5) < .05:
            m['khAmplitudeAt1p5'] = nearest['khAmplitude']
    if ctx.case.name == 'rayleighTaylor':
        middle = .5*(ctx.param('rho_low')+ctx.param('rho_high'))
        m['heavyMinY'] = float(a['positions'][a['densities'] > middle, 1].min())
        m['lightMaxY'] = float(a['positions'][a['densities'] <= middle, 1].max())
        m['mixingWidth'] = m['lightMaxY']-m['heavyMinY']
        for label, mask in (('heavy', a['phase'] > .5), ('light', a['phase'] < .5)):
            mask &= (initial['positions'][:, 1] > 0) & (initial['positions'][:, 1] < ctx.spec.L)
            weights = a['masses'][mask]
            m[label+'CenterY'] = float(np.sum(weights*a['positions'][mask, 1])/weights.sum())
            m[label+'CenterYInitial'] = float(np.sum(weights*initial['positions'][mask, 1])/weights.sum())
            m[label+'Y05'], m[label+'Y95'] = map(float, np.quantile(a['positions'][mask, 1], (.05, .95)))
    if a['positions'].shape[1] == 2 and np.ptp(a['phase']):
        lo = ctx.config.domain.min.detach().cpu().numpy()
        hi = ctx.config.domain.max.detach().cpu().numpy()
        def mix(data):
            p = (data['positions']-lo) % (hi-lo)+lo
            total = np.histogram2d(*p.T, bins=64, range=np.c_[lo, hi], weights=data['masses'])[0]
            heavy = np.histogram2d(*p.T, bins=64, range=np.c_[lo, hi], weights=data['masses']*data['phase'])[0]
            c = np.divide(heavy, total, out=np.zeros_like(total), where=total>0)
            return float((total*4*c*(1-c)).sum()/total.sum())
        m.update(gridMixing=mix(a), gridMixingInitial=mix(initial))
    return m


def worker(root, row):
    from warpSPHBootstrap import bootstrap
    bootstrap('float64')
    import torch
    torch.manual_seed(1234)
    np.random.seed(1234)
    from warpSPH.runner import RunConfiguration, getCase
    from warpSPH.cases.compressible.common import compressibleTimestep
    from warpSPH.runner.media import encodeFrames
    import importlib
    runnerModule = importlib.import_module('warpSPH.runner.runner')
    runnerModule.encodeFrames = lambda path, destination: encodeFrames(path, destination, framerate=10, fps=10, gif=False)
    from warpSPH.cases import importAll
    from pyvirtualdisplay import Display
    from vispy import app
    importAll()
    if row.get('callAudit'):
        from warpSPHCore import callAudit
        callAudit.enable(warmupSteps=2)
    if row.get('disableConductivitySource'):
        # Explicit diagnostic ablation only, retained for the frozen cohorts
        # that predate the repaired public strength multiplier. This override
        # always removes conduction, independently of the saved coefficient.
        ratesModule = importlib.import_module('warpSPH.modules.dissipation.rates')
        ratesModule.computeConductivityWarp = lambda particles, *args, **kwargs: torch.zeros_like(particles.internalEnergies)
        monaghanModule = importlib.import_module('warpSPH.schemes.compressible.monaghan')
        monaghanModule.computeConductivity = lambda particles, *args, **kwargs: torch.zeros_like(particles.internalEnergies)
    directory = root/row['id']
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'snapshots').mkdir(exist_ok=True)
    # Dependencies are shared editable checkouts; fingerprint their Python
    # trees per row so concurrent unrelated work cannot silently change the
    # provenance of a multi-hour comparison.
    import warpSPHCore, warpSPHIntegrators, warp
    dependencies = dict(torch=torch.__version__, warp=warp.__version__, precision='float64',
                        device=row['configuration']['simulation'].get('device', 'auto'))
    for package in (warpSPHCore, warpSPHIntegrators):
        packageRoot = Path(package.__file__).resolve().parent
        hashes = {str(p.relative_to(packageRoot)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(packageRoot.rglob('*.py'))}
        dependencies[package.__name__] = dict(path=str(packageRoot),
            treeSHA256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
    dump(directory/'environment.json', dependencies)
    manifest = json.loads((root/'manifest.json').read_text())
    dump(directory/'sourceHashes.json', {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                                        for name in manifest['sources']})
    config = RunConfiguration.fromDict(row['configuration'])
    config.simulation['exportRoot'] = str(directory/'run')
    base = getCase(row['case'])
    history = []
    configure = base.configureScheme
    build = base.buildSystem
    def configure_case(ctx):
        if configure:
            configure(ctx)
        # Some legacy samplers change this to 1 after initialization; restore
        # PESPH's configured support iterations in build_case below.
        ctx.scratch['evalSupportIterations'] = getattr(ctx.schemeConfig, 'adaptiveSupportIterations', None)
    def build_case(ctx):
        if row['case'] == 'linearWave':
            from warpSPH.caseUtils import sampleLinearWave
            system = sampleLinearWave(ctx.spec.nx, ctx.config, ctx.schemeConfig,
                ctx.SimulationState, ctx.SimulationSystem, *(ctx.param(k) for k in ('A', 'lamda', 'c_s', 'rho0', 'gamma')),
                supportScheme=ctx.schemeConfig.adaptiveSupportScheme, analyticMass=True)
        else:
            system = build(ctx)
        if row['scheme'].startswith('PESPH'):
            ctx.schemeConfig.adaptiveSupportIterations = ctx.scratch['evalSupportIterations']
        return system
    def setup(ctx, system):
        s = system.state
        if ctx.case.name == 'kelvinHelmholtz':
            phase = ((s.positions[:, 1]>.25)&(s.positions[:, 1]<.75)).to(s.masses.dtype)
        elif ctx.case.name == 'blob':
            phase = (s.materials == 1).to(s.masses.dtype)
        elif ctx.case.name == 'hydrostatic':
            phase = (s.masses > s.masses.min()*1.5).to(s.masses.dtype)
        elif ctx.case.name == 'rayleighTaylor':
            phase = (s.positions[:, 1] > .5*ctx.spec.L).to(s.masses.dtype)
        else:
            phase = torch.zeros_like(s.masses)
        ctx.scratch['evalPhase'] = phase
        ctx.scratch['evalInitial'] = arrays(ctx, system)
        return Recording(ctx, system, directory, row['scheme'])
    def record_diagnostics(ctx, system):
        values = diagnostics(base.diagnostics, ctx, system)
        history.append(dict(values, t=float(system.t)))
        return values
    visual = replace(base, configureScheme=configure_case, buildSystem=build_case,
                     timestep=base.timestep or compressibleTimestep,
                     diagnostics=record_diagnostics,
                     setupPlot=setup, updatePlot=lambda ctx, system, handle, step: handle.record(ctx, system,
                         force=ctx.spec.nSteps is not None and step+1 >= ctx.spec.nSteps))
    started = time.perf_counter()
    record = dict(id=row['id'], case=row['case'], name=row['name'], scheme=row['scheme'], status='running')
    if row.get('disableConductivitySource'):
        record['diagnosticOverrides'] = dict(disableConductivitySource=True)
    dump(directory/'result.json', record)
    with Display(visible=False, size=(1200, 480)):
        app.use_app('glfw')
        try:
            result = config.run(visual)
            record.update(status='failed' if result.diverged or result.stopReason else 'complete',
                          stopReason=result.stopReason, metrics=metrics(result.ctx, result, result.ctx.scratch['evalInitial']),
                          exportPath=result.exportPath, videoPath=result.videoPath)
            dump(directory/'history.json', result.trajectory)
            if row.get('callAudit'):
                from warpSPHCore import callAudit
                callAudit.writeReport(str(directory/'callAudit.txt'))
                callAudit.disable()
            dump(directory/'result.json', record)
        except Exception:
            record.update(status='error', error=traceback.format_exc(), wallSeconds=time.perf_counter()-started)
            dump(directory/'history.json', history)
            from warpSPH.runner.media import encodeFrames
            for imageDir in (directory/'run').glob('*/images'):
                record['videoPath'] = encodeFrames(str(imageDir), str(imageDir.parent), framerate=10, fps=10, gif=False)
            dump(directory/'result.json', record)
            print(record['error'], flush=True)
        finally:
            app.quit()
            import glfw
            glfw.terminate()
    return 0 if record['status'] == 'complete' else 2


def busy(pattern):
    lines = subprocess.check_output(['ps', '-eo', 'pid=,comm=,args='], text=True).splitlines()
    active = []
    for line in lines:
        pid, command, args = line.split(None, 2)
        tokens = args.split()
        if pid != str(os.getpid()) and command.startswith('python') and len(tokens)>1 and pattern in tokens[1]:
            active.append(line.strip())
    return active


def run_suite(root, manifest, wait_pattern, retry=False):
    for i, row in enumerate(manifest['rows']):
        directory = root/row['id']
        directory.mkdir(parents=True, exist_ok=True)
        saved = directory/'result.json'
        if saved.exists() and json.loads(saved.read_text()).get('status') != 'running' and not retry:
            continue
        if wait_pattern:
            while work := busy(wait_pattern):
                print(f'[PESPH queue] waiting for {wait_pattern}: {len(work)} process(es); {time.strftime("%H:%M:%S")}', flush=True)
                time.sleep(30)
        print(f'[PESPH suite] {i+1}/{len(manifest["rows"])} {row["id"]}', flush=True)
        with (directory/'log.txt').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--root', str(root), '--row', row['id']],
                                       cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED='1'))
            started = time.monotonic()
            while process.poll() is None:
                time.sleep(5)
                content = (directory/'log.txt').read_text(errors='replace')[-2500:].replace('\r', '\n').splitlines()
                print(f'[PESPH suite] {row["id"]} {time.monotonic()-started:.0f}s | '+(content[-1][-420:] if content else 'initializing'), flush=True)
                if time.monotonic()-started > 1800:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    dump(saved, dict(id=row['id'], case=row['case'], name=row['name'], scheme=row['scheme'],
                                     status='timeout', error='1800 s row watchdog; partial frames/log retained'))
            if not saved.exists() or json.loads(saved.read_text()).get('status') == 'running':
                dump(saved, dict(id=row['id'], case=row['case'], name=row['name'], scheme=row['scheme'],
                                 status='error', error=f'Worker exited {process.returncode}; inspect log.txt'))
        collect(root, manifest)
    report(root, manifest)


def profile_context(root, record):
    """Rebuild only analytic-reference parameters; never initialize a solver."""
    candidates = ([Path(record['exportPath'])/'caseSpec.json'] if record.get('exportPath') else
                  sorted((root/record['id']/'run').glob('*/caseSpec.json')))
    if not candidates or not candidates[-1].exists():
        return None
    saved = json.loads(candidates[-1].read_text())
    params = saved['params']
    context = SimpleNamespace(case=SimpleNamespace(name=record['case']),
        spec=SimpleNamespace(dim=saved['dim'], nx=saved['nx'], tLimit=saved['tLimit']), param=lambda k: params[k], scratch={})
    configPath = candidates[-1].with_name('config.json')
    if configPath.exists():
        domain = json.loads(configPath.read_text())['config']['domain']
        context.config = SimpleNamespace(domain=SimpleNamespace(**{k: np.asarray(v) for k, v in domain.items()}))
    if record['case'] == 'sedov':
        from warpSPH.caseUtils.compressible.sedov.sedovSolution import SedovSolution
        context.scratch['solution'] = SedovSolution(nDim=saved['dim'], gamma=params['gamma'],
            rho0=params['rho0'], E0=params['E0'], h0=2/saved['nx'])
    if record['case'] == 'kidder':
        from warpSPH.caseUtils.compressible.kidder.kidder import KidderIsentropicCapsuleAnalyticSolution
        context.scratch['solution'] = KidderIsentropicCapsuleAnalyticSolution(params['nu'],
            params['r_inner'], params['r_outer'], params['P_inner'], params['P_outer'], params['rho_outer'])
    return context


def collect(root, manifest):
    declared = {row['id']: row for row in manifest['rows']}
    records = [json.loads(p.read_text()) for row in manifest['rows'] if (p := root/row['id']/'result.json').exists()]
    # Recompute profile diagnostics uniformly from saved final states so rows
    # completed before a reporting improvement retain the same definitions.
    for record in records:
        if record.get('status') != 'complete':
            continue
        paths = sorted((root/record['id']/'snapshots').glob('*.npz'))
        context = profile_context(root, record)
        if not paths or context is None:
            continue
        a = dict(np.load(paths[-1]))
        x, ys, reference = profiles(context, a, at_particles=True)
        m = record['metrics']
        m.update(maxDensity=float(a['densities'].max()), maxPressure=float(a['pressures'].max()))
        if record['case'] == 'kidder':
            band = context.param('band')
            interior = np.ones(len(a['masses']), dtype=bool)
            interior[:band] = False
            interior[-band:] = False
            m.update(minFinalInteriorEnergy=float(a['internalEnergies'][interior].min()),
                negativeFinalEnergyInteriorParticles=int(np.sum(a['internalEnergies'][interior] <= 0)),
                negativeFinalEnergyBandParticles=int(np.sum(a['internalEnergies'][~interior] <= 0)))
        if reference is not None:
            mask = np.ones(len(x), dtype=bool)
            if record['case'] == 'kidder':
                mask[:context.param('band')] = False
                mask[-context.param('band'):] = False
            if record['case'] == 'sedov':
                mask &= x < context.scratch['solution'].shockState(float(a['time']))[1]*1.15
                m['shockRadiusBinWidthRelative'] = 1.5/80
            volume = a['masses'][mask]/a['densities'][mask]
            for name, y, yr in zip(('density', 'pressure', 'velocity'), ys, reference[1]):
                error = abs(y[mask]-yr[mask])
                m[name+'L1'] = float(error.mean())
                m[name+'VolumeWeightedL1'] = float(np.sum(volume*error)/volume.sum())
            thermoDensity = (a['densities'] if not record['scheme'].startswith('PESPH') else
                             a['pressures']/((context.param('gamma')-1)*a['internalEnergies']))
            error = abs(thermoDensity[mask]-reference[1][0][mask])
            thermoVolume = a['masses'][mask]/thermoDensity[mask]
            m['thermodynamicDensityL1'] = float(error.mean())
            m['thermodynamicDensityVolumeWeightedL1'] = float(np.sum(thermoVolume*error)/thermoVolume.sum())
    # Separate background quadrature offset from acoustic mode accuracy. This
    # is postprocessing the saved physical state, not changing initialization.
    for record in records:
        if record.get('case') != 'linearWave' or record.get('status') != 'complete':
            continue
        paths = sorted((root/record['id']/'snapshots').glob('*.npz'))
        caseSpec = Path(record['exportPath'])/'caseSpec.json'
        if not paths or not caseSpec.exists():
            continue
        a = dict(np.load(paths[-1]))
        params = json.loads(caseSpec.read_text())['params']
        cs, rho, amplitude = (params[k] for k in ('c_s', 'rho0', 'A'))
        dp = amplitude*np.sin(2*np.pi*(a['positions'][:, 0]-cs*float(a['time']))/params['lamda'])
        reference = (rho+dp/cs**2, rho*cs**2/params['gamma']+dp, dp/(rho*cs))
        for name, y, ref, scale in zip(('density','pressure','velocity'),
                    (a['densities'],a['pressures'],a['velocities'][:,0]), reference,
                    (amplitude/cs**2, amplitude, amplitude/(rho*cs))):
            error = y-ref
            record['metrics'][name+'MeanError'] = float(error.mean())
            record['metrics'][name+'MeanRemovedNormalizedL1'] = float(np.mean(abs(error-error.mean()))/scale)
            if name == 'velocity':
                record['metrics']['normalizedVelocityL1'] = float(np.mean(abs(error))/scale)
    for record in records:
        # Some native samplers leave the initial entropy cache zero. Derive
        # CRK's entropy from saved u and mass density; PESPH uses its own
        # pressure-volume thermodynamic entropy. Recompute ALL rows uniformly
        # from snapshots, including workers that predate this diagnostic fix.
        context = profile_context(root, record)
        if context is not None and record.get('status') != 'running':
            gamma = context.param('gamma')
            entropyHistory = []
            for path in sorted((root/record['id']/'snapshots').glob('*.npz')):
                with np.load(path) as a:
                    entropy = (a['entropies'] if record['scheme'].startswith('PESPH') else
                               (gamma-1)*a['internalEnergies']*a['densities']**(1-gamma))
                    def thermal_entropy(mask):
                        values = entropy[mask]
                        if not np.all(np.isfinite(values) & (values > 0)):
                            return None
                        return float(np.sum(a['masses'][mask]*np.log(values)/(gamma-1)))
                    values = dict(t=float(a['time']), physicalEntropy=thermal_entropy(slice(None)))
                    if record['case'] == 'kidder':
                        values['interiorEntropy'] = thermal_entropy(slice(context.param('band'), -context.param('band')))
                    entropyHistory.append(values)
            record['entropyHistory'] = entropyHistory
            if len(entropyHistory) > 1 and record.get('metrics'):
                for key, metric in (('physicalEntropy', 'entropyChange'), ('interiorEntropy', 'interiorEntropyChange')):
                    if key in entropyHistory[0]:
                        defined = all(h[key] is not None for h in entropyHistory)
                        record['metrics'][metric] = entropyHistory[-1][key]-entropyHistory[0][key] if defined else None
                        if key == 'physicalEntropy':
                            record['metrics']['globalEntropyDefinedForAllSnapshots'] = defined
        if record.get('status') in ('error', 'timeout') and 'metrics' not in record:
            # Preserve failed workers as measured partial trajectories. Values
            # refer to the last saved good snapshot, not the requested end time.
            paths = sorted((root/record['id']/'snapshots').glob('*.npz'))
            if paths:
                first, last = dict(np.load(paths[0])), dict(np.load(paths[-1]))
                energy = lambda a: float(np.sum(a['masses']*(a['internalEnergies']+.5*np.sum(a['velocities']**2, axis=1))))
                e0 = energy(first)
                record['metrics'] = dict(partialMetrics=True, tFinal=float(last['time']),
                    particles=len(last['masses']), initialEnergy=e0,
                    relativeEnergyChange=energy(last)/e0-1,
                    minPressure=float(last['pressures'].min()), minEnergy=float(last['internalEnergies'].min()),
                    minDensity=float(last['densities'].min()), maxDensity=float(last['densities'].max()),
                    maxPressure=float(last['pressures'].max()),
                    closedSystem=record['case'] not in ('kidder', 'rayleighTaylor', 'yee'),
                    targetTime=context.spec.tLimit if context is not None else None,
                    wallSeconds=record.get('wallSeconds', 1800. if record['status']=='timeout' else None))
                historyPath = root/record['id']/'history.json'
                if historyPath.exists():
                    history = json.loads(historyPath.read_text())
                    if history:
                        record['metrics'].update(nSteps=max(0, len(history)-1), lastDiagnosticTime=history[-1]['t'],
                            maxRelativeEnergyChange=max(abs(h['physicalEnergy']/e0-1) for h in history))
            logPath = root/record['id']/'log.txt'
            if logPath.exists():
                match = re.search(r'Assertion `([^`]+)` failed', logPath.read_text(errors='replace'))
                if match:
                    record['failureReason'] = match[1]
        m = record.get('metrics', {})
        if m.get('nSteps') and m.get('wallSeconds'):
            m['wallPerStepMs'] = 1000*m['wallSeconds']/m['nSteps']
            m['simulatedTimePerWallSecond'] = m['tFinal']/m['wallSeconds']
        if m.get('targetTime') is not None:
            m['reachedTarget'] = m['tFinal'] >= m['targetTime']-1e-12
            requested = declared[record['id']]['configuration']['simulation'].get('nSteps')
            record['evaluationScope'] = 'fixed-step' if requested is not None else 'physical-endpoint'
            if requested is not None:
                m['requestedSteps'] = requested
                m['reachedRequestedSteps'] = m.get('nSteps', 0) >= requested
                m['reachedRequestedGoal'] = m['reachedRequestedSteps']
            else:
                m['reachedRequestedGoal'] = m['reachedTarget']
            if record['status'] == 'complete' and not m['reachedRequestedGoal']:
                reason = ('Requested step count was not reached' if requested is not None else
                          'Requested physical end time was not reached')
                record.update(status='failed', stopReason=reason)
        flags = []
        if abs(m.get('injectedEnergyRecovery', 0.)) > 1e-3:
            flags.append('initial blast energy differs from requested E0')
        if m.get('closedSystem') and m.get('maxRelativeEnergyChange', 0.) > 1e-3:
            flags.append('closed-system energy change exceeds 0.1%')
        if m.get('velocityAlarmEvents', 0):
            flags.append('velocity alarm')
        if record.get('case') == 'hydrostatic' and m.get('pressureError', 0.) > .01:
            flags.append('static-equilibrium maximum pressure error exceeds 1%')
        if any(m.get(key, 1.) <= 0 for key in ('minPressure', 'minEnergy', 'minDensity')):
            flags.append('non-positive recorded thermodynamic state')
        if record.get('status') not in ('complete', 'running'):
            flags.append('incomplete trajectory; no end-time accuracy comparison')
        record['qualityFlags'] = flags
    for name in dict.fromkeys(row['name'] for row in manifest['rows']):
        baseline = sorted((root/(name+'__CRKSPH')/'snapshots').glob('*.npz'))
        if not baseline:
            continue
        with np.load(baseline[0]) as data:
            initialCRK = {key: data[key] for key in ('positions', 'masses', 'velocities')}
            initialCRKEnergy = float(np.sum(data['masses']*(data['internalEnergies']+.5*np.sum(data['velocities']**2, axis=1))))
        for record in (r for r in records if r['name'] == name and r.get('metrics')):
            snapshots = sorted((root/record['id']/'snapshots').glob('*.npz'))
            if not snapshots:
                continue
            with np.load(snapshots[0]) as data:
                matches = {key: np.array_equal(data[key], initialCRK[key]) for key in initialCRK}
                initialMomentum = np.sum(data['masses'][:, None]*data['velocities'], axis=0)
            with np.load(snapshots[-1]) as data:
                finalMomentum = np.sum(data['masses'][:, None]*data['velocities'], axis=0)
            record['metrics']['momentumChangeNorm'] = float(np.linalg.norm(finalMomentum-initialMomentum))
            record['metrics']['initialGeometryMatchesCRK'] = all(matches.values())
            record['metrics']['initialEnergyRelativeToCRK'] = record['metrics']['initialEnergy']/initialCRKEnergy-1
            if record['case'] in ('linearWave', 'sod', 'sod2d', 'sod3d', 'woodwardColella', 'triplePoint', 'blob') and abs(record['metrics']['initialEnergyRelativeToCRK']) > .001:
                record['qualityFlags'].append('initial energy differs from CRK by more than 0.1%; not a matched-energy comparison')
            if not all(matches.values()):
                record['qualityFlags'].append('initial positions/masses/velocities differ from CRK baseline')
    for record in records:
        if record.get('case') == 'kelvinHelmholtz' and record.get('status') == 'complete':
            history = json.loads((root/record['id']/'history.json').read_text())
            record['metrics']['khAmplitudeMax'] = max(h['khAmplitude'] for h in history)
            nearest = min(history, key=lambda h: abs(h['t']-1.5))
            if abs(nearest['t']-1.5) < .05:
                record['metrics']['khAmplitudeAt1p5'] = nearest['khAmplitude']
        if record.get('case') == 'yee' and record.get('status') == 'complete':
            history = json.loads((root/record['id']/'history.json').read_text())
            if len(history) > 1:
                e0, e1, ef = (history[i]['physicalEnergy'] for i in (0, 1, -1))
                record['metrics'].update(startupRelativeEnergyChange=e1/e0-1, postStartupRelativeEnergyChange=ef/e1-1)
                if abs(e1/e0-1) > .001:
                    record['qualityFlags'].append('stationary-vortex startup energy projection exceeds 0.1%')
        if record.get('case') == 'blob' and record.get('status') == 'complete':
            history = json.loads((root/record['id']/'history.json').read_text())
            for fraction, label in ((.5, 'Half'), (.1, 'TenPercent')):
                crossing = next((h for h in history if h['cloudMassFraction'] <= fraction*history[0]['cloudMassFraction']), None)
                record['metrics']['tKHTo'+label+'InitialColdMass'] = crossing['tKH'] if crossing else None
    dump(root/'analysisProtocol.json', dict(evaluatorSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analyticErrors='Native analytic solutions evaluated at particle coordinates; Sod interpolates a 100000-point reference grid. Plot curves use fewer samples and are not the numerical error reference.',
        weighting='L1 is a particle mean. densityL1 uses the native mass-density estimate. The distinct PESPH pressure-volume density is an auxiliary closure diagnostic, not an interchangeable mass-density estimate. Approximate volume-weighted L1 uses m/rhoMass; the additional thermodynamic weighting uses m/rhoThermo. Kidder excludes prescribed bands.',
        entropy='Sum m log(A)/(gamma-1), using accepted u and mass density for CRK, thermodynamic A for PESPH. Undefined global entropy is null; Kidder also reports interior entropy.',
        energy='Sum m (u+v²/2), referenced to the accepted initial state after each scheme initialization. Legacy case relativeEnergyDrift fields can reference pre-initialization values and are not the comparison metric. Driven Kidder/RT/Yee energy changes are not closed-system conservation tests.',
        momentum='momentumChangeNorm is the full-vector final-minus-initial norm, evaluated from saved states.',
        videos='Particle replays wrap display coordinates and use per-case temporal envelopes of frame 1–99% quantiles, shared across schemes. Raw trajectories and original movies remain unchanged. Static comparisons pool final-frame quantiles; profiles include full extrema.',
        mixing='Fixed 64²-bin mass-weighted original-label mixing proxy, not molecular diffusion.',
        shockRadius='Density-peak radial-bin estimator; bin width is 1.875% of analytic radius. Sub-bin errors do not imply sub-bin accuracy.'))
    dump(root/'results.json', records)
    return records


def replay_videos(root, manifest):
    """Publish wrapped particle movies with shared scales, after GPU runs finish."""
    from warpSPHBootstrap import bootstrap
    bootstrap('float64')
    from warpSPH.runner.media import encodeFrames
    from pyvirtualdisplay import Display
    from vispy import app
    import glfw
    jobs, ranges = [], {}
    for row in manifest['rows']:
        directory = root/row['id']
        path = directory/'result.json'
        snapshots = sorted((directory/'snapshots').glob('*.npz'))
        if not path.exists() or not snapshots:
            continue
        record = json.loads(path.read_text())
        if record.get('status') == 'running':
            continue
        with np.load(snapshots[0]) as first:
            particle2d = first['positions'].shape[1] == 2 and row['case'] not in ('sedov', 'sod2d')
            hasPhase = np.ptp(first['phase']) > 0
        if not particle2d:
            continue
        bounds = ranges.setdefault(row['name'], [[np.inf, -np.inf] for _ in range(3)])
        for snapshot in snapshots:
            with np.load(snapshot) as a:
                values = (a['densities'], a['pressures'], a['phase'] if hasPhase else np.linalg.norm(a['velocities'], axis=1))
                for i, y in enumerate(values):
                    finite = y[np.isfinite(y)]
                    if finite.size:
                        lo, hi = np.quantile(finite, (.01, .99))
                        bounds[i][0] = min(bounds[i][0], float(lo))
                        bounds[i][1] = max(bounds[i][1], float(hi))
        if hasPhase:
            bounds[2][:] = [0., 1.]
        jobs.append((row, directory, path, snapshots, record))
    for bounds in ranges.values():
        for pair in bounds:
            if not np.all(np.isfinite(pair)):
                pair[:] = [0., 1.]
            if pair[1] <= pair[0]:
                pad = max(abs(pair[0])*.05, 1e-9)
                pair[:] = [pair[0]-pad, pair[1]+pad]
    dump(root/'videoRanges.json', dict(definition='Per-case envelope of per-frame 1–99% quantiles, pooled across schemes and time; original phase fixed at 0–1.', ranges=ranges))
    with Display(visible=False, size=(1200, 480)):
        app.use_app('glfw')
        try:
            for row, directory, path, snapshots, record in jobs:
                if record.get('videoReplayVersion') == 2 and record.get('videoReplayRanges') == ranges[row['name']]:
                    continue
                ctx = profile_context(root, record)
                if ctx is None:
                    continue
                ctx.imagePath = str(directory/'replay'/'images')
                Path(ctx.imagePath).mkdir(parents=True, exist_ok=True)
                renderer = Recording(ctx, None, directory, row['scheme'])
                renderer.ranges = ranges[row['name']]
                for snapshot in snapshots:
                    with np.load(snapshot) as data:
                        renderer.record_arrays(ctx, dict(data))
                renderer.close()
                record.setdefault('originalVideoPath', record.get('videoPath'))
                record['videoPath'] = encodeFrames(ctx.imagePath, str(directory/'replay'), framerate=10, fps=10, gif=False)
                record['videoReplayVersion'] = 2
                record['videoReplayRanges'] = ranges[row['name']]
                record['videoReplayReason'] = 'Periodic display wrapping and common temporal-envelope color scales only; original trajectory, frames and video retained.'
                dump(path, record)
                print('[PESPH replay]', row['id'], flush=True)
        finally:
            app.quit()
            glfw.terminate()


def report(root, manifest):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = collect(root, manifest)
    fields = sorted({k for r in records for k in r.get('metrics', {})})
    with (root/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'scheme', 'status', 'qualityFlags', *fields])
        writer.writeheader()
        for r in records:
            writer.writerow(dict(name=r['name'], scheme=r['scheme'], status=r['status'],
                                 qualityFlags='; '.join(r.get('qualityFlags', [])), **r.get('metrics', {})))
    cards = []
    for name in dict.fromkeys(row['name'] for row in manifest['rows']):
        selected = [r for r in records if r['name'] == name]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        for r in selected:
            path = root/r['id']/'history.json'
            if not path.exists():
                continue
            h = json.loads(path.read_text())
            if not h:
                continue
            t = np.array([x['t'] for x in h])
            e = np.array([x['physicalEnergy'] for x in h])
            axes[0].plot(t, e/e[0]-1, label=r['scheme'])
            axes[1].plot(t, [x['kineticEnergy'] for x in h], label=r['scheme'])
            key = next((k for k in ('cloudMassFraction', 'khAmplitude', 'modeAmplitude', 'pressureError') if k in h[0]), 'physicalEntropy')
            if r['case'] == 'kidder':
                key = 'interiorEntropy'
            diagnostic = r.get('entropyHistory') if key in ('physicalEntropy', 'interiorEntropy') else None
            if diagnostic or key in h[0]:
                axes[2].plot([x['t'] for x in diagnostic] if diagnostic else t,
                             [x[key] for x in diagnostic or h], label=r['scheme'])
            axes[2].set_title(key)
        for ax, title in zip(axes[:2], ('Relative energy change', 'Kinetic energy')):
            ax.set_title(title)
        for ax in axes:
            ax.set_xlabel('Time'); ax.grid(alpha=.2)
        if axes[0].lines:
            axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(root/(name+'_histories.png'), dpi=140); plt.close(fig)
        available = []
        for r in selected:
            snapshots = sorted((root/r['id']/'snapshots').glob('*.npz'))
            if snapshots:
                a = dict(np.load(snapshots[-1]))
                context = profile_context(root, r)
                if context is not None:
                    a['displayPositions'] = display_positions(a, context.config.domain)
                    _, _, reference = profiles(context, a)
                    if reference is not None:
                        a.update(referenceX=reference[0], referenceDensities=reference[1][0],
                                 referencePressures=reference[1][1], referenceVelocities=reference[1][2])
                available.append((r, a))
        if available:
            fig, axes = plt.subplots(3, len(available), figsize=(5*len(available), 10), squeeze=False)
            particle2d = available[0][1]['positions'].shape[1] == 2 and not name.startswith('sedov') and not name.startswith('sod')
            hasPhase = particle2d and any(np.ptp(a['phase']) for _, a in available)
            for _, a in available:
                a['velocity'] = (a['phase'] if hasPhase else np.linalg.norm(a['velocities'], axis=1)) if particle2d or name.startswith('sedov') else a['velocities'][:, 0]
                if 'referenceVelocities' in a:
                    a['referenceVelocity'] = a['referenceVelocities']
            for q, quantity in enumerate(('densities', 'pressures', 'velocity')):
                label = ('original phase' if hasPhase else 'speed') if q == 2 and particle2d else quantity
                lo = min(np.quantile(a[quantity], .01) for _, a in available)
                hi = max(np.quantile(a[quantity], .99) for _, a in available)
                if q == 2 and hasPhase:
                    lo, hi = 0., 1.
                if not particle2d:
                    values = [a[quantity] for _, a in available]
                    referenceKey = 'reference'+quantity.title()
                    values += [a[referenceKey] for _, a in available if referenceKey in a]
                    lo, hi = min(np.min(v) for v in values), max(np.max(v) for v in values)
                for col, (r, a) in enumerate(available):
                    ax = axes[q, col]
                    if particle2d:
                        artist = ax.scatter(*a.get('displayPositions', a['positions'])[:, :2].T, c=a[quantity], s=2, vmin=lo, vmax=hi, cmap='viridis', rasterized=True)
                        fig.colorbar(artist, ax=ax); ax.set_aspect('equal')
                    else:
                        x = np.linalg.norm(a['positions'], axis=1) if name.startswith('sedov') else a['positions'][:, 0]
                        ax.scatter(x, a[quantity], s=2, alpha=.5)
                        referenceKey = 'reference'+quantity.title()
                        if referenceKey in a:
                            ax.plot(a['referenceX'], a[referenceKey], color='#b52e23', linewidth=1)
                        ax.set_ylim(lo-.1*(hi-lo+1e-9), hi+.1*(hi-lo+1e-9))
                    ax.set_title(f'{r["scheme"]} / {label} / t={float(a["time"]):.4g}')
                xlim = (min(ax.get_xlim()[0] for ax in axes[q]), max(ax.get_xlim()[1] for ax in axes[q]))
                ylim = (min(ax.get_ylim()[0] for ax in axes[q]), max(ax.get_ylim()[1] for ax in axes[q]))
                for ax in axes[q]:
                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
            warnings = '; '.join(r['scheme']+': '+', '.join(r.get('qualityFlags', [])) for r, _ in available if r.get('qualityFlags'))
            fig.suptitle(name+' | red: analytic where available'+(' | '+warnings if warnings else ''), fontsize=9)
            fig.tight_layout(); fig.savefig(root/(name+'_final.png'), dpi=150); plt.close(fig)
        table = []
        videos = []
        for r in selected:
            m = r.get('metrics', {})
            velocityKey = next((k for k in ('normalizedVelocityL1', 'speedL1', 'velocityL1') if k in m), 'velocityL1')
            values = [r['scheme'], r['status']+(' / CHECK' if r.get('qualityFlags') else ''), *(f'{m[k]:.5g}' if isinstance(m.get(k), (int, float)) else '' for k in
                      ('tFinal', 'initialEnergy', 'maxRelativeEnergyChange', 'densityL1', velocityKey, 'gridMixing', 'minPressure', 'medianStepMs', 'wallSeconds'))]
            configPaths = sorted((root/r['id']/'run').glob('*/config.json'))
            schemeCell = html.escape(r['scheme'])
            if configPaths:
                schemeCell = '<a href="'+html.escape(str(configPaths[-1].relative_to(root)))+'">'+schemeCell+'</a>'
            table.append('<tr><td>'+schemeCell+'</td>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values[1:])+'</tr>')
            if r.get('qualityFlags'):
                table.append('<tr><td colspan="11" style="text-align:left;color:#963c12">'+html.escape(r['scheme']+': '+'; '.join(r['qualityFlags']))+'</td></tr>')
            if r.get('failureReason'):
                table.append('<tr><td colspan="11" style="text-align:left;color:#963c12">'+html.escape(r['scheme']+': '+r['failureReason'])+'</td></tr>')
            if r.get('videoPath'):
                videos.append(f'<div><b>{r["scheme"]}</b><video controls preload="metadata" src="{Path(r["videoPath"]).relative_to(root)}"></video></div>')
            if r.get('error'):
                table.append('<tr><td colspan="11"><details><summary>Error</summary><pre>'+html.escape(r['error'])+'</pre></details></td></tr>')
        diagnosticsTable = []
        for key in ('initialEnergyRelativeToCRK', 'thermodynamicDensityL1', 'entropyChange', 'interiorEntropyChange', 'startupRelativeEnergyChange', 'postStartupRelativeEnergyChange',
                    'kineticEnergyRetention', 'pressureError', 'maxPeculiarVelocity',
                    'shapeDisplacement', 'densityRelativeL1', 'khAmplitude', 'khAmplitudeAt1p5',
                    'khAmplitudeMax', 'postShockDensityRelativeError', 'shockRadiusRelativeError',
                    'shockFrontRelativeError', 'tangentialVelocityL1', 'angularMomentumLoss',
                    'cloudMassFraction', 'cloudMassFractionInitial', 'cloudColdRetentionRelativeToInitial',
                    'cloudDenseFraction', 'cloudDenseFractionInitial', 'cloudMeanEnergy', 'cloudMeanEnergyInitial',
                    'tKHToHalfInitialColdMass', 'tKHToTenPercentInitialColdMass', 'modeAmplitude', 'heavyMinY',
                    'lightMaxY', 'mixingWidth', 'heavyCenterY', 'lightCenterY', 'injectedEnergyRecovery',
                    'minFinalInteriorEnergy', 'negativeFinalEnergyInteriorParticles', 'negativeFinalEnergyBandParticles'):
            if any(r.get('metrics', {}).get(key) is not None for r in selected):
                values = [f'{value:.6g}' if isinstance(value := r.get('metrics', {}).get(key), (int, float)) else '' for r in selected]
                diagnosticsTable.append('<tr><td>'+key+'</td>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
        extraTable = '<table><tr><th>Diagnostic</th>'+''.join('<th>'+r['scheme']+'</th>' for r in selected)+'</tr>'+''.join(diagnosticsTable)+'</table>'
        caseNote = ('<p>Kidder analytic errors and the entropy plot exclude the prescribed boundary bands. '
                    'Accepted-state positivity is also checked on the bands; negative band energies are reported separately. '
                    'Global thermodynamic entropy is undefined whenever any saved internal energy is non-positive.</p>'
                    if name == 'kidder' else '')
        finalImage = (f'<img src="{name}_final.png">' if available else
                      '<p>No initial state was recorded: inspect the preflight error above.</p>')
        cards.append(f'<section id="{name}"><h2>{name}</h2><table><tr>'+''.join('<th>'+h+'</th>' for h in
            ('Scheme', 'Status', 't', 'initial E', 'max |ΔE/E|', 'L1 rho', 'L1 v', 'grid mixing', 'min P', 'ms/step', 'wall s'))+
            '</tr>'+''.join(table)+'</table>'+extraTable+caseNote+f'<img src="{name}_histories.png">'+finalImage+
            '<div class="videos">'+''.join(videos)+'</div></section>')
    names = list(dict.fromkeys(row['name'] for row in manifest['rows']))
    statuses = {status: sum(r['status'] == status for r in records) for status in dict.fromkeys(r['status'] for r in records)}
    fig, ax = plt.subplots(figsize=(11, max(4, len(names)*.37)))
    y = np.arange(len(names))
    presentSchemes = tuple(dict.fromkeys(row['scheme'] for row in manifest['rows']))
    for offset, scheme in zip(np.linspace(-.3, .3, len(presentSchemes)), presentSchemes):
        values = [next((r.get('metrics', {}).get('medianStepMs', np.nan) for r in records if r['name']==name and r['scheme']==scheme), np.nan) for name in names]
        ax.barh(y+offset, values, height=.75/len(presentSchemes), label=scheme)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    hasTimings = any(np.isfinite(value) and value > 0 for patch in ax.containers for value in patch.datavalues)
    if hasTimings:
        ax.set_xscale('log')
    else:
        ax.text(.5,.5,'No accepted step timings',ha='center',transform=ax.transAxes)
    ax.set_xlabel('Median integrator ms/step'+(' (log scale; missing = unavailable)' if hasTimings else ' (unavailable)'))
    ax.grid(axis='x', alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root/'cost_comparison.png', dpi=140)
    plt.close(fig)
    document = ('<!doctype html><meta charset="utf-8"><title>PESPH / CRKSPH evaluation</title>'
        '<style>body{font:15px system-ui;margin:30px;background:#f5f6f7;color:#18212b}section{background:white;padding:20px;margin:25px 0}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}.videos{display:flex;gap:12px}.videos>div{width:33%}video{width:100%}pre{white-space:pre-wrap;text-align:left}</style>'
        '<h1>'+html.escape(' / '.join(presentSchemes))+'</h1><p>'+html.escape(manifest['comparison'])+'</p><p>'+html.escape(str(len(records))+' / '+str(len(manifest['rows']))+' rows recorded: '+', '.join(f'{n} {status}' for status, n in statuses.items()))+'</p><nav>'+ ' · '.join(f'<a href="#{name}">{name}</a>' for name in names)+'</nav><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in manifest['limitations'])+'</ul>'
        '<p>Full metrics: <a href="metrics.csv">CSV</a> · <a href="results.json">JSON</a>. Complete means the integration reached its target, not that accuracy/conservation gates passed. CHECK flags use explicit diagnostic thresholds, not a universal solver ranking. Particle color limits use pooled 1–99% ranges; profile limits include full extrema and analytic peaks. Failed/early rows and blast-energy mismatches are not matched-time/matched-energy accuracy winners. L1 rho is a particle mean absolute error; CSV also includes approximate volume weighting. L1 v is normalized by wave amplitude for linearWave, speed error for vortices, and absolute profile error otherwise. Grid mixing is a fixed 64² bin proxy of original particle labels, not physical molecular diffusion. Shock-radius bins are 1.875% of the analytic radius; smaller reported radius errors are not resolved precision. Median ms/step measures the integrator, including PESPH accepted-state support/closure refresh. All case timestep/CFL hooks run outside this timer. Use wall seconds for end-to-end cost, including initialization, compilation, diagnostics and video; CSV also records step count, wall/step and simulated-time throughput.</p><img src="cost_comparison.png">'+''.join(cards))
    fixedSteps = [row for row in manifest['rows'] if row['configuration']['simulation'].get('nSteps') is not None]
    if fixedSteps:
        document = document.replace('Complete means the integration reached its target, not that accuracy/conservation gates passed.',
            'Rows with an explicit step cap are fixed-step probes: complete means their requested step count was accepted. Other rows must reach their physical end time. Fixed-step completion does not establish full-endpoint accuracy or conservation acceptance.')
    if (root/'conductivity_coefficient_probe.json').exists():
        document = document.replace('<nav>', '<p style="background:#fff0d0;padding:16px"><b>As-evaluated conductivity defect:</b> '
            'the shared operator returns before applying thermalConductivity. Nominal strengths 0, 0.5 and 1 give identical '
            'nonzero rates in the saved operator probe. PESPH rows therefore use unscaled native conductivity; CRKSPH '
            'does not call that operator. These are native-preset comparisons, not an isolated pressure-formulation '
            'ablation. Explicit source-off controls and later source corrections are reported separately; these frozen '
            'rows do not validate those later changes.</p><nav>', 1)
    (root/'comparison.html').write_text(document)
    print('[PESPH suite] report:', root/'comparison.html', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan', 'run', 'worker', 'report', 'replay'))
    p.add_argument('--root', type=Path, default=Path('scripts/out_pesph_suite_20261022'))
    p.add_argument('--cases', nargs='+')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--row')
    p.add_argument('--retry', action='store_true')
    p.add_argument('--wait-pattern')
    a = p.parse_args()
    root = a.root.resolve()
    manifest = (json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() and a.action != 'plan'
                else make_plan(root, a.cases, a.quick, a.device))
    if a.action == 'worker':
        return worker(root, next(r for r in manifest['rows'] if r['id'] == a.row))
    if a.action == 'run':
        run_suite(root, manifest, a.wait_pattern, a.retry)
    elif a.action == 'report':
        report(root, manifest)
    elif a.action == 'replay':
        replay_videos(root, manifest)
    else:
        print(f'{len(manifest["rows"])} rows: {root/"manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
