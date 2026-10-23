"""Reproducible PESPH acoustic/refinement and Sod smoke validation.

CPU by default; each run streams progress, has a stall watchdog and records
Vispy profile frames plus video. Results are diagnostics, not a solver preset.
Run from the checkout with PYTHONPATH=src and the warp conda interpreter.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np


def reference(ctx, system):
    s = system.state
    x = s.positions[:, 0].detach().cpu().numpy()
    t = float(system.t)
    if ctx.case.name == 'linearWave':
        amp, cs, rho0 = (ctx.param(k) for k in ('A', 'c_s', 'rho0'))
        dp = amp*np.sin(2*np.pi*(x-cs*t)/ctx.param('lamda'))
        return x, (rho0+dp/cs**2, rho0*cs**2/ctx.param('gamma')+dp, dp/(rho0*cs))
    from warpSPH.caseUtils.compressible.sod.sodSolution import solve
    left = tuple(ctx.param('left_'+k) for k in ('pressure', 'rho', 'velocity'))
    right = tuple(ctx.param('right_'+k) for k in ('pressure', 'rho', 'velocity'))
    # Mirrored periodic tubes: compare x >= 0 with the right-moving problem.
    _, _, data = solve(left, right, (0., 1., .5), max(t, 1e-12),
                       gamma=ctx.param('gamma'), npts=4000)
    return x, tuple(np.interp(abs(x), data['x'], data[k])*sign for k, sign in
                    (('rho', 1), ('p', 1), ('u', np.sign(x))))


class Profile:
    def __init__(self, ctx, system):
        from vispy import scene
        self.canvas = scene.SceneCanvas(size=(1200, 460), bgcolor='white', show=False)
        grid = self.canvas.central_widget.add_grid(margin=18, spacing=25)
        self.title = scene.Label('', color='black', font_size=14)
        grid.add_widget(self.title, row=0, col=0, col_span=3)
        self.views, self.lines, self.refs = [], [], []
        for i, title in enumerate(('Mass density', 'Pressure', 'Velocity x')):
            grid.add_widget(scene.Label(title, color='black', font_size=12), row=1, col=i)
            v = grid.add_view(row=2, col=i, border_color='#bbb')
            v.camera = 'panzoom'
            self.views.append(v)
            self.lines.append(scene.visuals.Line(color='#1269a3', width=2, parent=v.scene))
            self.refs.append(scene.visuals.Line(color='#c54b31', width=1, parent=v.scene))
        self.draw(ctx, system, 0)

    def draw(self, ctx, system, step):
        from PIL import Image
        s = system.state
        x, refs = reference(ctx, system)
        order = np.argsort(x)
        ys = (s.densities, s.pressures, s.velocities[:, 0])
        for v, line, ref, y, r in zip(self.views, self.lines, self.refs, ys, refs):
            y = y.detach().cpu().numpy()
            line.set_data(np.c_[x[order], y[order]].astype(np.float32))
            ref.set_data(np.c_[x[order], r[order]].astype(np.float32))
            lo, hi = min(y.min(), r.min()), max(y.max(), r.max())
            pad = max((hi-lo)*.15, abs(hi)*1e-5, 1e-8)
            v.camera.set_range(x=(x.min(), x.max()), y=(lo-pad, hi+pad), margin=0)
        self.title.text = (f'{ctx.case.name} / {ctx.schemeConfig.variant.name} / '
                           f'N={len(x)} / t={float(system.t):.5f}    '
                           'blue: particles; red: analytic')
        Image.fromarray(self.canvas.render()).save(Path(ctx.imagePath)/f'frame_{step:07d}.png')

    def close(self):
        self.canvas.close()


def setupPlot(ctx, system):
    s = system.state
    ctx.scratch['pesphInitial'] = dict(
        mass=float(s.masses.sum()), energy=float(s.totalEnergies.sum()),
        entropy=float((s.masses*s.entropies).sum()),
        momentum=float((s.masses*s.velocities[:, 0]).sum()))
    return Profile(ctx, system)


def updatePlot(ctx, system, handle, step):
    handle.draw(ctx, system, step+1)


def measure(result):
    ctx, system = result.ctx, result.state
    s = system.state
    x, refs = reference(ctx, system)
    if ctx.case.name == 'linearWave':
        a, cs, rho0 = (ctx.param(k) for k in ('A', 'c_s', 'rho0'))
        scales, mask = (a/cs**2, a, a/(rho0*cs)), np.ones(len(x), dtype=bool)
    else:
        scales, mask = (1., 1., 1.), x >= 0
    errors = {}
    for name, y, ref, scale in zip(('density', 'pressure', 'velocity'),
                                  (s.densities, s.pressures, s.velocities[:, 0]), refs, scales):
        y = y.detach().cpu().numpy()
        errors[name+'L1'] = float(np.mean(abs(y[mask]-ref[mask]))/scale)
    initial = ctx.scratch['pesphInitial']
    return dict(case=ctx.case.name, variant=ctx.schemeConfig.variant.name, nx=ctx.spec.nx,
                particles=len(x), dtCFL=ctx.spec.cflFactor, time=float(system.t),
                steps=result.nSteps, diverged=result.diverged, stopReason=result.stopReason,
                wallSeconds=result.wallTime,
                meanStepMs=float(np.mean(result.series('stepTime_ms'))),
                medianStepMs=float(np.median(result.series('stepTime_ms')[1:])),
                massDrift=float(s.masses.sum())-initial['mass'],
                relativeEnergyDrift=(float(s.totalEnergies.sum())-initial['energy'])/initial['energy'],
                relativeEntropyChange=(float((s.masses*s.entropies).sum())-initial['entropy'])/initial['entropy'],
                momentumChange=float((s.masses*s.velocities[:, 0]).sum())-initial['momentum'],
                minPressure=float(s.pressures.min()), minInternalEnergy=float(s.internalEnergies.min()),
                minMassDensity=float(s.densities.min()), minSupport=float(s.supports.min()),
                errors=errors, exportPath=result.exportPath, videoPath=result.videoPath)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('scripts/out_pesph'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--quick', action='store_true', help='short plotting/runner check only')
    parser.add_argument('--only', choices=('wave', 'sod', 'all'), default='all')
    args = parser.parse_args()
    from warpSPHBootstrap import bootstrap
    bootstrap('float64')
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.linearWave import linearWaveCase
    from warpSPH.cases.compressible.sod import sodCase
    from pyvirtualdisplay import Display
    from vispy import app

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.only != 'sod':
        for variant in ('PressureEnergy', 'PressureEntropy'):
            for nx in ((48,) if args.quick else (48, 96)):
                jobs.append((linearWaveCase, variant, nx, .03 if args.quick else .5, .15, False))
    if args.only != 'wave':
        for cfl in ((.1,) if args.quick else (.1, .05)):
            jobs.append((sodCase, 'PressureEnergy', 128, .01 if args.quick else .1, cfl, True))
    rows = []
    with Display(visible=False, size=(1200, 460)):
        app.use_app('glfw')
        for case, variant, nx, end, cfl, diss in jobs:
            name = f'{case.name}_{variant}_{nx}_cfl{cfl}'
            print(f'\nPESPH PROBE {name}', flush=True)
            config = RunConfiguration(case.name, variant='classic' if case.name == 'sod' else 'default',
                simulation=dict(nx=nx, tLimit=end, device=args.device, precision='float64',
                                integrationScheme='rungeKutta4', cflFactor=cfl,
                                calibrateNormalization=True, progress=True, quiet=True,
                                plot=True, video=True, show=False, plotBackend='vispy',
                                plotInterval=5, store=False, exportRoot=str(args.out/name),
                                velocityAlarmPlotInterval=1, stallProgress=1e-3),
                params={'A': 1e-3} if case.name == 'linearWave' else {},
                solver=dict(scheme='PESPH', variant=variant, dissipation=diss))
            visualCase = replace(case, setupPlot=setupPlot, updatePlot=updatePlot)
            result = config.run(visualCase)
            row = measure(result)
            rows.append(row)
            (args.out/'metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(json.dumps(row, indent=2), flush=True)
            if result.diverged or result.stopReason:
                raise RuntimeError(f'{name}: {result.stopReason}')
        app.quit()
        import glfw
        glfw.terminate()


if __name__ == '__main__':
    main()
