"""Daemon-recorded 1D/3D planar Sod qualification of locally generated Owen tables.

Identical EOS on two material labels isolates geometry/support rather than a
new Riemann solver. Compare hermiteShell and lattice with weighting enabled
in both, before each fresh midpoint RHS, without the blob's absolute bounds.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from compare_spheralMaterials import save, warp_run
from compare_spheralSod1d import exact
from study_crkEnergySuite import context as base_context, Measures as BaseMeasures, summarize as base_summary
from study_densityEvolution import install_audit as install_density_audit
from study_idealH import study

SOURCES = (__file__, 'scripts/study_crkEnergySuite.py', 'scripts/study_densityEvolution.py',
    'scripts/compare_spheralMaterials.py', 'scripts/compare_spheralSod1d.py',
    'src/warpSPH/modules/adaptiveSupport/owenHermite.py',
    'src/warpSPH/modules/adaptiveSupport/wp_materialPsi.py',
    'src/warpSPH/modules/adaptiveSupport/optimalSupportOwen.py',
    'src/warpSPH/modules/crk/evolution.py', 'src/warpSPH/schemes/compressible/crkSPH.py',
    'src/warpSPH/systems/compSPH.py', 'src/warpSPH/configurations/crkSPH.py',
    'src/warpSPH/configurations/compressibleConfig.py', 'src/warpSPH/runner/configuration.py')


def source_hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path, SOURCES)}


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for dim, nx in ((1, 100), (3, 40)):
        dx = 1 / nx
        longitudinal = -1.5 + (np.arange(3 * nx) + .5) * dx
        nt = 16 if dim == 3 else 1
        axes = [longitudinal] + [(-nt / 2 + np.arange(nt) + .5) * dx] * (dim - 1)
        x = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, dim)
        q, _ = exact(x[:, 0], 0.)
        rho, P = q['density'], q['pressure']
        n = len(x)
        initial = dict(positions=x, velocities=np.zeros_like(x), densities=rho, pressures=P,
            internalEnergies=P / (.4 * rho), masses=rho * dx**dim, supports=np.full(n, 4 * dx),
            materials=(x[:, 0] >= 0).astype(np.int32), UIDs=np.arange(n, dtype=np.int64),
            kinds=np.zeros(n, dtype=np.int32))
        for table in ('hermiteShell', 'lattice'):
            name = f'sod-{dim}d-{table}'
            folder = root / name
            folder.mkdir()
            np.savez_compressed(folder / 'input.npz', **initial)
            steps = 200
            settings = dict(case='sod', dim=dim, nx=nx, transversePlanes=nt, gamma=1.4,
                domain=[-1.5, 1.5], domainAxes=[[-1.5, 1.5]] + [[-nt*dx/2, nt*dx/2]]*(dim-1),
                end=.2, steps=steps, exactSteps=steps, dt=.2/steps, timeGrid=[.2/steps]*steps,
                frameEvery=5, policy=table, gammaVariant=f'{dim}D {table} weighted',
                table=table, schedule='paper', thermalGamma=1.4, choice=dict(compatible=True, thermal='base'),
                inputPath=str((folder/'input.npz').resolve()), precision='float64', kernel='B8',
                comparisonPolicy='Only Owen lookup/gradient calibration varies; material weighting enabled in both, '
                    'unchanged relative bounds/relaxation, before-RHS, fresh midpoint, paper density, '
                    'geometric volumes, native compatible energy, averaged Q, Cl2/Cq1 and AV R/4')
            save(folder/'settings.json', settings)
            rows.append(name)
    sources = source_hashes()
    (root/'sources').mkdir()
    for name, digest in sources.items():
        p = Path(name)
        shutil.copyfile(p, root/'sources'/f'{p.stem}_{digest[:12]}{p.suffix}')
    save(root/'provenance.json', dict(rows=rows, sources=sources,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        scope='Physical dimensional spot checks, not a spatial/temporal convergence study. '
            'Two material IDs with the same ideal-gas EOS preserve the classical Sod analytic reference. '
            'Periodic padded reservoirs keep the central |x|<0.5 comparison causally isolated from the outer seam. '
            '3D has genuine 3D neighbors/forces and a central-plane video; no imposed planar force constraint. '
            'No Spheral table files or research support bounds are used.'))


def context(s):
    from warpSPH.runner import RunConfiguration
    import torch
    from warpSPHCore import DomainDescription
    from warpSPH.cases.common.plotting import Field, particlePlot
    case, _, sim, params, solver = base_context(s)
    original_configure = case.configureScheme
    def configure(c):
        original_configure(c)
        c.config.domain.min[:] = torch.tensor(np.asarray(s['domainAxes'])[:, 0], device=c.device, dtype=c.dtype)
        c.config.domain.max[:] = torch.tensor(np.asarray(s['domainAxes'])[:, 1], device=c.device, dtype=c.dtype)
    case = replace(case, name=f'owenDimensions{s["dim"]}d', configureScheme=configure)
    if s['dim'] == 3:
        setup, update = particlePlot([Field('densities', 'Density: central z plane'), Field('pressures', 'Pressure: central z plane')])
        def projected(c, system):
            state = system.state
            mask = state.UIDs.remainder(s['transversePlanes']) == s['transversePlanes']//2
            visible = copy.copy(system)
            visible.state = copy.copy(state)
            for name, tensor in vars(state).items():
                if isinstance(tensor, torch.Tensor) and tensor.ndim and tensor.shape[0] == len(state.masses):
                    value = tensor[mask]
                    if value.ndim == 2 and value.shape[1] == 3:
                        value = value[:, :2].contiguous()
                    setattr(visible.state, name, value)
            cfg = copy.copy(c.config)
            a = np.asarray(s['domainAxes'])[:2]
            cfg.domain = DomainDescription(torch.tensor(a[:, 0], device=c.device, dtype=c.dtype),
                torch.tensor(a[:, 1], device=c.device, dtype=c.dtype), torch.zeros(2, device=c.device, dtype=torch.bool), 2)
            return replace(c, config=cfg), visible
        def draw(c, system):
            return setup(*projected(c, system))
        def refresh(c, system, plot, step):
            return update(*projected(c, system), plot, step)
        draw.usesWarp = refresh.usesWarp = False
        case = replace(case, setupPlot=draw, updatePlot=refresh)
    solver.update(owenTable=s['table'], owenMaterialWeights=True, crkEvolution='paper', crkDensityPolicy='massReplacement',
        materials=[dict(name=name, eos=dict(type='idealGas', gamma=1.4)) for name in ('left', 'right')],
        crkViscosityParams=dict(oneSidedWork=False, viscosityKernelScale=4.), diffusionParams=dict(C_l=2., C_q=1.))
    return case, RunConfiguration(case.name, simulation=sim, params=params, solver=solver).prepare(case), sim, params, solver


class Measures(BaseMeasures):
    def __call__(self, a, t):
        row = super().__call__(a, t)
        row.update(minSupport=float(a['supports'].min()), maxSupport=float(a['supports'].max()),
            transverseVelocityMax=float(np.linalg.norm(a['velocities'][:, 1:], axis=1).max()),
            transversePlaneDensitySpread=0., transversePlaneSupportSpread=0., transversePositionDrift=0.)
        if self.settings['dim'] == 3:
            nt = self.settings['transversePlanes']
            for field, key in [('densities', 'transversePlaneDensitySpread'), ('supports', 'transversePlaneSupportSpread')]:
                row[key] = float(np.ptp(a[field].reshape(-1, nt*nt), axis=1).max())
            width = self.settings['domainAxes'][1][1] - self.settings['domainAxes'][1][0]
            delta = (a['positions'][:, 1:] - self.initial['positions'][:, 1:] + width/2) % width - width/2
            row['transversePositionDrift'] = float(abs(delta).max())
        return row


def summarize(root, solver, s, history, initial, final, wall, steps, remote, extra):
    row = base_summary(root, solver, s, history, initial, final, wall, steps, remote, extra)
    maxima = {k: max(abs(v[k]) for v in history) for k in history[0] if k != 'time'}
    identities = {k: bool(np.array_equal(initial[k], final[k])) for k in ('masses', 'materials', 'UIDs')}
    row['maxima'] = maxima
    row['gates'].update(identities=all(identities.values()), energy=maxima['relativeEnergyDrift'] < 1.e-10,
        momentum=maxima['momentumError'] < 1.e-10, planar=maxima['transverseVelocityMax'] < 1.e-8,
        analyticAccuracy=all(row['final'][k] < .15 for k in ('densityL1', 'pressureL1', 'velocityL1')))
    if s['dim'] == 3:
        row['gates']['periodicSupportFits'] = maxima['maxSupport'] < .5*(s['domainAxes'][1][1]-s['domainAxes'][1][0])
    row['identities'] = identities
    row['passed'] = all(row['gates'].values())
    save(root/'comparisonRun.json', row)
    print('DIMENSION RESULT', s['dim'], s['table'], 'passed', row['passed'], 'gates', row['gates'], flush=True)
    return row


def run_row(args):
    sys.path.insert(0, str(args.publisher_path))
    s = json.loads((args.output/'settings.json').read_text())
    expected = json.loads((args.output.parent/'provenance.json').read_text())['sources']
    assert source_hashes() == expected, 'Numerical source changed since preparation'
    density = install_density_audit(s)
    module = importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    solve = module.solveStageSupport
    support = dict(calls=0, table=s['table'], materialWeights=True, absoluteBounds=False, timing='before each fresh RHS')
    def observed(system, state, c, sc, **kwargs):
        assert sc.owenTable == s['table'] and sc.owenMaterialWeights and sc.adaptiveSupportIterations == 1
        assert sc.crkEnergyPartition == 'native' and sc.crkEnergyPredictor == 'raw'
        support['calls'] += 1
        return solve(system, state, c, sc, **kwargs)
    module.solveStageSupport = observed
    def initialize(c, system, a, settings):
        system.state.crkVolumes = None
        c.scratch['_titleProvenance'] = f'Owen dimensional qualification | {s["dim"]}D {s["table"]} weighted | fresh midpoint'
        save(args.output/'resolvedPolicy.json', dict(table=c.schemeConfig.owenTable, weighted=c.schemeConfig.owenMaterialWeights,
            dim=c.config.domain.dim, materials=[m.name for m in c.schemeConfig.materials],
            integration=c.config.integrationScheme.name, density=c.schemeConfig.crkEvolution,
            compatible=c.schemeConfig.compatibleEnergy, supportIterations=c.schemeConfig.adaptiveSupportIterations,
            absoluteBounds=False, domainAxes=s['domainAxes']))
    try:
        return warp_run(args, s, context, Measures, summarize, 'Owen dimensions', initialize_factory=initialize)
    finally:
        save(args.output/'densityAudit.json', density)
        save(args.output/'supportAudit.json', support)


def analyze(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((root/'provenance.json').read_text())
    assert source_hashes() == provenance['sources']
    rows, states = {}, {}
    for name in provenance['rows']:
        folder = root/name
        row = json.loads((folder/'warp/comparisonRun.json').read_text())
        s = row['settings']
        assert row['passed'], (name, row['gates'])
        initial = dict(np.load(folder/'input.npz'))
        a = dict(np.load(folder/'warp/finalState.npz'))
        for k in ('masses', 'materials', 'UIDs'):
            np.testing.assert_array_equal(initial[k], a[k])
        h = json.loads((folder/'warp/diagnosticHistory.json').read_text())
        np.testing.assert_allclose([v['time'] for v in h], np.r_[0, np.cumsum(s['timeGrid'])], rtol=0, atol=1.e-12)
        da = json.loads((folder/'densityAudit.json').read_text())
        sa = json.loads((folder/'supportAudit.json').read_text())
        assert da['rhs'] == sa['calls'] == 2*s['steps']
        assert len(da['resets']) == s['steps'] and da['continuityUpdates'] == 2*s['steps'] and da['negativeEOSCalls'] == 0
        row['densitySummary'] = dict(rhs=da['rhs'], resets=len(da['resets']), negativeEOSCalls=da['negativeEOSCalls'])
        row['supportSummary'] = sa
        rows[name], states[name] = row, a
    comparisons = {}
    for dim in (1, 3):
        a, b = (states[f'sod-{dim}d-{table}'] for table in ('hermiteShell', 'lattice'))
        comparisons[f'{dim}d'] = {k+'RelativeL1': float(np.mean(abs(a[k]-b[k]))/max(np.mean(abs(b[k])), 1.e-30))
            for k in ('densities', 'pressures', 'internalEnergies', 'supports')}
        comparisons[f'{dim}d']['positionRMS'] = float(np.sqrt(np.mean(np.sum((a['positions']-b['positions'])**2, axis=1))))
    save(report/'comparison.json', dict(provenance=provenance, rows=rows, policyComparisons=comparisons))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    xx = np.linspace(-.5, .5, 1800)
    q, _ = exact(xx, .2)
    for i, dim in enumerate((1, 3)):
        for ax, key, ref in zip(axes[i], ('densities', 'pressures', 'velocities'), ('density', 'pressure', 'velocity')):
            ax.plot(xx, q[ref], 'k-', lw=1, label='Exact Sod')
            for table, marker in [('lattice', 'x'), ('hermiteShell', '.')]:
                a = states[f'sod-{dim}d-{table}']
                s = rows[f'sod-{dim}d-{table}']['settings']
                mask = abs(a['positions'][:, 0]) < .5
                if dim == 3:
                    mask &= (a['UIDs'] % (s['transversePlanes']**2) == 0)
                yy = a[key][:, 0] if key == 'velocities' else a[key]
                ax.plot(a['positions'][mask, 0], yy[mask], marker, ms=3, label=table)
            ax.set(title=f'{dim}D {ref}, t=0.2', xlabel='x', xlim=(-.5, .5))
        axes[i, 0].legend()
    fig.savefig(report/'profiles.png', dpi=160)
    plt.close(fig)
    print('ANALYZED', len(rows), 'dimensional flow runs', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'study', 'row', 'analyze', 'verify'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--report', type=Path, default=Path('reports/materialEOS/owenDimensions'))
    p.add_argument('--publisher-path', type=Path, default=Path('/home/lu26029/dev/warpSPHRemote/python'))
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.mode == 'prepare':
        prepare(args.output)
    elif args.mode == 'study':
        study(args.output, row_script=__file__)
    elif args.mode == 'row':
        raise SystemExit(run_row(args))
    elif args.mode == 'analyze':
        analyze(args.output, args.report.resolve())
    else:
        sys.path.insert(0, str(args.publisher_path))
        from analyze_crkEnergySuite import verify
        verify(args.output, args.report.resolve())


if __name__ == '__main__':
    main()
