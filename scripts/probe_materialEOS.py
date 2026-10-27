"""Sequential, recorded contact/planar/RM validation with machine-readable metrics.

Example: python scripts/probe_materialEOS.py --mode contact --nx 64
The periodic seam is deliberately distant. Only central measurements qualify
the contact/shock interaction. Every run records video and progress.
"""
import argparse
import json
from pathlib import Path
import sys

from warpSPHBootstrap import bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('contact', 'planar', 'rm'), default='contact')
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--time', type=float, default=1.)
    parser.add_argument('--precision', default='float32')
    parser.add_argument('--scheme', choices=('CRKSPH', 'CompSPH', 'Monaghan'), default='CRKSPH')
    parser.add_argument('--execution', choices=('eager', 'graph'), default='eager')
    parser.add_argument('--amplitude', type=float, default=.01)
    parser.add_argument('--boost', type=float, default=0.)
    parser.add_argument('--mach', type=float, default=1.2)
    parser.add_argument('--switch', choices=('NoneSwitch', 'CullenDehnen2010'), default='NoneSwitch')
    parser.add_argument('--cfl', type=float, default=.15)
    parser.add_argument('--densityPolicy', '--density-policy', choices=('legacy', 'sameMaterial', 'massReplacement'), default='legacy')
    parser.add_argument('--live', action='store_true', help='Register with the resident Remote companion for live monitoring and controls.')
    parser.add_argument('--name', help='Name displayed in the Remote simulation list.')
    parser.add_argument('--xLeft', type=float, default=-4.)
    parser.add_argument('--xRight', type=float, default=6.)
    parser.add_argument('--output', default='results/materialEOS')
    args = parser.parse_args()
    if args.scheme != 'CRKSPH' and args.densityPolicy != 'legacy':
        parser.error('Material density policies are specific to CRKSPH')
    bootstrap(precision=args.precision)
    import torch
    from warpSPH.runner import run
    from warpSPH.cases.compressible.richtmyerMeshkov import richtmyerMeshkovCase

    params = dict(amplitude=args.amplitude if args.mode == 'rm' else 0.,
                  mach=1. if args.mode == 'contact' else args.mach, xLeft=args.xLeft, xRight=args.xRight,
                  boost=args.boost, viscositySwitch=args.switch)
    publisher = None
    case = richtmyerMeshkovCase
    if args.live:
        from warpsph_remote.publisher import SimulationPublisher
        publisher = SimulationPublisher(args.name or f'RM {args.mode} nx={args.nx} {args.switch}',
                                        measurements=True, diagnostic_interval=.05)
        case = publisher.bind_case(case)
        print(f'Remote run registered: {publisher.id} ({publisher.name})', flush=True)
    try:
        result = run(case, scheme=args.scheme, nx=args.nx, tLimit=args.time,
            schemeOverrides={'crkDensityPolicy': args.densityPolicy} if args.scheme == 'CRKSPH' else {},
            cflFactor=args.cfl, execution=args.execution, params=params, plot=True, video=True, progress=True,
            plotInterval=50, store=args.execution != 'graph', storeMode='trajectory', storeInterval=50,
            holdPlot=False, show=False, exportRoot=args.output, stallProgress=1e-3,
            velocityAlarmPlotInterval=1)
        if publisher is not None:
            publisher.recording(result.ctx)
            publisher.finish(result.ctx, result.state, result.nSteps)
            publisher.update(step=result.nSteps, time=float(result.state.t), dt=float(result.ctx.config.dt),
                             particles=int(result.state.state.positions.shape[0]))
            publisher.close('failed' if result.diverged or result.stopReason else 'completed', result.stopReason or '')
    except BaseException as error:
        if publisher is not None:
            from warpsph_remote.publisher import SimulationStopped
            publisher.close('stopped' if isinstance(error, SimulationStopped) else 'failed', str(error))
            if isinstance(error, SimulationStopped):
                print('Stopped through Remote; remaining study runs must be launched explicitly.', flush=True)
                return 130
        raise
    ctx, state = result.ctx, result.state.state
    central = (state.positions[:, 0] - args.boost * float(result.state.t)).abs() < .5
    peculiar = state.velocities.clone()
    peculiar[:, 0] -= args.boost
    values = dict(centralPressureError=(state.pressures[central] - 1).abs().max(),
                  centralMaxVelocity=torch.linalg.vector_norm(peculiar[central], dim=-1).max())
    summary = dict(mode=args.mode, nx=args.nx, precision=args.precision, scheme=args.scheme,
        execution=args.execution, viscositySwitch=args.switch, cfl=args.cfl,
        densityPolicy=getattr(ctx.schemeConfig, 'crkDensityPolicy', None),
        remoteRunID=publisher.id if publisher is not None else None,
        steps=result.nSteps, time=float(result.state.t),
        effectiveExecution=ctx.scratch.get('compSPHStepMode', 'eager'),
        initialAmplitude=float(result.series('interfaceCosAmplitude')[0]),
        finalAmplitude=float(result.series('interfaceCosAmplitude')[-1]),
        relativeEnergyDrift=float(result.series('relativeEnergyDrift')[-1]),
        **{k: float(v.detach().cpu()) for k, v in values.items()},
        shock=ctx.scratch['rmShock'], star=ctx.scratch['rmStar'])
    for name in ('circulationTime', 'circulationUpper', 'circulationLower', 'circulationPositive',
                 'circulationNegative', 'enstrophy', 'transverseKineticEnergy', 'markerLayerWidth',
                 'bubbleExtent', 'spikeExtent', 'interactionAlphaMean', 'interactionAlphaActiveVolume'):
        summary[name] = float(result.series(name)[-1])
    summary['initialMarkerLayerWidth'] = float(result.series('markerLayerWidth')[0])
    from warpSPH.caseUtils.compressible.richtmyerMeshkov.reference import weakShockReference
    gamma0, gamma1 = (m.eos.gamma for m in ctx.schemeConfig.materials)
    summary['weakShockReference'] = weakShockReference(ctx.param('rho0'), ctx.param('rho1'),
        ctx.param('pressure'), gamma0, gamma1, ctx.param('mach'), ctx.param('amplitude'), ctx.spec.L)
    summary.update(wallSeconds=result.wallTime,
        meanStepMilliseconds=float(result.series('stepTime_ms').mean()) if result.series('stepTime_ms').size else None,
        stoppedEarly=bool(result.diverged or result.stopReason), stopReason=result.stopReason)
    if args.mode == 'rm':
        import numpy as np
        times = result.series('t')
        amplitude = result.series('interfaceCosAmplitude')
        fit = (times >= 1.) & (times <= min(1.8, args.time))
        if fit.sum() >= 3:
            summary['growthFitWindow'] = [1., min(1.8, args.time)]
            summary['growthRate'] = float(np.polyfit(times[fit], amplitude[fit], 1)[0])
    if args.mode == 'planar':
        shock, star = summary['shock'], summary['star']
        since = summary['time'] + params.get('shockPosition', -1.) / shock['speed']
        if since > 0:
            origin = args.boost / shock['speed']
            contact = origin + star['velocity'] * since
            x = state.positions[:, 0]
            for label, wave in (('Left', star['waveLeft']), ('Right', star['waveRight'])):
                boundary = origin + wave * since
                low, high = sorted((contact, boundary))
                plateau = (x > low + .25 * (high - low)) & (x < high - .25 * (high - low))
                if plateau.any():
                    measured = float(state.pressures[plateau].median().detach().cpu())
                    velocity = float(state.velocities[plateau, 0].median().detach().cpu())
                    summary[f'plateau{label}PressureError'] = abs(measured / star['pressure'] - 1)
                    summary[f'plateau{label}VelocityError'] = abs(velocity / star['velocity'] - 1)
    path = Path(ctx.exportPath) / 'materialValidation.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    (path.parent / 'diagnosticHistory.json').write_text(json.dumps(result.trajectory, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    if result.diverged or result.stopReason:
        print(f'Run ended before successful completion: {result.stopReason}', file=sys.stderr, flush=True)
        return 1
    if args.mode == 'contact' and (summary['centralPressureError'] > .01 or summary['centralMaxVelocity'] > .01 * (5/9)**.5):
        print(f'Contact gate failed: {path}', file=sys.stderr, flush=True)
        return 1
    if args.mode == 'planar' and any(value > .02 for key, value in summary.items() if key.startswith('plateau') and key.endswith('Error')):
        print(f'Planar plateau gate failed: {path}', file=sys.stderr, flush=True)
        return 1
    if not torch.isfinite(state.internalEnergies).all() or (state.internalEnergies <= 0).any():
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
