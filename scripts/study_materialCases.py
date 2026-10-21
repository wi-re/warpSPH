"""Recorded, sequential, daemon-visible material controls.

python scripts/study_materialCases.py --phase pilot
python scripts/study_materialCases.py --phase refinement
Remote Stop or a numerical/qualification failure halts the queue. A manifest,
per-run diagnostics and final states are kept even when a gate fails.
"""
import argparse
import datetime
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time


def materials(g0=1.4, g1=5/3):
    return [dict(name=name, eos=dict(type='idealGas', gamma=g))
            for name, g in zip(('background', 'inclusion'), (g0, g1))]


def studyCases():
    def box(name, nx=64, duration=1., explicit=True, **params):
        return dict(name=name, case='hydrostatic', nx=nx, time=duration,
            params=dict(rho_high=4., gamma=1.5, viscositySwitch='NoneSwitch', **params),
            solver={'materials': materials(1.5, 1.5)} if explicit else {})
    def sod(name, nx=200, equal=False):
        return dict(name=name, case='twoGasSod', nx=nx, time=.15, params={},
                    solver={'materials': materials(5/3, 5/3) if equal else materials()})
    def bubble(name, nx=100, summed=False):
        return dict(name=name, case='shockBubble', nx=nx, time=.3,
                    params={'initialDensity': 'summed' if summed else 'prescribed'}, solver={})
    pilot = [box('box-legacy', explicit=False), box('box-labelled'),
        dict(box('box-two-gamma'), solver={'materials': materials()}),
        dict(box('ellipse-advected', shape='ellipse', boost=1.), solver={'materials': materials()}),
        sod('sod-equal-gamma', equal=True), sod('sod-two-gamma'),
        bubble('bubble-prescribed'), bubble('bubble-summed', summed=True)]
    refinement = [box('box-literature100', nx=100, duration=7.),
        dict(box('box-two-gamma100', nx=100, duration=7.), solver={'materials': materials()}),
        sod('sod-literature400', nx=400, equal=True),
        sod('sod-two-gamma400', nx=400), sod('sod-two-gamma800', nx=800),
        bubble('bubble-prescribed200', nx=200), bubble('bubble-summed200', nx=200, summed=True)]
    contact = [dict(bubble('bubble-contact-summed', summed=True), time=.02,
                    params={'initialDensity': 'summed', 'driverPressure': 1.}),
               dict(bubble('bubble-contact-prescribed'), time=.02,
                    params={'initialDensity': 'prescribed', 'driverPressure': 1.})]
    densityContact, densityRegression = [], []
    for policy in ('sameMaterial', 'massReplacement'):
        def withPolicy(job):
            return dict(job, name=job['name']+'-'+policy,
                solver={**job['solver'], 'crkDensityPolicy': policy})
        densityContact.append(withPolicy(dict(bubble('bubble-contact-prescribed'), time=.02,
            params={'initialDensity': 'prescribed', 'driverPressure': 1.})))
        densityRegression.extend([withPolicy(box('box-sharp', nx=100)),
            withPolicy(dict(box('ellipse-advected', shape='ellipse', boost=1.), solver={'materials': materials()})),
            withPolicy(sod('sod-two-gamma400', nx=400))])
    # Isolate Eq. 76's cross-material mass rule on the standard hydrostatic
    # case. Equal-gamma labels enable that rule without an EOS contrast.
    hydrostaticPolicy = [dict(name='hydrostatic-standard-'+policy,
        case='hydrostatic', nx=200, time=10., params={}, cflFactor=.3,
        retainInitial=True, solver=dict(materials=materials(5/3, 5/3),
                                       crkDensityPolicy=policy))
        for policy in ('legacy', 'massReplacement')]
    return dict(pilot=pilot, refinement=refinement, contact=contact,
                hydrostaticPolicy=hydrostaticPolicy,
                densityContact=densityContact, densityRegression=densityRegression,
                densityShock=[dict(name='lattice-shock-'+policy, case='sod2d', nx=100, time=.6, steps=220,
                    params={'equalMass': False, 'viscositySwitch': 'NoneSwitch'},
                    simulation={'supportMode': 'KernelMeanSymmetric'},
                    solver={'materials': materials(5/3, 5/3), 'crkDensityPolicy': policy})
                    for policy in ('sameMaterial', 'massReplacement')],
                densityBubble=[dict(bubble('bubble-prescribed100', nx=100), name='bubble-prescribed100-sameMaterial',
                    solver={'crkDensityPolicy': 'sameMaterial'})],
                densityLiterature=[dict(box('box-literature100', nx=100, duration=7.), name='box-literature100-'+policy,
                    solver={'materials': materials(1.5, 1.5), 'crkDensityPolicy': policy})
                    for policy in ('sameMaterial', 'massReplacement')],
                volumeContact=[dict(sod('contact-'+sampling+'-spheral'), time=.02,
                    params={'pressureRight': 1., 'equalMass': sampling == 'equalMass'},
                    solver={'materials': materials(), 'crkDensityPolicy': 'sameMaterial', 'crkEvolution': 'spheral'})
                    for sampling in ('equalSpacing', 'equalMass')],
                volumeRegression=[dict(box('box-volume100', nx=100),
                    solver={'materials': materials(1.5, 1.5), 'crkDensityPolicy': 'sameMaterial', 'crkEvolution': 'spheral'}),
                    dict(bubble('bubble-volume100'), time=.02, params={'initialDensity': 'prescribed', 'driverPressure': 1.},
                    solver={'crkDensityPolicy': 'sameMaterial', 'crkEvolution': 'spheral'})],
                densitySampling=[dict(sod('contact-'+sampling+'-'+policy), time=.02,
                    params={'pressureRight': 1., 'equalMass': sampling == 'equalMass'},
                    solver={'materials': materials(), 'crkDensityPolicy': policy})
                    for policy in ('sameMaterial', 'massReplacement') for sampling in ('equalSpacing', 'equalMass')])


def runCase(job, root, nsteps=None):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import numpy as np
    import torch
    from dataclasses import replace
    from warpSPH.runner import run
    from warpSPH.cases.compressible.hydrostatic import hydrostaticCase
    from warpSPH.cases.compressible.twoGasSod import twoGasSodCase, states
    from warpSPH.cases.compressible.shockBubble import shockBubbleCase
    from warpSPH.cases.compressible.sodND import sod2dCase
    from warpSPH.cases.compressible.common import compressibleDiagnostics
    from warpSPH.cases.common.plotting import Field, particlePlot
    from warpSPH.caseUtils.compressible.materialValidation import materialConservation, hostScalars
    from warpSPH.modules.crk import crkFactors
    from warpsph_remote.publisher import SimulationPublisher, SimulationStopped
    cases = {c.name: c for c in (hydrostaticCase, twoGasSodCase, shockBubbleCase)}
    if job['case'] == 'sod2d':
        def buildShock(ctx):
            system = sod2dCase.buildSystem(ctx)
            s = system.state
            # This regression intentionally treats the two Sod regions as
            # distinct physical materials; the standard Sod sampler has one.
            s.materials = s.emitters.clone()
            s.emitters = None  # One unlabeled source; material IDs carry the interface.
            ctx.scratch['prescribedDensity'] = s.densities.clone()
            ctx.scratch['initialSummedDensity'] = crkFactors(s, ctx.config.domain, ctx.config.kernel,
                system.adjacency, densityPolicy=ctx.schemeConfig.crkDensityPolicy)[1]
            ctx.scratch['materialInitialEnergy'] = (s.masses*(s.internalEnergies+.5*s.velocities.square().sum(-1))).sum().detach()
            ctx.scratch['materialInitialMasses'] = torch.stack([s.masses[s.materials == i].sum() for i in range(2)])
            return system
        def shockDiagnostics(ctx, system):
            return hostScalars(dict(compressibleDiagnostics(ctx, system), **materialConservation(ctx, system)))
        setup, update = particlePlot([Field('densities', 'Density'), Field('pressures', 'Pressure')], figsize=(10, 3))
        cases['sod2d'] = replace(sod2dCase, buildSystem=buildShock, diagnostics=shockDiagnostics,
                               setupPlot=setup, updatePlot=update)
    case = cases[job['case']]
    if job.get('retainInitial'):
        baseBuild = case.buildSystem
        def buildAndSave(ctx):
            system = baseBuild(ctx)
            root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(root/'initialState.npz', **{
                k: getattr(system.state, k).detach().cpu().numpy() for k in
                ('positions', 'velocities', 'densities', 'pressures', 'internalEnergies',
                 'materials', 'masses', 'supports', 'UIDs')})
            return system
        case = replace(case, buildSystem=buildAndSave)
    publisher = SimulationPublisher('Material check ' + job['name'], measurements=True, diagnostic_interval=.02)
    print(f'Remote run registered: {publisher.id} ({publisher.name})', flush=True)
    try:
        result = run(publisher.bind_case(case), scheme='CRKSPH',
            nx=job['nx'], tLimit=job['time'], nSteps=nsteps if nsteps is not None else job.get('steps'), params=job['params'],
            schemeOverrides=job['solver'], cflFactor=job.get('cflFactor', .15), execution='eager',
            plot=True, video=True, plotBackend='vispy', progress=True,
            plotInterval=25, store=True, storeMode='trajectory', storeInterval=50,
            holdPlot=False, show=False, exportRoot=str(root), stallProgress=1e-3,
            velocityAlarmPlotInterval=1, **job.get('simulation', {}))
        publisher.recording(result.ctx)
        publisher.finish(result.ctx, result.state, result.nSteps)
        publisher.update(step=result.nSteps, time=float(result.state.t), dt=float(result.ctx.config.dt),
                         particles=len(result.state.state.positions))
    except BaseException as error:
        publisher.close('stopped' if isinstance(error, SimulationStopped) else 'failed', str(error))
        if isinstance(error, SimulationStopped):
            return 130
        raise
    try:
        s = result.state.state
        final = dict(result.trajectory[-1])
        summary = dict(job, remoteRunID=publisher.id, steps=result.nSteps,
            time=float(result.state.t), wallSeconds=result.wallTime,
            meanStepMilliseconds=float(result.series('stepTime_ms').mean()),
            particles=len(s.positions), final=final, stopReason=result.stopReason,
            initialDensityRelativeL1=None, gates={})
        summary['densityPolicy'] = result.ctx.schemeConfig.crkDensityPolicy
        summary['evolution'] = result.ctx.schemeConfig.crkEvolution
        if job.get('retainInitial'):
            from warpSPH.configurations.crkSPH import crkSPHConfigToDict
            (root/'resolvedSolver.json').write_text(json.dumps(
                crkSPHConfigToDict(result.ctx.schemeConfig), indent=2, default=str)+'\n')
        gate = summary['gates']
        gate['completed'] = not (result.diverged or result.stopReason)
        gate['finitePositive'] = bool(torch.isfinite(s.internalEnergies).all() and
            torch.isfinite(s.densities).all() and (s.internalEnergies > 0).all() and (s.densities > 0).all())
        gate['energy'] = max(abs(result.series('relativeEnergyDrift'))) < 1e-4
        if job['case'] == 'hydrostatic':
            gate.update(pressure=max(result.series('pressureError')) < .01,
                        quiescent=max(result.series('maxPeculiarVelocity')) < .01,
                        shape=final['shapeDisplacement'] < .01)
            summary['reference'] = 'Exact stationary/uniformly translated contact; Frontiere et al. 2017 section 4.2 for equal-gamma square.'
            if summary['densityPolicy'] != 'legacy':
                gate['sharpDensity'] = max(result.series('densityRelativeL1')) < 1e-3
        else:
            rho = result.ctx.scratch['prescribedDensity']
            summed = result.ctx.scratch['initialSummedDensity']
            summary['initialDensityRelativeL1'] = float(((summed-rho).abs()/rho).mean().cpu())
            summary['initialDensityRelativeMax'] = float(((summed-rho).abs()/rho).max().cpu())
            gate['materialMass'] = final['materialMassError'] < 1e-6
        arrays = {name: getattr(s, name).detach().cpu().numpy() for name in
                  ('positions', 'velocities', 'densities', 'pressures', 'internalEnergies', 'materials', 'masses')}
        if job['case'] != 'hydrostatic':
            arrays['initialPrescribedDensity'] = result.ctx.scratch['prescribedDensity'].cpu().numpy()
            arrays['initialSummedDensity'] = result.ctx.scratch['initialSummedDensity'].cpu().numpy()
        if job['case'] == 'twoGasSod':
            from warpSPH.caseUtils.compressible.sod.twoGasReference import sodProfile
            exact = sodProfile(s.positions[:, 0], summary['time'], *states(result.ctx))
            arrays.update({'exact_'+k: v.cpu().numpy() for k, v in exact.items()})
            summary['reference'] = 'Exact left rarefaction/contact/right shock for independent gammas; Spheral SodSolutionGasGas ICs.'
            summary['waves'] = result.ctx.scratch['sodWaves']
            gate.update(densityL1=final['densityL1'] < .04, pressureL1=final['pressureL1'] < .04,
                        velocityL1=final['velocityL1'] < .04)
            if job['params'].get('pressureRight') == 1.:
                summary['reference'] = 'Exact stationary planar contact; tests equal-spacing versus equal-mass initialization.'
                gate.update(pressure=max(result.series('pressureError')) < .01,
                            quiescent=max(result.series('maxPeculiarVelocity')) < .01)
        if job['case'] == 'sod2d':
            summary['reference'] = 'Historical equal-spacing/mass-jump Sod 2D regression: survive 220 steps beyond the former step-164 viscosity failure, peak speed <2, positive energy and conservation.'
            gate['shockVelocity'] = max(result.series('maxVelocity')) < 2.
        if job['case'] == 'shockBubble':
            summary['reference'] = 'LLNL/spheral 116c71f ShockBubble-Variant-2d; uniform lattice with air2He1=air2He2=1, HeSeed=lattice. No quantitative shape oracle yet.'
            summary['referenceLimit'] = 'Completion/conservation gates only; not a literature-agreement pass.'
            if job['params'].get('driverPressure') == 1.:
                summary['reference'] = 'Exact stationary circular two-gas contact, no shock driver.'
                summary.pop('referenceLimit')
                gate.update(pressure=max(result.series('pressureError')) < .01,
                            quiescent=max(result.series('maxVelocity')) < .01)
        summary['gates'] = {k: bool(v) for k, v in gate.items()}
        summary['passed'] = all(gate.values())
        path = Path(result.ctx.exportPath)
        np.savez_compressed(path/'finalState.npz', **arrays)
        (path/'diagnosticHistory.json').write_text(json.dumps(result.trajectory, indent=2)+'\n')
        (path/'materialCaseValidation.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2), flush=True)
        publisher.close('completed' if summary['passed'] else 'failed', '' if summary['passed'] else 'qualification gate failed; see materialCaseValidation.json')
        return 0 if summary['passed'] else 1
    except BaseException as error:
        publisher.close('failed', str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=tuple(studyCases()), default='pilot')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run', help=argparse.SUPPRESS)
    parser.add_argument('--steps', type=int, help='Limit each child to this many steps for setup smoke checks.')
    args = parser.parse_args()
    jobs = studyCases()[args.phase]
    if args.run:
        return runCase(next(c for c in jobs if c['name'] == args.run), args.output, args.steps)
    if args.only:
        unknown = set(args.only) - {c['name'] for c in jobs}
        if unknown:
            parser.error(f'Unknown cases: {sorted(unknown)}')
        jobs = [c for c in jobs if c['name'] in args.only]
    root = args.output or Path('results/materialEOS')/('controls_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(phase=args.phase, status='running', cases=[dict(settings=j, status='pending') for j in jobs])
    def save():
        temp = root/'campaign.tmp'
        temp.write_text(json.dumps(manifest, indent=2)+'\n')
        temp.replace(root/'campaign.json')
    save()
    env = dict(os.environ, MPLCONFIGDIR='/tmp/warpsph-material-mpl')
    for entry in manifest['cases']:
        job = entry['settings']
        log = root/(job['name']+'.log')
        command = [sys.executable, __file__, '--phase', args.phase, '--run', job['name'], '--output', str(root/job['name'])]
        if args.steps is not None:
            command += ['--steps', str(args.steps)]
        entry.update(status='running', log=str(log), command=command, startedAt=time.time())
        save()
        print(f'START Material check {job["name"]}; log {log}', flush=True)
        with log.open('w') as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
            try:
                while child.poll() is None:
                    time.sleep(5)
                    with log.open('rb') as recent:
                        recent.seek(max(0, log.stat().st_size-6000))
                        rows = recent.read().decode(errors='replace').replace('\r', '\n').splitlines()
                    updates = [r for r in rows if '| t=' in r or 'Remote run registered:' in r or 'Traceback' in r]
                    if updates:
                        last = updates[-1]
                        fields = re.findall(r'(?:^|\|\s*)((?:t|pressureError|maxPeculiarVelocity|densityL1|pressureL1|velocityL1|bubbleWidth|relativeEnergyDrift)=[^ |]+)', last)
                        print(job['name']+': '+(' | '.join(fields) if fields else last[:300]), flush=True)
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                entry.update(status='interrupted', finishedAt=time.time())
                manifest['status'] = 'interrupted'
                save()
                raise
        entry.update(returncode=code, finishedAt=time.time(),
                     status='completed' if code == 0 else 'stopped' if code == 130 else 'failed')
        save()
        print(f'END {job["name"]}: {entry["status"]}', flush=True)
        if code:
            manifest['status'] = entry['status']
            save()
            return code
    manifest['status'] = 'completed'
    save()
    print('Campaign complete: '+str(root/'campaign.json'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
