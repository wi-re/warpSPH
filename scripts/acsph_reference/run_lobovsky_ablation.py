#!/usr/bin/env python3
"""Run one frozen cumulative Lobovsky ablation, with original ICs and video."""
import argparse
from dataclasses import fields,replace
import json
from pathlib import Path
import runpy
import sys

from ablation_matrix import Options,select


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','reference','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--row')
    ap.add_argument('--michel-trial',choices=('baseline','transport','bdf','oracle'))
    ap.add_argument('--steps',type=int,default=7641)
    ap.add_argument('--eager',action='store_true')
    ap.add_argument('--daemon',action=argparse.BooleanOptionalAction,default=None,help='Native daemon publisher; default on for long runs')
    ap.add_argument('--plot-interval',type=int,default=10)
    args=ap.parse_args()
    if args.michel_trial:
        from michel_trials import select as trial_select
        row=trial_select(args.michel_trial)
        options=Options(static_geometry=True,accepted_fields=True)
    else:
        if args.row is None:ap.error('--row or --michel-trial is required')
        row=select(args.row);options=Options(**row['options'])
    runtime=args.runtime.resolve();out=args.out.resolve()
    sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts/acsph_reference')]
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch
    import warpSPH.runner as runner
    from ablation_controls import install
    from diagnostic_state import map_tensors,solve_payload
    from warpSPH.modules.artificialCompressible import dualTime as dual
    from warpSPH.modules.artificialCompressible import dyngpusphReference as ref
    from warpSPH.systems import artificialCompressible as systems
    if args.michel_trial:
        import importlib
        imports={name:str(Path(importlib.import_module(name).__file__).resolve())
            for name in ('warpSPH','warpSPHCore','warpSPHIntegrators','warpSPHPlotting')}
        assert all(Path(path).is_relative_to(runtime) for path in imports.values()),imports
    clock=json.loads((args.reference/'result.json').read_text())['steps']
    assert 0<args.steps<=len(clock)
    milestones={next(i for i,r in enumerate(clock,1) if r['t']>=t)-1 for t in (1.25,1.44)}
    active={'step':0};original_run=runner.run
    def checkpoint(ctx,system,step,archive=False):
        payload={'schema':'reference-motion-1','step':step,
            'system':{name:getattr(system,name) for name in ('state','adjacency','domain','t',
                'positionsPrev','velocitiesPrev','positionsPrev2','velocitiesPrev2','dtPrev','positionStep','positionLow')},
            'configDomain':ctx.config.domain,
            'bodies':[{f.name:getattr(b,f.name) for f in fields(b) if not callable(getattr(b,f.name))}
                for b in ctx.schemeConfig.rigidBodies],
            'schemeDynamic':{k:v for k,v in vars(ctx.schemeConfig).items() if k.startswith(('_acsph','_dyngpusph'))},
            'ablationRow':row}
        pending=out/'checkpoint.pending';torch.save(map_tensors(payload,device='cpu'),pending)
        pending.replace(out/'latest-checkpoint.pt')
        if archive:
            import shutil
            shutil.copy2(out/'latest-checkpoint.pt',out/f'checkpoint-{step:06d}.pt')
        print('ABLATION CHECKPOINT',step,float(system.t),flush=True)
    def trap(original):
        def solve(problem,x0,v0,p0,*a,**kw):
            payload=solve_payload(problem,x0,v0,p0)
            try:return original(problem,x0,v0,p0,*a,**kw)
            except Exception as error:
                capture=out/'failure';capture.mkdir(exist_ok=True)
                torch.save(map_tensors(payload,device='cpu'),capture/'entry.pt')
                tb=error.__traceback__;loc=None
                while tb is not None:
                    candidate=tb.tb_frame.f_locals
                    if all(k in candidate for k in ('x','v','p','problem')):loc=candidate
                    tb=tb.tb_next
                if loc is not None:
                    names=('x','v','p','lam','norm','edges','cells','adjacency','last_position','last_pressure')
                    trial={k:loc[k] for k in names if k in loc}
                    trial['coordinates']=dual.positionCoordinates(loc['problem'],loc['x'])
                    torch.save(map_tensors(trial,device='cpu'),capture/'trial.pt')
                (capture/'receipt.json').write_text(json.dumps(dict(step=active['step']+1,
                    error=str(error),row=row,trialCaptured=loc is not None),indent=2)+'\n')
                raise
        return solve
    def run(case,**kwargs):
        configure=case.configureScheme;initialize=case.initialConditions;post=case.postStep
        def configured(ctx):
            configure(ctx)
            if args.michel_trial:
                (out/'runtime-imports.json').write_text(json.dumps(imports,indent=2)+'\n')
                from michel_trials import install as install_trial,trial_parameters
                ctx.schemeConfig.acParams.bdfShiftCorrection=row['options']['bdf_shift']
                receipt=install_trial(row)
                receipt['actualAcParams']=trial_parameters(ctx.schemeConfig)
            else:receipt=install(options)
            # Install before initialize's per-step timing wrapper. Generic
            # audit hooks still attach to the unchanged generic iteration.
            if options.source_operators:ref.solve=trap(ref.solve)
            else:
                import importlib
                step=importlib.import_module('warpSPH.schemes.incompressible.artificialCompressible')
                wrapped=trap(step.solveDualTime);step.solveDualTime=wrapped;dual.solveDualTime=wrapped
            (out/'ablation-row.json').write_text(json.dumps(dict(row=row,implementation=receipt,
                endpointSteps=args.steps,endpointTime=clock[args.steps-1]['t']),indent=2)+'\n')
        def initialized(ctx,system):
            initialize(ctx,system)
            ctx.scratch['ablationNonfluid']=system.state.kinds!=0
            ctx.scratch['ablationGeometry']=system.state.positions[system.state.kinds!=0].clone()
            ctx.scratch['ablationOffsets']=system.state.ghostOffsets.clone()
            print('ABLATION START',json.dumps(row),flush=True)
        def after(ctx,system,step):
            post(ctx,system,step);active['step']=step+1
            if options.static_geometry:
                assert torch.equal(system.state.positions[ctx.scratch['ablationNonfluid']],ctx.scratch['ablationGeometry'])
                assert torch.equal(system.state.ghostOffsets,ctx.scratch['ablationOffsets'])
            if args.steps>=100 and (step+1 in milestones or step+1==args.steps or (step+1>=6000 and (step+1)%200==0)):
                checkpoint(ctx,system,step+1,archive=step+1 in milestones)
        prepared=replace(case,configureScheme=configured,initialConditions=initialized,postStep=after)
        from live_run import daemon_enabled,run_published
        if daemon_enabled(args.steps,args.daemon):
            import warpsph_remote
            assert Path(warpsph_remote.__file__).resolve().is_relative_to(runtime)
            return run_published(original_run,prepared,name='ACSPH Lobovsky '+row['id'],out=out,**kwargs)
        return original_run(prepared,**kwargs)
    runner.run=run
    script=runtime/'scripts/acsph_reference/compare_lobovsky.py'
    sys.argv=[str(script),'--reference',str(args.reference.resolve()),'--out',str(out),
        '--steps',str(args.steps),'--precision','float32','--position-form','increment','--fused-residual',
        '--solver-model','dyngpusph-static-2d' if options.source_operators else 'warpsph',
        '--wall-diffusion','source','--acoustic-damper','source','--rk-treatment','source',
        '--wall-velocity','source-consistent','--wall-bounce-ramp','.5','--ghost-spacing','volume',
        '--shifting','dyngpusph-reference' if options.source_shift else 'warpsph-michel',
        '--history','source','--coefficients','mapped','--long-run','--plot-interval',str(args.plot_interval)]
    if args.eager:sys.argv.append('--eager')
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
