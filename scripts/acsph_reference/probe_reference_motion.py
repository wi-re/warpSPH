#!/usr/bin/env python3
"""Reproduce the frozen reference motion guard and capture its exact inputs.

No numerical expression or stopping condition is changed. A read-only callback
immediately before the original guard records the trial state. Physical prefix
diagnostics must reproduce the failed baseline exactly. Complete BDF/shift
history is checkpointed before the failing physical step.
"""
import argparse
from dataclasses import fields,replace
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','reference','baseline','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--step',type=int,default=6489)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=args.out.resolve();runtime=args.runtime.resolve();capture=root/'capture';capture.mkdir()
    shutil.copy2(__file__,root/'probe-source.py')
    sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts/acsph_reference')]
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch
    import warpSPH.runner as runner
    from diagnostic_state import map_tensors,solve_payload
    from warpSPH.modules.artificialCompressible import dyngpusphReference as reference
    from warpSPH.modules.artificialCompressible import dualTime
    expected=[json.loads(line) for line in (args.baseline/'progress.jsonl').read_text().splitlines()]
    assert len(expected)==args.step,'Baseline must end immediately before failing step'
    report={'targetStep':args.step,'runtime':str(runtime),'baseline':str(args.baseline.resolve()),
        'sourceScope':'Unchanged frozen numerical solve plus read-only pre-guard callback',
        'parityRows':0,'guardReproduced':False}
    def save():
        p=root/'audit.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
    save();active={'step':0}
    original=reference.solve;source=inspect.getsource(original)
    marker='    displacement=(coordinates-cells).norm(dim=-1).max()\n'
    assert source.count(marker)==1
    def observe(loc):
        if active['step']!=args.step:return
        problem=loc['problem'];x=loc['coordinates'];delta=(x-loc['cells']).norm(dim=-1)
        values,order=delta.topk(20);st=problem.state
        record={'step':active['step'],'iterations':loc['m']+1,'dt':problem.dtau*problem.schemeConfig.acParams.dtOverDtau,
            'bound':.5*float(st.supports.min()),'maximumDisplacement':float(values[0]),
            'topUIDs':st.UIDs[order].cpu().tolist(),'topKinds':st.kinds[order].cpu().tolist(),
            'topDisplacements':values.cpu().tolist(),'topStart':loc['cells'][order].cpu().tolist(),
            'topEnd':x[order].cpu().tolist(),'topVelocities':loc['v'][order].cpu().tolist(),
            'topPressures':loc['p'][order].cpu().tolist(),
            'maximumSpeed':float(loc['v'].norm(dim=-1).max()),
            'pressureMin':float(loc['p'].min()),'pressureMax':float(loc['p'].max()),
            'finite':all(bool(torch.isfinite(loc[key]).all()) for key in ('coordinates','v','p')),
            'diagnostics':problem.schemeConfig._dyngpusphSolveDiagnostics}
        (capture/'trial-summary.json').write_text(json.dumps(record,indent=2)+'\n')
        names=('x','v','p','lam','norm','edges','switch','cells','coordinates','last_position','last_pressure','adjacency')
        torch.save(map_tensors({key:loc[key] for key in names},device='cpu'),capture/'trial.pt')
        report['trial']=record;save();print('PRE-GUARD TRIAL',json.dumps(record),flush=True)
    # Execute an observational copy of precisely the qualified solve source.
    namespace=dict(original.__globals__);namespace['_motion_observer']=observe
    instrumented=source.replace(marker,marker+'    _motion_observer(locals())\n')
    (root/'observed-solve.py').write_text(instrumented)
    report['originalSolveSha256']=hashlib.sha256(source.encode()).hexdigest()
    report['observedSolveSha256']=hashlib.sha256(instrumented.encode()).hexdigest()
    exec(compile(instrumented,str(root/'observed-solve.py'),'exec'),namespace)
    observed=namespace['solve']
    def solve(problem,x0,v0,p0,verbose=False):
        active['step']+=1
        if active['step']==args.step:
            payload=solve_payload(problem,x0,v0,p0)
            torch.save(map_tensors(payload,device='cpu'),capture/'entry.pt')
            print('CAPTURED COMPLETE SOLVE ENTRY',active['step'],flush=True)
        return observed(problem,x0,v0,p0,verbose)
    reference.solve=solve
    original_run=runner.run
    system_fields=('state','adjacency','domain','t','positionsPrev','velocitiesPrev','positionsPrev2',
        'velocitiesPrev2','dtPrev','positionStep','positionLow')
    def run(case,**kwargs):
        old_post=case.postStep;old_diagnostics=case.diagnostics
        def post(ctx,system,step):
            old_post(ctx,system,step)
            if step+1==args.step-1:
                bodies=[{f.name:getattr(body,f.name) for f in fields(body) if not callable(getattr(body,f.name))}
                    for body in ctx.schemeConfig.rigidBodies]
                payload={'schema':'reference-motion-1','step':step+1,
                    'system':{name:getattr(system,name) for name in system_fields},'configDomain':ctx.config.domain,
                    'bodies':bodies,'schemeDynamic':{key:value for key,value in vars(ctx.schemeConfig).items()
                        if key.startswith(('_dyngpusph','_acsph'))}}
                torch.save(map_tensors(payload,device='cpu'),capture/'checkpoint.pt')
                print('CAPTURED COMPLETE SYSTEM HISTORY',step+1,float(system.t),flush=True)
        def diagnostics(ctx,system):
            row=old_diagnostics(ctx,system);actual={'t':float(system.t),**row};i=report['parityRows']
            assert actual==expected[i],f'Baseline diagnostics mismatch at accepted step{i}'
            report['parityRows']+=1
            if i%100==0 or i==args.step-1:save();print('PARITY',i,float(system.t),flush=True)
            return row
        return original_run(replace(case,postStep=post,diagnostics=diagnostics),**kwargs)
    runner.run=run
    script=runtime/'scripts/acsph_reference/compare_lobovsky.py'
    sys.argv=[str(script),'--reference',str(args.reference.resolve()),'--out',str(root/'replay'),
        '--steps',str(args.step),'--precision','float32','--position-form','increment',
        '--solver-model','dyngpusph-static-2d','--wall-bounce-ramp','.5','--ghost-spacing','volume',
        '--long-run','--plot-interval','10']
    start=time.monotonic()
    try:
        runpy.run_path(str(script),run_name='__main__')
        raise AssertionError('Expected candidate-motion guard was not raised')
    except RuntimeError as error:
        report['exception']=str(error)
        if active['step']!=args.step or str(error)!='Reference solve exceeded the declared half-support candidate motion bound':
            report['unexpectedFailure']=True;save();raise
        assert report['parityRows']==len(expected) and (capture/'trial.pt').exists()
        report['guardReproduced']=True
    finally:
        report['wallSeconds']=time.monotonic()-start;save()
    # The runner threw before finalizing video. Render accepted snapshots as
    # an explicitly labelled partial trajectory; trial fields stay separate.
    code=subprocess.call([sys.executable,str(runtime/'scripts/acsph_reference/render.py'),str(root/'replay'),
        '--label','warpSPH FP32 exact failed-prefix replay; accepted states only'])
    report.update(renderReturncode=code,status='captured' if code==0 else 'captured; render failed');save()
    print('MOTION CAPTURE COMPLETE',json.dumps({k:report[k] for k in ('guardReproduced','parityRows','wallSeconds','status')}),flush=True)
    if code:raise SystemExit(code)


if __name__=='__main__':main()
