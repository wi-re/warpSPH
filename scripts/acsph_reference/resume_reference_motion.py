#!/usr/bin/env python3
"""Complete-history reference replay after a candidate-motion checkpoint.

--expect-guard qualifies restoration by reproducing the original rejected
trial bitwise. Normal use continues the chosen frozen runtime, with video and
an explicit physical-step endpoint. Never uses a visual snapshot as a restart.
"""
import argparse
from dataclasses import fields,replace
import inspect
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','reference','capture','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--end-step',type=int,required=True)
    ap.add_argument('--expect-guard',action='store_true')
    ap.add_argument('--qualify-trial',action='store_true',help='Require the first repaired solve to equal the original rejected trial bitwise')
    ap.add_argument('--checkpoint',type=Path,help='Use a rolling complete checkpoint instead of capture/checkpoint.pt')
    ap.add_argument('--capture-guard-step',type=int,help='Capture and require a certified-motion stop at this absolute step')
    ap.add_argument('--baseline',type=Path,help='Accepted progress.jsonl directory for exact guarded replay verification')
    args=ap.parse_args();runtime=args.runtime.resolve()
    sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts/acsph_reference')]
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch,numpy as np,struct
    import warpSPH.runner as runner
    from diagnostic_state import map_tensors,solve_payload
    from warpSPH.modules.artificialCompressible import dyngpusphReference as ref
    checkpoint_path=args.checkpoint or args.capture/'checkpoint.pt'
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert saved['schema']=='reference-motion-1'
    start=saved['step'];assert args.end_step>start
    expected=torch.load(args.capture/'trial.pt',map_location='cpu',weights_only=False) if args.expect_guard or args.qualify_trial else None
    times=json.loads((args.reference/'result.json').read_text())['steps']
    out=args.out.resolve();audit={'startStep':start,'endStep':args.end_step,'expectedGuard':args.expect_guard,
        'runtime':str(runtime),'sourceCheckpoint':str(checkpoint_path.resolve())}
    def save():
        if out.exists():(out/'restart-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if args.capture_guard_step:
        assert args.capture_guard_step==args.end_step and args.baseline
        assert not args.expect_guard and not args.qualify_trial
        counter={'step':start};original=ref.solve;source=inspect.getsource(original)
        marker='        if not math.isfinite(values[2]) or maximum_motion>=budget.limit:\n'
        assert source.count(marker)==1
        capture_dir=out/'guard-capture'
        def capture(loc):
            assert counter['step']==args.capture_guard_step
            from warpSPH.modules.artificialCompressible.dualTime import positionCoordinates
            coordinates=positionCoordinates(loc['problem'],loc['x'])
            names=('x','v','p','lam','norm','edges','switch','cells','last_position','last_pressure','adjacency')
            trial={name:loc[name] for name in names};trial['coordinates']=coordinates
            torch.save(map_tensors(trial,device='cpu'),capture_dir/'trial.pt')
            motion=loc['scheme']._dyngpusphMotionDiagnostics
            audit['capturedGuardStep']=counter['step'];audit['motionCertificate']=motion
            record=dict(step=counter['step'],iterations=loc['m']+1,motion=motion,
                finite=all(bool(torch.isfinite(trial[k]).all()) for k in ('coordinates','v','p')),
                maximumSpeed=float(loc['v'].norm(dim=-1).max()),
                maximumEndpointMotion=float((coordinates-loc['cells']).norm(dim=-1).max()),
                pressureMin=float(loc['p'].min()),pressureMax=float(loc['p'].max()))
            (capture_dir/'trial-summary.json').write_text(json.dumps(record,indent=2)+'\n')
            save();print('CAPTURED CERTIFIED GUARD',json.dumps(record),flush=True)
        namespace=dict(original.__globals__);namespace['_capture_certified_guard']=capture
        source=source.replace(marker,marker+'            _capture_certified_guard(locals())\n')
        exec(compile(source,'observed-certified-motion-solve','exec'),namespace)
        observed=namespace['solve']
        def observe(problem,x0,v0,p0,verbose=False):
            counter['step']+=1
            if counter['step']==args.capture_guard_step:
                capture_dir.mkdir(parents=True,exist_ok=False)
                torch.save(map_tensors(solve_payload(problem,x0,v0,p0),device='cpu'),capture_dir/'entry.pt')
            return observed(problem,x0,v0,p0,verbose)
        ref.solve=observe
        expected_rows=[json.loads(s) for s in (args.baseline/'progress.jsonl').read_text().splitlines()]
        expected_rows=[r for r in expected_rows if r['t']>=float(saved['system']['t'])]
        assert len(expected_rows)==args.end_step-start
        audit['parityRows']=0
    elif args.expect_guard:
        assert args.end_step==start+1
        original=ref.solve;source=inspect.getsource(original)
        marker='    displacement=(coordinates-cells).norm(dim=-1).max()\n';assert source.count(marker)==1
        def compare(loc):
            audit['originalRejectedTrialBitwise']={name:torch.equal(loc[name].detach().cpu(),expected[name])
                for name in ('coordinates','v','p','x','lam','norm','edges')}
            save();assert all(audit['originalRejectedTrialBitwise'].values()),audit
            print('RESTART TRIAL BITWISE',json.dumps(audit['originalRejectedTrialBitwise']),flush=True)
        namespace=dict(original.__globals__);namespace['_compare_restart']=compare
        source=source.replace(marker,marker+'    _compare_restart(locals())\n')
        exec(compile(source,'qualified-restart-solve','exec'),namespace);ref.solve=namespace['solve']
    elif args.qualify_trial:
        original=ref.solve
        def qualified(*a,**kw):
            result=original(*a,**kw)
            if 'originalRejectedTrialBitwise' not in audit:
                rf=result.referenceFields
                actual=dict(coordinates=result.positions,v=result.velocities,p=result.pressures,
                    x=result.positionStep,lam=rf['lambdas'],norm=rf['normals'],edges=rf['edges'])
                audit['originalRejectedTrialBitwise']={name:torch.equal(value.detach().cpu(),expected[name])
                    for name,value in actual.items()}
                audit['motionCertificate']=a[0].schemeConfig._dyngpusphMotionDiagnostics
                save();assert all(audit['originalRejectedTrialBitwise'].values()),audit
                print('REPAIRED TRIAL BITWISE',json.dumps(audit['originalRejectedTrialBitwise']),flush=True)
            return result
        ref.solve=qualified
    original_run=runner.run
    def physical_snapshot(system,index):
        st=system.state;keep=st.kinds!=2
        a={name:getattr(st,name)[keep].detach().cpu().numpy() for name in ('positions','velocities','pressures','densities','masses','kinds')}
        n=len(a['positions'])
        with (out/'snapshots'/f'{index:06d}.bin').open('wb') as f:
            f.write(struct.pack('<id',n,float(system.t)))
            for key in ('positions','pressures','densities','velocities'):f.write(a[key].astype('<f8').tobytes())
            f.write(np.where(a['kinds']==0,1,3).astype('<i4').tobytes())
            for x in (np.zeros(n),a['masses']/a['densities'],np.zeros(n)):f.write(x.astype('<f8').tobytes())
    def checkpoint(ctx,system,index):
        payload={'schema':'reference-motion-1','step':index,
            'system':{name:getattr(system,name) for name in saved['system']},'configDomain':ctx.config.domain,
            'bodies':[{f.name:getattr(body,f.name) for f in fields(body) if not callable(getattr(body,f.name))} for body in ctx.schemeConfig.rigidBodies],
            'schemeDynamic':{key:value for key,value in vars(ctx.schemeConfig).items() if key.startswith(('_dyngpusph','_acsph'))}}
        pending=out/'latest-checkpoint.pending'
        torch.save(map_tensors(payload,device='cpu'),pending)
        pending.replace(out/'latest-checkpoint.pt')
        print('ROLLING CHECKPOINT',index,float(system.t),flush=True)
    def run(case,**kwargs):
        initial=case.initialConditions;post=case.postStep;diagnostics=case.diagnostics
        def initialize(ctx,system):
            initial(ctx,system);data=map_tensors(saved,device=ctx.device)
            for key,value in data['system'].items():setattr(system,key,value)
            ctx.config.domain=data['configDomain']
            for body,record in zip(ctx.schemeConfig.rigidBodies,data['bodies']):
                for key,value in record.items():setattr(body,key,value)
            assert len(ctx.schemeConfig.rigidBodies)==len(data['bodies'])
            for key,value in data['schemeDynamic'].items():setattr(ctx.schemeConfig,key,value)
            # The patched map, if any, qualifies its own graph again.
            ctx.schemeConfig._dyngpusphGraphValidated=False
            ctx.scratch['completed']=start;ctx.config.dt=times[start]['dt']
            (out/'snapshots/000000.bin').rename(out/'construction-initial.bin')
            physical_snapshot(system,start)
            audit['restoredSystemTensorEquality']={key:torch.equal(getattr(system,key).cpu(),value)
                for key,value in saved['system'].items() if isinstance(value,torch.Tensor)}
            audit['restoredParticleTensorEquality']={f.name:torch.equal(getattr(system.state,f.name).cpu(),getattr(saved['system']['state'],f.name))
                for f in fields(system.state) if isinstance(getattr(system.state,f.name),torch.Tensor)}
            assert all(audit['restoredSystemTensorEquality'].values()) and all(audit['restoredParticleTensorEquality'].values())
            audit['restartTime']=float(system.t);save();print('RESTORED COMPLETE HISTORY',start,float(system.t),flush=True)
        def after(ctx,system,step):
            post(ctx,system,step);absolute=step+1
            before_guard=args.capture_guard_step is not None and absolute==args.capture_guard_step-1
            if absolute%100==0 or absolute==args.end_step or before_guard:
                checkpoint(ctx,system,absolute)
            if absolute==args.end_step:physical_snapshot(system,absolute)
        def checked(ctx,system):
            row=diagnostics(ctx,system);actual={'t':float(system.t),**row};i=audit['parityRows']
            expected_row=expected_rows[i]
            keys=actual.keys() if i else actual.keys()-{'pseudoIterations','epsilonV','epsilonP','exitReason'}
            assert all(actual[k]==expected_row[k] for k in keys),f'Restart replay differs at row{i}'
            audit['parityRows']+=1;save()
            return row
        kwargs.update(nSteps=args.end_step-start,tLimit=times[args.end_step-1]['t'],dt=times[start]['dt'],resumeStepOffset=start)
        result=original_run(replace(case,initialConditions=initialize,postStep=after,
            diagnostics=checked if args.capture_guard_step else diagnostics),**kwargs)
        result.ctx.scratch['last_snapshot_step']=result.nSteps
        return result
    runner.run=run
    script=runtime/'scripts/acsph_reference/compare_lobovsky.py'
    sys.argv=[str(script),'--reference',str(args.reference.resolve()),'--out',str(out),
        '--steps',str(args.end_step),'--precision','float32','--position-form','increment',
        '--solver-model','dyngpusph-static-2d','--wall-bounce-ramp','.5','--ghost-spacing','volume',
        '--long-run','--plot-interval','10']
    tick=time.monotonic()
    try:
        runpy.run_path(str(script),run_name='__main__')
        assert not args.expect_guard,'Expected original guard was not raised'
        audit['status']='completed'
    except RuntimeError as error:
        if args.capture_guard_step:
            if not str(error).startswith('Reference solve exceeded certified candidate motion:'):raise
            assert audit.get('capturedGuardStep')==args.capture_guard_step and audit['parityRows']==len(expected_rows)
            audit['status']='certified guard reproduced; trial captured'
            audit['exception']=str(error)
        else:
            if not args.expect_guard or str(error)!='Reference solve exceeded the declared half-support candidate motion bound':raise
            assert all(audit.get('originalRejectedTrialBitwise',{}).values()) and audit.get('originalRejectedTrialBitwise')
            audit['status']='old guard reproduced; restart qualified'
    finally:
        audit['wallSeconds']=time.monotonic()-tick;save()
    print('RESTART COMPLETE',json.dumps(audit),flush=True)


if __name__=='__main__':main()
