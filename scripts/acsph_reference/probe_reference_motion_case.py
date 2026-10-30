#!/usr/bin/env python3
"""Non-evolving, same-input motion-guard controls from the captured failure.

The original guard is bypassed ONLY in these fixed-solve diagnostic copies,
so rejected trial fields can be compared. No physical step is accepted, no
shifting/finalization occurs, and no production/frozen source file is edited.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime',type=Path,required=True)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--factors',type=float,nargs='+',default=[2.,3.,4.])
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    runtime=args.runtime.resolve();sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts/acsph_reference')]
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch
    from diagnostic_state import map_tensors
    from warpSPH.modules.artificialCompressible import dyngpusphReference as ref
    from warpSPH.modules.artificialCompressible.dualTime import positionCoordinates
    entry=torch.load(args.capture/'entry.pt',map_location='cpu',weights_only=False)
    expected=torch.load(args.capture/'trial.pt',map_location='cpu',weights_only=False)
    report={'scope':'Fixed solve only; trial fields never accepted; no physical trajectory/video',
        'factors':{},'stageCertificate':'R +2max|stage position -entry position| < actual candidate cutoff is sufficient for all physical support pairs.'}
    original_iteration=inspect.getsource(ref.iteration)
    original_solve=inspect.getsource(ref.solve)
    captured=None
    for factor in args.factors:
        label='candidate-'+str(factor);directory=args.out/label;directory.mkdir()
        payload=map_tensors(entry,device='cuda:0');pr=payload['problem'];R=float(pr.state.supports[0])
        physical=pr.state.kinds!=2
        old=pr.adjacency
        bounce_cutoff=float(old.querySupports.min()) if old.querySupports is not None else R*float(pr.config.verletScale)
        bounce_entry_motion=float((pr.state.positions-old.queryPositions)[physical].norm(dim=-1).max()) if old.queryPositions is not None else 0.
        bounce_margin=bounce_cutoff-bounce_entry_motion-1.155*float(pr.config.dx)
        stage_rows=[];trial={};start=time.monotonic()
        def trace(stages,accepted,cells,m,adjacency,problem):
            cutoff=float(adjacency.querySupports.min())
            physical=problem.state.kinds!=2
            for name,x in list(enumerate(stages))+[('accepted',accepted)]:
                displacement=(x-cells)[physical]
                maximum=float(displacement.norm(dim=-1).max())
                box=float((displacement.max(0).values-displacement.min(0).values).norm())
                stage_rows.append({'iteration':m+1,'stage':name,'maximumMotion':maximum,
                    'relativeMotionBoundingBoxDiameter':box,'candidateCutoff':cutoff,
                    'triangleCertificate':R+2*maximum<cutoff,'relativeBoxCertificate':R+box<cutoff,
                    'coneStencilCertificate':maximum+.25*R<R,
                    'bounceCertificate':pr.bounce is None or maximum<bounce_margin})
        def capture(loc):
            for name in ('coordinates','v','p','x','lam','norm','edges'):
                trial[name]=loc[name].detach().cpu().clone()
            trial['diagnostics']=loc['scheme']._dyngpusphSolveDiagnostics
        namespace=dict(ref.solve.__globals__);namespace.update(_trace_motion=trace,_capture_trial=capture)
        src=original_iteration.replace('    for s in range(stages):\n','    stage_positions=[]\n    for s in range(stages):\n',1)
        src=src.replace('        coords=positionCoordinates(problem,xs)\n','        coords=positionCoordinates(problem,xs)\n        stage_positions.append(coords)\n',1)
        old='    return xN,vN,pN,lam,norm,edges,switch,metric,slopes[0][-1],slopes[2][-1]\n'
        assert src.count(old)==1
        src=src.replace(old,'    return (xN,vN,pN,lam,norm,edges,switch,metric,slopes[0][-1],slopes[2][-1],*stage_positions)\n')
        exec(compile(src,str(directory/'iteration.py'),'exec'),namespace)
        (directory/'iteration.py').write_text(src)
        src=original_solve.replace('search.supports=2*search.supports',f'search.supports={factor!r}*search.supports')
        old='        x,v,p,lam,norm,edges,switch,metric,last_position,last_pressure=out\n';assert src.count(old)==1
        src=src.replace(old,old.replace('=out\n','=out[:10]\n')+
            '        _trace_motion(out[10:],positionCoordinates(problem,x),cells,m,adjacency,problem)\n')
        old='    if float(displacement)>.5*float(problem.state.supports.min()):\n';assert src.count(old)==1
        src=src.replace(old,'    _capture_trial(locals())\n    if False: # fixed-solve diagnostic only; no physical acceptance\n')
        exec(compile(src,str(directory/'solve.py'),'exec'),namespace);(directory/'solve.py').write_text(src)
        result=namespace['solve'](pr,payload['x0'],payload['v0'],payload['p0'])
        torch.cuda.synchronize()
        d={'seconds':time.monotonic()-start,'iterations':result.iterations,
            'maximumMotionAllStages':max(row['maximumMotion'] for row in stage_rows),
            'maximumMotionRKOnly':max(row['maximumMotion'] for row in stage_rows if row['stage']!='accepted'),
            'maximumMotionAcceptedIterates':max(row['maximumMotion'] for row in stage_rows if row['stage']=='accepted'),
            'triangleCertificateAllStages':all(row['triangleCertificate'] for row in stage_rows),
            'triangleCertificateRKOnly':all(row['triangleCertificate'] for row in stage_rows if row['stage']!='accepted'),
            'relativeBoxCertificateAllStages':all(row['relativeBoxCertificate'] for row in stage_rows),
            'coneStencilCertificateAllStages':all(row['coneStencilCertificate'] for row in stage_rows),
            'bounceCertificateAllStages':all(row['bounceCertificate'] for row in stage_rows),
            'candidateCutoff':stage_rows[0]['candidateCutoff'],
            'finite':all(bool(torch.isfinite(trial[name]).all()) for name in ('coordinates','v','p'))}
        if factor==2.:
            d['originalTrialBitwise']={name:torch.equal(trial[name],expected[name]) for name in ('coordinates','v','p','x','lam','norm','edges')}
            assert all(d['originalTrialBitwise'].values()),d
            captured=trial
        if captured is not None:
            d['differenceFromCandidate2']={name:float((trial[name]-captured[name]).abs().max()) for name in ('coordinates','v','p')}
        (directory/'stage-motion.json').write_text(json.dumps(stage_rows,indent=2)+'\n')
        torch.save(trial,directory/'trial.pt');report['factors'][str(factor)]=d
        (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MOTION CONTROL',factor,json.dumps(d),flush=True)
        del payload,result,trial;torch.cuda.empty_cache()


if __name__=='__main__':main()
