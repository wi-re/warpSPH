#!/usr/bin/env python3
"""Fresh-process, same-input ACSPH diagnostic controls; no physical advance.

Promoting an FP32 entry to FP64 changes arithmetic, not its input accuracy.
Original tensor values, neighbor order, BDF and contact topology are retained.
The FP32 graph replay must match the captured original before using the bundle
as evidence. Tiny same-state controls omit video; the physical prefix records it.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime',type=Path,required=True)
    ap.add_argument('--entry',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--precision',choices=['float32','float64'],required=True)
    ap.add_argument('--controls',nargs='+',choices=['graph','eager','off','smooth','spacing-normals'],default=['graph','eager','off','smooth'])
    ap.add_argument('--original-result',type=Path)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    runtime=args.runtime.resolve()
    sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts/acsph_reference')]
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=args.precision)
    import torch
    from diagnostic_state import map_tensors, equation_residuals
    from warpSPH.modules.artificialCompressible import dualTime
    from audit_local import install
    torch.set_num_threads(4)
    install(out/'operators',observe=False,wall_diffusion='source',acoustic_damper='source',rk_treatment='source',wall_velocity='source')
    dtype=getattr(torch,args.precision)
    payload=torch.load(args.entry,map_location='cpu',weights_only=False)
    payload=map_tensors(payload,device='cuda:0',dtype=dtype)
    problem=payload['problem']
    inputs=[payload[k] for k in ('x0','v0','p0')]
    torch.save({'UIDs':problem.state.UIDs.cpu(),'numNeighbors':problem.adjacency.numNeighbors.cpu()},out/'identity.pt')
    original_iterate=dualTime.iterate
    reports={}
    start=time.monotonic()
    for label in args.controls:
        scheme=copy.copy(problem.schemeConfig);scheme.acParams=copy.copy(scheme.acParams)
        scheme.cudaGraph=True
        scheme._acsphGraphValidated=False
        scheme._acsphGraphDisabled=None if label=='graph' else 'per-particle diagnostic observation'
        bounce=problem.bounce if label in ('graph','eager') else None if label=='off' else replace(problem.bounce,ramp=.5)
        if label=='spacing-normals':
            # Static rectangular fixture only. Isolate the ghost-spacing unit
            # defect in bounce normals; retain original wall closures/topology.
            from fixture_geometry import lobovsky_tank_sdf
            from warpSPH.rigidBody.ghostParticles import _gridSnapGhostOffsets
            wall=problem.state.kinds==1
            offsets=_gridSnapGhostOffsets(problem.state.positions[wall],[lobovsky_tank_sdf],
                float(problem.config.dx),float(problem.state.supports.mean()))
            normal=problem.bounce.normals.clone()
            normal[wall]=torch.nn.functional.normalize(-offsets,dim=-1)
            bounce=replace(problem.bounce,normals=normal)
        trial=replace(problem,schemeConfig=scheme,bounce=bounce)
        count=[0]
        def observe(pr,*state):
            result=original_iterate(pr,*state)
            q,v,p,eps=result
            x=dualTime.positionCoordinates(pr,q)
            rate=dualTime.bdfSource(pr,q,v)[0]
            bounced=dualTime.bounceVelocities(pr.bounce,x,v)
            residual=(bounced-rate).index_select(0,pr.fluidIndex)
            e2=residual.square().sum(-1);values,indices=e2.topk(10);top=pr.fluidIndex[indices]
            row=dict(iteration=count[0],epsilonV=float(eps),top1_fraction=float(values[0]/e2.sum().clamp_min(1e-300)),
                top10_fraction=float(values.sum()/e2.sum().clamp_min(1e-300)),top_uids=pr.state.UIDs[top].cpu().tolist(),
                top_positions=x[top].cpu().tolist(),top_residuals=(bounced-rate)[top].cpu().tolist(),
                top_velocities=v[top].cpu().tolist(),top_bdf_rates=rate[top].cpu().tolist(),
                top_pressures=p[top].cpu().tolist())
            with (out/(label+'.jsonl')).open('a') as log:log.write(json.dumps(row)+'\n')
            if count[0]%25==0:print('CONTROL',args.precision,label,count[0],row['epsilonV'],row['top1_fraction'],flush=True)
            count[0]+=1
            return result
        dualTime.iterate=original_iterate if label=='graph' else observe
        tick=time.monotonic()
        result=dualTime.solveDualTime(trial,*[x.clone() for x in inputs])
        torch.cuda.synchronize()
        record=dict(iterations=result.iterations,epsilonV=result.epsilonV,wallSeconds=time.monotonic()-tick)
        fields={name:getattr(result,name).detach().cpu().clone() for name in ('positions','velocities','pressures','positionStep','positionLow')}
        if label=='graph' and args.original_result:
            expected=torch.load(args.original_result,map_location='cpu',weights_only=False)
            record['originalResultDifferences']={name:float((value.double()-expected[name].double()).abs().max()) for name,value in fields.items()}
            if args.precision=='float32':
                for name,value in fields.items():torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
                record['originalResultBitwiseEqual']=True
        if label=='graph':baseline=fields
        elif 'graph' in args.controls:
            record['graphResultDifferences']={name:float((value.double()-baseline[name].double()).abs().max()) for name,value in fields.items()}
        torch.save(fields,out/(label+'-result.pt'))
        record['equationResiduals']=equation_residuals(trial,inputs[0],result)
        reports[label]=record
        (out/'summary.json').write_text(json.dumps(dict(precision=args.precision,controls=reports,wallSeconds=time.monotonic()-start),indent=2)+'\n')
        print('RESULT',args.precision,label,json.dumps(record),flush=True)
    dualTime.iterate=original_iterate


if __name__=='__main__':main()
