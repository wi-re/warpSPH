"""Native scheme lifecycle parity when a rank has no chunks, then gains work.

Tiny repeated flat/owned-row A/Bs omit video; progress and per-phase timing stay
visible. Run under torchrun on Gloo or a qualified shared-GPU NCCL runtime.
"""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import argparse,contextlib,json,os,sys,time

import torch
import warp as wp
import torch.distributed as dist

from warpSPH.runner import RunConfiguration
from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
from warpSPH.distributedRun import _sourcePins,_runtime
from warpSPHCore import SupportScheme
from warpSPHCore.radiusSearch import RowChunkPolicy,buildChunkedVerletList
from warpSPHCore.distributed import RowChunkProcessGroup


def configuration(name,device,dim):
    if name in ('deltaSPH','sun2017DeltaSPH','iisph','dfsph','dfsphPredictor','dfsphShift'):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        if name in ('deltaSPH','sun2017DeltaSPH'):
            from test_chunkedDelta import configuration as native
            from warpSPH.cases.periodic.tgvWeaklyCompressible import tgvWeaklyCompressibleCase as base
        else:
            from test_chunkedProjection import activeConfiguration as native
            from warpSPH.cases.periodic.tgv import tgvCase as base
        cfg=native(name)
    else:
        from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base
        scheme,_,variant=name.partition(':')
        solver=dict(scheme=scheme)
        params=dict(accelerationRMS=.1,forcingInterval=.00025,seed=1234)
        if scheme in ('MFM','MFV'):solver.update(temporalScheme='ssprk',closure=variant or 'none')
        if variant and scheme=='PESPH':
            solver.update(variant=variant,dissipation=variant=='PressureEnergy')
            if variant=='PressureEntropy':params['thermalModel']='adiabatic'
        cfg=RunConfiguration('drivenTurbulence',simulation=dict(nx=8 if dim==2 else 6,dim=dim,
            maxDt=.0002,integrationScheme='rungeKutta4',storeMode='trajectory'),params=params,solver=solver)
    cfg.simulation.update(device=device,nSteps=4,plot=False,video=False,store=False,show=False,
        quiet=True,progress=True,execution='eager',cudaGraph=False,pipelineOutputs=False,
        stallProgress=1e-3,velocityAlarmPlotInterval=1)
    return cfg,base


def nativeState(result):
    return {k:v.detach().clone() for k,v in vars(result.state.state).items() if isinstance(v,torch.Tensor)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu');p.add_argument('--output',required=True)
    p.add_argument('--schemes',nargs='+',required=True);p.add_argument('--dim',type=int,choices=(2,3),default=2)
    p.add_argument('--replica-checks',choices=('exact','contract'),default='exact')
    args=p.parse_args();os.environ['WARPSPH_TEST_DEVICE']=args.device
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    wp.init()
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo',timeout=timedelta(seconds=120))
    comm=RowChunkProcessGroup(dist.group.WORLD,args.device,replicaChecks=args.replica_checks)
    output=Path(args.output)
    comm.localPhase('empty:test:output',lambda:output.mkdir(parents=True,exist_ok=True))
    reports=[]
    try:
        sources=_sourcePins();comm.agree('empty:test:sources',{k:v['digest'] for k,v in sources.items()})
        stream=torch.cuda.Stream() if args.device.startswith('cuda') else None
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            for name in args.schemes:
                cfg,base=configuration(name,args.device,args.dim)
                started=time.monotonic()
                print(f'empty-owner rank{comm.rank}: {name} flat reference',flush=True)
                reference=cfg.run(base);expected=nativeState(reference)
                referenceTime=float(reference.state.t);referenceDt=torch.as_tensor(reference.ctx.config.dt).clone()
                source=getattr(reference.state,'sourceTerms',None)
                forcing=None if source is None else source.acceleration.state_dict()
                # Capture expected state only; no reference adjacency stays live.
                referenceWall=reference.wallTime
                del reference
                ownership=[]
                def case(rows,phase):
                    def build(ctx):
                        def factory(state,config):
                            compatible=ctx.spec.scheme in ('CRKSPH','CompSPH')
                            components=state.positions.shape[1] if compatible else 1
                            policy=RowChunkPolicy(edgeBudget=1<<20,maxRows=rows,
                                bytesPerEdge=8+components*state.positions.element_size(),maxElementsPerEdge=components)
                            return buildChunkedVerletList(state,config.domain,policy=policy,
                                supportMode=SupportScheme.SuperSymmetric if cfg.case in ('tgv','tgv-wc') else config.supportMode,
                                verletScale=config.verletScale,communication=comm)
                        return base.buildSystem(ctx,neighborhoodFactory=factory)
                    def accepted(ctx,system,completed):
                        adjacency=system.adjacency
                        ranges=adjacency.plan.ownerRanges
                        active=sum(a!=b for a,b in ranges)
                        assert active==(1 if phase=='empty' else comm.size),(phase,ranges)
                        if phase=='empty' and comm.rank!=0:
                            assert len(adjacency.chunks)==0 and adjacency.memoryUsage()['pairCapacityBytes']==0
                            from warpSPHCore.radiusSearch import ChunkedPairField
                            work=getattr(getattr(system,'pairWork',None),'work',None)
                            if isinstance(work,ChunkedPairField):assert not work.buffers
                        ownership.append(dict(phase=phase,completed=completed,ranges=ranges,
                            localChunks=len(adjacency.chunks),localEdges=sum(c.metadata.liveEdges for c in adjacency)))
                    return replace(base,buildSystem=build,acceptedStep=accepted)
                print(f'empty-owner rank{comm.rank}: {name} empty-owner prefix',flush=True)
                cfg.simulation.update(nSteps=2,exportRoot=str(output/name))
                first=cfg.run(case(1<<20,'empty'))
                checkpoint=output/(name.replace(':','-')+'.h5')
                writeChunkedCheckpoint(checkpoint,first.ctx,first.state)
                n=len(first.state.state.positions);prefixWall=first.wallTime
                del first
                print(f'empty-owner rank{comm.rank}: {name} activate owner on restart',flush=True)
                cfg.simulation['resumeFrom']=str(checkpoint)
                resumed=cfg.run(case(max(1,n//(2*comm.size)),'active'))
                def compare():
                    actual=nativeState(resumed)
                    assert actual.keys()==expected.keys()
                    assert float(resumed.state.t)==referenceTime and resumed.ctx.scratch['completedSteps']==4
                    torch.testing.assert_close(torch.as_tensor(resumed.ctx.config.dt),referenceDt,rtol=0,atol=0)
                    source=getattr(resumed.state,'sourceTerms',None)
                    assert (source is None)==(forcing is None)
                    if source is not None:
                        actualForcing=source.acceleration.state_dict()
                        assert actualForcing['parameters']==forcing['parameters']
                        for field in ('knots','generator'):
                            torch.testing.assert_close(actualForcing[field],forcing[field],rtol=0,atol=0,msg=name+':forcing:'+field)
                    for field,value in expected.items():
                        tol=2e-12 if field=='sourceEnergyBudget' else 0
                        torch.testing.assert_close(actual[field],value,rtol=tol,atol=tol,msg=name+':'+field)
                comm.localPhase('empty:test:parity',compare)
                report=dict(scheme=name,dim=args.dim,status='passed',fields=len(expected),forcingClockChecked=forcing is not None,ownership=ownership,
                    wallSeconds=time.monotonic()-started,referenceWallSeconds=referenceWall,prefixWallSeconds=prefixWall,
                    resumedWallSeconds=resumed.wallTime)
                reports.append(report)
                print(f'empty-owner rank{comm.rank}: {name} passed in {report["wallSeconds"]:.3f}s',flush=True)
                del resumed,expected
        if stream is not None:stream.synchronize()
        record=dict(status='passed',runtime=_runtime(comm),sources=sources,variants=reports,communication=comm.summary())
        (output/f'rank-{comm.rank}.json').write_text(json.dumps(record,indent=2))
        print(f'empty-owner rank{comm.rank}: all lifecycle checks passed',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
