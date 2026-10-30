"""Tiny accepted-step restart controls across independent torchrun jobs.

No videos for these bounded A/B fixtures; progress/stall controls remain enabled.
The test oracle stores particle fields and per-row hashes, never pair arrays.
"""
from dataclasses import replace
from datetime import timedelta
from functools import wraps
from hashlib import sha256
import argparse
import contextlib
import importlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from warpSPH.enumTypes import AdaptiveSupportScheme
from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
from warpSPHCore import SupportScheme,setNeighborLanes,get_torch_offset_dtype
from warpSPHCore.distributed import RowChunkProcessGroup,DistributedContractError
from warpSPHCore.radiusSearch import RowChunkPolicy,buildChunkedVerletList,ChunkedPairField,ChunkedAdjacency


def rowHashes(adjacency,communication,field=None):
    out=torch.zeros((adjacency.numRows,32),dtype=torch.uint8,device=communication.device)
    if not isinstance(adjacency,ChunkedAdjacency):
        assert communication.size==1,'flat legacy fixtures are serial only'
        value=adjacency.paddedJ if field is None else field
        for row in range(adjacency.numRows):
            start,n=int(adjacency.edgeOffsets[row]),int(adjacency.numNeighbors[row])
            raw=value[start:start+n].detach().cpu().contiguous().numpy().tobytes()
            out[row]=torch.tensor(list(sha256(raw).digest()),dtype=torch.uint8,device=out.device)
        return out.cpu()
    for index,chunk in enumerate(adjacency):
        a,m=chunk.adjacency,chunk.metadata
        value=a.paddedJ if field is None else field.buffers[index]
        for local in range(m.rowCount):
            start,n=int(a.edgeOffsets[local]),int(a.numNeighbors[local])
            raw=value[start:start+n].detach().cpu().contiguous().numpy().tobytes()
            out[m.rowStart+local]=torch.tensor(list(sha256(raw).digest()),dtype=torch.uint8,device=out.device)
    communication.exchangeRows(out,adjacency.plan.ownerRanges,tag='test:row-hashes',signature=adjacency.plan.signature)
    return out.cpu()


def compare(actual,expected):
    assert actual['time']==expected['time'],(actual['time'],expected['time'])
    assert actual['completed']==expected['completed']
    for name,value in expected['state'].items():
        tolerance=2e-12 if name=='sourceEnergyBudget' else 0
        torch.testing.assert_close(actual['state'][name],value,rtol=tolerance,atol=tolerance,msg=lambda m:name+': '+m)
    for key in ('knots','generator'):
        torch.testing.assert_close(actual['forcing'][key],expected['forcing'][key],rtol=0,atol=0)
    assert actual['forcing']['parameters']==expected['forcing']['parameters']
    assert actual['dtKind']==expected['dtKind']
    torch.testing.assert_close(actual['dt'],expected['dt'],rtol=0,atol=0)
    torch.testing.assert_close(actual['finalRows'],expected['finalRows'],rtol=0,atol=0)
    tail=expected['stages'][-len(actual['stages']):]
    assert len(tail)==len(actual['stages'])
    for a,b in zip(actual['stages'],tail):
        assert a['time']==b['time']
        for key in ('dvdt','dudt','rows'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        assert (a['work'] is None)==(b['work'] is None)
        if a['work'] is not None:
            torch.testing.assert_close(a['work'],b['work'],rtol=0,atol=0)


def run(args,communication):
    stages=[]
    old=[]
    contracts=[]
    def patch(module,name,value):
        old.append((module,name,getattr(module,name)))
        setattr(module,name,value)
    def rejected(function,text):
        try:function()
        except (DistributedContractError,ValueError,OSError) as exc:
            assert text in str(exc),str(exc)
            contracts.append(text)
        else:raise AssertionError('expected coordinated checkpoint rejection: '+text)
    def build(ctx):
        module=importlib.import_module('warpSPH.cases.compressible.drivenTurbulence')
        def factory(state,config):
            policy=RowChunkPolicy(edgeBudget=args.edge_budget,maxRows=args.rows,
                bytesPerEdge=4+state.positions.shape[1]*state.positions.element_size(),maxElementsPerEdge=2)
            return buildChunkedVerletList(state,config.domain,policy=policy,communication=communication,
                supportMode=SupportScheme.SuperSymmetric,verletScale=config.verletScale)
        def forbidden(*a,**kw):
            raise AssertionError('restart constructed a monolithic neighbour list')
        if args.legacy:
            assert communication.size==1
            system=module.buildSPHSystem(ctx)
        else:
            patch(importlib.import_module('warpSPHCore.radiusSearch.verlet.build'),'radiusSearchCompactHashMap_',forbidden)
            patch(importlib.import_module('warpSPHCore.radiusSearch.compactHash.search'),'radiusSearchOnCompactHashMap',forbidden)
            system=module.buildSPHSystem(ctx,neighborhoodFactory=factory)
        sc=ctx.schemeConfig
        sc.adaptiveSupportScheme=AdaptiveSupportScheme.NoScheme
        sc.adaptiveSupportCorrections=args.scheme=='CompSPH'
        s=system.state
        s.velocities[:,0]=.13*torch.sin(6.283*s.positions[:,1])
        s.velocities[:,1]=.08*torch.cos(6.283*s.positions[:,0])
        s.internalEnergies*=1+.08*torch.cos(6.283*s.positions[:,0])
        from warpSPH.modules.eos import refreshEOS
        refreshEOS(s,sc)
        ctx.scratch['initialEnergy']=(s.masses.double()*(s.internalEnergies.double()+.5*s.velocities.double().square().sum(-1))).sum()
        native=ctx.bundle.stepFunction
        @wraps(native)
        def observed(system,*a,**kw):
            if args.contracts and not stages:
                rejected(lambda:writeChunkedCheckpoint(Path(args.checkpoint).with_name('midstage.h5'),ctx,system),'accepted-step boundary')
            result=native(system,*a,**kw)
            update=result[0]
            pw=getattr(system,'pairWork',None)
            work=pw.work if pw is not None and isinstance(pw.work,(ChunkedPairField,torch.Tensor)) else None
            communication.agree('test:work-present',work is not None)
            stages.append(dict(time=float(system.t),dvdt=update.dvdt.detach().cpu().clone(),
                dudt=update.dudt.detach().cpu().clone(),rows=rowHashes(system.adjacency,communication),
                work=None if work is None else rowHashes(pw.adjacency,communication,work)))
            return result
        ctx.bundle=replace(ctx.bundle,stepFunction=observed)
        return system
    solver=dict(scheme=args.scheme)
    if args.scheme in ('CRKSPH','CompSPH'):
        solver[('crk' if args.scheme=='CRKSPH' else 'comp')+'PairStorage']='recompute'
    simulation=dict(device=args.device,nx=8,dim=2,nSteps=args.steps,maxDt=.0002,
        plot=False,video=False,store=False,storeMode='states' if args.legacy else 'trajectory',quiet=True,show=False,
        progress=True,velocityAlarmPlotInterval=1,stallProgress=1e-3,integrationScheme='rungeKutta4')
    if args.legacy:simulation['neighborOmitI']=args.legacy=='csr'
    if args.action in ('resume','reject'):
        simulation['resumeFrom']=args.checkpoint
    configuration=RunConfiguration('drivenTurbulence',simulation=simulation,
        params=dict(accelerationRMS=args.amplitude,forcingInterval=.00025),solver=solver)
    try:
        result=configuration.run(replace(case,buildSystem=build))
    finally:
        for module,name,value in reversed(old):setattr(module,name,value)
    assert not result.diverged and result.nSteps==args.steps
    if args.action=='save':
        if args.legacy:
            from warpSPH.io.export import exportSimulationSystem
            from warpSPH.cases.compressible.drivenTurbulence import checkpointSignature
            result.ctx.scratch['checkpointSignature']=checkpointSignature(result.ctx)
            path=Path(args.checkpoint)
            export=path.parent/'legacy-export'
            exportSimulationSystem(str(export),'checkpoint',result.ctx.scheme,result.state,
                extraData=case.extraData(result.ctx,result.state))
            (export/'trajectory/checkpoint.h5').replace(path)
        else:
            writeChunkedCheckpoint(args.checkpoint,result.ctx,result.state)
        if args.contracts:
            path=Path(args.checkpoint)
            before=sha256(path.read_bytes()).hexdigest()
            rejected(lambda:writeChunkedCheckpoint(path,result.ctx,result.state),'already exists')
            assert sha256(path.read_bytes()).hexdigest()==before
            module=importlib.import_module('warpSPH.io.chunkCheckpoint')
            dump=module.dumpState
            def interrupted(*a,**kw):
                dump(*a,**kw)
                raise OSError('injected checkpoint write failure')
            temporary=path.with_name('interrupted.h5')
            if communication.rank==0:module.dumpState=interrupted
            try:
                rejected(lambda:writeChunkedCheckpoint(temporary,result.ctx,result.state),'injected checkpoint write failure')
            finally:module.dumpState=dump
            assert not temporary.exists() and not list(temporary.parent.glob('.interrupted.h5.*.tmp'))
            writeChunkedCheckpoint(temporary,result.ctx,result.state)
    data=dict(time=float(result.state.t),completed=result.ctx.scratch['completedSteps'],
        state={k:v.detach().cpu().clone() for k,v in vars(result.state.state).items() if isinstance(v,torch.Tensor)},
        forcing=result.ctx.scratch['driving'].state_dict(),dt=torch.as_tensor(result.ctx.config.dt).detach().cpu(),
        dtKind='tensor' if isinstance(result.ctx.config.dt,torch.Tensor) else 'python',stages=stages,
        finalRows=rowHashes(result.state.adjacency,communication))
    if args.action=='reference':
        def publish():
            if communication.rank==0:torch.save(data,args.oracle)
        communication.localPhase('test:publish-oracle',publish)
    elif args.action=='resume':
        expected=communication.localPhase('test:read-oracle',lambda:torch.load(args.oracle,map_location='cpu',weights_only=True))
        communication.localPhase('test:restart-parity',lambda:compare(data,expected))
    report=dict(status='passed',action=args.action,scheme=args.scheme,rank=communication.rank,
        ranks=communication.size,offsetDtype=str(get_torch_offset_dtype()),steps=args.steps,
        completed=data['completed'],rhs=len(stages),contracts=contracts,time=data['time'],
        checkpoint=args.checkpoint,rows=args.rows,edgeBudget=args.edge_budget)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=('reference','save','resume','reject'),required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--scheme',default='CRKSPH')
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--rows',type=int,default=3)
    parser.add_argument('--edge-budget',type=int,default=1024)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--oracle',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--legacy',choices=('csr','coo'))
    parser.add_argument('--contracts',action='store_true')
    parser.add_argument('--expect-error')
    parser.add_argument('--amplitude',type=float,default=.1)
    args=parser.parse_args()
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo',timeout=timedelta(seconds=45))
    communication=RowChunkProcessGroup(dist.group.WORLD,args.device)
    setNeighborLanes(1)
    try:
        stream=torch.cuda.Stream(device=communication.device) if communication.device.type=='cuda' else None
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            try:
                report=run(args,communication)
            except (DistributedContractError,ValueError,OSError) as exc:
                if args.expect_error is None:raise
                assert args.expect_error in str(exc),str(exc)
                communication.agree('test:expected-checkpoint-failure',args.expect_error)
                report=dict(status='passed',action='expected failure',rank=communication.rank,error=str(exc))
            else:
                assert args.expect_error is None,'expected restart to fail'
        if stream is not None:stream.synchronize()
        if args.device.startswith('cuda'):
            import ctypes
            version=ctypes.c_int()
            ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
            report['runtimeNccl']=version.value
            report['gpuUuid']=str(torch.cuda.get_device_properties(communication.device).uuid)
        Path(args.output).mkdir(parents=True,exist_ok=True)
        (Path(args.output)/f'{args.action}-rank{communication.rank}.json').write_text(json.dumps(report,indent=2))
        print(f'[checkpoint rank {communication.rank}] {args.action} {args.scheme}: passed',flush=True)
    finally:
        setNeighborLanes(None)
        dist.destroy_process_group()


if __name__=='__main__':main()
