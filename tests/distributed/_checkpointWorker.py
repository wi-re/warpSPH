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

from warpSPH.runner import RunConfiguration, RunAllocationError
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from warpSPH.enumTypes import AdaptiveSupportScheme
from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
from warpSPHCore import SupportScheme,setNeighborLanes,get_torch_offset_dtype
from warpSPHCore.distributed import RowChunkProcessGroup,DistributedContractError,DistributedAllocationError
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
        if args.action=='fail':
            def shouldFail():
                return communication.rank==args.fail_rank and ctx.scratch.get('completedSteps')==2
            if args.fail_site=='work':
                empty=ChunkedPairField.empty
                @classmethod
                def rejectedWork(cls,adjacency,**kwargs):
                    if shouldFail():raise torch.OutOfMemoryError('injected owned work allocation')
                    return empty(adjacency,**kwargs)
                patch(ChunkedPairField,'empty',rejectedWork)
            else:
                chunks=importlib.import_module('warpSPHCore.radiusSearch.chunked')
                allocate=chunks._allocateChunks
                def rejectedNeighbors(*a,**kw):
                    if shouldFail():raise MemoryError('injected neighbor allocation')
                    return allocate(*a,**kw)
                patch(chunks,'_allocateChunks',rejectedNeighbors)
                stage=importlib.import_module('warpSPH.schemes.compressible.'+('crkSPH' if args.scheme=='CRKSPH' else 'compSPH'))
                nativePairs=stage.stagePairList
                def rebuild(s,state,cfg,sc,**kw):
                    adj=nativePairs(s,state,cfg,sc,**kw)
                    if ctx.scratch.get('completedSteps')==2:
                        return buildChunkedVerletList(state,cfg.domain,policy=adj.plan.policy,
                            priorNeighborhood=adj,forceRebuild=True,communication=communication,
                            supportMode=SupportScheme.SuperSymmetric,verletScale=cfg.verletScale)
                    return adj
                patch(stage,'stagePairList',rebuild)
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
        exportRoot=args.output,progress=True,velocityAlarmPlotInterval=1,stallProgress=1e-3,integrationScheme='rungeKutta4')
    if args.legacy:simulation['neighborOmitI']=args.legacy=='csr'
    if args.action in ('resume','reject'):
        simulation['resumeFrom']=args.checkpoint
    configuration=RunConfiguration('drivenTurbulence',simulation=simulation,
        params=dict(accelerationRMS=args.amplitude,forcingInterval=.00025),solver=solver)
    allocationFailure=None
    try:
        result=configuration.run(replace(case,buildSystem=build))
    except RunAllocationError as error:
        if args.action!='fail':raise
        assert isinstance(error.__cause__,DistributedAllocationError)
        assert error.recoveryError is None,error.recoveryError
        assert error.checkpointPath and Path(error.checkpointPath).is_file()
        assert error.result.nSteps==2 and error.result.ctx.scratch['completedSteps']==2
        allocationFailure=dict(checkpoint=error.checkpointPath,primary=str(error.__cause__))
        communication.agree('test:recovery-published',error.checkpointPath)
        result=error.result
    finally:
        for module,name,value in reversed(old):setattr(module,name,value)
    assert (allocationFailure is not None and result.diverged) if args.action=='fail' else (not result.diverged and result.nSteps==args.steps)
    if args.action in ('save','fail'):
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
        checkpoint=args.checkpoint,rows=args.rows,edgeBudget=args.edge_budget,allocationFailure=allocationFailure)
    return report


def launcherFailure(args, communication):
    from warpSPH.distributedRun import runDistributed
    empty=ChunkedPairField.empty
    calls=0
    @classmethod
    def failing(cls,adjacency,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3 and communication.rank==0:
            raise torch.OutOfMemoryError('injected launcher allocation')
        return empty(adjacency,**kwargs)
    ChunkedPairField.empty=failing
    configuration=RunConfiguration('drivenTurbulence',simulation=dict(nx=8,dim=2,nSteps=4,maxDt=.0002,
        plot=False,video=False,store=False,execution='eager',cudaGraph=False,integrationScheme='rungeKutta2'),
        solver=dict(scheme='CRKSPH'))
    try:
        runDistributed(configuration,communication,outputDirectory=args.output,edgeBudget=1024,maxRows=3)
    except RunAllocationError as error:
        assert error.recoveryError is None,error.recoveryError
        manifest=json.loads((Path(args.output)/'manifest.json').read_text()) if communication.rank==0 else None
        failure=json.loads((Path(args.output)/f'failure-{communication.rank:05d}.json').read_text())
        for value in (failure, manifest) if manifest is not None else (failure,):
            assert value['status']=='failed' and value['recovery']['completed']==2
            assert value['recovery']['checkpointPath']==error.checkpointPath
        assert Path(error.checkpointPath).is_relative_to(Path(args.output).resolve())
        assert Path(error.checkpointPath).is_file()
        return dict(status='passed',action='launcher allocation failure',rank=communication.rank,recovery=failure['recovery'])
    else:
        raise AssertionError('launcher allocation failure was not injected')
    finally:
        ChunkedPairField.empty=empty


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=('reference','save','resume','reject','fail'),required=True)
    parser.add_argument('--replica-checks',choices=('exact','contract'),default='exact')
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
    parser.add_argument('--fail-site',choices=('work','neighbors'),default='work')
    parser.add_argument('--fail-rank',type=int,default=0)
    parser.add_argument('--launcher-failure',action='store_true')
    args=parser.parse_args()
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo',timeout=timedelta(seconds=45))
    communication=RowChunkProcessGroup(dist.group.WORLD,args.device,replicaChecks=args.replica_checks)
    setNeighborLanes(1)
    try:
        stream=torch.cuda.Stream(device=communication.device) if communication.device.type=='cuda' else None
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            try:
                report=launcherFailure(args,communication) if args.launcher_failure else run(args,communication)
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
