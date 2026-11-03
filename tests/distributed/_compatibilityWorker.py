"""Bounded Godunov/prepared/unsupported-context process compatibility matrix."""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse,contextlib,importlib,json,time

import torch
import torch.autograd.forward_ad as fwAD
import torch.distributed as dist
import warp as wp

from warpSPHCore import DomainDescription,KernelFunctions,OperationProperties,SupportScheme,setNeighborLanes
from warpSPHCore.autograd import SPHContext,PreparedLauncher
from warpSPHCore.distributed import RowChunkProcessGroup,DistributedContractError
from warpSPHCore.radiusSearch import buildChunkedVerletList,RowChunkPolicy,radiusSearchRowChunks_
from warpSPHCore.radiusSearch.verlet.build import buildVerletList_
from warpSPH.configurations.gsph import GSPHConfig
from warpSPH.configurations.moduleConfigurations.diffusionParameters import StateLimiter,RiemannSolver
from warpSPH.modules.godunov import computeGodunovWarp,computeGaussianDensityWarp,computeInutsukaWarp
from warpSPH.modules.godunov.chunked import checkGodunovChunks
from warpSPH.distributed import distributedRunStart,CHUNKED_SCHEMES

# Reuse the durable core's tiny independent fixture helpers without installation.
import importlib.util
_core=Path(importlib.import_module('warpSPHCore').__file__).resolve().parents[2]/'tests/distributed/_rowChunkWorker.py'
_spec=importlib.util.spec_from_file_location('_compat_core_fixture',_core)
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)
sample,equal=_fixture.sample,_fixture.equal


def operators(comm):
    records=[]
    for dim in (2,3):
        for n in (0,1,11):
            q,r,domain=sample(comm.device,nq=n,dim=dim)
            q.velocities=torch.sin(q.positions*2)
            q.pressures=.6+.1*torch.cos(torch.arange(n,dtype=q.positions.dtype,device=comm.device))
            props=OperationProperties(kernel=KernelFunctions.Wendland2,supportMode=SupportScheme.KernelMeanSymmetric)
            stored=(n+dim)%2==1
            adj=buildChunkedVerletList(q,domain,policy=RowChunkPolicy(edgeBudget=128,maxRows=2,bytesPerEdge=8),
                verletScale=1.5,supportMode=SupportScheme.SuperSymmetric,storeI=stored,communication=comm)
            flat=buildVerletList_(q.positions,q.positions,q.supports,q.supports,domain,verletScale=1.5,mode=SupportScheme.SuperSymmetric,storeI=stored)
            originalRho=q.densities.clone()
            hG=q.supports/3
            J=torch.sin(torch.arange(n*dim*dim,dtype=q.positions.dtype,device=comm.device)).reshape(n,dim,dim)*.03
            G=torch.cos(torch.arange(n*2*dim,dtype=q.positions.dtype,device=comm.device)).reshape(n,2,dim)*.02
            for order in (1,2):
                for centred in (False,True):
                    q.densities=originalRho.clone()
                    params=GSPHConfig().diffusionParams
                    params.stateLimiter=StateLimiter.VanLeerMonotonized.value
                    params.riemannSolver=RiemannSolver.Adaptive.value
                    params.riemannReconstruction=order==2
                    params.timeCentredEnergy=centred
                    kwargs=dict(domain=domain,params=params,gamma=1.4,order=order,
                        queryVelocityTensor=J if order==2 else None,queryStateGradients=G if order==2 else None)
                    gm=importlib.import_module('warpSPH.modules.godunov.wp_gsph')
                    captured={};launch=gm.launchOperator
                    def observe(spec,ctx,**extras):
                        captured.update(spec=spec,ctx=ctx,extras=extras)
                        return launch(spec,ctx,**extras)
                    first=computeGodunovWarp(q,props,adjacency=adj,**kwargs)
                    equal(first,computeGodunovWarp(q,props,adjacency=flat,**kwargs))
                    if centred:kwargs.update(queryAccelerations=first[0],halfDt=.0001)
                    with patch.object(gm,'launchOperator',observe):actual=computeGodunovWarp(q,props,adjacency=adj,**kwargs)
                    expected=computeGodunovWarp(q,props,adjacency=flat,**kwargs);equal(actual,expected)
                    prepared=PreparedLauncher(captured['ctx'])
                    equal(prepared.launch(captured['spec'],**captured['extras']),expected)
                    im=importlib.import_module('warpSPH.modules.godunov.wp_inutsuka');launch=im.launchOperator
                    with patch.object(im,'launchOperator',observe):
                        grad,rho=computeGaussianDensityWarp(q,props,domain=domain,adjacency=adj,gaussianH=hG)
                    equal((grad,rho),computeGaussianDensityWarp(q,props,domain=domain,adjacency=flat,gaussianH=hG))
                    equal(PreparedLauncher(captured['ctx']).launch(captured['spec'],**captured['extras']),(grad,rho))
                    q.densities=rho
                    kwargs.pop('queryAccelerations',None);kwargs.pop('halfDt',None)
                    kwargs.update(gradRho=grad,gaussianH=hG)
                    im=importlib.import_module('warpSPH.modules.godunov.wp_inutsuka');launch=im.launchOperator
                    first=computeInutsukaWarp(q,props,adjacency=adj,**kwargs)
                    equal(first,computeInutsukaWarp(q,props,adjacency=flat,**kwargs))
                    if centred:kwargs.update(queryAccelerations=first[0],halfDt=.0001)
                    with patch.object(im,'launchOperator',observe):actual=computeInutsukaWarp(q,props,adjacency=adj,**kwargs)
                    expected=computeInutsukaWarp(q,props,adjacency=flat,**kwargs);equal(actual,expected)
                    equal(PreparedLauncher(captured['ctx']).launch(captured['spec'],**captured['extras']),expected)
                    records.append(dict(dim=dim,rows=n,order=order,timeCentred=centred,prepared=True,preparedGaussian=True,storeI=stored))
                    print(f'[compat rank {comm.rank}] operator {len(records)}/24 passed',flush=True)
    return records


def guards(comm):
    q,r,domain=sample(comm.device,nq=11)
    adj=buildChunkedVerletList(q,domain,policy=RowChunkPolicy(edgeBudget=128,maxRows=2),
        verletScale=1.5,supportMode=SupportScheme.SuperSymmetric,communication=comm)
    failures=[]
    def rejected(tag,function,text):
        try:function()
        except (DistributedContractError,NotImplementedError) as exc:assert text in str(exc),str(exc)
        else:raise AssertionError('expected context rejection: '+tag)
        failures.append(tag)
    def forbidden(*a,**kw):raise AssertionError('unsupported context entered payload/callback')
    with patch.object(dist,'broadcast',forbidden):
        t=torch.zeros(11,device=comm.device,requires_grad=comm.rank==comm.size-1)
        rejected('reverse-ad',lambda:comm.exchangeRows(t,adj.plan.ownerRanges,tag='matrix:reverse',signature='same'),'AD')
        with fwAD.dual_level():
            primal=torch.zeros(11,device=comm.device)
            value=fwAD.make_dual(primal,torch.ones_like(primal)) if comm.rank==comm.size-1 else primal
            rejected('forward-ad',lambda:comm.exchangeRows(value,adj.plan.ownerRanges,tag='matrix:forward',signature='same'),'AD')
            dual=fwAD.make_dual(q.positions,torch.ones_like(q.positions)) if comm.rank==comm.size-1 else q.positions
            rejected('search-forward-ad',lambda:radiusSearchRowChunks_(dual,q.positions,q.supports,q.supports,domain,
                policy=RowChunkPolicy(edgeBudget=128,maxRows=2),communication=comm),'AD')
        spec=SimpleNamespace(nSteps=1,tLimit=None,integrationScheme='rungeKutta2',stallDtSteps=8,stallProgress=.001,
                             store=False,plot=False,video=False,resumeFrom=None)
        ctx=SimpleNamespace(scheme='GSPH',schemeConfig=SimpleNamespace(),spec=spec,scratch={'execution':'eager'})
        system=SimpleNamespace(adjacency=adj,state=q,sourceTerms=None)
        for name in CHUNKED_SCHEMES:
            ctx.scheme=name
            assert distributedRunStart(ctx,system) is comm
        for name in ('MFM','MFV','acSPH','deltaSPH','iisph'):
            ctx.scheme=name
            rejected(name,lambda:distributedRunStart(ctx,system),'support eager')
        ctx.scheme='GSPH';ctx.scratch['execution']='graph'
        rejected('runner-graph',lambda:distributedRunStart(ctx,system),'requires eager')
        ctx.scratch['execution']='eager'
        for feature in ('materials','walls','graph','reverse-ad'):
            cfg=SimpleNamespace(compute=SimpleNamespace(execution='eager'))
            sc=SimpleNamespace(materials=None,stepMode='eager')
            state=replace(q)
            if feature=='materials':sc.materials=[1] if comm.rank==comm.size-1 else None
            if feature=='walls':state.kinds=torch.ones_like(q.kinds) if comm.rank==comm.size-1 else q.kinds
            if feature=='graph':sc.stepMode='graph' if comm.rank==comm.size-1 else 'eager'
            if feature=='reverse-ad':state.positions=q.positions.clone().requires_grad_(comm.rank==comm.size-1)
            rejected('godunov-'+feature,lambda:checkGodunovChunks(SimpleNamespace(adjacency=adj,state=state),cfg,sc,.001),
                     'differentiation' if feature=='reverse-ad' else 'eager uniform-material')
        from warpSPH.io.legacyChunkCheckpoint import readLegacyCheckpoint
        rejected('godunov-legacy',lambda:comm.localPhase('matrix:legacy',lambda:readLegacyCheckpoint(None,ctx,system,None,comm.device)),
                 'Legacy chunk import')
    if comm.device.type=='cuda':
        from warpSPHCore.autograd.lanes import neighborLanes
        previous=neighborLanes()
        try:
            setNeighborLanes(1 if comm.rank==0 else 32)
            dispatch=importlib.import_module('warpSPHCore.autograd.row_dispatch')
            prepared=PreparedLauncher(SPHContext(q,OperationProperties(kernel=KernelFunctions.Wendland2),domain,adj))
            with patch.object(dispatch,'_launchLocal',forbidden):
                rejected('different-effective-lanes',lambda:prepared.launch(_fixture._DENSITY_SPEC),'disagree')
        finally:setNeighborLanes(previous)
        comm.agree('matrix:lanes-recovered',previous)
        # Matched eager interfaces reject *locally* during capture. No callback,
        # packet copy, host readback or c10d operation may be attempted there.
        serial=buildChunkedVerletList(q,domain,policy=RowChunkPolicy(edgeBudget=128,maxRows=2),
            verletScale=1.5,supportMode=SupportScheme.SuperSymmetric)
        q.velocities=torch.zeros_like(q.positions);q.pressures=torch.ones_like(q.supports)
        props=OperationProperties(kernel=KernelFunctions.Wendland2,supportMode=SupportScheme.KernelMeanSymmetric)
        params=GSPHConfig().diffusionParams;gradient=torch.zeros_like(q.positions)
        from warpSPH.modules.compSPH.chunkedWork import workInputs,checkChunkedStage
        from warpSPH.configurations.crkSPH import CRKSPHConfig
        native=SimpleNamespace(adjacency=serial,pairWork=SimpleNamespace(latest=None,weights=(1,),work=None))
        cfg=SimpleNamespace(compute=SimpleNamespace(execution='eager'));nativeScheme=CRKSPHConfig()
        before=comm.sequence;graph=torch.cuda.CUDAGraph()
        value=torch.zeros(1,device=comm.device)
        with torch.cuda.graph(graph):
            with patch.object(dist,'all_gather',forbidden):
                rejected('capture-agree',lambda:comm.agree('matrix:captured',None),'graph capture')
                rejected('capture-local',lambda:comm.localPhase('matrix:captured',forbidden),'graph capture')
                rejected('capture-replica',lambda:comm.assertReplicated('matrix:captured',dict(value=value)),'graph capture')
                rejected('capture-constructor',lambda:RowChunkProcessGroup(comm.group,comm.device),'graph capture')
                rejected('capture-serial-gsph',lambda:computeGodunovWarp(q,props,domain=domain,params=params,gamma=1.4,adjacency=serial),'graph capture')
                rejected('capture-serial-gaussian',lambda:computeGaussianDensityWarp(q,props,domain=domain,adjacency=serial),'graph capture')
                rejected('capture-serial-inutsuka',lambda:computeInutsukaWarp(q,props,domain=domain,params=params,gamma=1.4,gradRho=gradient,order=1,adjacency=serial),'graph capture')
                rejected('capture-serial-search',lambda:radiusSearchRowChunks_(q.positions,q.positions,q.supports,q.supports,domain,policy=serial.plan.policy),'graph capture')
                rejected('capture-serial-work-inputs',lambda:workInputs(serial,forbidden),'graph capture')
                rejected('capture-serial-native-stage',lambda:checkChunkedStage(native,q,cfg,nativeScheme,0,.001,'CRK'),'graph capture')
        assert comm.sequence==before
    comm.agree('matrix:guards-recovered',len(failures))
    return failures


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',required=True)
    parser.add_argument('--replica-checks',choices=('exact','contract'),default='exact');parser.add_argument('--lanes',type=int,default=1)
    args=parser.parse_args()
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo',timeout=timedelta(seconds=45))
    comm=RowChunkProcessGroup(dist.group.WORLD,args.device,replicaChecks=args.replica_checks)
    wp.init();setNeighborLanes(args.lanes);started=time.perf_counter()
    stream=torch.cuda.Stream(device=comm.device) if comm.device.type=='cuda' else None
    try:
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            cases=operators(comm);failures=guards(comm)
        if stream is not None:stream.synchronize()
        report=dict(status='passed',operators=cases,rejectedContexts=failures,wallSeconds=time.perf_counter()-started,
                    lanes=args.lanes,**comm.summary())
        if comm.device.type=='cuda':
            import ctypes
            version=ctypes.c_int();ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
            report['runtimeNccl']=version.value
        folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
        (folder/f'compat-rank{comm.rank}.json').write_text(json.dumps(report,indent=2))
        print(f'[compat rank {comm.rank}] all compatibility checks passed',flush=True)
    finally:setNeighborLanes(None);dist.destroy_process_group()

if __name__=='__main__':main()
