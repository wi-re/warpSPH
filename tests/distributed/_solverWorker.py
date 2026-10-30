"""Bounded distributed solver A/Bs, intended for Gloo or shared-GPU NCCL.

Video is omitted for these tiny repeated correctness fixtures. Progress and a
stall watchdog stay enabled. Timings under GPU contention are diagnostic only.
Flat references run first and are released before distributed initialization;
only tiny CPU oracle snapshots remain, never a production global pair fallback.
"""
from dataclasses import replace
from datetime import timedelta
from functools import wraps
import argparse
import contextlib
import gc
import importlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from warpSPH.runner import RunConfiguration
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as case
from warpSPH.enumTypes import AdaptiveSupportScheme
from warpSPHCore import setNeighborLanes, SupportScheme
from warpSPHCore.distributed import RowChunkProcessGroup
from warpSPHCore.radiusSearch import buildChunkedVerletList, RowChunkPolicy, ChunkedPairField


def snapshot(system, update):
    pw = getattr(system, 'pairWork', None)
    data = dict(dvdt=update.dvdt.detach().cpu().clone(), dudt=update.dudt.detach().cpu().clone())
    if pw is None:
        return data
    data['reducing'] = pw.halfVelocity is not None
    data['delta'] = None if pw.deltaU is None else pw.deltaU.detach().cpu().clone()
    if isinstance(pw.work, ChunkedPairField):
        data['work'] = [(c.metadata.edgeStart, w[:c.metadata.liveEdges].detach().cpu().clone(),
                         c.adjacency.paddedJ[:c.metadata.liveEdges].detach().cpu().clone())
                        for c,w in zip(pw.adjacency, pw.work.buffers)]
    elif pw.work is not None:
        data['work'] = [(0, pw.work[:pw.adjacency.numEdges].detach().cpu().clone(),
                         pw.adjacency.paddedJ[:pw.adjacency.numEdges].detach().cpu().clone())]
    else:
        data['work'] = None
    return data


def run(args, scheme, dim, tableau, support, change, communication=None):
    records, rebuilt, rhs, acceptedMemory = [], [], [], []
    workModule = None
    originals = []
    def patch(module, name, value):
        originals.append((module,name,getattr(module,name)))
        setattr(module,name,value)
    native = scheme in ('CRKSPH','CompSPH')
    if native:
        family = 'crk' if scheme == 'CRKSPH' else 'compSPH'
        workModule = importlib.import_module('warpSPH.modules.'+family+'.work')
        function = 'accumulateCRKWork' if scheme == 'CRKSPH' else 'accumulateCompWork'
        accumulate = getattr(workModule,function)
        def capture(system,state,update,*a,**kw):
            accumulate(system,state,update,*a,**kw)
            records.append(snapshot(system,update))
        patch(workModule,function,capture)
        schemeModule = importlib.import_module('warpSPH.schemes.compressible.'+('crkSPH' if scheme=='CRKSPH' else 'compSPH'))
        pairList = schemeModule.stagePairList
        def changing(system,state,config,sc,**kw):
            adj = pairList(system,state,config,sc,**kw)
            pw = system.pairWork
            pw._distributedTestVisits = getattr(pw, '_distributedTestVisits', 0) + 1
            if change and not pw.conservative and pw._distributedTestVisits == 2:
                if communication is None:
                    from warpSPHCore import buildVerletList
                    adj = buildVerletList(state,config.domain,priorNeighborhood=adj,
                        supportMode=SupportScheme.SuperSymmetric,verletScale=config.verletScale,
                        forceRebuild=True,storeI=False,queryOrder='input')
                else:
                    adj = buildChunkedVerletList(state,config.domain,priorNeighborhood=adj,
                        policy=replace(adj.plan.policy,maxRows=2),verletScale=config.verletScale,
                        supportMode=SupportScheme.SuperSymmetric,
                        # Only one rank requests it: all ranks must rebuild.
                        forceRebuild=communication.rank==communication.size-1)
                system.adjacency = adj
                rebuilt.append(1)
            return adj
        patch(schemeModule,'stagePairList',changing)
    def build(ctx):
        module = importlib.import_module('warpSPH.cases.compressible.drivenTurbulence')
        def factory(state,config):
            policy=RowChunkPolicy(edgeBudget=1024,maxRows=5,
                bytesPerEdge=4+dim*state.positions.element_size(),maxElementsPerEdge=dim)
            return buildChunkedVerletList(state,config.domain,policy=policy,
                verletScale=config.verletScale,supportMode=SupportScheme.SuperSymmetric,communication=communication)
        if communication is not None:
            # Guard before setup: even the initial density solve must use chunks.
            def forbidden(*a,**kw):
                raise AssertionError('distributed solver constructed a monolithic neighbour list')
            bm=importlib.import_module('warpSPHCore.radiusSearch.verlet.build')
            patch(bm,'radiusSearchCompactHashMap_',forbidden)
            cm=importlib.import_module('warpSPHCore.radiusSearch.compactHash.search')
            patch(cm,'radiusSearchOnCompactHashMap',forbidden)
            system=module.buildSPHSystem(ctx,neighborhoodFactory=factory)
        else:
            system=case.buildSystem(ctx)
        sc=ctx.schemeConfig
        sc.adaptiveSupportScheme=getattr(AdaptiveSupportScheme,support)
        sc.adaptiveSupportIterations=4
        sc.adaptiveSupportCorrections=scheme=='CompSPH'
        s=system.state
        s.velocities[:,0]=.13*torch.sin(6.283*s.positions[:,1])
        s.velocities[:,1]=.08*torch.cos(6.283*s.positions[:,0])
        if dim==3:
            s.velocities[:,2]=.06*torch.sin(6.283*s.positions[:,2])
        s.internalEnergies*=1+.08*torch.cos(6.283*s.positions[:,0])
        from warpSPH.modules.eos import refreshEOS
        refreshEOS(s,sc)
        ctx.scratch['initialEnergy']=(s.masses.double()*(s.internalEnergies.double()+.5*s.velocities.double().square().sum(-1))).sum()
        original=ctx.bundle.stepFunction
        @wraps(original)
        def counted(*a,**kw):
            result=original(*a,**kw)
            rhs.append(1)
            if not native:
                records.append(snapshot(a[0],result[0]))
            return result
        ctx.bundle=replace(ctx.bundle,stepFunction=counted)
        return system
    solver=dict(scheme=scheme)
    if native:
        solver[('crk' if scheme=='CRKSPH' else 'comp')+'PairStorage']='recompute'
    config=RunConfiguration('drivenTurbulence',simulation=dict(device=args.device,
        dim=dim,nx=8 if dim==2 else 6,nSteps=args.steps,maxDt=.0002,plot=False,video=False,
        store=False,storeMode='trajectory',quiet=True,show=False,progress=True,
        stallProgress=1e-3,velocityAlarmPlotInterval=1,integrationScheme=tableau),
        params=dict(accelerationRMS=.1,forcingInterval=args.clock_interval),solver=solver)
    def postStep(ctx,system,step):
        if case.postStep is not None:
            case.postStep(ctx,system,step)
        if communication is not None and args.device.startswith('cuda'):
            acceptedMemory.append(dict(step=step,torchAllocatedBytes=torch.cuda.memory_allocated(),
                torchReservedBytes=torch.cuda.memory_reserved(),torchPeakBytes=torch.cuda.max_memory_allocated()))
    start=time.perf_counter()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    try:
        result=config.run(replace(case,buildSystem=build,postStep=postStep))
    finally:
        for module,name,value in reversed(originals):
            setattr(module,name,value)
    assert result.nSteps==args.steps and not result.diverged
    assert min(result.trajectory[-1][k] for k in ('minDensity','minPressure','minEnergy'))>0
    assert abs(result.trajectory[-1]['relativeEnergyBudgetResidual'])<2e-6
    state={k:v.detach().cpu().clone() for k,v in vars(result.state.state).items() if isinstance(v,torch.Tensor)}
    data=dict(state=state,records=records,rhs=len(rhs),time=float(result.state.t),rebuilds=len(rebuilt),
        wallSeconds=time.perf_counter()-start,steps=result.nSteps,
        budgetResidual=result.trajectory[-1]['relativeEnergyBudgetResidual'],acceptedMemory=acceptedMemory)
    forcing=result.ctx.scratch['driving'].state_dict()
    data['forcingState']=forcing
    data['forcingKnots']=len(forcing['knots'])
    data['forcingInterval']=args.clock_interval
    if communication is not None:
        adj=result.state.adjacency
        communication.validateAdjacency(adj)
        a,b=adj.plan.ownerRanges[communication.rank]
        assert all(c.metadata.owner==communication.rank for c in adj)
        # These small periodic fixtures have neighbours on the other owner's rows.
        cross=sum(int(((c.adjacency.paddedJ[:c.metadata.liveEdges]<a)|
                       (c.adjacency.paddedJ[:c.metadata.liveEdges]>=b)).sum()) for c in adj)
        assert cross>0
        pw=getattr(result.state,'pairWork',None)
        workBytes=0 if pw is None or not isinstance(pw.work,ChunkedPairField) else pw.work.memoryUsage()['capacityBytes']
        data['ownership']=dict(rows=[a,b],chunks=len(adj.chunks),crossOwnerEdges=cross,
            generation=adj.plan.generation.number,signature=adj.plan.signature,
            pairBytes=adj.memoryUsage()['pairCapacityBytes'],workBytes=workBytes,
            globalRows=adj.numRows,globalEdges=adj.numEdges)
    # Do not keep result/stage systems alive across the oracle/distributed runs.
    del result
    gc.collect()
    return data


def compare(actual,expected):
    assert actual['time']==expected['time'] and actual['rhs']==expected['rhs'], f"time {actual['time']!r} vs {expected['time']!r}; RHS {actual['rhs']} vs {expected['rhs']}"
    assert actual['rebuilds']==expected['rebuilds']
    assert actual['state'].keys()==expected['state'].keys()
    assert actual['forcingState']['parameters']==expected['forcingState']['parameters']
    for key in ('knots','generator'):
        torch.testing.assert_close(actual['forcingState'][key],expected['forcingState'][key],rtol=0,atol=0)
    for name,value in expected['state'].items():
        # Only unique-owner FP64 global source sums regroup additions. Declare
        # their tolerance in advance; particle physics and W stay bitwise.
        if name=='sourceEnergyBudget':
            torch.testing.assert_close(actual['state'][name],value,rtol=2e-12,atol=2e-12)
        else:
            torch.testing.assert_close(actual['state'][name],value,rtol=0,atol=0,msg=lambda m:name+': '+m)
    assert len(actual['records'])==len(expected['records'])
    for a,b in zip(actual['records'],expected['records']):
        for name in ('dvdt','dudt'):
            torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
        if 'reducing' not in b:
            continue
        assert a['reducing']==b['reducing']
        if b['delta'] is not None:
            torch.testing.assert_close(a['delta'],b['delta'],rtol=0,atol=0)
        assert (a['work'] is None)==(b['work'] is None)
        if b['work'] is not None:
            _,bw,bj=b['work'][0]
            for base,aw,aj in a['work']:
                torch.testing.assert_close(aw,bw[base:base+len(aw)],rtol=0,atol=0)
                torch.testing.assert_close(aj,bj[base:base+len(aj)],rtol=0,atol=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replica-checks',choices=('exact','contract'),default='exact')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--clock-interval',type=float,default=.125)
    parser.add_argument('--lanes',type=int,default=1)
    parser.add_argument('--output')
    parser.add_argument('--suite',choices=('basic','extended'),default='basic')
    args=parser.parse_args()
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
    setNeighborLanes(args.lanes)
    variants=[('Monaghan',2,'rungeKutta2','NoScheme',False),
        ('CRKSPH',2,'rungeKutta4','NoScheme',False),
        ('CompSPH',2,'rungeKutta4','NoScheme',False),
        ('CRKSPH',2,'rungeKutta4','NoScheme',True),
        ('CompSPH',2,'rungeKutta4','NoScheme',True)]
    if args.suite=='extended':
        variants += [('CRKSPH',3,'rungeKutta2','Owen',False),
                     ('CompSPH',3,'rungeKutta2','Owen',False),
                     ('Monaghan',2,'rungeKutta4','Monaghan',False)]
    if args.steps>2:
        variants=[('CRKSPH',2,'rungeKutta4','NoScheme',True),
                  ('CompSPH',2,'rungeKutta4','NoScheme',True)]
    # All flat oracle runs precede the process-group / distributed solver phase.
    references=[run(args,*variant) for variant in variants]
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo',timeout=timedelta(seconds=45))
    communication=RowChunkProcessGroup(dist.group.WORLD,args.device,replicaChecks=args.replica_checks)
    summaries=[]
    try:
        stream=torch.cuda.Stream(device=communication.device) if communication.device.type=='cuda' else None
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            for variant,reference in zip(variants,references):
                actual=run(args,*variant,communication=communication)
                def parity():
                    try:
                        compare(actual,reference)
                    except Exception:
                        if args.output:
                            Path(args.output).mkdir(parents=True,exist_ok=True)
                            torch.save(dict(actual=actual,reference=reference),Path(args.output)/f'failed-{variant[0]}-rank{communication.rank}.pt')
                        raise
                communication.localPhase('test:solver-parity',parity)
                summary={k:v for k,v in actual.items() if k not in ('state','records','forcingState')}
                summary.update(scheme=variant[0],dim=variant[1],tableau=variant[2],support=variant[3],forcedReplay=variant[4])
                summaries.append(summary)
                print(f'[distributed solver rank {communication.rank}] {variant}: passed {actual["rhs"]} RHS',flush=True)
        if stream is not None:
            stream.synchronize()
        summary=dict(status='passed',pid=os.getpid(),variants=summaries,lanes=args.lanes,**communication.summary())
        if args.device.startswith('cuda'):
            import ctypes
            version=ctypes.c_int()
            ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
            summary['runtimeNccl']=version.value
            summary['gpuUuid']=str(torch.cuda.get_device_properties(communication.device).uuid)
            summary['loadedNccl']=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                         if 'libnccl.so' in line})
        if args.output:
            Path(args.output).mkdir(parents=True,exist_ok=True)
            (Path(args.output)/f'solver-rank{communication.rank}.json').write_text(json.dumps(summary,indent=2))
        print(f'[distributed solver rank {communication.rank}] all solver checks passed',flush=True)
    finally:
        setNeighborLanes(None)
        dist.destroy_process_group()


if __name__=='__main__':
    main()
