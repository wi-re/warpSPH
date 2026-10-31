"""Native periodic delta process continuation and accepted allocation recovery."""
from datetime import timedelta
from dataclasses import replace
from functools import wraps
from pathlib import Path
import argparse,contextlib,importlib,json,os,sys
from unittest.mock import patch
import torch
import warp as wp
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_chunkedDelta import configuration,chunkedCase,nativeThreeDimensional
from warpSPH.runner import RunAllocationError
from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
from warpSPHCore.distributed import RowChunkProcessGroup,DistributedAllocationError


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu');p.add_argument('--scheme',default='deltaSPH')
    p.add_argument('--action',choices=('reference','save','resume','fail'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--oracle',required=True)
    p.add_argument('--rows',type=int,default=3);p.add_argument('--fail-site',choices=('rhs','shift'),default='shift')
    p.add_argument('--replica-checks',default='exact')
    args=p.parse_args();os.environ['WARPSPH_TEST_DEVICE']=args.device
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo',timeout=timedelta(seconds=60))
    wp.init()
    comm=RowChunkProcessGroup(dist.group.WORLD,args.device,replicaChecks=args.replica_checks)
    try:
        held={};stages=[]
        def accepted(ctx,system,completed):
            held['ctx']=ctx
            if completed==2:held['accepted']={k:v.clone() for k,v in vars(system.state).items() if isinstance(v,torch.Tensor)}
        case=chunkedCase(communication=comm,rows=args.rows,accepted=accepted)
        build=case.buildSystem
        def observedBuild(ctx):
            held['ctx']=ctx
            system=build(ctx)
            native=ctx.stepFunction
            @wraps(native)
            def observe(system,*a,**kw):
                result=native(system,*a,**kw)
                stages.append(dict(time=float(system.t),velocity=result[0].dvdt.detach().cpu(),density=result[0].drhodt.detach().cpu()))
                return result
            ctx.bundle=replace(ctx.bundle,stepFunction=observe)
            return system
        case=replace(case,buildSystem=observedBuild)
        cfg=configuration(args.scheme,steps=4 if args.action in ('reference','fail') else 2,
            resume=args.checkpoint if args.action=='resume' else None)
        cfg.simulation['exportRoot']=args.output
        dispatch=importlib.import_module('warpSPHCore.autograd.row_dispatch');launch=dispatch._launchLocal
        def fault(spec,*a,**kw):
            target='computeDeltaShift' if args.fail_site=='shift' else 'computeDensityDiffusionDeltaSPH'
            if args.action=='fail' and held['ctx'].scratch.get('completedSteps')==2 and comm.rank==comm.size-1 and target in spec.kernel.key:
                raise torch.OutOfMemoryError('injected distributed delta '+args.fail_site)
            return launch(spec,*a,**kw)
        stream=torch.cuda.Stream() if args.device.startswith('cuda') else None
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            if args.action=='save':
                nativeThreeDimensional(args.scheme,comm)
            with patch.object(dispatch,'_launchLocal',fault):
                try:result=cfg.run(case)
                except RunAllocationError as error:
                    if args.action!='fail':raise
                    assert isinstance(error.__cause__,DistributedAllocationError)
                    assert error.recoveryError is None,error.recoveryError
                    result=error.result
                    assert result.nSteps==2
                    for name,value in held['accepted'].items():
                        torch.testing.assert_close(getattr(result.state.state,name),value,rtol=0,atol=0,msg=name)
                else:assert args.action!='fail','expected allocation failure'
            if args.action in ('save','fail'):writeChunkedCheckpoint(args.checkpoint,result.ctx,result.state)
            data=dict(time=float(result.state.t),completed=result.ctx.scratch['completedSteps'],
                state={k:v.detach().cpu() for k,v in vars(result.state.state).items() if isinstance(v,torch.Tensor)},
                dt=torch.as_tensor(result.ctx.config.dt).detach().cpu(),stages=stages)
            if args.action=='reference':
                comm.localPhase('delta:test:oracle',lambda:torch.save(data,args.oracle) if comm.rank==0 else None)
            if args.action=='resume':
                expected=comm.localPhase('delta:test:read-oracle',lambda:torch.load(args.oracle,weights_only=True,map_location='cpu'))
                assert data['time']==expected['time'] and data['completed']==expected['completed']
                for name,value in expected['state'].items():torch.testing.assert_close(data['state'][name],value,rtol=0,atol=0,msg=name)
                torch.testing.assert_close(data['dt'],expected['dt'],rtol=0,atol=0)
                for a,b in zip(stages,expected['stages'][-len(stages):]):
                    assert a['time']==b['time']
                    for key in ('velocity','density'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,msg=key)
        if stream is not None:stream.synchronize()
        output=Path(args.output);output.mkdir(exist_ok=True,parents=True)
        report=dict(status='passed',scheme=args.scheme,action=args.action,ranks=comm.size,completed=data['completed'],
            time=data['time'],communication=comm.summary())
        if comm.device.type=='cuda':
            import ctypes
            version=ctypes.c_int();ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
            report.update(runtimeNccl=version.value,gpuUuid=str(torch.cuda.get_device_properties(comm.device).uuid))
        (output/f'rank-{comm.rank}.json').write_text(json.dumps(report,indent=2))
        print(f'delta rank {comm.rank} {args.scheme} {args.action}: passed',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
