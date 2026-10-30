"""Explicit eager driven-SPH launcher/output adapter for owned row chunks.

The caller owns the initialized process group and device selection. Defaults of
the ordinary runner are unchanged. MPI/ROCm and physical site setup are separate.
"""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch

from warpSPHCore import get_torch_offset_dtype, get_torch_precision
from warpSPHCore.radiusSearch import RowChunkPolicy, buildChunkedVerletList
from .runner import RunConfiguration
from .io.chunkCheckpoint import writeChunkedCheckpoint


def _json(value):
    return json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'


def _atomicJson(path,value):
    temporary=path.with_name('.'+path.name+'.tmp')
    try:
        temporary.write_text(_json(value))
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def _sourcePins():
    import warpSPH,warpSPHCore,warpSPHIntegrators
    pins={}
    for module in (warpSPH,warpSPHCore,warpSPHIntegrators):
        package=Path(module.__file__).resolve().parent
        files={str(p.relative_to(package)):sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*.py'))}
        digest=sha256(_json(files).encode()).hexdigest()
        result=subprocess.run(['git','-C',str(package),'rev-parse','HEAD'],capture_output=True,text=True)
        pins[module.__name__]=dict(package=str(package),commit=result.stdout.strip() if result.returncode==0 else None,
                                  digest=digest,files=files)
    return pins


def _runtime(communication):
    import warp
    data=dict(rank=communication.rank,ranks=communication.size,backend=communication.backend,
        device=str(communication.device),host=socket.gethostname(),pid=os.getpid(),torch=torch.__version__,
        warp=warp.__version__,precision=str(get_torch_precision()),offsetDtype=str(get_torch_offset_dtype()))
    entry=getattr(sys.modules.get('__main__'),'__file__',None)
    if entry is not None and Path(entry).is_file():
        entry=Path(entry).resolve()
        data['entrypoint']=dict(path=str(entry),sha256=sha256(entry.read_bytes()).hexdigest())
    if communication.device.type=='cuda':
        import ctypes
        version=ctypes.c_int()
        library=ctypes.CDLL('libnccl.so.2')
        if library.ncclGetVersion(ctypes.byref(version))!=0:
            raise RuntimeError('Could not query loaded NCCL runtime.')
        data.update(cuda=torch.version.cuda,ncclRuntime=version.value,
            gpuUuid=str(torch.cuda.get_device_properties(communication.device).uuid))
        maps=Path('/proc/self/maps')
        if maps.exists():
            data['ncclLibraries']=sorted({s.split()[-1] for s in maps.read_text().splitlines() if 'libnccl.so' in s})
    return data


def runDistributed(configuration,communication,*,outputDirectory,edgeBudget,maxRows=None,checkpointEvery=0):
    """Collectively run an explicit configuration; publish diagnostics on rank 0.

    Output must be a fresh directory visible to every rank. Periodic and final
    checkpoints use accepted-state native files; each rank writes its own runtime
    report. All live peers receive local setup/IO failures through the group.
    """
    from .cases.compressible.drivenTurbulence import drivenTurbulenceCase,buildSPHSystem
    output=Path(outputDirectory).expanduser().resolve()
    def validate():
        if not isinstance(configuration,RunConfiguration) or configuration.case!='drivenTurbulence':
            raise ValueError('Distributed launcher currently supports drivenTurbulence only.')
        spec=configuration.resolve(drivenTurbulenceCase)
        if (spec.scheme not in ('Monaghan','CRKSPH','CompSPH') or spec.execution!='eager'
            or spec.cudaGraph or spec.pipelineOutputs or spec.store or spec.plot or spec.video):
            raise ValueError('Distributed launcher requires eager Monaghan/CRKSPH/CompSPH with automatic store/plot/video disabled.')
        if type(checkpointEvery) is not int or checkpointEvery<0:
            raise ValueError('checkpointEvery must be a nonnegative integer.')
        RowChunkPolicy(edgeBudget=edgeBudget,maxRows=maxRows)
        if communication.device.type=='cuda' and torch.version.hip is not None:
            raise NotImplementedError('ROCm launcher qualification is deferred.')
        return configuration.toDict()
    document=communication.localPhase('launcher:validate',validate)
    contract=json.loads(json.dumps(document))
    contract['simulation'].pop('device',None)
    communication.agree('launcher:configuration',(contract,str(output),edgeBudget,maxRows,checkpointEvery))
    def create():
        if communication.rank==0:output.mkdir(parents=True,exist_ok=False)
    communication.localPhase('launcher:output-directory',create)
    started=time.perf_counter()
    manifest=None
    try:
        pins=communication.localPhase('launcher:source-pins',_sourcePins)
        communication.agree('launcher:sources',{name:pin['digest'] for name,pin in pins.items()})
        runtime=communication.localPhase('launcher:runtime',lambda:_runtime(communication))
        communication.agree('launcher:stack',tuple(runtime[k] for k in ('torch','warp','precision','offsetDtype')))
        communication.localPhase('launcher:rank-start',lambda:_atomicJson(output/f'rank-{communication.rank:05d}.json',
            dict(status='running',runtime=runtime)))
        def publishStart():
            nonlocal manifest
            if communication.rank!=0:return
            placement=[json.loads((output/f'rank-{rank:05d}.json').read_text())['runtime'] for rank in range(communication.size)]
            manifest=dict(formatVersion=1,status='running',configuration=document,sources=pins,placement=placement,
                chunks=dict(edgeBudget=edgeBudget,maxRows=maxRows),checkpointEvery=checkpointEvery,
                timingPolicy='diagnostic; physical topology/capacity/performance require separate qualification')
            _atomicJson(output/'manifest.json',manifest)
        communication.localPhase('launcher:manifest-start',publishStart)
        configured=RunConfiguration.fromDict(document)
        configured.simulation.update(device=str(communication.device),quiet=True,progress=communication.rank==0,storeMode='trajectory')
        base=drivenTurbulenceCase
        def build(ctx):
            def factory(state,config):
                compatible=ctx.spec.scheme in ('CRKSPH','CompSPH')
                dim=state.positions.shape[1] if compatible else 1
                policy=RowChunkPolicy(edgeBudget=edgeBudget,maxRows=maxRows,
                    bytesPerEdge=4+(dim*state.positions.element_size() if compatible else 0),maxElementsPerEdge=dim)
                return buildChunkedVerletList(state,config.domain,policy=policy,communication=communication,
                    supportMode=config.supportMode,verletScale=config.verletScale)
            return buildSPHSystem(ctx,neighborhoodFactory=factory)
        def accepted(ctx,system,completed):
            if checkpointEvery and completed%checkpointEvery==0:
                writeChunkedCheckpoint(output/f'checkpoint-{completed:08d}.h5',ctx,system)
        def diagnostics(ctx,system):
            values=communication.localPhase('launcher:diagnostics',lambda:base.diagnostics(ctx,system))
            def publish():
                if communication.rank==0:
                    row=dict(completed=ctx.scratch['completedSteps'],time=float(system.t),**values)
                    with (output/'diagnostics.jsonl').open('a') as file:file.write(json.dumps(row,allow_nan=False)+'\n')
            communication.localPhase('launcher:diagnostics-output',publish)
            return values
        case=replace(base,buildSystem=build,acceptedStep=accepted,diagnostics=diagnostics)
        result=configured.run(case)
        if not result.diverged and result.nSteps>0:
            writeChunkedCheckpoint(output/'finalState.h5',result.ctx,result.state)
        report=dict(status='failed' if result.diverged else 'complete',runtime=runtime,
            steps=result.nSteps,completed=result.ctx.scratch['completedSteps'],time=float(result.state.t),
            wallSeconds=time.perf_counter()-started,runWallSeconds=result.wallTime,
            logicalMemory=result.state.adjacency.memoryUsage(),communication=communication.summary(),stopReason=result.stopReason)
        if communication.device.type=='cuda':
            report['torchAllocator']=dict(allocated=torch.cuda.memory_allocated(communication.device),
                reserved=torch.cuda.memory_reserved(communication.device))
        communication.agree('launcher:result',(report['status'],report['steps'],report['completed'],report['time']))
        communication.localPhase('launcher:rank-final',lambda:_atomicJson(output/f'rank-{communication.rank:05d}.json',report))
        def finish():
            if communication.rank==0:
                manifest.update(status=report['status'],steps=report['steps'],completed=report['completed'],time=report['time'])
                _atomicJson(output/'manifest.json',manifest)
        communication.localPhase('launcher:manifest-final',finish)
        if result.diverged:raise RuntimeError('Distributed run stopped: '+str(result.stopReason))
        return result
    except Exception as error:
        # Do not enter new collectives here: a hard peer failure may have left
        # other ranks inside c10d. The finite timeout/launcher terminates peers.
        try:
            _atomicJson(output/f'failure-{communication.rank:05d}.json',dict(status='failed',rank=communication.rank,
                error=type(error).__name__+': '+str(error)))
            if communication.rank==0 and manifest is not None:
                manifest.update(status='failed',error=type(error).__name__+': '+str(error))
                _atomicJson(output/'manifest.json',manifest)
        except OSError:pass
        raise
