"""Accepted periodic delta checkpoints: density/history, no frozen RHS or pairs."""
import json
import math
import os
from pathlib import Path
import tempfile

import h5py
import torch
from warpSPHCore.radiusSearch import ChunkedAdjacency,rowChunkCheckpointState,restoreRowChunkCheckpointState
from .hdf5 import dumpState,loadState
from .chunkCheckpoint import _local,_json,_fileDigest,_writeScalar,_readScalar


def _scope(ctx,system):
    from ..distributed import CHUNKED_WEAKLY_SCHEMES
    from ..modules.deltaSPH.chunked import checkDeltaChunks
    if (ctx.case.name!='tgv-wc' or ctx.spec.scheme not in CHUNKED_WEAKLY_SCHEMES
        or ctx.scratch.get('execution')!='eager' or not isinstance(system.adjacency,ChunkedAdjacency)
        or system.adjacency.kind!='verlet'):
        raise NotImplementedError('Delta checkpoints require eager periodic tgv-wc with resident chunks.')
    checkDeltaChunks(system,ctx.config,ctx.schemeConfig)
    system.adjacency.assertStorageValid()


def _signature(ctx):
    return _json(dict(case=ctx.case.name,parameters=ctx.spec.params,solver=ctx.exportFunction(ctx.schemeConfig),
        scheme=ctx.spec.scheme,precision=str(ctx.dtype),integrator=ctx.integrator.identifier.name,
        domain={name:getattr(ctx.spec,name) for name in ('nx','dim','L','periodic','kernel','n_h','samplingScheme')},
        timestep={name:getattr(ctx.config,name) for name in ('adaptiveDt','minDt','maxDt','dtGrowthFactor')}))


def _tensors(state):
    return {k:v for k,v in vars(state).items() if isinstance(v,torch.Tensor)}


def _validate(state):
    values=_tensors(state)
    if any(t.requires_grad or torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in values.values()):
        raise NotImplementedError('Delta checkpoints do not support AD.')
    if not all(bool(torch.isfinite(t).all()) for t in values.values()) or bool((state.densities<=0).any()):
        raise ValueError('Delta checkpoint requires finite accepted fields and positive density.')


def writeCheckpoint(path,ctx,system,*,overwrite=False):
    comm=system.adjacency.communication
    path=Path(path).expanduser().resolve()
    def validate():
        _scope(ctx,system)
        token=ctx.scratch.get('_rowChunkAccepted')
        if token is None or token[0]() is not system or token[2]!=float(system.t) or token[1]<1:
            raise ValueError('Delta checkpoint requires a runner-marked accepted-step boundary.')
        if type(token[1]) is not int or ctx.scratch.get('completedSteps')!=token[1]:
            raise ValueError('Delta checkpoint completed-step context is inconsistent.')
        if (not math.isfinite(float(ctx.config.dt)) or float(ctx.config.dt)<=0
            or not math.isfinite(float(system.t)) or float(system.t)<0):
            raise ValueError('Delta checkpoint requires a finite nonnegative time and positive next timestep.')
        if type(overwrite) is not bool:raise TypeError('Checkpoint overwrite must be a bool.')
        _validate(system.state)
        return token[1],_signature(ctx)
    completed,signature=_local(comm,'delta:checkpoint:validate',validate)
    recipe=_local(comm,'delta:checkpoint:history',lambda:rowChunkCheckpointState(system.adjacency))
    history=getattr(system.deltaStages,'surfaceStart',None)
    stats=getattr(system.deltaStages,'surfaceStats',None)
    if comm is not None:
        comm.agree('delta:checkpoint:contract',(str(path),completed,signature,recipe['checksum'],overwrite))
        comm.assertReplicated('delta:checkpoint:state',_tensors(system.state),exact=True)
        comm.assertReplicated('delta:checkpoint:clock',dict(time=torch.as_tensor(system.t,device=comm.device),
            dt=torch.as_tensor(ctx.config.dt,device=comm.device)),exact=True)
        comm.assertReplicated('delta:checkpoint:surface',{} if history is None else dict(surfaceStart=history),exact=True)
        comm.agree('delta:checkpoint:surface-stats',stats)
    def publish():
        if comm is not None and comm.rank!=0:return
        if path.exists() and not overwrite:raise FileExistsError(f'Checkpoint already exists: {path}')
        path.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
        os.close(descriptor);temporary=Path(temporary)
        try:
            with h5py.File(temporary,'w') as file:
                file.attrs.update(deltaRowChunkCheckpointVersion=1,complete=1,physicalSignature=signature,
                    completedSteps=completed,time=float(system.t),stateClass=type(system.state).__module__+'.'+type(system.state).__qualname__)
                dumpState(system.state,file.create_group('state'))
                search=file.create_group('neighborHistory')
                search.attrs.update(metadata=_json(recipe['metadata']),checksum=recipe['checksum'])
                for name,value in recipe['tensors'].items():search.create_dataset(name,data=value.numpy())
                clock=file.create_group('clock')
                _writeScalar(clock,'time',system.t);_writeScalar(clock,'nextDt',ctx.config.dt)
                surface=file.create_group('surfaceHistory')
                surface.attrs['statistics']=_json(stats)
                if history is not None:surface.create_dataset('start',data=history.detach().cpu().numpy())
                file.attrs['contentChecksum']=_fileDigest(file);file.flush()
            with temporary.open('rb') as file:os.fsync(file.fileno())
            if overwrite:os.replace(temporary,path)
            else:os.link(temporary,path);temporary.unlink()
            descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        finally:temporary.unlink(missing_ok=True)
    _local(comm,'delta:checkpoint:publish',publish)
    return str(path)


def restoreCheckpoint(path,ctx,system):
    prior=system.adjacency;comm=prior.communication;device=system.state.positions.device
    path=Path(path).expanduser().resolve()
    _scope(ctx,system)
    signature=_local(comm,'delta:checkpoint:signature',lambda:_signature(ctx))
    if comm is not None:comm.agree('delta:checkpoint:read-path',(str(path),signature))
    def read():
        with h5py.File(path,'r') as file:
            if file.attrs.get('deltaRowChunkCheckpointVersion')!=1 or file.attrs.get('complete')!=1:
                raise ValueError('Unsupported or incomplete delta checkpoint.')
            if file.attrs.get('physicalSignature')!=signature:raise ValueError('Delta checkpoint configuration is incompatible.')
            if file.attrs.get('stateClass')!=type(system.state).__module__+'.'+type(system.state).__qualname__:
                raise ValueError('Delta checkpoint state class is incompatible.')
            if set(file.keys())!={'state','neighborHistory','clock','surfaceHistory'}:raise ValueError('Incomplete delta checkpoint groups.')
            if file.attrs.get('contentChecksum')!=_fileDigest(file):raise ValueError('Delta checkpoint content checksum mismatch.')
            completed=int(file.attrs['completedSteps'])
            if completed<1 or ctx.spec.resumeStepOffset not in (0,completed):raise ValueError('Delta checkpoint step offset is incompatible.')
            time=_readScalar(file['clock/time'],device);dt=_readScalar(file['clock/nextDt'],device)
            if (float(time)<0 or float(dt)<=0 or not math.isfinite(float(time)) or not math.isfinite(float(dt))
                or (ctx.spec.nSteps is None and ctx.spec.tLimit<=float(time))):raise ValueError('Invalid delta checkpoint time/endpoint.')
            state=loadState(file['state'],device,type(system.state));state.UIDcounter=int(state.UIDcounter);_validate(state)
            if state.positions.shape!=system.state.positions.shape:raise ValueError('Delta checkpoint particle shape is incompatible.')
            group=file['neighborHistory']
            recipe=dict(metadata=json.loads(group.attrs['metadata']),checksum=group.attrs['checksum'],
                tensors={name:torch.from_numpy(ds[:]) for name,ds in group.items()})
            surface=file['surfaceHistory']
            history=torch.from_numpy(surface['start'][:]).to(device) if 'start' in surface else None
            if history is not None and (history.shape!=state.densities.shape or history.dtype!=torch.int32):
                raise ValueError('Invalid delta surface history layout.')
            stats=json.loads(surface.attrs['statistics'])
            return state,recipe,time,dt,completed,history,stats,file.attrs['contentChecksum']
    state,recipe,time,dt,completed,history,stats,checksum=_local(comm,'delta:checkpoint:read',read)
    if comm is not None:comm.agree('delta:checkpoint:read-content',(checksum,completed))
    adjacency=restoreRowChunkCheckpointState(recipe,device=device,policy=prior.plan.policy,communication=comm,storeI=prior.storeI)
    def install():
        from ..modules.deltaSPH.stages import DeltaStages
        system.state=state;system.adjacency=adjacency;system.t=time
        # Stage0 of the next real step computes fresh diffusion. Preserve only
        # accepted mask history used by that next step's surface diagnostics.
        system.deltaStages=DeltaStages(surfaceStart=history,surfaceStats=stats)
        if stats is not None:ctx.schemeConfig._surfaceMaskStats=stats
        ctx.config.dt=dt;ctx.spec.resumeStepOffset=completed
        ctx.scratch.update(completedSteps=completed,_rowChunkRestartVerified=True)
        ctx.scratch.pop('_rowChunkAccepted',None)
    _local(comm,'delta:checkpoint:install',install)
    return str(path)
