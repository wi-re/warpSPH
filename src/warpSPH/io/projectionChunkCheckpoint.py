"""Accepted projection checkpoints: native pressure histories and frozen search, no pairs."""
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
    from ..distributed import CHUNKED_PROJECTION_SCHEMES
    from ..modules.incompressible.chunked import checkProjectionChunks
    if (ctx.case.name != 'tgv' or ctx.spec.scheme not in CHUNKED_PROJECTION_SCHEMES
        or ctx.scratch.get('execution') != 'eager' or not isinstance(system.adjacency, ChunkedAdjacency)
        or system.adjacency.kind != 'verlet'):
        raise NotImplementedError('Projection checkpoints require eager periodic tgv with resident chunks.')
    checkProjectionChunks(system, ctx.config, ctx.schemeConfig)
    system.adjacency.assertStorageValid()


def _signature(ctx):
    return _json(dict(case=ctx.case.name,parameters=ctx.spec.params,solver=ctx.exportFunction(ctx.schemeConfig),
        scheme=ctx.spec.scheme,precision=str(ctx.dtype),integrator=ctx.integrator.identifier.name,
        domain={name:getattr(ctx.spec,name) for name in ('nx','dim','L','periodic','kernel','n_h','samplingScheme')},
        timestep={name:getattr(ctx.config,name) for name in ('adaptiveDt','minDt','maxDt','dtGrowthFactor','cflFactor')}))


def _tensors(state):
    return {k:v for k,v in vars(state).items() if isinstance(v,torch.Tensor)}


def _validate(state):
    values=_tensors(state)
    if any(t.requires_grad or torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in values.values()):
        raise NotImplementedError('Projection checkpoints do not support AD.')
    if not all(bool(torch.isfinite(t).all()) for t in values.values()) or bool((state.densities<=0).any()):
        raise ValueError('Projection checkpoint requires finite accepted fields and positive density.')


def writeCheckpoint(path,ctx,system,*,overwrite=False):
    comm=system.adjacency.communication
    path=Path(path).expanduser().resolve()
    def validate():
        _scope(ctx,system)
        token=ctx.scratch.get('_rowChunkAccepted')
        if token is None or token[0]() is not system or token[2]!=float(system.t) or token[1]<1:
            raise ValueError('Projection checkpoint requires a runner-marked accepted-step boundary.')
        if type(token[1]) is not int or ctx.scratch.get('completedSteps')!=token[1]:
            raise ValueError('Projection checkpoint completed-step context is inconsistent.')
        if (not math.isfinite(float(ctx.config.dt)) or float(ctx.config.dt)<=0
            or not math.isfinite(float(system.t)) or float(system.t)<0):
            raise ValueError('Projection checkpoint requires a finite nonnegative time and positive next timestep.')
        nominal=ctx.scratch.get('projectionNominalDt',0)
        if not math.isfinite(float(nominal)) or float(nominal)<=0:raise ValueError('Invalid projection nominal timestep.')
        if type(overwrite) is not bool:raise TypeError('Checkpoint overwrite must be a bool.')
        _validate(system.state)
        return token[1],_signature(ctx)
    completed,signature=_local(comm,'projection:checkpoint:validate',validate)
    recipe=_local(comm,'projection:checkpoint:history',lambda:rowChunkCheckpointState(system.adjacency))
    if comm is not None:
        comm.agree('projection:checkpoint:contract',(str(path),completed,signature,recipe['checksum'],overwrite,
            float(system.t).hex(),float(ctx.config.dt).hex(),float(ctx.scratch['projectionNominalDt']).hex()))
        comm.assertReplicated('projection:checkpoint:state',_tensors(system.state),exact=True)
        comm.assertReplicated('projection:checkpoint:clock',dict(time=torch.as_tensor(system.t,device=comm.device),
            dt=torch.as_tensor(ctx.config.dt,device=comm.device),
            nominalDt=torch.as_tensor(ctx.scratch['projectionNominalDt'],device=comm.device)),exact=True)
    def publish():
        if comm is not None and comm.rank!=0:return
        if path.exists() and not overwrite:raise FileExistsError(f'Checkpoint already exists: {path}')
        path.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
        os.close(descriptor);temporary=Path(temporary)
        try:
            with h5py.File(temporary,'w') as file:
                file.attrs.update(projectionRowChunkCheckpointVersion=1,complete=1,physicalSignature=signature,
                    completedSteps=completed,time=float(system.t),stateClass=type(system.state).__module__+'.'+type(system.state).__qualname__)
                dumpState(system.state,file.create_group('state'))
                search=file.create_group('neighborHistory')
                search.attrs.update(metadata=_json(recipe['metadata']),checksum=recipe['checksum'])
                for name,value in recipe['tensors'].items():search.create_dataset(name,data=value.numpy())
                clock=file.create_group('clock')
                _writeScalar(clock,'time',system.t);_writeScalar(clock,'nextDt',ctx.config.dt)
                context=file.create_group('context')
                _writeScalar(context,'nominalDt',ctx.scratch['projectionNominalDt'])
                file.attrs['contentChecksum']=_fileDigest(file);file.flush()
            with temporary.open('rb') as file:os.fsync(file.fileno())
            if overwrite:os.replace(temporary,path)
            else:os.link(temporary,path);temporary.unlink()
            descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        finally:temporary.unlink(missing_ok=True)
    _local(comm,'projection:checkpoint:publish',publish)
    return str(path)


def restoreCheckpoint(path,ctx,system):
    prior=system.adjacency;comm=prior.communication;device=system.state.positions.device
    path=Path(path).expanduser().resolve()
    _scope(ctx,system)
    signature=_local(comm,'projection:checkpoint:signature',lambda:_signature(ctx))
    if comm is not None:comm.agree('projection:checkpoint:read-path',(str(path),signature))
    def read():
        with h5py.File(path,'r') as file:
            if file.attrs.get('projectionRowChunkCheckpointVersion')!=1 or file.attrs.get('complete')!=1:
                raise ValueError('Unsupported or incomplete projection checkpoint.')
            if file.attrs.get('physicalSignature')!=signature:raise ValueError('Projection checkpoint configuration is incompatible.')
            if file.attrs.get('stateClass')!=type(system.state).__module__+'.'+type(system.state).__qualname__:
                raise ValueError('Projection checkpoint state class is incompatible.')
            if set(file.keys())!={'state','neighborHistory','clock','context'}:raise ValueError('Incomplete projection checkpoint groups.')
            if file.attrs.get('contentChecksum')!=_fileDigest(file):raise ValueError('Projection checkpoint content checksum mismatch.')
            completed=int(file.attrs['completedSteps'])
            if completed<1 or ctx.spec.resumeStepOffset not in (0,completed):raise ValueError('Projection checkpoint step offset is incompatible.')
            time=_readScalar(file['clock/time'],device);dt=_readScalar(file['clock/nextDt'],device)
            if (float(time)<0 or float(dt)<=0 or not math.isfinite(float(time)) or not math.isfinite(float(dt))
                or (ctx.spec.nSteps is None and ctx.spec.tLimit<=float(time))):raise ValueError('Invalid projection checkpoint time/endpoint.')
            state=loadState(file['state'],device,type(system.state));state.UIDcounter=int(state.UIDcounter);_validate(state)
            if state.positions.shape!=system.state.positions.shape:raise ValueError('Projection checkpoint particle shape is incompatible.')
            group=file['neighborHistory']
            recipe=dict(metadata=json.loads(group.attrs['metadata']),checksum=group.attrs['checksum'],
                tensors={name:torch.from_numpy(ds[:]) for name,ds in group.items()})
            nominal=_readScalar(file['context/nominalDt'],device)
            if not math.isfinite(float(nominal)) or float(nominal)<=0:raise ValueError('Invalid projection nominal timestep.')
            return state,recipe,time,dt,completed,nominal,file.attrs['contentChecksum']
    state,recipe,time,dt,completed,nominal,checksum=_local(comm,'projection:checkpoint:read',read)
    if comm is not None:comm.agree('projection:checkpoint:read-content',(checksum,completed))
    from ..configurations.compute import neighborSearchOptions
    # The next native RHS, rather than the disposable seed, owns this storage
    # contract. A mismatch would force a fresh spatial search and lose saved J
    # reduction order even though particle values and geometry were exact.
    adjacency=restoreRowChunkCheckpointState(recipe,device=device,policy=prior.plan.policy,
        communication=comm,storeI=neighborSearchOptions(ctx.config)['storeI'])
    def install():
        # All native physical pressures and timestep-scaled DF/CD impulses
        # belong to state and were loaded exactly, with their optionality intact.
        system.state=state;system.adjacency=adjacency;system.t=time
        ctx.config.dt=dt;ctx.spec.resumeStepOffset=completed
        ctx.scratch.update(projectionNominalDt=nominal,completedSteps=completed,_rowChunkRestartVerified=True)
        ctx.scratch.pop('_rowChunkAccepted',None)
    _local(comm,'projection:checkpoint:install',install)
    return str(path)
