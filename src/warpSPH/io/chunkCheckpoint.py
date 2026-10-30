"""Accepted-step native SPH checkpoints with no serialized pair arrays.

Explicit Python API for the bounded driven Monaghan/CRKSPH/CompSPH path. One
rank publishes an atomic canonical file; every rank reconstructs only its own
adjacency on restore. The file contains no communicator, device ordinal or W.
"""
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import weakref

import h5py
import numpy as np
import torch

from warpSPHCore.radiusSearch import (ChunkedAdjacency, rowChunkCheckpointState,
                                     restoreRowChunkCheckpointState)
from .hdf5 import dumpState, loadState

__all__ = ['writeChunkedCheckpoint', 'restoreChunkedCheckpoint']


def markCheckpointBoundary(ctx, system, completed):
    """Weak identity token: do not retain the previous system's pair buffers."""
    if isinstance(getattr(system,'adjacency',None), ChunkedAdjacency):
        ctx.scratch['_rowChunkAccepted'] = (weakref.ref(system), completed, float(system.t))


def _local(communication, tag, function):
    return function() if communication is None else communication.localPhase(tag,function)


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _plain(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,bytes):
        return value.decode()
    return value


def _fileDigest(file):
    digest=sha256()
    def visit(name,obj):
        attrs={key:_plain(value) for key,value in obj.attrs.items() if key!='contentChecksum'}
        digest.update(_json((name,attrs)).encode())
        if isinstance(obj,h5py.Dataset):
            data=np.asarray(obj[()])
            digest.update(_json((data.dtype.str,list(data.shape))).encode())
            digest.update(data.tobytes(order='C'))
    visit('',file)
    file.visititems(visit)
    return digest.hexdigest()


def _signature(ctx):
    from ..cases.compressible.drivenTurbulence import checkpointSignature
    return checkpointSignature(ctx)


def _stateTensors(state):
    return {k:v for k,v in vars(state).items() if isinstance(v,torch.Tensor)}


def _validateState(state):
    tensors=_stateTensors(state)
    if any(t.requires_grad or torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in tensors.values()):
        raise NotImplementedError('Chunk checkpoints do not support AD.')
    if any(getattr(state,k,None) is not None for k in ('ap_ij','av_ij','f_ij')):
        raise ValueError('Checkpoint state still contains per-edge stage arrays.')
    if not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise ValueError('Checkpoint particle state must be finite.')
    if any(bool((getattr(state,k)<=0).any()) for k in ('densities','internalEnergies','pressures')):
        raise ValueError('Checkpoint requires positive accepted thermodynamics.')
    if getattr(state,'sourceEnergyBudget',None) is None or state.sourceEnergyBudget.shape!=(2,):
        raise ValueError('Checkpoint requires the accepted two-component source budget.')


def _scope(ctx,system):
    from ..modules.meshless.sources import ConservedSources
    from ..modules.forcing import FourierOU
    if (ctx.spec.caseName!='drivenTurbulence' or getattr(ctx.scheme,'name',ctx.scheme) not in ('Monaghan','CRKSPH','CompSPH')
        or ctx.scratch.get('execution')!='eager' or not isinstance(system.adjacency,ChunkedAdjacency)
        or system.adjacency.kind!='verlet'):
        raise NotImplementedError('Chunk checkpoints currently qualify eager driven Monaghan/CRKSPH/CompSPH with resident chunks.')
    if not isinstance(system.sourceTerms,ConservedSources) or not isinstance(system.sourceTerms.acceleration,FourierOU):
        raise NotImplementedError('Chunk checkpoints currently require ConservedSources/FourierOU.')
    if getattr(system,'rerunRequest',None) is not None:
        raise ValueError('Cannot checkpoint an unaccepted compatible-work attempt.')
    system.adjacency.assertStorageValid()


def _writeScalar(group,name,value):
    tensor=isinstance(value,torch.Tensor)
    data=value.detach().cpu().numpy() if tensor else np.asarray(value,dtype=np.float64)
    if np.asarray(data).size!=1 or np.asarray(data).dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError(f'Checkpoint {name} must be a floating scalar.')
    if not math.isfinite(float(np.asarray(data).reshape(()))):
        raise ValueError(f'Checkpoint {name} must be finite.')
    dataset=group.create_dataset(name,data=data)
    dataset.attrs['representation']='tensor' if tensor else 'python'


def _readScalar(dataset,device):
    data=np.asarray(dataset[()])
    if data.size!=1 or data.dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError('Invalid checkpoint scalar layout.')
    kind=dataset.attrs['representation']
    if kind=='python':
        return float(data.reshape(()))
    if kind!='tensor':
        raise ValueError('Unknown checkpoint scalar representation.')
    return torch.from_numpy(data.copy()).to(device)


def writeChunkedCheckpoint(path,ctx,system,*,overwrite=False):
    """Collectively save an accepted result, e.g. after RunConfiguration.run().

    All ranks call with the same path. Default publication never overwrites an
    existing file. A failed write leaves no partially published checkpoint.
    Automatic output cadence is a separate launcher/output integration step.
    """
    communication=getattr(getattr(system,'adjacency',None),'communication',None)
    path=Path(path).expanduser().resolve()
    def validate():
        _scope(ctx,system)
        token=ctx.scratch.get('_rowChunkAccepted')
        if token is None or token[0]() is not system or token[2]!=float(system.t):
            raise ValueError('Checkpoint requires a runner-marked accepted-step boundary.')
        if type(overwrite) is not bool:
            raise TypeError('Checkpoint overwrite must be a bool.')
        completed=token[1]
        if type(completed) is not int or completed<1 or ctx.scratch['completedSteps']!=completed:
            raise ValueError('Checkpoint completed-step context is inconsistent.')
        _validateState(system.state)
        if not math.isfinite(float(ctx.config.dt)) or float(ctx.config.dt)<=0:
            raise ValueError('Checkpoint next timestep must be positive and finite.')
        return completed,_signature(ctx)
    completed,signature=_local(communication,'checkpoint:write-validate',validate)
    recipe=_local(communication,'checkpoint:write-history',lambda:rowChunkCheckpointState(system.adjacency))
    from ..distributed import validateSourceClock
    if communication is not None:
        communication.agree('checkpoint:write-contract',(str(path),completed,signature,recipe['checksum'],overwrite))
        communication.assertReplicated('checkpoint:write-state',_stateTensors(system.state))
        communication.assertReplicated('checkpoint:write-dt',dict(dt=torch.as_tensor(ctx.config.dt,device=communication.device)))
        validateSourceClock(system,communication)
    def write():
        if communication is not None and communication.rank!=0:
            return
        if path.exists() and not overwrite:
            raise FileExistsError(f'Checkpoint already exists: {path}')
        path.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
        os.close(descriptor)
        temporary=Path(temporary)
        try:
            with h5py.File(temporary,'w') as file:
                file.attrs.update(rowChunkCheckpointVersion=1,complete=1,physicalSignature=signature,
                    stateClass=type(system.state).__module__+'.'+type(system.state).__qualname__,
                    completedSteps=completed,time=float(system.t),sourceRanks=system.adjacency.plan.numOwners)
                dumpState(system.state,file.create_group('state'))
                history=file.create_group('neighborHistory')
                history.attrs.update(metadata=_json(recipe['metadata']),checksum=recipe['checksum'])
                for name,tensor in recipe['tensors'].items():
                    history.create_dataset(name,data=tensor.numpy())
                source=system.sourceTerms.acceleration.state_dict()
                forcing=file.create_group('forcing')
                forcing.attrs['parameters']=_json(source['parameters'])
                forcing.create_dataset('knots',data=source['knots'].numpy())
                forcing.create_dataset('generator',data=source['generator'].numpy())
                clock=file.create_group('clock')
                _writeScalar(clock,'time',system.t)
                _writeScalar(clock,'nextDt',ctx.config.dt)
                context=file.create_group('context')
                for name in ('initialMass','initialEnergy'):
                    context.create_dataset(name,data=torch.as_tensor(ctx.scratch[name],dtype=torch.float64).detach().cpu().numpy())
                file.attrs['contentChecksum']=_fileDigest(file)
                file.flush()
            with temporary.open('rb') as file:
                os.fsync(file.fileno())
            if overwrite:
                os.replace(temporary,path)
            else:
                os.link(temporary,path)
                temporary.unlink()
            directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
    _local(communication,'checkpoint:publish',write)
    return str(path)


def restoreChunkedCheckpoint(path,ctx,system):
    """Collectively restore into an explicitly seeded target chunk policy/group."""
    prior=system.adjacency
    communication=prior.communication
    path=Path(path).expanduser().resolve()
    def validateTarget():
        _scope(ctx,system)
        return _signature(ctx)
    signature=_local(communication,'checkpoint:target',validateTarget)
    if communication is not None:
        communication.agree('checkpoint:read-path',(str(path),signature))
    device=system.state.positions.device
    def read():
        from ..modules.forcing import FourierOU
        with h5py.File(path,'r') as file:
            if 'rowChunkCheckpointVersion' not in file.attrs:
                from .legacyChunkCheckpoint import readLegacyCheckpoint
                return readLegacyCheckpoint(file,ctx,system,signature,device)
            if file.attrs.get('rowChunkCheckpointVersion')!=1 or file.attrs.get('complete')!=1:
                raise ValueError('Unsupported or incomplete native chunk checkpoint.')
            if file.attrs['physicalSignature']!=signature:
                raise ValueError('Checkpoint physical/numerical configuration is incompatible.')
            if file.attrs['stateClass']!=type(system.state).__module__+'.'+type(system.state).__qualname__:
                raise ValueError('Checkpoint particle-state class is incompatible.')
            if set(file.keys())!={'state','neighborHistory','forcing','clock','context'}:
                raise ValueError('Incomplete checkpoint groups.')
            if file.attrs.get('contentChecksum')!=_fileDigest(file):
                raise ValueError('Chunk checkpoint content checksum mismatch.')
            completed=int(file.attrs['completedSteps'])
            if completed<1 or ctx.spec.resumeStepOffset not in (0,completed):
                raise ValueError('Checkpoint completed-step offset is incompatible.')
            time=_readScalar(file['clock/time'],device)
            dt=_readScalar(file['clock/nextDt'],device)
            if (not math.isfinite(float(time)) or float(time)<0 or not math.isfinite(float(dt)) or float(dt)<=0
                or (ctx.spec.nSteps is None and ctx.spec.tLimit<=float(time))):
                raise ValueError('Checkpoint time/timestep or requested endpoint is invalid.')
            state=loadState(file['state'],device,type(system.state))
            state.UIDcounter=int(state.UIDcounter)
            _validateState(state)
            if state.positions.shape!=system.state.positions.shape:
                raise ValueError('Checkpoint particle dimensions do not match target initialization.')
            history=file['neighborHistory']
            recipe=dict(metadata=json.loads(history.attrs['metadata']),checksum=history.attrs['checksum'],
                        tensors={name:torch.from_numpy(dataset[:]) for name,dataset in history.items()})
            source=FourierOU.from_state_dict(dict(parameters=json.loads(file['forcing'].attrs['parameters']),
                knots=torch.from_numpy(file['forcing/knots'][:]),generator=torch.from_numpy(file['forcing/generator'][:])))
            if source.parameters!=system.sourceTerms.acceleration.parameters:
                raise ValueError('Checkpoint forcing parameters are incompatible.')
            context={name:torch.as_tensor(file['context'][name][()],dtype=torch.float64,device=device)
                     for name in ('initialMass','initialEnergy')}
            if any(not math.isfinite(float(t)) or float(t)<=0 for t in context.values()):
                raise ValueError('Checkpoint initial mass/energy context is invalid.')
            return state,recipe,source,time,dt,completed,context,file.attrs['contentChecksum']
    state,recipe,source,time,dt,completed,context,checksum=_local(communication,'checkpoint:read',read)
    if communication is not None:
        communication.agree('checkpoint:read-content',(checksum,completed))
    adjacency=restoreRowChunkCheckpointState(recipe,device=device,policy=prior.plan.policy,
                                             communication=communication,storeI=prior.storeI)
    if checksum.startswith('legacy:'):
        from .legacyChunkCheckpoint import verifyLegacyPairs
        _local(communication,'checkpoint:legacy-pairs',lambda:verifyLegacyPairs(path,adjacency,checksum))
    def install():
        system.state=state
        system.adjacency=adjacency
        system.t=time
        system.sourceTerms=replace(system.sourceTerms,acceleration=source)
        for name in ('pairWork','rerunRequest','supportGrid','listGrid'):
            if hasattr(system,name):
                setattr(system,name,None)
        ctx.config.dt=dt
        ctx.scratch.update(context,driving=source,completedSteps=completed,nextPhysicalDt=dt,
                           checkpointSignature=signature,_rowChunkRestartVerified=True)
        ctx.scratch.pop('_rowChunkAccepted',None)
        ctx.spec.resumeStepOffset=completed
    _local(communication,'checkpoint:install',install)
    return str(path)
