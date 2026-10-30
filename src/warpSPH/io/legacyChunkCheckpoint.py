"""Bounded import of version-1 driven SPH COO/CSR history into owned chunks.

Legacy files have no content checksum or search-policy metadata. The caller's
matching config supplies mode/Verlet scale, as in the old flat restore path.
Reconstructed counts AND every live pair order are verified before installation.
"""
from hashlib import sha256
import math

import numpy as np
import torch

from warpSPHCore.radiusSearch import rowChunkCheckpointRecipe
from .hdf5 import loadState


def fileFingerprint(path):
    digest=sha256()
    with open(path,'rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    return 'legacy:'+digest.hexdigest()


def readLegacyCheckpoint(file,ctx,system,signature,device):
    """Read O(N) fields only; leave legacy J/I on disk for owned-slice checks."""
    from .chunkCheckpoint import _validateState
    if getattr(ctx.scheme,'name',ctx.scheme) not in ('Monaghan','CRKSPH','CompSPH'):
        raise NotImplementedError('Legacy chunk import currently qualifies Monaghan/CRKSPH/CompSPH only; use a native checkpoint for Godunov schemes.')
    from ..modules.forcing import FourierOU
    required=('checkpointVersion','checkpointSignature','checkpointCompletedSteps',
              'checkpointAdjacencyVersion','checkpointAdjacencyPresent','initialMass','initialEnergy','time')
    if any(name not in file.attrs for name in required):
        raise ValueError('Legacy checkpoint is missing accepted state/forcing/search context.')
    if file.attrs['checkpointVersion']!=1 or file.attrs['checkpointAdjacencyVersion']!=1:
        raise ValueError('Unsupported legacy checkpoint/history version.')
    if file.attrs['checkpointSignature']!=signature:
        raise ValueError('Checkpoint physical/numerical configuration is incompatible.')
    if not file.attrs['checkpointAdjacencyPresent'] or any(name not in file for name in
            ('state','dict_sourceAdjacency','dict_forcingCheckpoint','checkpointNextDt')):
        raise ValueError('Legacy checkpoint is missing native neighbor history or forcing state.')
    if any(name in file['state'] for name in ('ap_ij','av_ij','f_ij')):
        raise ValueError('Legacy expanded per-edge stage state is not supported by chunk import.')
    completed=int(file.attrs['checkpointCompletedSteps'])
    time=float(file.attrs['time'])
    dt=system.state.positions.new_tensor(float(file['checkpointNextDt'][()]))
    if (completed<1 or ctx.spec.resumeStepOffset not in (0,completed) or not math.isfinite(time) or time<0
        or not math.isfinite(float(dt)) or float(dt)<=0 or (ctx.spec.nSteps is None and ctx.spec.tLimit<=time)):
        raise ValueError('Legacy checkpoint time/timestep or completed-step offset is invalid.')
    state=loadState(file['state'],device,type(system.state))
    state.UIDcounter=int(state.UIDcounter)
    _validateState(state)
    if state.positions.shape!=system.state.positions.shape:
        raise ValueError('Checkpoint particle dimensions do not match target initialization.')
    saved=file['dict_sourceAdjacency']
    names=('queryPositions','referencePositions','querySupports','referenceSupports','numNeighbors','edgeOffsets','paddedJ')
    if any(name not in saved for name in names):
        raise ValueError('Incomplete legacy CSR history.')
    n=len(state.positions)
    if int(saved.attrs['numRows'])!=n or int(saved.attrs['numCols'])!=n:
        raise ValueError('Legacy neighbor dimensions do not match the particle state.')
    counts=torch.from_numpy(saved['numNeighbors'][:])
    offsets=torch.from_numpy(saved['edgeOffsets'][:])
    if counts.shape!=(n,) or counts.dtype!=torch.int32 or bool((counts<0).any()):
        raise ValueError('Invalid legacy neighbor counts.')
    prefix=torch.cat((torch.zeros(1,dtype=torch.int64),counts.cumsum(0,dtype=torch.int64)))
    total=int(prefix[-1])
    if offsets.shape!=(n,) or offsets.dtype not in (torch.int32,torch.int64) or not torch.equal(offsets.long(),prefix[:-1]):
        raise ValueError('Invalid legacy CSR offsets; expected an exact int64 count scan.')
    j=saved['paddedJ']
    if (int(saved.attrs['numEdges'])!=total or j.ndim!=1 or len(j)<total
        or j.dtype not in (np.dtype('int32'),np.dtype('int64'))):
        raise ValueError('Invalid legacy edge extent/ID dtype.')
    if 'paddedI' in saved:
        i=saved['paddedI']
        if i.shape!=j.shape or i.dtype not in (np.dtype('int32'),np.dtype('int64')):
            raise ValueError('Invalid legacy COO layout.')
    snapshots={name:torch.from_numpy(saved[name][:]) for name in names[:4]}
    for flag,q,r in [('samePositions','queryPositions','referencePositions'),('sameSupports','querySupports','referenceSupports')]:
        if not isinstance(saved.attrs[flag],(bool,np.bool_)):
            raise ValueError('Invalid legacy snapshot alias flag.')
        if saved.attrs[flag]:
            if not torch.equal(snapshots[q],snapshots[r]):
                raise ValueError('Legacy alias snapshots disagree.')
            snapshots[r]=snapshots[q]
    prior=system.adjacency
    recipe=rowChunkCheckpointRecipe(**snapshots,counts=counts,domain=ctx.config.domain,
        policy=prior.plan.policy,mode=ctx.config.supportMode,storeI=prior.storeI,
        verletScale=ctx.config.verletScale)
    forcing=file['dict_forcingCheckpoint']
    source=FourierOU.from_state_dict(dict(parameters=system.sourceTerms.acceleration.parameters,
        knots=torch.from_numpy(forcing['knots'][:]),generator=torch.from_numpy(forcing['generator'][:])))
    context={name:torch.tensor(float(file.attrs[name]),dtype=torch.float64,device=device)
             for name in ('initialMass','initialEnergy')}
    if any(not math.isfinite(float(t)) or float(t)<=0 for t in context.values()):
        raise ValueError('Invalid legacy initial mass/energy context.')
    return state,recipe,source,time,dt,completed,context,fileFingerprint(file.filename)


def verifyLegacyPairs(path,adjacency,fingerprint):
    """Check bounded owned slices; no global pair tensor/view is constructed."""
    import h5py
    with h5py.File(path,'r') as file:
        saved=file['dict_sourceAdjacency']
        for chunk in adjacency:
            m,a=chunk.metadata,chunk.adjacency
            start=adjacency.plan.prefix[m.rowStart]
            stop=adjacency.plan.prefix[m.rowStop]
            j=torch.from_numpy(saved['paddedJ'][start:stop])
            if not torch.equal(j.long(),a.paddedJ[:m.liveEdges].cpu().long()):
                raise ValueError('Legacy neighbor pair order/content does not match reconstructed search history.')
            if 'paddedI' in saved:
                i=torch.from_numpy(saved['paddedI'][start:stop])
                rows=torch.arange(m.rowStart,m.rowStop).repeat_interleave(a.numNeighbors.cpu().long())
                if not torch.equal(i.long(),rows):
                    raise ValueError('Legacy COO row IDs disagree with CSR counts.')
        # Padding is not live work, but malformed sentinel data is not accepted.
        # Read in fixed host blocks, independently of the old flat capacity.
        for name in ('paddedJ','paddedI'):
            if name not in saved:continue
            dataset=saved[name]
            for start in range(adjacency.numEdges,len(dataset),131072):
                if np.any(dataset[start:min(start+131072,len(dataset))]!=-1):
                    raise ValueError('Invalid legacy pair padding sentinel.')
    if fileFingerprint(path)!=fingerprint:
        raise ValueError('Legacy checkpoint changed while it was being imported.')
