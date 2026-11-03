"""Eager Godunov row-chunk contracts and local input preparation."""
import torch
from warpSPHCore import RowRole
from warpSPHCore.radiusSearch import ChunkedAdjacency


def centredQueryRole(ctx,extras):
    half=extras['halfDt']
    half=half.value if hasattr(half,'value') else half
    return RowRole.QUERY if extras['params'].timeCentredEnergy and half>0 else RowRole.REPLICATED


def godunovInputs(adjacency,function):
    """Callback is local only; agree errors before entering the operator."""
    if isinstance(adjacency,ChunkedAdjacency):
        if adjacency.communication is not None:
            return adjacency.communication.localPhase('godunov:inputs',function)
        if adjacency.queryPositions.is_cuda and torch.cuda.is_current_stream_capturing():
            raise NotImplementedError('Row-chunk Godunov inputs require eager execution outside CUDA graph capture.')
    return function()


def checkGodunovChunks(system,config,scheme,dt):
    adjacency=system.adjacency
    if not isinstance(adjacency,ChunkedAdjacency):return
    def validate():
        from ..mdbc._util import stateHasBoundaryParticles
        from ..crk.storage import derivativeInputs
        adjacency.assertStorageValid()
        if adjacency.communication is not None:adjacency.communication.validateAdjacency(adjacency)
        if (getattr(scheme,'materials',None) is not None or stateHasBoundaryParticles(system.state,config)
            or getattr(scheme,'stepMode','eager') not in (None,'eager')
            or getattr(getattr(config,'compute',None),'execution','eager')!='eager'):
            raise NotImplementedError('Chunked Godunov currently supports eager uniform-material flow without walls.')
        if derivativeInputs(*vars(system.state).values(),dt):
            raise NotImplementedError('Chunked Godunov does not support differentiation.')
    godunovInputs(adjacency,validate)


def chunkedVerletScale(adjacency,value):
    # The eager chunk planner agrees this scalar in its existing configuration
    # contract. Keep the ordinary device-tensor/grid path unchanged.
    if not isinstance(adjacency,ChunkedAdjacency):return value
    return godunovInputs(adjacency,lambda:float(value))
