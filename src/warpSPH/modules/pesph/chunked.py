"""Eager PESPH input boundaries; particle outputs exchange in core dispatch."""
import torch
from warpSPHCore.radiusSearch import ChunkedAdjacency


def pesphInputs(adjacency, function):
    if isinstance(adjacency, ChunkedAdjacency):
        if adjacency.queryPositions.is_cuda and torch.cuda.is_current_stream_capturing():
            raise NotImplementedError('Chunked PESPH requires eager execution outside CUDA graph capture.')
        if adjacency.communication is not None:
            return adjacency.communication.localPhase('pesph:inputs', function)
    return function()


def checkPESPHChunks(system, config, scheme, dt=None):
    if not isinstance(system.adjacency, ChunkedAdjacency):
        return
    def validate():
        from ..crk.storage import derivativeInputs
        if (getattr(scheme, 'materials', None) is not None or scheme.boundaryConditions
            or getattr(scheme, 'stepMode', 'eager') not in (None, 'eager')
            or getattr(getattr(config, 'compute', None), 'execution', 'eager') != 'eager'
            or not bool(config.domain.periodic.all()) or not bool((system.state.kinds == 0).all())):
            raise NotImplementedError('Chunked PESPH currently requires eager periodic single-material flow without boundary sources.')
        if system.sourceTerms is not None:
            from .sources import validatePESPHSource
            validatePESPHSource(system, scheme)
        if derivativeInputs(*vars(system.state).values(), dt):
            raise NotImplementedError('Chunked PESPH does not support differentiation.')
    pesphInputs(system.adjacency, validate)
