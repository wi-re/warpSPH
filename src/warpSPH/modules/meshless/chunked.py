"""Explicit qualification boundary for eager row-owned meshless stages."""
import torch
from warpSPHCore.radiusSearch import ChunkedAdjacency


def checkMeshlessChunks(system, config, scheme):
    adjacency=system.adjacency
    if not isinstance(adjacency,ChunkedAdjacency):
        return
    def validate():
        if config.dim not in (2,3):
            raise NotImplementedError('Chunked meshless stages currently qualify 2D/3D only.')
        if system.drivenParticles is not None or system.inflowReservoir is not None:
            raise NotImplementedError('Chunked meshless boundaries/reservoirs remain unqualified.')
        state=system.state
        if bool((state.kinds!=0).any()) or bool((state.materials!=0).any()) or scheme.materials is not None:
            raise NotImplementedError('Chunked meshless stages currently require one fluid material without boundary particles.')
        if getattr(scheme,'cudaGraph',False):
            raise NotImplementedError('Chunked meshless stages require eager execution.')
        return (scheme.mode,scheme.closure,scheme.closureSolver,scheme.temporalScheme,
                scheme.neighborSkin,scheme.cgTol,scheme.cgIters,scheme.closureCap)
    communication=adjacency.communication
    if communication is None:
        validate()
    else:
        communication._checkedAgree('meshless:scope',validate)
