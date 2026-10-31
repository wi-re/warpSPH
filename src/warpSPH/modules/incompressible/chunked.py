"""Eager periodic owned-row scope for native incompressible projections."""
from warpSPHCore.radiusSearch import ChunkedAdjacency


def checkProjectionChunks(system, config, scheme):
    adjacency = system.adjacency
    if not isinstance(adjacency, ChunkedAdjacency):
        return
    def validate():
        state = system.state
        if config.dim not in (2, 3) or not bool(config.domain.periodic.all()):
            raise NotImplementedError('Chunked projections require periodic 2D/3D domains.')
        if bool((state.kinds != 0).any()) or bool((state.materials != 0).any()):
            raise NotImplementedError('Chunked projections require one fluid material without wall particles.')
        if (scheme.surfaceDetectionConfig.active or getattr(scheme, 'boundaryProvider', None) is not None
            or scheme.rigidBodies or scheme.boundaryConditions or getattr(scheme, 'cudaGraph', False)
            or scheme.shiftProperties.active):
            raise NotImplementedError('Chunked projection walls/free surfaces/boundary conditions/graphs and external shifting remain unqualified.')
        if hasattr(scheme, 'dfsph') and scheme.dfsph.pressureBoundary != 'native':
            raise NotImplementedError('Chunked DFSPH currently qualifies native pressure geometry only.')
        return scheme.schemeName
    if adjacency.communication is None:
        validate()
    else:
        adjacency.communication._checkedAgree('projection:scope', validate)
