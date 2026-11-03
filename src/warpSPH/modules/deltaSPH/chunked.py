"""Bounded eager periodic qualification for native delta/Sun2017 stages."""
from warpSPHCore.radiusSearch import ChunkedAdjacency


def checkDeltaChunks(system,config,scheme):
    adjacency=system.adjacency
    if not isinstance(adjacency,ChunkedAdjacency):return
    def validate():
        if config.dim not in (2,3) or not bool(config.domain.periodic.all()):
            raise NotImplementedError('Chunked delta stages require periodic 2D/3D domains.')
        state=system.state
        if bool((state.kinds!=0).any()) or bool((state.materials!=0).any()):
            raise NotImplementedError('Chunked delta stages require one fluid material without boundary particles.')
        if (scheme.surfaceDetectionConfig.active or scheme.boundaryProvider is not None
            or scheme.rigidBodies or scheme.boundaryConditions or getattr(scheme,'cudaGraph',False)):
            raise NotImplementedError('Chunked delta walls/free surfaces/boundary conditions/graphs remain unqualified.')
        if scheme.shiftProperties.scheme.name!='deltaSPH':
            raise NotImplementedError('Chunked delta stages currently qualify the native delta shifting operator only.')
        return (scheme.schemeName,scheme.freezeDiffusionAcrossStages,scheme.freezeSurfaceMaskAcrossStages,
                scheme.shiftProperties.sun2017Eq7Shift)
    if adjacency.communication is None:validate()
    else:adjacency.communication._checkedAgree('delta:scope',validate)
