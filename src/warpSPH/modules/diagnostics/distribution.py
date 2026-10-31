"""Nearest-neighbour diagnostics as bounded row reductions, no pair expansion."""
from typing import Any
import warp as wp
from warpSPHCore.type_config import scalar_t,offset_t
from warpSPHCore.dataTypes import domainData,adjacencyData,gridData,kernelState
from warpSPHCore.autograd import OperatorSpec,OutputSpec,ExtraSpec,ExtraKind,SPHContext,launchOperator
from warpSPHCore.autograd.operator_spec import RowRole
from warpSPHCore.math import computeDistanceVec


@wp.kernel(enable_backward=False)
def _distribution(queryState:Any,referenceState:Any,domainState:domainData,
    useAdjacency:wp.bool,adjacencyState:adjacencyData,gridState:gridData,
    correctionData:Any,kernelProperties:kernelState,rowStart:wp.int32,
    nearest:wp.array(dtype=scalar_t),count:wp.array(dtype=scalar_t)):
    i=wp.tid()
    gi=i+rowStart
    minimum=scalar_t(wp.inf)
    total=scalar_t(0.)
    if queryState.kinds[i]==0:
        for n in range(adjacencyState.numNeighbors[i]):
            j=wp.int32(adjacencyState.neighborList[adjacencyState.neighborOffsets[i]+offset_t(n)])
            if j!=gi and referenceState.kinds[j]==0:
                delta=computeDistanceVec(queryState.positions[i],referenceState.positions[j],domainState)
                minimum=wp.min(minimum,wp.length(delta))
                total+=scalar_t(1.)
    nearest[i]=minimum
    count[i]=total


_SPEC=OperatorSpec(kernel=_distribution,rowChunked=True,
    outputs=(OutputSpec(dtype=scalar_t),OutputSpec(dtype=scalar_t)),
    extras=(ExtraSpec('rowStart',ExtraKind.SCALAR,RowRole.ROW_START),))


def distributionRows(particles,config,adjacency):
    from warpSPHCore import OperationProperties
    return launchOperator(_SPEC,SPHContext(query=particles,properties=OperationProperties(kernel=config.kernel),
        domain=config.domain,adjacency=adjacency),rowStart=0)
