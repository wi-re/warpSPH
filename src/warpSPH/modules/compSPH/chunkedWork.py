"""Native eager compatible work policy; pair buffers never leave their chunks."""
from warpSPHCore.radiusSearch import ChunkedAdjacency, ChunkedPairField
from ..crk.storage import derivativeInputs
from ...enumTypes import EnergyScheme


def checkChunkedStage(system, state, config, scheme, index, dt, family):
    pw = system.pairWork
    adjacency = pw.latest if pw is not None and pw.latest is not None else system.adjacency
    if isinstance(adjacency, ChunkedAdjacency) and adjacency.communication is not None:
        return adjacency.communication.localPhase('compatible:stage-contract',
            lambda: _checkChunkedStage(system, state, config, scheme, index, dt, family))
    return _checkChunkedStage(system, state, config, scheme, index, dt, family)


def _checkChunkedStage(system, state, config, scheme, index, dt, family):
    pw = system.pairWork
    adjacency = pw.latest if pw is not None and pw.latest is not None else system.adjacency
    if not isinstance(adjacency, ChunkedAdjacency):
        return
    from ..mdbc._util import stateHasBoundaryParticles
    if (not scheme.compatibleEnergy or scheme.energyScheme != EnergyScheme.CRK
        or getattr(scheme, family.lower() + 'PairStorage', 'recompute') != 'recompute'
        or getattr(scheme, 'materials', None) is not None
        or getattr(scheme, 'stepMode', 'eager') not in (None, 'eager')
        or getattr(getattr(config, 'compute', None), 'execution', 'eager') != 'eager'
        or getattr(scheme, 'supportSolveOnGrid', False) or getattr(scheme, 'inPlaceAdjacency', False)
        or pw is None or pw.weights is None or index is None):
        raise NotImplementedError('Chunked compatible work requires native recomputation, eager execution and integrator stage weights; materials/graph/flat storage controls are unqualified.')
    if family == 'CRK' and any(getattr(scheme, key, default) != default for key, default in (
        ('crkEnergyPartition','native'), ('crkEnergyPredictor','raw'), ('crkEvolution','summation'),
        ('crkDensityPolicy','legacy'), ('crkViscosityGradient','current'))):
        raise NotImplementedError('Chunked CRK compatible work supports native summation/current-gradient controls only.')
    if stateHasBoundaryParticles(state, config):
        raise NotImplementedError('Chunked compatible wall work is not qualified.')
    if derivativeInputs(*vars(state).values(), dt):
        raise NotImplementedError('Chunked compatible work does not support AD.')
    if adjacency.communication is not None:
        adjacency.communication.validateAdjacency(adjacency)
    elif adjacency.plan.numOwners != 1:
        raise NotImplementedError('Chunked compatible work requires a complete list or an explicit distributed process group.')
    ChunkedPairField._checkLayout(adjacency, (state.positions.shape[1],), state.positions.dtype)


def workBuffer(pairWork, state, adjacency, reducing):
    """Replay reduces immediately: only one unread pair placeholder is needed."""
    if reducing:
        return state.positions.new_empty((1, state.positions.shape[1]))
    if pairWork.work is None:
        if isinstance(adjacency, ChunkedAdjacency):
            return ChunkedPairField.empty(adjacency, trailingShape=(state.positions.shape[1],), dtype=state.positions.dtype)
        return state.positions.new_empty((adjacency.paddedJ.numel(), state.positions.shape[1]))
    if isinstance(adjacency, ChunkedAdjacency):
        pairWork.work.validate(adjacency)
    return pairWork.work
