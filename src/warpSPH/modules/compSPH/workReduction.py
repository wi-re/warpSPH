"""Reduce compatible pair work without materializing per-edge velocity gathers.

One thread per CSR row visits edges in order, as torch.segment_reduce does.
The Torch path retains automatic differentiation when inputs need a tape.
"""
from ...utils.adjacency import cooRows
import torch
import warp as wp
from warpSPHCore import scalar_t, index_t, offset_t, get_torch_precision
from ...math.scatter import segment_sum
from warpSPHCore.radiusSearch import ChunkedAdjacency, ChunkedPairField
from warpSPHCore.autograd.stream import torchWarpStream



@wp.func
def _pairDot(velocity: wp.array2d(dtype=scalar_t), work: wp.array2d(dtype=scalar_t),
             i: int, j: int, at: offset_t):
    dot = scalar_t(0.)
    for axis in range(velocity.shape[1]):
        dot += (velocity[j, axis] - velocity[i, axis]) * work[at, axis]
    return dot


@wp.kernel
def _pairWorkRows(velocity: wp.array2d(dtype=scalar_t), work: wp.array2d(dtype=scalar_t),
                  counts: wp.array(dtype=wp.int32), offsets: wp.array(dtype=offset_t),
                  neighbors: wp.array(dtype=index_t), output: wp.array(dtype=scalar_t)):
    i = wp.tid()
    total = scalar_t(0.)
    for edge in range(counts[i]):
        at = offsets[i] + offset_t(edge)
        j = wp.int32(neighbors[at])
        total += _pairDot(velocity, work, i, j, at)
    output[i] = total


@wp.kernel
def _chunkPairWorkRows(velocity: wp.array2d(dtype=scalar_t), work: wp.array2d(dtype=scalar_t),
                      counts: wp.array(dtype=wp.int32), offsets: wp.array(dtype=offset_t),
                      neighbors: wp.array(dtype=index_t), rowStart: wp.int32,
                      output: wp.array(dtype=scalar_t)):
    local = wp.tid()
    total = scalar_t(0.)
    for edge in range(counts[local]):
        at = offsets[local] + offset_t(edge)
        j = wp.int32(neighbors[at])
        total += _pairDot(velocity, work, local + rowStart, j, at)
    output[local + rowStart] = total


def _reduceChunkedPairWork(adjacency, velocity, work):
    communication = adjacency.communication
    if communication is None:
        return _reduceChunkedLocal(adjacency, velocity, work)
    communication.assertReplicated('compatible:half-velocity', dict(velocity=velocity))
    out = communication.localPhase('compatible:reduce-local-W', lambda: _reduceChunkedLocal(adjacency, velocity, work))
    communication.exchangeParticleOutputs(adjacency, (out,), tag='compatible:reduce-owned-W')
    return out


def _reduceChunkedLocal(adjacency, velocity, work):
    if not isinstance(work, ChunkedPairField):
        raise TypeError('Chunked compatible reduction requires a ChunkedPairField; no flat pair fallback exists.')
    work.validate(adjacency)
    tensors = (velocity, *work.buffers)
    if any(t.requires_grad or torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in tensors):
        raise NotImplementedError('Chunked compatible reduction does not support AD.')
    if (velocity.shape != (adjacency.numRows, *work.trailingShape) or velocity.ndim != 2
        or adjacency.numRows != adjacency.numCols or adjacency.queryPositions is not adjacency.referencePositions
        or (adjacency.plan.numOwners != 1 and adjacency.communication is None)
        or velocity.device != adjacency.queryPositions.device or velocity.dtype != get_torch_precision()
        or work.dtype != velocity.dtype):
        raise ValueError('Chunked compatible reduction needs complete self rows and matching velocity/work layouts.')
    if velocity.is_cuda and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError('Chunked compatible reduction does not support capture.')
    out = torch.empty(len(velocity), dtype=velocity.dtype, device=velocity.device)
    w = lambda t: wp.from_torch(t.contiguous())
    with torchWarpStream(velocity.device):
        v, output = w(velocity), w(out)
        for chunk, buffer in zip(adjacency, work.buffers):
            a, m = chunk.adjacency, chunk.metadata
            wp.launch(_chunkPairWorkRows, dim=m.rowCount, device=wp.device_from_torch(out.device),
                      inputs=[v, w(buffer), w(a.numNeighbors), w(a.edgeOffsets), w(a.paddedJ), m.rowStart, output])
    return out


def reducePairWork(adjacency, velocity, work):
    """sum_j (v_j-v_i).work_ij over live edges only, including empty rows."""
    if isinstance(adjacency, ChunkedAdjacency):
        return _reduceChunkedPairWork(adjacency, velocity, work)
    if (velocity.requires_grad or work.requires_grad or velocity.shape[0] == 0
            or velocity.dtype != get_torch_precision() or not hasattr(adjacency, 'edgeOffsets')
            or adjacency.numNeighbors.dtype != torch.int32):
        count = getattr(adjacency, 'numEdges', work.shape[0])
        i, j = cooRows(adjacency)[:count], adjacency.paddedJ[:count]
        return segment_sum(torch.einsum('ij,ij->i', velocity[j] - velocity[i], work[:count]), adjacency.numNeighbors)
    out = torch.empty(velocity.shape[0], dtype=velocity.dtype, device=velocity.device)
    w = lambda t: wp.from_torch(t.contiguous())
    with torchWarpStream(velocity.device):
        wp.launch(_pairWorkRows, dim=len(out), device=wp.device_from_torch(out.device),
                  inputs=[w(velocity), w(work), w(adjacency.numNeighbors), w(adjacency.edgeOffsets),
                          w(adjacency.paddedJ), w(out)])
    return out


@wp.kernel
def _referenceEntropyRows(velocity: wp.array2d(dtype=scalar_t), work: wp.array2d(dtype=scalar_t),
                  entropy: wp.array(dtype=scalar_t), tiny: scalar_t,
                  counts: wp.array(dtype=wp.int32), offsets: wp.array(dtype=offset_t),
                  neighbors: wp.array(dtype=index_t), output: wp.array(dtype=scalar_t)):
    i = wp.tid()
    total = scalar_t(0.)
    si = entropy[i]
    for edge in range(counts[i]):
        at = offsets[i] + offset_t(edge)
        j = wp.int32(neighbors[at])
        sj = entropy[j]
        dot = _pairDot(velocity, work, i, j, at)
        fraction = scalar_t(.5)
        if wp.abs(dot) > scalar_t(1.e-15) and wp.abs(si - sj) > scalar_t(1.e-15):
            numerator = wp.abs(sj) if dot > scalar_t(0.) else wp.abs(si)
            fraction = numerator / wp.max(wp.abs(si) + wp.abs(sj), tiny)
        total += fraction * dot
    output[i] = total


def reduceReferenceEntropyWork(adjacency, velocity, work, entropy):
    """Pinned native entropy weighting used by the paper-matching control.

    Preserve its strict thresholds and common-gamma contract. This is not the
    default CRK entropy policy (which has different thresholds).
    """
    tiny = torch.finfo(velocity.dtype).tiny
    if (any(t.requires_grad for t in (velocity, work, entropy)) or len(velocity) == 0
            or velocity.dtype != get_torch_precision() or not hasattr(adjacency, 'edgeOffsets')
            or adjacency.numNeighbors.dtype != torch.int32):
        count = getattr(adjacency, 'numEdges', work.shape[0])
        i, j = cooRows(adjacency)[:count], adjacency.paddedJ[:count]
        dot = torch.einsum('ij,ij->i', velocity[j] - velocity[i], work[:count])
        si, sj = entropy[i], entropy[j]
        fraction = torch.where(dot > 0., sj.abs(), si.abs()) / (si.abs() + sj.abs()).clamp_min(tiny)
        fraction = torch.where((dot.abs() > 1.e-15) & ((si - sj).abs() > 1.e-15), fraction, .5)
        return segment_sum(fraction * dot, adjacency.numNeighbors)
    out = torch.empty(len(velocity), dtype=velocity.dtype, device=velocity.device)
    w = lambda t: wp.from_torch(t.contiguous())
    with torchWarpStream(velocity.device):
        wp.launch(_referenceEntropyRows, dim=len(out), device=wp.device_from_torch(out.device),
                  inputs=[w(velocity), w(work), w(entropy), tiny, w(adjacency.numNeighbors),
                          w(adjacency.edgeOffsets), w(adjacency.paddedJ), w(out)])
    return out
