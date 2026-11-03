"""Explicit fixed-count particle axes and stable mixed-domain Morton permutations.

Only the audited CompSPH state/update schema is supported. Edge work is never
inferred from its length (E can equal N); it stays with its compute adjacency.
"""
from dataclasses import dataclass, fields
from typing import Any

import torch
import warp as wp
from warp.types import vector
from warpSPHCore import scalar_t
from warpSPHCore.radiusSearch.compactHash.morton import computeZOrderIndex64

# Deliberately explicit: additions to CompSPHState must declare their axis here.
PARTICLE_FIELDS = frozenset('''emitters bodyIDs positions velocities supports masses densities
kinds materials UIDs internalEnergies crkMassDensities crkVolumes crkQGradient totalEnergies
entropies pressures soundspeeds divergence alpha0s alphas entropiesPrev entropyRates
divergencePrevStep wadsleyD wadsleyXi wadsleyVsig wadsleyDPrev wallNormals wallVolumes'''.split())
GLOBAL_FIELDS = frozenset(('UIDcounter', 'entropiesPrevTime', 'divergencePrevStepTime',
                           'wadsleyDPrevTime', 'boundaryProjectionBudget'))
EDGE_FIELDS = frozenset(('ap_ij', 'av_ij', 'f_ij'))
ROW_UPDATE_FIELDS = frozenset('dxdt dvdt dudt dEdt drhodt passive dVdt sourceAcceleration sourceHeat'.split())
GLOBAL_UPDATE_FIELDS = frozenset(('sourceEnergyRates',))
UPDATE_FIELDS = ROW_UPDATE_FIELDS | GLOBAL_UPDATE_FIELDS


def validate_permutation(order, n, device):
    if (not isinstance(order, torch.Tensor) or order.dtype != torch.int64 or
            order.device != device or order.shape != (n,)):
        raise ValueError('permutation must be an int64 vector on the particle device')
    if not torch.equal(torch.sort(order).values, torch.arange(n, device=device)):
        raise ValueError('permutation must be a bijection of particle rows')


def validate_state_schema(state):
    from ..systems.compSPH import CompSPHState
    from ..caseUtils.drivenTurbulence import DrivenCompSPHState
    if type(state) not in (CompSPHState, DrivenCompSPHState):
        raise TypeError('particle layout currently requires the audited CompSPHState schema')
    names = {f.name for f in fields(state)}
    extra = {'sourceEnergyBudget'} if type(state) is DrivenCompSPHState else set()
    if names != PARTICLE_FIELDS | GLOBAL_FIELDS | EDGE_FIELDS | extra or set(vars(state)) != names:
        raise TypeError('particle fields changed: declare particle/global/edge axes before sorting')
    if extra and (not isinstance(state.sourceEnergyBudget, torch.Tensor) or state.sourceEnergyBudget.shape != (2,)):
        raise ValueError('sourceEnergyBudget must be the global work/heat vector')
    n = state.positions.shape[0]
    for name in PARTICLE_FIELDS:
        value = getattr(state, name)
        if value is not None and (not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != n):
            raise ValueError(f'{name} must have one value per particle')


def permute_state(state, order, *, identity=False):
    validate_state_schema(state)
    values = {}
    for name in (f.name for f in fields(state)):
        value = getattr(state, name)
        if name in EDGE_FIELDS:
            # Pair work is retained separately on the compute list, never on public rows.
            values[name] = None
        elif name in PARTICLE_FIELDS and value is not None:
            values[name] = value.clone() if identity else value[order]
        else:
            values[name] = value.clone() if isinstance(value, torch.Tensor) else value
    return type(state)(**values)


def validate_update_schema(update, n):
    from ..systems.compressibleMonaghan import CompressibleSystemUpdate
    if (type(update) is not CompressibleSystemUpdate or {f.name for f in fields(update)} != UPDATE_FIELDS
            or set(vars(update)) != UPDATE_FIELDS):
        raise TypeError('particle layout requires the audited compressible update schema')
    for name in ROW_UPDATE_FIELDS:
        value = getattr(update,name)
        if value is not None and (not isinstance(value,torch.Tensor) or not value.ndim or value.shape[0] != n):
            raise ValueError(f'update {name} must have one value per particle')
    if update.sourceEnergyRates is not None and (not isinstance(update.sourceEnergyRates,torch.Tensor) or update.sourceEnergyRates.shape != (2,)):
        raise ValueError('sourceEnergyRates must be the global work/heat vector')


def permute_update(update, order, *, identity=False):
    validate_update_schema(update,order.numel())
    values = {}
    for f in fields(update):
        value = getattr(update, f.name)
        if value is None:
            values[f.name] = None
        elif f.name in GLOBAL_UPDATE_FIELDS:
            values[f.name] = value.clone()
        elif f.name in ROW_UPDATE_FIELDS:
            values[f.name] = value.clone() if identity else value[order]
        else:
            raise ValueError(f'update {f.name} must have one value per particle')
    return type(update)(**values)


@dataclass(frozen=True, eq=False)
class ParticleLayout:
    """p[compute] = input, q[input] = compute; tensors are owned, read-only handles."""
    input_ids: torch.Tensor
    compute_ids: torch.Tensor
    p: torch.Tensor
    q: torch.Tensor
    identity: bool

    @classmethod
    def create(cls, ids, order):
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('particle IDs must be an integer vector')
        if ids.unique().numel() != ids.numel() or bool((ids < 0).any()):
            raise ValueError('particle IDs must be unique and nonnegative')
        validate_permutation(order, ids.numel(), ids.device)
        p = order.clone()
        q = torch.empty_like(p)
        q[p] = torch.arange(p.numel(), device=p.device)
        return cls(ids.clone(), ids[p].clone(), p, q,
                   torch.equal(p, torch.arange(p.numel(), device=p.device)))

    def for_input(self, ids):
        if torch.equal(ids, self.input_ids):
            return self
        if torch.equal(ids, self.compute_ids):
            # Accepted adoption keeps the same list and compute order.
            return ParticleLayout.create(ids, torch.arange(ids.numel(), device=ids.device))
        raise ValueError('particle membership/order changed without a layout transition')


@wp.kernel
def _morton(positions: wp.array(dtype=vector(length=Any, dtype=scalar_t)),
            lower: wp.array(dtype=scalar_t), upper: wp.array(dtype=scalar_t),
            periodic: wp.array(dtype=wp.bool),
            dim: wp.int32, bits: wp.int32, keys: wp.array(dtype=wp.int64)):
    i = wp.tid()
    cell = wp.vec3i(0)
    resolution = wp.int32(1) << bits
    for d in range(dim):
        unit = (positions[i][d] - lower[d]) / (upper[d] - lower[d])
        if periodic[d]:
            unit = unit - wp.floor(unit)
        cell[d] = wp.clamp(wp.int32(wp.floor(unit * scalar_t(resolution))), 0, resolution-1)
    keys[i] = computeZOrderIndex64(cell)


def morton_order(positions, domain, *, bits=21):
    """Stable 1–3D Z-order, wrapped periodic axes and clamped open axes."""
    if type(bits) is not int or not 1 <= bits <= 21:
        raise ValueError('Morton precision must be an integer between 1 and 21 bits')
    if positions.ndim != 2 or positions.shape[1] not in (1, 2, 3) or domain.dim != positions.shape[1]:
        raise ValueError('Morton positions must be an N by dim tensor, dim in 1..3')
    extent = domain.max-domain.min
    if not bool(((extent > 0) & torch.isfinite(extent)).all()):
        raise ValueError('Morton sorting requires a finite box with positive extents')
    if not bool(torch.isfinite(positions).all() & torch.isfinite(domain.min).all() & torch.isfinite(domain.max).all()):
        raise ValueError('Morton coordinates and bounds must be finite')
    keys = torch.empty(positions.shape[0], dtype=torch.int64, device=positions.device)
    if keys.numel():
        stream = wp.stream_from_torch(torch.cuda.current_stream(positions.device)) if positions.is_cuda else None
        wp.launch(_morton, dim=keys.numel(), inputs=[
            wp.from_torch(positions.contiguous(), dtype=vector(length=domain.dim, dtype=scalar_t), requires_grad=False),
            wp.from_torch(domain.min, requires_grad=False), wp.from_torch(domain.max, requires_grad=False),
            wp.from_torch(domain.periodic,requires_grad=False),
            domain.dim, bits, wp.from_torch(keys)], device=str(positions.device), stream=stream)
    return torch.argsort(keys, stable=True)


def remap_adjacency(adjacency, query_order, reference_order=None, *, return_edge_order=False):
    """Owned CSR/export view; preserve anchors, capacity, row-local order and widths.

    No hash grid or edge work is retained. Self-query/reference permutations can
    differ; omit reference_order only for an equally permuted self list.
    """
    from warpSPHCore import AdjacencyList
    if reference_order is None:
        reference_order = query_order
    validate_permutation(query_order, adjacency.numRows, adjacency.paddedJ.device)
    validate_permutation(reference_order, adjacency.numCols, adjacency.paddedJ.device)
    counts = adjacency.numNeighbors[query_order]
    offsets = torch.zeros_like(adjacency.edgeOffsets)
    if offsets.numel() > 1:
        offsets[1:] = torch.cumsum(counts[:-1], 0, dtype=offsets.dtype)
    edges = adjacency.numEdges if adjacency.numEdges is not None else int(counts.sum())
    delta = adjacency.edgeOffsets[query_order].long() - offsets.long()
    old_edges = torch.arange(edges, device=counts.device) + torch.repeat_interleave(delta, counts.long())
    inverse_reference = torch.empty_like(reference_order)
    inverse_reference[reference_order] = torch.arange(reference_order.numel(), device=counts.device)
    j = torch.full_like(adjacency.paddedJ, -1)
    j[:edges] = inverse_reference[adjacency.paddedJ[old_edges].long()].to(j.dtype)
    i = None
    if adjacency.paddedI is not None:
        i = torch.full_like(adjacency.paddedI, -1)
        i[:edges] = torch.repeat_interleave(torch.arange(adjacency.numRows, device=counts.device, dtype=i.dtype), counts.long())
    def anchors(query, reference):
        q = None if query is None else query[query_order]
        if query is reference and torch.equal(query_order, reference_order):
            return q, q
        return q, None if reference is None else reference[reference_order]
    qp, rp = anchors(adjacency.queryPositions, adjacency.referencePositions)
    qh, rh = anchors(adjacency.querySupports, adjacency.referenceSupports)
    result = AdjacencyList(i, j, counts, offsets, adjacency.numRows, adjacency.numCols,
                         qp, rp, qh, rh, hashMap=None, numEdges=edges)

    return (result, old_edges) if return_edge_order else result
