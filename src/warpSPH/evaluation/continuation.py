"""UID-aware adapters for the audited compressible continuation schemas.

History is a numerical input, while retained result stages are observations.
Transform the former at acceptance without rewriting the latter. Pair arrays
always travel with their original CSR until a native driver actually needs them.
"""
from dataclasses import dataclass, fields, is_dataclass, replace
from copy import copy
from enum import Enum

import torch

from .layout import (EDGE_FIELDS, GLOBAL_FIELDS, PARTICLE_FIELDS, ParticleLayout,
                     permute_state, permute_update, remap_adjacency,
                     validate_state_schema, validate_update_schema)


def uid_order(source, target):
    """Rows in source corresponding to target; a same-size foreign set is an error."""
    if source.dtype != target.dtype or source.device != target.device or source.shape != target.shape:
        raise ValueError('continuation particle membership differs')
    if torch.equal(source, target):
        return torch.arange(source.numel(), device=source.device)
    indices = torch.argsort(source, stable=True)
    sorted_ids = source[indices]
    locations = torch.searchsorted(sorted_ids, target)
    if (source.unique().numel() != source.numel() or target.unique().numel() != target.numel()
            or bool((locations >= source.numel()).any())
            or not torch.equal(sorted_ids[locations], target)):
        raise ValueError('continuation particle membership differs')
    return indices[locations]


def stage_view(stage, target_ids, *, native=False):
    """Map a StageResult, preserving explicit compute CSR/edge ownership."""
    from warpSPHIntegrators.state import StageResult
    from .sortedStep import SpatialStageContext
    if not isinstance(stage, StageResult) or not isinstance(stage.aux, tuple) or len(stage.aux) not in (2, 3):
        raise TypeError('continuation requires a compressible StageResult with UID-bearing auxiliary state')
    adjacency, state, *extra = stage.aux
    validate_state_schema(state)
    validate_update_schema(stage.update,state.UIDs.numel())
    spatial = extra[0] if extra else None
    if spatial is not None and not isinstance(spatial, SpatialStageContext):
        raise TypeError('unknown continuation stage auxiliary metadata')
    if spatial is not None and set(spatial.edge_fields) != EDGE_FIELDS:
        raise TypeError('undeclared continuation edge fields')
    if (state.UIDs.shape,state.UIDs.dtype,state.UIDs.device) != (target_ids.shape,target_ids.dtype,target_ids.device):
        raise ValueError('continuation particle membership differs')
    identity = torch.equal(state.UIDs,target_ids)
    if identity and spatial is not None:
        known = spatial.layout.for_input(state.UIDs)
        if not native:
            return stage._replace(aux=(spatial.adjacency if known.identity else None,state,replace(spatial,layout=known)))
        if known.identity:
            # Read-only history values already in the driver's coordinates.
            # New stage buffers are owned by the numerical driver; no gather
            # or CSR rewrite is needed for the lifetime of this layout.
            values = copy(state)
            for name,value in spatial.edge_fields.items():
                setattr(values,name,value)
            return stage._replace(aux=(spatial.adjacency,values))
    if identity and native and spatial is None:
        return stage
    order = uid_order(state.UIDs, target_ids)
    compute_ids = spatial.layout.compute_ids if spatial is not None else state.UIDs
    compute_order = uid_order(compute_ids, target_ids)
    layout = ParticleLayout.create(target_ids, uid_order(target_ids, compute_ids))
    adjacency = spatial.adjacency if spatial is not None else adjacency
    edge_fields = spatial.edge_fields if spatial is not None else {n: getattr(state, n) for n in EDGE_FIELDS}
    values = permute_state(state, order, identity=identity)
    update = permute_update(stage.update, order, identity=identity)
    if native:
        if adjacency is not None and not layout.identity:
            adjacency, edge_order = remap_adjacency(adjacency, compute_order, return_edge_order=True)
            edge_fields = {name: None if value is None else value.clone() for name, value in edge_fields.items()}
            for value in edge_fields.values():
                if value is not None:
                    value[:edge_order.numel()] = value[edge_order].clone()
        for name, value in edge_fields.items():
            setattr(values, name, value)
        return stage._replace(update=update, aux=(adjacency, values))
    metadata = (SpatialStageContext(layout, adjacency, edge_fields) if spatial is None
                else replace(spatial, layout=layout))
    return stage._replace(update=update, aux=(adjacency if layout.identity else None, values, metadata))


def history_view(history, target_ids, *, native=False):
    """Fixed-coefficient Adams/BDF and RK bookkeeping; no inferred tensor axes."""
    from warpSPHIntegrators.state.history import StepHistory
    from warpSPHIntegrators.state import StageResult
    from .sortedStep import _system_view
    if history is None:
        return None
    if type(history) is not StepHistory or history.context is not None:
        raise ValueError('history/reuse requires StepHistory without unaudited method-owned context')
    entries = []
    for entry in history:
        stage = stage_view(StageResult(aux=entry.aux, update=entry.update), target_ids, native=native)
        snapshot = entry.state
        if snapshot is not None:
            from ..systems.compSPH import CompSPHSystem
            if type(snapshot) is not CompSPHSystem:
                raise TypeError('history snapshots require CompSPHSystem')
            if not torch.equal(snapshot.state.UIDs, entry.aux[1].UIDs):
                raise ValueError('history update and state snapshot particle order differs')
            validate_state_schema(snapshot.state)
            if not torch.equal(snapshot.state.UIDs,target_ids):
                snapshot = _system_view(snapshot, permute_state(snapshot.state, uid_order(snapshot.state.UIDs, target_ids)))
            elif snapshot.adjacency is not None or snapshot.spatialContext is not None or snapshot.pairWork is not None:
                snapshot = _system_view(snapshot,permute_state(snapshot.state,torch.arange(target_ids.numel(),device=target_ids.device),identity=True))
        entries.append(entry._replace(update=stage.update, aux=stage.aux, state=snapshot))
    # The old pointer guard cannot describe newly allocated UID storage. These
    # drivers do not pass uid to pushed(); use the target's actual identity if
    # the caller supplied that optional guard. Membership was checked above.
    fingerprint = None if history._uid_fingerprint is None else (target_ids.data_ptr(), tuple(target_ids.shape))
    return StepHistory(history.maxlen, entries, _uid_fingerprint=fingerprint)


def _configuration(value):
    """Immutable identity/version description; unknown mutable inputs disable reuse."""
    if isinstance(value, torch.Tensor):
        return ('tensor', value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)
    if value is None or isinstance(value, (str, bool, int, float, Enum, torch.dtype, torch.device)):
        return value
    from ..configurations.moduleConfigurations.diffusionParameters import DiffusionParameters
    from ..configurations.crkSPH import CRKViscosity
    for schema in (DiffusionParameters, CRKViscosity):
        if type(value) is schema.instance_type:
            if set(vars(value)) != {'_ctype', *schema.vars}:
                raise TypeError('undeclared Warp parameter fields')
            return (schema.key, tuple((name, _configuration(getattr(getattr(value,name), 'value', getattr(value,name))))
                                      for name in schema.vars))
    if is_dataclass(value):
        return (type(value), tuple((f.name, _configuration(getattr(value, f.name))) for f in fields(value)))
    if isinstance(value, (tuple, list)):
        return tuple(_configuration(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _configuration(v)) for k, v in value.items())
    if callable(value):
        return ('callable', id(value))
    raise TypeError(f'unqualified configuration for derivative reuse: {type(value).__name__}')


@dataclass(frozen=True)
class ReuseStamp:
    time: float
    dt: float
    method: str
    configuration: tuple


@dataclass(frozen=True)
class EndpointDerivative:
    stamp: ReuseStamp
    ids: torch.Tensor
    update: object


def reuse_stamp(integrator, state, dt, config, schemeConfig):
    """FSAL is valid here only for a fixed-support, memoryless native RHS."""
    from ..enumTypes import AdaptiveSupportScheme, ViscositySwitch
    if (not integrator.fsal or integrator.implicit or schemeConfig.compatibleEnergy
            or schemeConfig.adaptiveSupportScheme != AdaptiveSupportScheme.NoScheme
            or schemeConfig.viscositySwitchParams.scheme != ViscositySwitch.NoneSwitch
            or getattr(schemeConfig, 'crkViscosityGradient', 'current') != 'current'):
        return None
    try:
        signature = (_configuration(config), _configuration(schemeConfig))
    except TypeError:
        return None
    return ReuseStamp(float(state.t), float(dt), integrator.identifier.name, signature)


def reusable_prior(stage, integrator, state, dt, config, schemeConfig):
    """Return a native prior only with an unchanged, stamped endpoint; else refresh."""
    expected = reuse_stamp(integrator, state, dt, config, schemeConfig)
    spatial = stage.aux[2] if len(stage.aux) == 3 else None
    if expected is None or spatial is None or spatial.reuse is None or spatial.reuse.stamp != expected:
        return None
    previous = stage.aux[1]
    order = uid_order(previous.UIDs, state.state.UIDs)
    for name in PARTICLE_FIELDS | GLOBAL_FIELDS:
        old, new = getattr(previous, name), getattr(state.state, name)
        if isinstance(old, torch.Tensor):
            old = old[order] if name in PARTICLE_FIELDS else old
            if not isinstance(new, torch.Tensor) or not torch.equal(old, new):
                return None
        elif old != new:
            return None
    native = stage_view(stage, state.state.UIDs, native=True)
    endpoint = spatial.reuse
    # CompSPH.finalize replaces the last stage's dudt with the step-average
    # rate for diagnostics. FSAL must use the raw endpoint derivative instead.
    return native._replace(update=permute_update(endpoint.update, uid_order(endpoint.ids, state.state.UIDs)))


def adopt_spatial_continuation(memory):
    """Adapter for integrator runtime.solve(continuation=...), after acceptance."""
    from warpSPHIntegrators.runtime import Continuation
    from .sortedStep import _adopt_state
    if not isinstance(memory, Continuation):
        raise TypeError('expected an accepted Continuation')
    state = _adopt_state(memory.state)
    ids = state.state.UIDs
    return Continuation(state, history_view(memory.history, ids),
                        None if memory.prior_step is None else stage_view(memory.prior_step, ids))
