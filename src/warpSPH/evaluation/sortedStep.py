"""Opt-in fixed-count Morton stepping at the simulator/integrator boundary.

One private coordinate order covers the entire step, including compatible-energy
replays and JFNK. Public state, errors and derivatives retain the caller's order.
Accepted adoption is explicit; retained stage records remain in evaluated order.
"""
from copy import copy
from dataclasses import dataclass, replace

import torch

from .compressible import CompressibleSpatialResources
from .layout import (EDGE_FIELDS, ParticleLayout, morton_order, permute_state,
                     permute_update, remap_adjacency, validate_state_schema)


@dataclass(frozen=True, eq=False)
class SpatialContext:
    layout: ParticleLayout
    adjacency: object
    settings: tuple


@dataclass(frozen=True, eq=False)
class SpatialStageContext:
    layout: ParticleLayout
    adjacency: object
    edge_fields: dict


def _system_view(system, particleState, adjacency=None, context=None):
    result = copy(system)
    result.state, result.adjacency, result.spatialContext = particleState, adjacency, context
    result.pairWork = result.supportGrid = result.listGrid = result.rerunRequest = None
    return result


def _settings(state, config, bits):
    domain = config.domain
    # Domain objects are immutable resources for a cache lifetime. Pointer/version
    # metadata catches replacement and ordinary in-place edits without device reads.
    bounds = tuple((value.data_ptr(), value._version, tuple(value.shape))
                   for value in (domain.min, domain.max, domain.periodic))
    return (bounds, domain.dim, float(config.verletScale),
            str(state.positions.dtype), str(state.positions.device), bits)


class MortonStepper:
    """Audited eager periodic CompSPH/CRK one-step tableaus; sorting is opt-in.

    Construct with a registered RK/DIRK/coupled integrator. Call with state, dt,
    config and schemeConfig (the native RHS is selected from schemeName).
    No history, FSAL reuse, custom RHS, external row-indexed inputs or topology
    changes are admitted by this first adapter. Resources are per call; failed
    or rejected calls publish no cache on their input or on this stepper.

    Sorting occurs at the first boundary, then only at an already-stale boundary
    list when sort_on_rebuild=True. Mid-step support/list renewals keep the chosen
    order. Pass sort_on_rebuild=False to retain the first compute order indefinitely.
    """
    def __init__(self, integrator, *, bits=21, sort_on_rebuild=True, observe=None):
        from warpSPHIntegrators.schemes.definitions.explicit import ExplicitDefinition
        from warpSPHIntegrators.schemes.definitions.dirk import DIRKDefinition
        from warpSPHIntegrators.schemes.definitions.coupled import CoupledDefinition
        if not isinstance(getattr(integrator, 'definition', None), (ExplicitDefinition, DIRKDefinition, CoupledDefinition)):
            raise ValueError('MortonStepper requires a registered one-step RK, DIRK or coupled tableau')
        if type(bits) is not int or not 1 <= bits <= 21 or type(sort_on_rebuild) is not bool:
            raise ValueError('bits must be in 1..21 and sort_on_rebuild must be bool')
        self.integrator, self.bits, self.sort_on_rebuild = integrator, bits, sort_on_rebuild
        self.resources = CompressibleSpatialResources(observe=observe)

    def __call__(self, state, dt, *, config, schemeConfig, **kwargs):
        from ..systems.compSPH import CompSPHSystem
        from ..schemes.compressible.compSPH import compSPH_step
        from ..schemes.compressible.crkSPH import crkSPH_step
        from ..modules.compSPH.supportGrid import solvesOnGrid
        from warpSPHCore import verletStaleFlag
        if type(state) is not CompSPHSystem or not state.state.positions.shape[0]:
            raise TypeError('MortonStepper requires a nonempty CompSPHSystem')
        validate_state_schema(state.state)
        forbidden = [kwargs.pop(k, None) for k in ('history', 'priorStep', 'warmStart', 'f')]
        if any(value is not None for value in forbidden):
            raise ValueError('MortonStepper does not yet adapt history/reuse/warmStart or custom RHS')
        solver, options = kwargs.get('solver'), kwargs.get('solver_opts') or {}
        if self.integrator.implicit:
            from warpSPHIntegrators.solvers import JFNKSolver
            if solver is not None and not isinstance(solver, JFNKSolver):
                raise ValueError('MortonStepper implicit qualification currently requires FD-JFNK')
            if getattr(solver, 'preconditioner', None) is not None or options.get('preconditioner') is not None:
                raise ValueError('preconditioners need an explicit vector/input layout adapter')
        if any(isinstance(kwargs.get(k), torch.Tensor) and kwargs[k].ndim for k in ('rtol', 'atol')):
            raise ValueError('particle-valued tolerances need an explicit layout adapter')
        if any(name.startswith('_') for name in kwargs):
            raise ValueError('MortonStepper requires a complete outer step')
        if not bool(config.domain.periodic.all()) or bool((state.state.kinds != 0).any()):
            raise ValueError('MortonStepper currently requires periodic fluid-only particles')
        if state.sourceTerms is not None or schemeConfig.boundaryConditions or getattr(schemeConfig, 'rigidBodies', None):
            raise ValueError('external sources/boundaries require explicit particle input adapters before sorting')
        if solvesOnGrid(schemeConfig) or getattr(schemeConfig, 'inPlaceAdjacency', False):
            raise ValueError('MortonStepper currently requires eager list mode')
        if getattr(schemeConfig, 'crkEvolution', 'summation') != 'summation':
            raise ValueError('MortonStepper currently requires summation CRK evolution')
        if getattr(schemeConfig, 'crkEnergyPartition', 'native') != 'native' or getattr(schemeConfig, 'crkEnergyPredictor', 'raw') != 'raw':
            raise ValueError('special CRK energy controls require separate sorting qualification')
        functions = {'CompSPH': compSPH_step, 'CRKSPH': crkSPH_step}
        if schemeConfig.schemeName not in functions:
            raise ValueError('MortonStepper supports CompSPH and CRKSPH only')
        settings = _settings(state.state, config, self.bits)
        old = state.spatialContext
        if old is not None and old.settings != settings:
            raise ValueError('spatial context settings changed; explicitly discard it before rebuilding')
        if old is None:
            layout = ParticleLayout.create(state.state.UIDs, morton_order(state.state.positions, config.domain, bits=self.bits))
            adjacency = None
        else:
            layout = old.layout.for_input(state.state.UIDs)
            adjacency = old.adjacency
        # Audited native initializers only assign metadata on this object; their
        # numerical stages clone buffers. Identity continuation needs no gather.
        computeState = copy(state.state) if layout.identity else permute_state(state.state, layout.p)
        if old is not None and self.sort_on_rebuild and (adjacency is None or bool(verletStaleFlag(
                computeState.positions, computeState.supports, adjacency, config.domain, config.verletScale))):
            order = morton_order(computeState.positions, config.domain, bits=self.bits)
            if not torch.equal(order, torch.arange(order.numel(), device=order.device)):
                # Compose B->C with the actual incoming A->B mapping, never with old input rows.
                layout = ParticleLayout.create(state.state.UIDs, layout.p[order])
                computeState = permute_state(state.state, layout.p, identity=layout.identity)
                adjacency = None
        private = _system_view(state, computeState, adjacency)
        with self.resources.session():
            result = self.integrator(private, dt, functions[schemeConfig.schemeName],
                                     config=config, schemeConfig=schemeConfig, **kwargs)
        if result.history is not None or result.quadrature is not None or result.transition is not None:
            raise ValueError('MortonStepper cannot publish unadapted history/quadrature/transition data')
        if result.state.rerunRequest is not None:
            raise RuntimeError('unfinished compatible-energy replay cannot publish a spatial context')
        context = SpatialContext(layout, result.state.adjacency, settings)
        publicState = result.state.state if layout.identity else permute_state(result.state.state, layout.q)
        public = _system_view(result.state, publicState,
                              result.state.adjacency if layout.identity else None, context)
        public.canonicalExport = True
        stages = []
        for stage in result.stages:
            adjacency, stageState = stage.aux
            stageContext = SpatialStageContext(layout, adjacency, {n: getattr(stageState, n) for n in EDGE_FIELDS})
            stages.append(stage._replace(update=stage.update if layout.identity else permute_update(stage.update, layout.q),
                                  aux=(adjacency if layout.identity else None,
                                       stageState if layout.identity else permute_state(stageState, layout.q), stageContext)))
        error = result.error
        if error is not None and not layout.identity:
            error = _system_view(error, permute_state(error.state, layout.q))
        return replace(result, state=public, stages=stages, error=error)


def _adopt_state(accepted_state):
    """Pure next-state transformation after the caller has accepted the step.

    Does not transform retained stages or histories. MortonStepper rejects those
    continuation inputs; use this only on its accepted one-step state. Numerical
    finalization, compatible-energy work and reruns have already completed.
    """
    context = accepted_state.spatialContext
    if context is None:
        raise ValueError('no spatial context to adopt')
    layout = context.layout.for_input(accepted_state.state.UIDs)
    if layout.identity:
        return accepted_state
    state = permute_state(accepted_state.state, layout.p)
    identity = ParticleLayout.create(state.UIDs, torch.arange(state.UIDs.numel(), device=state.UIDs.device))
    return _system_view(accepted_state, state, context.adjacency,
                        SpatialContext(identity, context.adjacency, context.settings))


def adopt_spatial_layout(accepted_result):
    """Adopt only after application acceptance; carry error fields with the state.

    Historic stage snapshots retain their evaluated input order and explicit IDs.
    Reuse/history are rejected by MortonStepper; no unadapted continuation memory
    is published. Numerical failure cannot be accepted through this adapter.
    """
    from warpSPHIntegrators.state import IntegrationResult
    if not isinstance(accepted_result, IntegrationResult) or not accepted_result.step_status.successful:
        raise ValueError('layout adoption requires a successful accepted step result')
    if accepted_result.history is not None or accepted_result.quadrature is not None or accepted_result.transition is not None:
        raise ValueError('layout adoption does not yet adapt continuation history/quadrature/transition')
    state = accepted_result.state
    context = state.spatialContext
    if context is None:
        raise ValueError('no spatial context to adopt')
    layout = context.layout.for_input(state.state.UIDs)
    error = accepted_result.error
    if error is not None and not layout.identity:
        error = _system_view(error, permute_state(error.state, layout.p))
    return replace(accepted_result, state=_adopt_state(state), error=error)


def canonical_view(system, stages=None, *, include_adjacency=True, include_stage_adjacency=True):
    """Owned UID-ordered frame/stage view, including independently remapped CSR.

    An export view is not a continuation checkpoint and carries no compute cache.
    All old snapshots and build-time anchors retain their original meaning.
    """
    if type(include_adjacency) is not bool or type(include_stage_adjacency) is not bool:
        raise TypeError('adjacency export choices must be bool')
    order = torch.argsort(system.state.UIDs, stable=True)
    context = system.spatialContext
    layout = context.layout.for_input(system.state.UIDs) if context is not None else None
    adjacency = context.adjacency if context is not None else system.adjacency
    reference_order = layout.q[order] if layout is not None else order
    view = _system_view(system, permute_state(system.state, order),
                        None if adjacency is None or not include_adjacency else remap_adjacency(adjacency, reference_order))
    view.canonicalExport = False
    if stages is None:
        return view, None
    outputs = []
    for stage in stages:
        adjacency, stageState, *extra = stage.aux
        order = torch.argsort(stageState.UIDs, stable=True)
        values = permute_state(stageState, order)
        edge_fields = {name: getattr(stageState, name) for name in EDGE_FIELDS}
        reference_order = order
        if extra:
            if len(extra) != 1 or not isinstance(extra[0], SpatialStageContext):
                raise ValueError('unknown stage auxiliary data cannot be canonically exported')
            spatial = extra[0]
            layout = spatial.layout.for_input(stageState.UIDs)
            adjacency = spatial.adjacency
            reference_order = layout.q[order]
            edge_fields = spatial.edge_fields
        if include_stage_adjacency and adjacency is not None:
            adjacency, edge_order = remap_adjacency(adjacency, reference_order, return_edge_order=True)
            for name, value in edge_fields.items():
                if value is not None:
                    reordered = value.clone()
                    reordered[:edge_order.numel()] = value[edge_order]
                    setattr(values, name, reordered)
        else:
            # Unkeyed pair arrays are omitted together with their adjacency.
            adjacency = None
        outputs.append(stage._replace(update=permute_update(stage.update, order), aux=(adjacency, values)))
    return view, outputs
