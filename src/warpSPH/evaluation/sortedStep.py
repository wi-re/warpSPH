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
    coverage_renewals: int = 0


@dataclass(frozen=True, eq=False)
class SpatialStageContext:
    layout: ParticleLayout
    adjacency: object
    edge_fields: dict
    reuse: object = None


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
    """Audited eager periodic CompSPH/CRK tableaus and fixed-step Adams/BDF.

    Construct with a registered RK/DIRK/coupled integrator. Call with state, dt,
    config and schemeConfig (the native RHS is selected from schemeName).
    History and explicit FSAL inputs are UID-adapted. Custom RHS, external
    row-indexed inputs and topology changes are not admitted. Resources are per call; failed
    or rejected calls publish no cache on their input or on this stepper.

    Sorting occurs at the first boundary, then only at an already-stale boundary
    list when sort_on_rebuild=True. Mid-step support/list renewals keep the chosen
    order. Pass sort_on_rebuild=False to retain the first compute order indefinitely.
    Implicit coverage loss may renew search envelopes and restart the complete
    step up to max_coverage_retries times (default 2); no Krylov basis is resumed.
    """
    def __init__(self, integrator, *, bits=21, sort_on_rebuild=True, observe=None, max_coverage_retries=2):
        from warpSPHIntegrators.schemes.definitions.explicit import ExplicitDefinition
        from warpSPHIntegrators.schemes.definitions.dirk import DIRKDefinition
        from warpSPHIntegrators.schemes.definitions.coupled import CoupledDefinition
        self.history_method = (getattr(integrator, 'identifier', None) is not None
            and integrator.identifier.name in {*(f'ab{i}' for i in range(2,6)),
                *(f'abm{i}' for i in range(2,6)), *(f'am{i}' for i in range(2,5)),
                *(f'bdf{i}' for i in range(1,6))}
            and getattr(integrator.function, '__module__', '') in {
                'warpSPHIntegrators.drivers.history.adams', 'warpSPHIntegrators.drivers.history.bdf'})
        if not self.history_method and not isinstance(getattr(integrator, 'definition', None), (ExplicitDefinition, DIRKDefinition, CoupledDefinition)):
            raise ValueError('MortonStepper requires an audited RK/DIRK/coupled tableau or fixed-step Adams/BDF method')
        if type(bits) is not int or not 1 <= bits <= 21 or type(sort_on_rebuild) is not bool:
            raise ValueError('bits must be in 1..21 and sort_on_rebuild must be bool')
        if type(max_coverage_retries) is not int or max_coverage_retries < 0:
            raise ValueError('max_coverage_retries must be a nonnegative integer')
        self.max_coverage_retries = max_coverage_retries
        self.integrator, self.bits, self.sort_on_rebuild = integrator, bits, sort_on_rebuild
        self.retain_stage_derivatives = False
        self.resources = CompressibleSpatialResources(observe=observe)

    def validate(self, state, *, config, schemeConfig):
        """Check fixed-layout eligibility without sorting or publishing resources."""
        from ..systems.compSPH import CompSPHSystem
        from ..modules.compSPH.supportGrid import solvesOnGrid
        if type(state) is not CompSPHSystem or not state.state.positions.shape[0]:
            raise TypeError('MortonStepper requires a nonempty CompSPHSystem')
        validate_state_schema(state.state)
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
        if schemeConfig.schemeName not in ('CompSPH', 'CRKSPH'):
            raise ValueError('MortonStepper supports CompSPH and CRKSPH only')
        if self.integrator.implicit and schemeConfig.compatibleEnergy:
            raise ValueError('implicit spatial evaluations require compatibleEnergy=False')
        if self.history_method and schemeConfig.compatibleEnergy:
            raise ValueError('multistep spatial evaluations require compatibleEnergy=False')

    def as_scheme(self):
        """Registered-scheme view for runtime.solve; pass native RHS and continuation adapter."""
        stepper = copy(self)
        # Generic runtime dense output consumes mathematical stage derivatives,
        # whereas the legacy frontend exposes a step-average diagnostic dudt.
        stepper.retain_stage_derivatives = True
        return self.integrator._replace(function=_SpatialDriver(stepper))

    def __call__(self, state, dt, *, config, schemeConfig, **kwargs):
        from ..schemes.compressible.compSPH import compSPH_step
        from ..schemes.compressible.crkSPH import crkSPH_step
        from warpSPHCore import verletStaleFlag
        self.validate(state, config=config, schemeConfig=schemeConfig)
        from .continuation import EndpointDerivative, history_view, stage_view, reusable_prior, reuse_stamp
        history = history_view(kwargs.pop('history', None), state.state.UIDs)
        prior = kwargs.pop('priorStep', None)
        if prior is not None:
            from warpSPHIntegrators.state import StageResult
            if type(prior) is tuple and len(prior) == 2:
                prior = StageResult(update=prior[0],aux=prior[1])
            if not self.integrator.fsal or self.integrator.implicit:
                raise ValueError('spatial priorStep reuse requires an explicit FSAL tableau')
            prior = stage_view(prior, state.state.UIDs)
        forbidden = [kwargs.pop(k, None) for k in ('warmStart', 'f')]
        if any(value is not None for value in forbidden):
            raise ValueError('MortonStepper does not yet adapt warmStart or custom RHS')
        if history is not None and history.maxlen < self.integrator.metadata.history_length:
            raise ValueError('insufficient continuation history capacity')
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
        functions = {'CompSPH': compSPH_step, 'CRKSPH': crkSPH_step}
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
        # Renewal is outside every solver/stage scope. Retry from the incoming
        # physical state, never the failed Newton iterate or a partial RK result.
        # Candidate search envelopes and their UIDs are the only retained data.
        from .coverage import CoverageInvalidated, CoverageRetryExhausted
        coverage, renewals = {}, 0
        while True:
            endpoint = None
            capture_endpoint = reuse_stamp(self.integrator, state, dt, config, schemeConfig) is not None
            native_rhs = functions[schemeConfig.schemeName]
            def rhs(system, dt, config, schemeConfig, verbose=False, stageIndex=None):
                nonlocal endpoint
                value = native_rhs(system, dt, config, schemeConfig, verbose=verbose, stageIndex=stageIndex)
                if capture_endpoint:
                    # Finalization may replace fields on the update object.
                    endpoint = copy(value[0])
                return value
            attemptState = computeState.initializeNewState() if self.integrator.implicit else computeState
            private = _system_view(state, attemptState, adjacency)
            private.retainStageDerivatives = self.history_method or self.retain_stage_derivatives
            inputs = dict(kwargs)
            if history is not None:
                inputs['history'] = history_view(history, computeState.UIDs, native=True)
            if prior is not None:
                candidate = reusable_prior(prior, self.integrator, state, dt, config, schemeConfig)
                if candidate is not None:
                    inputs['priorStep'] = stage_view(candidate, computeState.UIDs, native=True)
            try:
                with self.resources.session(coverage=coverage):
                    result = self.integrator(private, dt, rhs if capture_endpoint else native_rhs,
                                             config=config, schemeConfig=schemeConfig, **inputs)
                break
            except CoverageInvalidated as failure:
                if not self.integrator.implicit:
                    raise
                if renewals >= self.max_coverage_retries:
                    raise CoverageRetryExhausted(failure, renewals + 1) from failure
                envelope = failure.renewal.envelope
                if not torch.equal(envelope.input_ids, computeState.UIDs):
                    raise ValueError('cannot renew coverage in a different particle layout') from failure
                coverage[envelope.stage_path] = (envelope, envelope.build(config))
                renewals += 1
        if result.quadrature is not None or result.transition is not None:
            raise ValueError('MortonStepper cannot publish unadapted quadrature/transition data')
        if result.state.rerunRequest is not None:
            raise RuntimeError('unfinished compatible-energy replay cannot publish a spatial context')
        context = SpatialContext(layout, result.state.adjacency, settings, renewals)
        publicState = result.state.state if layout.identity else permute_state(result.state.state, layout.q)
        public = _system_view(result.state, publicState,
                              result.state.adjacency if layout.identity else None, context)
        public.canonicalExport = True
        stages = []
        stamp = reuse_stamp(self.integrator, result.state, dt, config, schemeConfig)
        reuse = None if stamp is None or endpoint is None else EndpointDerivative(stamp, computeState.UIDs.clone(), endpoint)
        for index, stage in enumerate(result.stages):
            adjacency, stageState = stage.aux
            stageContext = SpatialStageContext(layout, adjacency, {n: getattr(stageState, n) for n in EDGE_FIELDS},
                reuse if index == len(result.stages)-1 else None)
            stages.append(stage._replace(update=stage.update if layout.identity else permute_update(stage.update, layout.q),
                                  aux=(adjacency if layout.identity else None,
                                       stageState if layout.identity else permute_state(stageState, layout.q), stageContext)))
        error = result.error
        if error is not None and not layout.identity:
            error = _system_view(error, permute_state(error.state, layout.q))
        public_history = history_view(result.history, publicState.UIDs)
        if public_history is not None and public_history.latest is not None and reuse is not None:
            # RK bookkeeping records the last stage, the same endpoint used by
            # explicit priorStep. Adams/BDF never receive an FSAL stamp.
            latest = public_history.latest
            aux = latest.aux
            from warpSPHIntegrators.state.history import StepHistory
            latest = latest._replace(aux=(*aux[:2], replace(aux[2], reuse=reuse)))
            public_history = StepHistory(public_history.maxlen, (*public_history.entries[:-1], latest),
                                         _uid_fingerprint=public_history._uid_fingerprint)
        return replace(result, state=public, stages=stages, error=error, history=public_history)


class _SpatialDriver:
    """Keep tableau metadata available to the generic runtime without changing it."""
    def __init__(self, stepper):
        self.stepper = stepper

    def __getattr__(self, name):
        return getattr(self.stepper.integrator.function, name)

    def __call__(self, state, dt, f, *args, **kwargs):
        from ..schemes.compressible.compSPH import compSPH_step
        from ..schemes.compressible.crkSPH import crkSPH_step
        expected = compSPH_step if kwargs['schemeConfig'].schemeName == 'CompSPH' else crkSPH_step
        if args or f is not expected:
            raise ValueError('spatial runtime adapter requires native RHS and keyword configuration')
        if kwargs['schemeConfig'].compatibleEnergy:
            raise ValueError('spatial runtime integration requires compatibleEnergy=False')
        return self.stepper(state, dt, **kwargs)


def _adopt_state(accepted_state):
    """Pure next-state transformation after the caller has accepted the step.

    Does not transform retained observations. Continuation memory is transformed
    separately by the result/runtime adapters. Numerical
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
                        replace(context, layout=identity))


def adopt_spatial_layout(accepted_result):
    """Adopt only after application acceptance; carry error fields with the state.

    Historic stage snapshots retain their evaluated input order and explicit IDs.
    History follows the adopted state. Numerical failure cannot be accepted
    through this adapter; use adopt_spatial_continuation for runtime FSAL memory.
    """
    from warpSPHIntegrators.state import IntegrationResult
    if not isinstance(accepted_result, IntegrationResult) or not accepted_result.step_status.successful:
        raise ValueError('layout adoption requires a successful accepted step result')
    if accepted_result.quadrature is not None or accepted_result.transition is not None:
        raise ValueError('layout adoption does not yet adapt quadrature/transition')
    state = accepted_result.state
    context = state.spatialContext
    if context is None:
        raise ValueError('no spatial context to adopt')
    layout = context.layout.for_input(state.state.UIDs)
    error = accepted_result.error
    if error is not None and not layout.identity:
        error = _system_view(error, permute_state(error.state, layout.p))
    from .continuation import history_view
    adopted = _adopt_state(state)
    return replace(accepted_result, state=adopted, error=error,
                   history=history_view(accepted_result.history, adopted.state.UIDs))


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
