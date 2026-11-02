"""Eager, fixed-layout CompSPH/CRKSPH evaluation resources.

This first adapter isolates speculative scratch and pins a neighbor list per
stage of a JFNK linearization. Support is still solved for every physical input.
Coverage loss aborts the whole solve; callers renew/retry from owned input.
Sorting, population changes, graph capture and differentiable implicit support
solves are separate qualification gates.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..modules.compressibleStage import _ACTIVE, _prepare
from ..modules.compSPH.pairWork import beginPairWork
from ..modules.compSPH.supportGrid import solvesOnGrid


@dataclass
class _Evaluation:
    manager: object
    request: object
    pins: dict = field(default_factory=dict)
    prepared: object = None
    inputs: tuple = ()

    def prepare(self, system, config, schemeConfig, stageIndex):
        from warpSPHIntegrators.state import ResourceInvalidated
        request = self.request
        if request.kind != 'rhs' or request.state is not system:
            raise RuntimeError('compressible preparation requires the integrator RHS lifecycle')
        if self.prepared is not None:
            raise RuntimeError('a numerical evaluation may prepare its stage only once')
        if request.purpose in ('linear_action', 'preconditioner'):
            raise RuntimeError('an operator vector cannot be used as particle geometry')
        if request.purpose == 'jvp_probe':
            raise NotImplementedError('compressible spatial resources currently support FD JFNK, not exact JVP')
        if getattr(schemeConfig, 'schemeName', '') not in ('CompSPH', 'CRKSPH'):
            raise ValueError('spatial resources currently support CompSPH and CRKSPH only')
        if solvesOnGrid(schemeConfig) or getattr(schemeConfig, 'inPlaceAdjacency', False):
            raise ValueError('compressible spatial resources currently require eager list-mode support solves')
        implicit = request.find('nonlinear_solve') is not None or request.purpose == 'materialize'
        if implicit and schemeConfig.compatibleEnergy:
            raise ValueError('implicit spatial evaluations require compatibleEnergy=False; pair-work quadrature is RK-specific')
        if implicit and getattr(schemeConfig, 'crkEvolution', 'summation') != 'summation':
            raise ValueError('implicit spatial evaluations currently require summation CRK evolution')
        if implicit and system.sourceTerms is not None:
            raise ValueError('implicit spatial evaluations do not yet own external source transactions')
        linearization = request.find('linearization')
        pins = linearization.resource.pins if linearization is not None else None
        key = request.stage_path
        pinned = pins.get(key) if pins is not None else None
        if pinned is not None:
            system.adjacency = pinned
        label = 'crkSPH' if getattr(schemeConfig, 'schemeName', '') == 'CRKSPH' else 'compSPH'
        prepared = _prepare(system, config, schemeConfig, stageIndex, label)
        if pinned is not None and prepared.adjacency is not pinned:
            raise ResourceInvalidated(
                f'neighbor coverage changed in stage {key}; renew coverage and retry the complete solve')
        if pins is not None and pinned is None:
            pins[key] = prepared.adjacency
        self.prepared = prepared
        self.inputs = (system, config, schemeConfig, stageIndex)
        if self.manager.observe is not None:
            self.manager.observe(request, prepared)

    def consume(self, system, config, schemeConfig, stageIndex):
        if self.prepared is None:
            raise RuntimeError('spatial resources require preprocessing before the equation callback')
        oldSystem, oldConfig, oldScheme, oldIndex = self.inputs
        if (system is not oldSystem or config is not oldConfig or schemeConfig is not oldScheme or stageIndex != oldIndex):
            raise RuntimeError('prepared spatial context belongs to a different stage input')
        return self.prepared


class CompressibleSpatialResources:
    """Opt in with ``with resources.session(): driver(..., f=compSPH_step)``.

    Ordinary explicit RK retains its cross-stage compatible-energy work. JFNK
    residuals/probes/trials get private particle scratch and PairWork. Neighbor
    lists are out-of-place list-mode builds, frozen within each linearization,
    independently for coupled stages. Aux keeps the existing (adjacency, state)
    contract, so converged materialization still supplies finalization fields.

    ``observe(request, prepared)`` is an optional diagnostic callback. It must
    treat handles as read-only; the manager keeps no growing evaluation log.
    No automatic renewal or time-step acceptance is performed here.
    """
    def __init__(self, *, observe=None):
        if observe is not None and not callable(observe):
            raise TypeError('observe must be callable or None')
        self.observe = observe

    @contextmanager
    def session(self):
        # Lazy dependency: existing simulations still work with integrators 0.7.
        from warpSPHIntegrators.state import evaluation_resources
        with evaluation_resources(self):
            yield self

    @contextmanager
    def scope(self, request):
        evaluation = _Evaluation(self, request)
        saved = None
        system = request.state
        speculative = request.kind == 'rhs' and request.find('nonlinear_solve') is not None
        if speculative:
            # The RHS may update supports, EOS/switch fields and pair work.
            # Keep the nonlinear input and every other evaluation's scratch intact.
            saved = system.state, system.adjacency, system.pairWork
            system.state = system.state.initializeNewState()
            system.pairWork = beginPairWork(None)
        token = _ACTIVE.set(evaluation)
        try:
            yield evaluation
        finally:
            _ACTIVE.reset(token)
            if saved is not None:
                system.state, system.adjacency, system.pairWork = saved

    def freeze(self):
        raise NotImplementedError(
            'implicit differentiation of compressible spatial preparation is not qualified; use torch.no_grad() for FD JFNK')
