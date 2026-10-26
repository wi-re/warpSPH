"""Conserved MFM/MFV state and spatial lifecycle hooks.

The integrators package owns time staging. Geometry and primitive refresh run
through initialize/preprocess/finalize. Supports retain their historical
start-of-step policy; diagnostics describe the accepted conserved state. Derived fields use
copied semantics so stage clones do not duplicate values about to be recomputed.
"""

from .identity import ParticleIdentityState
from warpSPHIntegrators.state import BaseIntegrationSystem, BaseState, ComponentUpdateSpec, PositionUpdateSpec, constant, copied, get_reference_state, integrated, reference_state, tagged
from warpSPHIntegrators.state.fields import update_component, update_position
from dataclasses import dataclass
import torch
from typing import Optional
from warpSPHCore import *

__all__ = ['MeshlessState', 'MeshlessSystemUpdate', 'MeshlessSystem']


@dataclass
class MeshlessState(ParticleIdentityState):
    positions: torch.Tensor = integrated('dxdt', tags=('position',))
    # the conserved vector (m, m v, m e_tot); the single integrated physics quantity
    Q: torch.Tensor = integrated('dQdt', tags=('conserved',))
    supports: torch.Tensor = constant(tags=('particle_support',))

    kinds: torch.Tensor = constant(tags=('particle_kind',))
    materials: torch.Tensor = constant(tags=('particle_material',))
    UIDs: torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter: int = constant(tags=('particle_UIDcounter',))

    # the previous step's closure potential (N, dim): warm start for the CG solve
    lam: Optional[torch.Tensor] = constant(tags=('closure_lam',), default=None)
    # derived primitives (diagnostics), refreshed by the step and copied in finalize
    masses: torch.Tensor = copied(tags=('particle_mass',), default=None)
    densities: torch.Tensor = copied(tags=('particle_density',), default=None)
    velocities: torch.Tensor = copied(tags=('velocity',), default=None)
    internalEnergies: torch.Tensor = copied(tags=('internalEnergy',), default=None)
    totalEnergies: torch.Tensor = copied(tags=('energy',), default=None)
    entropies: torch.Tensor = copied(tags=('entropy',), default=None)
    pressures: torch.Tensor = copied(tags=('pressure',), default=None)
    soundspeeds: torch.Tensor = copied(tags=('soundSpeed',), default=None)


@dataclass
class MeshlessSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dQdt: torch.Tensor = tagged(tags=('conserved_derivative',))


@dataclass
class MeshlessSystem(BaseIntegrationSystem):
    state: MeshlessState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    meshlessStages: object = None
    drivenParticles: object = None
    inflowReservoir: object = None

    def initialize(self, dt, *args, conservedIntegration=False, **kwargs):
        if conservedIntegration:
            from ..modules.meshless.stages import MeshlessStages
            self.meshlessStages = MeshlessStages()
        return super().initialize(dt, *args, **kwargs)

    def preprocess(self, initialState, dt, *args, conservedIntegration=False, stageIndex=0, **kwargs):
        if conservedIntegration:
            from ..modules.meshless.stages import prepareStage
            prepareStage(self, dt, kwargs['config'], kwargs['schemeConfig'], stageIndex)
        return self

    def conserved_velocity(self, Q):
        return Q[:, 1:-1] / Q[:, 0:1]

    def wrap_positions(self, positions):
        from ..modules.meshless.context import wrap
        return wrap(positions, self.domain)

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        if verbose:   # self.t may be a device tensor: format it only when printing
            print(f'Initializing new state [t={self.t}]')
        newState = state.initializeNewState()
        if not kwargs.get('conservedIntegration', False):
            # The runner also clones once before its initial diagnostics, outside
            # an RHS lifecycle. Preserve those values for that public snapshot.
            for name in ('masses', 'densities', 'velocities', 'internalEnergies',
                         'totalEnergies', 'entropies', 'pressures', 'soundspeeds'):
                value = getattr(state, name)
                if value is not None:
                    setattr(newState, name, value.clone())
        from ..modules.meshless.geometry import inheritGeometry
        inheritGeometry(state, newState)
        return MeshlessSystem(state=newState, adjacency=self.adjacency, t=self.t, domain=self.domain, meshlessStages=self.meshlessStages, drivenParticles=self.drivenParticles, inflowReservoir=self.inflowReservoir)

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        # Legacy exact-delta callers apply the already time-centred net drift
        # here. The package-owned ConservedLagrangian driver stages positions
        # directly; it does not reinterpret this delta as an ordinary RHS.
        # Conserved momentum remains in Q; the velocity tag fills only the
        # generic update helper's unused kick slot for the legacy Euler path.
        update_position(self, update, spec, 'position', 'position_derivative', 'velocity')
        # wrap into the domain (periodic axes): the drift can cross a periodic edge, and the
        # core driver wraps at every move
        if self.domain is not None:
            self.state.positions = torch.where(
                self.domain.periodic,
                self.domain.min + torch.remainder(self.state.positions - self.domain.min,
                                                  self.domain.max - self.domain.min),
                self.state.positions)
        return self

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # advance the conserved vector: Q += dt * dQdt
        return update_component(self, update, spec, 'conserved', 'conserved_derivative')

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self

    def finalize(self, initialState, dt, returnValues, updateValues, weights=..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]
        lastState = returnValues[-1][1]
        if kwargs.get('conservedIntegration', False):
            # Time stepping has already accepted (x,Q). Preserve the support
            # update and closure warm start used by the historical method, but
            # never overwrite its diagnostics with a previous stage's (v,u,m).
            self.state.supports = self.meshlessStages.startState.supports.clone()
            self.state.lam = lastState.lam.clone() if lastState.lam is not None else None
            config = kwargs.get('config', args[0] if args else None)
            scheme = kwargs.get('schemeConfig', args[1] if len(args) > 1 else None)
            from ..modules.meshless.geometry import buildGeometry, stashGeometry
            from ..modules.meshless.primitives import refreshPrimitives, particleGamma
            if self.drivenParticles is not None:
                self.drivenParticles.position(self.state, self.t)
            geometry = (self.inflowReservoir.build(self.state, config, scheme, self.t)
                        if self.inflowReservoir is not None else
                        buildGeometry(self.state.positions, self.state.supports, self.state.lam,
                                      config, scheme, self.domain))
            if self.drivenParticles is not None:
                self.drivenParticles.conserved(self.state, geometry, self.t, float(scheme.gamma))
            refreshPrimitives(self.state, geometry, particleGamma(self.state, scheme), scheme.nngb, config.dim,
                              updateSupports=False)
            if self.inflowReservoir is None:
                stashGeometry(self.state.positions, self.state.supports, self.domain,
                              config.kernel, scheme, geometry, owner=self.state)
            return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
        self.state.supports.copy_(lastState.supports)
        lam = lastState.lam
        if self.state.lam is not None and lam is not None:
            self.state.lam.copy_(lam)
        for _f in ('masses', 'densities', 'velocities', 'internalEnergies', 'totalEnergies',
                   'entropies', 'pressures', 'soundspeeds'):
            value = getattr(lastState, _f)
            if value is not None:
                if getattr(self.state, _f) is None:
                    setattr(self.state, _f, value.clone())
                else:
                    getattr(self.state, _f).copy_(value)
        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
