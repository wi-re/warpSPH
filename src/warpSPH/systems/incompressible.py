"""State and integrator adapters for native IISPH and the DFSPH compositions."""

from .identity import ParticleIdentityState
from warpSPHIntegrators.state import (
    BaseIntegrationSystem,
    ComponentUpdateSpec,
    PositionUpdateSpec,
    constant,
    get_reference_state,
    integrated,
    reference_state,
    tagged,
)
from warpSPHIntegrators.state.fields import update_component, update_position
from dataclasses import dataclass
import torch
from typing import Optional, Any
from warpSPHCore import AdjacencyList, DomainDescription


from ..modules.boundaryConditions.pinned import applyPinnedVelocity

__all__ = [
    "IncompressibleState",
    "IncompressibleSystemUpdate",
    "IncompressibleSystem",
    "ProjectedSystem",
    "IISPHSystem",
    "DFSPHState",
    "DFSPHSystem",
]


@dataclass
class IncompressibleState(ParticleIdentityState):
    positions: torch.Tensor = integrated("dxdt", tags=("position",))
    velocities: torch.Tensor = integrated("dvdt", tags=("velocity",))
    supports: torch.Tensor = constant(tags=("particle_support",))
    masses: torch.Tensor = constant(tags=("particle_mass",))
    densities: torch.Tensor = integrated("drhodt", tags=("density",))

    kinds: torch.Tensor = constant(tags=("particle_kind",))
    materials: torch.Tensor = constant(tags=("particle_material",))
    UIDs: torch.Tensor = constant(tags=("particle_UID",))
    UIDcounter: int = constant(tags=("particle_UIDcounter",))

    pressures: torch.Tensor = constant(tags=("damping",), default=None)
    soundspeeds: torch.Tensor = constant(tags=("soundSpeed",), default=None)
    surfaceIndicators: torch.Tensor = constant(tags=("surfaceIndicator",), default=None)
    surfaceNormals: torch.Tensor = constant(tags=("surfaceNormal",), default=None)
    surfaceLambdas: torch.Tensor = constant(tags=("surfaceLambda",), default=None)

    ghostIndices: torch.Tensor = constant(tags=("ghostIndices",), default=None)
    ghostOffsets: torch.Tensor = constant(tags=("ghostOffsets",), default=None)

    # See `WeaklyCompressibleState`'s identical field -- `rigidBody/update.py`
    # rebuilds either state type via the same keyword set, so both must carry
    # it (WCSPH_DEFAULT_CLOSEOUT_PLAN.md item D).
    boundaryAccelerations: torch.Tensor = constant(tags=("boundaryAcceleration",), default=None)


@dataclass
class IncompressibleSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=("position_derivative",))
    dvdt: torch.Tensor = tagged(tags=("velocity_derivative",))
    drhodt: torch.Tensor = tagged(tags=("density_derivative",))
    passive: Optional[torch.Tensor] = tagged(tags=("passive_derivative",), default=None)


@dataclass
class IncompressibleSystem(BaseIntegrationSystem):
    state: IncompressibleState = reference_state(tags=("physics_state",))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    canonicalExport: bool = False

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        if verbose:  # self.t may be a device tensor: format it only when printing
            print(f"Initializing new state [t={self.t}]")
        return type(self)(
            state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain, canonicalExport=self.canonicalExport
        )

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(
            self, update, spec, "position", "position_derivative", "velocity", "velocity_derivative"
        )

    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, "velocity", "velocity_derivative")

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # updatedsystem = update_component(self, update, spec, 'internalEnergy', 'internalEnergy_derivative')
        # updatedsystem = update_component(updatedsystem, update, spec, 'totalEnergy', 'totalEnergy_derivative')
        updatedsystem = update_component(self, update, spec, "density", "density_derivative")
        return updatedsystem

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self



@dataclass
class IISPHSystem(IncompressibleSystem):
    """Ordinary integration of x and v with summed density and pressure carry.

    Inherit the normal derivative updates. Finalization retains only fields
    recomputed by the RHS; positions and velocities belong to the integrator.
    """

    boundarySupport: Optional[float] = None
    analyticBoundaryProvider: Optional[Any] = None
    contactBoundaryProvider: Optional[Any] = None

    def initializeNewState(self, *args, verbose=False, **kwargs):
        system = super().initializeNewState(*args, verbose=verbose, **kwargs)
        system.boundarySupport = self.boundarySupport
        system.analyticBoundaryProvider = self.analyticBoundaryProvider
        system.contactBoundaryProvider = self.contactBoundaryProvider
        return system

    def finalize(self, initialState, dt, returnValues, updateValues, weights=..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]
        lastState = returnValues[-1][1]
        for name in ('supports', 'densities', 'pressures', 'soundspeeds',
                     'surfaceIndicators', 'surfaceNormals', 'surfaceLambdas'):
            value = getattr(lastState, name)
            target = getattr(self.state, name)
            if value is None:
                setattr(self.state, name, None)
            elif target is not None and target.shape == value.shape:
                target.copy_(value)
            else:
                setattr(self.state, name, value.clone())

        schemeConfig = kwargs.get('schemeConfig', None)
        if schemeConfig is not None:
            from ..rigidBody.stages import advanceUnforcedBodies
            advanceUnforcedBodies(self, dt, schemeConfig)
        applyPinnedVelocity(self.state, kwargs.get('config', None), schemeConfig)
        return BaseIntegrationSystem.finalize(
            self, initialState, dt, returnValues, updateValues, weights, *args, **kwargs)


@dataclass
class ProjectedSystem(IncompressibleSystem):
    """One full native state map; the outer driver must not kick/drift twice."""

    projected_map = True

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        if verbose:  # self.t may be a device tensor: format it only when printing
            print(f"Initializing new state [t={self.t}]")
        return type(self)(
            state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain, canonicalExport=self.canonicalExport
        )

    # The step already integrated x, v and rho; the integrator must not add
    # `dt * update` on top (the update is all zeros anyway, but the
    # semi-implicit position step would otherwise fold in `dt * v_start`).
    def apply_position_update(self, update, spec, **kwargs):
        return self

    def apply_velocity_update(self, update, spec, **kwargs):
        return self

    def apply_quantity_update(self, update, spec, **kwargs):
        return self

    def apply_state_update(self, update, spec, **kwargs):
        return self

    def finalize(self, initialState, dt, returnValues, updateValues, weights=..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]
        lastState = returnValues[-1][1]
        from ..modules.incompressible.stages import carryProjectionFields

        carryProjectionFields(self.state, lastState)

        schemeConfig = kwargs.get("schemeConfig", None)
        if schemeConfig is not None:
            from ..rigidBody.stages import advanceUnforcedBodies

            advanceUnforcedBodies(self, dt, schemeConfig)

        applyPinnedVelocity(
            self.state, kwargs.get("config", None), kwargs.get("schemeConfig", None)
        )
        # The full map already advanced positions and velocities.
        return BaseIntegrationSystem.finalize(
            self, initialState, dt, returnValues, updateValues, weights, *args, **kwargs
        )


@dataclass
class DFSPHState(IncompressibleState):
    """Physical projection pressures and independent timestep-scaled histories."""

    divergencePressures: Optional[torch.Tensor] = constant(default=None, tags=('pressure',))
    densityPressureImpulse: Optional[torch.Tensor] = constant(default=None, tags=('pressure_history',))
    divergencePressureImpulse: Optional[torch.Tensor] = constant(default=None, tags=('pressure_history',))


@dataclass
class DFSPHSystem(ProjectedSystem):
    """One native DFSPH staged map, with IISPH's resolved static geometry."""

    boundarySupport: Optional[float] = None
    analyticBoundaryProvider: Optional[Any] = None
    contactBoundaryProvider: Optional[Any] = None

    def initializeNewState(self, *args, verbose=False, **kwargs):
        system = super().initializeNewState(*args, verbose=verbose, **kwargs)
        system.boundarySupport = self.boundarySupport
        system.analyticBoundaryProvider = self.analyticBoundaryProvider
        system.contactBoundaryProvider = self.contactBoundaryProvider
        return system

    def finalize(self, initialState, dt, returnValues, updateValues, weights=..., *args, **kwargs):
        lastState = returnValues[-1][1]
        for name in ('divergencePressures', 'densityPressureImpulse', 'divergencePressureImpulse'):
            value = getattr(lastState, name)
            setattr(self.state, name, None if value is None else value.clone())
        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
