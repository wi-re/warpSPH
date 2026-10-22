"""Weakly-compressible state and lifecycle adapter.

The integrators package owns RK/Verlet arithmetic. Modules own stage memory,
post-step shifting/contact physics, body advancement and lazy diagnostics.
"""
from dataclasses import dataclass
from typing import Optional
import copy
import torch
from warpSPHCore import *
from warpSPHIntegrators.state import BaseIntegrationSystem, ComponentUpdateSpec, PositionUpdateSpec, constant, get_reference_state, integrated, reference_state, tagged
from warpSPHIntegrators.state.fields import update_component, update_position
from .identity import ParticleIdentityState
from ..modules.mdbc._util import stateHasBoundaryParticles
from ..modules.mdbc.corrections import noPenShift, applyNoPenetration, _meanBoundaryNormal
from ..modules.shifting.transport import shiftTransport
from ..modules.boundaryConditions.pinned import applyPinnedVelocity
from ..modules.diagnostics.weaklyCompressible import _LazyStepDiagnostics, _statBlock, _accelStatBlocks
from ..rigidBody.stages import advanceBodies

__all__ = ['WeaklyCompressibleState', 'WeaklyCompressibleSystemUpdate', 'WeaklyCompressibleSystem']
# Existing unvalidated ceiling option stays disabled; no model promotion here.
_RESTORE_GRAVITY_ON_CEILING_NOPEN = False


@dataclass
class WeaklyCompressibleState(ParticleIdentityState):
    positions : torch.Tensor = integrated('dxdt', tags=('position',))
    velocities: torch.Tensor = integrated('dvdt', tags=('velocity',))
    supports : torch.Tensor = constant(tags=('particle_support',))
    masses : torch.Tensor = constant(tags=('particle_mass',))
    # A drift field (warpSPHIntegrators/drift.py): the continuity rate's
    # velocity-linear part (`-rho div v`, reported as `drhodt_kin`) is advanced
    # with the drifts, like the positions, under the Verlet-family integrators when
    # `schemeConfig.timeCentredContinuity` is on (CEILING_STICKING_PLAN.md §3, §6).
    densities : torch.Tensor = integrated('drhodt', tags=('density',), drift_key='drhodt_kin')

    kinds : torch.Tensor = constant(tags=('particle_kind',))
    materials : torch.Tensor = constant(tags=('particle_material',))
    UIDs : torch.Tensor = constant(tags=('particle_UID',))
    UIDcounter : int = constant(tags=('particle_UIDcounter',))

    pressures : torch.Tensor = constant(tags=('damping',), default=None)
    soundspeeds : torch.Tensor = constant(tags=('soundSpeed',), default=None)
    surfaceIndicators : torch.Tensor = constant(tags=('surfaceIndicator',), default=None)
    surfaceNormals : torch.Tensor = constant(tags=('surfaceNormal',), default=None)
    surfaceLambdas : torch.Tensor = constant(tags=('surfaceLambda',), default=None)

    ghostIndices : torch.Tensor = constant(tags=('ghostIndices',), default=None)
    ghostOffsets : torch.Tensor = constant(tags=('ghostOffsets',), default=None)

    # Per-particle rigid-body acceleration at boundary/ghost rows (zero
    # elsewhere), written by `rigidBody/update.py`'s `updateBodyParticlesWCSPH`
    # every step. `None` until at least one rigid body has been updated once
    # (readers must treat `None` as "zero everywhere", the same fallback this
    # field replaces -- `modules/mdbc/english2025.py`'s `a_b` used to be
    # hardcoded to zero because no such field existed at all;
    # WCSPH_DEFAULT_CLOSEOUT_PLAN.md item D).
    boundaryAccelerations : torch.Tensor = constant(tags=('boundaryAcceleration',), default=None)

@dataclass
class WeaklyCompressibleSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
    drhodt: torch.Tensor = tagged(tags=('density_derivative',))
    passive: Optional[torch.Tensor] = tagged(tags=('passive_derivative',), default=None)
    #: the continuity part of `drhodt` (`-rho div v`, wall pairs included), the
    #: density's velocity-linear rate; None from schemes that do not report it
    drhodt_kin: Optional[torch.Tensor] = tagged(default=None)


@dataclass
class WeaklyCompressibleSystem(BaseIntegrationSystem):
    state: WeaklyCompressibleState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    deltaStages: object = None
    def initialize(self, dt, *args, **kwargs):
        config = kwargs.get('schemeConfig')
        if any(getattr(config, name, False) for name in ('freezeDiffusionAcrossStages', 'freezeSurfaceMaskAcrossStages', 'surfaceMaskDiagnostics')):
            from ..modules.deltaSPH.stages import DeltaStages
            previous = self.deltaStages.surfaceStart if self.deltaStages is not None else None
            self.deltaStages = DeltaStages(previousSurfaceStart=previous)
        else:
            self.deltaStages = None
        return super().initialize(dt, *args, **kwargs)

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        if verbose:   # self.t may be a device tensor: format it only when printing
            print(f'Initializing new state [t={self.t}]')
        return WeaklyCompressibleSystem(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain, deltaStages=self.deltaStages)

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative', 'velocity', 'velocity_derivative')
    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # updatedsystem = update_component(self, update, spec, 'internalEnergy', 'internalEnergy_derivative')
        # updatedsystem = update_component(updatedsystem, update, spec, 'totalEnergy', 'totalEnergy_derivative')
        updatedsystem = update_component(self, update, spec, 'density', 'density_derivative')
        return updatedsystem

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self
    
    def drift_fields_enabled(self, schemeConfig=None, **kwargs):
        """Density as a drift field (warpSPHIntegrators/drift.py): on with
        `schemeConfig.timeCentredContinuity` (CEILING_STICKING_PLAN.md §3, §6)."""
        return bool(getattr(schemeConfig, 'timeCentredContinuity', False))

    def drift_rates(self, velocities, aux=None, config=None, schemeConfig=None, **kwargs):
        """The continuity rate `-rho div v` at this system's configuration with the
        fluid moving at `velocities` (wall rows re-derived from them by the wall
        BC, as the right-hand side does) -- the density's velocity-linear rate,
        `drhodt_kin`. Evaluated on a shallow copy; `aux` is the stage's
        `(adjacency, state)`, else this system's own adjacency."""
        from ..modules.mdbc.velocity import computeBoundaryVelocities
        from ..modules.momentum.inconsistent import computeMomentum
        adjacency = aux[0] if aux else self.adjacency
        state = copy.copy(self.state)
        fluid = (state.kinds == 0).unsqueeze(-1)
        state.velocities = torch.where(fluid, velocities, state.velocities)
        if stateHasBoundaryParticles(state, config):
            state.velocities = computeBoundaryVelocities(state, config, schemeConfig, adjacency)
        # analytic walls: `computeMomentum` adds the wall flux of the continuity equation for the velocities of this evaluation (the right-hand side adds the
        # same term for the stage velocity; the time-centred correction must see it for the mean velocity, or the density at a wall is advanced without
        # the wall's compression)
        rate = computeMomentum(state, config, schemeConfig, adjacency)
        rate = torch.where(fluid.squeeze(-1), rate, torch.zeros_like(rate))
        return WeaklyCompressibleSystemUpdate(dxdt=None, dvdt=None, drhodt=None, drhodt_kin=rate)
    def _noPenShift(self, config, schemeConfig):
        return noPenShift(self, config, schemeConfig)

    def finalize(self, initialState, dt, returnValues, updateValues, weights = ..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]  # Assuming the adjacency list is the last return value from the derivative function
        lastState = returnValues[-1][1]  # Assuming the state is the second return value from the derivative function
        self.state.supports.copy_(lastState.supports)

        if self.state.pressures is not None and lastState.pressures is not None:
            self.state.pressures.copy_(lastState.pressures)
        else:
            self.state.pressures = lastState.pressures.clone() if lastState.pressures is not None else None
        if self.state.soundspeeds is not None and lastState.soundspeeds is not None:
            self.state.soundspeeds.copy_(lastState.soundspeeds)
        else:
            self.state.soundspeeds = lastState.soundspeeds.clone() if lastState.soundspeeds is not None else None

        if self.state.surfaceIndicators is not None and lastState.surfaceIndicators is not None:
            self.state.surfaceIndicators.copy_(lastState.surfaceIndicators)
        else:
            self.state.surfaceIndicators = lastState.surfaceIndicators.clone() if lastState.surfaceIndicators is not None else None

        if self.state.surfaceNormals is not None and lastState.surfaceNormals is not None:
            self.state.surfaceNormals.copy_(lastState.surfaceNormals)
        else:
            self.state.surfaceNormals = lastState.surfaceNormals.clone() if lastState.surfaceNormals is not None else None

        if self.state.surfaceLambdas is not None and lastState.surfaceLambdas is not None:
            self.state.surfaceLambdas.copy_(lastState.surfaceLambdas)
        else:
            self.state.surfaceLambdas = lastState.surfaceLambdas.clone() if lastState.surfaceLambdas is not None else None

        config = kwargs.get('config', None)
        schemeConfig = kwargs.get('schemeConfig', None)
        dx, drhodt_shift, dudt, duCross = shiftTransport(self, dt, config, schemeConfig)

        # Restore boundary density/position, then protect density before shifting.
        midRho = returnValues[-1][1].densities
        self.state.densities = torch.where(self.state.kinds != 0, midRho, self.state.densities)

        self.state.positions = torch.where(
            (self.state.kinds != 0).unsqueeze(-1),
            initialState.state.positions, self.state.positions)
        self.state.densities = torch.nan_to_num(
            self.state.densities, nan=schemeConfig.fluid.restDensity,
            posinf=schemeConfig.fluid.restDensity, neginf=schemeConfig.fluid.restDensity
        ).clamp_min(0.05 * schemeConfig.fluid.restDensity)

        loneMask = None
        if getattr(schemeConfig, 'loneDensityReset', False):
            from ..modules.surfaceDetection import detectNoFluidNeighbours
            loneMask = (lastState.kinds == 0) & detectNoFluidNeighbours(lastState, config, self.adjacency)

        if schemeConfig.shiftProperties.active:
            if schemeConfig.shiftProperties.correctdrhodt:
                self.state.densities += drhodt_shift * dt
            if schemeConfig.shiftProperties.correctdvdt:
                self.state.velocities += (dudt + duCross) * dt
            self.state.positions += dx

        if loneMask is not None:
            self.state.densities = torch.where(
                loneMask, torch.full_like(self.state.densities, schemeConfig.fluid.restDensity),
                self.state.densities)

        nopenshiftDiag = applyNoPenetration(self, initialState, dt, config, schemeConfig,
            restoreGravity=_RESTORE_GRAVITY_ON_CEILING_NOPEN)
        advanceBodies(self, dt, config, schemeConfig)

        fluidMask = self.state.kinds == 0
        accelAll = (self.state.velocities - initialState.state.velocities) / dt
        self.stepDiagnostics = _LazyStepDiagnostics(accelAll, fluidMask, nopenshiftDiag)

        applyPinnedVelocity(self.state, kwargs.get('config', None), kwargs.get('schemeConfig', None))
        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
