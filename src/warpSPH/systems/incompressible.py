"""State/update/system triad for the divergence-free incompressible scheme
(DFSPH, `schemes/incompressible/divergenceFree.py`): unlike the weakly-compressible schemes, density
is itself an integrated quantity here (`drhodt`), corrected in `finalize` by a
pressure-Poisson solve (`modules.incompressible.solveIncompressible`) plus the
same delta-SPH-style particle shifting and rigid-body pose update
(`rigidBody.integrate`/`rigidBody.update`) that `weaklyCompressible.py` uses.
Field layout otherwise mirrors `WeaklyCompressibleState` closely enough that
`rigidBody.update.updateBodyParticlesWCSPH` rebuilds either interchangeably
via `type(particleState)`. Not re-exported from `systems/__init__.py` --
`schemes/incompressible/divergenceFree.py` and `schemes/builder.py` import it directly.
"""

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
from typing import Optional
from warpSPHCore import AdjacencyList, DomainDescription


from ..modules.boundaryConditions.pinned import applyPinnedVelocity

__all__ = [
    "IncompressibleState",
    "IncompressibleSystemUpdate",
    "IncompressibleSystem",
    "ProjectedSystem",
    "IISPHSystem",
    "DFSPHReferenceSystem",
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


#: TEMP (option 1 experiment): restore the VD+PS particle shift in
#: `IncompressibleSystem.finalize` that the `_solve` rewrite of `divergenceFree_step`
#: dropped. `divergenceFree_step` now folds the constant-density solve into the velocity
#: only (`inStepVelocity` semantics), so nothing regularises the particle
#: *distribution* -- on `tgv` the jittered lattice re-snaps in the first steps
#: (DFSPH_FINDINGS.md 1.16). This re-adds a post-integrator position shift plus
#: the Cornelis et al. Eq. 17 velocity resample.
#:
#: RESULT (measured): as `'cd'` with `divergenceFree.INSTEP_CD = False` (the true
#: pre-tmp VD+PS -- divergence projection in-step, constant density only as
#: this shift) it **fully resolves the tgv energy injection**: KE x1.000,
#: peak x1.000, monotone, and clean at nx=64 / 400 steps (no jitter=0 late
#: detonation). BUT `hydrostaticColumn` then NaNs -- the finalize position
#: shift alone cannot sustain a body force (Part 23); the column's safeguard
#: is the in-step CD velocity. Both on (`INSTEP_CD = True`) double-counts the
#: density error -> tgv KE x2.6, column NaN. `'delta'` + `INSTEP_CD = True`
#: is compatible (column holds) but too weak to undo the jitter snap
#: (tgv peak x1.05).
#:
#: `'auto'` (default) resolves the tension by *case*: run the shift where
#: there is no body force (`tgv`, the bounded box, `kolmogorov`, ...) -- it
#: regularises the distribution and `divergenceFree.INSTEP_CD = 'auto'` drops the
#: in-step CD there -- and skip it where a body force needs the in-step CD
#: (`hydrostaticColumn`). No case runs both, so no double-count. `True` /
#: `False` force it; `'cd'` is the shift mode (see `_PS_SHIFT_MODE`).
_RESTORE_PS_SHIFT = "auto"
#: 'cd'    -- the VD+PS shift proper: `solveIncompressible` (dvdt=0). Double-
#:            counts the density error when `divergenceFree.INSTEP_CD` is also on.
#: 'delta' -- a Fickian concentration-gradient shift (`solveShifting`), a
#:            different mechanism, so it can run *alongside* the in-step CD.
_PS_SHIFT_MODE = "cd"
#: The VD+PS shift has two parts -- toggle each to see which does the work:
#:   _PS_POSITION_SHIFT   -- `positions += dx` (dx = dt^2 * a_p).
#:   _PS_VELOCITY_RESAMPLE -- `velocities += gradV . dx` (Cornelis Eq. 17).
_PS_POSITION_SHIFT = True
_PS_VELOCITY_RESAMPLE = True
#: Apply the shift as a *velocity impulse* (`velocities += dx / dt`) instead of
#: a position move -- i.e. the same quantity the position shift uses, routed to
#: velocity. Predict: this stops regularising the distribution, so the tgv
#: injection returns (mirror of `divergenceFree.INSTEP_CD = False`, which routes the CD
#: solve to a position shift and breaks the column).
_PS_SHIFT_AS_VELOCITY = False
#: Free-surface-aware shift (DFSPH_FINDINGS.md 1.17): zero the shift on
#: `surfaceIndicators == 1` rows. `solveIncompressible`'s source `rho0 - rho*`
#: with `rho*` clamped at 0.9 drives the surface skin (legit ~0.5) toward that
#: floor every step -> `hydrostaticColumn` NaN; masking the surface is the
#: cheapest guard. `taper` scales by `surfaceLambdas` instead of a hard cut.
_PS_SURFACE_MASK = False  # False | 'hard' | 'taper'


@dataclass
class IncompressibleSystem(BaseIntegrationSystem):
    state: IncompressibleState = reference_state(tags=("physics_state",))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        if verbose:  # self.t may be a device tensor: format it only when printing
            print(f"Initializing new state [t={self.t}]")
        return type(self)(
            state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain
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

    def finalize(self, initialState, dt, returnValues, updateValues, weights=..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]
        lastState = returnValues[-1][1]
        self.state.supports.copy_(lastState.supports)
        if self.state.pressures is not None and lastState.pressures is not None:
            self.state.pressures.copy_(lastState.pressures)
        else:
            self.state.pressures = (
                lastState.pressures.clone() if lastState.pressures is not None else None
            )
        if self.state.soundspeeds is not None and lastState.soundspeeds is not None:
            self.state.soundspeeds.copy_(lastState.soundspeeds)
        else:
            self.state.soundspeeds = (
                lastState.soundspeeds.clone() if lastState.soundspeeds is not None else None
            )
        if self.state.surfaceIndicators is not None and lastState.surfaceIndicators is not None:
            self.state.surfaceIndicators.copy_(lastState.surfaceIndicators)
        else:
            self.state.surfaceIndicators = (
                lastState.surfaceIndicators.clone()
                if lastState.surfaceIndicators is not None
                else None
            )
        if self.state.surfaceNormals is not None and lastState.surfaceNormals is not None:
            self.state.surfaceNormals.copy_(lastState.surfaceNormals)
        else:
            self.state.surfaceNormals = (
                lastState.surfaceNormals.clone() if lastState.surfaceNormals is not None else None
            )
        if self.state.surfaceLambdas is not None and lastState.surfaceLambdas is not None:
            self.state.surfaceLambdas.copy_(lastState.surfaceLambdas)
        else:
            self.state.surfaceLambdas = (
                lastState.surfaceLambdas.clone() if lastState.surfaceLambdas is not None else None
            )
        self.state.densities.copy_(lastState.densities)
        config = kwargs.get("config", None)
        schemeConfig = kwargs.get("schemeConfig", None)
        from ..modules.shifting.projection import applyProjectionShift
        from ..schemes.incompressible.omniIncompressible import pressurePolicy

        applyProjectionShift(
            self,
            dt,
            config,
            schemeConfig,
            restore=_RESTORE_PS_SHIFT,
            mode=_PS_SHIFT_MODE,
            positionShift=_PS_POSITION_SHIFT,
            velocityResample=_PS_VELOCITY_RESAMPLE,
            shiftAsVelocity=_PS_SHIFT_AS_VELOCITY,
            surfaceMask=_PS_SURFACE_MASK,
            pressurePolicy=pressurePolicy(),
        )
        returnValues[-1] = (
            returnValues[-1][0],
            returnValues[-1][1],
            (returnValues[-1][2][0], returnValues[-1][2][1]),
            (None, None),
        )
        from ..rigidBody.stages import advanceUnforcedBodies

        advanceUnforcedBodies(self, dt, schemeConfig)
        applyPinnedVelocity(
            self.state, kwargs.get("config", None), kwargs.get("schemeConfig", None)
        )
        return super().finalize(
            initialState, dt, returnValues, updateValues, weights, *args, **kwargs
        )


@dataclass
class IISPHSystem(IncompressibleSystem):
    """Ordinary integration of x and v with summed density and pressure carry.

    Inherit the normal derivative updates. Finalization retains only fields
    recomputed by the RHS; positions and velocities belong to the integrator.
    """

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
    """State adapter for the reference DFSPH, omni and Band full maps.

    The package ProjectedStep driver evaluates one full projected map -- two
    pressure solves with a position advance between them -- so this system
    turns the integrator's own update application into a no-op and copies the
    step-advanced fields across in `finalize`. It deliberately does **not**
    run `IncompressibleSystem.finalize`'s VD+PS block (the constant-density
    shift solve, the Eq.-17 resample, the shifting term): those are exactly
    what the reference scheme replaces.
    """

    projected_map = True

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        if verbose:  # self.t may be a device tensor: format it only when printing
            print(f"Initializing new state [t={self.t}]")
        return type(self)(
            state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain
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
        # Skip IncompressibleSystem.finalize entirely -- go straight to the
        # integrator base's no-op hook.
        return BaseIntegrationSystem.finalize(
            self, initialState, dt, returnValues, updateValues, weights, *args, **kwargs
        )


# Compatibility for old imports and serialized Python class references.
DFSPHReferenceSystem = ProjectedSystem
