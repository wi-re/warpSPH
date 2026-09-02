"""State/update/system triad for the divergence-free incompressible scheme
(DFSPH, `schemes/dfsph.py`): unlike the weakly-compressible schemes, density
is itself an integrated quantity here (`drhodt`), corrected in `finalize` by a
pressure-Poisson solve (`modules.incompressible.solveIncompressible`) plus the
same delta-SPH-style particle shifting and rigid-body pose update
(`rigidBody.integrate`/`rigidBody.update`) that `weaklyCompressible.py` uses.
Field layout otherwise mirrors `WeaklyCompressibleState` closely enough that
`rigidBody.update.updateBodyParticlesWCSPH` rebuilds either interchangeably
via `type(particleState)`. Not re-exported from `systems/__init__.py` --
`schemes/dfsph.py` and `schemes/builder.py` import it directly.
"""

from warpSPHIntegrators import *
from dataclasses import dataclass
import torch
from typing import Optional
from warpSPHCore import *
from warpSPH.modules.surfaceDetection.wrapper import detectFreeSurface

from ..rigidBody.integrate import  integrateRigidBody
from ..rigidBody.update import updateBodyParticlesWCSPH

from ..modules.shifting.delta import computeDeltaShift
from ..modules.shifting.wrapper import solveShifting
from torch.profiler import profile, record_function, ProfilerActivity

__all__ = ['IncompressibleState', 'IncompressibleSystemUpdate', 'IncompressibleSystem',
           'DFSPHReferenceSystem']


@dataclass
class IncompressibleState(BaseState):
    positions : torch.Tensor = integrated('dxdt', tags=('position',))
    velocities: torch.Tensor = integrated('dvdt', tags=('velocity',))
    supports : torch.Tensor = constant(tags=('particle_support',))
    masses : torch.Tensor = constant(tags=('particle_mass',))
    densities : torch.Tensor = integrated('drhodt', tags=('density',))

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

@dataclass
class IncompressibleSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
    drhodt: torch.Tensor = tagged(tags=('density_derivative',))
    passive: Optional[torch.Tensor] = tagged(tags=('passive_derivative',), default=None)


from ..modules.incompressible import solveIncompressible
from ..configurations import ShiftApplication, DensityEvolution, resolveDensityEvolution
from ..modules.density import computeDensities
import copy

@dataclass
class IncompressibleSystem(BaseIntegrationSystem):
    state: IncompressibleState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        verbosePrint(verbose, f'Initializing new state [t={self.t}]')
        return IncompressibleSystem(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)

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
    
    def finalize(self, initialState, dt, returnValues, updateValues, weights = ..., *args, **kwargs):
        self.adjacency = returnValues[-1][0]  # Assuming the adjacency list is the last return value from the derivative function
        # Copy the last substeps values into the current state to ensure the final state is correct
        lastState = returnValues[-1][1]  # Assuming the state is the second return value from the derivative function
        # General attributes
        self.state.supports.copy_(lastState.supports)
        # self.state.densities.copy_(lastState.densities)

        # Incompressible system specific attributes (internal eneryg is integrated)
        # self.state.totalEnergies.copy_(lastState.totalEnergies)
        # self.state.entropies.copy_(lastState.entropies)
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

        # Under `DensityEvolution.summation` (the default) the carried density
        # is the one `dfsph_step` computed at the start of the step, and the
        # re-sum further down replaces it anyway. Under `continuity`/`hybrid`
        # this copy is exactly what used to make `integrateRho` inert: it
        # discards the `rho + dt*drhodt` the integrator just produced. See
        # `DensityEvolution`.
        # _densityEvolution = resolveDensityEvolution(
        #     kwargs['schemeConfig'].solverConfig) if kwargs.get('schemeConfig') is not None \
        #     else DensityEvolution.summation
        # if _densityEvolution is DensityEvolution.summation:
        #     if self.state.densities is not None and lastState.densities is not None:
        #         self.state.densities.copy_(lastState.densities)
        #     else:
        #         self.state.densities = lastState.densities.clone() if lastState.densities is not None else None
        self.state.densities.copy_(lastState.densities)

        velocity_magnitudes = torch.linalg.vector_norm(self.state.velocities, dim=-1)
        finite_velocity_magnitudes = velocity_magnitudes[torch.isfinite(velocity_magnitudes)]
        max_velocity_magnitude = (
            torch.max(finite_velocity_magnitudes).item()
            if finite_velocity_magnitudes.numel() > 0
            else float('nan')
        )
        config = kwargs.get('config', None)
        schemeConfig = kwargs.get('schemeConfig', None)

        # self.adjacency = buildVerletList(
        #     self.state, 
        #     config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
        #     priorNeighborhood = self.adjacency,
        #     verbose = False)

        # # print(f"Finalizing state at t={self.t + dt}, dt={dt}, with {self.state.positions.shape[0]} particles.")
        # # print(f'Maximum Velocity Magnitude: {max_velocity_magnitude}')
        # dxDeltaShift = None
        # if schemeConfig.shiftProperties.active:
        #     # shiftVector, self.adjacency = computeDeltaShift(
        #     #     currentState = self.state,
        #     #     config = config,
        #     #     schemeConfig = schemeConfig,
        #     #     domain = config.domain,
        #     #     adjacency = self.adjacency,
        #     # )
        #     with record_function("[warpSPH] - [deltaSPH] - solve shifting"):
        #         dxDeltaShift = solveShifting(
        #             systemState = self.state,
        #             config = config,
        #             schemeConfig = schemeConfig,
        #             adjacency = self.adjacency,
        #             dt = dt,
        #         )
        #         # print(f"Applied shifting update with max shift magnitude: {dx.norm(dim=1).max().item()}")

        #         du = dxDeltaShift / dt
        #         rho = self.state.densities
        #         u = self.state.velocities

        #         if schemeConfig.shiftProperties.correctdrhodt:
        #             with record_function("[warpSPH] - [deltaSPH] - compute drhodt_shift"):
        #                 drhodt_shift = warpOperation(
        #                     self.state,
        #                     operationProperties = OperationProperties(
        #                         operation=WarpOperation.Divergence,
        #                         kernel = config.kernel, 
        #                         supportMode = SupportScheme.Gather,
        #                         operationMode = OperationDirection.AllToAll,
        #                         gradientMode = GradientScheme.Summation
        #                     ),
        #                     queryValues = rho.view(-1,1) * du,
        #                     domain = config.domain,
        #                     adjacency = self.adjacency
        #                 ) - rho * warpOperation(
        #                     self.state,
        #                     operationProperties = OperationProperties(
        #                         operation=WarpOperation.Divergence,
        #                         kernel = config.kernel, 
        #                         supportMode = SupportScheme.Gather,
        #                         operationMode = OperationDirection.AllToAll,
        #                         gradientMode = GradientScheme.Difference
        #                     ),
        #                     queryValues = du,
        #                     domain = config.domain,
        #                     adjacency = self.adjacency
        #                 )
        #         if schemeConfig.shiftProperties.correctdvdt:
        #             with record_function("[warpSPH] - [deltaSPH] - compute dudt shift"):
        #                 dudt = -u * warpOperation(
        #                     self.state,
        #                     operationProperties = OperationProperties(
        #                         operation=WarpOperation.Divergence,
        #                         kernel = config.kernel, 
        #                         supportMode = SupportScheme.Gather,
        #                         operationMode = OperationDirection.AllToAll,
        #                         gradientMode = GradientScheme.Difference
        #                     ),
        #                     queryValues =  du,
        #                     domain = config.domain,
        #                     adjacency = self.adjacency
        #                 ).view(-1,1)

        #                 duCross = warpOperation(
        #                     self.state,
        #                     operationProperties = OperationProperties(
        #                         operation=WarpOperation.Divergence,
        #                         kernel = config.kernel, 
        #                         supportMode = SupportScheme.Gather,
        #                         operationMode = OperationDirection.AllToAll,
        #                         gradientMode = GradientScheme.Summation
        #                     ),
        #                     queryValues =  torch.einsum('ij,ik->ijk', u, du),
        #                     domain = config.domain,
        #                     adjacency = self.adjacency
                        # )

        # The step's second full density summation. `continuity` skips it (the
        # constant-density solve then runs on the integrated density too);
        # `hybrid` runs it for this solve only and puts the integrated density
        # back afterwards, because the shift repairs particle-distribution
        # drift that `drho/dt = -rho div v` cannot see. See `DensityEvolution`.
        # densityEvolution = resolveDensityEvolution(schemeConfig.solverConfig)
        # carriedDensities = None
        # if densityEvolution is not DensityEvolution.continuity:
        #     if densityEvolution is DensityEvolution.hybrid:
        #         carriedDensities = self.state.densities
        #     self.state.densities = computeDensities(self.state, config, schemeConfig, self.adjacency)

        # kernel = copy.deepcopy(config.kernel)
        # fs, fsm, n, renormalizationState_, lMin = detectFreeSurface(self.state, config, schemeConfig, schemeConfig.surfaceDetectionConfig, self.adjacency, returnNormals = True)

        # self.state.surfaceIndicator = fsm > 0.5

        # config.kernel = KernelFunctions.Spiky
        # dvdt_incomp, pressure_incomp, errors_incomp, pressures_incomp = solveIncompressible(
            # particles = self.state,
            # config = config,
            # schemeConfig = schemeConfig,
            # adjacency = self.adjacency,
            # dvdt = torch.zeros_like(self.state.velocities),
            # dt = dt,
            # verbose=False
        # )
        # config.kernel = kernel

        # if carriedDensities is not None:
        #     # `hybrid`: the summation density existed for the solve above only.
        #     self.state.densities = carriedDensities

        # Gated on `verbose`, which the integrator forwards from the caller.
        # These fired unconditionally on every step, which made a DFSPH run
        # unusable under --quiet and buried the end-of-run report in a log.
        # if kwargs.get('verbose', False):
        #     print(f"Finalizing state at t={self.t + dt}, dt={dt}, with {self.state.positions.shape[0]} particles.")
        #     print(f'Solver Iterations: {len(errors_incomp)}, Incompressible Error: {errors_incomp[0]:.4g}->{errors_incomp[-1]:.4g}')

        # gradVel = warpOperation(
        #     self.state,
        #     operationProperties = OperationProperties(
        #         operation=WarpOperation.Gradient,
        #         kernel = config.kernel, 
        #         supportMode = SupportScheme.Gather,
        #         operationMode = OperationDirection.AllToAll,
        #         gradientMode = GradientScheme.Difference
        #     ),
        #     queryValues =  self.state.velocities,
        #     domain = config.domain,
        #     adjacency = self.adjacency
        # )

        # dx = dt**2 * dvdt_incomp

        # `shiftProperties.active` used to be a no-op *that still paid for
        # itself*: `solveShifting`'s result was bound to `dx` above and then
        # shadowed by this line, and the only `positions += dx` that would have
        # applied it is commented out at the bottom of this method. So the flag
        # ran a full shifting solve every step and threw it away. It now does
        # what its name says -- the deltaSPH (explicit, concentration-gradient)
        # shift is applied *on top of* the implicit VD+PS shift, which is the
        # "relax the distribution more directly" configuration. Still opt-in:
        # every incompressible case ships `shifting=False`, so the default path
        # is byte-identical (`dxDeltaShift is None` short-circuits below).
        # if dxDeltaShift is not None:
            # dx = dx + dxDeltaShift

        # Resample against the *total* displacement: Cornelis et al. Eq. 17's
        # correction is a first-order Taylor step over however far the particle
        # actually moved, so if a second shift contributes to that move it
        # belongs here too.
        # proj_vel = torch.einsum('nij, nj -> ni', gradVel, dx)

        # The density half of that same resample. Under `summation` it is
        # pointless -- the next step re-sums from the shifted positions, so the
        # shift's effect on density is picked up exactly -- but under
        # `continuity`/`hybrid` nothing else ever tells the carried density that
        # the particles moved: `drho/dt = -rho div v` describes advection, not
        # the shift, so the carried field drifts from the truth by the whole
        # accumulated shift. This is Cornelis et al. Eq. 17 applied to `rho`
        # instead of `v`, i.e. exactly the correction `proj_vel` already is.
        # proj_rho = None
        # if densityEvolution is not DensityEvolution.summation:
        #     gradRho = warpOperation(
        #         self.state,
        #         operationProperties = OperationProperties(
        #             operation=WarpOperation.Gradient,
        #             kernel = config.kernel,
        #             supportMode = SupportScheme.Gather,
        #             operationMode = OperationDirection.AllToAll,
        #             gradientMode = GradientScheme.Difference
        #         ),
        #         queryValues = self.state.densities,
        #         domain = config.domain,
        #         adjacency = self.adjacency
        #     )
        #     proj_rho = torch.einsum('ni, ni -> n', gradRho, dx)

        # shiftApplication = schemeConfig.solverConfig.shiftApplication

        # if shiftApplication is not ShiftApplication.inStepVelocity:
        #     # `inStepVelocity` has already applied this solve's correction to
        #     # the velocity inside the step (`schemes/dfsph.py`), which is the
        #     # whole of DFSPH proper's constant-density treatment -- applying
        #     # the position shift on top of it corrects the same density error
        #     # twice per step, which on `tgv` injects energy rather than
        #     # removing it (kinetic energy *grows* 6.6x over 200 steps).
        #     self.state.positions += dx
        #     self.state.velocities += proj_vel
        #     if proj_rho is not None:
        #         self.state.densities = self.state.densities + proj_rho

        # if shiftApplication is ShiftApplication.positionAndVelocity:
        #     # Also apply the constant-density solution the way DFSPH proper
        #     # does -- to the velocity -- rather than only as the position shift
        #     # above. Without this the scheme has no velocity-level response to
        #     # a density *error* at all (the divergence-free solve enforces
        #     # `div v = 0`, which stops further compression but never undoes
        #     # existing compression), so wall-adjacent compression can only be
        #     # relieved by moving particles, which near a wall pushes them
        #     # through it. Dissipative -- applied here, after the integrator,
        #     # nothing ever removes the non-divergence-free part of it. See
        #     # `ShiftApplication`'s docstring, and `inStepVelocity` for the
        #     # placement that does not have that problem.
        #     self.state.velocities += dt * dvdt_incomp
        # print(f"Applied incompressible update with max position change magnitude: {dvdt_incomp.norm(dim=1).max().item() * dt}")

        # print(returnValues[-1][2])

        returnValues[-1] = (
            returnValues[-1][0], returnValues[-1][1], (returnValues[-1][2][0], returnValues[-1][2][1]), (None, None)
        )

        # initialRho = initialState.state.densities
        # midRho = returnValues[-1][1].densities
        
        # drhodtMid = updateValues[-1].drhodt
        # epsilon = -dt * drhodtMid / midRho
        # self.state.densities = initialRho * (2 - epsilon) / (2+epsilon)

        # if schemeConfig.shiftProperties.active:
        #     # if schemeConfig.shiftProperties.correctdrhodt:
        #     #     self.state.densities += drhodt_shift * dt
        #     if schemeConfig.shiftProperties.correctdvdt:
        #         self.state.velocities += (dudt + duCross) * dt
        #     self.state.positions += dx

        for rigidBody in schemeConfig.rigidBodies:
            rigidBody = integrateRigidBody(rigidBody, 0, 0, dt)
            self.state = updateBodyParticlesWCSPH(self.state, rigidBody)

        # Information for artificial viscosity switches
        # self.state.divergence.copy_(lastState.divergence)
        # self.state.alpha0s.copy_(lastState.alpha0s)
        # self.state.alphas.copy_(lastState.alphas)

        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)


@dataclass
class DFSPHReferenceSystem(IncompressibleSystem):
    """System for `IncompressibleSPHScheme.dfsphReference` (`schemes/
    dfsphReference.py`).

    Reference DFSPH does its whole time integration *inside* the step -- two
    pressure solves with a position advance between them -- so this system
    turns the integrator's own update application into a no-op and copies the
    step-advanced fields across in `finalize`. It deliberately does **not**
    run `IncompressibleSystem.finalize`'s VD+PS block (the constant-density
    shift solve, the Eq.-17 resample, the shifting term): those are exactly
    what the reference scheme replaces.
    """

    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        verbosePrint(verbose, f'Initializing new state [t={self.t}]')
        return DFSPHReferenceSystem(state=state.initializeNewState(),
                                    adjacency=self.adjacency, t=self.t,
                                    domain=self.domain)

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

    def finalize(self, initialState, dt, returnValues, updateValues, weights=...,
                 *args, **kwargs):
        self.adjacency = returnValues[-1][0]
        lastState = returnValues[-1][1]
        for name in ('positions', 'velocities', 'densities', 'supports',
                     'pressures', 'soundspeeds', 'surfaceIndicators',
                     'surfaceNormals', 'surfaceLambdas'):
            src = getattr(lastState, name, None)
            if src is None:
                continue
            dst = getattr(self.state, name, None)
            if dst is not None and torch.is_tensor(dst) and dst.shape == src.shape:
                dst.copy_(src)
            else:
                setattr(self.state, name, src.clone() if torch.is_tensor(src) else src)

        schemeConfig = kwargs.get('schemeConfig', None)
        if schemeConfig is not None:
            for rigidBody in schemeConfig.rigidBodies:
                rigidBody = integrateRigidBody(rigidBody, 0, 0, dt)
                self.state = updateBodyParticlesWCSPH(self.state, rigidBody)

        # Skip IncompressibleSystem.finalize entirely -- go straight to the
        # integrator base's no-op hook.
        return BaseIntegrationSystem.finalize(
            self, initialState, dt, returnValues, updateValues, weights,
            *args, **kwargs)
    