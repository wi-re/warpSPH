"""State/update/system triad for the weakly-compressible delta-SPH schemes
(`schemes/deltaSPH.py`, `schemes/divergenceFree.py`'s WCSPH path): density is
integrated via `drhodt`, with free-surface fields (`surfaceIndicators`/
`surfaceNormals`/`surfaceLambdas`) and the boundary-ghost bookkeeping
(`ghostIndices`/`ghostOffsets`) that mDBC and `rigidBody/` read and write.
`finalize` applies delta-SPH particle shifting, then integrates every rigid
body's pose and reprojects its particles (`rigidBody.integrate`/
`rigidBody.update`) each step. Its density update computes a Padé-pole-clamped
`epsilon` (kept well clear of the Padé(1,1) approximant's pole at +-2) but no
longer uses it -- the update itself switched to an unclamped exponential form,
so `epsilon` is now dead.
"""

from warpSPHIntegrators import *
from dataclasses import dataclass
import torch
from typing import Optional
from warpSPHCore import *

from ..rigidBody.integrate import  integrateRigidBody
from ..rigidBody.update import updateBodyParticlesWCSPH

from ..modules.shifting.delta import computeDeltaShift
from ..modules.shifting.wrapper import solveShifting
from torch.profiler import profile, record_function, ProfilerActivity

__all__ = ['WeaklyCompressibleState', 'WeaklyCompressibleSystemUpdate', 'WeaklyCompressibleSystem']


@dataclass
class WeaklyCompressibleState(BaseState):
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
class WeaklyCompressibleSystemUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dvdt: torch.Tensor = tagged(tags=('velocity_derivative',))
    drhodt: torch.Tensor = tagged(tags=('density_derivative',))
    passive: Optional[torch.Tensor] = tagged(tags=('passive_derivative',), default=None)


@dataclass
class WeaklyCompressibleSystem(BaseIntegrationSystem):
    state: WeaklyCompressibleState = reference_state(tags=('physics_state',))
    adjacency: Optional[AdjacencyList] = None
    domain: Optional[DomainDescription] = None
    t: float = 0.0
    def initializeNewState(self, *args, verbose=False, **kwargs):
        state = get_reference_state(self)
        verbosePrint(verbose, f'Initializing new state [t={self.t}]')
        return WeaklyCompressibleSystem(state=state.initializeNewState(), adjacency=self.adjacency, t=self.t, domain=self.domain)

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

        # Compressible system specific attributes (internal eneryg is integrated)
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
            # print(f'copying surface indicators: {self.state.surfaceIndicators is not None}, {lastState.surfaceIndicators is not None}')

        if self.state.surfaceNormals is not None and lastState.surfaceNormals is not None:
            self.state.surfaceNormals.copy_(lastState.surfaceNormals)
        else:
            self.state.surfaceNormals = lastState.surfaceNormals.clone() if lastState.surfaceNormals is not None else None

        if self.state.surfaceLambdas is not None and lastState.surfaceLambdas is not None:
            self.state.surfaceLambdas.copy_(lastState.surfaceLambdas)
        else:
            self.state.surfaceLambdas = lastState.surfaceLambdas.clone() if lastState.surfaceLambdas is not None else None

        # print(self.state)

        # print(f"Finalizing state at t={self.t + dt}, dt={dt}, with {self.state.positions.shape[0]} particles.")
        config = kwargs.get('config', None)
        schemeConfig = kwargs.get('schemeConfig', None)
        if schemeConfig.shiftProperties.active:
            # shiftVector, self.adjacency = computeDeltaShift(
            #     currentState = self.state,
            #     config = config,
            #     schemeConfig = schemeConfig,
            #     domain = config.domain,
            #     adjacency = self.adjacency,
            # )
            with record_function("[warpSPH] - [deltaSPH] - solve shifting"):
                dx = solveShifting(
                    systemState = self.state,
                    config = config,
                    schemeConfig = schemeConfig,
                    adjacency = self.adjacency,
                    dt = dt,
                )
                # print(f"Applied shifting update with max shift magnitude: {dx.norm(dim=1).max().item()}")

                du = dx / dt
                rho = self.state.densities
                u = self.state.velocities

                if schemeConfig.shiftProperties.correctdrhodt:
                    with record_function("[warpSPH] - [deltaSPH] - compute drhodt_shift"):
                        drhodt_shift = warpOperation(
                            self.state,
                            operationProperties = OperationProperties(
                                operation=WarpOperation.Divergence,
                                kernel = config.kernel, 
                                supportMode = SupportScheme.Gather,
                                operationMode = OperationDirection.AllToAll,
                                gradientMode = GradientScheme.Summation
                            ),
                            queryValues = rho.view(-1,1) * du,
                            domain = config.domain,
                            adjacency = self.adjacency
                        ) - rho * warpOperation(
                            self.state,
                            operationProperties = OperationProperties(
                                operation=WarpOperation.Divergence,
                                kernel = config.kernel, 
                                supportMode = SupportScheme.Gather,
                                operationMode = OperationDirection.AllToAll,
                                gradientMode = GradientScheme.Difference
                            ),
                            queryValues = du,
                            domain = config.domain,
                            adjacency = self.adjacency
                        )
                if schemeConfig.shiftProperties.correctdvdt:
                    with record_function("[warpSPH] - [deltaSPH] - compute dudt shift"):
                        dudt = -u * warpOperation(
                            self.state,
                            operationProperties = OperationProperties(
                                operation=WarpOperation.Divergence,
                                kernel = config.kernel, 
                                supportMode = SupportScheme.Gather,
                                operationMode = OperationDirection.AllToAll,
                                gradientMode = GradientScheme.Difference
                            ),
                            queryValues =  du,
                            domain = config.domain,
                            adjacency = self.adjacency
                        ).view(-1,1)

                        duCross = warpOperation(
                            self.state,
                            operationProperties = OperationProperties(
                                operation=WarpOperation.Divergence,
                                kernel = config.kernel, 
                                supportMode = SupportScheme.Gather,
                                operationMode = OperationDirection.AllToAll,
                                gradientMode = GradientScheme.Summation
                            ),
                            queryValues =  torch.einsum('ij,ik->ijk', u, du),
                            domain = config.domain,
                            adjacency = self.adjacency
                        )




        initialRho = initialState.state.densities
        midRho = returnValues[-1][1].densities
        
        drhodtMid = updateValues[-1].drhodt
        epsilon = -dt * drhodtMid / midRho
        epsilon = torch.clamp(epsilon, min=-1.5, max=1.5)  # Pade approximant has a pole at +-2, stay well clear of it
        # self.state.densities = initialRho * (2 - epsilon) / (2+epsilon)
        self.state.densities = initialRho * torch.exp(dt * drhodtMid / midRho)  # Use exponential update to avoid negative densities

        self.state.densities = torch.where(self.state.kinds != 0, midRho, self.state.densities)

        if schemeConfig.shiftProperties.active:
            if schemeConfig.shiftProperties.correctdrhodt:
                self.state.densities += drhodt_shift * dt
            if schemeConfig.shiftProperties.correctdvdt:
                self.state.velocities += (dudt + duCross) * dt
            self.state.positions += dx

        for rigidBody in schemeConfig.rigidBodies:
            rigidBody = integrateRigidBody(rigidBody, 0, 0, dt)
            self.state = updateBodyParticlesWCSPH(self.state, rigidBody)

        # Information for artificial viscosity switches
        # self.state.divergence.copy_(lastState.divergence)
        # self.state.alpha0s.copy_(lastState.alpha0s)
        # self.state.alphas.copy_(lastState.alphas)

        # print(self.state)

        # print(f'Surface particles: {self.state.surfaceIndicators.sum().item()} / {self.state.surfaceIndicators.shape[0]} ({100 * self.state.surfaceIndicators.sum().item() / self.state.surfaceIndicators.shape[0]:.2f}%)')

        return super().finalize(initialState, dt, returnValues, updateValues, weights, *args, **kwargs)
    