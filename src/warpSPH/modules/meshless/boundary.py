"""Prescribed moving reservoir particles for conserved meshless stages.

Reservoirs supply boundary work and momentum; conservation measurements must
account for those external exchanges. Masses stay fixed at their sampled
values, including in MFM. Positions are prescribed before geometry is built,
and a pressure prescription uses that geometry's volume, just like the fluid
RHS. An explicit specific-internal-energy prescription instead forms m*u and
lets pressure follow from the current thermodynamic volume.
"""
from dataclasses import dataclass
import torch


@dataclass
class DrivenParticles:
    mask: torch.Tensor
    referencePositions: torch.Tensor
    masses: torch.Tensor
    trajectory: object
    primitiveState: object
    # Optional prescribed u(time, positions); when present it replaces the
    # pressure prescription and forms E=m*u+KE independently of geometry.
    specificInternalEnergy: object = None

    def position(self, state, time):
        prescribed = self.trajectory(time, self.referencePositions)
        state.positions = torch.where(self.mask[:, None], prescribed, state.positions)

    def conserved(self, state, geometry, time, gamma):
        velocity, pressure = self.primitiveState(time, state.positions)
        mass = self.masses
        if self.specificInternalEnergy is None:
            thermal = pressure * geometry.volume / (gamma - 1)
        else:
            energy = torch.as_tensor(self.specificInternalEnergy(time, state.positions),
                                     device=state.Q.device, dtype=state.Q.dtype).expand_as(mass)
            torch._assert_async(torch.all(torch.isfinite(energy[self.mask]) & (energy[self.mask] > 0)),
                                'prescribed meshless internal energy must be finite and positive')
            thermal = mass * energy
        energy = thermal + .5 * mass * velocity.square().sum(-1)
        Q = torch.cat((mass[:, None], mass[:, None] * velocity, energy[:, None]), dim=-1)
        state.Q = torch.where(self.mask[:, None], Q, state.Q)


def initializeDrivenParticles(system, config, scheme):
    """Apply a driven initial state before the first diagnostic or timestep."""
    driven = system.drivenParticles
    if driven is None:
        return False
    from .geometry import buildGeometry, stashGeometry
    from .primitives import refreshPrimitives, particleGamma
    state = system.state
    driven.position(state, system.t)
    geometry = buildGeometry(state.positions, state.supports, state.lam, config, scheme, system.domain, owner=state)
    driven.conserved(state, geometry, system.t, float(scheme.gamma))
    state.lam = geometry.lam
    refreshPrimitives(state, geometry, particleGamma(state, scheme), scheme.nngb, config.dim, updateSupports=False)
    stashGeometry(state.positions, state.supports, system.domain, config.kernel, scheme, geometry, owner=state)
    state.boundaryProjectionBudget = state.Q.new_zeros(2, dtype=torch.float64)
    return True
