"""Prescribed moving reservoir particles for conserved meshless stages.

Reservoirs supply boundary work and momentum; conservation measurements must
account for those external exchanges. Masses stay fixed at their sampled
values, including in MFM. Positions are prescribed before geometry is built,
and thermal energy uses that geometry's volume, just like the fluid RHS.
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

    def position(self, state, time):
        prescribed = self.trajectory(time, self.referencePositions)
        state.positions = torch.where(self.mask[:, None], prescribed, state.positions)

    def conserved(self, state, geometry, time, gamma):
        velocity, pressure = self.primitiveState(time, state.positions)
        mass = self.masses
        energy = pressure * geometry.volume / (gamma - 1) + .5 * mass * velocity.square().sum(-1)
        Q = torch.cat((mass[:, None], mass[:, None] * velocity, energy[:, None]), dim=-1)
        state.Q = torch.where(self.mask[:, None], Q, state.Q)
