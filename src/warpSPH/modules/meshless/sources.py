"""Conservative body-force work and named finite-time thermal relaxation.

The integrator owns temporal stages. These are instantaneous ODE sources;
thermal relaxation uses finite gamma and is not an isothermal EOS.
"""
from dataclasses import dataclass
import math
import torch

__all__ = ['ConservedSources']


@dataclass
class ConservedSources:
    acceleration: object = None
    removeMeanAcceleration: bool = True
    targetInternalEnergy: float = 1.
    coolingTime: float = None

    def __post_init__(self):
        if self.coolingTime is not None:
            if not math.isfinite(self.coolingTime) or self.coolingTime <= 0:
                raise ValueError('coolingTime must be finite and positive, or None')
            if not math.isfinite(self.targetInternalEnergy) or self.targetInternalEnergy <= 0:
                raise ValueError('targetInternalEnergy must be finite and positive')

    def fields(self, mass, positions, velocity, energy, time):
        acceleration = torch.zeros_like(velocity) if self.acceleration is None else self.acceleration(positions, time)
        if acceleration.shape != velocity.shape:
            raise ValueError('source acceleration must have shape (N, dim)')
        if self.removeMeanAcceleration:
            acceleration = acceleration-(mass[:, None]*acceleration).sum(0)/mass.sum()
        momentum = mass[:, None]*acceleration
        work = (momentum*velocity).sum(-1)
        heat = torch.zeros_like(mass)
        if self.coolingTime is not None:
            heat = mass*(self.targetInternalEnergy-energy)/self.coolingTime
        return momentum, work, heat

    def rates(self, ctx, time):
        mass = ctx.Q[:, 0]
        momentum, work, heat = self.fields(mass, ctx.pos, ctx.vel, ctx.u, time)
        rate = torch.cat((torch.zeros_like(mass[:, None]), momentum, (work+heat)[:, None]), dim=1)
        # Signed energy exchange: positive heat is bath heating, negative is cooling.
        return rate, torch.stack((work.double().sum(), heat.double().sum()))
