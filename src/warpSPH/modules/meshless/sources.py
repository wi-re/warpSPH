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

    def rates(self, ctx, time):
        mass = ctx.Q[:, 0]
        acceleration = torch.zeros_like(ctx.vel) if self.acceleration is None else self.acceleration(ctx.pos, time)
        if acceleration.shape != ctx.vel.shape:
            raise ValueError('source acceleration must have shape (N, dim)')
        if self.removeMeanAcceleration:
            acceleration = acceleration-(mass[:, None]*acceleration).sum(0)/mass.sum()
        momentum = mass[:, None]*acceleration
        work = (momentum*ctx.vel).sum(-1)
        heat = torch.zeros_like(mass)
        if self.coolingTime is not None:
            heat = mass*(self.targetInternalEnergy-ctx.u)/self.coolingTime
        rate = torch.cat((torch.zeros_like(mass[:, None]), momentum, (work+heat)[:, None]), dim=1)
        # Signed energy exchange: positive heat is bath heating, negative is cooling.
        return rate, torch.stack((work.double().sum(), heat.double().sum()))
