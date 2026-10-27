"""Compatibility adapters for direct exact-delta callers.

The recurrences live in warpSPHIntegrators; mode/default selection is config policy.
The runner uses the package-owned Stepper and the normal system lifecycle.
"""
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from .context import StepContext, wrap

__all__ = ['TEMPORAL_SCHEMES', 'AUTO_TEMPORAL_SCHEME', 'resolve_temporal_scheme', 'Advance', 'HancockUpdate', 'advance', 'conservedHancockUpdate',
           'hancockStep', 'ssprkStep', 'kdkStep']

from ...configurations.meshless import TEMPORAL_SCHEMES, AUTO_TEMPORAL_SCHEME, resolve_temporal_scheme


@dataclass
class Advance:
    """What a temporal scheme returns: the exact deltas and the closure potential to warm-start the next step from."""
    dQdt: torch.Tensor
    dxdt: torch.Tensor
    lam: torch.Tensor


#: ``hancockUpdate(ctx) -> (dQdt, Qn)``: a mode's Euler update of the conserved vector from the start-of-step rate at ``dt``. ``dQdt`` is what
#: the integrator applies (``Q += dt dQdt``), ``Qn`` the updated vector (the drift needs its velocity).
HancockUpdate = Callable[[StepContext], Tuple[torch.Tensor, torch.Tensor]]


def conservedHancockUpdate(ctx: StepContext):
    from warpSPHIntegrators.drivers.mechanics.conserved_lagrangian import conservedEuler
    return conservedEuler(ctx.Q, ctx.dt, ctx.rates(ctx.schemeConfig.massFluxLimit))


def advance(ctx: StepContext, scheme: str, hancockUpdate: HancockUpdate = conservedHancockUpdate) -> Advance:
    """Compatibility adapter for callers requesting an exact-delta map directly."""
    from warpSPHIntegrators.drivers.mechanics.conserved_lagrangian import advance as integrate
    result = integrate(scheme, ctx.Q, ctx.pos, ctx.vel, ctx.dt, ctx.geom,
        rates=lambda geom, Q, dt: ctx.ratesAtRest(geom, Q),
        buildStage=lambda positions, geom: ctx.build(positions, geom.lam),
        velocity=lambda Q: Q[:, 1:-1] / Q[:, 0:1], wrap=lambda pos: wrap(pos, ctx.domain),
        hancockUpdate=lambda: hancockUpdate(ctx))
    return Advance(result.dQdt, result.dxdt, result.stage.lam)


def hancockStep(ctx, hancockUpdate=conservedHancockUpdate):
    return advance(ctx, 'hancock', hancockUpdate)


def ssprkStep(ctx):
    return advance(ctx, 'ssprk')


def kdkStep(ctx):
    return advance(ctx, 'kdk')
