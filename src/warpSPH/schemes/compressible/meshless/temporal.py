"""The temporal schemes of the meshless steps, and the exact-delta contract they work under.

A meshless step owns the *entire* time step and hands the integrator the exact per-step deltas ``dxdt = (x^{n+1} - x^n) / dt`` and
``dQdt = (Q^{n+1} - Q^n) / dt``: the integrator's forward-Euler update ``x += dt dxdt``, ``Q += dt dQdt`` then reproduces the step's result
exactly (`schemes/exactDelta.py`: why only single-evaluation integrators may drive such a step). Which result is the choice of
``schemeConfig.temporalScheme``:

``'hancock'``  the core driver's single stage. The face-flux rate at ``dt`` (the MUSCL-Hancock internal prediction), an Euler update of the
               conserved ``Q``, and a midpoint drift of the positions (``dxdt`` is the time-centred velocity ``(v + v_new) / 2``). The G1
               port-fidelity setting. The Euler update of ``Q`` is mode specific, see `hancockUpdate`.
``'ssprk'``    two-stage SSPRK. The rate at ``dt = 0`` (no internal prediction) at the start, a stage state at the midpoint drift (a fresh geometry
               warm-started from the first), the rate there, and ``Q^{n+1} = (Q* + Q + dt f2) / 2``; the final drift is time-centred.
``'kdk'``      kick-drift-kick. Kick half, drift the full ``dt`` with the half-kicked velocity (``dxdt`` = that velocity), kick half with the rate at
               the drifted state.

``'auto'`` is the D4 measurement per mode: MFM -> ``'ssprk'``, MFV -> ``'hancock'`` (`AUTO_TEMPORAL_SCHEME`). A scheme that needs a second
geometry (ssprk, kdk) builds it at the drifted positions with this step's supports and the first geometry's closure potential as warm start;
the closure warm start carried to the next step is the last geometry's.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from .common import StepContext, wrap

__all__ = ['TEMPORAL_SCHEMES', 'AUTO_TEMPORAL_SCHEME', 'resolve_temporal_scheme', 'Advance', 'HancockUpdate', 'advance', 'conservedHancockUpdate',
           'hancockStep', 'ssprkStep', 'kdkStep']

TEMPORAL_SCHEMES = ('hancock', 'ssprk', 'kdk')

#: the D4-measured per-mode default temporal scheme (what 'auto' resolves to)
AUTO_TEMPORAL_SCHEME = {'MFM': 'ssprk', 'MFV': 'hancock'}


def resolve_temporal_scheme(schemeConfig) -> str:
    """Resolve ``temporalScheme`` to a concrete scheme ('auto' -> the D4 per-mode default)."""
    ts = schemeConfig.temporalScheme
    if ts == 'auto':
        return AUTO_TEMPORAL_SCHEME[schemeConfig.mode]
    if ts in TEMPORAL_SCHEMES:
        return ts
    raise ValueError(f'temporalScheme {ts!r} is not recognised; expected "auto" or one of {list(TEMPORAL_SCHEMES)}.')


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
    """``dQdt`` exactly the rate (with the mass-flux limit of the config), ``Q^{n+1} = Q + dt dQdt``: MFM, and MFV's default."""
    dQdt = ctx.rates(ctx.schemeConfig.massFluxLimit)
    return dQdt, ctx.Q + ctx.dt * dQdt


def hancockStep(ctx: StepContext, hancockUpdate: HancockUpdate = conservedHancockUpdate) -> Advance:
    dQdt, Qn = hancockUpdate(ctx)
    vel_n = Qn[:, 1:-1] / Qn[:, 0:1]
    return Advance(dQdt=dQdt, dxdt=0.5 * (ctx.vel + vel_n), lam=ctx.geom.lam)


def ssprkStep(ctx: StepContext) -> Advance:
    dt, Q, vel, pos = ctx.dt, ctx.Q, ctx.vel, ctx.pos
    f1 = ctx.ratesAtRest(ctx.geom, Q)
    Qs = Q + dt * f1
    vels = Qs[:, 1:-1] / Qs[:, 0:1]
    poss = wrap(pos + 0.5 * dt * (vel + vels), ctx.domain)
    g2 = ctx.build(poss, ctx.geom.lam)
    f2 = ctx.ratesAtRest(g2, Qs)
    Qn = 0.5 * Qs + 0.5 * (Q + dt * f2)
    vel_n = Qn[:, 1:-1] / Qn[:, 0:1]
    return Advance(dQdt=(Qn - Q) / dt, dxdt=0.5 * (vel + vel_n), lam=g2.lam)


def kdkStep(ctx: StepContext) -> Advance:
    dt, Q, pos = ctx.dt, ctx.Q, ctx.pos
    f1 = ctx.ratesAtRest(ctx.geom, Q)
    Qh = Q + 0.5 * dt * f1
    velh = Qh[:, 1:-1] / Qh[:, 0:1]
    poss = wrap(pos + dt * velh, ctx.domain)
    g2 = ctx.build(poss, ctx.geom.lam)
    f2 = ctx.ratesAtRest(g2, Qh)
    Qn = Qh + 0.5 * dt * f2
    return Advance(dQdt=(Qn - Q) / dt, dxdt=velh, lam=g2.lam)


def advance(ctx: StepContext, scheme: str, hancockUpdate: HancockUpdate = conservedHancockUpdate) -> Advance:
    """Run the named temporal scheme (``'hancock'`` with the mode's `hancockUpdate`, ``'ssprk'``, ``'kdk'``)."""
    if scheme == 'hancock':
        return hancockStep(ctx, hancockUpdate)
    if scheme == 'ssprk':
        return ssprkStep(ctx)
    if scheme == 'kdk':
        return kdkStep(ctx)
    raise ValueError(f'unknown temporal scheme {scheme!r}')
