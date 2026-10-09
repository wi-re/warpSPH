"""Shared source-and-decay relaxation of a switched viscosity parameter.

Morris & Monaghan (1997) Eq. (5) and the Rosswog et al. (2000) variant evolve
`d alpha / dt = -(alpha - alpha_min) / tau + S` with `S >= 0` a shock source. Stepping the
decay explicitly is only stable for `dt < tau`; here the decay is taken implicitly,

    alpha_new = (alpha + dt (S + alpha_min / tau)) / (1 + dt / tau),

which is stable for any `dt / tau` (Sphenix, Borrow et al. 2022 Eq. 24, makes the same
argument), never undershoots `alpha_min`, and reaches the steady state `alpha_min + tau S`
for `dt >> tau`. The result is clamped to `[alpha_min, alpha_max]`.
"""

from __future__ import annotations

import torch

__all__ = ['relaxAlpha']


def relaxAlpha(alpha: torch.Tensor, source: torch.Tensor, tau: torch.Tensor, dt: float,
               alpha_min: float, alpha_max: float) -> torch.Tensor:
    return ((alpha + dt * (source + alpha_min / tau)) / (1.0 + dt / tau)).clamp(min=alpha_min, max=alpha_max)
