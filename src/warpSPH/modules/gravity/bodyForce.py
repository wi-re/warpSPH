"""A uniform body force (an acceleration of the momentum equation, `schemeConfig.bodyForce = (fx, fy)`): the pressure-gradient driver of a periodic channel, the boundaries repo's `bodyForce`.

Unlike gravity it is *only* in the momentum equation: it does not enter the hydrostatic term of the density diffusion or the no-penetration law, and it enters the analytic wall pressure condition
`dp/dn = rho (g + f - a_w) . n` only with `schemeConfig.bodyForceAtWall` (default True: a uniform force acts on the fluid at a wall like gravity; `analyticBoundary._evaluateWall`).
"""
from typing import Any

import torch

__all__ = ['computeBodyForce', 'bodyForceVector']


def bodyForceVector(schemeConfig: Any):
    """`schemeConfig.bodyForce` as a tuple of floats, or None when it is absent or zero."""
    f = getattr(schemeConfig, 'bodyForce', None)
    if f is None:
        return None
    f = tuple(float(c) for c in f)
    return f if any(c != 0.0 for c in f) else None


def computeBodyForce(currentState: Any, schemeConfig: Any) -> torch.Tensor:
    """The body force as an acceleration [N, dim] on the fluid rows (zeros elsewhere, and everywhere without a body force)."""
    out = torch.zeros_like(currentState.velocities)
    f = bodyForceVector(schemeConfig)
    if f is None:
        return out
    vec = torch.tensor(f[:out.shape[1]], dtype=out.dtype, device=out.device)
    return torch.where((currentState.kinds == 0).unsqueeze(-1), vec.expand_as(out), out)
