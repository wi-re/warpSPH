"""The linear / quadratic coefficient policy of the pairwise dissipation operator (AV_PLAN `CoefficientPolicy`)."""

from warpSPHCore import *
import warp as wp
from ....configurations.moduleConfigurations.diffusionParameters import BetaMode, DiffusionParameters

__all__ = ['switchedCoefficients']


@wp.func
def switchedCoefficients(alpha: scalar_t, C_l_: scalar_t, C_q_: scalar_t, viscosityParams: DiffusionParameters):
    """The linear and quadratic AV coefficients (alpha, beta) for a switched `alpha`.
    `computePi_pair` passes the pair mean alpha; one-sided formulations (CRKSPH's `Q_i` / `Q_j`) pass the particle's own.
    `BetaMode.Fixed`: beta = C_q whatever alpha is (PHANTOM / Garcia-Senz 2026 / Sphenix); `Coupled`: beta follows alpha
    (Chen & Nixon 2025), with the legacy `scaleBeta` (beta = alpha^2 C_l C_q) on top."""
    C_l = alpha * C_l_
    C_q = C_q_
    if viscosityParams.betaMode != wp.static(BetaMode.Fixed.value):
        C_q = alpha * C_q_
        if viscosityParams.scaleBeta:
            C_q = C_q * C_l
    return C_l, C_q
