"""Physical wall contribution to a particle summation density."""

import torch
from ...boundary.provider import bindBodies


def computeWallDensity(state, config, schemeConfig, *, representation='auto', massDensity=1.0):
    """Integrate solid mass density against W; sampled solids are in the particle sum.

    This evaluation needs geometry only, without contact-pressure reconstruction,
    gravity, or pressure-solver coefficients. Solid packing is calibrated during
    initialization; this function uses the ordinary kernel integral.
    """
    provider = getattr(schemeConfig, 'boundaryProvider', None)
    if representation == 'particles' or (representation == 'auto' and provider is None):
        return torch.zeros_like(state.masses)
    if provider is None:
        raise ValueError('Analytic wall density requires a boundaryProvider')
    bindBodies(provider, getattr(provider, 'rigidBodies', schemeConfig.rigidBodies))
    aggregate = provider.aggregate(state, float(state.supports[0]), config.kernel, lean=True)
    packing = float(getattr(schemeConfig, 'analyticWallMass', 1.0))
    return massDensity * (packing * aggregate.out['lam'].sum(0).to(state.masses))
