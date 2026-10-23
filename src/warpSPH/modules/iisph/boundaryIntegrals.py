"""Raw static-solid geometry for IISPH analytic pressure operators."""

from ...boundary.provider import bindBodies


def computeIISPHBoundaryIntegrals(state, config, schemeConfig):
    """Return the provider's unscaled per-body integrals, or None for samples.

    No pressure field, gravity extrapolation or contact interpolation is
    evaluated here. The provider owns the definitions and precision of its
    integral outputs; each physical operator applies its own mass factor.
    """
    policy = schemeConfig.iisph
    policy.validate()
    provider = getattr(schemeConfig, 'boundaryProvider', None)
    rigidBodies = getattr(provider, 'rigidBodies', schemeConfig.rigidBodies)
    if any(bool((rb.linearVelocity != 0).any()) or bool(rb.angularVelocity != 0)
           or getattr(rb, 'dynamic', False) for rb in rigidBodies):
        raise NotImplementedError("IISPH supports static bodies only")
    analytic = policy.boundaryRepresentation == 'analytic' or (
        policy.boundaryRepresentation == 'auto' and provider is not None)
    if analytic and provider is None:
        raise ValueError("Analytic IISPH boundaries require a boundaryProvider")
    if analytic and bool((state.kinds == 1).any()):
        raise ValueError("Analytic IISPH boundaries cannot also contain sampled solids; select particles explicitly")
    if provider is not None:
        bindBodies(provider, rigidBodies)
    if not analytic:
        return None
    if state.positions.shape[1] != 2:
        raise NotImplementedError("IISPH analytic solid integrals currently support 2D")
    return provider.aggregate(state, float(state.supports[0]), config.kernel, lean=True)
