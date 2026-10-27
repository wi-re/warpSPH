"""Raw static-solid geometry for IISPH analytic pressure operators."""


def computeIISPHBoundaryIntegrals(state, provider, support, kernel):
    """Return the selected provider's unscaled per-body integrals.

    No pressure field, gravity extrapolation or contact interpolation is
    evaluated here. The provider owns the definitions and precision of its
    integral outputs; each physical operator applies its own mass factor.
    """
    return provider.aggregate(state, support, kernel, lean=True)
