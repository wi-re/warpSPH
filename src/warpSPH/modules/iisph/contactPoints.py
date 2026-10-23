"""Closest surface contacts used by the IISPH wall pressure law."""


def computeIISPHContactPoints(state, schemeConfig):
    """Return closest wall points, or None when contact pressure is disabled."""
    policy = schemeConfig.iisph
    provider = getattr(schemeConfig, 'boundaryProvider', None)
    analytic = policy.boundaryRepresentation == 'analytic' or (
        policy.boundaryRepresentation == 'auto' and provider is not None)
    if policy.boundaryPressure != 'contactShepard' or not (
            analytic or bool((state.kinds == 1).any())):
        return None
    if provider is None:
        raise ValueError("Contact Shepard IISPH needs the solid contour in boundaryProvider; "
                         "for sampled walls select boundaryRepresentation='particles'")
    distance, normal, hit = provider.signed_distance(state.positions)
    if not bool(hit[state.kinds == 0].all()):
        raise ValueError("IISPH contact geometry did not cover every fluid query")
    return state.positions - distance[:, None] * normal.to(state.positions)
