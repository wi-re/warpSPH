"""Resolve static IISPH boundary inputs before timestep evaluation."""

import torch
from ...boundary.provider import bindBodies


def initializeIISPHBoundaries(state, schemeConfig):
    """Return host support, analytic provider and contact provider.

    Validation and automatic representation selection belong to setup. The
    returned providers are explicit inputs to geometry evaluation thereafter.
    """
    policy = schemeConfig.iisph
    policy.validate()
    provider = getattr(schemeConfig, 'boundaryProvider', None)
    bodies = getattr(provider, 'rigidBodies', schemeConfig.rigidBodies)
    if any(bool((body.linearVelocity != 0).any()) or bool(body.angularVelocity != 0)
           or getattr(body, 'dynamic', False) for body in bodies):
        raise NotImplementedError('IISPH supports static bodies only')
    support = float(state.supports[0])
    if not bool((torch.isfinite(state.supports) & (state.supports == support) & (state.supports > 0)).all()):
        raise NotImplementedError('IISPH requires fixed common support')
    sampled = bool((state.kinds == 1).any())
    representation = policy.boundaryRepresentation
    if representation == 'auto':
        representation = 'analytic' if provider is not None else 'particles'
    analytic = representation == 'analytic'
    if analytic and provider is None:
        raise ValueError('Analytic IISPH boundaries require a boundaryProvider')
    if analytic and sampled:
        raise ValueError('Analytic IISPH boundaries cannot also contain sampled solids; select particles explicitly')
    if analytic and state.positions.shape[1] != 2:
        raise NotImplementedError('IISPH analytic solid integrals currently support 2D')
    contact = policy.boundaryPressure == 'contactShepard' and (analytic or sampled)
    if contact and provider is None:
        raise ValueError('Contact Shepard IISPH needs the solid contour in boundaryProvider; '
                         "for sampled walls select boundaryRepresentation='particles'")
    if provider is not None:
        bindBodies(provider, bodies)
    policy.boundaryRepresentation = representation
    return support, provider if analytic else None, provider if contact else None
