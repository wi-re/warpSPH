"""Static IISPH solid integrals and closest-contact pressure extrapolation.

The contact law is per fluid query: all solid samples/integrals for that
query use the same Shepard pressure at its closest surface point. It is
distinct from extrapolating a separate pressure at every boundary particle.
The optional affine extension is q(x_solid)=q(contact)+g.(x_solid-contact).
"""

from typing import NamedTuple
from types import SimpleNamespace
import torch
from warpSPHCore import buildVerletList, SupportScheme
from ...configurations.compute import neighborSearchOptions
from .operators import IISPHKernelData, computeIISPHKernelGeometry, launchIISPHKernel


class IISPHContactWeights(NamedTuple):
    pairs: IISPHKernelData
    denominator: torch.Tensor


class IISPHBoundaryTerms(NamedTuple):
    gradient: torch.Tensor
    sampled: bool
    contactPoints: torch.Tensor = None
    covariance: torch.Tensor = None


def computeIISPHContactWeights(state, config, points, wallGradient, restVolumes):
    # Contact-centered support differs from fluid-centered support.
    query = SimpleNamespace(positions=points.contiguous(), supports=state.supports)
    adjacency = buildVerletList(query, config.domain, supportMode=SupportScheme.Gather,
        referenceParticles=state, **neighborSearchOptions(config))
    pairs = computeIISPHKernelGeometry(query, config, adjacency, reference=state)
    denominator = torch.empty_like(restVolumes)
    launchIISPHKernel('contact_denominator', state, [pairs.indices, pairs.offsets,
        pairs.counts, pairs.weights, restVolumes, state.kinds.to(torch.int32), wallGradient, denominator])
    missing = (state.kinds == 0) & (wallGradient.square().sum(-1) > 1.e-24) & (denominator <= 0)
    if bool(missing.any()):
        raise ValueError("IISPH contact pressure has no fluid support at a solid contact")
    return IISPHContactWeights(pairs, denominator)


def computeIISPHBoundaryTerms(state, config, schemeConfig):
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
    G = torch.zeros_like(state.positions)
    covariance = None
    if provider is not None:
        from ...boundary.provider import bindBodies
        bindBodies(provider, rigidBodies)
    if analytic:
        if state.positions.shape[1] != 2:
            raise NotImplementedError("IISPH analytic solid integrals currently support 2D")
        aggregate = provider.aggregate(state, float(state.supports[0]), config.kernel, lean=True)
        packing = float(getattr(schemeConfig, 'analyticWallMass', 1.0))
        if config.calibrateNormalization:
            from warpSPHCore.util.latticeDensity import latticeDensityFactor
            packing *= latticeDensityFactor(config.kernel, float(config.n_h), 2)
        G = packing * aggregate.out['G'].sum(0).to(G)
        covariance = packing * aggregate.out['Cov'].sum(0).to(state.positions)
    points = None
    if policy.boundaryPressure == 'contactShepard' and (analytic or bool((state.kinds == 1).any())):
        if provider is None:
            raise ValueError("Contact Shepard IISPH needs the solid contour in boundaryProvider; "
                             "for sampled walls select boundaryRepresentation='particles'")
        distance, normal, hit = provider.signed_distance(state.positions)
        if not bool(hit[state.kinds == 0].all()):
            raise ValueError("IISPH contact geometry did not cover every fluid query")
        points = state.positions - distance[:, None]*normal.to(state.positions)
    return IISPHBoundaryTerms(G, not analytic, points, covariance)
