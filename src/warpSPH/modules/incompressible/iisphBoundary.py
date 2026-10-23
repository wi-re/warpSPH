"""Static IISPH solid integrals and closest-contact pressure extrapolation.

The contact law is per fluid query: all solid samples/integrals for that
query use the same Shepard pressure at its closest surface point. It is
distinct from extrapolating a separate pressure at every boundary particle.
The optional affine extension is q(x_solid)=q(contact)+g.(x_solid-contact).
"""

from dataclasses import dataclass
from types import SimpleNamespace
import torch
from warpSPHCore import buildVerletList, SupportScheme
from ...configurations.compute import neighborSearchOptions
from .iisph import IISPHPairs


class ContactPressure:
    def __init__(self, operator, points):
        state, config = operator.state, operator.config
        self.operator = operator
        self.points = points.contiguous()
        query = SimpleNamespace(positions=self.points, supports=state.supports)
        # Contact-centered support differs from the fluid-centered support;
        # reusing the original pairs would truncate the extrapolation stencil.
        adjacency = buildVerletList(query, config.domain, supportMode=SupportScheme.Gather,
            referenceParticles=state, **neighborSearchOptions(config))
        self.pairs = IISPHPairs(query, config, adjacency, reference=state)
        self.denominator = torch.empty_like(operator.volume)
        self.pairs.launch('contact_denominator', [self.pairs.j, self.pairs.starts,
            self.pairs.counts, self.pairs.weights, operator.volume, operator.kinds,
            operator.G, self.denominator])
        missing = operator.fluid & (operator.G.square().sum(-1) > 1.e-24) & (self.denominator <= 0)
        if bool(missing.any()):
            raise ValueError("IISPH contact pressure has no fluid support at a solid contact")
        operator.pairs.launch('contact_diagonal', operator.pairs.base() +
            [operator.volume, operator.kinds, state.positions, self.points, state.supports,
             operator.G, operator.c, self.denominator, *operator.pairs.geometryParameters, operator.diagonal])

    def value(self, q):
        out = torch.empty_like(q)
        self.pairs.launch('contact_value', [self.pairs.j, self.pairs.starts, self.pairs.counts,
            self.pairs.weights, self.operator.volume, self.operator.kinds, self.denominator, q, out])
        return out


@dataclass
class IISPHBoundary:
    density: torch.Tensor
    gradient: torch.Tensor
    sampled: bool
    contactPoints: torch.Tensor = None
    gravity: torch.Tensor = None
    covariance: torch.Tensor = None

    def attach(self, operator):
        if self.contactPoints is None:
            return
        operator.contact = ContactPressure(operator, self.contactPoints)
        if self.gravity is None:
            return
        if self.sampled:
            operator.pairs.launch('sampled_gravity', operator.pairs.base() +
                [operator.volume, operator.kinds, operator.state.positions,
                 self.contactPoints, self.gravity, *operator.pairs.geometryParameters[:2], operator.constantAcceleration])
        else:
            # Cov[a,b] = integral (x_solid-x_i)[a] grad_i W[b].
            operator.constantAcceleration = (-torch.einsum('na,nab->nb', self.gravity, self.covariance)
                - ((operator.state.positions-self.contactPoints)*self.gravity).sum(-1, keepdim=True)*operator.G)
            operator.constantAcceleration = torch.where(operator.fluid[:, None], operator.constantAcceleration, 0)


def buildIISPHBoundary(state, config, schemeConfig, gravity):
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
    B = torch.zeros_like(state.masses)
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
        B = packing * aggregate.out['lam'].sum(0).to(B)
        G = packing * aggregate.out['G'].sum(0).to(G)
        covariance = packing * aggregate.out['Cov'].sum(0).to(state.positions)
    points = None
    extension = None
    if policy.boundaryPressure == 'contactShepard' and (analytic or bool((state.kinds == 1).any())):
        if provider is None:
            raise ValueError("Contact Shepard IISPH needs the solid contour in boundaryProvider; "
                             "for sampled walls select boundaryRepresentation='particles'")
        distance, normal, hit = provider.signed_distance(state.positions)
        if not bool(hit[state.kinds == 0].all()):
            raise ValueError("IISPH contact geometry did not cover every fluid query")
        points = state.positions - distance[:, None]*normal.to(state.positions)
        extension = gravity if policy.contactGravity else None
    return IISPHBoundary(B, G, not analytic, points, extension, covariance)
