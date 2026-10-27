"""Analytic inflow quadrature outside a finite Lagrangian particle interval.

Reservoir points are not integrated or exported. They complete the moment
volumes and supply exterior Riemann states, with no periodic seam. Physical
particles retain their original positions, masses and IDs at initialization.
"""
from dataclasses import dataclass
import torch
from warpSPHCore import DomainDescription
from .geometry import buildGeometry
from .operators import computeRates
from .primitives import particleGamma, primitives


@dataclass
class ReservoirGeometry:
    full: object
    count: int

    @property
    def volume(self):
        return self.full.volume[:self.count]

    @property
    def lam(self):
        return None


@dataclass
class InflowReservoir:
    referencePositions: torch.Tensor
    velocities: torch.Tensor
    supports: torch.Tensor
    owners: torch.Tensor
    density: float
    energy: float
    searchDomain: object

    def build(self, state, config, scheme, time):
        if scheme.closure != 'none':
            raise ValueError('open inflow reservoirs require solver.closure=none')
        ghost = self.referencePositions + time*self.velocities
        positions = torch.cat((state.positions, ghost))
        supports = torch.cat((state.supports, self.supports))
        full = buildGeometry(positions, supports, None, config, scheme, self.searchDomain)
        return ReservoirGeometry(full, len(state.positions))

    def fields(self, rho, velocity, pressure, gamma):
        ghostRho = torch.full_like(self.supports, self.density)
        ghostGamma = gamma[self.owners] if isinstance(gamma, torch.Tensor) else gamma
        ghostPressure = (ghostGamma-1)*ghostRho*self.energy
        fullGamma = torch.cat((gamma, ghostGamma)) if isinstance(gamma, torch.Tensor) else gamma
        return (torch.cat((rho, ghostRho)), torch.cat((velocity, self.velocities)),
                torch.cat((pressure, ghostPressure)), fullGamma)

    def rates(self, ctx, massFluxLimit, dt):
        rho, velocity, pressure, gamma = self.fields(ctx.rho, ctx.vel, ctx.P, ctx.gamma)
        ids = torch.cat((ctx.materials, ctx.materials[self.owners])) if ctx.materials is not None else None
        scheme = ctx.schemeConfig
        return computeRates(ctx.geom.full, rho, velocity, pressure, gamma, dt=dt,
            mode=scheme.mode, order=scheme.order, beta_min=scheme.betaMin, beta_max=scheme.betaMax,
            cond_crit=scheme.condCrit, psi1=scheme.psi1, psi2=scheme.psi2,
            timeCentredFrame=scheme.timeCentredFrame, retry=scheme.retry,
            massFluxLimit=massFluxLimit, faceGuard=scheme.faceGuard, materials=ids)[:ctx.geom.count]

    def timestep(self, state, geom, scheme):
        gamma = particleGamma(state, scheme)
        rho, velocity, pressure = primitives(state.Q, geom.volume, gamma)
        r, v, p, gamma = self.fields(rho, velocity, pressure, gamma)
        dts = (geom.full.timestep(r, v, p*gamma, 1., scheme.cfl) if isinstance(gamma, torch.Tensor)
               else geom.full.timestep(r, v, p, gamma, scheme.cfl))
        return dts[:geom.count]


def nohReservoir(state, config, scheme, density, energy):
    """Pad both incoming streams by twice the initial support radius."""
    import math
    dx = (config.domain.max[0]-config.domain.min[0])/len(state.positions)
    left, right = state.positions[:, 0].argmin(), state.positions[:, 0].argmax()
    layers = max(2, math.ceil(2*float(state.supports.max()/dx)))
    offset = torch.arange(1, layers+1, device=dx.device, dtype=dx.dtype)*dx
    ghosts = torch.cat((state.positions[left, 0]-offset, state.positions[right, 0]+offset))[:, None]
    velocities = torch.cat((torch.ones_like(offset), -torch.ones_like(offset)))[:, None]
    owners = torch.stack((left, right)).repeat_interleave(layers)
    h = state.supports[owners]
    padding = offset[-1]+2*h.max()
    domain = DomainDescription(config.domain.min-padding, config.domain.max+padding,
        torch.zeros_like(config.domain.periodic), config.dim)
    return InflowReservoir(ghosts, velocities, h, owners, density, energy, domain)
