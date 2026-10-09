"""A prescribed-velocity band of *fluid* particles: the wrapping frame of a periodic flow past a body (the boundaries repo's `Pinned`, `sim/pinned.py`).

The band is a union of slabs `|min_image(x_axis - centre)| < halfWidth` in the periodic box (a slab at the seam wraps around it).  Inside it the velocity is the free stream: the particles stay fluid
(density, pressure and mass enter every sum, they are advected with the stream, nothing is created or removed), but the momentum equation is replaced by the prescription -- the acceleration is zeroed (`enforceUpdates`),
the position shift is switched off (`pinnedKeepWeight`) and the velocity is set to the stream at the end of every step (`applyPinnedVelocity`, all three applied by the systems' `finalize`; the Dirichlet
function only changes a stage's copy of the state, which the integrator does not carry over).

Only the hard band (weight 1 inside, 0 outside) is ported; the oracle's `ramp` option is not used by any of its cases.  Use it as `schemeConfig.boundaryConditions += [pinnedBandBC(band)]`.
"""
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

from ...configurations.moduleConfigurations.boundaryConditions import BoundaryCondition, BoundaryConditionType

__all__ = ['PinnedBand', 'pinnedBandBC', 'pinnedKeepWeight', 'applyPinnedVelocity']


@dataclass(frozen=True)
class PinnedBand:
    slabs: Tuple[Tuple[int, float, float], ...]          # (axis, centre, halfWidth) per slab; the union is the band
    velocity: Tuple[float, ...] = (0.0, 0.0)
    lo: Tuple[float, ...] = (0.0, 0.0)                   # the periodic box the slabs wrap in
    hi: Tuple[float, ...] = (1.0, 1.0)
    periodic: Tuple[bool, ...] = (True, True)

    def inside(self, x: torch.Tensor) -> torch.Tensor:
        """[N] bool: the particles in the band (minimum image of the raw position along each slab's axis)."""
        m = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        for axis, centre, half in self.slabs:
            d = x[:, axis] - centre
            if self.periodic[axis]:
                L = self.hi[axis] - self.lo[axis]
                d = d - L * torch.round(d / L)
            m |= d.abs() < half
        return m


class PinnedBandSdf:
    """The band as the SDF of a `BoundaryCondition`: `d = -1` inside, `+1` outside (no distance, only membership), zero normals."""

    def __init__(self, band: PinnedBand):
        self.band = band

    def __call__(self, x: torch.Tensor):
        d = torch.where(self.band.inside(x), -torch.ones_like(x[:, 0]), torch.ones_like(x[:, 0]))
        return d, torch.zeros_like(x)


def pinnedBandBC(band: PinnedBand) -> BoundaryCondition:
    """The band as a boundary condition: velocity := stream and acceleration := 0 on the fluid particles inside."""
    stream = band.velocity

    def velocity(state, config, schemeConfig, positions, d, n, t, dt):
        u = torch.tensor(stream[:state.velocities.shape[1]], dtype=state.velocities.dtype, device=state.velocities.device)
        return torch.where((state.kinds == 0).unsqueeze(-1), u.expand_as(state.velocities), state.velocities)

    def acceleration(state, config, schemeConfig, positions, d, n, t, dt):
        return torch.zeros_like(state.velocities)

    return BoundaryCondition(type=BoundaryConditionType.dynamic, sdf=PinnedBandSdf(band),
                             dirichletFunctions={'velocities': velocity}, updateFunctions={'dvdt': acceleration})


def pinnedKeepWeight(state: Any, config: Any, schemeConfig: Any) -> Optional[torch.Tensor]:
    """[N, 1] weight of the position shift: 0 inside a pinned band, 1 elsewhere; None when the scheme has no band (the shift is then untouched)."""
    bcs = getattr(schemeConfig, 'boundaryConditions', None) or ()
    bands = [bc.sdf.band for bc in bcs if isinstance(bc.sdf, PinnedBandSdf)]
    if not bands:
        return None
    from ...math import getPeriodicPositions
    x = getPeriodicPositions(state.positions, config.domain)
    inside = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    for band in bands:
        inside |= band.inside(x)
    return (~inside).to(state.positions.dtype).unsqueeze(-1)


def applyPinnedVelocity(state: Any, config: Any, schemeConfig: Any) -> None:
    """End of a step: the fluid particles inside a pinned band take the stream velocity (in place); nothing happens without a band."""
    bcs = getattr(schemeConfig, 'boundaryConditions', None) or ()
    bands = [bc.sdf.band for bc in bcs if isinstance(bc.sdf, PinnedBandSdf)]
    if not bands:
        return
    from ...math import getPeriodicPositions
    x = getPeriodicPositions(state.positions, config.domain)
    fluid = state.kinds == 0
    v = state.velocities
    for band in bands:
        m = (band.inside(x) & fluid).unsqueeze(-1)
        u = torch.tensor(band.velocity[:v.shape[1]], dtype=v.dtype, device=v.device)
        v.copy_(torch.where(m, u.expand_as(v), v))
