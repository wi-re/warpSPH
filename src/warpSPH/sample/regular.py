"""Plain regular-lattice particle sampler: `buildPointCloud` lays out an
axis-aligned grid spanning `domain` (a single common spacing `dx` picked from
the domain's shortest, or longest if `shortEdge=False`, side so every
dimension is asked to use that spacing), and `sampleRegularParticles` wraps it
into a `ParticleSet` with uniform unit density and per-particle mass equal to
the cell volume/area each particle actually occupies. `jitter > 0` adds
uniform positional noise scaled by the (per-axis) cell spacing; `band` pads
`ns`/the coordinate range with extra layers on each side (used for ghost
regions). This is the sampler almost every case and other `sample/` module
builds on.

**Nominal `dx` vs. achieved `dn` -- why both exist.** `dx` only ever *proposes*
a spacing; each axis is then independently snapped to a whole number of cells
via `nd = ceil(l / dx)`, `dn = l / nd`, because `l` (that axis's length) is
essentially never an exact multiple of `dx`. `dn` is therefore the spacing a
particle is *actually* placed at, and it can differ from `dx` for two
different reasons: (1) structurally, on every axis but the one `dx` was
derived from, since that axis's own length has no reason to be a multiple of
some OTHER axis's spacing; and (2) on the defining axis itself, `l / dx`
should recompute to exactly `nx` (or `nx - 1`), but a few ULPs of floating-point
error in the round trip land it a hair on the wrong side of that integer, and
`ceil` -- a hard step function sitting right on the boundary -- turns that
into a whole extra row of cells. Reason (2) is not a resolution effect: it was
measured adding an entire extra cell layer from a `1e-6`-relative residual.

Mass, support and the periodic half-cell offset are all computed from the
*achieved* `dn`, not the nominal `dx` -- using the latter was exactly the bug:
a particle's assigned mass silently disagreed with the cell it was actually
placed in, by product-of-per-axis-snap amounts (~1.4% measured on
`sloshingTank` nx=100, 2D). See `LATTICE_DENSITY_PLAN.md` and
`cases/weaklyCompressible.calibrateRestDensityMasses`, whose only remaining job
is the *kernel* lattice-quadrature offset (`calibrateNormalization`) -- this
sampler no longer hands it a mass defect to paper over.
"""

from ..utils.domain import DomainDescription
from ..geometry import PointCloud, ParticleSet
from ..utils.support import volumeToSupport

import torch

__all__ = ['sampleRegularParticles']

def buildPointCloud(nx, domain: DomainDescription = None, targetNeighbors = 16, jitter = 0.0, band = 0, shortEdge = True, dx = None):
    periodicity = domain.periodic

    # `dx` proposes a spacing; each axis is then snapped to a whole number of
    # cells. Two ways to pick that proposal:
    #  * `dx=None` (default, unchanged): the shortest -- or with `shortEdge=False`
    #    the longest -- axis's `span/nx`. Fine for a roughly square domain, but a
    #    wide-shallow one (a sloshing tank) lets its *short* axis dictate a
    #    spacing the case never asked for, so `config.dx` ends up a lie that
    #    still feeds `dt` / `c_s` (DELTASPH_VALIDATION_PLAN 5.2.3).
    #  * an explicit `dx` (e.g. `config.dx`) is authoritative: every axis snaps
    #    to `round(span/dx)` cells, so the realised `dn` stays within one part
    #    in `nd` of `dx` on *every* axis. `round`, not `ceil` -- `ceil` sitting
    #    on an integer boundary turns a 1e-6 round-trip residual into a whole
    #    extra cell layer (see the module docstring, reason 2).
    explicitDx = dx is not None
    if explicitDx:
        dx = torch.as_tensor(float(dx))
    else:
        dxs = []
        for d in range(domain.dim):
            l = domain.max[d] - domain.min[d]
            dxs.append(l/(nx if periodicity[d] else nx-1))
        dx = torch.min(torch.tensor(dxs)) if shortEdge else torch.max(torch.tensor(dxs))

    spaces = []

    # `dns[d]` is the spacing actually realised on axis d -- see the module
    # docstring. `cellVolume` (= prod(dns)) is the true cell every particle
    # occupies; `dx` above only ever chose how many cells each axis gets.
    ns = []
    dns = []
    cellVolume = None
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        nd = (torch.round(l/dx) if explicitDx else torch.ceil(l/dx)).to(torch.int32)
        nd = torch.clamp(nd, min=(1 if periodicity[d] else 2))
        dn = l / (nd if periodicity[d] else nd-1)
        dns.append(dn)
        cellVolume = dn if cellVolume is None else cellVolume * dn

        # Both derived from `dn` (this axis's own achieved spacing), not the
        # shared nominal `dx` -- a periodic half-cell offset or ghost-band
        # width sized from the wrong axis's spacing would misalign the wrap.
        offset = dn/2 if periodicity[d] else 0
        offset -= dn * band

        x = torch.linspace(domain.min[d] + offset, domain.max[d] - offset, nd + band * 2, device = domain.min.device, dtype = domain.min.dtype)
        spaces.append(x)
        ns.append(nd + band * 2)

    dtype = spaces[0].dtype
    grid = torch.meshgrid(*spaces, indexing='xy')
    pos = torch.stack([g.flatten() for g in grid], dim=1)
    if jitter > 0:
        # Per-axis, from the achieved spacing: a domain whose axes were
        # snapped to different `dn` (the usual case -- see the module
        # docstring) would otherwise get a jitter width sized off whichever
        # axis happened to define the shared `dx`, not its own.
        dnsTensor = torch.stack([d.to(device = pos.device, dtype = pos.dtype) for d in dns])
        pos = pos + torch.rand_like(pos) * jitter * dnsTensor
    support = volumeToSupport(cellVolume, targetNeighbors, domain.dim)
    supports = torch.ones_like(pos[:, 0]) * support
    return PointCloud(positions = pos, supports = supports), cellVolume, support


def sampleRegularParticles(nx : int, domain : DomainDescription, targetNeighbors: int, jitter = 0.0, band = 0, shortEdge=True, dx = None):
    pc, cellVolume, support = buildPointCloud(nx, domain, targetNeighbors, jitter = jitter, band = band, shortEdge = shortEdge, dx = dx)
    return ParticleSet(positions = pc.positions, supports = pc.supports, masses = torch.ones_like(pc.positions[:, 0]) * cellVolume,
    densities = torch.ones_like(pc.positions[:, 0]))
