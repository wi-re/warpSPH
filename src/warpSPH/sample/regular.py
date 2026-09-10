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

**Explicit `dx`, non-periodic axis -- count intervals, not points, and round
`dn` down.** A non-periodic axis puts a particle on *both* faces, so `l` is
spanned by `nCells` gaps between `nCells + 1` points. Picking the point count
as `round(l / dx)` and then spanning `l` with `point_count - 1` gaps inflates
`dn` to `l / (round(l/dx) - 1)`, which is *always* a hair above `dx` (~1.2 % at
`l = 82 dx`) -- and a `dn > dx` first band opens a gap between the boundary
layer and the fluid at every wall. So the explicit-`dx` non-periodic path takes
`nCells = ceil(l / dx - 1e-3)` (the `- 1e-3` absorbs float32's ~1e-4 error on
`l/dx`, so an exact multiple does not gain a spurious layer -- reason (2)) and
`dn = l / nCells <= dx`: the lattice is a hair *tighter* than requested, never
looser, so the wall bands touch rather than gap. The periodic path already did
the right thing (`dn = l / round(l/dx)`, no endpoint), and the `dx = None` path
is left as it was.

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
    #  * an explicit `dx` (e.g. `config.dx`) is authoritative. A periodic axis
    #    snaps to `round(span/dx)` cells (`dn = span/nCells`, no endpoint). A
    #    non-periodic axis snaps to `ceil(span/dx - eps)` *cells* -- points are
    #    that + 1, one on each face -- so `dn = span/nCells <= dx`: a hair
    #    tighter than asked, never looser, so the boundary and fluid bands
    #    touch at a wall instead of gapping (see the module docstring). `eps`
    #    is a 1e-6 residual guard so an exact multiple does not gain a layer.
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
        if explicitDx:
            # `nCells = ceil(l/dx - 1e-3)` -> `dn = l/nCells <= dx`: a hair
            # *tighter* than the requested spacing, never looser, so the
            # boundary and fluid bands touch at a wall instead of gapping. The
            # `- 1e-3` (not `1e-6`) absorbs float32's ~1e-4 error on `l/dx` at a
            # few hundred cells, which would otherwise make `ceil` add a
            # spurious whole layer when `l` is an exact multiple of `dx`. For a
            # canonical periodic box (`l/dx == nx` exactly) this is identical to
            # the old `round(l/dx)`; it only bites a *padded* periodic axis,
            # where it keeps `dn <= dx` too. A non-periodic axis carries a point
            # on each face, so it needs `nCells + 1` points for `nCells` gaps --
            # the off-by-one the old `points = round(l/dx)` / `dn = l/(points-1)`
            # got wrong, always inflating `dn` above `dx`.
            nCells = torch.clamp(torch.ceil(l / dx - 1e-3).to(torch.int32), min=1)
            dn = l / nCells
            nd = nCells if periodicity[d] else nCells + 1
        else:
            nd = torch.clamp(torch.ceil(l / dx).to(torch.int32),
                             min=(1 if periodicity[d] else 2))
            dn = l / (nd if periodicity[d] else nd - 1)
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
