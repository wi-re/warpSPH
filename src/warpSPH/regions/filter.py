"""Clip a non-boundary region's particles against every `Boundary`-type region
in a scene, dropping particles inside the boundary's SDF (`sdfValues < 0`).
Boundary regions themselves pass through unchanged. Mutates and returns
`region` in place.

A site exactly on the interface (`sdfValues == 0`) is kept as **fluid**, which
makes the solid/fluid partition total: `regions/sample.py` claims `< 0` for the
boundary band, so keeping only `> 0` here left `== 0` claimed by neither and
punched a one-site void through both bands. That is not hypothetical -- the
Marrone 3.4 obstacle is built from exact multiples (`H = W/10`, `toe_x` an
integer number of `dx` from the centre), so its top and back faces land exactly
on lattice rows and lost a full row from the wall band, on the faces the jet
hits hardest (`DELTASPH_VALIDATION_PLAN.md` 5.10; same failure the `toe_x`
seam fix in `_marroneSharpEdgeSDF` addressed for the vertical seam). The zero
goes to fluid rather than solid so the solid stays exactly the region the SDF
calls negative -- claiming it for the boundary instead would grow every
lattice-coincident wall by a row and bias the effective wall position half a
cell into the fluid. Where nothing is lattice-coincident no site reads exactly
0 and this is a no-op.
"""

from ..configurations.region import RegionType
from ..geometry import ParticleSet

__all__ = ['filterRegion']


def filterRegion(region, regions):
    particles = region.particles
    if region.type == RegionType.Boundary:
        return region
    for region_ in regions:
        if region_.type == RegionType.Boundary:
            sdfValues, sdfNormals = region_.sdf(particles.positions)
            mask = sdfValues >= 0          # `== 0` is fluid -- see the module docstring
            particles = ParticleSet(
                positions = particles.positions[mask],
                supports = particles.supports[mask],
                masses = particles.masses[mask],
                densities = particles.densities[mask]            
            )
            region.particles = particles
    return region

