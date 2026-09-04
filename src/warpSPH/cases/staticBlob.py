"""Static square blob in free space (2D, fully periodic), divergence-free
incompressible SPH.

Baseline case 1 of `DFSPH_IMPROVEMENT_PLAN.md`'s "what is left" item 2: the
scheme's basic-correctness check that the dam-break mechanism question (item 1)
waits on. The other incompressible cases all start with a flow already moving
(`tgv`, `shearWave`, `kolmogorovIncompressible`) or a wall-bounded one
(`randomFlowIncompressible`, `dambreak`), so none of them can answer the
question this one exists for: **does the scheme manufacture motion out of a
fluid that has none?**

**The physics.** A square blob of fluid at rest in a periodic box -- no
gravity, no forcing, no initial velocity. The exact solution is that *nothing
happens*: the blob stays where it was, with zero velocity and an unchanged
shape, for all time. It is a free-surface case (the rest of the box is empty),
which is deliberate: the free-surface treatment is exactly the part of the
scheme the dam-break findings put in question (`§1.10`), and a surface in
motionless fluid is the cleanest place to see whether it moves anything.

**The figures of merit**, and what each is for:

- `maxVelocity` / `kineticEnergy` -- must stay ~0. Anything that appears is
  spurious, since no force acts on the blob.
- `dispRms` / `dispMax` -- per-particle displacement from the initial
  (post-relaxation) position: the "shape unchanged" axis, measured per
  particle so a clump of drifting surface particles cannot hide inside a bulk
  average.
- `centroidDrift` -- the centre of mass stays where it started. A whole-blob
  translation is the grosser failure (net spurious momentum); the
  per-particle numbers above catch the local ones.
- `maxDensity` -- no clumping. The free-surface compaction `§1.10` measures on
  `dambreak` (the constant-density solve packing the surface layer toward
  `rho0`) would show up here as `maxDensity` rising toward 1 while the surface
  itself creeps inward. The surface *deficit* `minDensity` ~ 0.5 is sampling
  geometry, not a defect -- half a surface particle's support is empty -- so
  only the upper bound is a health check.

The sampled lattice is jittered first (`shuffleParticles`, the same
de-correlation `tgv` and `shearWave` get through
`caseUtils.incompressible.relaxLattice`): a perfectly regular lattice is an
unstable equilibrium whose own setup transient would be measured as drift,
and the displacement figures of merit are taken from the jittered state, so
the test grades the run, not the setup. What this case deliberately does
*not* do is the constant-density pre-relaxation that `relaxLattice` adds:
that solve drives a free surface toward its `0.9` clamp floor, and on this
state it shifts the surface layer by ~10% of the domain per step, collapsing
the blob within a few steps -- the exact free-surface compaction the case
exists to measure (`§1.10`; measured in `DFSPH_IMPROVEMENT_PLAN.md` Part 23).
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..modules import shuffleParticles
from ..runner import Case, RunContext, caseMain, registerCase
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .plotting import Field, particlePlot
from .weaklyCompressible import (WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, buildRegionSystem,
                                 configureWeaklyCompressible, fluidRegion,
                                 shapeSdf,
                                 particleDistributionMetrics)

__all__ = ['staticBlobCase']


def configureScheme(ctx: RunContext) -> None:
    configureWeaklyCompressible(ctx)

    schemeConfig = ctx.schemeConfig
    # `configureWeaklyCompressible` wires the shared `inviscid`/`alpha`
    # (artificial-viscosity) knobs `deltaSPH` uses; DFSPH has no such term
    # (see `kolmogorovIncompressible`), so this scheme runs with a plain
    # physical viscosity instead.
    schemeConfig.diffusionParams.inviscid = False
    schemeConfig.diffusionParams.viscidNu = ctx.param('nu')
    schemeConfig.shiftProperties.active = ctx.param('shifting', False)


def buildSystem(ctx: RunContext):
    half = ctx.param('blobHalfSize')
    # The box primitive is centred on its own origin, which is the domain
    # centre: no offset needed. At the defaults the blob is a 32-by-32
    # lattice square in a 128-by-128 periodic box, its surface 32 spacings
    # from every periodic image, so the box interacts with nothing but itself.
    sdf = shapeSdf('box', args=[[half, half]])
    return buildRegionSystem(ctx, [fluidRegion(ctx, sdf)])


def initialConditions(ctx: RunContext, system) -> None:
    # Jitter the sampled lattice only -- see the module docstring for why the
    # constant-density pre-relaxation is not run on a free-surface state.
    system.state.positions = shuffleParticles(
        system.state, ctx.config, ctx.schemeConfig, 0,
        jitterAmount=ctx.param('jitter'))
    system.state.velocities[:] = 0.0
    # The displacement figures of merit are measured from here, after the
    # jitter: the setup transient is not the thing under test.
    ctx.scratch['initialPositions'] = system.state.positions.clone()


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    particles = state.state
    fluid = particles.kinds == 0
    positions = particles.positions[fluid]
    velocities = particles.velocities[fluid]
    masses = particles.masses[fluid]
    densities = particles.densities[fluid]

    d = {
        'maxVelocity': torch.linalg.norm(velocities, dim=-1).max().detach().cpu().item(),
        'kineticEnergy': (0.5 * masses * (velocities ** 2).sum(dim=-1)).sum().detach().cpu().item(),
        'minDensity': densities.min().detach().cpu().item(),
        'densityP05': torch.quantile(
            densities.detach().float(), 0.05).cpu().item(),
        'maxDensity': densities.max().detach().cpu().item(),
        'densityStd': densities.std().detach().cpu().item(),
    }
    d.update(particleDistributionMetrics(ctx, state))

    initial = ctx.scratch.get('initialPositions')
    if initial is not None:
        disp = positions - initial[fluid]
        d['dispRms'] = torch.sqrt((disp ** 2).sum(dim=-1).mean()).detach().cpu().item()
        d['dispMax'] = torch.linalg.norm(disp, dim=-1).max().detach().cpu().item()
        d['centroidDrift'] = torch.linalg.norm(
            positions.mean(dim=0) - initial[fluid].mean(dim=0)).detach().cpu().item()
    return d


setupPlot, updatePlot = particlePlot([
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    # The surface legitimately reads ~0.5 (half-empty support), so the panel
    # spans the real range rather than the bulk-only 0.99-1.01 of the
    # periodic cases: what the case grades is whether the surface *moves*.
    Field('densities', 'densities', colorMap='RdBu', colorMapKind='diverging',
          flip=True, midPoint=1.0, vMin=0.4, vMax=1.1),
])


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    return {k: ctx.param(k) for k in staticBlobCase.params}


staticBlobCase = registerCase(Case(
    name='staticBlob',
    scheme='divergenceFree',
    description='Static square blob in free space (2D, periodic): nothing should happen.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    timestep=kolmogorovIncompressibleTimestep,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='05-staticBlob',
        dim=2,
        nx=128,
        L=1.0,
        tLimit=1.0,
        # `kolmogorovIncompressibleTimestep` applies `cflFactor` to the
        # particle diameter, so this is Bender & Koschier's published 0.4,
        # as on the other adaptive incompressible cases.
        cflFactor=0.4,
        kernel='Wendland2',
        integrationScheme='semiImplicitEuler',
        supportMode='SuperSymmetric',
        dt=1e-3,
        minDt=1e-8,
        maxDt=1e-2,
        plotInterval=10,
        storeInterval=500,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        # The blob's surface is the point of the case, so surface detection
        # runs: `solveIncompressible`'s `minShift` gauge guard reads
        # `surfaceIndicators` to decide the constant mode is forceless.
        freeSurface=True,
        # Characteristic half-size of the square, in domain units.
        blobHalfSize=0.25,
        shifting=False,
        # Lattice de-correlation (`shuffleParticles`, `shiftIters=0`); see
        # `initialConditions` for why the constant-density pre-relaxation
        # `tgv`/`shearWave` use is not run here.
        jitter=0.01,
    ),
))


if __name__ == '__main__':
    caseMain(staticBlobCase)
