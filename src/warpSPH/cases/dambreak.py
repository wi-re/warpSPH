"""Dam break with optional obstacle (2D), weakly compressible.

The script form of this case was `datagen/weaklyCompressible/generator.py`,
whose ~65 argparse flags are now this case's `params`. The `caseUtils` helpers it
calls still take an argparse-style namespace -- they are shared with the
notebooks -- so :func:`caseArgs` rebuilds one from the spec rather than
rewriting them.

Running it under the incompressible scheme
------------------------------------------

`--scheme divergenceFree` works and needs no wiring, and it is the only free
surface that scheme does not break (`squarePatch` under the same scheme is a
known method limitation). Three things to know before reading any result from
it -- all measured, see `DFSPH_IMPROVEMENT_PLAN.md` §1.10 and Part 19, and
`scripts/probe_dambreakIncompressible.py`:

- **Pass `--integrationScheme semiImplicitEuler`.** This case defaults to
  `rungeKutta2`, and the pressure-projection derivation is specific to
  semi-implicit Euler: a multi-stage integrator solves each stage as if it were
  final and then blends, so the blended velocity is not divergence-free.
  Nothing in the code enforces this yet.
- **`dambreakTimestep` gives `--scheme divergenceFree` Bender & Koschier's
  advective CFL** instead of inheriting the weakly-compressible acoustic `dt`
  fixed once at setup. `deltaSPH` runs are untouched -- `Case.timestep` is one
  hook shared by every scheme a case might run under (see
  `randomFlowIncompressible`'s docstring), so this hook only acts under
  `divergenceFree` and returns `config.dt` unchanged otherwise.
- **Pass `--cflFactor 0.2`, not the published 0.4.** Measured
  (`DFSPH_IMPROVEMENT_PLAN.md` Part 20): unlike `randomFlowIncompressible
  --bounded`, where 0.4 is the landed default, this case **diverges** at 0.4
  (NaN by step 30) and at 0.3 (NaN by step 76) -- the falling column's impact
  is a sharper event than that case's gentle bounded flow, and the CFL's
  lagged `vMax` does not see it coming (§1.6). 0.25 survives but with a
  markedly worse density excursion (`rho_max` 1.23) than 0.2 (1.11); **0.2 is
  the recommended value.** Even so, it is not the free win Part 19 guessed at:
  it buys ~1.7x fewer steps over the full run (1769 against the fixed-`dt`
  baseline's 3000), not ~5x, and `rho_max` over the whole run is 1.11 against
  the baseline's 1.004 -- adaptive stepping here trades some density accuracy
  for fewer steps, it does not dominate the fixed `dt` on both axes.
- **It is markedly over-dissipative here.** Against `deltaSPH` on identical
  geometry, resolution and `dt`, the surge front runs out at about half speed
  and 88% of the kinetic energy disappears just as the falling column should be
  turning into horizontal run-out. This is the case that exposed it; the
  periodic and wall-bounded incompressible cases cannot see it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import torch

from ..caseUtils import (SimulationProperties, buildDomain, buildPresetObstacles,
                         buildRegions, sampleNoise, setupFreestream, setupKolmogorov)
from ..configurations.moduleConfigurations.gravity import GravityType
from ..enumTypes import isIncompressibleScheme
from ..initializers import initializeWeaklyCompressibleSimulation
from ..modules import setupWeaklyCompressibleTimestep
from ..runner import Case, RunContext, caseMain, registerCase
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .weaklyCompressible import particleDistributionMetrics
from .plotting import (Field, buildFieldPlotter, openWindow, pumpEvents,
                       refreshFieldPlotter)

__all__ = ['dambreakCase', 'caseArgs', 'simulationProperties',
           'DAMBREAK_FIELDS', 'DAMBREAK_FIELDS_DENSITY', 'dambreakFields']


def freeSurface(ctx: RunContext) -> bool:
    if ctx.param('fillRatio') < 1.0:
        return True
    return not (ctx.param('semiPeriodic') or ctx.param('fullyPeriodic'))


def caseArgs(ctx: RunContext) -> SimpleNamespace:
    """The argparse-shaped namespace `caseUtils` expects, built from the spec."""
    values = dict(ctx.spec.params)
    values.update(
        nx=ctx.spec.nx,
        L=ctx.spec.L,
        band=ctx.param('band'),
        caseName=ctx.spec.caseName,
        timeLimit=ctx.spec.tLimit,
        plot=ctx.spec.plot,
        plotInterval=ctx.spec.plotInterval,
        exportInterval=ctx.spec.exportInterval,
    )
    return SimpleNamespace(**values)


def simulationProperties(ctx: RunContext) -> SimulationProperties:
    return SimulationProperties(
        device=ctx.device,
        dtype=ctx.dtype,
        nx=ctx.spec.nx,
        dim=ctx.spec.dim,
        L=ctx.spec.L,
        W=ctx.param('W'),
        dx=ctx.spec.L / ctx.spec.nx,
        band=ctx.param('band'),
        n_h=ctx.spec.n_h,
        targetDt=ctx.param('targetDt'),
        freeSurface=freeSurface(ctx),
        semiPeriodic=ctx.param('semiPeriodic'),
        fullyPeriodic=ctx.param('fullyPeriodic'),
    )


def configureScheme(ctx: RunContext) -> None:
    args = caseArgs(ctx)
    simSetup = simulationProperties(ctx)
    ctx.scratch['args'] = args
    ctx.scratch['simSetup'] = simSetup

    # buildDomain widens the box by `band` particle layers for the boundary.
    domain, interiorDomain = buildDomain(simSetup)
    ctx.config.domain = domain
    ctx.config.nx = simSetup.nx + 2 * simSetup.band
    ctx.config.dx = simSetup.dx
    ctx.scratch['interiorDomain'] = interiorDomain

    schemeConfig = ctx.schemeConfig
    schemeConfig.surfaceDetectionConfig.active = simSetup.freeSurface
    schemeConfig.gravityConfig.active = not ctx.param('disableGravity')
    schemeConfig.gravityConfig.type = GravityType.Directional
    schemeConfig.gravityConfig.magnitude = ctx.param('gravityMagnitude')
    schemeConfig.gravityConfig.origin = ctx.param('gravityDirection')
    schemeConfig.bandwith = simSetup.L / ctx.param('bandWidth') / ctx.config.dx


def buildSystem(ctx: RunContext):
    args = ctx.scratch['args']
    simSetup = ctx.scratch['simSetup']
    dx = ctx.config.dx

    # Snapping the obstacle to the particle lattice keeps its sampled surface
    # free of the half-cell sliver a non-aligned SDF would produce.
    maxExtent = round(ctx.param('maxExtent') / dx) * dx
    offsetX = round(ctx.param('offsetX') / dx) * dx
    presets = buildPresetObstacles(maxExtent, offsetX, ctx.spec.L,
                                   ctx.param('fillRatio'), ctx.param('aoa'))
    obstacle = presets.get(ctx.param('obstacleType'))
    if obstacle is None:
        raise ValueError(f"Unknown obstacleType {ctx.param('obstacleType')!r}. "
                         f'Known: {sorted(presets)}')
    obstacle['offsetY'] = round(obstacle['offsetY'] / dx) * dx

    ctx.schemeConfig.regions = buildRegions(ctx.config, ctx.schemeConfig, simSetup, args,
                                            ctx.config.domain, ctx.scratch['interiorDomain'],
                                            obstacle)
    ctx.schemeConfig.boundaryConditions = []
    ctx.scratch['obstacle'] = obstacle

    return initializeWeaklyCompressibleSimulation(
        ctx.schemeConfig.regions, ctx.config, ctx.schemeConfig,
        ctx.SimulationSystem, ctx.SimulationState, verbose=ctx.spec.verbose)


def initialConditions(ctx: RunContext, system) -> None:
    args = ctx.scratch['args']
    simSetup = ctx.scratch['simSetup']

    sampleNoise(system, ctx.config, ctx.schemeConfig, simSetup, args)
    setupFreestream(system, ctx.config, ctx.schemeConfig, simSetup, args)
    setupKolmogorov(system, ctx.config, ctx.schemeConfig, simSetup, args)

    # The sound speed and dt are chosen together: dt follows from the acoustic
    # CFL, so this is what finally fixes config.dt for the run.
    ctx.schemeConfig.fluid.fixedSoundSpeed, ctx.config.dt = setupWeaklyCompressibleTimestep(
        ctx.config, ctx.schemeConfig, system, ctx.param('targetDt'), verbose=ctx.spec.verbose)


def dambreakTimestep(ctx: RunContext, state) -> float:
    """Bender & Koschier's advective CFL, but only under `--scheme
    divergenceFree`.

    `deltaSPH`'s own dt is fixed once, at setup, by `setupWeaklyCompressibleTimestep`
    in `initialConditions` above -- `adaptiveDt` is not otherwise exercised
    per step (nothing in the run loop revisits `config.dt` for a case without a
    `timestep` hook), so returning it unchanged here reproduces that path
    exactly. `divergenceFree` has no acoustic term and needs a real advective
    dt instead, which is `kolmogorovIncompressibleTimestep`'s formula --
    reused as-is, the same way `randomFlowIncompressible` reuses it for its own
    bounded case. dambreak has no `nu` param and runs inviscid under this
    scheme (`configureScheme` never touches `diffusionParams`), so that
    function's viscous term is always inert here.
    """
    if not isIncompressibleScheme(ctx.scheme):
        return ctx.config.dt
    return kolmogorovIncompressibleTimestep(ctx, state)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    particles = state.state
    fluid = particles.kinds == 0
    velocities = particles.velocities[fluid]
    d = {
        'maxVelocity': torch.linalg.norm(velocities, dim=-1).max().detach().cpu().item(),
        'kineticEnergy': (0.5 * particles.masses[fluid]
                          * (velocities ** 2).sum(dim=-1)).sum().detach().cpu().item(),
        'maxDensity': particles.densities[fluid].max().detach().cpu().item(),
        'minDensity': particles.densities[fluid].min().detach().cpu().item(),
        # Spray-robust companion to `minDensity` -- see
        # `weaklyCompressible.weaklyCompressibleDiagnostics`.
        'densityP05': torch.quantile(
            particles.densities[fluid].detach().float(), 0.05).cpu().item(),
    }
    d.update(particleDistributionMetrics(ctx, state))
    # Wall-penetration watch (DFSPH_FINDINGS.md 1.6): fluid particles pushed
    # more than half a spacing past the interior tank AABB. The `c637785`
    # rewrite dropped the mDBC no-penetration shift from `dfsph_step`; this is
    # how a re-grade of the wall-crossing metrics is read off this case.
    interior = ctx.scratch.get('interiorDomain')
    if interior is not None:
        dx = ctx.config.dx
        pos = particles.positions[fluid]
        lo = interior.min.to(pos)
        hi = interior.max.to(pos)
        past = torch.maximum(lo - pos, pos - hi)          # >0 == outside, per axis
        pen = (past > 0.5 * dx).any(dim=-1)
        d['nPenetrating'] = int(pen.sum().detach().cpu().item())
        d['maxPenetrationDx'] = float(
            torch.clamp(past.max(), min=0.0).detach().cpu().item() / dx)
    return d


#: The two panels a dam break actually ships with (`plotDensity=False`).
#: Velocity is the flow; the cyclic-coloured particle IDs are how you see the
#: fluid fold over itself at the free surface, which no scalar field shows.
DAMBREAK_FIELDS = [
    Field('velocities', 'Particle Velocity Magnitude', colorMap='viridis',
          mapping='L2Norm', plotTitleGap=0.08),
    Field('UIDs', 'Particle IDs', colorMap='twilight', colorMapKind='cyclic',
          midPoint=None, plotTitleGap=0.08),
]

#: `--plotDensity`: the same, with the density panel between them.
DAMBREAK_FIELDS_DENSITY = [
    DAMBREAK_FIELDS[0],
    Field('densities', 'Particle Density', colorMap='RdBu', colorMapKind='diverging',
          flip=True, midPoint=1.0, plotTitleGap=0.08),
    DAMBREAK_FIELDS[1],
]


def dambreakFields(ctx: RunContext):
    """The panel list this run plots -- two, or three with `--plotDensity`."""
    return DAMBREAK_FIELDS_DENSITY if ctx.param('plotDensity') else DAMBREAK_FIELDS


def _figsize(ctx: RunContext):
    # The dam break box is much wider than it is tall, so this case carries its
    # own figure size rather than the 11x5 default.
    return (ctx.param('plotWidth'), ctx.param('plotHeight'))


def setupPlot(ctx: RunContext, state):
    plotter = buildFieldPlotter(ctx, state, dambreakFields(ctx), figsize=_figsize(ctx))
    openWindow(ctx, plotter)
    return plotter


def updatePlot(ctx: RunContext, state, plotter, step: int) -> None:
    refreshFieldPlotter(ctx, state, plotter, dambreakFields(ctx), step=step)
    pumpEvents(plotter)


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    simSetup = ctx.scratch['simSetup']
    data = {k: v for k, v in ctx.spec.params.items() if not isinstance(v, (list, dict))}
    data.update(
        nx=ctx.spec.nx, L=ctx.spec.L, n_h=ctx.spec.n_h, timeLimit=ctx.spec.tLimit,
        freeSurface=simSetup.freeSurface, dx=simSetup.dx,
        obstacleText=(f"obstacle_{ctx.param('maxExtent'):.4g}_{ctx.param('aoa'):.4g}"
                      f"_{ctx.param('offsetX'):.4g}" if ctx.param('obstacleActive')
                      else 'no_obstacle'),
    )
    return data


dambreakCase = registerCase(Case(
    name='dambreak',
    scheme='deltaSPH',
    description='Dam break with optional obstacle (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    timestep=dambreakTimestep,
    defaults=dict(
        caseName='3-dambreak',
        dim=2,
        nx=128,
        L=2.0,
        n_h=4.0,
        kernel='Wendland4',
        integrationScheme='rungeKutta2',
        supportMode='KernelMeanSymmetric',
        tLimit=4.0,
        dt=None,
        adaptiveDt=True,
        cflFactor=0.3,
        minDt=1e-8,
        storeMode='trajectory',
        exportInterval=0.002,
        plotInterval=10,
    ),
    params=dict(
        W=4.0,
        band=5,
        targetDt=0.0005,
        # The column is `fluidWidth * W` wide by `fillRatio * L` tall, in the
        # bottom-left corner of a `W x L` tank. These two give 0.667 x 1.333 in
        # the 4 x 2 tank: the canonical Koshizuka & Oka proportions, a column
        # twice as tall as it is wide with six of its widths of run-out.
        #
        # parser.py's default was `fluidWidth = 5/2 * 1/3`, i.e. a 3.33-wide
        # slab covering 83% of the tank -- not a dam break at all, and not a
        # shape any shipped configuration used: every line of
        # `datagen/weaklyCompressible/cases/dambreak.sh` passes an explicit
        # `--fluidWidth` (5/12, 1/4 or 1/12 against fillRatio 1/3, 1/2, 2/3),
        # so the default was never exercised. Same class of stale default as
        # `obstacleType` below. (The first pass at this only moved
        # `fluidWidth`, which left a 1.333 x 0.667 column -- wider than it is
        # tall, i.e. the comment above and the values disagreed. Both are set
        # here now.)
        fillRatio=2.0 / 3.0,
        fluidWidth=1.0 / 6.0,
        semiPeriodic=False,
        fullyPeriodic=False,

        disableGravity=False,
        gravityMagnitude=9.81,
        gravityDirection=[0.0, -1.0],

        obstacleActive=False,
        # `circle` was parser.py's default but is not a preset key any more --
        # generator.py crashes on its own defaults because of it.
        obstacleType='circleMiddle',
        offsetX=3.0 / 4.0,
        aoa=0.0,
        maxExtent=1.0 / 16.0,

        enableFreestream=False,
        forcingWidth=2.0 / 16.0,
        freeStreamVelocity=1.0,

        enableNoise=False,
        octaves=3,
        lacunarity=2,
        persistence=0.5,
        baseFrequency=2,
        kind='perlin',
        seed=45906734,
        noiseAmplitude=1.0,
        bandWidth=16.0,

        enableKolmogorovForcing=False,
        kolmogorovForcingAmplitude=1 / 3,
        kolmogorovForcingWavenumber=2,

        markerSize=4,
        plotWidth=28,
        plotHeight=8,
        plotDensity=False,
    ),
))


if __name__ == '__main__':
    caseMain(dambreakCase)
