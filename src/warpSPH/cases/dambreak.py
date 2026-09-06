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
from ..enumTypes import isArtificialCompressibleScheme, isIncompressibleScheme
from ..initializers import initializeWeaklyCompressibleSimulation
from ..modules import setupWeaklyCompressibleTimestep
from ..modules.liu import interpolateLiuLiu
from warpSPHCore import OperationDirection
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

    # buildDomain widens the box by `band` particle layers for the boundary;
    # those band layers are the tank walls (bed, back wall, downstream impact
    # wall), sampled from the widened `domain` SDF -- the geometry needs no
    # further work, only physical parameter values (`ACSPH_PLAN.md` §4.5).
    #
    # The box is walled on every side, so the domain is non-periodic by
    # default. (It was briefly flipped to `periodic=True` while root-causing
    # the Eq. (61) wall moment's minimum-image handling -- `ACSPH_PLAN.md`
    # decision 3.) `wallPeriodic=True` restores the periodic wrap for
    # diagnosing whether near-wall behaviour is a domain-edge neighbour-search
    # artefact; the pressure probe forces its own non-periodic gather either
    # way so it is unaffected.
    domain, interiorDomain = buildDomain(simSetup)
    if not ctx.param('wallPeriodic'):
        domain.periodic = torch.zeros_like(domain.periodic)
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

    if isArtificialCompressibleScheme(ctx.scheme):
        _configureArtificialCompressibleExtra(ctx)


def _configureArtificialCompressibleExtra(ctx: RunContext) -> None:
    """ACSPH-only additions on top of the scheme-agnostic block above
    (`ACSPH_PLAN.md` §4.5/Part 7: the paper's own 2D dam-break case, four
    wall pressure probes and KE against Lobovsky et al.'s experiment).

    Unlike the other cases wired for ACSPH, `dambreak`'s `configureScheme`
    never went through `configureWeaklyCompressible`/`configureArtificialCompressible`
    in the first place -- it builds `schemeConfig` by hand from `caseArgs` --
    so there is no shared helper to dispatch to here; this only adds the
    handful of things that helper would otherwise have done.
    """
    schemeConfig = ctx.schemeConfig
    schemeConfig.shiftProperties.active = False

    # Eq. (46) as the paper writes it has no body-force / acceleration
    # constraint -- `modules/timestep/artificialCompressible.py` adds one
    # behind `dt_accelerationConstraint` (default on) because a gravity-driven
    # walled case generally needs it. On this violent dam break that
    # constraint is *active and binding* through the wall impact: the ACSPH
    # pressure field's `dvdt` spikes to O(1e6) there, driving `dt` down to
    # ~5e-5 (≈10x below the advective limit) where the [0.8,1.2] BDF2 clamp
    # then only lets it recover slowly -- a full `tLimit=2` run costs ~10x
    # what delta-SPH does. `acAccelConstraint=False` restores the paper's
    # literal constraint set (advective + viscous only); see `ACSPH_PLAN.md`
    # §4.5 / §5.6 and the authors' question there.
    if not ctx.param('acAccelConstraint'):
        schemeConfig.dt_accelerationConstraint = False

    # Eq. (48)'s U_char (ACSPH_PLAN.md §5.5): sqrt(g H), the free-fall speed
    # over the column's own height -- the same choice `hydrostaticColumn`
    # makes for the same reason (the only velocity scale in a column at rest
    # under gravity).
    if schemeConfig.acParams.uChar is None:
        depth = ctx.param('fillRatio') * ctx.spec.L
        schemeConfig.acParams.uChar = float(
            (ctx.param('gravityMagnitude') * depth) ** 0.5)

    # The ACSPH step owns its whole real-time advance and returns an exact
    # per-step delta (`schemes/artificialCompressible.py`); any multi-stage
    # integrator would run the dual-time solve once per stage and blend,
    # which is silently wrong. `configureArtificialCompressible` enforces
    # this for cases that go through it; this case does not, so it is
    # enforced here instead, the same way and for the same reason.
    from warpSPHIntegrators import getIntegrator
    from warpSPHIntegrators.integration import IntegrationSchemeType
    wanted = IntegrationSchemeType.forwardEuler
    current = ctx.config.integrationScheme
    if current is not wanted and current is not IntegrationSchemeType.explicitEuler:
        name = getattr(current, 'name', current)
        print(f"[warpSPH] artificialCompressible: overriding integrationScheme "
              f"{name!r} -> 'forwardEuler'. The step returns an exact per-step "
              f"delta, which only a single-evaluation integrator applies unchanged.")
        ctx.config.integrationScheme = wanted
        ctx.integrator = getIntegrator(wanted)


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

    if isArtificialCompressibleScheme(ctx.scheme):
        # No sound speed to back-solve a `dt` from; seed `targetDt` and let
        # `dambreakTimestep`'s Eq. (46) branch take over from step 2 on.
        ctx.config.dt = ctx.param('targetDt')
        return

    # The sound speed and dt are chosen together: dt follows from the acoustic
    # CFL, so this is what finally fixes config.dt for the run.
    #
    # `machTarget` (default None) selects how c0 is set:
    #   None  -- legacy: back-solve c0 out of `targetDt` (c0 ~ 1/dx, so the Mach
    #            number drifts well past 0.1 at fine resolution -- see
    #            `DELTASPH_VALIDATION_PLAN.md` Part 1).
    #   float -- Sun et al. 2017 Eq. (2): c0 = U_max / machTarget with
    #            U_max = `referenceVelocity` or sqrt(2 g H) (dam-break front
    #            speed), so the run stays genuinely weakly compressible at any
    #            resolution. dt then adapts each step via `dambreakTimestep`.
    machTarget = ctx.param('machTarget')
    if machTarget is not None:
        H = ctx.param('fillRatio') * ctx.spec.L
        uMax = ctx.param('referenceVelocity')
        if uMax is None:
            uMax = float((2.0 * ctx.param('gravityMagnitude') * H) ** 0.5)
        ctx.schemeConfig.fluid.fixedSoundSpeed, ctx.config.dt = setupWeaklyCompressibleTimestep(
            ctx.config, ctx.schemeConfig, system, ctx.param('targetDt'),
            verbose=ctx.spec.verbose, uMaxExpected=uMax, machTarget=machTarget)
    else:
        ctx.schemeConfig.fluid.fixedSoundSpeed, ctx.config.dt = setupWeaklyCompressibleTimestep(
            ctx.config, ctx.schemeConfig, system, ctx.param('targetDt'), verbose=ctx.spec.verbose)


def dambreakTimestep(ctx: RunContext, state) -> float:
    """Per-step adaptive dt, dispatched by scheme.

    * `divergenceFree` -- Bender & Koschier's advective CFL
      (`kolmogorovIncompressibleTimestep`), the same reuse
      `randomFlowIncompressible` makes. dambreak has no `nu` param and runs
      inviscid here, so that function's viscous term is inert.
    * `artificialCompressible` -- De Courcy et al. 2024 Eq. (46)
      (`modules.timestep.computeTimestep` -> ACSPH branch).
    * `deltaSPH` (weakly compressible) -- Sun et al. 2017 Eq. (5): the min of
      the viscous, acoustic and acceleration constraints
      (`modules.timestep.computeTimestep` -> weakly-compressible branch). The
      acceleration term needs the last stage update, stashed by the runner as
      `ctx.scratch['lastStageUpdate']`. Previously this branch returned
      `config.dt` unchanged (fixed dt for the whole run) -- that skipped the
      acceleration constraint a gravity-driven impact needs
      (`DELTASPH_VALIDATION_PLAN.md` Part 1).
    """
    from ..modules.timestep import computeTimestep
    if isArtificialCompressibleScheme(ctx.scheme):
        return computeTimestep(state, ctx.config, ctx.schemeConfig, dt=ctx.config.dt)
    if isIncompressibleScheme(ctx.scheme):
        return kolmogorovIncompressibleTimestep(ctx, state)
    return computeTimestep(state, ctx.config, ctx.schemeConfig, dt=ctx.config.dt,
                           systemUpdate=ctx.scratch.get('lastStageUpdate'))


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
    # rewrite dropped the mDBC no-penetration shift from `divergenceFree_step`; this is
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

    # Downstream-wall pressure probes (`ACSPH_PLAN.md` §4.5, Lobovsky et al.
    # 2014): a first-order MLS (Liu-Liu) interpolation of the fluid pressure at
    # fixed sensor points on the impact wall, emitted every step so the
    # trajectory carries the P(t) signal each experimental sensor records.
    # `pressureProbeHeights` lists heights above the tank bed in the case's own
    # length unit; empty (the default) skips this so no existing run changes.
    # First order rather than a bare Shepard gather so the local fit carries
    # the pressure gradient and needs no separate Adami hydrostatic correction
    # at the one-sided wall support; points with < 5 fluid neighbours fall back
    # to 0 (pre-arrival / thin run-up sheet), which `pProbe*Nnbr` makes visible.
    probeHeights = ctx.param('pressureProbeHeights')
    if probeHeights and interior is not None and getattr(particles, 'pressures', None) is not None:
        allPos = particles.positions
        xWall = float(interior.max[0].item()) - float(ctx.param('pressureProbeInset'))
        yBed = float(interior.min[1].item())
        pts = torch.tensor([[xWall, yBed + float(z)] for z in probeHeights],
                           device=allPos.device, dtype=allPos.dtype)
        # The probe sits ~1 kernel support from the +x domain edge; if the run
        # is periodic (`wallPeriodic`) the MLS gather would wrap across to the
        # back wall. Force a non-periodic domain for the interpolation only.
        import copy as _copy
        probeConfig = _copy.copy(ctx.config)
        dom = ctx.config.domain
        probeConfig.domain = type(dom)(dom.min, dom.max,
                                       torch.zeros_like(dom.periodic), dom.dim)
        # `pressureProbeSupportScale` widens the MLS gather radius. A real
        # transducer integrates over its face (Marrone 2011 uses a phi = 90 mm
        # disc); a scale > 1 averages the point reading over a comparable patch
        # so the impulsive first-contact / jet-tip spikes a point probe catches
        # are smoothed the way an area-integrated signal is. Default 1.0 leaves
        # every existing run unchanged.
        val, _grad, nnbr, _A, _b = interpolateLiuLiu(
            pts, referenceParticles=particles, referenceQuantities=particles.pressures,
            config=probeConfig, neighbor_threshold=4,
            direction=OperationDirection.FluidToFluid,
            supportScale=float(ctx.param('pressureProbeSupportScale') or 1.0))
        val = val.clamp(min=0.0).detach().cpu()
        nnbr = nnbr.detach().cpu()
        H = ctx.param('fillRatio') * ctx.spec.L
        g = ctx.param('gravityMagnitude')
        pRef = ctx.schemeConfig.fluid.restDensity * g * H      # rho0 g H
        t = float(state.t) if getattr(state, 't', None) is not None else 0.0
        d['tStar'] = t * (g / H) ** 0.5
        for k in range(len(probeHeights)):
            d[f'pProbe{k}'] = float(val[k])
            d[f'pProbe{k}Star'] = float(val[k]) / pRef
            d[f'pProbe{k}Nnbr'] = int(nnbr[k])
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
        # Sun et al. 2017 §2 integrate the δ-SPH system with RK4 (frozen
        # diffusion is a performance option on top, not required for
        # correctness). `DELTASPH_VALIDATION_PLAN.md` Part 1.
        integrationScheme='rungeKutta4',
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
        # Downstream-wall pressure sensors (`ACSPH_PLAN.md` §4.5): heights above
        # the tank bed, in the case's length unit. Empty -> no probing. See
        # `diagnostics`; `scripts/probe_acsphDambreakLobovsky.py` sets these to
        # Lobovsky et al. 2014's five-sensor matrix at that paper's tank scale.
        pressureProbeHeights=[],
        pressureProbeInset=0.0,
        # MLS gather radius multiplier for the wall probe (1.0 = one kernel
        # support). > 1 mimics a finite-face transducer's area integration --
        # see `diagnostics`; `scripts/probe_deltaSPHMarrone.py` uses it to
        # approach Marrone's φ = 90 mm probe disc.
        pressureProbeSupportScale=1.0,
        # Expected front speed U_max for the Sun Eq. (2) sound-speed pick
        # (`initialConditions`, `machTarget` path). None -> sqrt(2 g H), the
        # free-fall estimate. `scripts/probe_deltaSPHMarrone.py` sets it to
        # Marrone 2011's measured 1.95 sqrt(g H) so c0 = c0Ratio * sqrt(g H)
        # reproduces that paper's Mach number exactly.
        referenceVelocity=None,
        # Restore the periodic domain wrap (default: walled/non-periodic). A
        # diagnostic for near-wall neighbour-search artefacts; the probe gather
        # stays non-periodic regardless.
        wallPeriodic=False,
        # ACSPH only: keep the non-paper acceleration constraint in Eq. (46)'s
        # timestep (default). Set False for the paper's literal advective +
        # viscous constraint set -- see `_configureArtificialCompressibleExtra`.
        acAccelConstraint=True,
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
