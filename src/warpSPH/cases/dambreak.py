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
from typing import Any, Dict, Optional

import torch

from ..caseUtils import (SimulationProperties, buildDomain, buildPresetObstacles,
                         buildRegions, sampleNoise, setupFreestream, setupKolmogorov)
from ..caseUtils.weaklyCompressible import alignInteriorDomainToLattice
from ..caseUtils.weaklyCompressible import buildObstacleSDF
from ..configurations.moduleConfigurations.gravity import GravityType
from ..configurations.moduleConfigurations.shifting import ShiftingProjectionScheme, ShiftingScheme
from ..enumTypes import isArtificialCompressibleScheme, isIncompressibleScheme
from ..initializers import initializeWeaklyCompressibleSimulation
from ..modules import setupWeaklyCompressibleTimestep
from ..modules.liu import interpolateLiuLiu
from warpSPHCore import AdjacencyList, OperationDirection
from ..runner import Case, RunContext, caseMain, registerCase
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .weaklyCompressible import (distributionMetricsFromHost, particleDistributionMetricsDevice,
                                 stepAccelerationDiagnostics)
from ..utils.syncFree import cachedKindIndex, sortedQuantiles
from ..utils.cudaGraph import _readHost, forkedStream
from .plotting import (Field, buildFieldPlotter, openWindow, pumpEvents,
                       refreshFieldPlotter, _export)

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
    #
    # `buildDomain` always returns `domain.periodic` all-True (matching the
    # `13-open-flow.ipynb` notebook's own direct `buildDomainDescription(...,
    # True, ...)` call) -- minimum-image wrap is handled internally from that
    # flag (`buildCompactHashMap` et al.), no position ever needs to be
    # written back into the box. This case only zeroes it back out for the
    # genuinely walled box; `semiPeriodic`/`fullyPeriodic` (`channelFlow.
    # openFlowCase`'s wraparound channel) must keep what `buildDomain` set.
    domain, interiorDomain = buildDomain(simSetup)
    if not (simSetup.semiPeriodic or simSetup.fullyPeriodic
            or ctx.param('wallPeriodic')):
        domain.periodic = torch.zeros_like(domain.periodic)
    # The tank is an axis-aligned box, so its walls are free to sit at any phase
    # within the fixed sampling lattice; snap them onto the lattice mid-gaps so
    # the first particle band lands ~dx/2 proud of each flat wall instead of a
    # lattice row landing on the wall (which leaves the boundary layer a full dx
    # out and a fluid row on the wall). Inward-only, < 1 dx, and `domain` / the
    # lattice are untouched. `alignBoundaryLattice=False` restores the raw phase.
    if ctx.param('alignBoundaryLattice', True):
        alignInteriorDomainToLattice(domain, interiorDomain, simSetup.dx)
    ctx.config.domain = domain
    ctx.config.nx = simSetup.nx + 2 * simSetup.band
    ctx.config.dx = simSetup.dx
    ctx.scratch['interiorDomain'] = interiorDomain

    schemeConfig = ctx.schemeConfig
    schemeConfig.surfaceDetectionConfig.active = simSetup.freeSurface
    schemeConfig.gravityConfig.active = not ctx.param('disableGravity')
    schemeConfig.gravityConfig.type = GravityType.Directional
    schemeConfig.gravityConfig.magnitude = ctx.param('gravityMagnitude')
    schemeConfig.gravityConfig.direction = ctx.param('gravityDirection')
    schemeConfig.bandwith = simSetup.L / ctx.param('bandWidth') / ctx.config.dx
    # `None` (default): don't touch it -- let whichever scheme was selected
    # keep its own `freezeDiffusionAcrossStages` default (`deltaSPH`: False;
    # `sun2017DeltaSPH`: True, `Sun2017DeltaSPHConfig`). Only an explicit
    # True/False override here should be able to stomp that choice.
    freezeParam = ctx.param('freezeDiffusionAcrossStages')
    if freezeParam is not None and hasattr(schemeConfig, 'freezeDiffusionAcrossStages'):
        schemeConfig.freezeDiffusionAcrossStages = freezeParam

    # `shifting`: the delta-SPH / delta+-SPH switch, and on *this* case it is a
    # conformance knob rather than a tuning one.
    #
    # Marrone et al. 2011 Sec. 3's dam-break cases are plain delta-SPH with no
    # PST -- `DELTASPH_VALIDATION_PLAN.md` Sec. 5.1's spec table says so, and
    # Part 2.1 makes "stable with `shiftProperties.active = False`" the
    # acceptance gate. The case could not honour that, because
    # `ShiftProperties.active` defaults to **True** and the only `= False` here
    # is in `_configureArtificialCompressibleExtra`, i.e. the ACSPH branch that
    # the delta-SPH path never reaches. So every delta-SPH run of this case has
    # been delta+-SPH (`DELTASPH_VALIDATION_PLAN.md` Sec. 5.1.1). This param is
    # what lets the Marrone runs actually be delta-SPH.
    #
    # `None` (the default) leaves the selected scheme's own setting alone, so
    # no existing run changes; `False` is the Marrone Sec. 3 configuration.
    shiftingParam = ctx.param('shifting', None)
    if shiftingParam is not None and hasattr(schemeConfig, 'shiftProperties'):
        schemeConfig.shiftProperties.active = bool(shiftingParam)

    # Physical viscosity. `dambreak` has always run inviscid on the delta-SPH
    # path (the `# inviscid here` note in `dambreakTimestep`) -- Marrone 2011
    # §3.1/§3.4.1 and the Lobovsky dam break are all inviscid. §3.4.2 is the
    # exception: the same sharp-edged-obstacle flow *with* real viscosity at
    # Re = √(gH)·H/ν ∈ {1000, 10000} (`DELTASPH_VALIDATION_PLAN.md` §5.2.2).
    # `inviscid=True` (default) leaves every existing run unchanged; `False`
    # swaps the artificial-viscosity term for the Monaghan & Gingold physical
    # Laplacian with `nu` (`modules/deltaSPH/velocityDissipation.py`). Walls
    # stay free-slip -- the no-slip half of §3.4.2 is a separate piece.
    if not isArtificialCompressibleScheme(ctx.scheme) and hasattr(schemeConfig, 'diffusionParams'):
        schemeConfig.diffusionParams.inviscid = ctx.param('inviscid', True)
        if ctx.param('alpha', None) is not None:
            schemeConfig.diffusionParams.inviscidAlpha = ctx.param('alpha')
        if not ctx.param('inviscid', True):
            schemeConfig.diffusionParams.viscidNu = ctx.param('nu', 0.0)

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
    # Michel et al. 2022 shifting, as De Courcy et al. 2024 Sec. 4.5 run this
    # case. It used to be forced off here, and without it the run pairs from
    # step 1 (`pairedFraction` ~0.2, `nnDistP01` ~0.06 dx by the impact),
    # which seeds the wall leak and the dt collapse in
    # FREESLIP_DAMBREAK_FINDINGS.md. `--shifting False` still turns it off.
    shiftingParam = ctx.param('shifting', None)
    schemeConfig.shiftProperties.active = True if shiftingParam is None else bool(shiftingParam)
    schemeConfig.shiftProperties.scheme = ShiftingScheme.michel2022
    schemeConfig.shiftProperties.projectionScheme = ShiftingProjectionScheme.michel2022

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

    if ctx.param('noPenetrationShift'):
        schemeConfig.noPenetrationShift = True

    # `acParams.epsilonV`'s own docstring (`configurations/artificialCompressible.py`):
    # `None` (default) leaves the paper's `-6.0` alone. This repo runs float32
    # by default, where `-6.0` is measured to never converge on this case
    # (plateaus ~-5.8, so every step burns the full 200-iteration pseudo-time
    # cap) -- pass `-5.0` for a >6x step-time cut with no measured behaviour
    # change (`FREESLIP_DAMBREAK_FINDINGS.md`).
    epsilonV = ctx.param('epsilonV')
    if epsilonV is not None:
        schemeConfig.acParams.epsilonV = float(epsilonV)

    # Viscous stabilisation, De Courcy et al. 2024 Sec. 4: `nu = alpha_nu h c0 / K`
    # with `alpha_nu = 0.01` in every case, and `c0 = 50 sqrt(g H)` for the dam
    # break (Sec. 4.5, the delta-SPH sound speed ACSPH borrows only to fix nu).
    # Without it the run is effectively inviscid (`acParams.nu = 1e-6`, ~4000x
    # less at nx=70) and a thinly-supported free-surface particle at a wall
    # contact runs away in negative pressure (FREESLIP_DAMBREAK_FINDINGS.md).
    # `acAlphaNu=None` keeps the directly-set physical `acParams.nu`.
    schemeConfig.acParams.cavitationProjection = ctx.param('acCavitationProjection', 'off')
    schemeConfig.acParams.isolatedZeroPressure = bool(ctx.param('acIsolatedZeroPressure', False))

    alphaNu = ctx.param('acAlphaNu')
    if alphaNu is not None:
        depth = ctx.param('fillRatio') * ctx.spec.L
        schemeConfig.acParams.alphaNu = float(alphaNu)
        schemeConfig.acParams.referenceSoundSpeedForViscosity = float(
            50.0 * (ctx.param('gravityMagnitude') * depth) ** 0.5)

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
    # analytic walls (wallRepresentation='analytic'): which wall pressure condition (modules/analyticBoundary/wallTerms.py)
    ctx.schemeConfig.analyticWallPressure = ctx.param('analyticWallPressure')

    # Stash the obstacle's own SDF (negative inside the solid) so `diagnostics`
    # can measure fluid penetration into the obstacle -- the interior-AABB
    # penetration watch there only sees the tank walls, not a solid island in
    # the flow or a concave fillet inside the AABB (the Marrone 2011 Fig. 19
    # `marroneSharpEdge` geometry has both). Same snapped params `buildRegions`
    # sampled the boundary from.
    if ctx.param('obstacleActive'):
        ctx.scratch['obstacleSDF'] = buildObstacleSDF(
            obstacle['obstacleType'], obstacle['offsetX'], obstacle['offsetY'],
            obstacle['maxExtent'], obstacle['aspectRatio'], obstacle['aoa'],
            ctx.config, ctx.schemeConfig, ctx.spec.L, ctx.param('W'),
            interior=ctx.scratch['interiorDomain'])

    return initializeWeaklyCompressibleSimulation(
        ctx.schemeConfig.regions, ctx.config, ctx.schemeConfig,
        ctx.SimulationSystem, ctx.SimulationState, verbose=ctx.spec.verbose)


def initialConditions(ctx: RunContext, system) -> None:
    args = ctx.scratch['args']
    simSetup = ctx.scratch['simSetup']

    # The runner's velocity alarm (`runner/velocityAlarm.py`) flags |v| far
    # above this: the same `U_max` the Eq. (2) sound speed uses below, for
    # every scheme -- the flow starts from rest, so nothing generic applies.
    uMax = ctx.param('referenceVelocity')
    if uMax is None:
        uMax = float((2.0 * ctx.param('gravityMagnitude') * ctx.param('fillRatio') * ctx.spec.L) ** 0.5)
    ctx.velocityScale = float(uMax)
    ctx.velocityScaleSource = ('referenceVelocity' if ctx.param('referenceVelocity') is not None
                               else 'dam-break front speed sqrt(2 g H)')

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

    # `hydrostaticInit`: stamp the fluid density with the weakly-compressible
    # hydrostatic profile `rho(z) = rho0 (1 + g max(H-z,0) / c0^2)` instead of a
    # uniform `rho0`. Same idea as `tgv-wc`'s `initialPressure`: a weakly
    # compressible scheme carries its pressure in the density (`p = c0^2(rho -
    # rho0)`), so a uniform-`rho0` still pool starts with its entire hydrostatic
    # pressure field missing and has to compress into it -- a release-from-rest
    # transient that the delta-SPH diffusion (~ delta h c0, so weaker at finer
    # dx) damps ever more slowly. Measured on the English 2022 Sec. 4.1 still
    # tank: clean by t ~ 1 s at dp = 0.02 (H/dx = 25) but still ringing at
    # t = 4 s at dp = 0.01 (H/dx = 50), biasing the hydrostatic profile to
    # ~20 % of rho g H (`DELTASPH_VALIDATION_PLAN.md` Sec. 5.2.1). Default off,
    # so no existing dam-break run changes; a still-water case wants it on.
    if ctx.param('hydrostaticInit', False):
        c0 = float(ctx.schemeConfig.fluid.fixedSoundSpeed)
        rho0 = float(ctx.schemeConfig.fluid.restDensity)
        g = float(ctx.param('gravityMagnitude'))
        H = ctx.param('fillRatio') * ctx.spec.L
        bedY = -ctx.spec.L / 2.0
        st = system.state
        fluid = st.kinds == 0
        depth = torch.clamp(H - (st.positions[:, 1] - bedY), min=0.0)
        rhoHydro = rho0 * (1.0 + g * depth / (c0 ** 2))
        st.densities = torch.where(fluid, rhoHydro.to(st.densities.dtype), st.densities)
        if st.pressures is not None:
            st.pressures = torch.where(fluid, (c0 ** 2 * (st.densities - rho0)).to(st.pressures.dtype),
                                       st.pressures)


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


#: 7-point Gauss-Legendre quadrature on [-1, 1], used by `diagnostics`'
#: `pressureProbeDiscRadius` path to area-integrate a wall pressure probe over
#: a flush transducer disc instead of reading a single point. In 2D (no
#: out-of-plane extent) the disc-area integral reduces to a 1D integral over
#: its vertical chord: at height offset s from the disc centre (|s| <= R,
#: x = s/R), the chord width is 2*sqrt(R^2-s^2) = 2R*sqrt(1-x^2), so the
#: area-weighted average is (2/pi) * integral_{-1}^{1} P(R x) sqrt(1-x^2) dx.
#: `_DISC_CHORD_WEIGHTS` folds the sqrt(1-x^2) chord factor into the standard
#: Gauss-Legendre weights and is renormalised to sum to 1 at use (so the
#: (2/pi) prefactor, which is exactly what makes unweighted quadrature sum to
#: 1, need not be carried separately).
_GAUSS7_NODES = (-0.9491079123427585, -0.7415311855993945, -0.4058451513773972,
                 0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585)
_GAUSS7_WEIGHTS = (0.1294849661688697, 0.2797053914892766, 0.3818300505051189,
                   0.4179591836734694, 0.3818300505051189, 0.2797053914892766,
                   0.1294849661688697)
_DISC_CHORD_WEIGHTS = tuple(w * max(0.0, 1.0 - x * x) ** 0.5
                            for x, w in zip(_GAUSS7_NODES, _GAUSS7_WEIGHTS))
_DISC_CHORD_WEIGHT_SUM = sum(_DISC_CHORD_WEIGHTS)


def _readScalars(dev: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Device scalars -> Python floats in a single device->host transfer.
    float64 holds every float32 value and every count here exactly."""
    if not dev:
        return {}
    keys = list(dev)
    vals = torch.stack([dev[k].detach().to(torch.float64).reshape(()) for k in keys]).cpu().tolist()
    return dict(zip(keys, vals))


def _diagnosticsDevice(ctx: RunContext, state) -> Dict[str, torch.Tensor]:
    """Every per-step scalar of `diagnostics` except the pressure probes, as
    0-d device tensors and without a host sync (fluid rows via the per-run
    cached index, the distribution metrics and step statistics through their
    device variants) -- so the runner can replay it from a CUDA graph per
    Verlet generation (`utils/cudaGraph.py:GraphedDiagnostics`). Keys
    prefixed `dist.` / `step.` are those helpers' values."""
    particles = state.state
    fluidIndex = cachedKindIndex(particles.kinds, 0, ctx.config)
    velocities = particles.velocities.index_select(0, fluidIndex)
    rhoF = particles.densities.index_select(0, fluidIndex)
    rhoFq = rhoF.detach().float()
    dev = {
        'maxVelocity': torch.linalg.norm(velocities, dim=-1).max(),
        'kineticEnergy': (0.5 * particles.masses.index_select(0, fluidIndex)
                          * (velocities ** 2).sum(dim=-1)).sum(),
        'maxDensity': rhoF.max(),
        'minDensity': rhoF.min(),
        # Spray-robust companions to `min`/`maxDensity` -- see
        # `weaklyCompressible.weaklyCompressibleDiagnostics`. `maxDensity` at a
        # violent impact tracks a single jet-tip particle whose peak *sharpens*
        # with resolution (a fragmenting-jet case like Marrone 2011 §3.4 is not
        # converged in that quantity even at H/dx = 234); `densityP99` is the
        # bulk-compressibility measure a convergence check should band.
        # (both from one sort, bitwise `torch.quantile`'s)
        **dict(zip(('densityP05', 'densityP99'), sortedQuantiles(rhoFq, (0.05, 0.99)))),
    }
    for k, v in (particleDistributionMetricsDevice(ctx, state) or {}).items():
        dev['dist.' + k] = v
    stepDiag = getattr(state, 'stepDiagnostics', None)
    if hasattr(stepDiag, 'deviceValues'):
        for k, v in stepDiag.deviceValues(fluidIndex).items():
            dev['step.' + k] = v
    # Wall-penetration watch (DFSPH_FINDINGS.md 1.6): fluid particles pushed
    # more than half a spacing past the interior tank AABB. The `c637785`
    # rewrite dropped the mDBC no-penetration shift from `divergenceFree_step`; this is
    # how a re-grade of the wall-crossing metrics is read off this case.
    interior = ctx.scratch.get('interiorDomain')
    if interior is not None:
        dx = ctx.config.dx
        pos = particles.positions.index_select(0, fluidIndex)
        lo = interior.min.to(pos)
        hi = interior.max.to(pos)
        past = torch.maximum(lo - pos, pos - hi)          # >0 == outside, per axis
        pen = (past > 0.5 * dx).any(dim=-1)
        dev['nPenetrating'] = pen.sum()
        dev['maxPenetrationDx'] = torch.clamp(past.max(), min=0.0)

        # Penetration into a solid obstacle / concave wall fillet: the AABB
        # watch above cannot see these (a solid island in the flow, or a fillet
        # inside the tank AABB). `obstacleSDF` is negative inside the solid, so
        # `-sdf` clamped at 0 is the depth a fluid particle has sunk in.
        obSDF = ctx.scratch.get('obstacleSDF')
        if obSDF is not None:
            with torch.no_grad():
                sd = obSDF(pos)
            inside = sd < 0
            dev['nObstaclePen'] = inside.sum()
            dev['maxObstaclePenDx'] = torch.clamp(-sd.min(), min=0.0)
    hashMap = _probeHashMap(ctx, state)
    if hashMap is not None:
        setup = _probeSetup(ctx, state)
        for key, pts in setup.queries.items():
            dev['probe.' + key] = _mlsPressureDevice(setup, particles, pts, hashMap)
    return dev


def _diagnosticsPending(ctx: RunContext, state):
    """Start `_diagnosticsDevice` -- replayed from a CUDA graph when the
    runner set one up (`ctx.scratch['graphedDiagnostics']`), else eagerly --
    and return a thunk that reads its values back (one transfer). The GPU
    work runs while the caller does the pressure probes' host work."""
    graphed = ctx.scratch.get('graphedDiagnostics')
    if graphed is not None:
        return graphed.launch(ctx, state, _diagnosticsDevice)
    dev = _diagnosticsDevice(ctx, state)
    return lambda: _readHost(dev)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    pending = _diagnosticsPending(ctx, state)
    if _probeHashMap(ctx, state) is None:
        # probes fitted here, on their own stream: their host reads need not
        # wait for the GPU work just queued, which overlaps their host work
        with forkedStream(ctx.scratch, 'probeStream', state.state.positions.device):
            probes = _pressureProbes(ctx, state)
        host = pending()
    else:
        # probes fitted in `_diagnosticsDevice`, read back with the rest
        host = pending()
        probes = _pressureProbes(ctx, state, {k[6:]: v for k, v in host.items()
                                              if k.startswith('probe.')})
    d = {k: host[k] for k in ('maxVelocity', 'kineticEnergy', 'maxDensity', 'minDensity',
                              'densityP05', 'densityP99')}
    d.update(distributionMetricsFromHost(
        {k[5:]: v for k, v in host.items() if k.startswith('dist.')}))
    stepDiag = getattr(state, 'stepDiagnostics', None)
    if hasattr(stepDiag, 'fromHost'):
        d.update(stepDiag.fromHost({k[5:]: v for k, v in host.items() if k.startswith('step.')}))
    else:
        d.update(stepAccelerationDiagnostics(state))
    dx = ctx.config.dx
    if 'nPenetrating' in host:
        d['nPenetrating'] = int(host['nPenetrating'])
        d['maxPenetrationDx'] = float(host['maxPenetrationDx'] / dx)
    if 'nObstaclePen' in host:
        d['nObstaclePen'] = int(host['nObstaclePen'])
        d['maxObstaclePenDx'] = float(host['maxObstaclePenDx'] / dx)
    d.update(probes)
    return d


def _probeSetup(ctx: RunContext, state):
    """The run-constant side of the pressure probes, built once per run
    (`ctx.scratch`): query points on the device, the non-periodic probe
    config, the reference pressure. `None` when the case has no probes.

    Pressure probes -- a first-order MLS (Liu-Liu) interpolation of the fluid
    pressure at fixed sensor points, emitted every step so the trajectory
    carries the P(t) signal each experimental sensor records. First order
    rather than a bare Shepard gather so the local fit carries the pressure
    gradient and needs no separate Adami hydrostatic correction at the
    one-sided wall support; points with too few / near-coplanar fluid
    neighbours fall back to a 0th-order Shepard gather, then to 0 (pre-arrival
    / thin run-up sheet), which `*Nnbr` makes visible.

    - `pressureProbeHeights` (`ACSPH_PLAN.md` §4.5, Lobovsky et al. 2014):
    sensors on the +x impact wall, given as heights above the tank bed in
    the case length unit; optional φ-disc area integration.
    - `surfacePressureProbes` (`DELTASPH_VALIDATION_PLAN.md` §5.2.2, Marrone
    2011 §3.4): sensors anywhere on the solid surface -- the 45° edge, the
    obstacle roof, the concave fillet arc -- none axis-aligned, so each is
    `[x, y, nx, ny]` (centred-domain point + outward normal into the fluid)
    and the query point is pushed `surfacePressureProbeInset` spacings along
    that normal to sit in the fluid rather than exactly on the wall.
    Both are empty by default so no existing run changes.
    """
    particles = state.state
    if getattr(particles, 'pressures', None) is None:
        return None                     # e.g. the initial state, before any step
    cached = ctx.scratch.get('_probeSetup', False)
    if cached is not False:
        return cached
    interior = ctx.scratch.get('interiorDomain')
    canProbe = interior is not None
    probeHeights = ctx.param('pressureProbeHeights')
    surfaceProbes = ctx.param('surfacePressureProbes')
    haveSurface = surfaceProbes is not None and len(surfaceProbes) > 0
    setup = None
    if canProbe and (probeHeights or haveSurface):
        allPos = particles.positions
        # The probes sit ~1 kernel support from a domain edge; if the run is
        # periodic (`wallPeriodic`) the MLS gather would wrap to the far wall.
        # Force a non-periodic domain for the interpolation only.
        import copy as _copy
        probeConfig = _copy.copy(ctx.config)
        dom = ctx.config.domain
        probeConfig.domain = type(dom)(dom.min, dom.max,
                                       torch.zeros_like(dom.periodic), dom.dim)
        # `pressureProbeSupportScale` widens the MLS gather radius -- a cheap
        # single-point approximation to a finite-face transducer's area
        # integration, superseded by `pressureProbeDiscRadius` for a true disc
        # integral; kept for cases that still want it. Default 1.0 is unchanged.
        supportScale = float(ctx.param('pressureProbeSupportScale') or 1.0)
        H = ctx.param('fillRatio') * ctx.spec.L
        g = ctx.param('gravityMagnitude')
        setup = SimpleNamespace(
            probeConfig=probeConfig, supportScale=supportScale, H=H, g=g,
            pRef=ctx.schemeConfig.fluid.restDensity * g * H,          # rho0 g H
            dtype=allPos.dtype, queries={}, probeHeights=probeHeights,
            nSurface=0,
            # The Verlet list's hash map finds every fluid neighbour of a
            # fixed query point for as long as the list is valid (a reference
            # within h now was within verletScale*h at the build -- the same
            # argument the mDBC ghost query relies on), so the probes can skip
            # building their own; not for a widened gather or a periodic run.
            verletHash=supportScale == 1.0 and not bool(dom.periodic.any()))
        if probeHeights:
            xWall = float(interior.max[0].item()) - float(ctx.param('pressureProbeInset'))
            yBed = float(interior.min[1].item())
            discRadius = float(ctx.param('pressureProbeDiscRadius') or 0.0)
            # `pressureProbeDiscRadius > 0`: area-integrate over a flush wall
            # transducer disc (Marrone 2011 uses φ = 90 mm) via the 7-point
            # Gauss-Legendre chord quadrature above, instead of one point per
            # probe height. 0.0 (default): unchanged single-point behaviour.
            sOffsets = [discRadius * x for x in _GAUSS7_NODES] if discRadius > 0.0 else [0.0]
            setup.discRadius, setup.nP, setup.nS = discRadius, len(probeHeights), len(sOffsets)
            # Non-extrapolating companions (scripts/probe_marrone31P1Field.py,
            # 2026-09-26): the reading at the wall evaluates a first-order MLS
            # fit AT the wall, i.e. extrapolates the fitted gradient over the
            # last ~dx of one-sided support. A disordered near-wall layer
            # (delta-SPH without PST) gives a spurious gradient there --
            # Marrone 3.1 P1 read 0.83 rho g H while the adjacent fluid averaged
            # 0.48. Recorded alongside, not instead, so existing numbers stay
            # comparable: the same disc 1 dx into the fluid is the second
            # column of query points.
            setup.queries['wall'] = torch.tensor(
                [[xQuery, yBed + float(z) + s_]
                 for xQuery in (xWall, xWall - float(ctx.config.dx))
                 for z in probeHeights for s_ in sOffsets],
                device=allPos.device, dtype=allPos.dtype)
        if haveSurface:
            arr = torch.as_tensor(surfaceProbes, dtype=allPos.dtype,
                                  device=allPos.device).view(-1, 4)
            nrm = arr[:, 2:4] / arr[:, 2:4].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            inset = float(ctx.param('surfacePressureProbeInset') or 0.0) * float(ctx.config.dx)
            setup.queries['surf'] = arr[:, 0:2] + inset * nrm
            setup.nSurface = arr.shape[0]
    ctx.scratch['_probeSetup'] = setup
    return setup


def _probeHashMap(ctx: RunContext, state):
    """The Verlet hash map the probes may query instead of building their own
    (see `_probeSetup`), or `None`."""
    setup = _probeSetup(ctx, state)
    adjacency = getattr(state, 'adjacency', None)
    if setup is None or not setup.verletHash or not isinstance(adjacency, AdjacencyList):
        return None
    return getattr(adjacency, 'hashMap', None)


def _mlsPressureDevice(setup, particles, pts, hashMap=None) -> torch.Tensor:
    """(M,2) query points -> (4, M) device tensor [P, nNeighbours,
    wellConditioned, Shepard]: the first-order MLS fluid-pressure fit with the
    density2025.py / wallPressure.py fallback ladder: the MLS fit where
    well-conditioned, else a 0th-order Shepard gather (`b[:,0] / A_g[:,0,0]`),
    else 0. A query point can have plenty of neighbours (double digits) and
    still fail the determinant check -- a thin sheet running along a wall is
    exactly this, near-coplanar neighbours but many of them -- and
    `interpolateLiuLiu` hard-zeros the first-order fit there by design;
    without the Shepard tier every such point reads a flat 0 regardless of how
    much real neighbour support it had. The Shepard tier is only a 0th-order
    average of the (one-sided) neighbourhood, so it reads biased *low* against
    a real pressure gradient -- `pSurf{k}WC` records which tier a probe used.
    Clamped >= 0. The 4th row is the plain Shepard gather at every point (no
    gradient term), also clamped >= 0.

    With `hashMap` (the Verlet list's, `_probeHashMap`) the gather queries it
    and the fit runs sync-free -- graph-capturable; the same fit up to float
    rounding (candidate order, `inv_ex` for `pinv`). Without, a hash map is
    built for the call, exactly as the probes always did."""
    v, _g, nn, A_g, b, wc = interpolateLiuLiu(
        pts, referenceParticles=particles,
        referenceQuantities=particles.pressures,
        config=setup.probeConfig, neighbor_threshold=4,
        direction=OperationDirection.FluidToFluid, supportScale=setup.supportScale,
        adjacency=hashMap, syncFree=hashMap is not None)
    shepDen = A_g[:, 0, 0]
    shepVal = torch.where(shepDen > 0, b[:, 0] / shepDen.clamp_min(1e-12),
                          torch.zeros_like(shepDen))
    v = torch.where(wc, v, shepVal).clamp(min=0.0)
    # float32 holds the counts exactly
    return torch.stack([v.detach(), nn.detach().to(v.dtype), wc.to(v.dtype),
                        shepVal.clamp(min=0.0).detach()])


def _pressureProbes(ctx: RunContext, state, mls: Optional[Dict[str, torch.Tensor]] = None
                    ) -> Dict[str, float]:
    """`diagnostics`' wall pressure probes (`tStar` and the `pProbe*` /
    `pSurf*` columns); empty when the case has none. `mls`: the fits already
    computed on the device (`_mlsPressureDevice`, host copies keyed like
    `setup.queries`); any missing is computed here (one transfer each)."""
    setup = _probeSetup(ctx, state)
    d: Dict[str, float] = {}
    if setup is None:
        return d
    particles = state.state

    def _mlsPressure(key):
        host = mls.get(key) if mls else None
        if host is None:
            host = _mlsPressureDevice(setup, particles, setup.queries[key]).cpu()
        host = host.to(setup.dtype)
        return host[0], host[1].to(torch.int32), host[2] > 0.5, host[3]

    pRef = setup.pRef
    t = float(state.t) if getattr(state, 't', None) is not None else 0.0
    d['tStar'] = t * (setup.g / setup.H) ** 0.5

    if setup.probeHeights:
        nP, nS, discRadius = setup.nP, setup.nS, setup.discRadius

        def _discColumn(val, shep, nnbr):
            val, shep, nnbr = val.view(nP, nS), shep.view(nP, nS), nnbr.view(nP, nS)
            if discRadius > 0.0:
                # Weight each quadrature sample by its disc chord factor,
                # excluding only genuinely dry samples (`nnbr <= 1`,
                # matching density2025.py's own Shepard-tier floor) and
                # renormalising over the rest -- a sample with real
                # neighbour support always contributes its (MLS-or-Shepard)
                # value rather than a hard 0.
                baseW = torch.tensor(_DISC_CHORD_WEIGHTS, dtype=val.dtype)
                w = baseW.unsqueeze(0) * (nnbr > 1).to(val.dtype)
                wSum = w.sum(dim=1)

                def avg(q):
                    return torch.where(wSum > 0, (q * w).sum(dim=1) / wSum.clamp_min(1e-12),
                                       torch.zeros_like(wSum))
                return avg(val), avg(shep), nnbr.amax(dim=1)
            return val[:, 0], shep[:, 0], nnbr[:, 0]

        # Both columns (at the wall, 1 dx into the fluid) in one fit; every
        # query point's fit is independent of the others.
        valAll, nnbrAll, _wc, shepAll = _mlsPressure('wall')
        m = nP * nS
        valOut, shepOut, nnbrOut = _discColumn(valAll[:m], shepAll[:m], nnbrAll[:m])
        valIn1, _shepIn1, _nIn1 = _discColumn(valAll[m:], shepAll[m:], nnbrAll[m:])
        #   pProbe{k}In1  -- the same MLS disc 1 dx into the fluid;
        #   pProbe{k}Shep -- the plain Shepard disc at the wall (no gradient
        #                    term; biased low under a real gradient).
        for k in range(nP):
            d[f'pProbe{k}'] = float(valOut[k])
            d[f'pProbe{k}Star'] = float(valOut[k]) / pRef
            d[f'pProbe{k}Nnbr'] = int(nnbrOut[k])
            d[f'pProbe{k}In1Star'] = float(valIn1[k]) / pRef
            d[f'pProbe{k}ShepStar'] = float(shepOut[k]) / pRef

    if setup.nSurface:
        val, nnbr, wc, _shep = _mlsPressure('surf')
        for k in range(setup.nSurface):
            d[f'pSurf{k}'] = float(val[k])
            d[f'pSurf{k}Star'] = float(val[k]) / pRef
            d[f'pSurf{k}Nnbr'] = int(nnbr[k])
            d[f'pSurf{k}WC'] = int(bool(wc[k]))
    return d


#: The two panels a dam break actually ships with (`plotDensity=False`).
#: Velocity is the flow; density is what the boundary treatment is graded on --
#: a wall that attracts, a tongue that de-densifies, or a corner that pumps
#: shows up here and in no other field (`DELTASPH_VALIDATION_PLAN.md` 5.5-5.9).
#: `scaling='Symmetric'` is what makes `midPoint` take effect at all --
#: `warpSPHPlotting.math.getBounds` only applies it on the Symmetric branch, so
#: a diverging map left on the default `Linear` silently becomes a plain min-max
#: stretch with "white" wherever the data minimum happens to be, not at rho0
#: (5.6).
_DAMBREAK_DENSITY_FIELD = Field(
    'densities', 'Particle Density', colorMap='managua', colorMapKind='diverging',
    flip=True, midPoint=1.0, scaling='Symmetric', boundary='Visualize',
    plotTitleGap=0.08)

#: The cyclic-coloured particle IDs show the fluid folding over itself at the
#: free surface, which no scalar field does. Kept for `--plotIDs`, but not the
#: default: on these validation runs the question is almost always what the
#: density is doing at a wall.
_DAMBREAK_UID_FIELD = Field(
    'UIDs', 'Particle IDs', colorMap='twilight', colorMapKind='cyclic',
    midPoint=None, plotTitleGap=0.08)

DAMBREAK_FIELDS = [
    Field('velocities', 'Particle Velocity Magnitude', colorMap='viridis',
          mapping='L2Norm', plotTitleGap=0.08),
    _DAMBREAK_DENSITY_FIELD,
]

#: `--plotDensity`: velocity, density and the ID panel together.
DAMBREAK_FIELDS_DENSITY = [
    DAMBREAK_FIELDS[0],
    _DAMBREAK_DENSITY_FIELD,
    _DAMBREAK_UID_FIELD,
]


def dambreakFields(ctx: RunContext):
    """The panel list this run plots -- two, or three with `--plotDensity`."""
    return DAMBREAK_FIELDS_DENSITY if ctx.param('plotDensity') else DAMBREAK_FIELDS


def _figsize(ctx: RunContext):
    # The dam break box is much wider than it is tall, so this case carries its
    # own figure size rather than the 11x5 default.
    return (ctx.param('plotWidth'), ctx.param('plotHeight'))


def setupPlot(ctx: RunContext, state):
    # `exportFrame0=False` + a manual export after `openWindow`: the vispy
    # canvas has not settled to its real physical size until `openWindow`'s
    # first show()/layout pass, so exporting frame 0 before it (as
    # `buildFieldPlotter` does by default, for the no-window notebook case)
    # captured every run's frame 0 at a different resolution than the rest
    # of the video (measured: 2688x768 vs the steady 1440x768) -- the video's
    # first frame rendered squished/wrong-aspect relative to every later one.
    plotter = buildFieldPlotter(ctx, state, dambreakFields(ctx), figsize=_figsize(ctx),
                                exportFrame0=False)
    openWindow(ctx, plotter)
    _export(ctx, plotter, 0, dpi=300)
    return plotter


def updatePlot(ctx: RunContext, state, plotter, step: int) -> None:
    refreshFieldPlotter(ctx, state, plotter, dambreakFields(ctx), step=step)
    if getattr(ctx.spec, 'show', True):   # no live window (openWindow skipped it) -> nothing to repaint
        pumpEvents(plotter)


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    simSetup = ctx.scratch['simSetup']
    # `None`-default params (`alpha`, `referenceVelocity`, `shifting`, ...) must
    # be filtered the same way list/dict ones are: `writeInitialData` sets each
    # surviving value as an HDF5 attribute, and `h5py` has no native type for
    # `None` (`numpy.array(None)` is `dtype=object`) -- writing one raises
    # `TypeError: Object dtype dtype('O') has no native HDF5 equivalent` and
    # aborts `store=True` runs outright, not just this attribute.
    data = {k: v for k, v in ctx.spec.params.items()
            if v is not None and not isinstance(v, (list, dict))}
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
        # Sun/Marrone/DualSPHysics all use Wendland C2 (h/dp=2, support=2h=4dp
        # -- the same 4*dx this case's n_h=4.0 already gives). This case used
        # to default to C4; an A/B on the Marrone dam break
        # (`scripts/probe_deltaSPHMarrone.py --kernel Wendland2`) came out
        # marginally cleaner at every resolution tested than C4, and C4 was
        # never a deliberate physics choice recorded anywhere in this plan --
        # just the case's inherited default. `DELTASPH_VALIDATION_PLAN.md`
        # Part 1 / Part 3.
        kernel='Wendland2',
        # `symplecticEuler` default since WCSPH_DEFAULT_CLOSEOUT_PLAN.md item
        # E -- not unstable on any case tested (Marrone 3.1/3.4 at multiple
        # resolutions/durations, sloshingTank's full 7s combo; the earlier
        # divergences recorded elsewhere in this codebase were traced to
        # `mdbcNoPenShiftMode='derivative'`, superseded by `'finalize'`, not
        # to the integrator itself). Sun et al. 2017 §2 integrate the δ-SPH
        # system with RK4 (`DELTASPH_VALIDATION_PLAN.md` Part 1) -- a script
        # reproducing that exact numerical method for a literature comparison
        # (e.g. `scripts/probe_deltaSPHMarrone.py`'s Sun2017/Marrone legs)
        # must now pass `--integrationScheme rungeKutta4` explicitly rather
        # than relying on this case's own default; several of this plan's own
        # recorded comparison numbers were produced under the *old* implicit
        # RK4 default and are not reproducible bit-for-bit with a bare
        # no-override command anymore.
        integrationScheme='symplecticEuler',
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
        # Tank wall boundary condition. **FREE SLIP is the default**, because
        # that is what every reference this case is graded against specifies:
        # Marrone 2011 Sec. 3 (3.1 and 3.4.1), Lobovsky et al. 2014 / Buchner
        # 2002, and De Courcy et al. 2024 Sec. 4.5 ("a free-slip condition is
        # imposed on the boundary conditions due to the relatively high
        # Reynolds number considered").
        #
        # `'constant'` was the historical default and is NOT free slip, despite
        # adding no physical viscosity term: the wall particles sit at v = 0
        # (nothing writes them; there is no rigid body) while
        # `computeVelocityDiffusion` runs `AllToAll`, so the *artificial*
        # viscosity drags the fluid against a stationary bed -- an effective
        # NO-slip wall. `DELTASPH_VALIDATION_PLAN.md` 5.7 measured the cost:
        # the front runs 3.4 m/s instead of 4.13 (reference ~3.5-4.7), impact
        # moves from t* ~ 2.8 to ~ 2.6 (reference 2.3), and `pb@front` goes
        # negative (-0.014/-0.018) as the front passes, i.e. an *attracting*
        # wall that pulls particles into the boundary band. It was kept as the
        # default only so recorded runs would not move; every run since has
        # passed `freeSlip` explicitly, so the default is now the correct one.
        # Pass `'constant'` to reproduce a pre-flip number, or `'noSlip'` for
        # Marrone Sec. 3.4.2's viscous half, which genuinely wants it.
        wallBC='freeSlip',
        # 'particles' (default): boundary particles + ghost nodes (mDBC). 'analytic': the tank walls are a
        # warpSPHBoundaries body (exact boundary integrals, no wall particles); plain tank only.
        wallRepresentation='particles',
        # analytic walls only: 'hydrostatic' (the wall pressure gradient rho (g - a_w) in all directions: exact for a fluid in hydrostatic balance) or
        # 'normal' (only dp/dn = rho (g - a_w) . n, as the boundary particles' ghost extrapolation: no tangential force on a fluid in free fall)
        analyticWallPressure='hydrostatic',
        targetDt=0.0005,
        # Downstream-wall pressure sensors (`ACSPH_PLAN.md` §4.5): heights above
        # the tank bed, in the case's length unit. Empty -> no probing. See
        # `diagnostics`; `scripts/probe_acsphDambreakLobovsky.py` sets these to
        # Lobovsky et al. 2014's five-sensor matrix at that paper's tank scale.
        pressureProbeHeights=[],
        pressureProbeInset=0.0,
        # MLS gather radius multiplier for the wall probe (1.0 = one kernel
        # support). > 1 mimics a finite-face transducer's area integration --
        # see `diagnostics`. Superseded by `pressureProbeDiscRadius` for a
        # true disc integral; kept for cases that still want the cheaper
        # single-point-with-wide-support approximation.
        pressureProbeSupportScale=1.0,
        # Physical radius (case length unit) of a flush wall transducer disc
        # to area-integrate each probe over, rather than reading one point.
        # 0.0 (default) -> old single-point behaviour, unchanged.
        # `scripts/probe_deltaSPHMarrone.py` sets this to Marrone 2011's
        # φ = 90 mm probe disc (radius 0.045 m) -- see `diagnostics`.
        pressureProbeDiscRadius=0.0,
        # Sensors anywhere on the solid surface -- not just the axis-aligned +x
        # wall. Each entry is `[x, y, nx, ny]`: a point in centred-domain coords
        # and the outward unit normal pointing into the fluid. `diagnostics`
        # pushes the query point `surfacePressureProbeInset` spacings along the
        # normal and MLS-interpolates the fluid pressure there, emitting
        # `pSurf{k}` / `pSurf{k}Star` / `pSurf{k}Nnbr`. Empty -> skipped.
        # `scripts/probe_deltaSPHMarrone34.py` sets Marrone 2011 §3.4's nine
        # probes on the 45° edge, the obstacle roof and the fillet arc.
        # A list, not a tuple: `buildArgumentParser` only emits flags for
        # scalars/lists and `test_everyCaseDeclaresItsParamsAsScalarsOrLists`
        # enforces that, so a tuple here is silently un-overridable from the CLI.
        surfacePressureProbes=[],
        # 0.0 = query exactly on the wall point. The first-order MLS fit
        # recovers a linear field from the one-sided wall stencil, so on a
        # hydrostatic column the on-wall probe reads ρg·depth to < 1 % (floor
        # +0.4 %, side walls ±1 %, corners ±0.1 %) with no Shepard fallback --
        # `scripts/probe_hydrostaticPressureProbe.py`. A positive inset just
        # displaces the sample point `inset` spacings up the normal and reads
        # the (lower) pressure there, so it *degrades* the wall reading.
        surfacePressureProbeInset=0.0,
        # Viscosity on the delta-SPH path. `inviscid=True` (default, Marrone
        # §3.1/§3.4.1 + the Lobovsky dam break) uses the artificial-viscosity
        # term with coefficient `alpha`; `inviscid=False` (Marrone §3.4.2) uses
        # the physical Monaghan & Gingold Laplacian with kinematic viscosity
        # `nu`. See `configureScheme`; `scripts/probe_deltaSPHMarrone34.py --Re`
        # sets `nu = √(gH)·H / Re`. `alpha` (None -> leave the scheme's own
        # artificial-viscosity coefficient) only matters on the inviscid path.
        inviscid=True,
        alpha=None,
        nu=0.0,
        # Sun et al. 2017 Sec. 2: freeze the delta-SPH diffusive terms across
        # RK sub-stages instead of recomputing them fresh at each stage's
        # intermediate state. None (default) -> don't override; whichever
        # scheme is selected keeps its own default (`--scheme sun2017DeltaSPH`
        # already defaults this True). Pass True/False explicitly to force it
        # either way regardless of scheme. `DELTASPH_VALIDATION_PLAN.md`
        # Part 1/5.1.
        freezeDiffusionAcrossStages=None,
        # Particle shifting (the delta+ PST). None (default) -> leave the
        # selected scheme's own setting, which is ON -- `ShiftProperties.active`
        # defaults True. False is Marrone et al. 2011 Sec. 3's actual
        # configuration and this plan's own acceptance gate; see
        # `configureScheme` and `DELTASPH_VALIDATION_PLAN.md` Sec. 5.1.1.
        shifting=None,
        # Start the fluid on the weakly-compressible hydrostatic density profile
        # rather than a uniform rho0 -- see `initialConditions`. Off by default
        # (a collapsing dam-break column does not want it); a still-water case
        # (English 2022 §4.1) does. `DELTASPH_VALIDATION_PLAN.md` §5.2.1.
        hydrostaticInit=False,
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
        # Snap the interior wall box onto the sampling-lattice mid-gaps so the
        # first particle band sits ~dx/2 proud of each flat wall
        # (`alignInteriorDomainToLattice`). Default on; False keeps the raw
        # lattice phase (a row can land on the wall) for A/B.
        alignBoundaryLattice=True,
        # ACSPH only: keep the non-paper acceleration constraint in Eq. (46)'s
        # timestep (default). Set False for the paper's literal advective +
        # viscous constraint set -- see `_configureArtificialCompressibleExtra`.
        acAccelConstraint=True,
        # ACSPH only: `ArtificialCompressibleSPHConfig.noPenetrationShift`, the
        # mDBC wall-confinement safeguard -- off by default repo-wide (it is
        # not in De Courcy et al. 2024, only the free-slip velocity mirror
        # is). `FREESLIP_DAMBREAK_FINDINGS.md` item 1: under a violent dam-break
        # impact the mirror alone lets fluid leak past the wall (measured at
        # nx=24: 77 particles / 7.4 dx of penetration by t* = 8.09), and at
        # finer resolution (nx=70) that same leak has been observed to fling a
        # single particle to extreme velocity, which pins Eq. (46)'s advective
        # `dt` at its floor and freezes simulated time (`stallDtSteps`). With
        # this on at nx=24 the leak drops to 0 particles / 1.05 dx by the same
        # t*, peak velocity drops 9.68 -> 5.95 m/s, and the run needs *fewer*
        # steps (no escaper tightening the CFL) -- not yet confirmed at nx=70,
        # the resolution where the stall itself was observed.
        noPenetrationShift=False,
        # ACSPH only: override `acParams.epsilonV` (the paper's `-6.0`
        # default). `None` leaves it alone; see `_configureArtificialCompressibleExtra`
        # for why a float32 run wants `-5.0`.
        epsilonV=None,
        # ACSPH only: the paper's artificial viscosity `alpha_nu` (Sec. 4:
        # 0.01, with `c0 = 50 sqrt(g H)`). `None` falls back to the physical
        # `acParams.nu`; see `_configureArtificialCompressibleExtra`.
        acAlphaNu=0.01,
        # ACSPH only: `acParams.cavitationProjection` -- unilateral wall
        # contact where air can reach ('vicinity'), `MDBC_CONTACT_LINE_PLAN.md`.
        # 'off' is the paper.
        acCavitationProjection='off',
        # ACSPH only: `acParams.isolatedZeroPressure` -- p = 0 on rows with an
        # empty support (MDBC_CONTACT_LINE_PLAN.md §11.1). Off is the paper.
        acIsolatedZeroPressure=False,
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

        markerSize=None,
        plotWidth=28,
        plotHeight=8,
        plotDensity=False,
    ),
))


if __name__ == '__main__':
    caseMain(dambreakCase)
