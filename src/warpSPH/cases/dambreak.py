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
from ..caseUtils.weaklyCompressible import alignInteriorDomainToLattice
from ..caseUtils.weaklyCompressible import buildObstacleSDF
from ..configurations.moduleConfigurations.gravity import GravityType
from ..enumTypes import isArtificialCompressibleScheme, isIncompressibleScheme
from ..initializers import initializeWeaklyCompressibleSimulation
from ..modules import setupWeaklyCompressibleTimestep
from ..modules.liu import interpolateLiuLiu
from warpSPHCore import OperationDirection
from ..runner import Case, RunContext, caseMain, registerCase
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .weaklyCompressible import particleDistributionMetrics, stepAccelerationDiagnostics
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
    schemeConfig.gravityConfig.origin = ctx.param('gravityDirection')
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
            ctx.config, ctx.schemeConfig, ctx.spec.L, ctx.param('W'))

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
        # Spray-robust companions to `min`/`maxDensity` -- see
        # `weaklyCompressible.weaklyCompressibleDiagnostics`. `maxDensity` at a
        # violent impact tracks a single jet-tip particle whose peak *sharpens*
        # with resolution (a fragmenting-jet case like Marrone 2011 §3.4 is not
        # converged in that quantity even at H/dx = 234); `densityP99` is the
        # bulk-compressibility measure a convergence check should band.
        'densityP05': torch.quantile(
            particles.densities[fluid].detach().float(), 0.05).cpu().item(),
        'densityP99': torch.quantile(
            particles.densities[fluid].detach().float(), 0.99).cpu().item(),
    }
    d.update(particleDistributionMetrics(ctx, state))
    d.update(stepAccelerationDiagnostics(state))
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

        # Penetration into a solid obstacle / concave wall fillet: the AABB
        # watch above cannot see these (a solid island in the flow, or a fillet
        # inside the tank AABB). `obstacleSDF` is negative inside the solid, so
        # `-sdf` clamped at 0 is the depth a fluid particle has sunk in.
        obSDF = ctx.scratch.get('obstacleSDF')
        if obSDF is not None:
            with torch.no_grad():
                sd = obSDF(pos)
            inside = sd < 0
            d['nObstaclePen'] = int(inside.sum().detach().cpu().item())
            d['maxObstaclePenDx'] = float(
                torch.clamp(-sd.min(), min=0.0).detach().cpu().item() / dx)

    # Pressure probes -- a first-order MLS (Liu-Liu) interpolation of the fluid
    # pressure at fixed sensor points, emitted every step so the trajectory
    # carries the P(t) signal each experimental sensor records. First order
    # rather than a bare Shepard gather so the local fit carries the pressure
    # gradient and needs no separate Adami hydrostatic correction at the
    # one-sided wall support; points with too few / near-coplanar fluid
    # neighbours fall back to a 0th-order Shepard gather, then to 0 (pre-arrival
    # / thin run-up sheet), which `*Nnbr` makes visible.
    #
    #  - `pressureProbeHeights` (`ACSPH_PLAN.md` §4.5, Lobovsky et al. 2014):
    #    sensors on the +x impact wall, given as heights above the tank bed in
    #    the case length unit; optional φ-disc area integration.
    #  - `surfacePressureProbes` (`DELTASPH_VALIDATION_PLAN.md` §5.2.2, Marrone
    #    2011 §3.4): sensors anywhere on the solid surface -- the 45° edge, the
    #    obstacle roof, the concave fillet arc -- none axis-aligned, so each is
    #    `[x, y, nx, ny]` (centred-domain point + outward normal into the fluid)
    #    and the query point is pushed `surfacePressureProbeInset` spacings along
    #    that normal to sit in the fluid rather than exactly on the wall.
    # Both are empty by default so no existing run changes.
    canProbe = interior is not None and getattr(particles, 'pressures', None) is not None
    probeHeights = ctx.param('pressureProbeHeights')
    surfaceProbes = ctx.param('surfacePressureProbes')
    haveSurface = surfaceProbes is not None and len(surfaceProbes) > 0
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

        def _mlsPressure(pts):
            """(M,2) query points -> (P:(M,) cpu, nNeighbours:(M,) cpu,
            wellConditioned:(M,) bool cpu), the first-order MLS fluid-pressure
            fit with the density2025.py / wallPressure.py fallback ladder: the
            MLS fit where well-conditioned, else a 0th-order Shepard gather
            (`b[:,0] / A_g[:,0,0]`), else 0. A query point can have plenty of
            neighbours (double digits) and still fail the determinant check --
            a thin sheet running along a wall is exactly this, near-coplanar
            neighbours but many of them -- and `interpolateLiuLiu` hard-zeros
            the first-order fit there by design; without the Shepard tier every
            such point reads a flat 0 regardless of how much real neighbour
            support it had. The Shepard tier is only a 0th-order average of the
            (one-sided) neighbourhood, so it reads biased *low* against a real
            pressure gradient -- `pSurf{k}WC` records which tier a probe used.
            Clamped >= 0."""
            v, _g, nn, A_g, b, wc = interpolateLiuLiu(
                pts, referenceParticles=particles,
                referenceQuantities=particles.pressures,
                config=probeConfig, neighbor_threshold=4,
                direction=OperationDirection.FluidToFluid, supportScale=supportScale)
            shepDen = A_g[:, 0, 0]
            shepVal = torch.where(shepDen > 0, b[:, 0] / shepDen.clamp_min(1e-12),
                                  torch.zeros_like(shepDen))
            v = torch.where(wc, v, shepVal).clamp(min=0.0)
            return v.detach().cpu(), nn.detach().cpu(), wc.detach().cpu()

        H = ctx.param('fillRatio') * ctx.spec.L
        g = ctx.param('gravityMagnitude')
        pRef = ctx.schemeConfig.fluid.restDensity * g * H          # rho0 g H
        t = float(state.t) if getattr(state, 't', None) is not None else 0.0
        d['tStar'] = t * (g / H) ** 0.5

        if probeHeights:
            xWall = float(interior.max[0].item()) - float(ctx.param('pressureProbeInset'))
            yBed = float(interior.min[1].item())
            discRadius = float(ctx.param('pressureProbeDiscRadius') or 0.0)
            # `pressureProbeDiscRadius > 0`: area-integrate over a flush wall
            # transducer disc (Marrone 2011 uses φ = 90 mm) via the 7-point
            # Gauss-Legendre chord quadrature above, instead of one point per
            # probe height. 0.0 (default): unchanged single-point behaviour.
            sOffsets = [discRadius * x for x in _GAUSS7_NODES] if discRadius > 0.0 else [0.0]
            pts = torch.tensor(
                [[xWall, yBed + float(z) + s] for z in probeHeights for s in sOffsets],
                device=allPos.device, dtype=allPos.dtype)
            val, nnbr, _wc = _mlsPressure(pts)
            val = val.view(len(probeHeights), len(sOffsets))
            nnbr = nnbr.view(len(probeHeights), len(sOffsets))
            if discRadius > 0.0:
                # Weight each quadrature sample by its disc chord factor,
                # excluding only genuinely dry samples (`nnbr <= 1`, matching
                # density2025.py's own Shepard-tier floor) and renormalising
                # over the rest -- a sample with real neighbour support always
                # contributes its (MLS-or-Shepard) value rather than a hard 0.
                baseW = torch.tensor(_DISC_CHORD_WEIGHTS, dtype=val.dtype)
                validSample = nnbr > 1
                w = baseW.unsqueeze(0) * validSample.to(val.dtype)
                wSum = w.sum(dim=1)
                valOut = torch.where(wSum > 0, (val * w).sum(dim=1) / wSum.clamp_min(1e-12),
                                     torch.zeros_like(wSum))
                nnbrOut = nnbr.amax(dim=1)
            else:
                valOut = val[:, 0]
                nnbrOut = nnbr[:, 0]
            for k in range(len(probeHeights)):
                d[f'pProbe{k}'] = float(valOut[k])
                d[f'pProbe{k}Star'] = float(valOut[k]) / pRef
                d[f'pProbe{k}Nnbr'] = int(nnbrOut[k])

        if haveSurface:
            arr = torch.as_tensor(surfaceProbes, dtype=allPos.dtype,
                                  device=allPos.device).view(-1, 4)
            nrm = arr[:, 2:4] / arr[:, 2:4].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            inset = float(ctx.param('surfacePressureProbeInset') or 0.0) * float(ctx.config.dx)
            val, nnbr, wc = _mlsPressure(arr[:, 0:2] + inset * nrm)
            for k in range(arr.shape[0]):
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
        # Tank wall boundary condition. `'constant'` is the historical default
        # and is kept so no existing run changes -- but note what it means: the
        # wall particles sit at v = 0 (nothing writes them; there is no rigid
        # body) and `computeVelocityDiffusion` runs `AllToAll`, so the
        # artificial-viscosity term drags the fluid against a stationary wall.
        # That is an effective NO-slip bed, while Marrone 2011 Sec. 3 specifies
        # FREE slip. `'freeSlip'` matches the fluid tangentially and exerts no
        # such drag. See `DELTASPH_VALIDATION_PLAN.md` 5.7.
        wallBC='constant',
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
        surfacePressureProbes=(),
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
