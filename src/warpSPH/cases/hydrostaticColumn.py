"""Hydrostatic column (2D, wall-bounded), divergence-free incompressible SPH.

Baseline case 3 of `DFSPH_IMPROVEMENT_PLAN.md`'s "what is left" item 2: a
fluid with gravity in a container, at rest. Nothing much should happen -- the
velocities stay ~0, no drift, and the column holds.

**Status: the scheme currently does NOT pass this case.** `divergenceFree`
diverges in this case (Part 23 of the plan): the column falls, the free
surface compacts, and the velocity and pressure grow without bound. At
`nx=32` (fixed `dt=1e-2`) the run hits NaN/Inf within ~5-8 steps; at `nx=64`
(adaptive `dt`) it does not reach Inf by `t=1`, but it is already grossly
unphysical by step ~25 -- max velocity ~14 in a unit domain (peaking at ~73),
displacements exceeding the whole domain, and a fitted pressure slope ~10x
the hydrostatic value and still climbing. The case is kept as the *failing*
baseline that exposes the limitation; its figures of merit are what a correct
scheme would show, and the "why the scheme fails it" section below records
the diagnosis.

**The physics.** A column of fluid fills the bottom `fillRatio` of a walled
box; gravity points down; the flat top is a free surface. The exact solution
is static: `v = 0`, and the pressure is linear in depth,

    p(y) = p0 - rho0 g y        (g > 0, gravity in -y)

with the surface the level where `p` reaches its gauge value. Two axes are
measured separately, for the same reason `shearWave` separates amplitude from
density:

- `pressureSlopeRatio` -- the fitted `dp/dy`, divided by the analytic
  `-rho0 * g`: 1.0 is the exact hydrostatic gradient. This is the axis that
  grades the pressure *field*.
- `pressureResidual` -- the rms departure of the pressure from the best
  straight line, normalised by the column's own pressure drop
  `rho0 g H`: the spurious-noise axis, independent of the slope. Free-surface
  pressure noise is exactly this -- a field whose gradient is right on
  average but that is not a straight line.

The fit runs over the **bulk only**: a `bulkMargin`-spacing skin off the free
surface (where the profile legitimately departs from the straight line, since
the surface is where the pressure meets its gauge level) and a `wallMargin`
off the walls (where the boundary treatment, not the bulk physics, rules).

**The density axis is graded spray-robust.** Under `--scheme iisph` (and to a
lesser extent the others) the bulk slosh throws 1-3 fluid particles 1-3 dx
above the free surface; isolated, they read `rho` ~0.14-0.3 by kernel
deficiency and dominate the plain `minDensity`, which then tracks cosmetic
ballistic spray rather than the column (DFSPH_FINDINGS.md Part 33/34). So the
diagnostics also report `densityP05` (5th percentile of the fluid density) and
`embeddedMinDensity` (min over fluid rows more than 1 dx below the
95th-percentile surface height) -- both grade the column body, and both hold
~0.9 for a run `iisph` completes clean while `minDensity` swings 0.2-0.6.

**One caveat on the pressure axis.** The stored pressure is the
divergence-free solve's (the step's projection), which `solveDivergenceFree`
re-centres to zero mean over the fluid every iteration -- a pure gauge choice,
consistent with `DFSPH_IMPROVEMENT_PLAN.md` §1.4/§1.5 (the operator's constant
mode is null). So the level `p0` is not a figure of merit; the *gradient* is.
And for this scheme the hydrostatic support does not live in the stored DF
pressure: the DF projection enforces `div v = 0`, and its source is exactly
zero for the uniform gravity velocity `v* = dt*g` (see below), so the stored
field is flat, not hydrostatic. A correct scheme would show the balance in
this gradient; here it is a diagnostic of the failure, not of correctness.

**Why the scheme fails it (Part 23).** Not a case-setup artifact -- the
failure is specific to the quiescent, wall-bounded, free-surface-under-gravity
state and is the limitation this case exists to expose. Two coupled mechanisms:

1. **The DF projection cannot balance a uniform body force.** Its source is
   `-div(v*)`, and the mDBC operator's divergence is exactly 0 for the
   gravity-driven `v* = dt*g` (the wall is velocity-transparent to uniform
   normal motion, and a uniform field is divergence-free). So the DF solve's
   correct solution is a flat pressure field -- it enforces `div v = 0`, which
   it cannot use to oppose gravity. The column's support must come from the
   constant-density solve's fall-and-push-back cycle instead.
2. **That cycle is unstable here.** (a) The DF Jacobi carries a
   boundary-layer mode the mean-residual convergence test cannot see: the
   per-iteration mean residual decays toward the tolerance while the
   wall/surface-adjacent pressure amplitude grows monotonically, and the
   0.75x warm start of the large hydrostatic gradient feeds it -- so each
   step's early exit is clean on the mean and locally divergent. (b) The
   constant-density solve's pressure drifts under the `nonNegativeClamp`
   gauge -- the free-surface guard downgrades `minShift` to the clamp, which
   does not pin the constant null mode -- compounding the free-surface
   compaction. `|div v'|/|div v*|` is ~4.3 (amplification, not damping),
   `|a_p|max` grows 35 -> 49 -> 79 -> 2549 over the first five steps, and the
   free surface then compacts to 1e12. The documented "better wall"
   configurations do not stabilise it: `inStepVelocity` diverges *faster*,
   `forceGauge` (keeping `minShift` on the surface) diverges, and a zero
   initial pressure diverges more slowly but still diverges (Part 23 A/B).

**Initial state.** The sampled lattice is jittered first (the same
de-correlation `tgv`, `shearWave` and `staticBlob` apply): a perfectly regular
lattice is an unstable equilibrium whose own setup transient would be measured
as motion, and the displacement figures of merit are taken from the jittered
state. The jitter moves fluid particles only -- the wall band is the
container and stays put. The constant-density pre-relaxation `tgv`/`shearWave`
get is deliberately not run here: it drives the free surface toward its `0.9`
clamp floor before the run even starts (see `staticBlob`'s docstring for the
measurement). Fluid *and* wall pressure are initialised to the analytic
hydrostatic profile in the DF gauge (see `initialConditions`): the exact
at-rest state is the correct starting point for this test, and a zero start
only trades "maintain the balance" for "build it up from zero", which the
0.75x warm start turns into a slower version of the same divergence.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..configurations.moduleConfigurations.gravity import GravityType
from ..configurations.region import BCType
from ..enumTypes import IncompressibleSPHScheme
from ..modules import shuffleParticles
from ..runner import Case, RunContext, caseMain, registerCase
from .kolmogorovIncompressible import kolmogorovIncompressibleTimestep
from .plotting import Field, particlePlot
from .randomFlow import BOUNDED_BAND
from .weaklyCompressible import (WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, boundaryRegion,
                                 buildRegionSystem, configureWeaklyCompressible,
                                 domainBoundarySdf, fluidRegion, shapeSdf)

__all__ = ['hydrostaticColumnCase']


def columnSdf(ctx: RunContext):
    """The fluid box: full interior width, bottom-anchored, `fillRatio` deep."""
    L = ctx.spec.L
    fill = ctx.param('fillRatio')
    halfHeight = 0.5 * fill * L
    centre = [0.0, -0.5 * L + halfHeight]
    return shapeSdf('box', args=[[0.5 * L, halfHeight]], offset=centre)


def configureScheme(ctx: RunContext) -> None:
    # The container walls are the point of the case: with `band` at its
    # (periodic) default of 0 the wall region encloses no volume and the case
    # would silently run as a free-falling column, so fall back to
    # `randomFlow`'s bounded band exactly the way `randomFlow --bounded` does.
    if not ctx.param('band'):
        ctx.spec.params['band'] = BOUNDED_BAND
    configureWeaklyCompressible(ctx)

    schemeConfig = ctx.schemeConfig
    schemeConfig.surfaceDetectionConfig.active = True
    # `configureWeaklyCompressible` wires the shared `inviscid`/`alpha`
    # (artificial-viscosity) knobs `deltaSPH` uses; DFSPH has no such term
    # (see `kolmogorovIncompressible`), so this scheme runs with a plain
    # physical viscosity instead.
    schemeConfig.diffusionParams.inviscid = False
    schemeConfig.diffusionParams.viscidNu = ctx.param('nu')
    schemeConfig.shiftProperties.active = ctx.param('shifting', False)
    # Post-solve fluid XSPH velocity smoother in `dfsph_step`, in units of
    # omniSPH's own `XSPH_FLUID = 0.05` (DFSPH_FINDINGS.md 1.16). The residual
    # `|v|` this case shows under `divergenceFree` is a bounded, undamped
    # free-surface limit cycle; this otherwise-inviscid scheme has no other
    # sink for it. `1.0` takes nx=64 KE 3e-5 -> 5e-7 and nx=128 KE 7e-4 ->
    # 3e-7 (`densityP05` / `embeddedMinDensity` -> 1.000 at nx=128) for a ~3%
    # `densityP05` cost at nx=64. Default 0.0 -- opt-in, and only this case.
    schemeConfig.xsphFilterScale = ctx.param('xsphScale', 0.0)
    schemeConfig.gravityConfig.active = True
    schemeConfig.gravityConfig.type = GravityType.Directional
    schemeConfig.gravityConfig.magnitude = ctx.param('gravityMagnitude')
    schemeConfig.gravityConfig.origin = ctx.param('gravityDirection')
    # The surface-detection bandwidth is measured in particle spacings.
    schemeConfig.bandwith = ctx.spec.L / ctx.param('bandWidth') / ctx.config.dx


def buildSystem(ctx: RunContext):
    # `wallBC` selects the container's boundary condition: `freeSlip` (the
    # historical default -- reflect the normal component, no tangential drag)
    # or `noSlip` (mirror the fluid velocity -> a real viscous wall once
    # `nu > 0`, `DFSPH_FINDINGS.md` 1.14). Both go through
    # `computeBoundaryVelocities`; the schemes pick it up in step 2.
    wallBC = BCType[ctx.param('wallBC')]
    regions = [fluidRegion(ctx, columnSdf(ctx)),
               boundaryRegion(ctx, domainBoundarySdf(ctx), kind=wallBC)]
    return buildRegionSystem(ctx, regions)


def initialConditions(ctx: RunContext, system) -> None:
    # Jitter the fluid only -- see the module docstring for why the
    # constant-density pre-relaxation is not run on a free-surface state.
    particles = system.state
    # particles.positions = shuffleParticles(
    #     particles, ctx.config, ctx.schemeConfig, 0,
    #     jitterAmount=ctx.param('jitter'))
    particles.velocities[:] = 0.0

    # Optional rest-density calibration (Part 33, `dfsphReference` only in
    # practice). `tgv`/`kolmogorovIncompressible` normalise the fluid mass so
    # the sampled lattice integrates to `rho0` (DFSPH_FINDINGS.md §1.1); this
    # case does not, so at `n_h = 4` the at-rest bulk reads ~0.95 rho0. That
    # ~5% deficit is the constant-density solve's source `s = 1 - rho/rho0` at
    # rest: on step 1 the two forced Jacobi iterations (minIters = 2) see
    # `s > 0` everywhere and drive the IC hydrostatic seed to exactly 0
    # (Part 31), so the reference scheme cold-starts and the column slumps --
    # the slump feeds the free-slip vortex pair Part 32 could only damp at the
    # wall. Calibrating the bulk to `rho0` makes `s ~ 0` at rest, so the seed
    # survives and the column starts balanced. Off by default: `divergenceFree`
    # is graded against the uncalibrated sampling and §1.1 shows calibration
    # hurts the periodic cases; this is the bounded free-surface case where
    # `rho = rho0` at rest is the physically correct target.
    if ctx.param('calibrateRestDensity'):
        from ..modules import computeDensities
        dens = computeDensities(particles, ctx.config, ctx.schemeConfig, None)
        fluid0 = particles.kinds == 0
        y = particles.positions[:, 1]
        dx = ctx.config.dx
        ylo = y[fluid0].min()
        yhi = y[fluid0].max()
        bulk = fluid0 & (y > ylo + 3.0 * dx) & (y < yhi - 4.0 * dx)
        ref = float(dens[bulk].median()) if int(bulk.sum()) >= 8 \
            else float(dens[fluid0].median())
        rho0 = ctx.schemeConfig.fluid.restDensity
        particles.masses = particles.masses / ref * rho0
        if ctx.spec.verbose:
            check = computeDensities(particles, ctx.config, ctx.schemeConfig, None)
            print(f'[hydrostaticColumn] rest-density calibration: bulk median '
                  f'{ref:.4f} -> {float(check[bulk].median()):.4f} (rho0={rho0})')

    # the above code is garbage

    # Start from the exact at-rest state, not from zero pressure. The step's
    # projection warm-starts from 75% of the incoming pressure (see
    # `solveDivergenceFree`), so a zero start spends its first steps *building*
    # the hydrostatic balance, and the early-termination residual of each of
    # those steps leaks a fraction of `g` into the velocity -- the column
    # visibly falls before it is supported. Initialising the analytic profile
    # makes the run grade whether the scheme *maintains* the balance, which is
    # what the case exists to measure.
    #
    # Gauge: `solveDivergenceFree` re-centres the *fluid* pressure to zero
    # mean every iteration, and the boundary rows are frozen at their incoming
    # values -- so the boundary is initialised at the same shifted profile,
    # not the raw one. A raw initialisation would leave a fluid-versus-wall
    # jump of the fluid mean's size at every contact, a spurious wall force
    # with nothing to do with the scheme under test. Above the free surface
    # the gauge pressure is 0 (the clamp), matching the air side.
    positions = particles.positions
    fluid = particles.kinds == 0
    surfaceY = positions[fluid, 1].max()
    p = ctx.schemeConfig.fluid.restDensity * ctx.param('gravityMagnitude') \
        * (surfaceY - positions[:, 1])
    p = torch.clamp(p, min=0.0)
    if ctx.param('wallPressure') == 'zero':
        p = torch.where(fluid, p, torch.zeros_like(p))
    # `dfsphReference` (DFSPH proper) and `iisph` both carry a *non-negative*
    # pressure/kappa and warm-start their constant-density solve from it
    # directly, so they want the raw hydrostatic profile (0 at the surface,
    # rho0 g H at the floor). `divergenceFree` (VD+PS) re-centres the fluid
    # pressure to zero mean every iteration, so it is initialised at that
    # shifted profile instead -- a raw start would open a fluid-vs-wall jump
    # of the mean's size. See the module docstring.
    if ctx.scheme in (IncompressibleSPHScheme.dfsphReference,
                      IncompressibleSPHScheme.iisph):
        particles.pressures = p
    else:
        particles.pressures = p - p[fluid].mean()

    ctx.scratch['initialPositions'] = positions.clone()


def hydrostaticDiagnostics(ctx: RunContext, state) -> Dict[str, float]:
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
        'maxDensity': densities.max().detach().cpu().item(),
        'densityStd': densities.std().detach().cpu().item(),
    }

    # Spray-robust density FOMs (plan item 1 / Part 33). The plain `minDensity`
    # on this case is dominated by 1-3 fluid particles thrown 1-3 dx above the
    # free surface by the bulk slosh: isolated, reading low by kernel deficiency,
    # never the same particles two samples running (measured in
    # `probe_dfsphReferenceColumnSurface.py`). Cosmetic ballistic spray, not
    # structural loss. These two grade the column body instead:
    #   `densityP05`         -- 5th percentile of the fluid density, so a
    #                           handful of low outliers cannot move it;
    #   `embeddedMinDensity` -- the min over fluid rows more than 1 dx below the
    #                           95th-percentile surface height, i.e. with the
    #                           ballistic skin removed outright.
    dx = ctx.config.dx
    fluidY = positions[:, 1]
    surfY95 = torch.quantile(fluidY, 0.95)
    d['densityP05'] = torch.quantile(densities, 0.05).detach().cpu().item()
    embedded = fluidY < surfY95 - dx
    if bool(embedded.any()):
        d['embeddedMinDensity'] = densities[embedded].min().detach().cpu().item()

    initial = ctx.scratch.get('initialPositions')
    if initial is not None:
        disp = positions - initial[fluid]
        d['dispRms'] = torch.sqrt((disp ** 2).sum(dim=-1).mean()).detach().cpu().item()
        d['dispMax'] = torch.linalg.norm(disp, dim=-1).max().detach().cpu().item()

    pressures = particles.pressures
    if pressures is None:
        # Before the first step the divergence-free solve has not run yet.
        return d

    L = ctx.spec.L
    dx = ctx.config.dx
    g = ctx.param('gravityMagnitude')
    rho0 = ctx.schemeConfig.fluid.restDensity
    surfaceY = positions[:, 1].max()
    bulk = ((positions[:, 1] < surfaceY - ctx.param('bulkMargin') * dx)
            & (positions[:, 1] > -0.5 * L + ctx.param('wallMargin') * dx))
    if bulk.sum() < 8:
        return d

    y = positions[bulk, 1]
    p = pressures[fluid][bulk]
    # Least-squares line p = a + b y.
    yBar, pBar = y.mean(), p.mean()
    denom = ((y - yBar) ** 2).sum()
    # `denom == 0` (a single bulk y-level, e.g. a very coarse column) collapses
    # the fit -- fall back to a zero slope, and keep `b` a tensor so the
    # `.detach()` calls below still work.
    b = (((y - yBar) * (p - pBar)).sum() / denom if denom > 0
         else torch.zeros((), device=y.device, dtype=y.dtype))
    a = pBar - b * yBar
    residual = p - (a + b * y)
    # The analytic gradient: dp/dy = -rho0 g for gravity (0, -g).
    d['pressureSlope'] = b.detach().cpu().item()
    d['pressureSlopeRatio'] = (b / (-rho0 * g)).detach().cpu().item()
    d['pressureResidual'] = (
        residual.pow(2).mean().sqrt() / (rho0 * g * ctx.param('fillRatio') * L)
    ).detach().cpu().item()
    return d


setupPlot, updatePlot = particlePlot([
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm', boundary = 'Visualize'),
    # Pressure is the point of the case; span the hydrostatic range the
    # defaults imply (rho0 g H = 9.81 * 0.5, gauge-centred, so ~ +/- 2.5).
    Field('densities', 'densities', colorMap='viridis', boundary = 'Visualize'),
])


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    return {k: ctx.param(k) for k in hydrostaticColumnCase.params}


hydrostaticColumnCase = registerCase(Case(
    name='hydrostaticColumn',
    scheme='divergenceFree',
    description='Hydrostatic column in a walled box (2D): at rest under gravity, '
                'pressure linear in depth.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=hydrostaticDiagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    timestep=kolmogorovIncompressibleTimestep,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='07-hydrostaticColumn',
        dim=2,
        nx=128,
        L=1.0,
        tLimit=1.0,
        periodic=True,
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
        # The column fills the bottom half of the box: a 0.5-deep column in a
        # 1.0 box, hydrostatic pressure drop rho0 g H = 9.81 * 0.5.
        fillRatio=0.5,
        gravityMagnitude=9.81,
        gravityDirection=[0.0, -1.0],
        bandWidth=16.0,
        shifting=False,
        # Part 33: normalise the fluid mass so the at-rest bulk summation
        # density lands on `rho0` (see `initialConditions`). Off keeps the
        # historical sampling; `dfsphReference` probes turn it on to stop the
        # step-1 IC-seed cold start.
        calibrateRestDensity=False,
        # Container boundary condition (`BCType` name): `freeSlip` (default,
        # historical) or `noSlip` (a viscous no-slip wall once `nu > 0` --
        # decays the free-slip bulk slosh, `DFSPH_FINDINGS.md` 1.14).
        wallBC='freeSlip',
        # Post-solve fluid XSPH scale for `divergenceFree` (`DFSPH_FINDINGS.md`
        # 1.16), units of `XSPH_FLUID = 0.05`. 0 = the graded default; ~1.0
        # decays the free-surface residual limit cycle.
        xsphScale=0.0,
        # Lattice de-correlation (`shuffleParticles`, `shiftIters=0`); see
        # `initialConditions` for why the constant-density pre-relaxation
        # `tgv`/`shearWave` use is not run here.
        jitter=0.01,
        # Bulk margins for the pressure fit, in particle spacings.
        bulkMargin=8.0,
        wallMargin=6.0,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(hydrostaticColumnCase)
