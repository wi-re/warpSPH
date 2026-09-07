"""Taylor-Green vortex (2D), weakly compressible.

The script form of this case was
`examples/weaklyCompressible/05-taylor-green-vortex.ipynb` -- the same vortex as
`warpSPH.cases.tgv`, but integrated by deltaSPH with an explicit physical
viscosity instead of by the divergence-free incompressible solver.

The analytic answer is `KE(t) = KE(0) exp(-4 nu k^2 t)`, which
:func:`effectiveViscosity` fits back out of the run.

That makes this case the family's viscosity calibration: it is the one setup
whose dissipation has a closed-form answer, so it is where "what did I ask for"
(`nu`, or `alpha` when `--inviscid`; :func:`viscosityScales` converts between
them) can be held against "what did the scheme actually apply"
(:func:`effectiveViscosity`). Sweeping `nu` over decades and fitting each run is
the notebook's last section.

.. note::
   This docstring used to record that the measured `nu_eff` lands *consistently
   below* the prescribed one, because `computeVelocityDiffusion`'s Monaghan
   switch (`approachOnly=True`) zeroed the viscous term for separating pairs and
   so halved it. That is **no longer true**: `schemes/deltaSPH.py` passes
   `approachOnly=False` (Marrone et al. 2011 Eq. (5b) / Sun et al. 2017 Eq. (1)
   are two-sided; `DELTASPH_VALIDATION_PLAN.md` Part 1), which is the Monaghan &
   Gingold velocity Laplacian proper. The decay now tracks `exp(-4 nu k^2 t)`
   to a couple of percent rather than at half rate.

Sun et al. 2019 benchmark no. 1
-------------------------------
`literature/sun2019_consistent-particle-shifting-delta-plus-sph.pdf` Sec. 3.1
uses exactly this flow -- four counter-rotating vortices on a periodic
`[0, L]^2`, `Re = U L / nu` -- as the delta+-SPH consistency benchmark, and it
is the one case in that suite with no boundary of any kind, so it measures the
scheme and nothing else. Its Eq. (22), `f(t) = f(t0) exp[-16 pi^2 nu (t - t0)]`,
is this module's `analyticDecayRate` written for `k = 2 pi / L`, and it governs
the *pressure* as well as the kinetic energy -- which is what makes Figs. 6-9
discriminating, since a weakly compressible scheme's pressure is
`c0^2 (rho - rho0)` and therefore reads the volume error directly. The three
diagnostics those figures need are `kineticEnergy` (already here), `pCentre`
(:func:`diagnostics`, MLS-interpolated at the box centre) and `volumeError`
(Sun Eq. (23)). `scripts/probe_deltaPlusTGV.py` drives and scores it.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from warpSPHCore import OperationDirection

from ..modules import alphaToNu, nuToAlpha, shuffleParticles
from ..modules.liu import interpolateLiuLiu
from ..runner import Case, RunContext, caseMain, registerCase
from ..sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS,
                                 configureWeaklyCompressible, paramExtraData,
                                 setupTimestep, weaklyCompressibleDiagnostics)

__all__ = ['tgvWeaklyCompressibleCase', 'effectiveViscosity', 'analyticDecayRate',
           'analyticKineticEnergy', 'analyticPressureAmplitude',
           'analyticCentrePressure', 'viscosityScales', 'wavenumber', 'phase',
           'MIN_STABLE_ALPHA']

#: Empirically, an artificial viscosity below this stops being reliably stable.
#: It is what makes "the largest Reynolds number this discretisation can carry"
#: a computable number rather than a matter of taste -- see
#: :func:`viscosityScales`.
MIN_STABLE_ALPHA = 0.01


def wavenumber(ctx: RunContext) -> float:
    """The TGV wavenumber actually stamped onto the velocity field."""
    return ctx.param('k') / 2.0


def analyticDecayRate(ctx: RunContext) -> float:
    """`4 nu k^2`, the exponential rate of the kinetic-energy decay."""
    return 4.0 * ctx.param('nu') * wavenumber(ctx) ** 2


def analyticKineticEnergy(ctx: RunContext) -> float:
    """`KE(0)` of the continuum vortex, `rho0 uMag^2 L^2 / 4`.

    The mean of `u^2 + v^2` over a whole number of periods of the TGV field is
    `uMag^2 / 2`, so the continuum answer needs no integration -- and comparing
    it to what the sampled particles actually carry is the cheapest check that
    the lattice, the masses and the shuffle all came out right.
    """
    return 0.25 * ctx.param('uMag') ** 2 * ctx.param('rho0') * ctx.spec.L ** 2


def viscosityScales(ctx: RunContext, state) -> Dict[str, float]:
    """The viscosity of the run as configured, in every form it has one.

    `nu` and `alpha` are the same dissipation written two ways -- physical
    kinematic viscosity and the deltaSPH artificial-viscosity coefficient --
    related by `nu = alpha c0 h / (2(n+2))` (Sun et al. 2016 against Marrone et
    al. 2012). Whichever one the case was given, the other follows once the
    sound speed and the mean support radius exist, which is why this takes a
    state rather than only the spec -- the built system before the run, or a
    `RunResult.state` after it; anything carrying `.state.supports`.

    Also returns the Reynolds number that implies, and the largest one this
    discretisation can carry: `alpha` cannot usefully go below
    :data:`MIN_STABLE_ALPHA`, and that floor is a viscosity floor, hence a
    Reynolds ceiling.
    """
    dim = ctx.spec.dim
    c0 = float(ctx.schemeConfig.fluid.fixedSoundSpeed)
    h = float(state.state.supports.mean().detach().cpu())
    # The velocity scale is uMag and the length scale is half the box: one
    # vortex, not the periodic tile that holds four of them.
    scale = ctx.param('uMag') * ctx.spec.L / 2

    if ctx.param('inviscid'):
        alpha = ctx.param('alpha')
        nu = alphaToNu(alpha, c0, h, dim)
    else:
        nu = ctx.param('nu')
        alpha = nuToAlpha(nu, c0, h, dim)
    nuLimit = alphaToNu(MIN_STABLE_ALPHA, c0, h, dim)

    return {
        'nu': nu, 'alpha': alpha, 'c0': c0, 'h': h,
        'Re': scale / nu if nu > 0 else float('inf'),
        'nuLimit': nuLimit, 'ReLimit': scale / nuLimit,
    }


def effectiveViscosity(result) -> float:
    """Fit `nu_eff` from a completed run's kinetic-energy history."""
    ts = result.series('t')
    energies = result.series('kineticEnergy')
    mask = (ts > 0) & (energies > 0)
    slope = np.polyfit(ts[mask], np.log(energies[mask] / energies[0]), 1)[0]
    k = result.ctx.param('k') / 2.0
    return -slope / (4 * k ** 2)


def configureScheme(ctx: RunContext) -> None:
    configureWeaklyCompressible(ctx)
    # The TGV box is [0, L]^2, not the symmetric box the shared block builds.
    domain = ctx.config.domain
    domain.min = torch.zeros(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype)
    domain.max = torch.ones(ctx.spec.dim, device=ctx.device, dtype=ctx.dtype) * ctx.spec.L

    # delta-SPH vs delta+-SPH is exactly this switch: the PST is what the `+`
    # denotes (Sun et al. 2017 Eq. (7)). The box is fully periodic, so there is
    # no free surface for the projection scheme to treat and the raw shift
    # applies everywhere -- which is what makes this the clean measurement of
    # the PST's own effect on the pressure and volume fields (Sun et al. 2019
    # Sec. 3.1). `shifting` defaults to None = leave the scheme's own default
    # (on) alone, so no existing run changes; pass False for the delta-SPH leg
    # of the A/B.
    shifting = ctx.param('shifting', None)
    if shifting is not None and hasattr(ctx.schemeConfig, 'shiftProperties'):
        ctx.schemeConfig.shiftProperties.active = bool(shifting)


def buildSystem(ctx: RunContext):
    system = setupBasicWeaklyCompressibleInitialState(
        ctx.spec.nx, ctx.config, ctx.schemeConfig, ctx.SimulationState, ctx.SimulationSystem)
    # A perfectly regular lattice is an unstable SPH equilibrium; the shuffle is
    # what keeps the early trajectory free of lattice noise.
    if ctx.param('shuffleIters'):
        system.state.positions = shuffleParticles(
            system.state, ctx.config, ctx.schemeConfig, ctx.param('shuffleIters'),
            jitterAmount=ctx.param('jitter'))
    return system


def phase(ctx: RunContext) -> float:
    """The quarter-period offset stamped onto the velocity field.

    An even wavenumber puts the vortex centres on the domain boundary; the
    shift moves them back into the interior. It also decides what the *centre*
    of the box is -- a vortex core (pressure minimum) at `phase = 0`, a
    stagnation point (pressure maximum) at `phase = pi/2` -- which is why
    :func:`analyticCentrePressure` has to read it too rather than assume, and
    why the `phase` param exists at all: Sun et al. 2019 Sec. 3.1 reads its
    pressure signal at the centre of the box, and their Fig. 8 pressure profile
    along `y = 0.5 L` shows that point is a *maximum*, i.e. their layout is the
    `pi/2` one. A non-integer `k` (Sun's `k = 2 pi / L` is `k_param = 4 pi`)
    would otherwise fall through the even-`k` rule to `phase = 0` and put a
    vortex core under the probe -- the same flow, but the mean-pressure drift
    the benchmark measures then reads with the opposite sign.
    """
    explicit = ctx.param('phase', None)
    if explicit is not None:
        return float(explicit)
    return np.pi / 2 if ctx.param('k') % 2 == 0 else 0.0


def initialConditions(ctx: RunContext, system) -> None:
    k = wavenumber(ctx)
    uMag = ctx.param('uMag')
    ph = phase(ctx)

    positions = system.state.positions
    system.state.velocities[:, 0] = (
        uMag * torch.cos(k * positions[:, 0] + ph) * torch.sin(k * positions[:, 1] + ph))
    system.state.velocities[:, 1] = (
        -uMag * torch.sin(k * positions[:, 0] + ph) * torch.cos(k * positions[:, 1] + ph))

    setupTimestep(ctx, system)

    # The sound speed only exists after `setupTimestep`, which is why the
    # density has to be stamped after the velocity rather than with it.
    #
    # `initialPressure`: start the run *on* the analytic solution rather than
    # from a uniform `rho0`. A weakly compressible scheme carries its pressure
    # in the density, `p = c0^2 (rho - rho0)`, so a uniform-density start is a
    # start with the whole TGV pressure field missing -- the flow has to radiate
    # that field into existence, and the resulting acoustic transient is of the
    # same order as the signal (measured at L/dx = 50, c0 = 10 U: the centre
    # pressure rings over `p/P0` in [-2, -0.5] about its correct mean of -1).
    # Sun et al. 2019 Sec. 3.1 compares against the analytic pressure from
    # `t = 0`, so its benchmark needs this on. Default off, so no existing run
    # of this case changes.
    if ctx.param('initialPressure', False):
        c0 = float(ctx.schemeConfig.fluid.fixedSoundSpeed)
        rho0 = float(ctx.param('rho0'))
        # p = -(rho0 uMag^2 / 4) (cos 2(kx + ph) + cos 2(ky + ph)); the sign is
        # the one that makes the vortex cores low pressure and the stagnation
        # points high, i.e. the integral of `-(u . grad) u`.
        p = -0.25 * rho0 * uMag ** 2 * (
            torch.cos(2.0 * (k * positions[:, 0] + ph))
            + torch.cos(2.0 * (k * positions[:, 1] + ph)))
        system.state.densities = rho0 + p / (c0 ** 2)


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def analyticPressureAmplitude(ctx: RunContext) -> float:
    """`P0`, the peak pressure of the continuum vortex at `t = 0`.

    The TGV pressure field is `p = -(rho0 uMag^2 / 4)(cos 2kx + cos 2ky)`, so
    its extremes are `+/- rho0 uMag^2 / 2` and Sun et al. 2019 Sec. 3.1's
    normalising `P0` -- "the maximum pressure value in the flow domain at the
    initial time" -- is `rho0 uMag^2 / 2`.
    """
    return 0.5 * ctx.param('rho0') * ctx.param('uMag') ** 2


def analyticCentrePressure(ctx: RunContext) -> float:
    """The signed continuum pressure at the box centre at `t = 0`.

    `-P0` when the centre is a vortex core (`phase = 0`, which is Sun's own
    `[0, L]^2` layout) and `+P0` when the quarter-period shift has put a
    stagnation point there instead (`phase = pi/2`, this case's default `k=2`
    layout). Both are `+/- P0` exactly, but the sign is not a detail: it is what
    a measured `pCentre` has to be compared against, and Eq. (22) decays the
    signed value, not its magnitude.
    """
    centre = 0.5 * ctx.spec.L
    # p = -(rho0 u^2 / 4)(cos 2(kx+ph) + cos 2(ky+ph)); at x = y = L/2 the two
    # cosines are the same one, hence the factor 2.
    argument = 2.0 * (wavenumber(ctx) * centre + phase(ctx))
    return float(-0.5 * ctx.param('rho0') * ctx.param('uMag') ** 2 * np.cos(argument))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    d = weaklyCompressibleDiagnostics(ctx, state)
    particles = state.state
    fluid = particles.kinds == 0 if hasattr(particles, 'kinds') else slice(None)

    # Sun et al. 2019 Eq. (23): the percentage drift of the total particle
    # volume `sum_j m_j / rho_j` away from the box it has to tile. In a
    # periodic box with no source or sink of fluid this is exactly zero for the
    # continuum, so every count of it is discretisation error -- and Sun's
    # Fig. 9 is the diagnosis behind Fig. 7: a scheme whose volumes drift is a
    # scheme whose mean density, hence mean pressure `c0^2 (rho - rho0)`, has
    # drifted with them. It is the one metric that separates the delta+-SPH
    # PST's cumulative volume error from delta-SPH's one-off initial
    # rearrangement.
    volume = (particles.masses[fluid] / particles.densities[fluid]).sum()
    boxVolume = float(ctx.spec.L) ** int(ctx.spec.dim)
    d['totalVolume'] = float(volume.detach().cpu().item())
    d['volumeError'] = abs(d['totalVolume'] / boxVolume - 1.0) * 100.0

    # Sun's Figs. 5 and 7: the pressure at the centre of the domain, which
    # decays by the same Eq. (22) as the kinetic energy. Read by a first-order
    # MLS (Liu-Liu) fit rather than by picking the nearest particle -- the
    # particles move, so a nearest-particle read would sample a different point
    # of a sinusoid at every step and alias the signal it is meant to measure.
    # The box is periodic and the query sits deep in the bulk, so the fit is
    # always well conditioned here; the Shepard tier is kept only so the
    # diagnostic can never emit a hard zero (see `cases/dambreak.py`, whose
    # wall probe is one-sided and does need it).
    if getattr(particles, 'pressures', None) is not None:
        centre = torch.full((1, ctx.spec.dim), 0.5 * float(ctx.spec.L),
                            device=particles.positions.device,
                            dtype=particles.positions.dtype)
        val, _grad, nnbr, A_g, b, wellConditioned = interpolateLiuLiu(
            centre, referenceParticles=particles,
            referenceQuantities=particles.pressures, config=ctx.config,
            neighbor_threshold=4, direction=OperationDirection.FluidToFluid)
        shepDen = A_g[:, 0, 0]
        shepVal = torch.where(shepDen > 0, b[:, 0] / shepDen.clamp_min(1e-12),
                              torch.zeros_like(shepDen))
        val = torch.where(wellConditioned, val, shepVal)
        d['pCentre'] = float(val[0].detach().cpu().item())
        d['pCentreNnbr'] = int(nnbr[0].detach().cpu().item())
        p0 = analyticPressureAmplitude(ctx)
        d['pCentreStar'] = d['pCentre'] / p0 if p0 else float('nan')

    # The bulk pressure level the centre signal is measured against. Sun's
    # Fig. 8 point is that delta-SPH and the 2017 delta+-SPH keep the sinusoid's
    # *shape* while its mean level walks off; separating the two needs the mean.
    d['pMean'] = float(particles.pressures[fluid].mean().detach().cpu().item()) \
        if getattr(particles, 'pressures', None) is not None else float('nan')
    return d


tgvWeaklyCompressibleCase = registerCase(Case(
    name='tgv-wc',
    scheme='deltaSPH',
    description='Taylor-Green vortex (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='05-taylorGreenVortex',
        nx=256,
        L=2 * np.pi,
        tLimit=2.0,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        targetDt=0.001,
        inviscid=False,
        nu=0.01,
        k=2,
        # None -> the even-`k` rule in `phase()`; set explicitly to place a
        # stagnation point (pi/2) or a vortex core (0) at the box centre.
        phase=None,
        uMag=1.0,
        shuffleIters=128,
        jitter=1.0,
        # None -> whatever the selected scheme defaults to (on, for deltaSPH);
        # False runs the plain delta-SPH leg of the PST A/B. See
        # `configureScheme`.
        shifting=None,
        # Start on the analytic pressure field instead of a uniform rho0 --
        # see `initialConditions`. Off by default (unchanged behaviour); the
        # Sun 2019 Sec. 3.1 benchmark needs it on.
        initialPressure=False,
        # Sun et al. 2017 Eq. (2) sound speed, via the shared `setupTimestep`:
        # `machTarget=None` (the default) keeps the legacy `targetDt`
        # back-solve, so nothing changes unless a run asks for it. The Sun 2019
        # Sec. 3.1 benchmark wants `machTarget=0.1` with
        # `referenceVelocity=uMag`, i.e. `c0 = 10 U`.
        machTarget=None,
        referenceVelocity=None,
    ),
))


if __name__ == '__main__':
    caseMain(tgvWeaklyCompressibleCase)
