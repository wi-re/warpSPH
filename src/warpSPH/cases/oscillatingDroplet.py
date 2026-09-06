"""Oscillating droplet under a central potential (2D), weakly compressible.

The script form of this case was
`examples/weaklyCompressible/04-oscillating-droplet.ipynb`. A circular droplet
is given a straining velocity field and held together by a radial potential;
the exact solution oscillates between two ellipses with period T = 4.827 A, and
that period is what this run is measured against.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..enumTypes import isArtificialCompressibleScheme
from ..runner import Case, RunContext, caseMain, registerCase
from .plotting import particlePlot
from .weaklyCompressible import (VELOCITY_DENSITY_FIELDS, WEAKLY_COMPRESSIBLE_DEFAULTS,
                                 WEAKLY_COMPRESSIBLE_PARAMS, buildRegionSystem,
                                 configureArtificialCompressible,
                                 configureWeaklyCompressible, fluidRegion,
                                 paramExtraData, setupTimestep, shapeSdf,
                                 weaklyCompressibleDiagnostics)

__all__ = ['oscillatingDropletCase', 'DROPLET_STRETCH', 'DROPLET_PERIOD',
           'analyticEnvelope', 'analyticSolution']

#: Long semi-axis of the droplet at maximum elongation, in units of `R`.
#: The short one is `R / DROPLET_STRETCH`: the flow is incompressible, so the
#: ellipse has the same area as the circle it started as.
DROPLET_STRETCH = 1.931843

#: Oscillation period, in units of the strain time `1 / A`.
DROPLET_PERIOD = 4.827


def analyticEnvelope(R: float = 1.0, stretch: float = DROPLET_STRETCH):
    """`(long, short)` semi-axes of the two extreme ellipses.

    The droplet passes through the circle it started as and through two
    ellipses that are each other's 90-degree rotation -- long axis horizontal
    at one extreme, vertical at the other. `04-oscillating-droplet.ipynb` draws
    all three over the particles as the check that the amplitude, not just the
    period, came out right.
    """
    return stretch * R, R / stretch


def analyticSolution(t, A: float = 1.0, B: float = 1.0, R: float = 1.0, rho0: float = 1.0):
    """Monaghan & Rafiee (2013)'s exact inviscid/incompressible solution
    (`literature/`, filed under its own title "A simple SPH algorithm for
    multi-fluid flow with high density ratios", IJNMF 71(5) 537-561, DOI
    10.1002/fld.3671), specialised to a single fluid (their two-region result
    with the outer density set to zero) -- what `ACSPH_PLAN.md` Part 8 step 6
    needs Table 1/2's `IRMSE(KE)`/`IRMSE(a)` against, and what `DROPLET_STRETCH`/
    `DROPLET_PERIOD` above only sample at two isolated instants.

    Appendix A derives the elliptical-drop motion (semi-axes `a` (x), `b` (y),
    strain rate `sigma`) under a quadratic potential of strength `B`
    (`computePotentialFieldGravity`'s acceleration is `-magnitude**2 * r`, so
    `Omega**2 = B**2` in the paper's notation) from Eqs. (A.15)-(A.24):

        dsigma/dt = (sigma**2 + Omega**2) * (b**2 - a**2) / (a**2 + b**2)
        da/dt = sigma * a
        db/dt = -sigma * b

    with `sigma(0) = A` (the initial strain rate, `vx = A*x`), `a(0) = b(0) =
    R`. (The OCR'd PDF text is easy to misread as `dsigma/dt = -Omega**2 +
    2*(b**2-a**2)/(a**2+b**2)` -- the "2" is `Omega`'s superscript, not a
    separate coefficient; that misreading fails the check below.) Verified
    against the two already-encoded constants at the `A=B=R=1` default: this
    ODE's first peak of `a(t)` lands at 1.931852 at `t=1.207`, next at
    `t=6.034` -- against `DROPLET_STRETCH=1.931843` and
    `DROPLET_PERIOD=4.827` (`6.034-1.207`), matching to the integrator's own
    tolerance.

    Kinetic energy follows from Appendix A Eq. (A.28)/(A.29) specialised to
    one fluid (`rho_o=0`): `KE(t) = (pi/8) * rho0 * sigma(t)**2 * a(t)*b(t) *
    (a(t)**2+b(t)**2)`, using `integral-over-the-ellipse(x**2+y**2) =
    (pi/4)*a*b*(a**2+b**2)` for a uniform-density ellipse.

    `t` may be a scalar or array-like; returns `(a, b, kineticEnergy)`, each
    shaped like `t`.
    """
    from scipy.integrate import solve_ivp

    omega2 = float(B) ** 2
    tArr = np.atleast_1d(np.asarray(t, dtype=float))
    tMax = float(tArr.max()) if tArr.size else 0.0

    def rhs(_, y):
        a, b, sigma = y
        dsigma = (sigma * sigma + omega2) * (b * b - a * a) / (a * a + b * b)
        return [sigma * a, -sigma * b, dsigma]

    evalPoints = np.sort(np.unique(np.concatenate([[0.0], tArr, [tMax]])))
    sol = solve_ivp(rhs, [0.0, tMax], [float(R), float(R), float(A)],
                    t_eval=evalPoints, rtol=1e-10, atol=1e-12, max_step=1e-3,
                    dense_output=True)
    a, b, sigma = sol.sol(tArr)
    ke = (np.pi / 8.0) * rho0 * sigma ** 2 * a * b * (a ** 2 + b ** 2)

    scalarInput = np.ndim(t) == 0
    if scalarInput:
        return float(a[0]), float(b[0]), float(ke[0])
    return a, b, ke


def _measuredSemiAxes(particles):
    """`(a, b)`: the x/y semi-axes a uniform-density ellipse would need to
    match the fluid's own mass-weighted second moments about its centroid --
    `Var(x) = a**2/4` for a uniform ellipse aligned with the axes, which this
    droplet always is (the initial `vx=A*x, vy=-A*y` field has no rotational
    component, so the principal axes never leave x/y). What `analyticSolution`'s
    `a(t)`/`b(t)` are measured against.
    """
    fluid = particles.kinds == 0
    pos = particles.positions[fluid]
    mass = particles.masses[fluid]
    totalMass = mass.sum()
    centroid = (pos * mass[:, None]).sum(dim=0) / totalMass
    rel = pos - centroid
    varX = (mass * rel[:, 0] ** 2).sum() / totalMass
    varY = (mass * rel[:, 1] ** 2).sum() / totalMass
    return 2.0 * torch.sqrt(varX), 2.0 * torch.sqrt(varY)


def configureScheme(ctx: RunContext) -> None:
    # The restoring force is a potential field centred on the droplet, not a
    # directional gravity, so it is configured before the shared block runs.
    ctx.spec.params.setdefault('gravity', True)
    ctx.spec.params.setdefault('gravityType', 'PotentialField')
    ctx.spec.params.setdefault('gravityMagnitude', ctx.param('B'))
    ctx.spec.params.setdefault('gravityDirection', [0.0, 0.0])
    if isArtificialCompressibleScheme(ctx.scheme):
        return _configureArtificialCompressible(ctx)
    configureWeaklyCompressible(ctx)


def _configureArtificialCompressible(ctx: RunContext) -> None:
    """The ACSPH branch of `configureScheme` (`ACSPH_PLAN.md` §4.3, the
    paper's own headline validation case: `IRMSE(KE)`/`IRMSE(semi-major
    axis)` against the analytic period, and the vehicle for reproducing
    Tables 1-2, the CFL_t/`Δt/Δτ`/RK-stage-count sweep -- the real acceptance
    gate for the dual-time machinery per Part 8 step 6).

    `configureArtificialCompressible` already understands `PotentialField`
    gravity (`resolveEnum(GravityType, ...)`), so the shared block above
    (setting `gravity`/`gravityType`/`gravityMagnitude`/`gravityDirection`
    before dispatch) needs nothing droplet-specific changed here.
    """
    configureArtificialCompressible(ctx)

    schemeConfig = ctx.schemeConfig
    schemeConfig.shiftProperties.active = False
    # Eq. (48)'s U_char (ACSPH_PLAN.md §5.5, never defined per case by the
    # paper): the only velocity scale in the problem is the initial straining
    # field's own magnitude at the droplet's edge, A*R.
    if schemeConfig.acParams.uChar is None:
        schemeConfig.acParams.uChar = float(ctx.param('A') * ctx.param('R'))


def buildSystem(ctx: RunContext):
    return buildRegionSystem(
        ctx, [fluidRegion(ctx, shapeSdf('circle', ctx.param('R')))])


def initialConditions(ctx: RunContext, system) -> None:
    strain = ctx.param('A')
    positions = system.state.positions
    system.state.velocities[:, 0] = strain * positions[:, 0]
    system.state.velocities[:, 1] = -strain * positions[:, 1]
    if isArtificialCompressibleScheme(ctx.scheme):
        # No sound speed to back-solve a `dt` from (`setupTimestep` is WCSPH-
        # only); seed the same `targetDt` WCSPH would have tuned to, and let
        # `dropletTimestep`'s Eq. (46) hook take over from step 2 on.
        ctx.config.dt = ctx.param('targetDt')
    else:
        setupTimestep(ctx, system)


def dropletTimestep(ctx: RunContext, state) -> float:
    """Eq. (46) for ACSPH; the fixed acoustic-CFL `dt` otherwise (unchanged
    WCSPH behaviour -- this case has never had a per-step adaptive `dt`)."""
    if isArtificialCompressibleScheme(ctx.scheme):
        from ..modules.timestep import computeTimestep
        return computeTimestep(state, ctx.config, ctx.schemeConfig, dt=ctx.config.dt)
    return ctx.config.dt


setupPlot, updatePlot = particlePlot(VELOCITY_DENSITY_FIELDS)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    out = weaklyCompressibleDiagnostics(ctx, state)
    a, b = _measuredSemiAxes(state.state)
    out['semiAxisA'] = a.detach().cpu().item()
    out['semiAxisB'] = b.detach().cpu().item()
    # Eq. (65)'s `s_RK`/`m_iter`: pseudo-iterations this step took, for ACSPH
    # (`ctx.scratch['lastStageUpdate']`, `runner.py`'s per-step stash of the
    # scheme's own update object -- `diagnostics` never otherwise sees it).
    # `1` for every other scheme (one function evaluation per RK stage; the
    # RK stage count itself is a run-level constant, not a per-step one, so
    # it belongs to the sweep script that consumes this column, not here).
    update = ctx.scratch.get('lastStageUpdate')
    out['pseudoIterations'] = float(getattr(update, 'pseudoIterations', 1))
    return out


oscillatingDropletCase = registerCase(Case(
    name='droplet',
    scheme='deltaSPH',
    description='Oscillating droplet in a central potential (2D), weakly compressible deltaSPH.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    timestep=dropletTimestep,
    defaults=dict(
        WEAKLY_COMPRESSIBLE_DEFAULTS,
        caseName='04-oscillatingDroplet',
        nx=192,
        L=6.0,
        # One full oscillation period, DROPLET_PERIOD / A with A = 1.
        tLimit=DROPLET_PERIOD,
        plotInterval=10,
    ),
    params=dict(
        WEAKLY_COMPRESSIBLE_PARAMS,
        freeSurface=True,
        targetDt=0.00025,
        R=1.0,
        A=1.0,
        B=1.0,
        markerSize=8,
    ),
))


if __name__ == '__main__':
    caseMain(oscillatingDropletCase)
