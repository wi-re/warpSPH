"""Shear-wave decay (2D, fully periodic), divergence-free incompressible SPH.

Cornelis et al.'s reference case, ported per `DFSPH_IMPROVEMENT_PLAN.md` §4
item 8. It exists because every other incompressible case here grades this
codebase against *itself*: `tgv`'s decay rate has no published counterpart
(§5 Q5), `kolmogorovIncompressible` has no analytic solution at all, and
`randomFlowIncompressible` is a statistical steady state. This one has an exact
answer that is known in closed form and is trivial to state.

**The physics.** A transverse sinusoidal shear wave on a periodic box:

    u_x(y, 0) = u0 sin(k_w y),   u_y = 0,   k_w = 2 pi k / L

Both nonlinear terms vanish identically. `(u . grad) u = u_x d_x u_x e_x = 0`
because `u_x` depends only on `y`, and `div u = d_x u_x = 0` for the same
reason. So the incompressible Navier-Stokes solution is exactly

    u_x(y, t) = u0 sin(k_w y) exp(-nu k_w^2 t),   p = const

**for all time, at any amplitude** -- and, unlike the Taylor-Green vortex,
**with zero pressure gradient**. TGV is also an exact solution, but it is one
in which a nontrivial pressure field balances a nontrivial advection term, so
an error in the pressure solve and an error in the advection are measured
together. Here the exact pressure is constant, so *every* pressure the solver
produces is an artifact, and every departure of the amplitude from the analytic
exponential is numerical dissipation. At the default `nu = 0` the analytic
answer is that nothing happens at all.

**What it grades, and why those are two separate axes.** [C]'s Fig. 3 and
Fig. 4 report the sinus amplitude and the maximum density on this case
separately, and the separation is the point:

- `amplitudeRatio` -- the velocity field projected onto the analytic mode,
  divided by the analytic amplitude at that time. This is **artificial
  viscosity**: 1.0 is exact, below 1.0 is dissipation. It is the axis [C]'s
  abstract is about ("the DI source term suffers from significant artificial
  viscosity").
- `maxDensity` / `densityStd` -- **volume error and disorder**. A scheme can
  hold the amplitude perfectly while the sampling degrades, or keep a pristine
  sampling by damping the flow. One number cannot see both.

Those are exactly the axes the `ShiftApplication` modes trade off against each
other (§1.2), which is why this case is the one that can grade them: the
position shift is momentum-neutral and should cost amplitude nothing, while the
velocity modes feed a permanent residual into momentum and should show up here
as dissipation.

**Not yet done: the comparison against [C]'s published curves.** The paper's
figures are not in this repository and this case does not hard-code numbers
read off them. What is here is the case and this codebase's measurement on it;
grading that against Fig. 3 and Fig. 4 needs the paper in hand.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from warpSPHCore import sphKernelScale

from ..caseUtils.incompressible import relaxLattice
from ..modules import computeDensities
from ..runner import Case, RunContext, caseMain, registerCase
from ..sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
from .weaklyCompressible import particleDistributionMetrics
from .plotting import Field, particlePlot

__all__ = ['shearWaveCase', 'analyticAmplitude']


def waveNumber(ctx: RunContext) -> float:
    """`k_w = 2 pi k / L`, the wavenumber of the imposed shear mode."""
    return 2.0 * np.pi * ctx.param('k') / ctx.spec.L


def analyticAmplitude(ctx: RunContext, t: float) -> float:
    """`u0 exp(-nu k_w^2 t)` -- the exact amplitude of the shear mode.

    Exact for all `t`, not asymptotically: the nonlinear term vanishes
    identically for this field, so the momentum equation reduces to the heat
    equation on `u_x`.
    """
    kw = waveNumber(ctx)
    return float(ctx.param('uMag') * np.exp(-ctx.param('nu') * kw ** 2 * t))


def configureScheme(ctx: RunContext) -> None:
    ctx.schemeConfig.surfaceDetectionConfig.active = False
    ctx.schemeConfig.diffusionParams.inviscid = False
    ctx.schemeConfig.diffusionParams.viscidNu = ctx.param('nu')
    ctx.schemeConfig.shiftProperties.active = ctx.param('shifting')


def buildSystem(ctx: RunContext):
    system = setupBasicWeaklyCompressibleInitialState(
        ctx.spec.nx, ctx.config, ctx.schemeConfig, ctx.SimulationState, ctx.SimulationSystem)

    # Normalise mass so the sampled density lands on rho0, matching `tgv` and
    # `kolmogorovIncompressible` for this same scheme. It matters more here
    # than there: `maxDensity` is one of the two things this case reports, and
    # an uncalibrated mass would put a constant offset straight into it.
    rho0 = ctx.param('rho0')
    densities = computeDensities(system.state, ctx.config, ctx.schemeConfig, None)
    system.state.masses = system.state.masses / densities.mean() * rho0
    return system


def initialConditions(ctx: RunContext, system) -> None:
    relaxLattice(ctx, system, ctx.param('relaxSteps'), ctx.param('relaxDt'),
                 ctx.param('jitter'))

    kw = waveNumber(ctx)
    positions = system.state.positions
    system.state.velocities[:] = 0.0
    system.state.velocities[:, 0] = ctx.param('uMag') * torch.sin(kw * positions[:, 1])


def shearWaveTimestep(ctx: RunContext, state) -> float:
    """The same advective + viscous CFL `kolmogorovIncompressible` uses.

    Shared verbatim rather than imported so that the two cases can be tuned
    apart later; the advective term is [BK]'s published `dt <= 0.4 d / |v_max|`
    in their units (`d` = the particle diameter `dx`), so `cflFactor` is
    directly comparable with the 0.4 they state. See `DFSPH_IMPROVEMENT_PLAN.md`
    Part 12.

    Unlike that case this flow has a fixed velocity scale -- `|u|` starts at
    `uMag` and can only decay -- so the adaptive `dt` here is nearly constant
    and the case is a clean cost comparison as well as a clean accuracy one.
    """
    particles = state.state
    h = particles.supports.mean().item()
    vMax = particles.velocities.norm(dim=1).max().item()
    nu = ctx.param('nu')
    kernelScale = float(sphKernelScale(ctx.config.kernel.value, ctx.config.dim))
    dt_adv = ctx.config.cflFactor * ctx.config.dx / max(vMax, 1e-3)
    dt_visc = 0.125 * h ** 2 / kernelScale / nu if nu > 0 else float('inf')
    return float(min(max(min(dt_adv, dt_visc), ctx.config.minDt), ctx.config.maxDt))


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    particles = state.state
    fluid = particles.kinds == 0 if hasattr(particles, 'kinds') else slice(None)
    positions = particles.positions[fluid]
    velocities = particles.velocities[fluid]
    densities = particles.densities[fluid]
    masses = particles.masses[fluid]

    kw = waveNumber(ctx)
    basis = torch.sin(kw * positions[:, 1])
    # Mass-weighted projection onto the analytic mode. The 2x is the
    # normalisation of `sin` over a period (`<sin, sin> = 1/2`), so for a field
    # that is exactly `A sin(k_w y)` on a uniform sampling this returns `A`.
    amplitude = (2.0 * (masses * velocities[:, 0] * basis).sum() / masses.sum())
    amplitude = amplitude.detach().cpu().item()

    t = getattr(state, 't', 0.0)
    t = t.detach().cpu().item() if isinstance(t, torch.Tensor) else float(t)
    analytic = analyticAmplitude(ctx, t)

    # Everything the shear mode does not account for: the transverse component,
    # which is exactly zero in the exact solution, and whatever is left of
    # `u_x` once its own sinusoidal content is removed. This is the *disorder*,
    # and it is deliberately measured against the **measured** amplitude rather
    # than the analytic one. Subtracting the analytic mode instead would fold
    # the amplitude error back in, and at `nu = 0.01` that dominates it
    # completely -- the two modes then differ by 50% of `u0`, which swamps a
    # disorder signal two orders of magnitude smaller and makes the column read
    # the same thing `amplitudeRatio` already reports. The two axes are only
    # independent if this one subtracts what was actually there.
    residual = velocities.clone()
    residual[:, 0] = residual[:, 0] - amplitude * basis
    residualRms = torch.sqrt((residual ** 2).sum(dim=-1).mean()).detach().cpu().item()

    out = {
        'amplitude': amplitude,
        'amplitudeRatio': amplitude / analytic if analytic else float('nan'),
        'residualVelocity': residualRms / ctx.param('uMag'),
        'transverseVelocity': velocities[:, 1].abs().max().detach().cpu().item() / ctx.param('uMag'),
        'kineticEnergy': (0.5 * masses * (velocities ** 2).sum(dim=-1)).sum().detach().cpu().item(),
        'minDensity': densities.min().detach().cpu().item(),
        'maxDensity': densities.max().detach().cpu().item(),
        'densityStd': densities.std().detach().cpu().item(),
    }
    out.update(particleDistributionMetrics(ctx, state))
    return out


setupPlot, updatePlot = particlePlot([
    Field('velocities', 'velocities', colorMap='viridis', mapping='L2Norm'),
    Field('densities', 'densities', colorMap='RdBu', colorMapKind='diverging',
          flip=True, midPoint=1.0),
])


def extraData(ctx: RunContext, state) -> Dict[str, Any]:
    return {k: ctx.param(k) for k in shearWaveCase.params}


shearWaveCase = registerCase(Case(
    name='shearWave',
    scheme='divergenceFree',
    description='Shear-wave decay (2D, periodic): an exact solution with zero pressure gradient.',
    buildSystem=buildSystem,
    configureScheme=configureScheme,
    initialConditions=initialConditions,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=extraData,
    timestep=shearWaveTimestep,
    defaults=dict(
        caseName='04-shearWave',
        dim=2,
        nx=128,
        L=1.0,
        n_h=4.0,
        periodic=True,
        kernel='Wendland2',
        integrationScheme='semiImplicitEuler',
        supportMode='SuperSymmetric',
        tLimit=2.0,
        dt=1e-3,
        adaptiveDt=True,
        # `shearWaveTimestep` applies this to the particle diameter, so it is
        # Bender & Koschier's published constant (Part 12), as on the other
        # two adaptive incompressible cases.
        cflFactor=0.4,
        minDt=1e-8,
        maxDt=1e-2,
    ),
    params=dict(
        rho0=1.0,
        # Zero by default: with no physical viscosity the analytic answer is
        # that the amplitude never changes, so `amplitudeRatio` reads as pure
        # numerical dissipation with nothing to disentangle it from. Set `nu`
        # to grade against a nontrivial published decay instead.
        nu=0.0,
        k=1,
        uMag=1.0,
        shifting=False,
        relaxSteps=32,
        relaxDt=1e-3,
        jitter=0.01,
    ),
))


if __name__ == '__main__':
    caseMain(shearWaveCase)
