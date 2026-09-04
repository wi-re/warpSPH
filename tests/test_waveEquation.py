"""Standalone checks for the wave-equation scheme that do not go through the
`Case`/CLI machinery: a spatial-convergence check against the closed-form
standing-wave solution, and a check that gradients reach a smooth source's
`position`/`magnitude` tensors. See `docs/historic_plans/WAVE_EQUATION_PLAN.md` steps 5 and 6.
"""

from __future__ import annotations

import math

import pytest
import torch

from warpSPH.configurations import WaveCaseConfig, WaveEquationConfig, buildConfig
from warpSPH.sample.bySamplingScheme import sampleParticles
from warpSPH.sample.waveSystem import sampleSmoothPointSourceWaveSystem
from warpSPH.schemes.waveEquation import f_wave_equation
from warpSPH.systems.waveSystem import WaveSystemStatev3, WaveSystemv3, computeDt
from warpSPH.utils import buildDomainDescription
from warpSPHCore import SupportScheme, buildVerletList

#: The scheme's own Laplacian is stiffer than `computeDt`'s acoustic CFL
#: number accounts for (see `cases/waveEquation.py`'s default override for
#: the same reason): 0.3, the `SimulationConfig`/`CaseSpec` default, is
#: unstable by nx=256 in 1D even with zero damping here; 0.1 stays stable and
#: convergent at every resolution these tests use.
_CFL_FACTOR = 0.1


def _buildStandingWaveSystem(nx: int, dim: int, c: float = 1.0, L: float = 2.0):
    """A `WaveSystemv3` with `u(x,0) = sin(k.x)`, `v(x,0) = 0`, constant `c`,
    zero damping, on a periodic domain -- `k = pi` puts exactly one
    wavelength across `L = 2`, so it wraps cleanly. Closed form:
    `u(x,t) = sin(k.x) cos(c k t)`.
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                     dtype=dtype, dx=L / nx, cflFactor=_CFL_FACTOR)

    k = math.pi
    particles = sampleParticles(nx, config)
    x = particles.positions[:, 0]
    u0 = torch.sin(k * x)
    v0 = torch.zeros_like(u0)

    n = x.shape[0]
    state = WaveSystemStatev3(
        positions=particles.positions, supports=particles.supports,
        masses=particles.masses, densities=particles.densities,
        kinds=torch.zeros(n, device=device, dtype=torch.int32),
        materials=torch.zeros(n, device=device, dtype=torch.int32),
        UIDs=torch.arange(n, device=device, dtype=torch.int32),
        UIDcounter=n,
        u=u0, v=v0, c=torch.full_like(u0, c), damping=torch.zeros_like(u0),
    )
    adjacency = buildVerletList(state, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
    system = WaveSystemv3(state=state, adjacency=adjacency, domain=config.domain,
                          t=torch.tensor(0.0, device=device, dtype=dtype))
    return system, config, integrator, k


def _standingWaveError(nx: int, dim: int, c: float = 1.0, L: float = 2.0) -> float:
    """Run to a quarter period and return the RMS error against the analytic
    standing wave. A quarter period (rather than a full one) is deliberate:
    at a full period the numeric and analytic solutions both round-trip back
    to the initial condition, which would hide phase/amplitude error instead
    of measuring it.
    """
    system, config, integrator, k = _buildStandingWaveSystem(nx, dim, c, L)
    caseConfig = WaveCaseConfig(defaultSpeed=c)
    dt = computeDt(system, config, caseConfig, None, [], verbose=False)

    tEnd = (2 * math.pi / (c * k)) * 0.25
    nSteps = max(1, round(tEnd / dt))
    dtActual = tEnd / nSteps

    schemeConfig = WaveEquationConfig()
    for _ in range(nSteps):
        stepResult = integrator.function(state=system, f=f_wave_equation, dt=dtActual,
                                         config=config, verbose=False, schemeConfig=schemeConfig)
        system = stepResult.state

    x = system.state.positions[:, 0]
    analytic = torch.sin(k * x) * math.cos(c * k * tEnd)
    return torch.sqrt(torch.mean((system.state.u - analytic) ** 2)).item()


@pytest.mark.parametrize('dim,resolutions', [
    (1, (32, 64, 128)),
    (2, (16, 32, 64)),
    (3, (8, 16, 24)),
])
def test_standingWaveErrorShrinksWithResolution(dim, resolutions):
    """The base pipeline's accuracy check: a smooth, resolved standing wave
    converges as the particle spacing refines, in every dimension the N-D
    generalization (`docs/historic_plans/WAVE_EQUATION_PLAN.md` step 1) now supports."""
    errors = [_standingWaveError(nx, dim) for nx in resolutions]
    assert all(after < before for before, after in zip(errors, errors[1:])), (
        f'dim={dim}: errors {errors} did not shrink monotonically with resolution')
    assert errors[-1] < 0.5 * errors[0], (
        f'dim={dim}: error only went from {errors[0]:.3e} to {errors[-1]:.3e} '
        f'over resolutions {resolutions}')


# --- Gradients reach a smooth source's position and magnitude ---------------

def test_gradientsReachSourcePositionAndMagnitude():
    """`docs/historic_plans/WAVE_EQUATION_PLAN.md` step 5's verification: `position`/`magnitude`
    are real leaf tensors, and a scalar probe of `u` after a short rollout
    has a finite, non-zero gradient w.r.t. both -- through the smooth
    kernel-weighted bump `sampleSmoothPointSourceWaveSystem` stamps onto `u`,
    not the (non-differentiable) SDF id-grid path `shape_generation.py` uses
    for the 2D case's own sources.
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dim, nx, L = 2, 32, 2.0
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, _integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                      dtype=dtype, dx=L / nx, cflFactor=_CFL_FACTOR)
    caseConfig = WaveCaseConfig(defaultSpeed=1.0, domainDamping=True)

    # Off-centre so the probe's sensitivity to `position` isn't cancelled by
    # domain symmetry.
    position = torch.tensor([0.3, -0.2], device=device, dtype=dtype, requires_grad=True)
    magnitude = torch.tensor(5.0, device=device, dtype=dtype, requires_grad=True)

    system = sampleSmoothPointSourceWaveSystem(nx, config, caseConfig, position=position,
                                                magnitude=magnitude, radius=0.2)
    schemeConfig = WaveEquationConfig()

    dt = 0.005
    for _ in range(5):
        update, adjacency = f_wave_equation(system, dt, config, schemeConfig)
        system.adjacency = adjacency
        system.state.u = system.state.u + dt * update.dudt
        system.state.v = system.state.v + dt * update.dvdt

    probe = (system.state.u ** 2).sum()
    probe.backward()

    assert position.grad is not None
    assert torch.isfinite(position.grad).all()
    assert (position.grad != 0).any(), 'gradient w.r.t. source position is exactly zero'

    assert magnitude.grad is not None
    assert torch.isfinite(magnitude.grad).all()
    assert magnitude.grad != 0, 'gradient w.r.t. source magnitude is exactly zero'
