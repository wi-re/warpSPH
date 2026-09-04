"""Phase 1 of `warpier_forward_mode_plan.md`: a working, tested forward-mode
sensitivity `d(u(x,T))/d(source position, magnitude)` on the wave-equation
testbed, cross-checked against the existing reverse-mode result -- no new
`warpSPHCore` math, since Tier 1 (a value-tangent JVP) covers it entirely.

The trick this exercises: because `f_wave_equation` is exactly linear in the
integrated pair `(u, v)` for fixed `c`/`damping`/positions/adjacency (the
Laplacian operator and the `c**2 * (.) - damping * v` combination are both
linear in their `(u, v)` arguments), relaunching the *same* step sequence on
a tangent state `(du, dv)` in place of `(u, v)` computes the JVP exactly --
no new kernel, no forward-mode AD through `warpOperation`. This is the
special case `warpier_forward_mode_plan.md` calls "Tier 1"; it only holds
because positions/support/adjacency are frozen here (this scheme never moves
particles) and CRK/renorm corrections are off (`f_wave_equation` uses none).

Both the forward-tangent rollout and the reverse-mode reference below use the
*same* explicit-Euler step sequence (`docs/historic_plans/WAVE_EQUATION_PLAN.md` step 5's
already-validated pattern, reused verbatim from
`test_gradientsReachSourcePositionAndMagnitude` in `test_waveEquation.py`)
rather than the `rungeKutta2` integrator `_buildStandingWaveSystem` uses
elsewhere in this file's sibling tests. That is deliberate: the JVP identity
being checked (`d(probe)/d(direction)` via forward-mode tangent propagation
equals the same quantity via a reverse-mode gradient dotted with the
perturbation direction) only holds exactly when both sides differentiate the
*identical* computational graph. Reusing the tested Euler pattern for both
sides keeps that graph identical instead of merely order-matched. The
Tier-1 relaunch trick itself is not specific to Euler -- any step sequence
built purely from linear combinations of `f_wave_equation` calls (including
`rungeKutta2`) would work equally well.
"""

from __future__ import annotations

import pytest
import torch

from warpSPH.configurations import WaveCaseConfig, WaveEquationConfig, buildConfig
from warpSPH.caseUtils.waveEquation.damping import DampingProfiles, createDampingProfile
from warpSPH.sample.waveSystem import _wendlandKernelBump
from warpSPH.schemes.waveEquation import f_wave_equation
from warpSPH.systems.waveSystem import WaveSystemStatev3, WaveSystemv3, sampleInitialWaveState
from warpSPHCore import SupportScheme, buildVerletList

_CFL_FACTOR = 0.1


def _buildPrimalAndTangentSystems(nx, config, caseConfig, position, magnitude, dposition,
                                   dmagnitude, radius):
    """Seed `u0`/`du0` via `torch.func.jvp` on the plain-torch bump formula
    `sampleSmoothPointSourceWaveSystem` uses (plan step 1), then build the two
    `WaveSystemv3` instances plan step 2 rolls forward: same positions,
    supports, masses, densities, `c`, `damping` and adjacency in both, differing
    only in `(u, v)` vs. `(du, dv)`.
    """
    particleState = sampleInitialWaveState(nx, config, caseConfig)
    if caseConfig.domainDamping:
        particleState.damping = createDampingProfile(
            particleState, config, DampingProfiles.borderDamping_strong)
    positions = particleState.positions.detach()

    def u0Fn(pos, mag):
        distances = torch.linalg.norm(positions - pos, dim=-1)
        return mag * _wendlandKernelBump(distances, radius)

    u0, du0 = torch.func.jvp(u0Fn, (position, magnitude), (dposition, dmagnitude))
    u0, du0 = u0.detach(), du0.detach()
    v0 = torch.zeros_like(u0)
    dv0 = torch.zeros_like(du0)

    n = positions.shape[0]

    def _makeState(u, v):
        return WaveSystemStatev3(
            positions=positions, supports=particleState.supports,
            masses=particleState.masses, densities=particleState.densities,
            kinds=torch.zeros(n, device=config.device, dtype=torch.int32),
            materials=torch.zeros(n, device=config.device, dtype=torch.int32),
            UIDs=torch.arange(n, device=config.device, dtype=torch.int32),
            UIDcounter=n,
            u=u, v=v, c=particleState.c, damping=particleState.damping,
        )

    primalState = _makeState(u0, v0)
    tangentState = _makeState(du0, dv0)
    adjacency = buildVerletList(primalState, config.domain, 1.0, SupportScheme.SuperSymmetric, None)

    t0 = torch.tensor(0.0, device=config.device, dtype=config.dtype)
    primalSystem = WaveSystemv3(state=primalState, adjacency=adjacency, domain=config.domain, t=t0)
    tangentSystem = WaveSystemv3(state=tangentState, adjacency=adjacency, domain=config.domain, t=t0)
    return primalSystem, tangentSystem


def _rolloutEuler(system, dt, nSteps, config, schemeConfig):
    """The explicit-Euler step sequence shared by the forward-tangent and
    reverse-mode paths -- see the module docstring for why they must match.
    """
    for _ in range(nSteps):
        update, adjacency = f_wave_equation(system, dt, config, schemeConfig)
        system.adjacency = adjacency
        system.state.u = system.state.u + dt * update.dudt
        system.state.v = system.state.v + dt * update.dvdt
    return system


def _reverseModeDirectionalDerivative(nx, config, caseConfig, schemeConfig, position, magnitude,
                                       dposition, dmagnitude, radius, dt, nSteps, w):
    """`d(sum(w * u(T)))/d(position, magnitude)`, dotted with the same
    perturbation direction used to seed the forward-tangent rollout --
    step 3's reverse-mode reference, reusing
    `test_gradientsReachSourcePositionAndMagnitude`'s pattern verbatim.
    """
    position = position.clone().requires_grad_(True)
    magnitude = magnitude.clone().requires_grad_(True)

    particleState = sampleInitialWaveState(nx, config, caseConfig)
    if caseConfig.domainDamping:
        particleState.damping = createDampingProfile(
            particleState, config, DampingProfiles.borderDamping_strong)

    distances = torch.linalg.norm(particleState.positions - position, dim=-1)
    particleState.u = magnitude * _wendlandKernelBump(distances, radius)

    adjacency = buildVerletList(particleState, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
    t0 = torch.tensor(0.0, device=config.device, dtype=config.dtype)
    system = WaveSystemv3(state=particleState, adjacency=adjacency, domain=config.domain, t=t0)

    system = _rolloutEuler(system, dt, nSteps, config, schemeConfig)

    probe = (w * system.state.u).sum()
    probe.backward()

    return (position.grad * dposition).sum() + magnitude.grad * dmagnitude


def _checkForwardMatchesReverse(dim, nx, L, position, magnitude, dposition, dmagnitude, radius,
                                 dt, nSteps, probeSeed=0):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    from warpSPH.utils import buildDomainDescription
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, _integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device, dtype=dtype,
                                      dx=L / nx, cflFactor=_CFL_FACTOR)
    caseConfig = WaveCaseConfig(defaultSpeed=1.0, domainDamping=True)
    schemeConfig = WaveEquationConfig()

    position = position.to(device=device, dtype=dtype)
    magnitude = magnitude.to(device=device, dtype=dtype)
    dposition = dposition.to(device=device, dtype=dtype)
    dmagnitude = dmagnitude.to(device=device, dtype=dtype)

    primalSystem, tangentSystem = _buildPrimalAndTangentSystems(
        nx, config, caseConfig, position, magnitude, dposition, dmagnitude, radius)

    primalSystem = _rolloutEuler(primalSystem, dt, nSteps, config, schemeConfig)
    tangentSystem = _rolloutEuler(tangentSystem, dt, nSteps, config, schemeConfig)

    n = primalSystem.state.u.shape[0]
    generator = torch.Generator(device='cpu').manual_seed(probeSeed)
    w = torch.randn(n, generator=generator).to(device=device, dtype=dtype)

    forwardDot = (w * tangentSystem.state.u).sum()
    reverseDot = _reverseModeDirectionalDerivative(
        nx, config, caseConfig, schemeConfig, position, magnitude, dposition, dmagnitude, radius,
        dt, nSteps, w)

    return forwardDot, reverseDot


@pytest.mark.parametrize('dim,nx,L,position,magnitude,dposition,dmagnitude,nSteps', [
    (1, 64, 2.0, torch.tensor([0.3]), torch.tensor(5.0), torch.tensor([0.4]), torch.tensor(-0.6), 3),
    (1, 64, 2.0, torch.tensor([0.3]), torch.tensor(5.0), torch.tensor([0.4]), torch.tensor(-0.6), 6),
    (2, 32, 2.0, torch.tensor([0.3, -0.2]), torch.tensor(5.0), torch.tensor([0.4, -0.3]),
     torch.tensor(0.7), 3),
    (2, 32, 2.0, torch.tensor([0.3, -0.2]), torch.tensor(5.0), torch.tensor([0.4, -0.3]),
     torch.tensor(0.7), 6),
])
def test_forwardTangentRolloutMatchesReverseModeDirectionalDerivative(
        dim, nx, L, position, magnitude, dposition, dmagnitude, nSteps):
    """Plan step 3: the Tier-1 relaunch trick's tangent rollout agrees with
    an independent reverse-mode directional derivative, at a couple of
    rollout lengths (probe times) and in both 1D and 2D.
    """
    forwardDot, reverseDot = _checkForwardMatchesReverse(
        dim, nx, L, position, magnitude, dposition, dmagnitude, radius=0.2, dt=0.005,
        nSteps=nSteps)

    assert torch.isfinite(forwardDot) and torch.isfinite(reverseDot)
    assert forwardDot.item() != 0.0, 'forward-mode directional derivative is exactly zero'
    torch.testing.assert_close(forwardDot, reverseDot, rtol=1e-3, atol=1e-4)
