"""Shared IC helper for running the compressible battery cases under the meshless
(MFM / MFV) schemes (COMPRESSIBLE_SPH_PLAN E2).

The battery cases' samplers build a `CompressibleState` (densities / pressures /
velocities). The meshless schemes integrate the conserved vector
``Q = (m, m v, m e_tot)`` instead, so this converts the same IC fields into a
`MeshlessState`: ``Q = conserved(m, rho, v, P)`` (the G1 pattern) with the
primitives filled in as the diagnostics the step refreshes anyway, so the
pre-step readers (the velocity alarm, the t = 0 diagnostics row, the plot) see
the IC values.

Imported lazily from the cases' meshless branches: this module pulls in the
`warpSPHCore.mfm` backend, which a non-meshless run has no reason to load.
"""

import torch

from warpSPHCore.mfm import conserved

from ..systems.meshless import MeshlessState

__all__ = ['buildMeshlessState', 'toMeshlessSystem', 'buildPressureEquilibriumSystem', 'buildKidderMeshlessSystem']


def buildMeshlessState(positions, supports, masses, densities, velocities, pressures,
                       gamma, kinds, materials, UIDs, *, emitters=None, bodyIDs=None):
    """The meshless initial state from the compressible IC fields.

    `masses` / `densities` / `pressures` (N,), `velocities` (N, dim),
    `positions` (N, dim), `supports` (N,), integer `kinds` / `materials` /
    `UIDs` (N,) -- exactly what the battery samplers assemble for
    `CompressibleState`. Returns a `MeshlessState`: the conserved Q integrated,
    the primitives as diagnostics, `lam` unset for the first closure solve.
    """
    gamma = gamma if isinstance(gamma, torch.Tensor) else float(gamma)
    Q = conserved(masses, densities, velocities, pressures, gamma)
    state = MeshlessState(
        positions=positions,
        Q=Q,
        supports=supports,
        kinds=kinds,
        materials=materials,
        emitters=emitters,
        bodyIDs=bodyIDs,
        UIDs=UIDs,
        UIDcounter=int(UIDs.max().item()) + 1,
    )
    # Diagnostics, the same formulas `meshlessStep` refreshes every step
    e_tot = Q[:, -1] / masses.clamp_min(1e-30)
    v2 = torch.einsum('ij,ij->i', velocities, velocities)
    state.masses = masses
    state.densities = densities
    state.velocities = velocities
    state.pressures = pressures
    state.totalEnergies = Q[:, -1]
    state.internalEnergies = e_tot - 0.5 * v2
    state.soundspeeds = torch.sqrt(gamma * pressures / densities.clamp_min(1e-30))
    state.entropies = pressures / densities.clamp_min(1e-30) ** gamma
    return state


def toMeshlessSystem(compSystem, config, schemeConfig, *, inflowReservoir=None):
    """The shared E2 case path: a built compressible system to the meshless form.

    Each battery case's meshless branch runs the case's own sampler unchanged
    with CompressibleState / CompressibleSystem (the scheme's own state types
    would not carry the IC fields the sampler assembles), then hands the result
    here: the conserved Q from the IC's (m, rho, v, P) (`buildMeshlessState`),
    the `MeshlessSystem` on the case's domain, and `config.dt` from the
    dispatcher (the core signal timestep, `modules/timestep/meshless.py`).
    """
    from ..modules.timestep import computeTimestep
    from ..systems.meshless import MeshlessSystem

    st = compSystem.state
    gamma = float(schemeConfig.gamma)
    if schemeConfig.materials is not None:
        from ..modules.eos.materials import validateMaterialState, materialRuntime
        validateMaterialState(st, schemeConfig)
        gamma = materialRuntime(schemeConfig, st.densities).gammas[st.materials.long()]
    state = buildMeshlessState(
        st.positions, st.supports, st.masses,
        st.densities, st.velocities, st.pressures,
        gamma,
        st.kinds, st.materials, st.UIDs, emitters=st.emitters, bodyIDs=st.bodyIDs)
    system = MeshlessSystem(state=state, domain=config.domain, inflowReservoir=inflowReservoir)
    config.dt = computeTimestep(system, config, schemeConfig, dt=None if config.adaptiveDt else config.dt)
    return system


def buildPressureEquilibriumSystem(ctx):
    """Sample a contact using number-volume geometry and physical cell masses.

    SPH density/support adaptation across a mass jump is not a meshless volume
    partition. Initialize the energy with the same geometric volumes that the
    first RHS reads, so uniform pressure survives primitive refresh.
    """
    from ..sample.regular import sampleRegularParticles
    from ..modules.meshless.geometry import buildGeometry, stashGeometry
    from ..systems.meshless import MeshlessSystem
    from ..modules.timestep import computeTimestep

    cfg, scheme = ctx.config, ctx.schemeConfig
    particles = sampleRegularParticles(ctx.spec.nx, cfg.domain, cfg.targetNeighbors)
    pos, supports = particles.positions, particles.supports
    L = cfg.domain.max[0] - cfg.domain.min[0]
    if ctx.param('shape') == 'ellipse':
        mask = (pos[:, 0] / (L / 4)).square() + (pos[:, 1] / (L / 8)).square() < 1
    else:
        mask = (pos[:, 0].abs() < L / 4) & (pos[:, 1].abs() < L / 4)
    density = torch.full_like(particles.masses, ctx.param('rho_low'))
    density[mask] = ctx.param('rho_high')
    mass = density * particles.masses
    geometry = buildGeometry(pos, supports, None, cfg, scheme, cfg.domain)
    velocity = torch.zeros_like(pos)
    velocity[:, 0] = ctx.param('boost')
    labels = torch.zeros(len(pos), device=pos.device, dtype=torch.int32)
    materialIDs = labels.clone()
    gamma = scheme.gamma
    if scheme.materials is not None:
        if len(scheme.materials) != 2:
            raise ValueError('hydrostatic requires exactly two EOS entries when materials is configured')
        materialIDs = mask.to(torch.int32)
        from ..modules.eos.materials import materialRuntime
        gamma = materialRuntime(scheme, mass).gammas[materialIDs.long()]
    state = buildMeshlessState(pos, supports, mass, mass / geometry.volume,
        velocity, torch.ones_like(mass), gamma, labels, materialIDs,
        torch.arange(len(pos), device=pos.device, dtype=torch.int32))
    state.lam = geometry.lam
    stashGeometry(pos, supports, cfg.domain, cfg.kernel, scheme, geometry, owner=state)
    system = MeshlessSystem(state=state, domain=cfg.domain)
    cfg.dt = computeTimestep(system, cfg, scheme, dt=cfg.dt)
    return system


def buildKidderMeshlessSystem(ctx, solution):
    """Planar shell with analytic moving bands around an evolved interior."""
    from ..modules.meshless.boundary import DrivenParticles
    from ..modules.meshless.geometry import buildGeometry
    from ..modules.meshless.primitives import refreshPrimitives
    from ..modules.meshless.context import supports_from_volume
    from ..systems.meshless import MeshlessSystem
    from ..modules.timestep import computeTimestep

    cfg, scheme = ctx.config, ctx.schemeConfig
    pos = torch.linspace(solution.r0, solution.r1, ctx.spec.nx,
                         device=cfg.device, dtype=cfg.dtype)[:, None]
    dx = (solution.r1 - solution.r0) / (ctx.spec.nx - 1)
    supports = supports_from_volume(torch.full_like(pos[:, 0], dx), scheme.nngb, 1)
    radius = pos[:, 0]

    def initialDensity(radius):
        weight = (radius.square() - solution.r0**2) / (solution.r1**2 - solution.r0**2)
        return ((1 - weight) * solution.rho0**solution.gamma1 +
                weight * solution.rho1**solution.gamma1)**(1 / solution.gamma1)

    density = initialDensity(radius)
    mass = density * dx
    geometry = buildGeometry(pos, supports, None, cfg, scheme, cfg.domain)
    labels = torch.zeros(len(pos), dtype=torch.int32, device=pos.device)
    initialPressure = solution.S * density**solution.gamma
    state = buildMeshlessState(pos, supports, mass, mass / geometry.volume,
        torch.zeros_like(pos), initialPressure, scheme.gamma, labels, labels.clone(),
        torch.arange(len(pos), device=pos.device, dtype=torch.int32))
    index = torch.arange(len(pos), device=pos.device)
    mask = (index < ctx.param('band')) | (index >= len(pos) - ctx.param('band'))

    def scale(time):
        return (1 - (time / solution.tau)**2)**.5

    def trajectory(time, reference):
        return reference * scale(time)

    def primitives(time, positions):
        h = scale(time)
        velocity = -positions * time / (solution.tau * h)**2
        pressure = initialPressure / h**(solution.nu * solution.gamma)
        return velocity, pressure

    driven = DrivenParticles(mask, pos.clone(), mass.clone(), trajectory, primitives)
    system = MeshlessSystem(state=state, domain=cfg.domain, drivenParticles=driven)
    driven.conserved(state, geometry, 0., float(scheme.gamma))
    refreshPrimitives(state, geometry, float(scheme.gamma), scheme.nngb, cfg.dim, updateSupports=False)
    cfg.dx = dx
    cfg.dt = computeTimestep(system, cfg, scheme, dt=cfg.dt)
    return system
