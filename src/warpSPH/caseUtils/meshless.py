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

__all__ = ['buildMeshlessState', 'toMeshlessSystem']


def buildMeshlessState(positions, supports, masses, densities, velocities, pressures,
                       gamma, kinds, materials, UIDs):
    """The meshless initial state from the compressible IC fields.

    `masses` / `densities` / `pressures` (N,), `velocities` (N, dim),
    `positions` (N, dim), `supports` (N,), integer `kinds` / `materials` /
    `UIDs` (N,) -- exactly what the battery samplers assemble for
    `CompressibleState`. Returns a `MeshlessState`: the conserved Q integrated,
    the primitives as diagnostics, `lam` unset for the first closure solve.
    """
    gamma = float(gamma)
    Q = conserved(masses, densities, velocities, pressures, gamma)
    state = MeshlessState(
        positions=positions,
        Q=Q,
        supports=supports,
        kinds=kinds,
        materials=materials,
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


def toMeshlessSystem(compSystem, config, schemeConfig):
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
    state = buildMeshlessState(
        st.positions, st.supports, st.masses,
        st.densities, st.velocities, st.pressures,
        float(schemeConfig.gamma),
        st.kinds, st.materials, st.UIDs)
    system = MeshlessSystem(state=state, domain=config.domain)
    config.dt = computeTimestep(system, config, schemeConfig, dt=None)
    return system
