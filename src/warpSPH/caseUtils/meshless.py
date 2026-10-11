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

__all__ = ['buildMeshlessState']


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
