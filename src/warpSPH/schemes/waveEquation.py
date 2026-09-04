"""The scalar wave equation on SPH particles.

Solves ``d2u/dt2 = c^2 laplacian(u)`` as a first-order system in ``(u, v)``,
with a PML-style absorbing term ``- damping * v`` folded into the acceleration
rather than applied after integration, which absorbs outgoing waves far better
than post-step damping does.

This is deliberately *not* a fluid scheme. It exists as a demo of the SPH
operators acting on unstructured, particle-like data -- a Laplacian on
scattered points -- which is the shape of problem graph/point-cloud ML models
are usually posed on. The heterogeneous wave speed ``c`` and the
randomisable sources/obstacles in :mod:`warpSPH.configurations.waveEquationConfig`
are there to make families of such samples, not to model any particular
physical setup.

Positions are static: the adjacency built once by `finalizeWaveSystemSetup`
(:mod:`warpSPH.sample.waveSystem`) is reused every step rather than rebuilt,
so this is an explicit, non-moving-neighbourhood scheme. Implicit time
integration and forward-mode AD are not wired up here; see
`docs/historic_plans/WAVE_EQUATION_PLAN.md` for that follow-up.

See :mod:`warpSPH.systems.waveSystem` for the state it integrates,
:mod:`warpSPH.cases.waveEquation` for the registered `Case`, and the README
section "The wave system" for how the pieces fit together.
"""

from ..systems import WaveSystemUpdatev3, WaveSystemv3
from ..configurations import SimulationConfig, WaveEquationConfig
from warpSPHCore import OperationProperties, WarpOperation, warpOperation

__all__ = ['f_wave_equation']


def f_wave_equation(
    system: WaveSystemv3,
    dt: float,
    config: SimulationConfig,
    schemeConfig: WaveEquationConfig,
    verbose: bool = False,
):
    state = system.state

    # Positions are fixed for this (explicit, non-moving) case, so the
    # adjacency built once by `finalizeWaveSystemSetup` is reused every step
    # rather than rebuilt.
    laplacian_u = warpOperation(
        state, queryValues = state.u,
        domain = system.domain, adjacency = system.adjacency,
        operationProperties = OperationProperties(
            operation=WarpOperation.Laplacian,
            kernel = schemeConfig.kernel,
            supportMode = schemeConfig.supportMode,
            laplacianMode = schemeConfig.laplacianMode,
            gradientMode = schemeConfig.gradientMode,
        ),
    )

    # Apply PML-style damping to the derivatives
    # This absorbs waves more effectively than post-integration damping
    dudt = state.v
    dvdt = state.c**2 * laplacian_u - state.damping * state.v

    update = WaveSystemUpdatev3(dudt=dudt, dvdt=dvdt)

    return update, system.adjacency
