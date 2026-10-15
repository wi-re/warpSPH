"""`JFNK_PLAN.md` Phase B, step 1: the rudimentary WCSPH acoustic core
(`warpSPH.schemes.experimental.acousticCore.f_acoustic_core`) -- continuity + EOS +
pressure force, no dissipation. Foundation tests only: that the step
function is wired correctly (finite output, momentum conservation, the
right divergence sign) and that the `AcousticCoreState`/`AcousticCoreSystem`
pair actually satisfies `warpSPHIntegrators`' protocol well enough to run
through a generic implicit DIRK step with `JFNKSolver`, mirroring
`test_implicitWaveEquation.py`'s Step 4 for the wave equation.

The full three-way (explicit / Picard / JFNK) stability comparison Phase B's
plan calls for is a separate, longer-running follow-up; this file is the
"does the new scheme even work" layer underneath it.
"""

from __future__ import annotations

import torch

from warpSPH.sample.acousticCore import buildPeriodicVortexAcousticCoreSystem
from warpSPH.schemes.experimental.acousticCore import f_acoustic_core

from warpSPHIntegrators.solvers import FixedPointSolver, JFNKSolver
from warpSPHIntegrators.state import get_reference_state
from warpSPHIntegrators import get_scheme


def test_acousticCoreStepProducesFiniteBoundedDerivatives():
    system, config, schemeConfig = buildPeriodicVortexAcousticCoreSystem()
    update, adjacency = f_acoustic_core(system, dt=1e-4, config=config, schemeConfig=schemeConfig)

    assert torch.isfinite(update.dxdt).all()
    assert torch.isfinite(update.dvdt).all()
    assert torch.isfinite(update.drhodt).all()
    torch.testing.assert_close(update.dxdt, system.state.velocities)


def test_acousticCoreSymmetricPressureForceConservesMomentum():
    """`GradientScheme.Symmetric` is the momentum-conserving form
    (`JFNK_PLAN.md` Phase C's linearity note): `sum_i m_i dv_i/dt` should be
    ~zero on a periodic domain with no external forcing, since every
    pairwise contribution to the pressure force is antisymmetric. A real
    correctness check on the operator wiring, not just "it runs".
    """
    system, config, schemeConfig = buildPeriodicVortexAcousticCoreSystem()
    update, _adjacency = f_acoustic_core(system, dt=1e-4, config=config, schemeConfig=schemeConfig)

    totalMomentumRate = (system.state.masses[:, None] * update.dvdt).sum(dim=0)
    scale = (system.state.masses[:, None] * update.dvdt).abs().sum(dim=0).clamp_min(1e-12)
    torch.testing.assert_close(totalMomentumRate / scale, torch.zeros_like(totalMomentumRate),
                                rtol=0, atol=1e-4)


def test_acousticCoreCompressingFlowIncreasesDensity():
    """Sign check on the continuity term. `vx = -A*sin(kx)`, `vy = -A*sin(ky)`
    (period `L`, so it stays continuous across the periodic seam, unlike a
    field linear in absolute position) has analytic divergence
    `-A*k*(cos(kx)+cos(ky))`: maximally negative (compressing) at the domain
    centre `(0, 0)` and maximally positive (expanding) at the domain corner
    `(L/2, L/2)` (periodic-identified with `(-L/2, -L/2)`). `drho/dt =
    -rho*div(v)` should be positive at the former and negative at the latter.
    """
    system, config, schemeConfig = buildPeriodicVortexAcousticCoreSystem(uMag=0.0)
    L = float(config.domain.max[0] - config.domain.min[0])
    x, y = system.state.positions[:, 0], system.state.positions[:, 1]
    k = 2 * torch.pi / L
    amplitude = 0.05
    system.state.velocities = torch.stack([-amplitude * torch.sin(k * x), -amplitude * torch.sin(k * y)], dim=1)

    update, _adjacency = f_acoustic_core(system, dt=1e-4, config=config, schemeConfig=schemeConfig)

    compressRegion = (x.abs() < 0.15 * L) & (y.abs() < 0.15 * L)
    expandRegion = ((x - L / 2).abs() < 0.15 * L) & ((y - L / 2).abs() < 0.15 * L)
    assert compressRegion.any() and expandRegion.any()
    assert (update.drhodt[compressRegion] > 0).all(), (
        f'expected compression near (0,0) to increase density, '
        f'got min drhodt={update.drhodt[compressRegion].min().item():.3e}'
    )
    assert (update.drhodt[expandRegion] < 0).all(), (
        f'expected expansion near (L/2,L/2) to decrease density, '
        f'got max drhodt={update.drhodt[expandRegion].max().item():.3e}'
    )


def test_acousticCoreWiredThroughGenericDIRKWithJFNK():
    """Mirrors `test_implicitWaveEquation.py`'s
    `test_jfnkThroughGenericDIRKAgreesWithHandRolledCG`'s wiring half (no
    hand-rolled reference exists for this scheme): one backward-Euler step
    through the generic driver, with `JFNKSolver`, should converge to a
    finite, bounded state. This is the actual point of Phase B being built
    on `warpSPHIntegrators`' protocol at all -- if `AcousticCoreState`/
    `AcousticCoreSystem` didn't satisfy it correctly, this would fail or
    silently no-op rather than solve anything.
    """
    system, config, schemeConfig = buildPeriodicVortexAcousticCoreSystem()
    scheme = get_scheme('Backward Euler (implicit)')
    solver = JFNKSolver(matvec='fd', tol=1e-8, max_iterations=20)

    dt = 1e-3
    result = scheme(system, dt, f_acoustic_core, config, schemeConfig, solver=solver)
    resultState = get_reference_state(result.state)

    assert torch.isfinite(resultState.positions).all()
    assert torch.isfinite(resultState.velocities).all()
    assert torch.isfinite(resultState.densities).all()
    assert (resultState.densities > 0).all()


def test_acousticCorePicardAlsoRunsOneStep():
    """Sanity companion to the JFNK check above: `FixedPointSolver` (the
    library's existing default) should also be able to take one step on
    this scheme without erroring, at a `dt` mild enough that Picard's fixed
    iteration count is not expected to fail (unlike Phase B's later,
    dedicated stability comparison at a much larger `dt`).
    """
    system, config, schemeConfig = buildPeriodicVortexAcousticCoreSystem()
    scheme = get_scheme('Backward Euler (implicit)')
    solver = FixedPointSolver(iterations=5)

    dt = 1e-4
    result = scheme(system, dt, f_acoustic_core, config, schemeConfig, solver=solver)
    resultState = get_reference_state(result.state)

    assert torch.isfinite(resultState.positions).all()
    assert torch.isfinite(resultState.velocities).all()
    assert torch.isfinite(resultState.densities).all()
