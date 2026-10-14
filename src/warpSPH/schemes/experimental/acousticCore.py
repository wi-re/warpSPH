"""The rudimentary WCSPH acoustic core: continuity + EOS + pressure force,
and nothing else -- no artificial or physical viscosity, no surface-aware
pressure treatment, no gravity, no boundaries (`JFNK_PLAN.md` Phase B).

Built from exactly the same three warpSPHCore operator calls the plan
identified as already expressible with zero new JVP derivation:

- Continuity, `drho/dt = -rho * Divergence(v)`, the same call
  `modules/momentum/inconsistent.py`'s `computeMomentum` makes.
- The EOS, `p = isoThermalEOS(rho, rho0, c_s)` -- pointwise, no
  warpSPHCore operator at all, differentiable through plain torch autograd.
- The pressure-gradient force, `-Gradient_symmetric(p) / rho`, the same
  call `modules/pressure/symmetricForce.py`'s `computePressureForceSymmetric`
  makes (adapted here to `AcousticCoreState` directly, since that function is
  typed against `CompressibleState`).

This is the hypothesis-testing rung between the wave equation (Phase A) and
`deltaSPH_step` (Phase C/D): with an implicit solver of sufficient accuracy,
does weakly-compressible SPH stay stable with *zero* dissipation, or is
artificial viscosity load-bearing for stability independent of time
integration? Positions move here (unlike the wave equation's static-neighbor
case), so the caller is responsible for rebuilding adjacency each real step;
this function reuses whatever adjacency it is handed for every stage inside
one step, exactly as `f_wave_equation` does.

`JFNK_PLAN.md` Phase E1 adds one optional term on top of the Phase B core:
a Kolmogorov-flow body acceleration (`schemeConfig.forcingAmplitude != 0`),
folded directly into `dv/dt`. Pointwise in `positions` (wrapped periodic via
`getPeriodicPositions`, matching `cases/periodic/kolmogorov.py`'s own convention), so
it costs no new `warpSPHCore` operator and differentiates through plain
torch autograd -- same "no new derivation needed" shape as the EOS term.

Phase E1.5 adds two more optional terms, motivated directly by Phase E1's own
finding: the Kolmogorov base flow is a linearly unstable shear flow, so
zero-dissipation JFNK eventually diverges too (later than explicit/Picard,
but not never) -- dissipation is physically load-bearing against that
instability, unlike Phase B's vortex. Both terms are a single
`warpOperation(..., WarpOperation.Laplacian, ...)` call each, since
`Laplacian` is already one of the six value+geometry JVP-wrapped operators --
zero new `warpSPHCore` derivation, same "already expressible" pattern this
module's own docstring opens with:
- Density diffusion: a plain Fickian `densityDiffusionCoefficient * h *
  soundSpeed * Laplacian(densities)` added to `drho/dt` -- the same
  `delta * h / xi * c_s` prefactor shape `modules/deltaSPH/densityDiffusion.py`
  uses for the real scheme's own delta-SPH term (with `xi` folded into the
  coefficient here, since this core has no renormalized-gradient machinery to
  match it against), minus that module's flux/renormalization machinery.
- Velocity diffusion: `velocityDiffusionCoefficient * Laplacian(velocities)`
  added to `dv/dt` -- the classic Brookshaw/Morris SPH-viscosity Laplacian
  formula, computed by the same operator with no modification, since
  `Laplacian` of a vector field is already shaped componentwise
  (`wp_laplacian.py`'s `_computeSPHLaplacian_stateBackend`: "Laplacian of a
  scalar field is a scalar field, of a vector field a vector field").
  `AcousticCoreConfig.laplacianMode` defaults to the eponymous `Brookshaw` --
  its geometry-JVP kernel was originally scalar-field-only and crashed the
  exact-JVP path on this vector term (a real finding, caught building this),
  fixed at the root in `warpSPHCore` 2026-08-25 (generalized to generic
  `Any`-typed fields, no formula change) rather than routed around; see that
  config's own docstring for the full finding.
Both coefficients default to `0.0` (off), recovering Phase B/E1 exactly --
opt-in, like `forcingAmplitude`.
"""

import math
from typing import Optional

import torch
import torch.autograd.forward_ad as fwAD

from warpSPHCore import (
    GradientScheme, OperationDirection, OperationProperties, SupportScheme,
    WarpOperation, buildVerletList, warpOperation,
)

from ...configurations import AcousticCoreConfig, SimulationConfig
from ...math import getPeriodicPositions
from ...modules.eos.weaklyCompressible import isoThermalEOS
from ...systems.acousticCore import AcousticCoreSystem, AcousticCoreSystemUpdate

__all__ = ['f_acoustic_core', 'f_acoustic_core_rebuildAdjacency']


def f_acoustic_core(
    system: AcousticCoreSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: AcousticCoreConfig,
    verbose: bool = False,
):
    state = system.state

    divV = warpOperation(
        state, queryValues=state.velocities,
        domain=system.domain, adjacency=system.adjacency,
        operationProperties=OperationProperties(
            operation=WarpOperation.Divergence,
            kernel=schemeConfig.kernel,
            supportMode=schemeConfig.supportMode,
            operationMode=OperationDirection.AllToAll,
            gradientMode=schemeConfig.continuityGradientMode,
        ),
    )
    drhodt = -state.densities * divV

    pressures = isoThermalEOS(state.densities, schemeConfig.restDensity, schemeConfig.soundSpeed)

    gradP = warpOperation(
        state, queryValues=pressures,
        domain=system.domain, adjacency=system.adjacency,
        operationProperties=OperationProperties(
            operation=WarpOperation.Gradient,
            kernel=schemeConfig.kernel,
            supportMode=schemeConfig.supportMode,
            gradientMode=schemeConfig.pressureGradientMode,
        ),
    )
    dvdt = -gradP / state.densities[:, None]

    if schemeConfig.densityDiffusionCoefficient != 0.0:
        laplacianRho = warpOperation(
            state, queryValues=state.densities,
            domain=system.domain, adjacency=system.adjacency,
            operationProperties=OperationProperties(
                operation=WarpOperation.Laplacian,
                kernel=schemeConfig.kernel,
                supportMode=schemeConfig.supportMode,
                gradientMode=schemeConfig.laplacianGradientMode,
                laplacianMode=schemeConfig.laplacianMode,
            ),
        )
        drhodt = drhodt + schemeConfig.densityDiffusionCoefficient * state.supports * schemeConfig.soundSpeed * laplacianRho

    if schemeConfig.velocityDiffusionCoefficient != 0.0:
        laplacianV = warpOperation(
            state, queryValues=state.velocities,
            domain=system.domain, adjacency=system.adjacency,
            operationProperties=OperationProperties(
                operation=WarpOperation.Laplacian,
                kernel=schemeConfig.kernel,
                supportMode=schemeConfig.supportMode,
                gradientMode=schemeConfig.laplacianGradientMode,
                laplacianMode=schemeConfig.laplacianMode,
            ),
        )
        dvdt = dvdt + schemeConfig.velocityDiffusionCoefficient * laplacianV

    if schemeConfig.forcingAmplitude != 0.0:
        wrappedPositions = getPeriodicPositions(state.positions, system.domain)
        forcingX = schemeConfig.forcingAmplitude * torch.sin(
            schemeConfig.forcingWavenumber * math.pi * wrappedPositions[:, 1])
        forcing = torch.zeros_like(dvdt)
        forcing[:, 0] = forcingX
        dvdt = dvdt + forcing

    dxdt = state.velocities

    update = AcousticCoreSystemUpdate(dxdt=dxdt, dvdt=dvdt, drhodt=drhodt)

    return update, system.adjacency


class _PositionSupportsView:
    """Minimal `queryParticles`/`referenceParticles` stand-in for
    `buildVerletList` (`warpSPHCore/radiusSearch/verlet/build.py`), which
    only ever reads `.positions`/`.supports` off whatever it's handed. Used
    instead of `dataclasses.replace(state, positions=...)` so this stays
    agnostic to `AcousticCoreState`'s own field machinery (`integrated()`/
    `constant()` descriptors) -- it never constructs a real state object at
    all, just the two attributes the neighbor search actually touches.
    """
    __slots__ = ('positions', 'supports')

    def __init__(self, positions: torch.Tensor, supports: torch.Tensor):
        self.positions = positions
        self.supports = supports


def f_acoustic_core_rebuildAdjacency(
    system: AcousticCoreSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: AcousticCoreConfig,
    verbose: bool = False,
):
    """Same physics as `f_acoustic_core`, but rebuilds (or at minimum
    re-validates) the neighbor list at the top of every invocation, instead
    of reusing whatever adjacency the DIRK/JFNK driver was handed once at
    the start of the real step.

    `f_acoustic_core`'s own docstring is explicit that positions move within
    a step (unlike the wave equation's static-neighbor case) and that the
    *caller* is responsible for rebuilding adjacency once per real step --
    every DIRK stage, every outer Newton iteration, and every inner GMRES
    matvec inside one step all see the exact same `AdjacencyList` object as
    a result. The real production scheme, by contrast
    (`warpSPH/src/warpSPH/schemes/deltaSPH.py:69-74`), calls
    `buildVerletList(currentState, config.domain, verletScale=config.verletScale,
    supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=adjacency,
    verbose=False)` unconditionally at the top of *every single RHS
    evaluation* -- so for an implicit stage built from that scheme, the
    neighbor list would get re-checked (and, once particles have moved past
    the Verlet skin, actually rebuilt) on every inner Newton/GMRES call, not
    frozen for the whole implicit solve. This function is the ablation:
    identical physics to `f_acoustic_core`, but with that same
    `buildVerletList(..., priorNeighborhood=...)` call inserted at the top,
    mirroring `deltaSPH_step`'s exact call shape (down to the keyword names)
    rather than a bespoke one. `f_acoustic_core` itself is untouched -- this
    is a new sibling function, not a modification, so nothing that already
    depends on `f_acoustic_core`'s frozen-adjacency behavior is at risk.

    **Dual-tensor handling (`JFNK_PLAN.md`'s neighbor-list-staleness
    ablation, item 2)**: under `jvp_matvec` (`warpSPHIntegrators/jfnk.py`,
    JFNK_PLAN.md A3), `state.positions` -- an `integrated` field -- may be a
    forward-mode dual tensor (wrapped inside
    `torch.autograd.forward_ad.dual_level()`), and is very likely to
    actually be one in practice here: `dx/dt = v`, so the position tangent
    is zero only when the velocity tangent is exactly zero, which isn't the
    steady-state case JVP's zero-tangent skip (A3's finding) was built for.
    Checked directly, not assumed: calling `buildVerletList` on dual-wrapped
    positions does **not** crash, in either the `priorNeighborhood=None`
    path or the `priorNeighborhood=<real adjacency>` path that exercises
    `_verlet_validity_metrics`'s distance computation against a *plain*
    prior adjacency -- warp's `wp.from_torch()` bridge silently reads
    through to the dual tensor's primal buffer, and the returned adjacency's
    `i`/`j`/`numNeighbors` were confirmed bit-identical to calling
    `buildVerletList` on the manually-unpacked primal directly (see this
    task's own scratch probe). So this path was *not* known to be broken --
    a real, verified negative finding, not a guess. The explicit primal
    extraction below is kept anyway, as the semantically-correct thing to do
    regardless (the adjacency structure is discrete/non-differentiable and
    should never carry a tangent at all, and leaving `adjacency.queryPositions`
    dual-wrapped risks it being read after its `dual_level()` has exited --
    a real hazard in general, though traced through `JFNKSolver.solve`'s own
    call order and confirmed not to actually manifest today: every return
    path re-evaluates `step(Y)` outside any `dual_level()` immediately
    before returning, so a dual-tainted adjacency from inside a GMRES matvec
    never survives to become the stage's or the real step's `system.adjacency`).
    Costs nothing extra either way -- `fwAD.unpack_dual` is a no-op read on a
    plain tensor.
    """
    state = system.state

    primalPositions, _ = fwAD.unpack_dual(state.positions)
    queryView = _PositionSupportsView(primalPositions, state.supports)

    adjacency = buildVerletList(
        queryView, system.domain, verletScale=config.verletScale,
        supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=system.adjacency, verbose=verbose,
    )

    divV = warpOperation(
        state, queryValues=state.velocities,
        domain=system.domain, adjacency=adjacency,
        operationProperties=OperationProperties(
            operation=WarpOperation.Divergence,
            kernel=schemeConfig.kernel,
            supportMode=schemeConfig.supportMode,
            operationMode=OperationDirection.AllToAll,
            gradientMode=schemeConfig.continuityGradientMode,
        ),
    )
    drhodt = -state.densities * divV

    pressures = isoThermalEOS(state.densities, schemeConfig.restDensity, schemeConfig.soundSpeed)

    gradP = warpOperation(
        state, queryValues=pressures,
        domain=system.domain, adjacency=adjacency,
        operationProperties=OperationProperties(
            operation=WarpOperation.Gradient,
            kernel=schemeConfig.kernel,
            supportMode=schemeConfig.supportMode,
            gradientMode=schemeConfig.pressureGradientMode,
        ),
    )
    dvdt = -gradP / state.densities[:, None]

    if schemeConfig.densityDiffusionCoefficient != 0.0:
        laplacianRho = warpOperation(
            state, queryValues=state.densities,
            domain=system.domain, adjacency=adjacency,
            operationProperties=OperationProperties(
                operation=WarpOperation.Laplacian,
                kernel=schemeConfig.kernel,
                supportMode=schemeConfig.supportMode,
                gradientMode=schemeConfig.laplacianGradientMode,
                laplacianMode=schemeConfig.laplacianMode,
            ),
        )
        drhodt = drhodt + schemeConfig.densityDiffusionCoefficient * state.supports * schemeConfig.soundSpeed * laplacianRho

    if schemeConfig.velocityDiffusionCoefficient != 0.0:
        laplacianV = warpOperation(
            state, queryValues=state.velocities,
            domain=system.domain, adjacency=adjacency,
            operationProperties=OperationProperties(
                operation=WarpOperation.Laplacian,
                kernel=schemeConfig.kernel,
                supportMode=schemeConfig.supportMode,
                gradientMode=schemeConfig.laplacianGradientMode,
                laplacianMode=schemeConfig.laplacianMode,
            ),
        )
        dvdt = dvdt + schemeConfig.velocityDiffusionCoefficient * laplacianV

    if schemeConfig.forcingAmplitude != 0.0:
        wrappedPositions = getPeriodicPositions(state.positions, system.domain)
        forcingX = schemeConfig.forcingAmplitude * torch.sin(
            schemeConfig.forcingWavenumber * math.pi * wrappedPositions[:, 1])
        forcing = torch.zeros_like(dvdt)
        forcing[:, 0] = forcingX
        dvdt = dvdt + forcing

    dxdt = state.velocities

    update = AcousticCoreSystemUpdate(dxdt=dxdt, dvdt=dvdt, drhodt=drhodt)

    return update, adjacency
