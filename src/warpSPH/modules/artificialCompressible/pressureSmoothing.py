"""`D^p`, the ACSPH pressure-smoothing operator (De Courcy et al. 2024
Eqs. 32-37; `ACSPH_PLAN.md` Secs. 1.4 and 4.3).

Returns the **unscaled** operator -- the `k2 = 0.1 h beta` prefactor of Eq. (24)
belongs to the caller (`schemes/artificialCompressible.py`), which is the only
place `beta` exists.

Nothing here is new numerics. AC-2 and AC-2L are the delta-SPH density-diffusion
operator with the pressure in place of the density, which
`computeScalarFieldDiffusion` supports directly:

    AC-2  (Eq. 32)      == DensityDiffusionScheme.densityOnly
    AC-2L (Eqs. 33-34)  == DensityDiffusionScheme.deltaSPH, with <grad p>^L
    AC-4  (Eq. 35)      == DensityDiffusionScheme.densityOnly applied *twice*
                           (see below)

See Part 3 of the plan for why our *unprojected* psi reproduces the paper's
*projected* Eq. (33) exactly (`gradW || x_ij` for any isotropic kernel;
measured to 2e-15 by `scripts/probe_deltaSPHPsiProjection.py`) -- and for the
one condition under which it does not: gradient renormalisation on `gradW`
itself. `computeScalarFieldDiffusion` builds its own `OperationProperties` with
no `renormalizationState`, so that path is off by construction here; `L` enters
only through `<grad p>^L`, which is where Eq. (34) wants it.

**AC-4 (Eq. 35)** is `-h**2 * Sum_j 2(D^Delta_i - D^Delta_j)(x_ij.gradW_ij)/|x_ij|**2 V_j`,
where `D^Delta` is AC-2's own (unrenormalised) output -- *not* AC-2L's, per
the plan's own Part 1.4 numbering ("Two neighbour loops, no correction.
Inherits AC-2's truncation error but weaker."). `DensityDiffusionScheme.densityOnly`'s
own flux is exactly `2(f_i-f_j)(x_ij.gradW_ij)/|x_ij|**2` (module docstring,
`modules/deltaSPH/wp_densityDelta.py`), i.e. precisely Eq. (35)'s neighbour
sum with `f = D^Delta` -- so AC-4 is `densityOnly` applied to `densityOnly`'s
own output, no new kernel.

⚠ **Verified formula, but genuinely unstable *at full strength* -- not a
transcription error.** Re-checked character-by-character against the actual
PDF (not just `ACSPH_PLAN.md`'s own transcription) and matches. Run *alone*
on `hydrostaticColumn` (nx=24) and `rotatingSquarePatch` (nx=24, no walls, so
not a boundary-treatment artefact) it diverges within ~15 real steps
(pressure reaching O(1e6), velocity O(1e5)): AC-2's own raw output is already
large near a free surface/wall by construction (the known-bad truncation
AC-2L exists to fix), and nesting it without any renormalisation amplifies
whatever is already there roughly 20x per pass, measured on the initial
hydrostatic state alone, before any dynamics.

**Cross-checked by AC-JST below, which uses this same operator and does
*not* blow up** -- run on the same two cases plus `oscillatingDroplet` for
60 real steps, all three stay bounded (`‖v‖` peaks in the 0.3-5 range, not
`1e5`). AC-JST's `epsilon_4` caps AC-4's contribution at `kappa_4 = 1/32`
(against this formula's own implicit `epsilon_4 = 1` when used alone) and
zeroes it out entirely once the JST switch `chi_i` exceeds
`kappa_4/kappa_2 = 1/16`, i.e. anywhere the flow is not close to smooth --
so the ~20x amplification measured above is throttled by at least that
factor of 32 in every configuration that matters, consistent with the
paper's own report that AC-4 alone "struggles to maintain a converged
kinetic energy" while AC-JST behaves like AC-2L. This confirms the AC-4
*formula* is correct (a real sign or coefficient error would show up in the
JST blend too, just scaled down, not disappear) -- the standalone
instability is a real property of running it unblended at fixed
resolution, not an implementation bug. **Still not validated for
production use standalone** (`PressureSmoothingScheme.biharmonic`); use
`.jst` if AC-4's fourth-order behaviour in smooth regions is wanted without
the standalone risk. See `ACSPH_PLAN.md` Part 8 step 8.
"""

from typing import Any, Optional, Union

import torch
from torch.profiler import record_function
from warpSPHCore import (AdjacencyList, CompactHashMap, OperationProperties,
                          SupportScheme, WarpOperation)

from ...configurations.simulationConfig import SimulationConfig
from ...enumTypes import DensityDiffusionScheme, PressureSmoothingScheme
from ..deltaSPH import computeScalarFieldDiffusion
from ..density.gradRhoL import computeGradRhoL
from .wp_jstSwitch import computeJstSwitchWarp

__all__ = ['computePressureSmoothing']


def computePressureSmoothing(
    currentState: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
    renormalizationState: Any = None,
    pressures: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """`D^p_i` for the configured `acParams.pressureSmoothing`, unscaled.

    `pressures` defaults to `currentState.pressures`; pass it explicitly when
    the dual-time loop is evaluating at a stage value rather than the state's
    own. `renormalizationState` is the `L` from `detectFreeSurface` -- reused
    rather than recomputed, since AC-2L needs exactly the matrix Eq. (34) does.
    """
    scheme = schemeConfig.acParams.pressureSmoothing
    p = currentState.pressures if pressures is None else pressures

    with record_function("[warpSPH] - (acsph) - computePressureSmoothing"):
        if scheme is PressureSmoothingScheme.laplacian:
            return computeScalarFieldDiffusion(
                currentState, config, adjacency,
                DensityDiffusionScheme.densityOnly, field=p)

        if scheme is PressureSmoothingScheme.renormalizedBiLaplacian:
            return _biLaplacian(currentState, config, schemeConfig, adjacency,
                                renormalizationState, p)

        if scheme is PressureSmoothingScheme.biharmonic:
            return _biharmonic(currentState, config, adjacency, p)

        if scheme is PressureSmoothingScheme.jst:
            return _jst(currentState, config, schemeConfig, adjacency,
                       renormalizationState, p)

        raise NotImplementedError(
            f"pressureSmoothing={scheme.name} is not implemented -- known "
            f"schemes are PressureSmoothingScheme.renormalizedBiLaplacian "
            f"(AC-2L, the paper's default), .laplacian (AC-2, its negative "
            f"control), .biharmonic (AC-4) and .jst (AC-JST).")


def _biLaplacian(currentState, config, schemeConfig, adjacency, renormalizationState, p):
    """AC-2L (Eqs. 33-34), factored out so `_jst` can reuse it without
    recomputing `<grad p>^L`."""
    gradPL = computeGradRhoL(currentState, config, schemeConfig, adjacency,
                             renormalizationState, field=p)
    return computeScalarFieldDiffusion(
        currentState, config, adjacency,
        DensityDiffusionScheme.deltaSPH, gradFieldL=gradPL, field=p)


def _biharmonic(currentState, config, adjacency, p):
    """AC-4 (Eq. 35), factored out so `_jst` can reuse it. See the module
    docstring's stability caveat -- unchanged by being reused here."""
    dDelta = computeScalarFieldDiffusion(
        currentState, config, adjacency,
        DensityDiffusionScheme.densityOnly, field=p)
    h2 = currentState.supports ** 2
    return -h2 * computeScalarFieldDiffusion(
        currentState, config, adjacency,
        DensityDiffusionScheme.densityOnly, field=dDelta)


def _jst(currentState, config, schemeConfig, adjacency, renormalizationState, p):
    """AC-JST (Eqs. 36-37): AC-2L at the free surface (dilated set `V`),
    otherwise `epsilon_2 * AC-2L + epsilon_4 * AC-4`, blended by `chi_i`
    (Eq. 37's smoothness switch, `wp_jstSwitch.py`).

    **`epsilon_4 = max(0, kappa_4 - epsilon_2)`, not the paper-printed
    `min`** (`ACSPH_PLAN.md` §5.1: as printed, `min` makes the JST operator
    vanish identically in smooth flow, the opposite of its stated design;
    standard JST literature uses `max`). `acParams.jstUsePrintedMin` reproduces
    the printed form for comparison; default is `max`.

    `surfaceIndicators` (the dilated free-surface set `V`, `schemes/artificialCompressible.py`'s
    per-real-step `detectFreeSurface` call, frozen for the whole dual-time
    loop like everything else that reads it) is read directly off
    `currentState` -- `None`/absent (surface detection inactive) means no
    particle is in `V`, so every particle uses the interior blend.
    """
    acParams = schemeConfig.acParams
    biLap = _biLaplacian(currentState, config, schemeConfig, adjacency,
                         renormalizationState, p)
    biharm = _biharmonic(currentState, config, adjacency, p)

    chi = computeJstSwitchWarp(
        currentState,
        OperationProperties(operation=WarpOperation.Density, kernel=config.kernel,
                            supportMode=SupportScheme.SuperSymmetric),
        config.domain, p, adjacency=adjacency)

    eps2 = acParams.kappa2 * torch.clamp(chi, max=1.0)
    if acParams.jstUsePrintedMin:
        eps4 = torch.clamp(acParams.kappa4 - eps2, max=0.0)
    else:
        eps4 = torch.clamp(acParams.kappa4 - eps2, min=0.0)

    interior = eps2 * biLap + eps4 * biharm

    surfaceIndicator = getattr(currentState, 'surfaceIndicators', None)
    if surfaceIndicator is None:
        return interior
    inV = surfaceIndicator.to(torch.bool)
    return torch.where(inV, biLap, interior)
