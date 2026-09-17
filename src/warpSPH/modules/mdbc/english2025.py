"""mDBC boundary-particle density extrapolation via English, Vacondio et al.
2025's "pressure cloning" (`literature/english2025_river-flows-past-
bridges.pdf`, Eqs. 8-11) -- a third alternative to `density2025.py`'s English
et al. 2022 Eq. (12) fitted-gradient extrapolation and `densityBand.py`'s
Band et al. 2018-style fitted-gradient extrapolation (both Taylor-shift a
FITTED gradient from the ghost node to the boundary particle).

`BOUNDARY_DENSITY_PLAN.md` §3.1 found that this Taylor shift is not free: a
correctly-computed, well-conditioned gradient still gets multiplied by a
lever arm (`relPos`) that grows with the boundary particle's layer depth,
amplifying either fit noise or the gradient's own discretization error into
an unphysical extrapolated value at depth. English 2025 replaces the fitted
gradient with a KNOWN, noise-free physical quantity instead:

    rho_g = Shepard/MLS VALUE fit AT the ghost                (paper's Eq. 8)
    P_g   = c0^2 (rho_g - rho0)                                (Eq. 9)
    P_b   = P_g + rho0 * dot(g - a_b, relPos)                  (Eq. 10, corrected)
    rho_b = rho0 + P_b / c0^2                                  (Eq. 11)

`relPos = x_b - x_g` is exactly this codebase's own convention
(`density2025.py`'s `relPos = -currentState.ghostOffsets[...]`), reused
unmodified. The `P_b` sign here is CORRECTED relative to the paper's own
printed Eq. (10) as extracted by `pdftotext` -- cross-checked directly
against DualSPHysics' own `JSphCpu::Mdbc2PressClone`
(`JSphCpu_mdbc.cpp`), the reference implementation of this exact scheme; see
`BOUNDARY_DENSITY_PLAN.md` §5.1 for the full derivation. `P_g`'s linear
(isothermal) `c0^2 (rho - rho0)` relation is DualSPHysics' own convention for
this boundary pressure-density map specifically -- independent of whichever
bulk EOS (`stiffTait`, etc.) the fluid solver itself uses elsewhere.

The VALUE fit at the ghost (`rho_g`) is `densityBand.py`'s decoupled 0th-order
Shepard term (`BOUNDARY_DENSITY_PLAN.md` §2-3): exact and well-posed whenever
there is >=1 real fluid neighbour, no matrix inversion at all, since there is
no gradient block to decouple it from here -- this scheme never fits a
gradient in the first place. `densityBand.py`'s Mb-precision-cancellation fix
(gate "has real neighbours" on the exact integer fluid-neighbour count, not a
scale-sensitive threshold on `Mb` itself) is reused verbatim rather than
re-derived, since it was a real, non-obvious bug there.

`BOUNDARY_DENSITY_PLAN.md` §5.2's toy depth-sweep validated this combination
(analytic extrapolation + Band's Tikhonov/Shepard value) against the exact
§3.1 failure mode: error stays flat (~1.2x shallow-to-deep) where both
existing fitted-gradient schemes grow 5-11x. Not yet validated on a real
case -- see `scripts/probe_deltaSPHMarrone.py --mdbcDensityScheme
english2025` / `scripts/probe_deltaSPHMarrone34.py --mdbcDensityScheme
english2025`.

`a_b` (boundary acceleration) is always zero here -- this codebase does not
track a per-particle rigid-body acceleration field for boundary particles
yet, so this is static-wall-only (fine for Marrone 3.1/3.4 and the dam-break
suite; flagged in the plan §5.3 for a moving-boundary case later).
"""

from typing import Any, Optional, Union

import torch
from torch.profiler import record_function

from warpSPHCore import *

from ...configurations.simulationConfig import SimulationConfig
from ..liu.wp_mat import computeLiuMatricesWarp
from ..gravity.wrapper import computeGravity
from ._util import stateHasBoundaryParticles

__all__ = ['computeMdbcDensityEnglish2025']

# DIAGNOSTIC ONLY (BOUNDARY_DENSITY_PLAN.md §10 ceiling-hover investigation).
# `scripts/probe_ceilingHover.py` sets this to a list before `run()`; every
# call appends one dict of per-ghost-row diagnostics. `None` (default): the
# `if _CAPTURE is not None` check below is the only added cost, everywhere
# else this module is used. Revert (`git checkout`) after use.
_CAPTURE: Optional[list] = None


def computeMdbcDensityEnglish2025(currentState: Any, config: SimulationConfig, schemeConfig: Any,
                                  adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    if not stateHasBoundaryParticles(currentState, config):
        return currentState.densities
    dim = currentState.positions.shape[1]
    if dim != 2:
        raise NotImplementedError(
            "computeMdbcDensityEnglish2025: only 2D is validated so far "
            f"(matches densityBand.py's own scope); got dim={dim}")

    with record_function("[warpSPH] - (mdbc) - computeMdbcDensityEnglish2025"):
        rho0 = schemeConfig.fluid.restDensity
        c0 = schemeConfig.fluid.fixedSoundSpeed
        ghost = currentState.kinds == 2
        if not bool(ghost.any()):
            return currentState.densities
        bIndices = currentState.ghostIndices[ghost]
        # r_b - r_g, same convention as density2025.py's English Eq. (12) and
        # densityBand.py's Taylor shift.
        relPos = -currentState.ghostOffsets[ghost]

        # -- Value-only fit at the ghost: Band et al. 2018's decoupled
        # 0th-order (Shepard) term, `densityBand.py`'s `alpha` -- no gradient
        # block computed at all, since this scheme has none to Taylor-shift.
        props = OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.SuperSymmetric,
            operationMode=OperationDirection.FluidToGhost)

        def gather(referenceValues):
            return warpOperation(currentState, props, domain=config.domain,
                                 referenceValues=referenceValues, adjacency=adjacency)

        ones = torch.ones_like(currentState.densities)
        M = gather(ones)
        Sq = gather(currentState.densities)

        # Exact integer fluid-neighbour count, not a scale-sensitive
        # threshold on `M` itself -- `densityBand.py`'s Mb-precision fix
        # (BOUNDARY_DENSITY_PLAN.md §3 item 1), reused verbatim.
        _, _, _, nNbFluid = computeLiuMatricesWarp(
            queryPositions=currentState.positions[ghost],
            referenceParticles=currentState, referenceQuantities=currentState.densities,
            operationProperties=OperationProperties(
                kernel=config.kernel, supportMode=SupportScheme.Scatter,
                operationMode=OperationDirection.FluidToGhost),
            domain=config.domain,
            adjacency=adjacency.hashMap if isinstance(adjacency, AdjacencyList) else None)

        hasAny = nNbFluid >= 1
        Mb = M[ghost].clamp_min(0.0)
        # Regularize the Shepard ratio by adding the SAME epsilon to both
        # numerator and denominator, rather than flooring the denominator
        # alone (as this used to). A single-neighbour ghost row's weight can
        # be a genuinely tiny but internally-consistent kernel value (a
        # particle sitting almost exactly on the kernel's cutoff radius,
        # `r ~ 0.99999h` -- confirmed directly, `BOUNDARY_DENSITY_PLAN.md`
        # §9.4/§9.6): `Sq = w*rho_j` and `Mb = w` share that SAME `w`, so in
        # exact arithmetic `Sq/Mb = rho_j` regardless of how small `w` is --
        # the weight cancels. Flooring only `Mb` (the old `MbSafe =
        # Mb.clamp_min(1e-30)`) broke that cancellation: once `Mb` dropped
        # below the floor, `alpha` became `(tiny Sq) / (arbitrary floor)`,
        # not `rho_j` -- confirmed exactly (`Mb=9.6e-42` -> `alpha=9.6e-12`
        # instead of ~1.0), which is what fed the catastrophic pressure swing
        # that diverged the run. Adding the SAME `eps` to both preserves the
        # cancellation at every scale instead: `Mb=0` (no neighbours) ->
        # `alpha=rho0` exactly, matching the explicit fallback below; `Mb`
        # comparable to or below `eps` (a vanishing/underflowing weight) ->
        # `alpha` blends smoothly toward `rho0`, not a wrong near-zero value;
        # `Mb >> eps` (a healthy stencil) -> `alpha ~= Sq/Mb`, unaffected.
        # `eps = 1e-30` reuses the OLD (buggy) denominator-only floor's own
        # threshold, applied symmetrically instead -- deliberately NOT a
        # larger value like `1e-8`. A first attempt at `1e-8` fixed the
        # crash but measurably worsened sloshingTank's later behaviour
        # (earlier onset, more frequent large excursions) -- because `1e-8`
        # sits ABOVE this case's typical *non-degenerate* single-neighbour
        # scale (`Mb ~ 1e-20` at `nNbFluid=1`, confirmed empirically), it was
        # silently overriding every ordinary N=1 row's raw Shepard value
        # toward `rho0`, not just the genuinely-underflowed one -- a much
        # bigger behaviour change than intended, discarding real information
        # the OLD code (bug aside) actually used successfully. `1e-30` sits
        # far below that normal N=1 scale (negligible perturbation, matches
        # the old code's behaviour there almost exactly) while still
        # dominating genuine underflow (`Mb ~ 1e-42`) enough to fall back to
        # `rho0` instead of the wrong `9.6e-12`. `BOUNDARY_DENSITY_PLAN.md`
        # §9.6/§9.7 has the full comparison.
        _ALPHA_EPS = 1e-30
        alpha = (Sq[ghost] + _ALPHA_EPS * rho0) / (Mb + _ALPHA_EPS)

        # -- Eq. (10), corrected sign (module docstring / plan §5.1):
        # P_b = P_g + rho0 * dot(g - a_b, relPos). `a_b = 0` (static walls).
        gAll = computeGravity(currentState, config, schemeConfig, adjacency)
        g_b = gAll[bIndices]
        P_g = c0 ** 2 * (alpha - rho0)
        P_b = P_g + rho0 * torch.einsum('nu,nu->n', g_b, relPos)
        rho_b = rho0 + P_b / c0 ** 2

        # No-neighbour fallback: rest density, matching density2025.py /
        # densityBand.py's own zero-neighbour behaviour.
        rho_b = torch.where(hasAny, rho_b, torch.full_like(rho_b, rho0))
        rho_b = torch.nan_to_num(rho_b, nan=rho0, posinf=rho0, neginf=rho0)

        if _CAPTURE is not None:
            _CAPTURE.append(dict(
                boundaryUID=(currentState.UIDs[bIndices].detach().cpu().numpy()
                             if currentState.UIDs is not None else None),
                boundaryPos=currentState.positions[bIndices].detach().cpu().numpy(),
                ghostPos=currentState.positions[ghost].detach().cpu().numpy(),
                relPos=relPos.detach().cpu().numpy(),
                Mb=Mb.detach().cpu().numpy(),
                alpha=alpha.detach().cpu().numpy(),
                hasAny=hasAny.detach().cpu().numpy(),
                nNbFluid=nNbFluid.detach().cpu().numpy(),
                g_b=g_b.detach().cpu().numpy(),
                P_g=P_g.detach().cpu().numpy(),
                P_b=P_b.detach().cpu().numpy(),
                rho_b=rho_b.detach().cpu().numpy(),
            ))

        merged = currentState.densities.clone()
        merged[bIndices] = rho_b
        return merged
