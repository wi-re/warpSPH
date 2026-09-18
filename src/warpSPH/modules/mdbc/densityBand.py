"""mDBC boundary-particle density extrapolation, Band et al. 2018-style
(`literature/band2018_mls-pressure-boundaries.pdf`, "MLS pressure boundaries
for divergence-free and viscous SPH fluids") -- an alternative to
`density2025.py`'s English et al. 2022 ghost-node extrapolation + smooth
determinant/neighbour-count ramp.

Prototyped and validated against synthetic few-particle boundary stencils in
`scripts/probe_bandMlsPressureBoundary.py` (see that script's docstring and
the conversation history in `DELTASPH_VALIDATION_PLAN.md` around the "few
particle large sheet" open item). Two structural differences from
`density2025.py`, both aimed at the same failure mode -- a hard
`where(wellConditioned, ...)` (or a ramp on a scalar that itself jumps
discontinuously across a discrete neighbour-count change) snapping a sparse
boundary stencil between a 1st-order MLS fit and a 0th-order Shepard
fallback:

  1. The fit is evaluated directly AT the boundary particle (its own fluid
     neighbours, via `OperationDirection.FluidToBoundary`) -- no ghost node,
     no separate `(r_b - r_g) . grad` Taylor-correction transport step.
     REVISED (see below): this only holds for Band's own single-layer
     (Akinci-style) boundary sampling, where every boundary particle sits
     right at the fluid interface. This codebase's boundary bands have real
     DEPTH (multiple layers into the wall) -- confirmed directly against a
     rendered dam-break frame (managua colormap, corrected marker size):
     particles more than ~1 layer deep have poor-to-nonexistent fluid
     overlap, so fitting AT their own position starves the fit of real
     neighbours exactly where Band's paper assumes it never has to. Fixed by
     querying at the GHOST NODE instead (mirrored near the surface
     regardless of the boundary particle's own depth -- the same node
     `density2025.py`'s English Eq. 12 path already uses) and Taylor-
     shifting the fitted linear field to the boundary particle via
     `ghostOffsets`/`ghostIndices`, exactly like that module. Because the
     fit is linear, evaluating it at the ghost and then shifting by
     `(r_b - r_g)` is algebraically identical to evaluating it directly at
     `r_b` -- so this keeps Band's decoupled/Tikhonov-damped system exactly
     as derived below, just anchored at a position that actually has fluid
     support at every layer.
  2. The moment system is translated by the neighbourhood's own kernel-
     weighted centroid `d_b` (computed from the SAME raw moments via the
     standard closed form `Var(w,x) = E[w x^2] - E[w x]^2/E[w]`, i.e. no
     second gather pass is needed). This decouples the 0th-order Shepard
     term `alpha_b` (exact and well-posed whenever there is at least one
     neighbour; ALGEBRAICALLY IDENTICAL to a Shepard-only fit, not a
     separately-computed fallback formula) from the `dim x dim` gradient
     block, which is inverted via a Tikhonov-damped eigendecomposition
     (smooth per-eigenvalue degradation, replacing the paper's own plain-SVD
     "safe inversion" -- the toy probe found that plain `pinv` still
     amplifies per-particle density noise catastrophically right at a
     marginal crossing, 10-20x worse than `density2025.py`'s existing ramp
     across every tested geometry; Tikhonov damping is what actually gets a
     1.5-17x improvement over the ramp instead).

No anti-attraction (`>= rho0`) clamp here, unlike `density2025.py` -- ONE
formula, no separate "fallback share" to selectively clamp. Per the plan:
validate this on the Marrone dam-break case unclamped first; only reintroduce
a clamp (and decide where it would attach -- most likely gated on the
gradient block's own conditioning rather than resurrecting a boolean
well/ill-conditioned split) if a receding-wall case actually needs it.

2D only (the `dim == 2` closed forms below); `computeMdbcDensityBand` raises
on any other dimensionality rather than silently doing the wrong thing.
"""

from typing import Any, Optional, Union

import torch
from torch.profiler import record_function

from warpSPHCore import *

from ...configurations.simulationConfig import SimulationConfig
from ._util import stateHasBoundaryParticles

__all__ = ['computeMdbcDensityBand']

#: Tikhonov damping scale for the (decoupled) gradient block's eigen-inverse,
#: in units of the block's own mean |eigenvalue| -- self-normalizing, so one
#: constant works across the wildly different neighbourhood scales a
#: multi-resolution/adaptive-support run can produce. Calibrated in
#: `scripts/probe_bandMlsPressureBoundary.py` against synthetic few-particle
#: stencils carrying ~0.3% IID density noise: lambda in [3e-3, 3e-2] all beat
#: `density2025.py`'s existing ramp there; 5e-3 sits in the middle of that
#: range.
_TIKHONOV_LAMBDA = 5e-3

#: Symmetric Shepard-ratio regularizer -- ported verbatim from
#: `english2025.py`'s `_ALPHA_EPS` fix (`BOUNDARY_DENSITY_PLAN.md` §9.4-§9.7).
#: This module had the identical asymmetric-floor bug english2025.py's had
#: pre-fix: flooring only the denominator (`MbSafe = Mb.clamp_min(1e-30)`)
#: breaks the numerator/denominator weight cancellation that should hold at
#: any scale (a single fluid neighbour has `Sx = w*x_j`, `Sq = w*rho_j`,
#: `Mb = w` all sharing the SAME `w`, so `Sq/Mb = rho_j` exactly regardless of
#: how small `w` is -- until only `Mb` is floored, at which point a
#: catastrophically underflowing `w` (e.g. `Mb ~ 1e-42`, a ghost's kernel
#: support grazing a single fluid particle almost exactly at the cutoff
#: radius) turns a should-be-~rho0 value into `(tiny Sq)/(arbitrary 1e-30
#: floor)` ~ 1e-12*rho0 instead). Same calibration reasoning applies
#: unchanged (`1e-30` reuses the old floor's own threshold, applied
#: symmetrically instead of asymmetrically -- deliberately not `1e-8`, which
#: sits above this case family's typical *non-degenerate* single-neighbour
#: `Mb` scale and would silently override real information for every
#: ordinary marginal row, not just the genuinely-underflowed ones).
_ALPHA_EPS = 1e-30

#: Debug-only: when `_DEBUG['on']` is set (a probe/diagnostic script's job,
#: never a production run), each call records the worst (|rho_b - rho0|)
#: boundary row's internals into `_DEBUG['worst']`. Not thread-safe, not for
#: production use -- a cheap hook for chasing a specific instability without
#: re-deriving the whole call from outside.
_DEBUG = {'on': False, 'worst': None}


def computeMdbcDensityBand(currentState: Any, config: SimulationConfig, schemeConfig: Any,
                           adjacency: Optional[Union[AdjacencyList, CompactHashMap]],
                           tikhonovLambda: float = _TIKHONOV_LAMBDA) -> torch.Tensor:
    if not stateHasBoundaryParticles(currentState, config):
        return currentState.densities
    dim = currentState.positions.shape[1]
    if dim != 2:
        raise NotImplementedError(
            "computeMdbcDensityBand: only 2D is implemented (the closed-form "
            "moment-decoupling below is written out for dim==2); got "
            f"dim={dim}")

    with record_function("[warpSPH] - (mdbc) - computeMdbcDensityBand"):
        rho0 = schemeConfig.fluid.restDensity
        ghost = currentState.kinds == 2
        if not bool(ghost.any()):
            return currentState.densities
        bIndices = currentState.ghostIndices[ghost]
        # r_b - r_g, same convention as density2025.py's English Eq. (12)
        relPos = -currentState.ghostOffsets[ghost]

        props = OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.SuperSymmetric,
            operationMode=OperationDirection.FluidToGhost)

        def gather(referenceValues):
            return warpOperation(currentState, props, domain=config.domain,
                                 referenceValues=referenceValues, adjacency=adjacency)

        # Positions shifted by a single global constant (the domain origin)
        # before forming second-order moments -- cheap mitigation of the
        # catastrophic-cancellation risk in `Sxx - Sx^2/M`-style closed forms
        # (raw coordinates far from the origin square to values much larger
        # than the local variance they need to resolve). A fully local
        # per-particle frame would remove this entirely but needs a custom
        # accumulation kernel rather than reusing the existing `Interpolate`
        # primitive nine times; not done here -- flag if this is promoted
        # beyond a validation prototype on larger domains than the ~3 m
        # Marrone dam break.
        origin = config.domain.min
        x = currentState.positions[:, 0] - origin[0]
        y = currentState.positions[:, 1] - origin[1]
        rho = currentState.densities
        ones = torch.ones_like(rho)

        M = gather(ones)
        Sx = gather(x)
        Sy = gather(y)
        Sxx = gather(x * x)
        Sxy = gather(x * y)
        Syy = gather(y * y)
        Sq = gather(rho)
        Sxq = gather(x * rho)
        Syq = gather(y * rho)

        # `M` (the Shepard denominator) can be many orders of magnitude below
        # a well-populated stencil's scale -- a boundary particle whose ghost
        # support just barely grazes a single fluid particle near the exact
        # kernel cutoff (q~1, W~0) gives M ~ 1e-10 against a healthy ~1e-3.
        # Gating "has real neighbours" on a continuous threshold against THAT
        # (`Mb > 1e-12`, the original version of this code) is a scale- and
        # precision-sensitive test: found by direct diagnosis on the Marrone
        # dam-break IC (nx=20, t=0) to admit exactly this near-zero-overlap
        # case as "valid", after which `Gxx = Sxx - dbx*Sx`-style closed forms
        # cancel two O(M)-scale numbers that are ALREADY near float32's noise
        # floor -- amplifying that roundoff through the (correctly, smoothly)
        # Tikhonov-damped but still nonzero-response gradient block produced
        # boundary densities up to 1.40 at rest (should be exactly 1.0
        # everywhere; verified this is pure precision, not physics: identical
        # run in float64 reproduces 1.0 to 1e-13 everywhere). Gate on an EXACT
        # integer fluid-neighbour count instead -- `computeLiuMatricesWarp`'s
        # own `neighCounts` (raw `r_ij < h` count, kind-filtered by
        # `operationMode`, no precision ambiguity) -- mirroring
        # `interpolateLiuLiu`'s neighbour-count floor rather than inventing a
        # new scale-dependent one.
        from ..liu.wp_mat import computeLiuMatricesWarp
        _, _, _, nNbFluid = computeLiuMatricesWarp(
            queryPositions=currentState.positions[ghost],
            referenceParticles=currentState, referenceQuantities=currentState.densities,
            operationProperties=OperationProperties(
                kernel=config.kernel, supportMode=SupportScheme.Scatter,
                operationMode=OperationDirection.FluidToGhost),
            domain=config.domain,
            adjacency=adjacency.hashMap if isinstance(adjacency, AdjacencyList) else None)

        hasAny = nNbFluid >= 1
        # dim+1 = 3 fluid neighbours is the minimum that can determine a 2D
        # gradient at all (matches the toy prototype's `n < 3` gate in
        # `scripts/probe_bandMlsPressureBoundary.py`); below that the
        # (rank-deficient by construction, not just ill-conditioned) 2x2
        # block is skipped entirely rather than handed to the eigensolver.
        hasGradient = nNbFluid >= 3

        Mb = M[ghost].clamp_min(0.0)
        # Denominator-only floor kept ONLY for the eigenvalue-scale reference
        # below (`mbRef`/`absoluteFloor`) -- that's a numerical-scale floor,
        # not a value-ratio, so it doesn't have a cancellation identity to
        # break. `dbx`/`dby`/`alpha` below use the symmetric `_ALPHA_EPS`
        # form instead (see that constant's docstring).
        MbSafe = Mb.clamp_min(1e-30)

        # `dbx`/`dby`: same denominator floor as `alpha`, no numerator-side
        # epsilon needed -- their numerators (`Sx`/`Sy`) already vanish along
        # with `Mb` in the no/underflowing-neighbour case, so the symmetric
        # floor's fallback is "centroid at the ghost's own position" (offset
        # 0), which is already the intended safe default here.
        dbx = Sx[ghost] / (Mb + _ALPHA_EPS)
        dby = Sy[ghost] / (Mb + _ALPHA_EPS)
        alpha = torch.where(hasAny, (Sq[ghost] + _ALPHA_EPS * rho0) / (Mb + _ALPHA_EPS),
                            torch.full_like(Mb, rho0))

        Gxx = Sxx[ghost] - dbx * Sx[ghost]
        Gxy = Sxy[ghost] - dbx * Sy[ghost]
        Gyy = Syy[ghost] - dby * Sy[ghost]
        rhsx = Sxq[ghost] - dbx * Sq[ghost]
        rhsy = Syq[ghost] - dby * Sq[ghost]

        G = torch.stack([torch.stack([Gxx, Gxy], dim=-1),
                         torch.stack([Gxy, Gyy], dim=-1)], dim=-2)   # (Nb, 2, 2)
        rhs = torch.stack([rhsx, rhsy], dim=-1)                      # (Nb, 2)

        # symmetric by construction -> eigh, not a general SVD/pinv; smooth
        # per-eigenvalue Tikhonov damping instead of a hard rtol truncation
        # (see module docstring: plain pinv still amplifies noise badly here).
        #
        # `scale` self-normalizes to THIS stencil's own eigenvalues so one
        # constant `tikhonovLambda` works across resolutions -- but that
        # self-normalization fails exactly where damping matters most: a
        # near-collinear 3-7 particle stencil (the minimum-population case
        # `hasGradient` admits at all) has BOTH eigenvalues tiny, not just
        # one, so `evals.abs().mean()` collapses right along with them and
        # the damping floor vanishes. Confirmed directly on the Marrone
        # dam-break stress config (nx=35, no PST, un-frozen): every runaway
        # sample during the first-impact collapse showed both eigenvalues
        # O(1e-8)-O(1e-10) and a `gradTerm` swinging +-1.5, producing
        # `rho_b` in [0.3, 2.9] at a single boundary row from one step to
        # the next.
        #
        # First fix tried: floor `scale` at `Mb * dx^2` (this row's own
        # Shepard weight times one particle-spacing squared). INSUFFICIENT --
        # re-broke on the same case, worse (evals down to 1e-20, rho_b up to
        # 8.5e7): `Mb` for a marginal 3-neighbour stencil can itself collapse
        # towards zero (those 3 neighbours sitting right at the kernel's
        # outer edge, W~0) for exactly the same reason a single row's own
        # eigenvalues can, so a PER-ROW floor built from that row's own `Mb`
        # inherits the identical fragility it was meant to guard against.
        # Fixed by using a floor that CANNOT collapse with any one row: the
        # median `Mb` across this call's gradient-eligible rows (>= 3
        # fluid neighbours), a robust stand-in for "what a normally-populated
        # stencil's Shepard weight looks like right now" that a minority of
        # degenerate outliers cannot drag down. Shared across every row in
        # the call, not recomputed per row -- that is the point.
        evals, evecs = torch.linalg.eigh(G)                          # (Nb,2), (Nb,2,2)
        dxRef = getattr(config, 'dx', None)
        dxRef = float(dxRef) if dxRef is not None else float(currentState.supports.mean())
        if bool(hasGradient.any()):
            mbRef = float(torch.median(Mb[hasGradient]))
        else:
            mbRef = float(MbSafe.max())
        absoluteFloor = mbRef * (dxRef ** 2)
        scale = torch.maximum(evals.abs().mean(dim=-1, keepdim=True),
                              torch.full_like(Mb, absoluteFloor).unsqueeze(-1)).clamp_min(1e-300)
        damped = evals / (evals ** 2 + (tikhonovLambda * scale) ** 2)
        Ginv = torch.einsum('nij,nj,nkj->nik', evecs, damped, evecs)
        gradC = torch.einsum('nij,nj->ni', Ginv, rhs)                # (Nb, 2)
        gradC = torch.where(hasGradient.unsqueeze(-1), gradC, torch.zeros_like(gradC))

        # Evaluate the fit AT THE GHOST, then Taylor-shift to the boundary
        # particle by `relPos = r_b - r_g` -- algebraically identical to
        # evaluating the same linear fit directly at `r_b` (see module
        # docstring), but every moment above was gathered around the ghost's
        # own (well-populated-by-construction) fluid neighbourhood rather
        # than the boundary particle's own (possibly many layers deep, poorly
        # supported) position.
        xtg = torch.stack([x[ghost] - dbx, y[ghost] - dby], dim=-1)
        xtb = xtg + relPos
        gradTerm = torch.einsum('ni,ni->n', gradC, xtb)
        rho_b_raw = alpha + gradTerm

        # Reintroduced anti-attraction guard (DELTASPH_VALIDATION_PLAN.md
        # 5.2.3's m2dbc guard, dropped from this module's first version on
        # the theory that Tikhonov damping alone would be enough). It
        # wasn't: on the Marrone dam-break stress config (nx=35, no PST,
        # un-frozen) the unclamped scheme still diverges at t~=0.42 -- a
        # feedback loop, not a single bad sample, at MODERATE neighbour
        # counts (6-12, not the 3-neighbour degenerate floor the det/eigen
        # fixes above target) -- and the trace runs specifically INTO
        # tension (rho_b 0.96 -> 0.94 -> 0.88 ...) each step, exactly the
        # attraction pathology `density2025.py`'s clamp exists for. Unlike
        # that module, there is no separately-computed "fallback share" here
        # to clamp in isolation -- one formula, not two -- so this blends
        # the SAME m2dbc-clamped fallback value in smoothly, weighted by
        # this row's own conditioning (the smallest damped eigenvalue
        # relative to the Tikhonov floor already computed above), rather
        # than a hard well/ill-conditioned switch: `wCond` -> 1 once the
        # gradient fit clears the floor comfortably (raw value trusted in
        # full, including legitimate sub-rho0 excursions under a receding
        # wall -- density2025.py's docstring's whole reason for moving the
        # clamp off the well-conditioned share), -> 0 at/under the floor
        # (falls back to the clamped Shepard-only value alone).
        floorScale = tikhonovLambda * scale.squeeze(-1)
        wCond = torch.clamp(evals.abs().min(dim=-1).values / floorScale.clamp_min(1e-300), 0.0, 1.0)
        alphaClamped = torch.clamp(alpha, min=rho0)
        rho_b = wCond * rho_b_raw + (1.0 - wCond) * alphaClamped

        rho_b = torch.where(hasAny, rho_b, torch.full_like(rho_b, rho0))
        rho_b = torch.nan_to_num(rho_b, nan=rho0, posinf=rho0, neginf=rho0)

        if _DEBUG['on']:
            dev = (rho_b - rho0).abs()
            if dev.numel():
                j = int(torch.argmax(dev))
                _DEBUG['worst'] = dict(
                    rho_b=float(rho_b[j]), rho_b_raw=float(rho_b_raw[j]), alpha=float(alpha[j]),
                    gradTerm=float(gradTerm[j]), wCond=float(wCond[j]),
                    nNbFluid=int(nNbFluid[j]), evals=evals[j].tolist(),
                    xtb=xtb[j].tolist(), pos=currentState.positions[int(bIndices[j])].tolist())

        merged = currentState.densities.clone()
        merged[bIndices] = rho_b
        return merged
