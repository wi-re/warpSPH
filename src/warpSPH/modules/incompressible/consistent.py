"""Bender, Westhofen & Jeske 2023, "Consistent SPH Rigid-Fluid Coupling"
(VMV) -- the boundary treatment their constraint-based derivation produces,
as a selectable mode.

The paper defines the density constraint `C_i = rho0 - sum_j m_j W_ij -
sum_k m~_k W_ik` for *fluid particles only*, and observes that a static
boundary particle's position is constant, so `dC_i/dx_k = 0`. Three
consequences fall out, and this codebase's relationship to each is different:

- **Eq. 32 (the diagonal).** The boundary enters `||sum_j m_j gradW_ij +
  sum_k m~_k gradW_ik||^2` and *not* `sum_j ||m_j gradW_ij||^2`. That is
  exactly `BoundaryOperatorTerms.staticBoundary`'s alpha half, which this
  codebase already has (found independently from SPlisHSPlasH's
  `computeDFSPHFactor`; see `DFSPH_IMPROVEMENT_PLAN.md` Part 9).
- **Eq. 34 (the Laplacian).** The boundary term is `sum_k m~_k a^p_i .
  gradW_ik` -- the neighbour's own pressure acceleration is absent. That is
  `staticBoundary`'s operator half.
- **Eq. 33 (the pressure acceleration).** `a^p_i = -sum_j m_j (p_i/rho_i^2 +
  p_j/rho_j^2) gradW_ij - sum_k m~_k (p_i/rho_i^2) gradW_ik` -- **no boundary
  pressure value appears at all.** This codebase's symmetric pressure gradient
  reduces to exactly that whenever boundary pressures are zero, which
  `BoundaryPressureMode.plain`/`mdbcDensity` already arrange. It was
  `mdbcMlsPressure` -- Band et al.'s MLS extrapolation, which the paper is
  written against -- that departed from it; that mode is removed (pre-merge
  cleanup pass, 09-04, DFSPH_IMPROVEMENT_PLAN.md), so today only `consistent`
  itself is a live counterexample to "boundary pressures are zero".

So the operator side is already the paper's. What is genuinely different is
the *state* the boundary particles enter the solve with. The paper treats them
as "static fluid particles" at the fluid's rest density: `rho_k = rho0`, with
mass `m~_k = rho0 / sum_l W_kl` (Akinci et al.'s volume correction, `l` over
boundary neighbours only). This codebase gives them an mDBC-extrapolated
density -- which reaches 1.3+ in a compressed wall band -- and a nominal mass
`rho0 * dx^d`. Those values do reach the solve: every SPH sum in it weights a
neighbour by its apparent volume `m_j / rho_j`.

`applyConsistentCoupling` is the context manager that puts the boundary rows
into the paper's state for the duration of a solve and restores them
afterwards. It deliberately does *not* touch `computeMdbcDensity` in
`schemes/divergenceFree.py`: the extrapolated density stays available to everything
outside the pressure solve, so the A/B isolates the pressure solve alone.
"""

from contextlib import contextmanager
from typing import Any

import torch

from warpSPHCore import (OperationDirection, OperationProperties, SupportScheme,
                         WarpOperation, warpOperation)

from ...configurations import BoundaryPressureMode

__all__ = ['applyConsistentCoupling', 'akinciBoundaryMass']


def akinciBoundaryMass(particles: Any, config: Any, adjacency: Any,
                       rho0: float, scale: float = 1.0) -> torch.Tensor:
    """Akinci et al.'s boundary mass `m~_k = rho0 / sum_l W_kl`, `l` over the
    *boundary* neighbours of `k` (the paper's Section 3.3).

    `WarpOperation.Interpolate` computes `sum_l (m_l/rho_l) f_l W_kl`, so
    passing `f = rho/m` recovers the bare kernel sum `sum_l W_kl`.

    Note this correction was designed for a *one-layer* boundary sampling,
    where it makes the single layer stand in for the whole solid half-space.
    This codebase samples a five-layer band, so the layers behind the
    interface already supply that volume and the correction inflates the
    interface layer instead -- which is why it is a separate, default-off
    flag rather than part of the mode. On that band `sum_l W_kl` comes out
    large enough that `m~_k` is numerically indistinguishable from the nominal
    particle volume (`DFSPH_IMPROVEMENT_PLAN.md` Part 25), so the correction is
    effectively inert; `scale` is an explicit multiplier for callers that need
    the boundary term carried at a different weight (`dfsphReference` uses 2.0).
    """
    inv = particles.densities / particles.masses
    kernelSum = warpOperation(
        particles,
        OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Interpolate,
            supportMode=SupportScheme.Gather,
            operationMode=OperationDirection.BoundaryToBoundary),
        queryValues=inv, domain=config.domain, adjacency=adjacency)
    # `BoundaryToBoundary` returns 0 for every row that is not `kind == 1`
    # (fluid and ghost), and `rho0 / 0` is a mass large enough to destroy a
    # simulation if it ever reached a neighbour sum. It does not today --
    # `OperationDirection.AllToAll` and the default `TrueAllToToAll` both
    # exclude ghosts, and fluid rows keep their own mass -- but the value must
    # not be a landmine, so anything without a real boundary neighbourhood
    # falls back to its nominal mass.
    usable = kernelSum > (0.1 * particles.densities / particles.masses.clamp(min=1e-30))
    return scale * torch.where(usable, rho0 / kernelSum.clamp(min=1e-12), particles.masses)


@contextmanager
def applyConsistentCoupling(particles: Any, config: Any, schemeConfig: Any,
                            adjacency: Any, mode: BoundaryPressureMode):
    """Put `kind == 1` (Boundary) rows into the paper's boundary state for the
    duration of a pressure solve, and restore them on the way out. `kind == 2`
    (Ghost) rows are left alone -- they are mDBC evaluation points, not
    interacting particles.

    A no-op unless `mode is BoundaryPressureMode.consistent`, and a no-op in
    any case when there are no boundary particles.
    """
    if mode is not BoundaryPressureMode.consistent:
        yield
        return

    boundary = particles.kinds == 1
    if not bool(boundary.any()):
        yield
        return

    rho0 = schemeConfig.fluid.restDensity
    savedDensities = particles.densities
    savedMasses = particles.masses
    try:
        # "we assume that they have the same rest density rho0 as the fluid
        # since we treat these particles as static fluid particles" (Sec. 3.3).
        # `kind == 1` only -- NOT `kind != 0`: `kind == 2` ghost particles are
        # mDBC evaluation points, not interacting particles (they are excluded
        # from every neighbour sum by `OperationDirection.AllToAll`), and the
        # paper has no notion of them. Overwriting their density is inert today
        # but wrong in principle, and a landmine for any future kernel that
        # does read ghost rows.
        particles.densities = torch.where(
            boundary, torch.full_like(savedDensities, rho0), savedDensities)
        volScale = float(getattr(schemeConfig.solverConfig,
                                 'akinciBoundaryVolumeScale', 1.0))
        if getattr(schemeConfig.solverConfig, 'akinciBoundaryVolume', False):
            psi = akinciBoundaryMass(particles, config, adjacency, rho0, scale=volScale)
            # `kind == 1` only: ghost particles are an mDBC construct with no
            # counterpart in the paper, and they have no boundary neighbourhood
            # to compute a volume correction from.
            particles.masses = torch.where(particles.kinds == 1, psi, savedMasses)
        elif volScale != 1.0:
            # No Akinci substitution, but the caller still wants the boundary
            # apparent volume the solve sees reweighted -- scale the nominal
            # mass directly (`kind == 1` only). Strict no-op at the default 1.0.
            particles.masses = torch.where(
                particles.kinds == 1, savedMasses * volScale, savedMasses)
        yield
    finally:
        particles.densities = savedDensities
        particles.masses = savedMasses
