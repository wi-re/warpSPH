"""Extended fluid/boundary PPE iteration for the local Band variant."""

from typing import Any
import torch
from warpSPHCore.linalg import runStationary
from ...configurations.compute import resolveCheckSchedule
from .convergence import ConvergenceCheckSchedule
from .policies import BandPressurePolicy
from .sources import pressureBoundarySource
from .pressureBoundaries import (
    bandRestVolumes,
    bandActualVolumes,
    bandRelaxation,
    bandInjectVolumes,
    bandDiagonal,
    bandWellPosedMask,
    bandVelocityDivergence,
    bandConstantModeRatio,
    bandPressureAccel,
    bandApplyOperator,
)


def solvePressureBoundaries(
    state: Any,
    config: Any,
    schemeConfig: Any,
    adjacency: Any,
    *,
    solveRows: torch.Tensor,
    fluid: torch.Tensor,
    rho0: float,
    vStar: torch.Tensor,
    warmStart: torch.Tensor,
    dt: float,
    policy: BandPressurePolicy = None,
):
    """The extended-PPE relaxed Jacobi over `p = [p_f ; p_b]`.

    Returns `(a_p, p, nIter, err)` with `a_p` the final Eq. 8 pressure
    acceleration (fluid rows only), `err` the last mean-residual value.
    """
    policy = BandPressurePolicy() if policy is None else policy
    V0 = bandRestVolumes(state, config, adjacency, rho0)
    V = bandActualVolumes(state, config, adjacency, V0, rho0)
    omega = bandRelaxation(state, V0, rho0, policy.OMEGA_FLUID)

    with bandInjectVolumes(state, V):
        diag = bandDiagonal(state, config, schemeConfig, adjacency, V, dt)
        # Drop kernel-deficient rows (near-null diagonal -- free surface, thin
        # gap) from the unknown set: they carry no meaningful pressure equation
        # and `omega / a_ii` detonates there. Held at `p = 0` instead of
        # regularised with a large diagonal shift (which over-compresses the
        # column). Recompute the source only over the rows that survive.
        rows = bandWellPosedMask(diag, fluid, solveRows)

        divVstar = bandVelocityDivergence(state, config, adjacency, vStar, rows)
        source = pressureBoundarySource(V0, V, dt=dt, divergence=divVstar)
        source = torch.where(rows, source, torch.zeros_like(source))

        # A small Tikhonov shift still helps the conditioning of what remains.
        if policy.DIAG_TIKHONOV and bool(fluid.any()):
            shift = policy.DIAG_TIKHONOV * float(diag[fluid].abs().median())
        else:
            shift = 0.0
        diagEff = diag - shift
        diagBad = (diagEff.abs() < policy.ALPHA_FLOOR) | ~rows
        invDiag = torch.where(diagBad, torch.zeros_like(diagEff), omega / diagEff.clamp(max=-1e-30))

        # Closed-domain gauge: is the constant a null mode of A? (See
        # `policy.CLOSED_DOMAIN_GAUGE`.) One extra operator application per step.
        if policy.CLOSED_DOMAIN_GAUGE == "always":
            gauge = True
        elif policy.CLOSED_DOMAIN_GAUGE == "auto":
            gauge = (
                bandConstantModeRatio(state, config, adjacency, diag, V, fluid, rows, dt)
                < policy.NULL_MODE_THRESHOLD
            )
        elif policy.CLOSED_DOMAIN_GAUGE == "off":
            gauge = False
        else:
            raise ValueError(f"Unknown policy.CLOSED_DOMAIN_GAUGE: {policy.CLOSED_DOMAIN_GAUGE!r}")

        p = torch.where(rows, warmStart, torch.zeros_like(warmStart))
        # The loop, its device-side stopping test and the one history read are `warpSPHCore.linalg.runStationary`; the error stays on the
        # device and the convergence flag is read back only at the check schedule's checkpoints (`pressureSolver.convergenceCheckSchedule`;
        # `every` reproduces the historical per-iteration reads -- `rows.any()` and the boolean-mask mean -- bitwise).
        hasRows = bool(rows.any())
        rowsIndex = rows.nonzero().squeeze(1)
        tol = float(schemeConfig.solverConfig.pressureSolver.tolerance)
        schedule = ConvergenceCheckSchedule(
            schemeConfig,
            "band2018pb",
            resolveCheckSchedule(schemeConfig.solverConfig.pressureSolver, config),
            policy.DENSITY_MIN_ITERATIONS - 1,
            policy.DENSITY_MAX_ITERATIONS,
        )  # `it + 1 >= min` is `it >= min - 1`

        def step(p, it):
            a_p = bandPressureAccel(state, config, adjacency, p, V, fluid)
            Ap = bandApplyOperator(state, config, adjacency, a_p, dt, rows)
            if shift:
                Ap = Ap - shift * p  # `(A - shift*I) p`
            p = p + invDiag * (source - Ap)
            if gauge:
                # Pin the unconstrained constant instead of clamping (a closed
                # domain has no free surface, so no tensile instability).
                p = p - torch.where(
                    rows, p.index_select(0, rowsIndex).mean().expand_as(p), torch.zeros_like(p)
                )
                p = torch.where(rows, p, torch.zeros_like(p))
            else:
                p = p.clamp(min=0.0)
            bad = diagBad | (~torch.isfinite(p)) | (p.abs() > 1e25) | ~rows
            p = torch.where(bad, torch.zeros_like(p), p)
            if hasRows:
                residual = torch.clamp(Ap - source, min=policy.RESIDUAL_FLOOR)
                err = residual.index_select(0, rowsIndex).mean()
            else:
                err = torch.zeros((), dtype=p.dtype, device=p.device)
            return p, err, None

        out = runStationary(
            step,
            p,
            maxIters=policy.DENSITY_MAX_ITERATIONS,
            tolerance=tol,
            inclusive=True,
            schedule=schedule,
        )
        p = out.x
        it = out.iterations - 1
        err = out.stats[-1] if out.stats else 0.0
        a_p = bandPressureAccel(state, config, adjacency, p, V, fluid)
    return a_p, p, it + 1, err
