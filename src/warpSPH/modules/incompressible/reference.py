"""Local DFSPH pressure solve (the composed reference variant).

The divergence equation uses opDt=dt and a source with units 1/time;
omni uses opDt=dt**2 and a dimensionless source. Do not interchange them.
"""

from typing import Any
import torch
from warpSPHCore import (
    GradientScheme,
    OperationProperties,
    SupportScheme,
    WarpOperation,
    warpOperation,
)
from warpSPHCore.linalg import runStationary
from ...configurations.compute import resolveCheckSchedule
from .convergence import ConvergenceCheckSchedule
from .wallPressure import wallPressureExtrapolation
from .wp_dfsph_factor import computeDFSPHFactor
from .sources import referenceSource
from .operators import pressureAcceleration as _pressureAccel

__all__ = ["computeReferenceFactor", "referenceDensityRate", "solveReference"]
from ..momentum.incompressible import computeMomentumIncompressible
from ..util import countNeighbors


def computeReferenceFactor(
    state: Any, config: Any, schemeConfig: Any, adjacency: Any, wall: Any = None
) -> torch.Tensor:
    """Local stiffness denominator for physical pressure, negated and floored.

    `computeDFSPHFactor` includes ALL non-ghost neighbor squares for fluid
    queries, including static boundary neighbors. This deliberately differs
    from upstream DFSPH's fluid-only back-reaction sum. Its volume-only scaling is
    also distinct from the exact composed IISPH diagonal on nonuniform states.
    The volume-only denominator acts on normalized pressure p/rho0. The force
    below consumes physical p, so divide by rho0 to match its units. Omitting
    this factor makes the pressure iteration depend on density units.
    The analytic correction adds 2 S.G + |G|^2 without a 1/rho weight.
    """
    apparentArea = state.masses / state.densities
    diag = computeDFSPHFactor(state, config, schemeConfig, adjacency, apparentVolumes=apparentArea)
    if wall is not None:
        from ...modules.analyticBoundary import wallAlphaCorrection

        gradSum = warpOperation(
            state,
            OperationProperties(
                kernel=config.kernel,
                operation=WarpOperation.Gradient,
                gradientMode=GradientScheme.Naive,
                supportMode=SupportScheme.Gather,
            ),
            queryValues=apparentArea * state.densities / state.masses,
            domain=config.domain,
            adjacency=adjacency,
        )
        diag = diag - wallAlphaCorrection(
            wall, gradSum, torch.ones_like(state.densities)
        )  # wallAlphaCorrection is the negative IISPH-diagonal term: -(2 S.G + |G|^2) / rho
    factor = torch.clamp(-diag, max=-1e-8)
    rho0 = schemeConfig.fluid.restDensity
    return factor if rho0 == 1.0 else factor / rho0


def referenceDensityRate(
    state: Any,
    config: Any,
    schemeConfig: Any,
    adjacency: Any,
    advectionVelocities: torch.Tensor,
    wall: Any = None,
    bodyVelocity: bool = False,
) -> torch.Tensor:
    """`Drho/Dt` from the same scatter-divergence operator both `divergenceFree`
    solvers form their source from (`computeMomentumIncompressible`, `-rho0 div
    v`) -- so this reference inherits that operator rather than adding a second
    convention. `wall` (analytic walls): the wall's flux `-rho0 sum_b (f_b - f) . G_b`
    (`wallDivergence`), the body velocity `f_b` with `bodyVelocity` (a velocity field), 0 for an
    acceleration (a static wall takes no reaction)."""
    out = computeMomentumIncompressible(
        currentState=state,
        config=config,
        schemeConfig=schemeConfig,
        adjacency=adjacency,
        advectionVelocities=advectionVelocities,
    )
    if wall is not None:
        from ...modules.analyticBoundary import wallDivergence

        out = out - schemeConfig.fluid.restDensity * wallDivergence(
            wall, advectionVelocities, bodyVelocity=bodyVelocity
        )
    return out


def solveReference(
    state: Any,
    config: Any,
    schemeConfig: Any,
    adjacency: Any,
    *,
    vEnter: torch.Tensor,
    rho0: float,
    warmStart: torch.Tensor,
    opDt: float,
    solverCfg: Any,
    minIters: int,
    fluidMask: torch.Tensor,
    mode: str,
    surfaceMask: torch.Tensor = None,
    warmStartDamped: bool = False,
    wallPressure: str = None,
):
    """DFSPH pressure Jacobi (SPlisHSPlasH `pressureSolveIteration` /
    `divergenceSolveIteration`), the LINEAR form (Part 28). The earlier
    re-summed fixed point recomputed `Drho/Dt` from the trial velocity
    `v* = vEnter + dt*a_p` each iteration, which is nonlinear in `p` and
    step-size-sensitive (Part 27: the faithful factor regressed `staticBlob`
    through exactly this). SPlisHSPlasH instead solves the *linear* system
    `A p = s` with a Jacobi iteration, so this now mirrors it:

      * source `s`, computed once from `vEnter` (the velocity after the
        non-pressure forces), not re-summed -- in SPlisHSPlasH's convention
        (see the sign notes below):
          `mode == 'density'`:    `s = 1 - rho/rho0 - dt*Drho/Dt(vEnter)/rho0`
            (= `1 - rho/rho0 + h div v`, their `computeDensityAdv` fed to
            `s = 1 - densityAdv`);
          `mode == 'divergence'`: `s = -Drho/Dt(vEnter)/rho0`
            (= `+div v`, their `computeDensityChange` fed to `s = -...`).
      * each iteration: `a_p = accel(p)`; `aij_pj = Drho/Dt(a_p)/rho0 * opDt`
        (= `-div(a_p) * opDt`, their `compute_aij_pj` with `*= h^2` / `*= h`);
        `p = max(p - 0.5*(s - aij_pj)/(opDt*sum_grad_p_k), 0)` -- the 0.5 is
        SPlisHSPlasH's fixed relaxation, *not* the residual-minimizing step
        Part 26 found blows up along free-surface near-null directions.

    Sign notes (Part 28, measured not derived): their `delta` operator
    (difference-form `V_i sum (v_i - v_j) gradW`) is the NEGATIVE of the
    continuum divergence, while this codebase's scatter Divergence (inside
    `_drhodt = -rho0 * div`) IS the continuum one (probe: a `div = +1` field
    gives `_drhodt ~= -1.0` in the bulk). Their `factor =
    1/(sum_grad_p_k * h^k) > 0` (sum of squared kernel-gradient norms) and
    both solves iterate `p -= 0.5*(s - aij_pj)*factor`, so with
    `invDiag = 1/(opDt*sum_grad_p_k) > 0` the step here is the same `p -=`
    for both modes. Getting any of the three signs (source, aij_pj, step)
    wrong is a DIVERGING iteration (spectral radius > 1): the first draft
    (sign-flipped `aij_pj` and the naive `p -=` step) doubled the DF pressure
    every iteration.

    `aij_pj` reuses `_drhodt` (the same scatter-divergence operator as the
    source, applied to the acceleration field), so no new kernel.
    Warm-started from `warmStart` (the carried kappa). Returns `(a_p, p,
    nIters, err)`, where `err` is the DFSPH-style one-sided compression-only
    metric: `residuum = min(resid, 0)` per particle (the DF solve zeroes it
    -- and the DF source and warm start -- for particles with < 7 (2D) /
    < 20 (3D) neighbours; the CD solve has no such guard), then `err =
    rho0 * mean(-residuum)` over the fluid. Note the tolerance calibration
    differs from the reference (its eta is a percent of rho0, scaled by 1/h
    in the DF solve); the config tolerances here are kept as-is.

    The former `iisph` wrapper called this density mode. Native IISPH now
    lives in `modules.iisph`. Pinned SPlisHSPlasH IISPH uses rest-volume
    continuity, its separate density-dependent coefficients, half the
    previous pressure, omega=0.5, and a signed average on pressure-active
    rows evaluated with new diagonal / old off-diagonal iterates. Its
    clamp alone does not specify the solve policy.

    `surfaceMask` (Part 30, step 3; divergence solve only): rows the
    (dilated) `detectFreeSurface` flags are held at `kappa^v = 0` for the
    whole solve — source / warm start / metric residuum zeroed and the
    pressure pinned to 0 at every iteration, so the returned field (and
    the warm start it carries into the next step) is 0 on those rows. The
    pin is Part 26's gauge re-run under the linear solve; the
    source/warm-start/metric zeroing mirrors the reference's deficiency
    guard so the pinned rows cannot hold the one-sided metric above
    tolerance (which would cost the solve its early exit). Never applied
    to the constant-density solve — Part 26 measured that mask there as a
    bad trade (sub-surface over-compression).

    `warmStartDamped` (Part 31): seed with the reference's
    `USE_WARMSTART` / `USE_WARMSTART_V` formula instead of the full carry:
    `0.5*min(warmStart, cap)/opDt`, gated on the one-sided source being
    negative (compressed), zero otherwise. `warmStart` is then the
    dt-scaled STORED field (CD: `p*dt**2`, DF: `p*dt` — the reference
    multiplies the solved pressure by `h**k` at the end of each step so
    the stored value is dt-invariant), and `opDt` (`dt**2` CD / `dt` DF)
    converts it back to pressure units. Caps in stored units
    (re-verified in TimeStepDFSPH.cpp): CD 2.5e-4, DF 0.5.
    """
    dt = opDt if mode == "divergence" else opDt**0.5  # opDt is dt or dt**2
    # SPlisHSPlasH's fixed relaxation is 0.5 -- measured here (Part 29,
    # `probe_dfsphReferenceContraction.py`, hydrostaticColumn nx=32) that 0.5
    # is OUTSIDE this composed operator's Jacobi convergence window: the
    # step-1 divergence solve grows ~1.2x/iteration in its asymptotic phase
    # (spectral radius > 1) while 0.3 decays in all four (step, mode) states
    # and 0.1/0.05 regrow late. The window is ~[0.2, 0.35]; 0.3 is the
    # reference value's nearest stable neighbour. Native IISPH also runs
    # 0.5 in 2D; this composed-operator measurement does not justify changing
    # that distinct method's relaxation.
    omega = 0.3
    maxIters = solverCfg.maxIterations
    tol = solverCfg.tolerance
    # |diag|: opDt * |computeDFSPHFactor|  (_factor returns <= 0).
    from ...modules.analyticBoundary import resolveWall

    wall = resolveWall(state, config, schemeConfig, adjacency)
    invDiag = 1.0 / (opDt * (-computeReferenceFactor(state, config, schemeConfig, adjacency, wall)))

    # Fixed source, computed once from vEnter (the linear form), in
    # SPlisHSPlasH's sign convention. Two operator facts fix the signs:
    #   (1) `referenceDensityRate(v) = -rho0 * div_std(v)` -- this codebase's scatter
    #       Divergence IS the continuum divergence (positive for expansion),
    #       so `_drhodt > 0` for compression.
    #   (2) SPlisHSPlasH's `delta` operator (difference-form
    #       `V_i sum (v_i - v_j) gradW`) is the NEGATIVE of the continuum
    #       divergence, so their sources/aij_pj read (div_std = div):
    #   density:    s = 1 - rho/rho0 + h div(v)  = 1 - rho/rho0 - dt*referenceDensityRate(v)/rho0
    #   divergence: s = +div(v)                  = -referenceDensityRate(v)/rho0
    drhodtEnter = referenceDensityRate(
        state, config, schemeConfig, adjacency, vEnter, wall, bodyVelocity=True
    )
    source = referenceSource(state.densities, drhodtEnter, rho0=rho0, dt=dt, mode=mode)

    # SPlisHSPlasH's particle-deficiency guard: the DF (divergence) solve is
    # "not performed" for a particle with too few neighbours -- < 7 in 2D,
    # < 20 in 3D -- which concretely means its source is zeroed in the setup
    # (both warm-start branches then start it from p = 0) and its residuum is
    # zeroed in the metric; the CD solve has no such guard. The pressure
    # UPDATE still runs for those particles (their aij_pj is relaxed toward 0,
    # not their source). The count is the fluid+boundary neighbour count
    # (`AllToAll` excludes ghost references, matching their per-point-set sum)
    # and is evaluated ONCE: the Verlet list and positions are fixed during
    # the solve.
    deficient = torch.zeros_like(fluidMask)
    if mode == "divergence":
        guardThreshold = 7 if config.domain.dim == 2 else 20
        nNeighbours = countNeighbors(state, config, schemeConfig, adjacency)
        if wall is not None:
            # an analytic wall is not a particle: it counts as the neighbours it replaces, its completeness `lambda` times the bulk neighbour count (a boundary-particle wall row has them in `countNeighbors`)
            bulk = fluidMask & (state.densities > 0.99 * rho0)
            nBulk = (
                nNeighbours[bulk].float().median()
                if bool(bulk.any())
                else nNeighbours[fluidMask].float().median()
            )
            nNeighbours = nNeighbours + (wall.lam.sum(0) * nBulk).to(nNeighbours.dtype)
        deficient = fluidMask & (nNeighbours < guardThreshold)
    # Part 30 (step 3): the free-surface gauge rows (the dilated
    # detectFreeSurface mask, divergence solve only). They are held at
    # kappa^v = 0 for the whole solve, so they join the reference-deficient
    # rows in the source / warm-start / metric zeroing, and the pressure
    # update is additionally pinned back to 0 on them each iteration.
    pin = None if (mode != "divergence" or surfaceMask is None) else (surfaceMask & fluidMask)
    exempt = deficient if pin is None else (deficient | pin)
    if mode == "divergence":
        source = torch.where(exempt, torch.zeros_like(source), source)

    if warmStartDamped:
        # Part 31: the reference's `USE_WARMSTART` / `USE_WARMSTART_V` seed.
        # `warmStart` holds the dt-scaled stored field (CD: p*dt**2, DF:
        # p*dt), so `0.5*min(stored, cap)/opDt` is in pressure units with
        # opDt = dt**2 (CD) / dt (DF). The gate is the reference's
        # `densityAdv > 1` (CD) / clamped `delta > 0` (DF) in this code's
        # sign convention -- `source = 1 - densityAdv` / `source = -delta` --
        # evaluated AFTER the exemption zeroing, so deficient/pinned rows
        # seed 0 exactly as the reference's zeroed densityAdv does.
        cap = 2.5e-4 if mode == "density" else 0.5
        p = torch.where(
            fluidMask & (source < 0.0),
            0.5 * warmStart.clamp(max=cap) / opDt,
            torch.zeros_like(warmStart),
        )
    else:
        p = torch.where(fluidMask, warmStart, torch.zeros_like(warmStart))
        if mode == "divergence":
            # The reference's deficient DF particles start from p = 0 (their
            # densityAdv = 0 gates both warm-start and guess branches to 0),
            # not from the carried kappa^v; the gauge's pinned rows do the
            # same.
            p = torch.where(exempt, torch.zeros_like(p), p)
    # The loop, its device-side stopping test and the one history read are `warpSPHCore.linalg.runStationary`; the error stays on the device and
    # the convergence flag is read back only at the check schedule's checkpoints (`solverCfg.convergenceCheckSchedule`; `every` reproduces the
    # historical per-iteration reads -- three syncs each: `fluidMask.any()`, the boolean-mask mean and the float -- bitwise).
    hasFluid = bool(fluidMask.any())
    fluidIndex = fluidMask.nonzero().squeeze(1)
    schedule = ConvergenceCheckSchedule(
        schemeConfig,
        f"dfsphReference.{mode}",
        resolveCheckSchedule(solverCfg, config),
        max(0, minIters - 1),
        maxIters,
    )  # `it + 1 >= minIters` is `it >= minIters - 1`

    def step(p, it):
        pIn = (
            wallPressureExtrapolation(state, config, adjacency, p, fluidMask, mode=wallPressure)
            if wallPressure
            else p
        )
        a_p = _pressureAccel(state, config, adjacency, pIn, fluidMask, wall, rho0)
        # SPlisHSPlasH's `aij_pj = delta_{a_p} = -div_std(a_p)`, scaled by
        # opDt (h^2 CD / h DF) to match their `aij_pj *= h^2` / `*= h`.
        # `referenceDensityRate(a_p) = -rho0 div_std(a_p)`, so `aij_pj = +referenceDensityRate(a_p)/rho0 * opDt`.
        # (The OLD `-referenceDensityRate(...)` was the sign-flipped operator, which is what
        # made the naive `p -=` update the diverging iteration.)
        ap = referenceDensityRate(state, config, schemeConfig, adjacency, a_p, wall) / rho0 * opDt
        resid = source - ap
        # SPlisHSPlasH's `factor = 1/(sum_grad_p_k * h^k) > 0` (sum_grad_p_k is
        # a sum of squared kernel-gradient norms), and both solves iterate
        # `p -= 0.5*(s - aij_pj)*factor` (their matrices are negative in this
        # convention: positive p -> aij_pj < 0). `invDiag > 0` == |factor|, so
        # the same step is `p -= omega*(s-ap)*invDiag` for BOTH modes.
        p = (p - omega * resid * invDiag).clamp(min=0.0)
        p = torch.where(fluidMask, p, torch.zeros_like(p))
        if pin is not None:
            # Hold the gauge's surface rows at 0 INSIDE the solve (Part 30):
            # they are not unknowns, so their Jacobi update is discarded and
            # the interior iterates against a zero Dirichlet row set.
            p = torch.where(pin, torch.zeros_like(p), p)
        if hasFluid:
            # DFSPH-style metric (not native IISPH's signed pressure-active
            # average): ONE-SIDED, compression-only --
            # `residuum = min(resid, 0)` per particle, then `err = rho0 *
            # mean(-residuum)` over the fluid. Under-compressed particles
            # (resid > 0, which the one-sided pressure cannot fix) contribute
            # nothing, so a solve with nothing left to correct exits early
            # instead of running to its cap as the two-sided `mean|resid|`
            # would. The DF kernel zeroes the residuum of the deficient
            # particles (and of the gauge's pinned rows, Part 30); the CD
            # kernel has no such guard.
            residuum = torch.minimum(resid, torch.zeros_like(resid))
            if mode == "divergence":
                residuum = torch.where(exempt, torch.zeros_like(residuum), residuum)
            err = rho0 * (-residuum).index_select(0, fluidIndex).mean()
        else:
            err = torch.zeros((), dtype=resid.dtype, device=resid.device)
        return p, err, None

    out = runStationary(step, p, maxIters=maxIters, tolerance=tol, schedule=schedule)
    p = out.x
    it = out.iterations - 1
    err = out.stats[-1] if out.stats else 0.0

    pIn = (
        wallPressureExtrapolation(state, config, adjacency, p, fluidMask, mode=wallPressure)
        if wallPressure
        else p
    )
    a_p = _pressureAccel(state, config, adjacency, pIn, fluidMask, wall, rho0)
    return a_p, p, it + 1, err
