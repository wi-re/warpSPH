"""Reference DFSPH step -- Bender & Koschier 2015/2017, matching SPlisHSPlasH's
`TimeStepDFSPH.cpp` as closely as a *composed* implementation (built from the
existing warp operators, not dedicated kernels) allows.

Why this exists
---------------
`IncompressibleSPHScheme.divergenceFree` (`schemes/dfsph.py`) is Cornelis et
al.'s VD+PS step, not DFSPH: it enforces `div v = 0` with a projection and then
applies the *density-invariance* correction as a momentum-neutral **position
shift** (`x += dt**2 a_p`, `DFSPH_IMPROVEMENT_PLAN.md` 1.2/1.3). A position
shift cannot sustain a body-force balance, so that scheme cannot hold a
quiescent hydrostatic column (Part 23) -- the DF projection's source is
identically zero for a uniform gravity velocity, and the lagged shift cycle
that is left over is an amplifier.

DFSPH proper applies **both** corrections to the *velocity*, as warm-started
pressure impulses `v -= dt a_p(kappa)`. That is what lets the pressure field
accumulate the hydrostatic profile and hold the column. This module is that
scheme, kept deliberately minimal and self-contained: no VD projection, no
Eq.-17 position-shift/resample, no free-surface gauge, no damped warm
start, and no XSPH velocity filter by default (the Part 30
`FREE_SURFACE_GAUGE`, the Part 31 `DAMPED_WARM_START`, and the Part 32
`XSPH_FLUID_EPSILON` / `XSPH_BOUNDARY_EPSILON` toggles are all off), no
`minShift`, no `rhoStar` 0.9 clamp.

STATUS (troubleshooting artifact, not a landed solver). Structurally complete
and it demonstrates the thesis directly: with the velocity warmstart + a
one-sided constant-density solve, `hydrostaticColumn` holds the *exact*
hydrostatic pressure gradient (`dp/dy` ratio ~ 1.0) and `|v| < 1` for the first
~10-15 steps, where `divergenceFree` NaNs by step 6. `impact` (dynamic
free-surface collision, no walls) is stable. The Part 29 linear solve
(one-sided metric + 2D <7-neighbour guard + omega 0.3 + reference budget 100)
killed the ratchet: `hydrostaticColumn` now runs to ~1100 steps with bounded
pressure and a bounded post-slump slosh (|v|max ~1.3-1.7), and the
`staticBlob` A/B recovered (Part 27's 70.9/inf -> max |v| 1.15/1.28). A
late-time (t ≈ 1.1 s, step ~1150) free-surface degradation failure hits 2 of
3 1500-step runs (surface rho_min 0.6 -> 0.14, then blowup or collapse into
a uniform rho-0.139 soup with inf velocities). Part 30 re-ran step 3's
free-surface gauge (the `FREE_SURFACE_GAUGE` toggle: hold `kappa^v` = 0 on
`detectFreeSurface`-flagged rows, pressure pinned to 0 at every iteration of
the divergence solve) under this linear solve -- a measured negative (the
degradation's onset is unchanged, the surface degrades deeper and never
recovers, the slosh rises ~30-40%), so the toggle ships off. Part 31
re-ran the reference's damped warm start (the `DAMPED_WARM_START` toggle:
seed each solve with `0.5*min(carried, cap)/dt**k` gated on the row being
compressed, the carrier dt-scaled, the step-1 IC pressure zeroed) against
the full-kappa carry -- also a measured negative (onset unchanged, surface
depth mildly favourable at n=2, ~5x the CD iterations), and it exposed that
the baseline's IC seed self-destructs at step 1 (the two forced CD
iterations drive the hydrostatic seed to 0). The toggle ships off; the
late-time degradation now survived three levers (Parts 26, 30, 31). Part 32
resolved the decision point with a run video: the post-slump motion is the
classic free-slip effect (particles slide freely down the walls, deflect
along the floor, set up two counter-rotating vortices), and the code
confirms the wall is *exactly* free-slip (the deltaSPH viscosity term is
normal-only in both the alpha and the nu branch, and the case runs
`nu = 0`), so nothing decays the slosh. The reference's XSPH velocity
filter is now the `XSPH_FLUID_EPSILON` / `XSPH_BOUNDARY_EPSILON` toggles
(default 0.0): wall-only drag (boundary epsilon = 0.1) is the first lever
to hold the late-time surface (rho_min ends ~0.5 vs ~0.23 baseline) but
does not decay the slosh, and fluid-only XSPH is a measured negative
(earlier, deeper degradation; diverges ~step 1200). The remaining target
is the slosh's origin, the initial collapse.
The runner's divergence check is now `~isfinite` (not `isnan`), so the
inf-velocity soup reports `diverged=True`. It is **not** yet a landed
general scheme.

Harden-track progress (`DFSPH_IMPROVEMENT_PLAN.md` Parts 24-25):

- **Step 1 (wall boundary force) -- done.** The wall-adjacent kappa runaway on
  `hydrostaticColumn` (`kappa_max` 5 -> 130 -> clamp, `|v|max` ~100 by step 25)
  came from the boundary term in `A p` being carried at ~half the weight it
  needs on this codebase's five-layer `BOUNDED_BAND`, where the Akinci volume
  `rho0 / sum_l W_kl` is numerically the nominal particle volume and so
  corrects nothing. `akinciBoundaryVolumeScale = 2.0` (set below) bounds
  `kappa_max` at 6.81 and holds `|v|max` < 1 through 25+ steps.
- **Steps 3-4 (free-surface gauge, contractive divergence solve).** Step 3's
  gauge was explored under the old nonlinear solver (Part 26: worse slump,
  deferred) and re-run under the Part 29 linear solve in Part 30 -- a
  measured negative (onset unchanged ~step 300-400, deeper unrecovered
  surface, +30-40% slosh), so the `FREE_SURFACE_GAUGE` toggle ships off.
  Step 4 was re-attempted as the *linear* SPlisHSPlasH Jacobi in Part 28:
  the re-summed fixed point was
  replaced with the linear fixed-source Jacobi (the structure the faithful
  factor of Part 27 is derived for); after fixing the source/`aij_pj`/step
  sign conventions against the reference source, the physics was right but
  the iteration did not contract (CD oscillated at the 64-iter cap, DF
  diverged by step 2; NaN at step 6). Part 29 closed it: the reference's
  one-sided compression-only convergence metric and its 2D <7-neighbour
  deficiency guard (zeroing the DF source and warm start) were adopted; a
  relaxation sweep (`probe_dfsphReferenceContraction.py`) showed the
  reference's omega = 0.5 lies OUTSIDE this composed operator's Jacobi
  window (~[0.2, 0.35] measured) and it was set to omega = 0.3; the
  iteration budget was set to the reference's 100. `hydrostaticColumn`
  (nx=32) runs `diverged=False` for hundreds of steps: every solve converges
  (2-100 iterations), pressures stay bounded (CD kappa <= ~11, DF <= ~10),
  and |v|max settles to a bounded post-slump slosh (~1.3-1.7) instead of the
  old ratchet (|v|max 0.01 -> 1.76 -> 1732 -> NaN). The column's initial
  slump is the one-sided pressure responding to an initial state with no
  standing over-compression (rho <= rho0 everywhere), not an instability.
  The `staticBlob` A/B (Part 27's regression) recovered with the linear
  solve (70.9/inf -> 1.15/1.28 max |v|). At step ~1150 (t ≈ 1.1 s) a
  free-surface degradation failure hits 2 of 3 1500-step runs (surface
  rho_min 0.6 -> 0.14, then blowup or a uniform rho-0.139 soup with inf
  velocities); Part 30's gauge re-run did not close it, and Part 31's
  reference damped warm start (`DAMPED_WARM_START` toggle, default off)
  did not either (onset unchanged, depth mildly favourable at n=2, ~5x
  the CD iterations). Part 32 resolved the decision point with a run
  video: the post-slump motion is the classic free-slip effect (two
  counter-rotating vortices fed by wall sliding + floor deflection), the
  code confirms the wall is exactly free-slip (normal-only viscosity
  term, `nu = 0`), and the reference's XSPH velocity filter
  (`XSPH_FLUID_EPSILON` / `XSPH_BOUNDARY_EPSILON`, default 0.0) is the
  first lever to hold the late-time surface (wall-only drag, boundary
  epsilon = 0.1: rho_min ends ~0.5 vs ~0.23) -- without decaying the
  slosh, and with fluid-only XSPH a measured negative (diverges ~step
  1200). The remaining target is the slosh's origin, the initial
  collapse.

Per-step algorithm (SPlisHSPlasH order):

 1. neighbourhood, summation density (boundary particles included)
 2. extend the boundary velocity field per `BCType` (`computeBoundaryVelocities`);
    non-pressure accelerations -- gravity + physical viscosity
    (`computeVelocityDiffusion`, `viscidNu`) + forcing + XSPH (off by
    default) -- into `v` : `v += dt a_nonp`
 3. **constantDensitySolver** (`correctDensityError`): warm-start `kappa` from
    the carried field; relaxed-Jacobi solve of `A p = rho0 - rho*` with the
    one-sided (compression-only) source; `v += dt a_p`
 4. `x += dt v`
 5. rebuild neighbourhood + density on the new positions
 6. **divergenceSolver** (`correctDivergenceError`): relaxed-Jacobi solve of
    `A p = -Drho/Dt`, one-sided; `v += dt a_p`
 7. carry `kappa` for the next step's warm start

Boundary: static particles enter both solves as "static fluid" at `rho0` with
Akinci et al. rest-density volumes (`applyConsistentCoupling` +
`akinciBoundaryVolume`), which is what SPlisHSPlasH runs the hydrostatic test
with. They contribute to the fluid density and to the `i`-side pressure force
but take no reaction (`BoundaryOperatorTerms.staticBoundary`). NOTE: this reuses
the case's existing multi-layer, non-periodic boundary band, so there is no
thin single layer sitting on a periodic edge; a genuine one-layer Akinci
sampling would need an oversized domain or periodic interactions disabled to
avoid wraparound artifacts.

The step does its own integration and hands the integrator a no-op update;
`DFSPHReferenceSystem` (systems/incompressible.py) copies the advanced fields
across in `finalize` and skips the VD+PS machinery entirely.
"""

from __future__ import annotations

from typing import Any

import torch

from warpSPHCore import (OperationProperties, SupportScheme, WarpOperation,
                         buildVerletList, warpOperation)

from ..configurations import BoundaryPressureMode
from ..modules.boundaryConditions import (computeForcing, enforceDirichlet,
                                          enforceUpdates)
from ..modules.deltaSPH import computeVelocityDiffusion
from ..modules.density import computeDensities
from ..modules.gravity import computeGravity
from ..modules.incompressible.consistent import applyConsistentCoupling
from ..modules.incompressible.wallPressure import wallPressureExtrapolation
from ..modules.incompressible.wp_dfsph_factor import computeDFSPHFactor
from ..modules.mdbc import computeBoundaryVelocities
from ..modules.momentum.incompressible import computeMomentumIncompressible
from ..modules.pressure.iisph import computePressureAccelIISPH
from ..modules.surfaceDetection import detectFreeSurface
from ..modules.util import countNeighbors
from ..systems.incompressible import IncompressibleSystemUpdate

__all__ = ['dfsphReference_step', 'iisph_step']

# Part 30 (harden-track step 3): the free-surface `kappa^v` gauge, re-run
# under the Part 29 linear solve. When True, the divergence solve holds
# `kappa^v` at 0 on the rows the case's own (dilated) `detectFreeSurface`
# flags: their DF source, warm start, and metric residuum are zeroed and
# their pressure is pinned to 0 at every iteration, so the carried field
# that warm-starts the next step is 0 on those rows. DF solve only — Part
# 26 measured that masking the constant-density solve the same way
# over-compresses the sub-surface layer. Default False = the Part 29
# baseline; the probes (`--gauge`) toggle it for the A/B.
FREE_SURFACE_GAUGE = False

# Part 31: the reference's damped warm start (SPlisHSPlasH
# `USE_WARMSTART` / `USE_WARMSTART_V`), tested against this scheme's full
# kappa carry (the Part 29/30 baseline). The reference does not carry the
# solved pressure across steps as-is: it stores `p*h**2` (constant-density)
# / `p*h` (divergence) -- dt-invariant -- and seeds the next solve with
# `0.5*min(stored, cap)/h**k`, GATED on the row being compressed (CD:
# `densityAdv > 1`; DF: clamped `delta > 0`; both are "the one-sided source
# is negative" in this code's sign convention), zero otherwise. Caps in
# stored units (re-verified in TimeStepDFSPH.cpp, 08-30): CD 2.5e-4,
# DF 0.5. Default False = the full carry; the probes (`--warmStart`)
# toggle it for the A/B.
DAMPED_WARM_START = False

# Part 32: the reference's XSPH velocity filter (SPlisHSPlasH `XSPH.cpp`),
# the only tangential-stress mechanism the scheme family has. The deltaSPH
# viscosity term (`wp_viscosityDelta.py`) projects each pair's relative
# velocity onto the pair axis and clamps it to the approaching case --
# purely normal, compression-only, in BOTH the `alpha` and the `nu`
# branch -- so at this case's `nu = 0` / `alpha = 0.01` there is no
# tangential stress anywhere: the wall is exactly free-slip and the bulk
# vorticity is undamped. Measured consequence: the column's post-slump
# slosh is a wall-fed counter-rotating vortex pair that the pressure solve
# can bound but nothing can decay, and the pair keeps working the free
# surface (the late-time degradation Parts 26/30/31 could not close). The
# reference's XSPH is `a_i -= (1/h) eps sum_j (m_j/rho_j) (v_i - v_j)
# W(x_i - x_j)` over fluid AND static-boundary neighbours (its boundary
# branch uses the boundary volume; in this codebase the boundary's
# `m/rho` IS that volume), applied as a non-pressure acceleration -- the
# boundary's v = 0 is what makes it a wall drag. Coefficients default 0.0
# = the reference default (no shipped SPlisHSPlasH scene enables XSPH
# either); the probes' `--xspH` / `--xspHBoundary` toggle them for the A/B.
XSPH_FLUID_EPSILON = 0.0
XSPH_BOUNDARY_EPSILON = 0.0

# Physical (laminar) viscosity for this scheme family is the ordinary path:
# `computeVelocityDiffusion` (already called in step 2, driven by
# `schemeConfig.diffusionParams.viscidNu` / `inviscid`) plus the boundary
# velocity field extended by `computeBoundaryVelocities` (also step 2, driven
# by each boundary region's `BCType` -- `freeSlip` reflection / `noSlip`
# mirror / `extended` MLS -- exactly as `schemes/deltaSPH.py` and
# `schemes/dfsph.py` do). Part 39's `_physViscosity` + `PHYS_VISCOSITY_*`
# module globals (a hand-rolled Brookshaw Laplacian and a hand-rolled Adami
# no-slip mirror) duplicated that machinery and were removed; the wall no-slip
# is `BCType.noSlip` on the boundary region (`hydrostaticColumn`'s `wallBC`
# param). See `DFSPH_FINDINGS.md` 1.14 / Part 39.

# Part 33: drop the divergence-free pass and run only the constant-density
# solve as a velocity impulse -- i.e. plain IISPH ([I], Ihmsen et al. 2014,
# Alg. 1). The DF Jacobi has been the fragile component (Parts 26/28/29, the
# Part 33 rest-density-calibration blow-up), and `hydrostaticColumn` is
# IISPH's own benchmark, so this both isolates the instability and stands as
# the missing [I] baseline next to `divergenceFree` (VD+PS) and the full
# DFSPH path. Default False = the two-solve DFSPH order.
SKIP_DIVERGENCE_SOLVE = False

#: Per-iterate wall-pressure closure (Part 41), the same fix
#: `omniIncompressible` ships by default. The composed Jacobi puts the wall
#: into `alpha` only (`applyConsistentCoupling`'s Akinci band) and runs the
#: pressure accel at boundary `p == 0`, so the near-wall iteration matrix is
#: inconsistent and does not contract at a wall band (Part 41: omniSPH's
#: `densitySolve` recomputes an MLS wall pressure every iterate; this port did
#: not). When set (`'shepard'` / `'mls'`), every `_jacobiSolve` iterate refills
#: the `kind == 1` rows with `modules.incompressible.wallPressure`'s
#: extrapolation of the current fluid `p` before the pressure accel. Default
#: `None` here -- `dfsphReference` is a troubleshooting artifact and `iisph`
#: is a landed baseline; this is graded as an A/B first
#: (`probe_hydrostaticColumnIisph.py`). omniSPH applies it in the density
#: solve only; `WALL_PRESSURE_ON_DIVERGENCE` also applies it to the DF Jacobi
#: (the fragile one, Parts 26/28/29/37) for the experiment.
WALL_PRESSURE_MODE = None
WALL_PRESSURE_ON_DIVERGENCE = False


def _rebuildAdjacency(state: Any, system: Any, config: Any):
    adjacency = buildVerletList(
        state, config.domain, verletScale=config.verletScale,
        supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=system.adjacency, verbose=False)
    system.adjacency = adjacency
    return adjacency


def _factor(state: Any, config: Any, schemeConfig: Any, adjacency: Any) -> torch.Tensor:
    """DFSPH factor `alpha_i`, SPlisHSPlasH `computeDFSPHFactor` form: the
    sum-of-squares term runs over fluid neighbours only (a static boundary
    particle takes no reaction) and the boundary enters only the vector-sum
    term, i.e. `BoundaryOperatorTerms.staticBoundary` / Bender-Westhofen-Jeske
    2023 Eq. 32. `computeDFSPHFactor` returns the positive stiffness denominator
    `sum_grad_p_k`; it is negated here so the value is <= 0 -- the sign both
    `divergenceFree.py` solvers iterate against -- and floored away from zero to
    bound the division. Runs inside `applyConsistentCoupling`, so boundary rows
    carry their Akinci apparent volume in `state.masses`."""
    apparentArea = state.masses / state.densities
    diag = computeDFSPHFactor(state, config, schemeConfig, adjacency,
                              apparentVolumes=apparentArea)
    return torch.clamp(-diag, max=-1e-8)


def _drhodt(state: Any, config: Any, schemeConfig: Any, adjacency: Any,
            advectionVelocities: torch.Tensor) -> torch.Tensor:
    """`Drho/Dt` from the same scatter-divergence operator both `divergenceFree`
    solvers form their source from (`computeMomentumIncompressible`, `-rho0 div
    v`) -- so this reference inherits that operator rather than adding a second
    convention."""
    return computeMomentumIncompressible(
        currentState=state, config=config, schemeConfig=schemeConfig,
        adjacency=adjacency, advectionVelocities=advectionVelocities)


def _pressureAccel(state, config, adjacency, p, fluidMask):
    a_p = computePressureAccelIISPH(
        state=state, pressureValues=p, config=config,
        supportScheme=SupportScheme.Scatter, adjacency=adjacency)
    # Zero `a_p` for every non-fluid row `i` (`kind == 1` boundary AND `kind ==
    # 2` ghost): a static particle takes no reaction (`staticBoundary` /
    # `operatorMovesBoundary=False`), and ghosts are not real particles at all.
    # `computePressureAccelIISPH` already excludes `kind == 2` *neighbours* from
    # the sum (default `OperationDirection.AllToAll`), so this only touches the
    # output rows.
    return torch.where(fluidMask.unsqueeze(-1), a_p, torch.zeros_like(a_p))


def _xspH(state: Any, config: Any, schemeConfig: Any, adjacency: Any,
          fluidMask: torch.Tensor) -> torch.Tensor:
    """The reference's XSPH velocity filter, as a non-pressure acceleration
    (see the `XSPH_FLUID_EPSILON` note). `warpOperation(Interpolate)`
    computes `sum_j f_j (m_j/rho_j) W_ij` over all neighbours (fluid +
    static boundary, `AllToAll`), so the reference's per-epsilon sums
    collapse to two interpolations with the per-kind epsilon folded into
    the reference values: with `c_j` = `XSPH_FLUID_EPSILON` /
    `XSPH_BOUNDARY_EPSILON` by kind,
    `- (1/h) sum_j c_j (m_j/rho_j) (v_i - v_j) W_ij`
    = `-(1/h) (interp(c v) - v_i interp(c))`. The boundary takes no
    reaction (static, like the rest of this scheme); the output is
    masked to the fluid rows."""
    c = torch.where(fluidMask,
                    torch.full_like(state.densities, XSPH_FLUID_EPSILON),
                    torch.full_like(state.densities, XSPH_BOUNDARY_EPSILON))
    props = OperationProperties(
        kernel=config.kernel, operation=WarpOperation.Interpolate,
        supportMode=SupportScheme.SuperSymmetric)
    cv = warpOperation(state, props, domain=config.domain,
                       referenceValues=state.velocities * c.unsqueeze(-1),
                       adjacency=adjacency)
    cc = warpOperation(state, props, domain=config.domain,
                       referenceValues=c, adjacency=adjacency)
    a = -(cv - state.velocities * cc.unsqueeze(-1)) / state.supports.unsqueeze(-1)
    return torch.where(fluidMask.unsqueeze(-1), a, torch.zeros_like(a))


def _jacobiSolve(state: Any, config: Any, schemeConfig: Any, adjacency: Any, *,
                 vEnter: torch.Tensor, rho0: float, warmStart: torch.Tensor,
                 opDt: float, solverCfg: Any, minIters: int,
                 fluidMask: torch.Tensor, mode: str,
                 surfaceMask: torch.Tensor = None,
                 warmStartDamped: bool = False,
                 wallPressure: str = None):
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
    nIters, err)`, where `err` is SPlisHSPlasH's one-sided compression-only
    metric: `residuum = min(resid, 0)` per particle (the DF solve zeroes it
    -- and the DF source and warm start -- for particles with < 7 (2D) /
    < 20 (3D) neighbours; the CD solve has no such guard), then `err =
    rho0 * mean(-residuum)` over the fluid. Note the tolerance calibration
    differs from the reference (its eta is a percent of rho0, scaled by 1/h
    in the DF solve); the config tolerances here are kept as-is.

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
    dt = opDt if mode == 'divergence' else opDt ** 0.5   # opDt is dt or dt**2
    # SPlisHSPlasH's fixed relaxation is 0.5 -- measured here (Part 29,
    # `probe_dfsphReferenceContraction.py`, hydrostaticColumn nx=32) that 0.5
    # is OUTSIDE this composed operator's Jacobi convergence window: the
    # step-1 divergence solve grows ~1.2x/iteration in its asymptotic phase
    # (spectral radius > 1) while 0.3 decays in all four (step, mode) states
    # and 0.1/0.05 regrow late. The window is ~[0.2, 0.35]; 0.3 is the
    # reference value's nearest stable neighbour. The reference runs 0.5 in
    # 3D, where the composed operator is better conditioned.
    omega = 0.3
    maxIters = solverCfg.maxIterations
    tol = solverCfg.tolerance
    # |diag|: opDt * |computeDFSPHFactor|  (_factor returns <= 0).
    invDiag = 1.0 / (opDt * (-_factor(state, config, schemeConfig, adjacency)))

    # Fixed source, computed once from vEnter (the linear form), in
    # SPlisHSPlasH's sign convention. Two operator facts fix the signs:
    #   (1) `_drhodt(v) = -rho0 * div_std(v)` -- this codebase's scatter
    #       Divergence IS the continuum divergence (positive for expansion),
    #       so `_drhodt > 0` for compression.
    #   (2) SPlisHSPlasH's `delta` operator (difference-form
    #       `V_i sum (v_i - v_j) gradW`) is the NEGATIVE of the continuum
    #       divergence, so their sources/aij_pj read (div_std = div):
    #   density:    s = 1 - rho/rho0 + h div(v)  = 1 - rho/rho0 - dt*_drhodt(v)/rho0
    #   divergence: s = +div(v)                  = -_drhodt(v)/rho0
    drhodtEnter = _drhodt(state, config, schemeConfig, adjacency, vEnter)
    if mode == 'density':
        source = 1.0 - state.densities / rho0 - dt * drhodtEnter / rho0
    else:
        source = -drhodtEnter / rho0

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
    guardThreshold = 7 if config.domain.dim == 2 else 20
    nNeighbours = countNeighbors(state, config, schemeConfig, adjacency)
    deficient = fluidMask & (nNeighbours < guardThreshold)
    # Part 30 (step 3): the free-surface gauge rows (the dilated
    # detectFreeSurface mask, divergence solve only). They are held at
    # kappa^v = 0 for the whole solve, so they join the reference-deficient
    # rows in the source / warm-start / metric zeroing, and the pressure
    # update is additionally pinned back to 0 on them each iteration.
    pin = None if (mode != 'divergence' or surfaceMask is None) \
        else (surfaceMask & fluidMask)
    exempt = deficient if pin is None else (deficient | pin)
    if mode == 'divergence':
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
        cap = 2.5e-4 if mode == 'density' else 0.5
        p = torch.where(fluidMask & (source < 0.0),
                        0.5 * warmStart.clamp(max=cap) / opDt,
                        torch.zeros_like(warmStart))
    else:
        p = torch.where(fluidMask, warmStart, torch.zeros_like(warmStart))
        if mode == 'divergence':
            # The reference's deficient DF particles start from p = 0 (their
            # densityAdv = 0 gates both warm-start and guess branches to 0),
            # not from the carried kappa^v; the gauge's pinned rows do the
            # same.
            p = torch.where(exempt, torch.zeros_like(p), p)
    err = 0.0
    it = 0
    for it in range(maxIters):
        pIn = wallPressureExtrapolation(
            state, config, adjacency, p, fluidMask, mode=wallPressure) \
            if wallPressure else p
        a_p = _pressureAccel(state, config, adjacency, pIn, fluidMask)
        # SPlisHSPlasH's `aij_pj = delta_{a_p} = -div_std(a_p)`, scaled by
        # opDt (h^2 CD / h DF) to match their `aij_pj *= h^2` / `*= h`.
        # `_drhodt(a_p) = -rho0 div_std(a_p)`, so `aij_pj = +_drhodt(a_p)/rho0 * opDt`.
        # (The OLD `-_drhodt(...)` was the sign-flipped operator, which is what
        # made the naive `p -=` update the diverging iteration.)
        ap = _drhodt(state, config, schemeConfig, adjacency, a_p) / rho0 * opDt
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
        if fluidMask.any():
            # SPlisHSPlasH's convergence metric: ONE-SIDED, compression-only --
            # `residuum = min(resid, 0)` per particle, then `err = rho0 *
            # mean(-residuum)` over the fluid. Under-compressed particles
            # (resid > 0, which the one-sided pressure cannot fix) contribute
            # nothing, so a solve with nothing left to correct exits early
            # instead of running to its cap as the two-sided `mean|resid|`
            # would. The DF kernel zeroes the residuum of the deficient
            # particles (and of the gauge's pinned rows, Part 30); the CD
            # kernel has no such guard.
            residuum = torch.minimum(resid, torch.zeros_like(resid))
            if mode == 'divergence':
                residuum = torch.where(
                    exempt, torch.zeros_like(residuum), residuum)
            err = float(rho0 * (-residuum)[fluidMask].mean())
        if it + 1 >= minIters and err < tol:
            break

    pIn = wallPressureExtrapolation(
        state, config, adjacency, p, fluidMask, mode=wallPressure) \
        if wallPressure else p
    a_p = _pressureAccel(state, config, adjacency, pIn, fluidMask)
    return a_p, p, it + 1, err


def dfsphReference_step(system: Any, dt: float, config: Any, schemeConfig: Any,
                        verbose: bool = False, skipDivergence: bool = None):
    # `skipDivergence` overrides the module flag when passed explicitly; the
    # `iisph` scheme (see `iisph_step`) uses this to select plain IISPH
    # without touching the process-global toggle.
    if skipDivergence is None:
        skipDivergence = SKIP_DIVERGENCE_SOLVE
    st = system.state
    fluid = st.kinds == 0
    fcol = fluid.unsqueeze(-1)
    rho0 = schemeConfig.fluid.restDensity

    solver = schemeConfig.solverConfig
    psCfg = solver.pressureSolver
    dfCfg = solver.divergenceFreeSolver
    # Boundary particles enter both solves as static fluid at rho0 with Akinci
    # rest-density volumes -- SPlisHSPlasH's boundary model for this test.
    solver.akinciBoundaryVolume = True
    # Part 25 (harden track step 1): on this codebase's multi-layer BOUNDED_BAND
    # the Akinci `m~_k = rho0 / sum_l W_kl` is numerically equal to the nominal
    # particle volume, so the boundary term in `A*p` (which is what tells the
    # one-sided constant-density drive that a standing `kappa` is doing its job)
    # under-resolves the wall by ~2x and wall-adjacent `kappa` runs away.
    # Carrying the boundary apparent volume at 2x removes the runaway (measured:
    # |v|max over 25 steps 213 -> 0.97, `hydrostaticColumn` nx=32). Not landed
    # as a general default -- it is a property of this band, not of DFSPH -- so
    # it lives on the reference scheme only. A faithful single-layer Akinci
    # sampling (module docstring) would not need it.
    if solver.akinciBoundaryVolumeScale == 1.0:
        solver.akinciBoundaryVolumeScale = 2.0
    # Part 29: SPlisHSPlasH runs BOTH solves with maxIterations = 100 (the
    # config ships 64 CD / 32 DF). At the measured-stable omega = 0.3 the
    # trajectories converge within 100 (the step-1 divergence solve reaches
    # its tolerance at ~iteration 90), so carry the reference budget; the
    # early exit means the extra headroom costs nothing on easy steps.
    if psCfg.maxIterations < 100:
        psCfg.maxIterations = 100
    if dfCfg.maxIterations < 100:
        dfCfg.maxIterations = 100

    # --- 1. neighbourhood + summation density (boundary included) ------------
    adjacency = _rebuildAdjacency(st, system, config)
    st.densities = computeDensities(st, config, schemeConfig, adjacency)

    if st.pressures is None:
        st.pressures = torch.zeros_like(st.densities)
    kappa = st.pressures.clone()          # carried constant-density warm start
    # kappa^v (divergence solve) is warm-started too -- SPlisHSPlasH warmstarts
    # both. This scheme has no acoustic term, so the unused `soundspeeds`
    # constant field is repurposed as its carrier (documented hack; avoids a
    # new state field for the reference).
    if st.soundspeeds is None or st.soundspeeds.shape != st.densities.shape:
        st.soundspeeds = torch.zeros_like(st.densities)
    kappaV = st.soundspeeds.clone()
    if DAMPED_WARM_START and system.t == 0.0:
        # First step: the case's IC wrote a RAW pressure (the hydrostatic
        # profile) into the carrier, but the damped carry is dt-scaled
        # (p*dt**2 / p*dt, see the carry block below). The reference has no
        # IC pressure at all -- its field starts at 0 -- so seed from 0.
        # (A checkpoint resume at t > 0 carries a properly scaled field and
        # skips this.)
        kappa = torch.zeros_like(kappa)
        kappaV = torch.zeros_like(kappaV)

    # --- 2. non-pressure accelerations -------------------------------------
    enforceDirichlet(system, system.t, dt, config, schemeConfig)
    # Extend the boundary velocity field per each boundary region's `BCType`
    # (`freeSlip` reflection / `noSlip` mirror / `extended` MLS), exactly as
    # `schemes/deltaSPH.py` and `schemes/dfsph.py` do -- this is what gives
    # `computeVelocityDiffusion` (below) a real no-slip wall when the case
    # asks for one. A strict no-op with no `kind == 1` particles (periodic
    # cases). The boundary rows keep the extended velocity for the rest of the
    # step, so the constant-density / divergence sources see it too.
    st.velocities = computeBoundaryVelocities(st, config, schemeConfig, adjacency)
    a_nonp = computeGravity(st, config, schemeConfig, adjacency)
    a_nonp = a_nonp + computeVelocityDiffusion(st, config, schemeConfig, adjacency)
    if XSPH_FLUID_EPSILON != 0.0 or XSPH_BOUNDARY_EPSILON != 0.0:
        # Part 32: the reference's XSPH velocity filter (module flag note) --
        # the only tangential-stress term in the scheme family.
        a_nonp = a_nonp + _xspH(st, config, schemeConfig, adjacency, fluid)
    forcing = computeForcing(system, dt, system.t, config, schemeConfig)
    a_nonp = a_nonp + forcing / st.masses.view(-1, 1)
    st.velocities = st.velocities + dt * torch.where(fcol, a_nonp, torch.zeros_like(a_nonp))

    # --- 3. constant-density solver (SPlisHSPlasH correctDensityError) -------
    # kappa is warm-started from the carried field, so `v* = v + dt*a_p(kappa)`
    # inside the first iteration already carries the standing hydrostatic
    # support -- a balanced column then reads `rho* ~ rho0` and kappa is
    # maintained rather than rebuilt from zero. The velocity impulse is applied
    # once, from the converged kappa.
    vEnter = st.velocities.clone()
    with applyConsistentCoupling(st, config, schemeConfig, adjacency,
                                 BoundaryPressureMode.consistent):
        a_p_cd, kappa, nCd, errCd = _jacobiSolve(
            st, config, schemeConfig, adjacency, vEnter=vEnter, rho0=rho0,
            warmStart=kappa, opDt=dt * dt, solverCfg=psCfg,
            minIters=max(psCfg.minIterations, 2), fluidMask=fluid,
            mode='density', warmStartDamped=DAMPED_WARM_START,
            wallPressure=WALL_PRESSURE_MODE)
    st.velocities = vEnter + dt * a_p_cd

    # --- 4. advance positions with the corrected velocity ------------------
    st.positions = st.positions + dt * torch.where(
        fcol, st.velocities, torch.zeros_like(st.velocities))

    # --- 5. rebuild neighbourhood + density on the new positions ----------
    adjacency = _rebuildAdjacency(st, system, config)
    st.densities = computeDensities(st, config, schemeConfig, adjacency)

    # --- 6. divergence-free solver (correctDivergenceError) --------------
    # Part 30 (step 3): the free-surface gauge (module flag
    # FREE_SURFACE_GAUGE) holds kappa^v = 0 on the rows the case's own
    # (dilated) detectFreeSurface flags. The mask is read off the post-move
    # state, the same state the DF solve integrates. It is a no-op when the
    # case leaves surface detection inactive (the wrapper returns an
    # all-false mask).
    surfaceMask = None
    if skipDivergence:
        # Part 33: run only the constant-density solve, applied as a velocity
        # impulse -- i.e. plain IISPH ([I] Alg. 1: one density projection per
        # step, no divergence-free pass). The DF Jacobi has been the fragile
        # component through Parts 26/28/29 (and the Part 33 calibration test),
        # and the hydrostatic column is IISPH's home benchmark, so this
        # isolates "is the DF solve the instability" and doubles as the
        # missing [I] baseline. `kappaV` is left untouched (no DF warm start
        # to carry).
        a_p_df = torch.zeros_like(st.velocities)
        nDf, errDf = 0, 0.0
    else:
        if FREE_SURFACE_GAUGE:
            _, fs, _, _ = detectFreeSurface(
                st, config, schemeConfig, schemeConfig.surfaceDetectionConfig,
                adjacency, returnNormals=False)
            surfaceMask = fluid & (fs > 0.5)
        vEnterDf = st.velocities.clone()
        with applyConsistentCoupling(st, config, schemeConfig, adjacency,
                                     BoundaryPressureMode.consistent):
            a_p_df, kappaV, nDf, errDf = _jacobiSolve(
                st, config, schemeConfig, adjacency, vEnter=vEnterDf, rho0=rho0,
                warmStart=kappaV, opDt=dt, solverCfg=dfCfg,
                minIters=max(dfCfg.minIterations, 1), fluidMask=fluid,
                mode='divergence', surfaceMask=surfaceMask,
                warmStartDamped=DAMPED_WARM_START,
                wallPressure=(WALL_PRESSURE_MODE if WALL_PRESSURE_ON_DIVERGENCE
                              else None))
        st.velocities = vEnterDf + dt * a_p_df

    # --- 7. carry kappa / kappa^v for the next step's warm start --------
    if DAMPED_WARM_START:
        # The reference stores the pressure scaled by that step's dt (CD:
        # dt**2, DF: dt -- the `*= h**k` blocks at the end of
        # pressureSolve / divergenceSolve) so the next step's damped seed
        # 0.5*min(stored, cap)/dt**k is dt-invariant. The carrier therefore
        # holds a dt-scaled field on this path, not the raw pressure (the
        # step-1 IC zeroing above is the only raw value it ever holds).
        st.pressures = kappa * dt * dt
        st.soundspeeds = kappaV * dt
    else:
        st.pressures = kappa
        st.soundspeeds = kappaV

    if verbose:
        vmax = float(st.velocities[fluid].norm(dim=-1).max()) if fluid.any() else 0.0
        kf = kappa[fluid]
        gaugeTag = f'   fsGauge={int(surfaceMask.sum())}' if surfaceMask is not None else ''
        print(f'[dfsphReference] t={system.t + dt:.4g}  '
              f'CD {nCd:2d} it err {errCd:.3g}   DF {nDf:2d} it err {errDf:.3g}   '
              f'|v|max {vmax:.4g}   kappa[{float(kf.min()):+.3g}, {float(kf.max()):+.3g}]{gaugeTag}')

    # The step integrated its own x/v/rho; hand the integrator a no-op update
    # (DFSPHReferenceSystem.{apply_*_update} are no-ops, finalize copies across).
    zerosV = torch.zeros_like(st.velocities)
    update = IncompressibleSystemUpdate(
        dxdt=zerosV.clone(), dvdt=zerosV.clone(),
        drhodt=torch.zeros_like(st.densities),
        passive=torch.zeros_like(st.densities, dtype=torch.bool))
    enforceUpdates(update, system, dt, system.t, config, schemeConfig)
    return update, adjacency, st, ([], [errCd, errDf])


def iisph_step(system: Any, dt: float, config: Any, schemeConfig: Any,
               verbose: bool = False):
    """Plain IISPH (Ihmsen et al. 2014, *Implicit Incompressible SPH*):
    a single constant-density projection per step, applied as a velocity
    impulse -- no divergence-free pass, no VD+PS position shift.

    Shares `dfsphReference`'s step body with the divergence solve switched
    off (`skipDivergence=True`). The constant-density Jacobi already uses
    `computeAlpha`'s / `computeDFSPHFactor`'s IISPH diagonal `a_ii` and the
    one-sided `rho0 - rho_adv` source, so this *is* [I]'s Algorithm 1.

    Part 33: this is the first configuration in the codebase to hold
    `hydrostaticColumn` -- stable geometry, stable bulk density, correct
    hydrostatic pressure gradient (`pressureSlopeRatio` ~ 1.0) over 2000
    steps -- and it holds `staticBlob` where the two-solve `dfsphReference`
    diverges. The divergence-free Jacobi was the catastrophic-instability
    source (the inf-velocity soup of Parts 29/30); removing it leaves a
    bounded, undamped free-slip bulk slosh and cosmetic ballistic surface
    spray, neither of which diverges or degrades. `DFSPH_IMPROVEMENT_PLAN.md`
    Part 33.
    """
    return dfsphReference_step(system, dt, config, schemeConfig,
                               verbose=verbose, skipDivergence=True)
