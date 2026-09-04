"""`RelaxedJacobiSolverConfig` (min/max iterations, tolerance, relaxation factor)
and `IncompressibleSolverConfig`, which bundles two of the former
(`pressureSolver`, `divergenceFreeSolver`) plus `integrateRho`. Embedded as
`.solverConfig` on `IncompressibleSPHConfig` and read via
`schemeConfig.solverConfig.{pressureSolver,divergenceFreeSolver}.*` by
`modules/incompressible/{incompressible,divergenceFree}.py` and
`schemes/divergenceFree.py`. `buildDefaultPSConfig`/`buildDefaultDFConfig` give the
pressure and divergence-free solvers different tuned defaults (iteration caps,
tolerances, relaxation) rather than sharing one default.
"""

__all__ = ['PressureSolverType', 'JacobiRelaxationMode', 'JacobiConvergenceCriterion', 'BoundaryPressureMode', 'ShiftPressureGauge', 'ShiftApplication', 'BoundaryOperatorTerms', 'DensityEvolution', 'resolveDensityEvolution', 'resolveBoundaryOperatorTerms', 'RelaxedJacobiSolverConfig', 'buildDefaultPSConfig', 'buildDefaultDFConfig', 'IncompressibleSolverConfig', 'buildDefaultIncompressibleSolverConfig']

from ...enumTypes import *
from typing import Optional, Union, List
from dataclasses import dataclass, field
import torch
from enum import Enum

class PressureSolverType(Enum):
    """How the incompressible pressure Poisson equation ``A p = b`` is solved.

    The operator is the IISPH matrix-free pressure operator
    ``A = dt * (IISPH pressure shift o IISPH pressure accel)`` with source term
    ``b`` the IISPH divergence, preconditioned by the IISPH diagonal (``1/D``).
    ``relaxedJacobi`` (the default) keeps the historical matrix-free relaxed
    Jacobi iteration byte-for-byte; the Krylov options are opt-in alternatives
    that solve the same ``A p = b``. See ``INCOMPRESSIBLE_SOLVER_PLAN.md``.
    """
    relaxedJacobi = 0   # default: the existing relaxed-Jacobi path (unchanged)
    cg = 1              # (preconditioned) conjugate gradient -- gated on an SPD/symmetry probe
    bicg = 2            # bi-conjugate gradient -- needs the adjoint matvec A^T
    bicgStab = 3        # bi-conjugate gradient stabilized
    gmres = 4           # (restarted) generalized minimal residual
    minres = 5          # minimum residual -- for this symmetric (not necessarily definite) operator


class JacobiRelaxationMode(Enum):
    """How the relaxed-Jacobi path chooses its per-step relaxation size
    (only used when ``solverType`` is ``relaxedJacobi``).

    The update is ``p <- p + omega * D^-1 * r`` with ``D = diag(A)``. Because
    ``D^-1 A`` is similar to the symmetric ``|D|^-1/2 (-A) |D|^-1/2 >= 0``,
    ``fixed`` converges iff ``omega < 2/rho(D^-1 A)`` -- a state-dependent
    stability window (measured ~0.355 on the TGV operator family, so the
    historical omega=0.5 default diverges and 0.3 sits inside with ~15%
    margin). ``optimal`` removes the window entirely: each step uses the
    exact residual minimizer ``omega_k = (r . A D^-1 r)/||A D^-1 r||^2``,
    which costs the same single matvec as the fixed step and decreases the
    residual monotonically for any starting size. See
    ``docs/regression/incompressible_pressure_solver_choice.md``.
    """
    fixed = 0    # default: constant relaxationFactor (byte-identical history)
    optimal = 1  # per-step exact residual-minimizing size (IISPH solver only)


class BoundaryPressureMode(Enum):
    """How `kind==1` boundary particles are handled by the incompressible
    pressure solvers (`solveDivergenceFree`/`solveIncompressible`) and by
    `schemes/divergenceFree.py`'s mDBC wiring.

    In all three modes, boundary particles are excluded from the pressure
    *unknowns*: their pressure is held fixed (not driven by the
    Jacobi/Krylov update) for the duration of a solve, they are excluded
    from the gauge-fixing mean, and their pressure acceleration (`a_p`) is
    zeroed post-solve -- the one-way-coupling contract already enforced
    downstream by `nonFluidMask` in `divergenceFree_step`. What differs is the value
    their pressure is held at, and how their density is computed:

    - `plain`: no mDBC at all. Boundary density comes from plain SPH
      summation like a fluid particle; boundary pressure is held at 0.
    - `mdbcDensity`: boundary density is mDBC-extrapolated
      (`computeMdbcDensity`, Liu-Liu MLS from fluid neighbors); boundary
      pressure is still held at 0 (no pressure extrapolation).
    - `mdbcMlsPressure`: same density extrapolation as `mdbcDensity`, plus
      the fluid pressure field is itself Liu-Liu MLS-projected onto
      boundary particles after each `solveDivergenceFree` call
      (`computeMdbcPressure`), so boundary particles carry a physically
      consistent pressure for the *next* step's force computation on fluid
      neighbors, rather than an artificial zero-pressure wall.
    - `consistent`: Bender, Westhofen & Jeske 2023, "Consistent SPH
      Rigid-Fluid Coupling". Their constraint-based derivation of DFSPH
      defines the density constraint for *fluid* particles only, so a static
      boundary particle contributes to `dC_i/dx_i` and has no constraint (and
      no pressure) of its own. This mode is that paper end to end: the
      operator terms of `BoundaryOperatorTerms.staticBoundary` are forced on
      (their Eqs. 32 and 34), boundary pressure is pinned at exactly 0 rather
      than at whatever the state carries (their Eq. 33 has no boundary
      pressure term to give a value to), and -- the part that is genuinely
      new here -- boundary rows enter the solve at `rho = rho0`, as "static
      fluid particles", instead of at the mDBC-extrapolated density this
      codebase otherwise feeds them. That last one reaches every SPH sum in
      the solve, since each weights a neighbour by `m_j / rho_j`. The mDBC
      density extrapolation still runs and is still used everywhere outside
      the pressure solve. See `modules/incompressible/consistent.py` and
      `DFSPH_IMPROVEMENT_PLAN.md` Part 11.
    """
    plain = 0
    mdbcDensity = 1
    mdbcMlsPressure = 2
    consistent = 3


class ShiftPressureGauge(Enum):
    """How `solveIncompressible` pins the constant (null-space) component of
    its pressure field each iteration.

    Both of this library's callers use `solveIncompressible` as an *implicit
    particle-shifting* solve, not a momentum pressure solve
    (`systems/incompressible.py`'s `finalize`, which feeds its output straight
    into a position shift `dx = dt**2 * a_p`; and `cases/tgv.py`'s lattice
    relaxation). Its "pressure" is therefore a shifting potential, and its
    operator has the same (near-)constant null space every pure-Neumann PPE
    has -- but unlike `solveDivergenceFree`, whose source term (a divergence)
    is mean-zero by pair antisymmetry, this solver's source term
    (`rho0 - rhoStar`) carries a persistent negative mean that no pressure
    field can remove: the SPH summation density's particle average rises
    quadratically with disorder and is bounded below by its lattice value, so
    `mean_i rho_i == rho0` is unattainable for any disordered configuration
    (`scripts/probe_densityBiasVsDisorder.py`). The solver reacts by driving
    the only mode with a nonzero mean response -- the constant one, whose
    response is weak -- to ever larger amplitude, i.e. it winds up like an
    integral controller with an unreachable setpoint.

    - `nonNegativeClamp`: the historical behavior. `clamp(p, min=0)` each
      iteration. Physically motivated (a shifting potential that never pulls
      particles together) but it is a floor, not a gauge: nothing pins the
      constant mode, so it drifts upward without bound. Measured on
      `kolmogorovIncompressible` at nx=128: the mean climbs to 2.4e6 and the
      run NaNs at step 574.
    - `minShift`: subtract the fluid minimum instead, `p -= p[fluid].min()`.
      Non-negative by construction *and* gauge-fixed (the constant mode is
      pinned by the field's own shape rather than left free), and, unlike the
      clamp, it never discards the field's negative part -- it only
      translates it, so the shift forces the solve actually computed survive.
      Same case/resolution: bounded at ~29 over 1000 steps with no trend,
      and mean/max density error roughly halved.

    Mean-centering (`p -= p[fluid].mean()`, what `solveDivergenceFree` does)
    is *not* an option here: it was tested and diverges within ~150 steps,
    because it gives up the non-negativity that keeps the shift from pulling
    particles together, and because the resulting near-zero mean removes the
    background-pressure de-clumping force this solver relies on. See
    `DFSPH_IMPROVEMENT_PLAN.md` Part 4 for the full comparison.

    **`minShift` applies wherever the constant mode is forceless, which
    includes this codebase's walls.** `solveIncompressible` falls back to
    `nonNegativeClamp` only for a solve that has *free-surface* particles,
    where the kernel support really is truncated: the gradients stop summing
    to zero, so a uniform pressure exerts a large real force and translating
    the field is no longer a free choice.

    Until Part 14 the same fallback also fired whenever any pressure row was
    pinned (`kind != 0`), i.e. on every wall-bounded case, on the argument
    that Dirichlet data already fixes the constant. Both halves of that were
    measured wrong: this codebase's walls have complete support
    (`scripts/probe_wallSupportCompleteness.py`; `BOUNDED_BAND = 5` is wider
    than the kernel), and the divergence that justified the fallback (t=0.69
    on the bounded `randomFlowIncompressible` at nx=128) was measured at 3x
    [BK]'s CFL -- at the published CFL that configuration is stable and
    better. Paired with `BoundaryOperatorTerms.staticBoundary` it holds the
    bounded case's density band at 4.48e-3 against 1.78e-1 for the old
    defaults. See `DFSPH_IMPROVEMENT_PLAN.md` Parts 13 and 14.
    """
    nonNegativeClamp = 0
    minShift = 1


class ShiftApplication(Enum):
    """How `IncompressibleSystem.finalize` applies `solveIncompressible`'s
    constant-density solution.

    DFSPH proper (Bender & Koschier) runs two solves per step and applies both
    to the *velocity*: a divergence-free projection and a constant-density
    correction. This scheme applies only the first to velocity; the second is
    repurposed as an implicit particle shift, i.e. a one-shot *position*
    displacement `dx = dt**2 * a_p`. That works in a periodic domain and is
    what every case here has always used.

    It does not work against a wall. Nothing in the scheme then produces a
    velocity-level response to a density *error* -- the divergence-free solve
    only enforces `div v = 0`, which prevents further compression but never
    undoes existing compression -- so wall-adjacent compression can only be
    relieved by moving particles, and near a wall that pushes them through it.
    Measured on the bounded `randomFlowIncompressible` at nx=128: the shift
    reaches ~1.2 particle spacings per step, fluid accumulates inside the
    boundary band at `rho = 1.30-1.36`, and the run NaNs at t=5.54.

    - `positionShift`: the historical behavior, and the default.
    - `positionAndVelocity`: additionally applies the constant-density solution
      as a velocity correction, `v += dt * a_p`. On that bounded case it is a
      large improvement -- the run reaches t=8.0 at the *default* CFL (387
      steps, no timestep penalty), near-wall `mean|rho-1|` drops 0.30 -> 0.033,
      `rho_max` stays at 1.147 instead of climbing past 1.6, penetration
      plateaus around 250 particles instead of accumulating, and the position
      shift itself collapses from ~1.2 spacings to ~0.1, since the velocity
      correction relieves compression continuously instead of in lumps.

    **Both velocity modes are opt-in because neither is physics-neutral, and
    `tgv` is where that shows.** Against the analytic decay rate
    `KE(t) = KE(0) exp(-4 nu k^2 t)`, `positionShift` gives 0.55x (the value
    `tests/test_physics.py` documents and asserts, and it is monotone);
    `positionAndVelocity` gives 3.2x and `inStepVelocity` 3.4x, both
    non-monotone. That is not a placement artifact -- it is the same
    unreachable setpoint `ShiftPressureGauge` documents. The SPH summation
    density's particle average cannot equal `rho0` for a disordered
    configuration, so the constant-density solve never converges, and feeding
    its permanent residual into the *momentum* equation is a permanent
    unphysical forcing. Applied as a position shift the same residual is
    momentum-neutral -- it only reorganises particles -- which is why the
    shift formulation is benign in the bulk and why it is still the default.

    The obvious next experiment, not done: drive the velocity correction with
    the *attainable* part of the source only (project out the structurally
    unreachable mean, which Part 4 measured), leaving the position shift to
    use the raw source as it does now.

    - `inStepVelocity`: DFSPH proper. The correction is computed and applied
      *inside* the step (`schemes/divergenceFree.py`), folded into the same `dvdt` the
      integrator advects with, and the position shift is dropped entirely --
      two velocity-level solves per step and no repositioning, which is the
      original formulation. Best wall behavior of the three by a wide margin:
      on the bounded case near-wall `mean|rho-1|` is 9.7e-3 (against 3.3e-2
      for `positionAndVelocity` and 0.30 at the default's death), only 63
      particles ever inside the boundary band (against 239 and 4506), and
      `rho` stays within [0.986, 1.140].

      Keeping the position shift *as well* is a mistake worth naming: it
      corrects the same density error twice per step, and on `tgv` that
      injects energy instead of removing it -- kinetic energy *grows* 6.6x
      over 200 steps. `finalize` therefore skips the shift in this mode.

    See `DFSPH_IMPROVEMENT_PLAN.md` Part 5.
    """
    positionShift = 0
    positionAndVelocity = 1
    inStepVelocity = 2


class JacobiConvergenceCriterion(Enum):
    """Which statistic of the residual the relaxed-Jacobi loops compare against
    ``tolerance``.

    The two solvers shipped two different tests and neither was reachable from
    the config, so `DFSPH_IMPROVEMENT_PLAN.md` §1.7 -- "the stopping criterion
    is broken" -- could be argued but not measured. These are the same two
    tests plus the published one, as one setting. With ``r = b - A p`` over
    fluid rows:

    - ``flooredOneSided``: ``mean(clamp(-r, min=-tolerance))``.
      `solveIncompressible`'s historical test, and its default. One-sided (only
      over-compression counts towards the error) *and* floored, so an
      under-dense particle contributes at most ``-tolerance`` rather than its
      full negative value and cannot cancel an over-dense one. Neither
      published criterion floors: [BK] Alg. 3 tests ``rho_avg - rho0 > eta``
      and [I] §5.1 the same shape, both on the plain average.
    - ``oneSided``: ``mean(-r)``, i.e. the published form with the floor
      removed. This is the one-line difference §1.7 identifies.
    - ``meanAbsolute``: ``mean(|r|)``. Both divergence-free loops' historical
      test, and their default. The only one of the three that is a norm, so the
      only one that cannot be satisfied by cancellation.

    The three are *not* interchangeable at a fixed `tolerance` -- they have
    different scales, and a value tuned for one is meaningless for another.
    Changing the criterion means re-tuning `tolerance` with it.

    Orthogonal to this, `rtol`/`atol` add a relative disjunct
    (``mean|r| <= atol + rtol * mean|b|``) to whichever statistic is chosen.

    See `modules/incompressible/convergence.py` and Part 15.
    """
    flooredOneSided = 0
    oneSided = 1
    meanAbsolute = 2


class BoundaryOperatorTerms(Enum):
    """Which pressure-operator terms a *static* neighbour (`kind != 0`:
    boundary, ghost) is allowed to contribute to.

    The incompressible solvers iterate `A p = b` with

        A p = dt^k * div_i( a_p )    where a_p = -grad p / rho    (per particle)

    and precondition it with `computeAlpha`'s IISPH diagonal
    `alpha_i ~ |sum_j V_j gradW_ij|^2 / m_i + sum_j V_j^2/m_j |gradW_ij|^2`.
    Both sums, and the divergence, currently run over one `AllToAll` neighbour
    set, so a boundary particle is treated exactly like a fluid one.

    Two of those contributions describe the *neighbour's* response to `p_i`,
    and they are wrong for a particle that never moves. `schemes/divergenceFree.py`
    zeroes `dxdt`/`dvdt` for every `kind != 0` row, so boundary and ghost
    particles are static by construction:

    - `alpha`'s **second sum** is `dp_i/dx_j` -- how much `rho_i` changes
      because neighbour `j` accelerated under `i`'s pressure. [BK] §3.2 states
      it directly: "since `F^p_{j<-i} = 0` if particle j is not dynamic, the
      equation for `kappa^v_i` must be adapted accordingly for static boundary
      particles." SPlisHSPlasH's `TimeStepDFSPH::computeDFSPHFactor` implements
      exactly that -- its boundary loop accumulates into `grad_p_i` (the first
      sum) and never into `sum_grad_p_k` (the second).
    - the **divergence's `a_j` term**: `dx_p_i = sum_j V_j (a_i - a_j).gradW_ij`
      counts the neighbour's pressure displacement, which is zero for a static
      particle. SPlisHSPlasH's `TimeStepIISPH::pressureSolveIteration` does the
      same thing on the other side -- its boundary loop keeps only `i`'s own
      displacement (`sum += V_j * dij_pj_i.gradW`) and drops the neighbour's.

    The two are one physical statement applied to the diagonal and to the
    off-diagonal, so `staticBoundary` changes both together and the diagonal
    stays the true diagonal of the operator being iterated. The two
    single-sided values exist to measure them separately (a mismatched
    diagonal is a rescaled relaxation factor, not a different equation --
    see `JacobiRelaxationMode`) and are diagnostics, not recommendations.

    Only reaches cases that sample `kind != 0` particles; a no-op on the
    periodic ones -- measured, not assumed: the disputed term is 40% of the
    diagonal in the wall-adjacent bin, 55% for particles that have crossed the
    wall, and *exactly* zero beyond 3 particle spacings.

    **`staticBoundary` on both solvers is the default** (Part 14), and it is
    only worth having together with `ShiftPressureGauge.minShift`: the two are
    one fix applied at two points, and they compose 5.4x better than
    independently. Measured on the bounded `randomFlowIncompressible`, nx=128,
    900 steps, at the published CFL (`cflFactor=0.4`, i.e. [BK]'s constant --
    recorded as `cflFactor=0.1` before Part 12 rewrote the condition in
    particle diameters instead of support radii; the timestep is the same
    one), as the density band `mean max(|rho_max-1|, |rho_min-1|)` over the
    second half of the run:

    | gauge | PS terms | DF terms | band | t_final |
    |---|---|---|---|---|
    | clamp | `full` | `full` | 1.78e-1 | 4.690 |
    | `minShift` | `full` | `full` | 1.43e-1 | 6.458 |
    | `minShift` | `staticBoundary` | `full` | 6.49e-3 | 6.231 |
    | `minShift` | `full` | `staticBoundary` | 7.11e-2 | 6.301 |
    | **`minShift`** | **`staticBoundary`** | **`staticBoundary`** | **4.48e-3** | 6.150 |

    So the operator wants to be the same on both sides: splitting it costs
    1.45x with the divergence-free solve left historical and 16x the other way
    round. Under the *clamp* gauge the same split was worse than that -- it
    diverged at t=1.65 -- and why it did is still unexplained (the boundary
    velocity is not the cause: `BCType.zeros`, the DFSPH convention, delays
    the divergence from step 283 to 482 without preventing it; nor is the
    Jacobi stability window, `rho(D^-1 A)` 6.3777 against 6.3782; nor the
    iteration budget, 32 -> 96 -> 192 delays it and then reverses). Under
    `minShift` the mismatched half-state no longer diverges at all, which
    narrows that open question to the clamp.

    One caveat, measured: **it is a published-CFL result.** At 3x [BK]'s limit
    (`cflFactor=1.2` in the current units, the pre-Part-12 default of 0.3
    support radii) `staticBoundary` diverges at t=1.41 where `full` reaches
    t=5.54 -- a smaller `|alpha|` at the wall means a larger Jacobi step,
    which is not survivable at 1.2 particle spacings of displacement per step.
    Same entanglement `ShiftPressureGauge.minShift` has, and at that timestep
    no configuration survives, `full` included (Part 13).

    See `DFSPH_IMPROVEMENT_PLAN.md` Parts 9, 13 and 14.
    """
    full = 0            # historical: static neighbours contribute every term
    staticBoundary = 1  # default. [BK]/SPlisHSPlasH: drop their reaction in *both* alpha and the operator
    diagonalOnly = 2    # diagnostic: drop it in alpha only (operator unchanged)
    operatorOnly = 3    # diagnostic: drop it in the operator only (alpha unchanged)

    @property
    def alphaIncludesBoundaryReaction(self) -> bool:
        """Whether `computeAlpha`'s second sum runs over static neighbours."""
        return self in (BoundaryOperatorTerms.full, BoundaryOperatorTerms.operatorOnly)

    @property
    def operatorMovesBoundary(self) -> bool:
        """Whether a static neighbour's own pressure acceleration enters the
        divergence the solvers iterate."""
        return self in (BoundaryOperatorTerms.full, BoundaryOperatorTerms.diagonalOnly)


class DensityEvolution(Enum):
    """Where the density field a step runs on comes from: a fresh SPH
    summation, or the continuity equation integrated in time.

    Weakly-compressible SPH conventionally integrates `drho/dt = -rho div v`
    and never re-sums (`schemes/deltaSPH.py` does exactly that, which is why
    it carries a density-diffusion term at all). This scheme re-sums twice per
    step instead: once at the top of `divergenceFree_step`, and once inside
    `IncompressibleSystem.finalize` before the constant-density solve. The
    continuity term is still computed and still handed to the integrator --
    `update.drhodt` -- but `finalize` overwrote the result unconditionally, so
    `integrateRho=True` had no effect on the density that survived a step
    (`DFSPH_IMPROVEMENT_PLAN.md` Part 3's audit). This enum is what
    `integrateRho` was supposed to be, and the two re-sums are separately
    controllable because they answer different questions.

    - `summation` (default, and byte-identical history): re-sum at the top of
      every step and again in `finalize`. Exact by construction -- the density
      always matches the particle positions -- and the reason this scheme has
      never needed a density-diffusion term.
    - `continuity`: integrate, never re-sum. The WCSPH standard, and the
      cheaper path (one fewer full neighbour pass per step, plus one more the
      `finalize` skip removes).
    - `hybrid`: integrate the carried density -- so the divergence-free solve,
      the mDBC extrapolation and `drhodt` all run on it -- but give the
      *constant-density/shifting* solve a fresh summation density, which is
      then discarded rather than carried. This exists because the two solves
      need different things from a density field. The divergence-free solve
      only needs `div v`, which the continuity equation tracks exactly. The
      shifting solve exists to repair *particle-distribution* drift, and the
      continuity equation cannot see that: `drho/dt = -rho div v` is blind to
      any rearrangement at fixed divergence, which is precisely the error the
      shift is there to remove. Integrating through the shift also never
      accounts for the shift's own displacement, so a purely integrated
      density drifts away from the true summation value by however much the
      scheme has shifted -- `scripts/probe_densityEvolution.py` measures that
      drift directly rather than assuming it is small.

    Two interactions worth knowing before enabling either non-default value:
    `BoundaryPressureMode.plain` skips the mDBC extrapolation, so under
    `continuity`/`hybrid` nothing updates `kind != 0` rows at all (their
    `drhodt` is zeroed by `divergenceFree_step`) and they freeze at their initial value;
    and the carried density is only as good as the integrator's own order,
    which for this scheme's `semiImplicitEuler` default is first order.

    See `DFSPH_IMPROVEMENT_PLAN.md` Part 10.
    """
    summation = 0
    continuity = 1
    hybrid = 2


def resolveDensityEvolution(solverConfig) -> 'DensityEvolution':
    """`DensityEvolution` for a solver config, honouring the legacy
    `integrateRho` bool. `integrateRho=True` meant "do not re-sum at the top of
    the step" and was inert because `finalize` re-summed anyway; it now maps to
    `continuity`, which is what its name and docstring always claimed. An
    explicit `densityEvolution` wins."""
    evolution = getattr(solverConfig, 'densityEvolution', DensityEvolution.summation)
    if evolution is DensityEvolution.summation and getattr(solverConfig, 'integrateRho', False):
        return DensityEvolution.continuity
    return evolution


def resolveBoundaryOperatorTerms(solverConfig, solver) -> 'BoundaryOperatorTerms':
    """`BoundaryOperatorTerms` for one of the two solvers.

    The setting lives on `RelaxedJacobiSolverConfig`, so the constant-density
    and divergence-free solves can run different operators -- the split Part 9
    could not express and Part 14 measured. The bundle-level
    `IncompressibleSolverConfig.boundaryOperatorTerms` is kept as an
    *override*: `None` (the default) means "use each solver's own setting",
    and any other value forces both solvers to it, which is what every probe
    script and every recorded A/B in `DFSPH_IMPROVEMENT_PLAN.md` sets.

    `solver` is the per-solver `RelaxedJacobiSolverConfig`
    (`solverConfig.pressureSolver` or `.divergenceFreeSolver`).
    """
    override = getattr(solverConfig, 'boundaryOperatorTerms', None)
    if override is not None:
        return override
    return getattr(solver, 'boundaryOperatorTerms', BoundaryOperatorTerms.full)


@dataclass
class RelaxedJacobiSolverConfig:
    minIterations: int = field(default=1, metadata={"description": "Minimum number of iterations for the relaxed Jacobi solver"})
    maxIterations: int = field(default=10, metadata={"description": "Maximum number of iterations (used by both the relaxed-Jacobi and the Krylov paths)"})
    tolerance: float = field(default=1e-3, metadata={"description": "Tolerance for the relaxed Jacobi solver (mean |residual|; ignored by the Krylov paths)"})
    relaxationFactor: float = field(default=0.5, metadata={"description": "Relaxation factor for the relaxed Jacobi solver (ignored by the Krylov paths and by relaxationMode='optimal')"})
    relaxationMode: JacobiRelaxationMode = field(default=JacobiRelaxationMode.fixed, metadata={"description": "Relaxation mode for the relaxed-Jacobi path: fixed (constant relaxationFactor, byte-identical default) or optimal (per-step exact residual-minimizing step; same matvec count, monotonically decreasing residual, no stability window; divergenceFree/IISPH solver only)"})
    solverType: PressureSolverType = field(default=PressureSolverType.relaxedJacobi, metadata={"description": "Pressure solver: relaxedJacobi (default) or a Krylov method (cg/bicg/bicgStab/gmres/minres)"})
    rtol: float = field(default=1e-5, metadata={"description": "Relative residual tolerance (converge when ||r|| <= atol + rtol*||b||). Read by the Krylov solvers as their primary test, and by the relaxed-Jacobi loops as a DISJUNCT alongside the absolute tolerance test -- either one satisfied ends the solve. 0 disables the relative test on the Jacobi path. The Jacobi path measures both norms as means of absolute values over fluid rows. See DFSPH_IMPROVEMENT_PLAN.md 1.7 and Part 15."})
    atol: float = field(default=0.0, metadata={"description": "Absolute residual floor for the rtol test (0 = purely relative). Read by the Krylov solvers and by the relaxed-Jacobi loops' relative disjunct; distinct from tolerance, which is the absolute test on the configured convergenceCriterion's statistic."})
    restart: int = field(default=30, metadata={"description": "GMRES restart length (ignored by the other solvers)"})
    krylovFp64: bool = field(default=False, metadata={"description": "Run the Krylov recurrence in float64 while the SPH matvec stays float32 (opt-in; improves the residual by roughly an order of magnitude on this ill-conditioned operator at negligible extra cost)"})
    convergenceCriterion: JacobiConvergenceCriterion = field(default=JacobiConvergenceCriterion.meanAbsolute, metadata={"description": "Which residual statistic the relaxed-Jacobi loop compares against tolerance: flooredOneSided (mean(clamp(-r, min=-tolerance)) -- solveIncompressible's historical test, one-sided and floored so under-dense particles cannot cancel over-dense ones), oneSided (mean(-r) -- the published form, [BK] Alg. 3 and [I] 5.1, without the floor), or meanAbsolute (mean(|r|) -- both divergence-free loops' historical test, and the only one of the three that is a norm). The three have different scales, so tolerance has to be re-tuned alongside this. buildDefaultPSConfig/buildDefaultDFConfig carry the shipped values. Ignored by the Krylov paths, which have their own rtol/atol contract. See JacobiConvergenceCriterion's docstring and DFSPH_IMPROVEMENT_PLAN.md 1.7 and Part 15."})
    boundaryOperatorTerms: BoundaryOperatorTerms = field(default=BoundaryOperatorTerms.staticBoundary, metadata={"description": "Which pressure-operator terms a static (kind != 0) neighbour contributes to *in this solver*: full (boundary and ghost particles are treated exactly like fluid ones in both computeAlpha's sums and the divergence the solvers iterate) or staticBoundary (the published formulation -- a particle that never moves takes no reaction force, so it is dropped from computeAlpha's second sum AND from the divergence's neighbour-acceleration term). The two single-sided values diagonalOnly/operatorOnly are diagnostics. The two solvers are configured separately because the operator they build is the only thing they share; the setting was measured on both crossed (Part 14) and staticBoundary on BOTH is the default, because splitting it is 1.45x worse on the constant-density side alone and 16x worse on the divergence-free side alone. IncompressibleSolverConfig.boundaryOperatorTerms, if set, overrides both. A no-op on cases with no kind != 0 particles. See BoundaryOperatorTerms' docstring and DFSPH_IMPROVEMENT_PLAN.md Parts 9, 13 and 14."})


def buildDefaultPSConfig() -> RelaxedJacobiSolverConfig:
    return RelaxedJacobiSolverConfig(
        minIterations=2,
        maxIterations=64,
        tolerance=5e-4,
        relaxationFactor=0.3,
        # Both solvers run the published static-boundary operator (Part 14).
        # Stated explicitly here rather than left to the field default, since
        # this pair of builders is where the shipped tuning is read off.
        boundaryOperatorTerms=BoundaryOperatorTerms.staticBoundary,
        # The constant-density solve's historical test (Part 15). It is the
        # one §1.7 calls broken; it is still the default because the
        # replacements measured worse -- see Part 15.
        convergenceCriterion=JacobiConvergenceCriterion.flooredOneSided,
    )
def buildDefaultDFConfig() -> RelaxedJacobiSolverConfig:
    return RelaxedJacobiSolverConfig(
        minIterations=2,
        maxIterations=32,
        tolerance=2.5e-3,
        relaxationFactor=0.3,
        boundaryOperatorTerms=BoundaryOperatorTerms.staticBoundary,
        # Both divergence-free loops' historical test (Part 15).
        convergenceCriterion=JacobiConvergenceCriterion.meanAbsolute,
    )


@dataclass 
class IncompressibleSolverConfig:
    pressureSolver: RelaxedJacobiSolverConfig = field(default_factory=buildDefaultPSConfig, metadata={"description": "Configuration for the pressure solver"})
    divergenceFreeSolver: RelaxedJacobiSolverConfig = field(default_factory=buildDefaultDFConfig, metadata={"description": "Configuration for the divergence-free solver"})
    integrateRho: bool = field(default=False, metadata={"description": "Legacy alias for densityEvolution=continuity, kept for config round-tripping. It used to mean 'skip the summation at the top of the step', which was inert because finalize re-summed unconditionally (DFSPH_IMPROVEMENT_PLAN.md Part 3). resolveDensityEvolution maps True to DensityEvolution.continuity; set densityEvolution directly instead."})
    densityEvolution: DensityEvolution = field(default=DensityEvolution.summation, metadata={"description": "Where each step's density comes from: summation (the default and byte-identical history -- a fresh SPH summation at the top of every step and again in finalize), continuity (the WCSPH standard: integrate drho/dt = -rho div v and never re-sum), or hybrid (carry the integrated density through the step, but give the constant-density/shifting solve a fresh summation density that is not carried forward -- the shift repairs particle-distribution drift, which the continuity equation is blind to by construction). See DensityEvolution's docstring and DFSPH_IMPROVEMENT_PLAN.md Part 10."})
    boundaryPressureMode: BoundaryPressureMode = field(default=BoundaryPressureMode.mdbcDensity, metadata={"description": "How kind==1 boundary particles are handled by the pressure solvers: plain (no mDBC), mdbcDensity (mDBC density extrapolation only, matching this scheme's historical always-on behavior), or mdbcMlsPressure (mDBC density + MLS-projected boundary pressure)"})
    mdbcPressureRelaxation: float = field(default=0.3, metadata={"description": "Under-relaxation for BoundaryPressureMode.mdbcMlsPressure's boundary pressure update (new = old + factor*(projected - old), matching the divergence-free solver's own default relaxationFactor). Ignored by plain/mdbcDensity. The one-step-lagged MLS projection closes a positive feedback loop with the fluid pressure solve (a larger boundary pressure drives a larger nearby fluid pressure gradient, which projects to an even larger boundary pressure next step); without damping this diverges within single-digit steps even on a well-sampled boundary (see DFSPH_IMPROVEMENT_PLAN.md's mdbcMlsPressure instability finding)."})
    shiftApplication: ShiftApplication = field(default=ShiftApplication.positionShift, metadata={"description": "How finalize applies solveIncompressible's constant-density solution: positionShift (the default; a one-shot position displacement, this scheme's historical behavior) or positionAndVelocity (additionally apply it as a velocity correction, as DFSPH proper does). The latter is what makes the wall-bounded case stable -- it reaches t=8.0 at the default CFL instead of NaN-ing at t=5.54, with 9x lower near-wall density error -- but it is dissipative: it drives tgv's kinetic-energy decay to 1.93x the analytic rate. Opt-in for that reason. See ShiftApplication's docstring and DFSPH_IMPROVEMENT_PLAN.md Part 5."})
    shiftPressureGauge: ShiftPressureGauge = field(default=ShiftPressureGauge.minShift, metadata={"description": "How solveIncompressible (the implicit particle-shifting solve) pins the constant null-space component of its pressure field: minShift (the default; subtract the fluid minimum -- non-negative and gauge-fixed) or nonNegativeClamp (the historical clamp(p, min=0); a floor, not a gauge, so the constant mode drifts up without bound and NaNs kolmogorovIncompressible at nx=128/step 574). Only differs on solves where the constant mode is genuinely free -- see ShiftPressureGauge's docstring and DFSPH_IMPROVEMENT_PLAN.md Part 4."})
    forceShiftPressureGauge: bool = field(default=False, metadata={"description": "Bypass solveIncompressible's remaining guard, which downgrades ShiftPressureGauge.minShift to nonNegativeClamp on any solve that has free-surface particles. Experiment hook only (default False = shipped behaviour). The guard used to fire on pinned pressure rows too, i.e. on every wall-bounded case, and this flag existed to re-run the A/B that had rejected minShift there; that A/B was measured at 3x the published CFL, the re-run reversed it, and Part 13's factorial and Part 14's landing made minShift-on-bounded the default, so the pinned-row half of the guard is gone. What is left is the free-surface half, which is the case where kernel support genuinely is truncated and a constant pressure genuinely is not forceless -- untested, and off by default."})
    boundaryOperatorTerms: Optional[BoundaryOperatorTerms] = field(default=None, metadata={"description": "Bundle-level OVERRIDE for both solvers' boundaryOperatorTerms. None (the default) means each solver uses its own RelaxedJacobiSolverConfig.boundaryOperatorTerms, which is where the shipped defaults live; setting it to full/staticBoundary/diagonalOnly/operatorOnly forces both solvers to that value, which is what every A/B recorded in DFSPH_IMPROVEMENT_PLAN.md does. The setting moved per-solver in Part 14, because the constant-density and divergence-free solves want different operators; this field is kept so the single-knob form still works. See BoundaryOperatorTerms' docstring and resolveBoundaryOperatorTerms."})
    akinciBoundaryVolume: bool = field(default=False, metadata={"description": "Whether BoundaryPressureMode.consistent also replaces boundary particle masses with Akinci et al.'s volume correction m~_k = rho0 / sum_l W_kl (l over boundary neighbours only), as the paper specifies. Default False because that correction is derived for a ONE-LAYER boundary sampling, where it makes the single layer stand in for the whole solid half-space; this codebase samples a five-layer band (randomFlow.BOUNDED_BAND), so the layers behind the interface already supply that volume and the correction inflates the interface layer instead. Ignored by every other BoundaryPressureMode."})
    akinciBoundaryVolumeScale: float = field(default=1.0, metadata={"description": "Multiplier on the boundary apparent volume that BoundaryPressureMode.consistent feeds every pressure-solve SPH sum (whether or not akinciBoundaryVolume is on -- it scales the substituted mass either way). Default 1.0 is a strict no-op and every A/B in DFSPH_IMPROVEMENT_PLAN.md keeps its meaning. dfsphReference sets it to 2.0: on the multi-layer BOUNDED_BAND the Akinci m~_k = rho0/sum_l W_kl comes out numerically equal to the nominal particle volume (measured, hydrostaticColumn nx=32), so the paper's correction is inert here and the boundary term in A*p under-resolves the wall by ~2x -- the one-sided constant-density drive then never releases and wall-adjacent kappa runs away (Part 24 step 1 / Part 25). 2.0 removes the runaway (|v|max over 25 steps 213 -> 0.97). Ignored by every other BoundaryPressureMode."})
    mdbcNoPenetrationShift: bool = field(default=True, metadata={"description": "Whether divergenceFree_step applies computeMdbcNoPenShift's soft per-particle velocity-damping correction near mDBC boundaries. Default True preserves the scheme's historical always-on behavior; the original DFSPH paper (Bender & Koschier) has no such term and relies on the pressure projection alone to prevent penetration, so this is an experimental A/B toggle (DFSPH_IMPROVEMENT_PLAN.md) to check whether it is actually helping or is a crutch that makes the near-wall density error worse -- not a permanent design decision."})
    warmStartConstantDensity: bool = field(default=False, metadata={"description": "Only with shiftApplication=inStepVelocity: seed each step's constant-density (solveIncompressible) solve with the previous step's constant-density pressure (carried on the unused soundspeeds field) instead of the historical cold zero start. Without it a standing pressure field -- a quiescent hydrostatic column -- is rebuilt from zero every step and the fluid falls before the solve supports it; this is the missing ingredient behind Part 23's inStepVelocity divergence on hydrostaticColumn (see DFSPH_IMPROVEMENT_PLAN.md Part 24). Experiment hook, default False."})

def buildDefaultIncompressibleSolverConfig() -> IncompressibleSolverConfig:
    return IncompressibleSolverConfig(
        pressureSolver=buildDefaultPSConfig(),
        divergenceFreeSolver=buildDefaultDFConfig(),
        integrateRho=False
    )
