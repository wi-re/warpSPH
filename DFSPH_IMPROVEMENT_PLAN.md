# warpSPH — Incompressible (VD+PS / DFSPH) Improvement Plan

Working document for the incompressible SPH path (`schemes/dfsph.py`,
registered as `IncompressibleSPHScheme.divergenceFree`). Twelve sessions of
investigation, 2026-08-26 to 2026-08-28.

This file was rewritten on 2026-08-28 to remove superseded reasoning and
retracted claims. **The full narrative, including every hypothesis that was
later falsified, is in git history** (`git log -p DFSPH_IMPROVEMENT_PLAN.md`);
what is kept here is what a reader needs in order to act. Sections are ordered
so the durable content comes first and the current-state overview comes last.

**Units note.** `cflFactor` on the incompressible cases multiplies the particle
**diameter** `dx`, which is the form Bender & Koschier state their condition in
(`dt <= 0.4 d/|v_max|`). Before Part 12 it multiplied the support radius
`h = n_h dx`; with `n_h = 4` the two conventions differ by 4x for the same
number, so **every `cflFactor` recorded in git history means 4x more travel per
step than the same number means today**. Every number in this file is restated
in the current (diameter) units:

| this file says | git history says | what it is |
|---|---|---|
| **`cflFactor = 0.4`** ("the published CFL") | `cflFactor = 0.1` | 0.4 spacings of travel per step |
| **`cflFactor = 1.2`** ("the legacy CFL") | `cflFactor = 0.3` | 1.2 spacings per step, 3x published |
| `cflFactor = 0.5` | `cflFactor = 0.125` | Part 5's empirical stability point |

---

## 1. Lessons learned — the durable physics

These are the findings that everything else rests on. Each was measured, not
argued, and each survived every subsequent session.

### 1.1 The density setpoint is structurally unreachable

For equal masses, `sum_i rho_i = m sum_{ij} W(x_i - x_j)`, which by Parseval on
a periodic domain is `m sum_k What(k) |rhohat(k)|^2` with `What(k) >= 0` for any
positive-definite kernel (Wendland is one). A perfect lattice puts all spectral
weight on reciprocal-lattice vectors where the bandwidth-limited `What` is ~0;
*any* disorder moves weight to small `k` where `What` is large, and every such
contribution is non-negative.

**The lattice minimises the particle-averaged summation density.** So
`mean_i rho_i == rho0` is unattainable for any disordered configuration, and
`sourceTerm = rho0 - rhoStar` carries a permanently signed mean. Measured with
no dynamics at all (`scripts/probe_densityBiasVsDisorder.py`) — jitter sweep
against `mean(rho-1)`:

| jitter/dx | 0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|---|---|---|
| `mean(rho-1)` | 5.6e-8 | 2.2e-6 | 8.7e-6 | 3.4e-5 | 2.1e-4 | 8.5e-4 | 3.3e-3 | 1.2e-2 |

Always positive, rising as the square of the displacement (~3.9x per doubling
against 4x predicted). In the developed `kolmogorovIncompressible` flow the
floor is ~2.7e-3, against a configured PS tolerance of 5e-4.

**There is no upstream bug to find.** Both `kolmogorovIncompressible.py:71` and
`tgv.py:53` additionally pin `rho0` *to* that lattice minimum by normalising
mass — which places the setpoint exactly at the floor. That is a real code
decision, but removing it makes things worse (§2), because the source's
negative mean is load-bearing.

### 1.2 The permanent residual is safe on positions and unsafe in momentum

This is the single most important consequence of §1.1, and it is what the
literature split turns on.

- Applied as a **position shift** (`dx = dt^2 a_p`), the residual is
  momentum-neutral — it only reorganises particles, and its negative mean *is*
  the de-clumping drive that keeps the sampling ordered.
- Applied to **velocity**, the same residual is a permanent unphysical forcing.
  Measured: both velocity modes damp `tgv`'s kinetic energy at ~3.3x the
  analytic rate (against 0.55x for the position shift) and make the decay
  non-monotone.

That is why this scheme uses a position shift, and it is not a shortcut — it is
the formulation that is robust to an unreachable setpoint. Cornelis et al.
state the same conclusion from the other direction: their abstract says "the DI
source term suffers from significant artificial viscosity", and their Fig. 3
measures it.

**Narrowed by Part 16.** The `tgv` measurement is real and reproduces, but the
mechanism above does not explain it on its own. On the `shearWave` case — an
exact solution whose pressure is *constant*, so there is no pressure error and
no advection to conflate with dissipation — **all three `ShiftApplication`
modes dissipate identically**, to 0.1% at `nu = 0.01`, and at `nu = 0` the
position shift is the **most** dissipative of the three. So "applied to
velocity the residual is a permanent unphysical forcing" cannot be asserted as
a general property of the modes: on a flow with no pressure gradient it costs
nothing measurable. Whatever drives `tgv`'s 3.3x requires a real pressure field
or a real advection term, and which one is unmeasured. What *does* separate the
modes on the shear wave is the other axis — the position shift carries 1.8x the
volume error and 2.4x the wall time. See §4, Part 16.

**Confirmed on the bounded case, at 2.1x (Part 18).** At a pinned `dt` on the
inviscid bounded case — where this scheme has no artificial viscosity, so all
loss is numerical — the velocity modes retain 41% of the initial kinetic energy
at t=6 against the position shift's 81%. That is what this section claims, and
it is the measurement that justifies the default: they buy 2x lower density
error for 2.1x the energy loss, with **identical** wall behaviour (zero
particles past the wall for all three modes, over 1201 steps).

**But the `tgv` number is still withdrawn (Part 17).** The 3.2x/3.4x does not
reproduce: at `tgv`'s own default resolution the three modes' decay ratios are
0.615 / 0.693 / 0.674, within 12% of each other, and the velocity modes' ratio
moves by 2x between nx=128 and nx=256 while `positionShift` holds 0.58-0.62
everywhere. A penalty that halves under refinement **cannot** be the permanent
residual this section attributes it to, because that residual is
resolution-independent — §1.1 argues it and Part 16 measured it flat across an
8x range. What survives is narrower and still real: the velocity modes are
**non-monotone at every resolution**, so a decaying flow with no forcing gains
energy on some steps. That is the defect; the mechanism above is not its
cause.

### 1.3 This scheme is VD+PS, not a mis-named DFSPH

`dfsph_step` + `IncompressibleSystem.finalize` implement Cornelis et al.'s
VD+PS step for step and in the right order:

| [C] | this codebase |
|---|---|
| `v* = v + dt a_nonp` (Eq. 2) | `dvdt` assembled in `schemes/dfsph.py` |
| `dt grad^2 p* = rho0 div v*` (Eq. 12) | `solveDivergenceFree`, source `-divergence` |
| `v' = v* - dt grad p*/rho0` (Eq. 13) | `dvdt_pressure` |
| `x** = x + dt v'` (Eq. 14) | integrator |
| `dt grad^2 p** = (rho0 - rho**)/dt` (Eq. 15) | `solveIncompressible`, source `rho0 - rhoStar` |
| `x(t+dt) = x** - dt^2 grad p**/rho0` (Eq. 16) | `dx = dt^2 dvdt_incomp`, `positions += dx` |
| `v(t+dt) = v' + grad v' . (x(t+dt) - x**)` (Eq. 17) | `proj_vel` einsum |

Including [C] Eq. 19's detail of evaluating `grad v'` on the *current*
neighbourhood rather than the shifted one.

The registered scheme name `divergenceFree` is accurate. The misnomer is
confined to the file/function names `dfsph.py`/`dfsph_step`, whose correct
target is `vdps.py` — **not** a rescue of a broken `dfsph.py`. A real DFSPH
scheme would be a *different* method, and specifically the one [C] was written
to replace.

### 1.4 `solveIncompressible` is a shifting potential, not a stress

Both call sites (`systems/incompressible.py::finalize`, `cases/tgv.py`'s lattice
relaxation) feed its output into a position shift. So:

- its non-negativity is not "a fluid cannot sustain tension" — it is "the shift
  never pulls particles together", which is the tensile-instability guard;
- its constant mode is not a physical pressure level, which makes translating
  it a free choice wherever the operator's constant mode is null.

### 1.5 This codebase's walls have complete support, so a constant pressure is forceless there

With complete support the kernel gradients sum to zero, so a constant pressure
field produces ~zero acceleration: the operator has a constant null space and
the constant is a gauge. Truncate the support and the same constant produces a
large net force along the truncation normal.

The literature's one-layer boundaries (Akinci, as used by [BK] and [C]) *are*
truncated. **This codebase's are not.** `randomFlow.BOUNDED_BAND = 5` samples a
five-layer solid band against `h = 4` spacings — the band is wider than the
kernel. Measured (`scripts/probe_wallSupportCompleteness.py`, bounded case,
nx=128, 120 steps):

| depth (spacings) | n | Shepard `sum V_j W_ij` | `\|A.1\|/\|A.rand\|` |
|---|---|---|---|
| (<0, inside wall) | 84 | 1.00857 | 0.245 |
| [0,1) | 436 | **1.00100** | **0.194** |
| [2,3) | 455 | 0.99786 | 0.189 |
| [10,inf) bulk | 11608 | 0.99997 | 0.166 |

Shepard is ~1.00 down to the wall, and the constant mode is as near-null there
(0.194) as in the bulk (0.166). **Free surfaces are genuinely truncated; walls
here are not.** One nuance: `|sum V_j grad W_ij|` *is* elevated ~4x at the wall
(6.3 vs 1.3), but that is the volume field being discontinuous across the
interface (boundary rows carry mDBC-extrapolated densities, hence different
`V_j = m_j/rho_j`), not missing neighbours. The composite operator absorbs it.

### 1.6 A wall cannot stop a particle that crosses a full spacing in one step

The wall's force is mediated by boundary-particle kernel contributions computed
from the configuration at the *start* of the step. The distance over which that
contribution goes from "no wall" to "fully inside the wall" is one particle
spacing. A particle allowed to cross a full spacing per step traverses the
entire first boundary layer before the wall ever pushes on it — **the wall is
not weak, it is late.**

Measured sweep on `randomFlowIncompressible --bounded`, nx=128:

| `cflFactor` (spacings/step) | outcome | penetrating | near-wall `\|rho-1\|` |
|---|---|---|---|
| 1.2 (legacy default) | **NaN at t=5.54** | 4506, growing | 0.30 |
| 1.0 | **NaN at t=5.09** | 15994, growing | 0.41 |
| 0.5 | survives to t=8.0 | 653, steady | 0.040 |
| 0.2 | survives to t=8.0 | 201, *declining* | 0.022 |

The transition is sharp and sits between half a spacing and one. [BK] §3.1's
published constant is `dt <= 0.4 d/|v_max|` with `d` the particle **diameter**;
[B] Table 2's own scenes land at 0.1–0.35 spacings/step. No published run goes
near one spacing.

Caveat carried forward: `n_h = 4` was fixed throughout, so "half a spacing" and
"one eighth of `h`" are not distinguished by this data. One sweep at a
different `n_h` would settle which governs.

### 1.7 The constant-density solve does not converge — it integrates

The claim this section used to make was "the stopping criterion is broken, and
it is the last unexplained thing". Part 15 measured it, and the criterion turns
out not to be the defect. Two things are:

**The periodic cases converge; only the bounded one does not.** At the shipped
defaults `kolmogorovIncompressible` at nx=128 terminates *both* solvers at 3
iterations on every one of 200 steps, with the constant-density statistic at
2.26e-5 against a 5e-4 tolerance — a factor of 22 inside it. The
"never terminates under every gauge, every `dt`, every solver" claim was taken
under the clamp gauge and never re-checked after `minShift` landed. Like every
other error in this document, non-termination is a **near-wall** phenomenon
(§1.6).

**On the bounded case the constant-density solve is an integrator, not a
solve.** Measured along a fixed iterate path (`probe_stoppingCriterion.py
--mode trace`, 200 steps, nx=128):

| iteration k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| `mean\|r\|` | 1.085e-3 | 1.077e-3 | 1.074e-3 | 1.071e-3 | 1.069e-3 | 1.065e-3 | 1.056e-3 |
| pressure range `p_max - p_min` | 0.97 | 1.76 | 3.37 | 6.68 | 13.9 | 28.8 | 59.1 |

**64 sweeps remove 2.7% of the residual and grow the pressure field 61x,
linearly in `k` at about 0.92 per iteration.** That is what an iteration whose
increment `omega r / alpha` is nearly constant does — and the increment is
nearly constant because `A p` is nearly zero for the field being built, i.e.
the source lives almost entirely in the operator's near-null space, which is
§1.1 measured inside a single solve instead of across steps.

Three consequences:

1. **No residual criterion can fire here, and no repair of the criterion
   changes that.** The residual is not shrinking, so the tolerance's *value*
   and *form* are both beside the point.
2. **`maxIterations` on `pressureSolver` is a gain, not a budget.** It sets the
   shift amplitude. That is why §1.7's old table showed accuracy improving with
   iteration count — that was the gain rising, not a solve converging — and it
   is a physical parameter wearing a numerical parameter's name.
3. **The divergence-free solve is the opposite and is fine.** Its residual
   contracts 2.9x over 32 sweeps and its pressure range grows 34% and settles.
   It converges; it is merely stopped early by its cap.

**The floor is not the defect.** §1.7 used to name it as the one-line fix:
this codebase floors each particle's negative residual contribution at
`-tolerance` (`mean(clamp(-r, min=-tolerance))`) where both papers take a plain
one-sided average, so under-dense particles cannot cancel over-dense ones.
Measured along the same fixed path, the three statistics on the
constant-density solve are 1.033e-3 (floored), 1.020e-3 (one-sided, the
published form) and 1.086e-3 (mean absolute) — **within 6% of each other, and
all a factor of two above the tolerance**. Removing the floor changes nothing.
Retracted; see §3.

**Where the floor does matter is the divergence-free solve, and switching it
buys the published iteration count at the published price.** There the
one-sided average is 1.96e-3 against mean-absolute's 1.57e-2 — **8x smaller,
purely by cancellation** — and it is already below the 2.5e-3 tolerance at the
*first* iteration. Run end to end at the shipped tolerances (900 steps,
bounded, nx=128):

| PS criterion | DF criterion | PS iters | DF iters | band, 2nd half | t_final | wall s |
|---|---|---|---|---|---|---|
| `flooredOneSided` | `meanAbsolute` *(shipped)* | 64.0 | 31.9 | **4.4810e-3** | 6.1498 | 126.3 |
| `flooredOneSided` | `oneSided` | 64.0 | **3.0** | 6.8773e-3 | 6.0397 | 93.3 |
| `oneSided` | `meanAbsolute` | 64.0 | 31.9 | **4.4810e-3** | 6.1498 | 125.6 |
| `oneSided` | `oneSided` | 64.0 | 3.0 | 6.8773e-3 | 6.0397 | 93.0 |

Two things to read off. **Swapping the constant-density criterion is
bit-identical** — 4.4810e-3 and t=6.1498 to every digit either way — which is
the cleanest possible confirmation that it never fires. And **adopting the
published criterion on the divergence-free solve collapses it to 3.0
iterations**, which is [BK]'s reported 4.5 to within the difference between two
cases, **for 1.53x the density error**. The published iteration count is
purchasable here by choosing the statistic, and what it costs is exactly the
iterations it skips. That is a reason to be careful about comparing iteration
counts across papers, not a reason to adopt the criterion.

**The iteration budget is nevertheless well chosen** — measured, not assumed
(`--mode budget`, 900 steps, bounded, uncontended):

| PS cap | DF cap | band, 2nd half | wall s |
|---|---|---|---|
| 128 | 32 | 3.54e-3 | 203.2 |
| **64** | **32** *(shipped)* | **4.48e-3** | **127.2** |
| 96 | 16 | 4.20e-3 | 147.3 |
| 64 | 16 | 4.84e-3 | 108.1 |
| 32 | 32 | 6.54e-3 | — |
| 16 | 32 | 1.06e-2 | — |
| 8 | 32 | 1.71e-2 | — |

Cutting the constant-density budget is expensive (16x fewer iterations is 6.8x
the error) and doubling it saturates (1.26x the accuracy for 1.6x the cost);
the divergence-free budget is nearly flat (32 → 8 costs 24%). The obvious
reallocation — buy constant-density iterations with divergence-free ones — does
not pay: (96, 16) is 6% better than the shipped pair for 16% more time. The
shipped (64, 32) sits on the frontier.

**So the practical statement is the one the numbers support: this solver's
iteration count is a tuning parameter that happens to be spelled as a
convergence budget, the tolerance on it is decorative on the bounded case and
binding on the periodic ones, and the honest fix is to say so rather than to
adjust a threshold that nothing can reach.**

### 1.8 Static boundary particles do not take reaction forces

[BK] §3.2, one sentence: "since `F^p_{j<-i} = 0` if particle j is not dynamic,
the equation for `kappa^v_i` must be adapted accordingly for static boundary
particles." Every term describing *the neighbour's* response to `p_i` is absent
for a static neighbour; every term describing `i`'s own response stays.
SPlisHSPlasH implements this literally
(`TimeStepDFSPH.cpp::computeDFSPHFactor`, `compute_aij_pj`), and [BWJ23] Eqs.
32/34 *derive* it — their density constraint is defined for fluid particles
only, so `dC_i/dx_k = 0` for a static boundary.

This codebase computed both terms. The two that differ:

- `computeAlpha`'s second sum `sum_j V_j^2/m_j |gradW_ij|^2` — i.e. `dp_i/dx_j`;
- the divergence operator's `a_j` term — the neighbour's own pressure
  displacement.

In the wall-adjacent bin they are **40% of the diagonal** (55% inside the band)
and **exactly zero beyond 3 spacings**, so this is a wall effect only and a
strict no-op on periodic cases.

> Trap for anyone reading SPlisHSPlasH as ground truth: its **IISPH** keeps
> `-d_ji` in `aii`'s boundary loop (`TimeStepIISPH.cpp:287-292`) — the term its
> own matvec drops. Its DFSPH has no such mismatch, and [I] §4 agrees with the
> DFSPH file.

### 1.9 The continuity equation cannot see particle rearrangement

`drho/dt = -rho div v` describes advection of a density field. The real density
error in this scheme is dominated by particle *rearrangement* at essentially
zero divergence, which that equation cannot represent at all. Measured: under
pure `continuity` the carried density reports `|rho-1|` = 7.1e-3 while a fresh
summation on the same positions reads 4.3e-2. The carried field converges to a
smooth, plausible-looking lie.

This is not the position shift's doing — dropping the shift entirely
(`inStepVelocity`) leaves the drift at 3.5e-2. It is the flow's own
rearrangement.

The corollary is the useful part: **the divergence-free solve genuinely only
needs `div v`**, which the continuity equation tracks exactly. Only the
shifting solve needs a density that matches the positions. Hence `hybrid`.

### 1.10 The scheme is over-dissipative where the flow is violent, and that was invisible until a free surface was put in front of it

Nineteen sessions tuned this scheme against periodic and wall-bounded cases,
where it is now good: the bounded `randomFlowIncompressible` holds a density
band of 4.5e-3 and the periodic cases are healthy. The first genuinely
recognisable scenario it was asked to run — a dam break — exposes something
none of those cases could.

- **It works**, which is itself new: `dambreak --scheme divergenceFree` runs
  3000 steps to t=1.5 with fluid density in [0.907, 1.004] and produces a
  recognisable collapse and run-out. The only other free-surface incompressible
  case, `rotatingSquarePatch`, is broken.
- **The run-out is about half speed.** The surge front travels 1.50 by t=0.7
  against `deltaSPH`'s 2.82 on identical geometry, resolution and `dt`.
- **88% of the kinetic energy disappears between t=0.5 and t=0.8**, exactly
  when the falling column should be turning into horizontal run-out, while
  `deltaSPH` is still gaining. The peak is 2x higher (7.42 against 3.61) and
  then collapses.

Three things this is *not*, each checked: not a timestep artifact (the run is
5x **finer** in time than [BK]'s condition requires, because `dambreak` has no
incompressible `timestep` hook); not the `ShiftApplication` mode (Part 18 has
`positionShift` as the least dissipative of the three); and not divergence or
instability (nothing blows up, the density band is good throughout).

The related observation, and the lead: **the free-surface density deficit
disappears.** A particle at a flat free surface reads about `0.5 rho0` under a
summation density because half its kernel support is empty, and it does — the
minimum fluid density is 0.518 at t=0.02. By t=0.2 it is ~0.98 and it stays
there, while the surface keeps only three quarters of the bulk's neighbour
count, i.e. while it is still geometrically a surface. Its neighbours must
therefore be packed closer than the bulk's. That is the constant-density solve
compacting the surface layer to reach a setpoint the geometry forbids — §1.1's
unreachable setpoint again, now at a free surface instead of in a disordered
bulk, and the same thing §4 records it doing at the rotating patch's corners.

Whether that compaction is what costs the energy is **now settled, and the
answer is no**: Part 21 forced the compaction's clamp off and the run NaNs in
4 steps (there is no dissipation left to measure), and Part 22's energy budget
measures the wall-side shifts (no-penetration) at negligible — the channel is
the incompressibility cycle itself (the DF projection plus the Eq. 17
position-shift resample), with the Monaghan viscosity secondary. The compaction
is real and the clamp that produces it is load-bearing; it is not the sink.

See §4, Part 19, Part 21, Part 22, and `probe_dambreakEnergyBudget.py`.

### 1.11 Method note: measure, do not derive from pseudocode

Three over-claims in a row (Part 7's tolerance argument, Part 8's `rho`-power
algebra, Part 8's solver-ordering claim) all came from reasoning off published
pseudocode without checking what the surrounding machinery does with the values.
Every claim that was *measured* instead settled in one run. `INCOMPRESSIBLE_
SOLVER_PLAN.md` already contained one of the results a session was spent
re-deriving. **Read the repo's own prior measurements first, then measure.**

---

## 2. Negative results — do not re-run these

Each was tested, and each failed. Recorded so nobody spends a session on them
again.

| hypothesis | result |
|---|---|
| **Move the setpoint off the lattice floor** (`--setpointEps`) | Kills `kolmogorovIncompressible` **150 steps sooner**. The source's negative mean is the shifting solve's de-clumping drive; cancelling it is a static mean-centering. Under `minShift` the apparent `nIter` 64→6 is the stopping criterion going slack, not convergence (`rhoErr` 3.5x worse). |
| **`computeAlpha` carries an extra power of `rho`** | Falsified decisively. `probe_operatorDiagonal.py` extracts the diagonal exactly: `diag(A)/alphas` = 1.00011 ± 0.0051 periodic, 0.99987 ± 0.0160 bounded with `rho` out to **1.303**. `omega_eff = omega` exactly. (`INCOMPRESSIBLE_SOLVER_PLAN.md` had already measured rel-L2 3.3e-7.) |
| **Mean-centering the shift pressure** (what the VD solver does) | Worst of four gauges: NaN at step 155. The VD solver's source is a divergence, mean-zero by pair antisymmetry, so recentering costs it nothing; this solver's is not. |
| **Centre-then-clamp** | NaN at step 136. Chops the field's shape every iteration rather than translating it. |
| **Zero-meaning the source** (textbook compatibility projection) | Makes the solver *converge* (nIter 64→13.4) and the density *worse* (`mean\|rho-1\|` 3.9e-3 vs 2.9e-3, `rhoStd` 3.4x). Only part of the source's mean is unreachable; the rest is the real de-clump signal. |
| **Capping the implicit shift** (`--shiftCap`) | 0.25 spacings/step still diverges (t=5.28); 0.05 diverges *much earlier* (t=1.45). This is [C] §6's argument about user-tuned shift magnitudes, measured. |
| **deltaSPH shift stacked on the implicit shift** | 4% (noise) at default magnitude, then monotonically worse, then NaN at `shiftProperties.CFL=3.0`. Same [C] §6 argument from the other side. Note it was also *never running* — `finalize` shadowed `dx`; fixed, then rejected. |
| **Confining the velocity correction to a wall band** | Diverges at t=4.17 vs t=8.0 unscoped; widening the band is worse (t=2.74). The correction's value is not wall-local — it improves the *bulk* 2x — and a shell edge injects a velocity discontinuity. Mode implemented, measured, removed. |
| **Scaling the velocity correction down** | No sweet spot. Bounded case stable at λ=0.25; `tgv` already 1.4x analytic and non-monotone there. |
| **Moving the MLS projection earlier in the step** (`--mlsBeforeSolve`) | 232 penetrating vs 228. It changed the *lag*, not the *statefulness* — the real fix is moving it inside the Jacobi iteration (§4, item 5), which is untried. |
| **Removing `mdbcNoPenetrationShift`** | Strictly worse on 2 of 3 crossing metrics (543 vs 441 crossing; worst depth 16.2dx vs 10.4dx). Not a crutch worth removing. |
| **Reflecting the mDBC normal component** (the published `-2` form) | A wash with the no-pen shift on, *worse* with it off, much worse paired with `staticBoundary` (`\|rho-1\|` 7.98e-2 vs 3.00e-2). Report the deviation; do not "fix" it blind. |
| **Boundary velocity is the divergence-free half-state's problem** | Setting boundary velocity to the rigid body's (`zeros`, the DFSPH convention) delays the divergence from step 283 to 482 and does not prevent it. |
| **The Jacobi stability window explains it** | `rho(D^-1 A)` = 6.3777 (`full`) vs 6.3782 (`staticBoundary`) — identical to four digits, dominant mode is a bulk mode either way. |
| **The iteration budget explains it** | DF cap 32→96 delays (283→597 steps); 192 is *worse* than 96. Not an under-iteration. |
| **Applying `staticBoundary` to the divergence-free solve alone** | Turns a finite 901-step run into divergence at t=1.65. See §4 item 3 — still unexplained. |
| **Akinci `m~_k` as the particles' actual mass** (the faithful reading) | Diverges in **9 steps**. The correction assumes a one-layer sampling; `BOUNDED_BAND = 5` already contains the volume it adds, so the density sum double-counts it (~15% phantom compression at step 0). |
| **One-sided source clamping** | Published negative result — [I] §3.2: "causes implausible alignments of single particles at the fluid surface for CG and Jacobi." |
| **Krylov on the clamped solve** | Published negative result, confirmed end-to-end: post-hoc `gauge='nonnegative'` gives `rhoErr` 3.6e-1 (MINRES, garbage) or NaN at step 4 (CG, BiCGStab). [I] §3.2 and [BK] §5 both state it. |
| **Reordering `inStepVelocity` to [BK] Alg. 1** | Demoted, not run. Under `semiImplicitEuler` the position advances with the *updated* velocity, making the codebase's ordering equivalent to [BK]'s up to loop phase. The surviving difference is a one-step lag (`DF` computed before `CD`), not an absence of projection. |
| **Adopting the published stopping criterion** | Bit-identical on the constant-density solve (its test never fires under any of the three statistics) and 1.53x *worse* on the divergence-free one, where it collapses the solve to 3.0 iterations by cancellation. It buys [BK]'s iteration count and pays for it in density error (§1.7, Part 15). |
| **Re-tuning the iteration budget** | The shipped (64, 32) is on the accuracy/cost frontier. 128 PS buys 1.26x for 1.6x the wall time; 16 DF loses 8%; the reallocation the "one converges, one does not" picture suggests, (96, 16), is 6% better for 16% more time. No free win (§1.7). |
| **`forceShiftPressureGauge` at a free surface** (`dambreak`) | NaN in 4 steps. Bypassing the clamp fallback does not reduce the over-dissipation Part 19 measures — the run does not survive long enough to reach it. Rules out "relax the free-surface pressure handling" as a fix and confirms the clamp is load-bearing, not optional damping (§4, Part 21). |
| **`dambreak`'s published CFL** (`cflFactor = 0.4`, [BK]'s constant, safe on every other incompressible case here) | NaN by step 30; even `0.3` NaNs by step 76. Unlike every wall-bounded case measured so far, this one needs `cflFactor = 0.2` (§4, Part 20) — the falling column's impact is sharper than `randomFlowIncompressible`'s bounded shear and the CFL's lagged `vMax` does not see it coming. |
| **`dambreak --scheme divergenceFree` at the case's default `nx = 128`** | Diverges at step 88 (t ≈ 0.175, mid free-fall, before the column reaches the floor): maxDensity 1.23, maxVelocity 4.65, "NaN detected in velocities". At `nx = 64` the same case runs past t = 1.0, but the free surface and boundary show clustering and distortion artifacts — the surface is not clean at either resolution, and the coarser one is the only one that survives. Finer is worse here. Do not spend compute on a full-resolution incompressible dam break until the baseline test cases (item 2 below) pass. |
| **`dfsphReference` free-surface `kappa^v` mask** (harden step 3) | On `hydrostaticColumn`, holding `detectFreeSurface`'s flagged rows (~27% of fluid) at `kappa^v = 0` in the divergence solve cleans the `dp/dy` fit (tracks ~1.0 vs raw -2..+3) but makes the column slump *faster* (`|v|max` 23 by step 59 vs ~2 at step 55 without). Masking the constant-density solve the same way: `rho_max` 2.5 in 20 steps. SPlisHSPlasH's `< 20`-neighbour guard never fires at `n_h = 4` (surface particles keep 53+ neighbours). Parked — the slump is a CD-solve problem (§4, Part 26). **Re-run under the Part 29 linear solve (Part 30): same sign.** The gauge (now an A/B toggle, `FREE_SURFACE_GAUGE`, default off) does not delay the late-time surface degradation (onset ~step 300-400 in both arms), degrades the surface deeper (rho_min 0.15-0.21 vs 0.25-0.38) and blocks the recovery the gauge-off survivor shows, and raises the slosh ~30-40% (|v|max 1.8-2.0 vs 1.3-1.5) over 1500-step runs. Closed as a lever for this failure mode. |
| **`dfsphReference` damped warm start** (harden step 5) | The reference's `USE_WARMSTART` / `USE_WARMSTART_V` — seed `0.5·min(p·h^k, cap)/h^k` gated on compression (CD cap 2.5e-4, DF cap 0.5, stored units; the carried field is dt-scaled) against the full-`kappa` carry (Part 31): onset of the late-time surface degradation unchanged (step 226-429 across all four runs), end-state comparable, surface depth mildly favourable at n=2 (rho_min low 0.259-0.260 vs 0.227-0.243, not conclusive), ~5x the CD iterations (median 22 vs 4), no blow-up in either arm this batch (0/4 — batch-stochastic). It exposed a baseline defect: the full-carry arm's IC hydrostatic seed (max 6.15) is destroyed by step 1's two forced CD iterations, so the baseline is an effective cold start. Not a fix — the late-time degradation now survives three levers (Part 26, Part 30, Part 31). Toggle ships off. |
| **`dfsphReference` linear optimal-step divergence solve** (harden step 4) | The SPD operator `A(p) = -dt·_drhodt(a_p(p))` with the exact residual-minimizing step converges the DF solve in 14–25 iters (vs a permanent 32) for ~13 steps on `hydrostaticColumn`, then regresses `staticBlob` hard (`|v|max` 19 by step 2): the optimal step needs null-mode handling, and `solveDivergenceFree`'s per-iteration mean-centre is the spurious-force move §1.5 forbids at a free surface. The re-summed fixed-`omega` form is uglier but has no such failure mode. `|kappa^v|` clamp not tried (§4, Part 26). |
| `convergenceCriterion` (per solver) | `flooredOneSided` (PS) / `meanAbsolute` (DF) | Each solver's historical statistic, now one setting instead of two inline tests. `oneSided` is the published form. On the constant-density solve the swap is **bit-identical** (its criterion never fires); on the divergence-free solve it collapses the solve to 3.0 iterations for 1.53x the density error (§1.7). | 3% better for 23% more time on the bounded case, against 115% better on the periodic one. At a wall the error is set by the boundary treatment, not by how well the PPE is solved. |

---

## 3. Corrections — claims this document previously made and retracted

Kept because each was acted on or nearly acted on.

1. **"`mdbcMlsPressure` is the most stable and most accurate mode"** (Part 2) —
   **wrong.** That was measured at the legacy CFL out to t≈1.5. Over 900 steps
   at the published CFL it is the **worst configuration measured**
   (`\|rho-1\|` 1.86e-1, worse than the shipped baseline's 1.78e-1, `rho_max`
   1.334). Under `inStepVelocity` it NaNs at t=0.21, inverting the ranking a
   third way. **A boundary-mode ranking does not transfer across formulations
   or across timesteps.**
2. **"Published solvers converge because their tolerance sits above the
   structural floor"** (Part 7 Q1) — **wrong.** [BK] runs 1e-4, five times
   *tighter* than this codebase's 5e-4, and converges in 4.5 iterations. The
   real defect is the stopping criterion's form (§1.7).
3. **"`computeAlpha` carries an extra power of `rho`"** (Part 8) — **wrong**,
   see §2.
4. **"`inStepVelocity` is DFSPH with the solvers swapped, and nothing projects
   the density correction"** (Parts 5/6/8) — **overstated**, see §2.
5. **"Kernel support is truncated at a wall"** (`ShiftPressureGauge`'s
   docstring, Part 4) — **false for this codebase's walls**, see §1.5. Half of
   `minShift`'s scoping justification does not hold.
6. **"`minShift` diverges on bounded cases at t=0.69"** (Part 4) — **that was
   measuring the timestep.** At the published CFL `minShift` on the bounded
   case does not diverge, covers 38% more simulated time per step budget, has
   19% lower density error, and costs half the wall time.
7. **"The MINRES win is a complete-support result"** (Part 8) — the
   *observation* stands (2.15x periodic, 3% bounded) but the *explanation* is
   wrong, since the bounded case has complete support. The better explanation
   is §1.6's: solver quality cannot fix an error that is not a solver error.
8. **"The bounded divergence-free harm is the extrapolated boundary
   velocity"** (Part 9) — **retracted**, see §2. Three mechanisms tested, all
   eliminated; the observable that survives is §4 item 3.
9. **"`consistent + akinciBoundaryVolume` is the best configuration measured"**
   (Part 11) — **true only under the clamp gauge.** Paired with
   `ShiftPressureGauge.minShift` it NaNs at step 137. And `consistent`'s own
   contribution over `staticBoundary` (4.7% under the clamp) falls to 0.007% —
   i.e. nothing — once the gauge is fixed. Both measured in Part 13's factorial
   (§4), where Part 11's own numbers reproduce exactly, so this is an
   interaction rather than a contradiction.
10. **"The nx=128 blowup is run-to-run chaotic sensitivity"** (Part 4) — no.
   The case is deterministic; re-running reproduces `pMean` = 2.3838e6 and NaN
   at step 574 exactly. That determinism is what made every subsequent A/B
   trustworthy.
11. **"The win belongs to `solveIncompressible`"** (Part 9) — **retracted.**
   Under the clamp, `staticBoundary` scoped to the shifting solve gave 2.88e-2
   against 3.00e-2 for both, which read as "the divergence-free half
   contributes nothing". Under `minShift` the same split is 6.49e-3 against
   4.48e-3, i.e. the divergence-free half contributes 1.45x, and dropping it
   gives the *worst* divergence-free residual of any gauged row. The operator
   wants to be the same on both sides (Part 14).
12. **"The harm is a property of running the two solves on inconsistent
   operators"** (Part 9) — **narrowed to the clamp.** The divergence-free
   half-state NaN'd at t=1.65 under the clamp gauge; under `minShift` the same
   configuration runs all 901 steps. Whether the halved per-sweep contraction
   behind it is also gone is unmeasured (§4 item 3).
13. **"The stopping criterion is broken, and the floor is the one-line fix"**
   (§1.7, Parts 4-13) — **retracted twice over.** The floored one-sided
   average, the published unfloored one and the mean absolute all read within
   6% of each other on the constant-density solve and all sit a factor of two
   above the tolerance, so the floor is not what keeps it from terminating.
   And the premise was already half false: at the shipped defaults the
   *periodic* cases terminate both solvers in 3 iterations. What is actually
   happening is not a criterion defect at all (new §1.7).
14. **"Both velocity modes damp `tgv`'s kinetic energy at ~3.3x the analytic
   rate"** (Part 5, §1.2, §6) — **withdrawn.** It does not reproduce at any
   configuration tried, and the statistic is not stable: at `tgv`'s own default
   nx=256 the three modes read 0.615 / 0.693 / 0.674, and the velocity modes'
   ratio moves 2x with resolution and 12% with duration. The qualitative
   observation it accompanied — that their decay is non-monotone — reproduces
   everywhere and is the part worth keeping. The bounded half of the same
   comparison (§6's table) is superseded for a different reason: it was taken
   at the legacy CFL. See Part 17. **The underlying claim is nevertheless
   right** — Part 18 measures the velocity modes losing 59% of an inviscid
   flow's kinetic energy against the default's 19%. It was the number and the
   mechanism that were wrong, not the conclusion.
15. **"`positionAndVelocity` has a 5x larger worst-case density excursion"**
   (Part 17) — **retracted, and it was my own protocol error.** The three modes
   were compared at the case's *adaptive* `dt`, so they ran at three different
   timesteps: the velocity modes damp the flow, the CFL condition hands a
   slower flow a larger `dt`, and the excursion followed from the timestep. At
   pinned `dt` `positionAndVelocity` has the *lowest* max `rho` of the three.
   `probe_boundedIncompressibleBlowup.py`'s docstring requires a fixed `dt` for
   exactly this comparison and the requirement was not followed. See Part 18.
16. **"The knee is at about 0.2 and the published 0.4 is not near it"**
   (Part 12) — **retracted.** True at the shipped boundary configuration, and
   an artifact of it: the 2.83x that a halved timestep bought was the near-wall
   band, and once the band is gone halving buys 1.17x. The sweep's own numbers
   reproduce; the inference does not survive the defaults it was measured
   under. Same shape as corrections 1 and 9: a conclusion drawn at one
   configuration, inverted by a change underneath it — and, as there, the
   thing underneath was the boundary treatment.

---

## 4. The combined default change — measured

**Three of the four have landed and the fourth is rejected.** `cflFactor`
went in with Part 12; the factorial (Part 13, below) measured the combination
at 40x and Part 14 landed the two changes behind it — `ShiftPressureGauge.
minShift` on wall-bounded solves and `BoundaryOperatorTerms.staticBoundary` on
both solvers. `BoundaryPressureMode.consistent` is not landing: it is inert
once the gauge is right, and its `akinciBoundaryVolume` variant diverges.
The table below is the history of how each was measured *alone*, which is
what makes Part 13's interaction legible; the shipped configuration is the
`minShift` + `staticBoundary` row of Part 13's factorial, not any row here.

All rows: `randomFlowIncompressible --bounded`, nx=128, 900 steps.

| change | at published CFL (0.4) | at legacy CFL (1.2) |
|---|---|---|
| **`cflFactor` 1.2 → 0.4** *(landed, Part 12)* | bounded case reaches t=8.0 at *stock* settings, near-wall `\|rho-1\|` 2.6e-2 (vs 0.30 at the legacy default's death); `kolmogorovIncompressible` 23% better; `tgv` provably inert (its `dt` is pinned at 1e-3, the CFL never binds); also removes Part 4's step-574 NaN under the *historical* gauge | — |
| **`ShiftPressureGauge.minShift` on bounded solves** (via `forceShiftPressureGauge`) | no divergence over 901 steps, t=6.458 vs clamp's 4.690, `\|rho-1\|` 1.43e-1 vs 1.78e-1, **half the wall time** | diverges at t=0.69 vs clamp's t=5.54 |
| **`BoundaryOperatorTerms.staticBoundary`** | `\|rho-1\|` 3.00e-2 vs 1.78e-1 (**5.9x**), `rho_max` 1.247→1.007, 35% more simulated time, same cost | dies at t=1.41 vs `full`'s t=5.54 |
| **`BoundaryPressureMode.consistent`** | `\|rho-1\|` **2.86e-2** (6.2x vs shipped), t=6.463; +`akinciBoundaryVolume` gives **2.38e-2**, the best row measured | dies at t=1.56 (t=3.68 with Akinci) |

The mechanism behind the entanglement is the same in all four: each makes the
solve less damped at the wall (smaller `|alpha|` → larger Jacobi step), which
is not survivable at 1.2 spacings of displacement per step.

### The factorial — run (Part 13, `probe_fourWayDefaults.py`)

**`cfl × gauge × boundary` is a 2 x 2 x 4, not a 2^4.**
`BoundaryPressureMode.consistent` *forces* `BoundaryOperatorTerms.
staticBoundary` inside both solvers (§6), so crossing them yields
bit-identical rows. The boundary axis is one ladder: `shipped`
(`mdbcDensity`+`full`) / `static` (`mdbcDensity`+`staticBoundary`) /
`consistent` / `akinci` (`consistent`+`akinciBoundaryVolume`).

Bounded `randomFlowIncompressible`, nx=128, 900 steps. **Ten of the sixteen
rows are configurations this document had already recorded, and all ten
reproduced to every digit on file** — so the harness is validated and the new
rows are trustworthy.

**At the published CFL (0.4):**

| gauge | boundary | `rho` range | `\|rho-1\|` 2nd half | t_final | DF resid | outcome |
|---|---|---|---|---|---|---|
| shipped | shipped | [0.902, 1.247] | 1.7782e-1 | 4.690 | 7.52e-2 | ok ✓*Part 9* |
| shipped | static | [0.950, 1.007] | 3.0033e-2 | 6.347 | 1.57e-2 | ok ✓*Part 9* |
| shipped | consistent | [0.955, 1.008] | 2.8620e-2 | 6.463 | 1.55e-2 | ok ✓*Part 11* |
| shipped | akinci | [0.951, 1.019] | 2.3847e-2 | 6.330 | 1.24e-2 | ok ✓*Part 11* |
| minShift | shipped | [0.863, 1.154] | 1.4318e-1 | 6.458 | 1.19e-2 | ok ✓*Part 8* |
| **minShift** | **static** | **[0.995, 1.007]** | **4.4810e-3** | 6.150 | 6.50e-3 | **best measured** |
| minShift | consistent | [0.995, 1.006] | 4.4813e-3 | 6.171 | 6.54e-3 | ok |
| minShift | akinci | — | — | 0.817 | 1.9e+6 | **NaN @ 137** |

Three results, none of which either change shows on its own:

1. **The interaction is the finding.** `minShift` alone is worth 1.24x and
   `static` alone 5.9x; composing independently predicts **2.42e-2**. Measured:
   **4.48e-3 — 5.4x better than that prediction**, 40x the shipped default and
   5.3x the best configuration previously known. The two changes are not
   additive knobs, they are one fix applied at two points: the gauge stops the
   constant mode winding up, and the static-boundary operator stops the wall
   manufacturing the error the gauge was absorbing.
2. **`consistent` is inert once the gauge is right** — 4.4813e-3 against
   `static`'s 4.4810e-3, a 0.007% difference. Under the clamp it was worth 4.7%
   (§ Part 11). So [BWJ23]'s `rho_k = rho0` boundary *state* was compensating
   for the gauge, not contributing in its own right. Its operator terms, which
   `static` already has, are the whole of its value.
3. **Part 11's best row does not survive the gauge fix.**
   `akinciBoundaryVolume` was the best measured configuration (2.38e-2) — under
   the clamp, where it still reproduces exactly. Under `minShift` it NaNs at
   step 137 with a divergence-free residual of 1.9e6. **This is the fourth time
   a boundary ranking in this document has inverted** when something underneath
   it changed (Part 2 → Part 11 on `dt`; Part 6 on formulation; now Part 11 →
   here on the gauge).

**At the legacy CFL (1.2), all eight diverge — including the shipped default.**

| gauge | boundary | steps | t_final | `rho` range | `\|rho-1\|` |
|---|---|---|---|---|---|
| shipped | shipped | 258 | 5.535 | [0.139, 2.452] | 5.38e-1 ✓*Part 9* |
| shipped | akinci | 184 | 3.677 | [0.894, 1.139] | 3.50e-2 ✓*Part 11* |
| shipped | consistent | 90 | 1.557 | [0.842, 1.241] | 9.25e-2 ✓*Part 11* |
| shipped | static | 80 | 1.412 | [0.840, 1.271] | 8.68e-2 ✓*Part 9* |
| minShift | shipped | 38 | 0.690 | [0.681, 1.257] | 2.16e-1 ✓*Part 8* |
| minShift | akinci | 34 | 0.603 | [0.951, 1.285] | 3.60e-2 |
| minShift | static | 33 | 0.585 | [0.934, 1.267] | 3.42e-2 |
| minShift | consistent | 30 | 0.529 | [0.950, 1.349] | 4.43e-2 |

> **Do not rank the `1.2` half by `|rho-1|`.** That column is a mean over the
> second half of the steps a run *survived*, so a row that dies at step 30
> reports its error over steps 15-30, before the error has developed, and a row
> that reaches 258 reports over steps 129-258. The column is only comparable
> between rows of similar length. `t_final` is the meaningful metric here, and
> every row is a divergence.

**This reframes the entanglement, and it removes the trade-off.** The story was
"each change is better at 0.4 and worse at 1.2", which reads as a trade. It is
not one: **at 1.2 there is no viable configuration at all.** The shipped
default's apparent survival to t=5.54 is the misleading part — it is surviving
with `rho` in [0.139, 2.452], i.e. 145% over-dense and 86% evacuated, which is
not a simulation anyone should want. The better configurations "die sooner"
only in the sense that they stop rather than continue producing nonsense.

So the question was never "0.4 with the new defaults against 1.2 with the old
ones". **The legacy CFL was already broken; the published CFL is what makes any
of this measurable, and it has landed.** Nothing is blocking the rest.

### The per-solver split, and the landing (Part 14, `probe_perSolverBoundaryTerms.py`)

**Landed.** `ShiftPressureGauge.minShift` now applies on wall-bounded solves,
and `BoundaryOperatorTerms.staticBoundary` is the default on **both** solvers.
Running `randomFlowIncompressible --bounded` at nx=128 with no overrides at all
now reproduces Part 13's best cell exactly — `rho` in [0.99491, 1.00712], band
4.4810e-3, t=6.1498, DF residual 6.4951e-3 — so the 40x is the shipped
behaviour, not a configuration someone has to know to ask for.

Two things had to be decided first.

**Where the setting lives.** `boundaryOperatorTerms` moved onto
`RelaxedJacobiSolverConfig`, so the constant-density and divergence-free solves
can carry different operators — the split Part 9 wanted and could not express.
`IncompressibleSolverConfig.boundaryOperatorTerms` is kept as an *override*:
`None` (the new default) means "each solver's own", and any other value forces
both. Every probe in this document sets the bundle-level knob, so every A/B on
file keeps its meaning unchanged, which is what let the reproduction rows below
be checked at all.

**What the split should be.** §4 item 2 wanted `pressureSolver =
staticBoundary` with the divergence-free solver left `full`, on Part 9's
finding that the win belonged to the shifting solve. §10's closing list
objected that a split is exactly the mismatched-operator configuration behind
the unexplained
contraction collapse. Both were inferences from clamp-gauge measurements, and
Part 13's lesson is that those do not survive the gauge fix. So it was measured
— `minShift` throughout, published CFL, bounded `randomFlowIncompressible`,
nx=128, 900 steps, same band metric as the factorial:

| gauge | PS terms | DF terms | `rho` range | band, 2nd half | t_final | DF resid | PS resid | wall s |
|---|---|---|---|---|---|---|---|---|
| clamp | `full` | `full` | [0.902, 1.247] | 1.7782e-1 | 4.690 | 7.52e-2 | 2.51e-3 | 115.6 |
| `minShift` | `full` | `full` | [0.863, 1.154] | 1.4318e-1 | 6.458 | 1.19e-2 | 4.76e-4 | 61.8 |
| `minShift` | `staticBoundary` | `full` | [0.988, 1.014] | 6.4889e-3 | 6.231 | 1.71e-2 | 9.24e-4 | 123.9 |
| `minShift` | `full` | `staticBoundary` | [0.965, 1.091] | 7.1102e-2 | 6.301 | 9.20e-3 | 4.94e-4 | 74.0 |
| **`minShift`** | **`staticBoundary`** | **`staticBoundary`** | **[0.995, 1.007]** | **4.4810e-3** | 6.150 | 6.50e-3 | 9.59e-4 | 124.9 |

**Three of those five rows are configurations Part 13 published, and all three
reproduced to every recorded digit** (1.7782e-1, 1.4318e-1, 4.4810e-3) — the
fourth harness in this document validated that way, and here it also confirms
that moving the setting per-solver is bit-for-bit inert at the old values.

Three findings:

1. **The operator wants to be the same on both sides.** Splitting it costs
   1.45x with the divergence-free solve left historical (6.49e-3) and 16x the
   other way round (7.11e-2). So §4 item 2's proposal is rejected by
   measurement and §10 item 4's caution is upheld: `both` is what landed. Note
   the two halves are not two doses of one effect — `ps-only` has the *worst*
   divergence-free residual of any `minShift` row (1.71e-2, worse even than
   `full`/`full`'s 1.19e-2) while `df-only` has a good one (9.20e-3) and a bad
   density band. Each solve's operator shows up mostly in the *other* metric.
2. **Under the gauge, the divergence-free half-state does not diverge.**
   `df-only` is the configuration that turned a finite 901-step run into a NaN
   at t=1.65 (§2, §4 item 3) — under the clamp. Under `minShift` it runs the
   full 901 steps to t=6.30 with `rho` in [0.965, 1.091]. So the divergence
   that made "mismatched operators" look dangerous belongs to the clamp, and
   the risk §10 flagged for landing a per-solver default is not there.
   What is *not* settled: whether the halved per-sweep contraction that §4
   item 3 describes is also gone, or merely no longer fatal. That was measured
   under the clamp and has not been re-measured.
3. **It is free.** Two back-to-back 901-step comparisons disagree on the sign
   of an ~8% wall-time difference (124.9s vs 115.6s in the first, 123.7s vs
   123.9s in a later re-run), so the per-step cost is inside run-to-run
   variation on a shared GPU. Per unit of *physics* it is a clear win: the
   same 901 steps cover 31% more simulated time (t=6.150 against 4.690),
   because the adaptive `dt` is no longer being held down by a wall band that
   is 40x more compressed.

**What actually changes, case by case** (measured, not assumed —
`kinds != 0` and surface-flag counts taken from the running solves):

| case | non-fluid rows | surface-flagged | effect of the landing |
|---|---|---|---|
| `randomFlowIncompressible --bounded` | 2760 / 6856 | 0 | the whole change: 1.78e-1 → 4.48e-3 |
| `tgv`, `kolmogorovIncompressible`, `randomFlowIncompressible` (periodic) | 0 | 0 | strict no-op — `staticBoundary` has nothing to drop and `minShift` was already the default |
| `rotatingSquarePatch --scheme divergenceFree` | 0 | 96 / 100 | strict no-op — the guard's surviving free-surface half keeps the clamp, and there are no boundary rows to change the operator |

So the guard did not go away, it got scoped to the case it was right about.
`solveIncompressible` still downgrades `minShift` to `nonNegativeClamp` when
free-surface particles are present, where support genuinely is truncated and a
constant pressure genuinely is not forceless (§1.5); it no longer downgrades
merely because pressure rows are pinned. `forceShiftPressureGauge` survives as
the bypass for that remaining half.

**Verified:** full suite 241 passed / 1 skipped (the two known flakes did not
fire), `gradcheck_incompressible.py` passes both `includeBoundaryReaction`
branches, `run_sweep.py` 30/30, and the config round-trips with the new
`Optional` override and the two new per-solver fields. The two end rows were
then re-run *after* the defaults were flipped and reproduced again, which is
the check that the old configuration is still reachable and still identical —
the new defaults changed what you get by saying nothing, not what any explicit
setting means.

### The CFL question, re-run at the new defaults (Part 14)

Part 12's sweep found a knee at `cflFactor = 0.2` and made the case that the
published 0.4 was permissive on a bounded case. §10's list held that back on
the grounds that it had been measured where the near-wall band dominated the
error, and that the landing would remove 40x of that band. It does, and the
knee goes with it. Same probe, same case, same protocol — every run to t=3.0,
`randomFlowIncompressible --bounded`, nx=128 — before and after:

| `cflFactor` | steps | mean`\|rho-1\|` **old defaults** | mean`\|rho-1\|` **Part 14 defaults** | gain |
|---|---|---|---|---|
| **0.4** (published) | 457 / 472 | 5.17e-3 | **9.91e-4** | **5.2x** |
| 0.2 | 903 / 956 | 1.83e-3 | 8.50e-4 | 2.2x |
| 0.1 | 1830 / 1935 | 1.37e-3 | 7.32e-4 | 1.9x |

Read down the new column instead of across it: **halving the published
timestep now buys 1.17x, and halving it again 1.16x** — against 2.83x and
1.33x before. The knee at 0.2 was the boundary treatment, not the timestep.
And the new configuration at the published 0.4 is better than the old one at
0.1 (9.91e-4 against 1.37e-3) for a quarter of the steps.

So **item 2 closes: keep `cflFactor = 0.4`.** The case for departing from a
cited constant rested on a 2.8x that no longer exists. The tail agrees — the
sub-rest-density fraction at 0.4 falls 12.2% → 5.8%, and `rho_max` over the
whole run 1.202 → 1.006, so the worst particles are no longer strongly
timestep-limited either.

Two notes on reading the table. The step counts barely move (472 against 457
at 0.4), so the accuracy is not being bought by the adaptive `dt` quietly
shrinking — it is the same timestep on a better-conditioned near-wall state.
And the wall-clock columns are **not** comparable across the two sweeps: they
were run in different sessions with different GPU contention, which is enough
to swamp the per-step difference entirely (see the previous section's point 3).

### The stopping criterion — measured, and it was the wrong suspect (Part 15, `probe_stoppingCriterion.py`)

**No default changed; the finding did.** The full analysis is in the rewritten
§1.7. What landed in code is the machinery that made it measurable:

- **One configurable criterion across all three relaxed-Jacobi loops.** They
  used to spell two different tests inline and neither was reachable from the
  config, so "the criterion is broken" could be argued but not tested.
  `JacobiConvergenceCriterion` names the three forms — `flooredOneSided`
  (`solveIncompressible`'s historical test), `oneSided` ([BK] Alg. 3 / [I]
  §5.1) and `meanAbsolute` (both divergence-free loops' historical test) — and
  `modules/incompressible/convergence.py` computes them. The defaults are each
  solver's own historical statistic, so this is inert: the two end rows of
  Part 14's probe reproduce bit for bit.
- **`rtol` wired into the relaxed-Jacobi path as a disjunct** (§4 item 4's
  ask), with the same `mean|r| <= atol + rtol*mean|b|` contract the Krylov path
  states. It is inert at the shipped `rtol = 1e-5` and provably so: on the
  bounded case `mean|r| / mean|b|` is about 0.97 *at the last iteration*, which
  is the finding below in one number.

The measurement is `--mode trace`, which disables early exit
(`minIterations = maxIterations`, `rtol = 0`) so that every criterion is
evaluated **along the same iterate path** — same states, same iterates, three
readings, directly comparable rather than three different simulations.

Four results:

1. **The premise was half wrong.** `kolmogorovIncompressible` at nx=128
   terminates both solvers in 3 iterations on every step, statistic 2.26e-5
   against a 5e-4 tolerance. Non-termination is a bounded-case phenomenon and
   has been since `minShift` became the default in Part 4; nobody re-checked.
2. **The floor is not the defect.** On the constant-density solve the three
   statistics read 1.033e-3 / 1.020e-3 / 1.086e-3 — within 6%, all a factor of
   two above the tolerance. §1.7's "it is one line" is retracted.
3. **The constant-density solve does not converge in any norm.** 64 sweeps
   remove 2.7% of the residual and grow the pressure range 61x, linearly in the
   iteration count. It is an integrator; `maxIterations` is a gain. That is
   §1.1's unreachable setpoint observed *inside* one solve rather than across
   steps, and it explains why every criterion this document has tried failed
   in the same way.
4. **The published criterion's shape flatters published iteration counts, and
   the price is measured.** On the divergence-free solve the one-sided average
   is 8x smaller than the mean absolute (1.96e-3 vs 1.57e-2) purely by
   cancellation, and it is already under tolerance at the first iteration.
   Adopting it end to end takes that solve from 31.9 iterations to **3.0** —
   [BK]'s reported 4.5, near enough — **at 1.53x the density error** (6.88e-3
   against 4.48e-3). The same swap on the constant-density solve is
   **bit-identical**, which is the cleanest confirmation that its criterion
   never fires at all. Table in §1.7.

And one negative result worth its own line: **the shipped iteration budget is
on the accuracy/cost frontier** and should not change. Doubling the
constant-density budget buys 1.26x for 1.6x the wall time; halving the
divergence-free one loses 8%; the reallocation that the "one converges, one
does not" picture suggests — buy PS iterations with DF ones — measures worse
than the shipped pair at equal cost. Table in §1.7.

**What this changes downstream.** Item 9 (warm start) was gated on "fix the
stopping criterion first, because warm-starting a solver that is winding up
carries the wind-up across steps". That is now precise rather than
precautionary, and it splits the item: the divergence-free solve converges, so
warm-starting it is the ordinary optimisation [BK] describes; the
constant-density solve is an integrator, so warm-starting it would carry a
*linear ramp* across steps and is contraindicated outright. Item 11 (the real
`dfsph` scheme) inherits a sharper warning too: DFSPH proper puts the
constant-density solve into the *momentum* equation, where an amplitude set by
an iteration count becomes a force set by an iteration count.

### The shear-wave case — ported, and what it says (Part 16, `cases/shearWave.py`)

**Ported.** `shearWave` is registered, in the sweep, and carries four
assertions in `tests/test_physics.py`. It is the first incompressible case here
that grades this codebase against something other than itself.

**Why this case and not another.** A transverse sinusoidal shear wave,
`u_x = u0 sin(k_w y)`, `u_y = 0`, on a periodic box. Both nonlinear terms
vanish identically — `(u . grad) u = u_x d_x u_x e_x = 0` because `u_x` depends
only on `y`, and `div u = d_x u_x = 0` for the same reason — so

    u_x(y, t) = u0 sin(k_w y) exp(-nu k_w^2 t),   p = const

is exact for all time at any amplitude, **with zero pressure gradient**. `tgv`
is also an exact solution, but one in which a real pressure field balances a
real advection term, so a pressure error and a dissipation error are measured
together there. Here the exact pressure is constant, so every pressure the
solver produces is an artifact and every departure of the amplitude is
dissipation. At `nu = 0` the exact answer is that nothing happens.

**Four results.**

**1. The scheme is not very dissipative on this flow, and it converges to a
floor.** At `nu = 0`, `t = 1.0`, shipped defaults:

| nx | amplitude | deficit | max `rho` | disorder | max `\|v_y\|` |
|---|---|---|---|---|---|
| 32 | 0.992480 | 7.52e-3 | 1.00359 | 4.54e-2 | 1.33e-1 |
| 64 | 0.997221 | 2.78e-3 | 1.00324 | 2.45e-2 | 1.13e-1 |
| 128 | 0.998602 | 1.40e-3 | 1.00375 | 1.48e-2 | 7.23e-2 |
| 256 | 0.998942 | 1.06e-3 | 1.00361 | 9.64e-3 | 4.35e-2 |

The amplitude deficit falls 2.7x, 2.0x, then only 1.32x — converging onto a
floor near 1e-3 rather than to zero.

**2. The volume error does not converge at all.** `max rho` is 1.0036 ± 0.0003
across an **8x resolution range** — flat, while everything else improves.
That is §1.1 reproduced from a completely different direction: the summation
density's excess over `rho0` is set by how disordered the sampling is, not by
how fine it is, so refinement cannot remove it. §1.1 argued this spectrally and
demonstrated it with no dynamics at all (`probe_densityBiasVsDisorder.py`);
this is the same statement measured in a running simulation with an exactly
known answer.

**3. The physical viscosity is applied at about half its prescribed value —
independently confirming `tgv`'s 0.55x.** Graded against a *moving* target
(nx=128, t=2.0, amplitude reported relative to `exp(-nu k_w^2 t)`, so 1.0 is
exact at every `nu`):

| `nu` | analytic decay | measured/analytic | implied decay-rate ratio | disorder |
|---|---|---|---|---|
| 0 | 1.000000 | 0.995140 | — (pure artifact) | 2.07e-2 |
| 0.001 | 0.924012 | 1.040851 | **0.493** | 1.19e-3 |
| 0.01 | 0.453600 | 1.501154 | **0.486** | 7.79e-3 |

`tests/test_physics.py` asserts `tgv`'s kinetic energy decays at ~0.55x the
analytic rate and explains it as the Monaghan viscosity switch — viscosity is
deactivated for separating pairs, so roughly half the pairs dissipate at any
instant. **That explanation now has a second, independent measurement behind
it**: 0.49 on a flow whose exact pressure is constant, where it *cannot* be
pressure error, at two viscosities an order of magnitude apart. It is the first
number in this document that two unrelated cases agree on for a stated reason.

**4. The `ShiftApplication` question — the reason the case was ported — comes
back the other way.** §1.2 predicts the two velocity modes should dissipate
more, since they feed a permanent residual into momentum while the position
shift is momentum-neutral; on `tgv` that shows as 3.3x the analytic decay rate
against the shift's 0.55x. On this flow it does not happen. nx=128, t=2.0:

| mode | `nu = 0` amplitude | `nu = 0` max `rho` | `nu = 0.01` amplitude | `nu = 0.01` max `rho` | wall s (`nu = 0`) |
|---|---|---|---|---|---|
| `positionShift` *(default)* | 0.995100 | **1.00365** | 1.501436 | **1.00321** | 36.7 |
| `positionAndVelocity` | **0.998732** | 1.00268 | 1.499894 | 1.00177 | 15.4 |
| `inStepVelocity` | 0.997899 | 1.00312 | 1.501020 | 1.00202 | 19.2 |

**At `nu = 0.01` the three agree on dissipation to 0.1%** (1.5014 / 1.4999 /
1.5010 — all the same 0.49x half-viscosity), and at `nu = 0` the *default* is
the most dissipative of the three, by 4x in deficit. So the dissipation penalty
§1.2 attributes to the velocity modes is **not intrinsic to them**: on a flow
with no pressure gradient and no advection it disappears entirely. Whatever
produces `tgv`'s 3.3x needs one of those two things, and this case does not say
which.

What *does* separate the modes here, consistently at both viscosities, is the
other axis: **the position shift carries the largest volume error** (1.8x the
density excess of `positionAndVelocity` at both `nu`) and costs the most wall
time (2.4x at `nu = 0`). That is the opposite ranking to the one §1.2's
argument implies, on the axis this case can actually see.

**This is one case and it does not settle the default.** The velocity modes are
still the ones with a documented `tgv` dissipation problem and a documented
wall advantage (§6), and this flow has no wall and no pressure. What it does is
remove the *general* argument — "applied to velocity the residual is a
permanent unphysical forcing, applied to position it is momentum-neutral" —
from the list of things that can be asserted without qualification. §1.2 has
been narrowed accordingly.

**Still open: the comparison against [C]'s own curves.** The paper's Fig. 3 and
Fig. 4 are not in this repository, and neither the case nor this section
hard-codes numbers read off them. Everything above is this codebase measured
against an analytic solution, which is a real reference but not the published
one. Grading against [C] needs the paper in hand.

### `ShiftApplication`, re-measured at the current defaults (Part 17, `probe_shiftApplication.py`)

**The quantitative case for the shipped default does not survive re-measurement,
and the mechanism §1.2 attributes it to is the wrong one. The default has not
been changed, because what replaces that case is a judgement call rather than a
number.**

The evidence this question rested on had rotted in two different ways. The
bounded-case table in §6 was taken at the **legacy CFL** — it says so, "so
`positionShift` is at its death state" — which Part 13 later showed has no
viable configuration at all, and it predates both Part 14 defaults. And the
`tgv` decay ratios turn out not to be a stable statistic.

**`tgv`, artificial viscosity.** `decay/analytic`, fitted log-linearly over the
run, at three configurations:

| configuration | `positionShift` | `positionAndVelocity` | `inStepVelocity` |
|---|---|---|---|
| nx=128, 200 steps | 0.580 | 1.266 | 1.219 |
| nx=128, 500 steps | 0.585 | 1.423 | 1.306 |
| **nx=256** *(tgv's own default)*, 200 steps | 0.615 | **0.693** | **0.674** |

**At `tgv`'s own default resolution the three modes agree to within 12%.** The
recorded 3.2x / 3.4x does not reproduce anywhere, and the ratio for the two
velocity modes moves by 2x with resolution and 12% with duration, while
`positionShift` sits at 0.58-0.62 in every configuration — stable under
refinement, exactly as `tests/test_physics.py` claims for it.

That resolution dependence is the substantive point, not the disagreement.
**§1.2 attributes the velocity modes' dissipation to the permanent residual of
an unreachable setpoint** — and §1.1 and Part 16 both establish that that
residual is *resolution-independent* (Part 16 measured the volume error flat to
0.0003 across an 8x range). A penalty that halves from nx=128 to nx=256 cannot
be that residual. It is discretisation error. Combined with Part 16 — where the
three modes dissipate identically on a flow with no pressure gradient — the
mechanism §1.2 states is not what produces the effect §6 rejects the modes for.

**What does survive on `tgv`: the velocity modes are non-monotone at every
resolution**, including nx=256 where their net rate matches. A decaying viscous
flow with no forcing whose kinetic energy rises on some steps is having energy
injected, and that is a real defect regardless of what a fitted slope says
about the average. `positionShift` is monotone everywhere.

**The bounded case, at the published CFL and the Part 14 defaults.** nx=128,
900 steps — the run §6's table should have been:

| mode | band, 2nd half | sustained max `rho` | worst-ever `rho` | t_final | wall s |
|---|---|---|---|---|---|
| `positionShift` *(default)* | 4.4810e-3 | 1.0044 | [0.9949, 1.0071] | 6.150 | 124.7 |
| `positionAndVelocity` | **3.0074e-3** | **1.0033** | [0.9865, 1.0367] | **7.566** | 124.6 |
| `inStepVelocity` | 1.2753e-2 | 1.012 | [0.9866, 1.0202] | 7.449 | 202.8 |

> **Superseded in part by Part 18.** Every row here is at the case's *adaptive*
> `dt`, so the three modes ran at three different timesteps — the velocity modes
> damp the flow, and the CFL condition then hands them a larger `dt`. The
> excursion column below is a consequence of that, not of the modes. Part 18
> redoes this at a pinned `dt`.

"Sustained" is `max rho` sampled at deciles of the run; it is quoted because the
two summaries disagree for `positionAndVelocity`, and the disagreement is the
information. Its worst-ever `rho` is 1.0367 against the default's 1.0071 — 5x
the excursion — but at every decile it sits at 1.0026-1.0038, *below* the
default's steady 1.004. So its excursions are transient spikes on an otherwise
better-behaved run, where the default is uniformly mediocre.

**Read together, at the current defaults:**

| | `positionShift` | `positionAndVelocity` | `inStepVelocity` |
|---|---|---|---|
| `tgv` rate (nx=256) | 0.615 | 0.693 | 0.674 |
| `tgv` monotone | **yes** | no | no |
| `shearWave` amplitude | 0.9952 | **0.9987** | 0.9979 |
| `shearWave` max `rho` | 1.00335 | **1.00291** | 1.00315 |
| bounded band | 4.48e-3 | **3.01e-3** | 1.28e-2 |
| bounded excursion | **1.0071** | 1.0367 | 1.0202 |
| bounded wall s | 124.7 | **124.6** | 202.8 |

`inStepVelocity` is out: 2.8x the bounded band and 63% more wall time, with
nothing it wins. Between the other two, **`positionAndVelocity` is better on
every sustained metric across all three cases, at identical cost**, and worse
on exactly two things: energy monotonicity on `tgv`, and transient density
excursions on the bounded case.

**So the default stays, for now, and the reason is stated rather than
measured.** Both of the things `positionAndVelocity` loses on are *bounded-worst-
case* properties and both of the things it wins on are *averages*, and this
document has been wrong before by ranking on an average (§4's warning about the
legacy-CFL half of Part 13's factorial). Flipping a default on a 1.5x mean
improvement that comes with a 5x worse tail is not the trade this project has
been making. What would settle it is a metric nobody here has: whether those
transient excursions are the beginning of the wall accumulation that killed the
legacy-CFL runs, or noise on a run that is otherwise fine. `probe_boundedIncompressibleBlowup.py`
measures exactly that (penetration count and worst depth per step) and has not
been pointed at these three modes at the current defaults.

**What has changed is the argument, not the setting.** §1.2's mechanism claim
and §6's 3.2x are both withdrawn; the case for `positionShift` now rests on
energy monotonicity and tail behaviour, which is narrower and honest.

### The tail, measured — and `ShiftApplication` settles (Part 18, `probe_boundedIncompressibleBlowup.py`)

**The default is right, and for the first time the reason survives scrutiny.**
Part 17 left it resting on two worst-case properties, one of which turns out to
be an artifact of Part 17's own protocol. Both are now measured, at a **pinned
`dt`** — which the probe's own docstring has always required for an A/B whose
variants change the velocity field, and which Part 17's bounded table did not
use.

nx=128, `dt = 5e-3` fixed, to t=6.0 (1201 steps), published CFL configuration,
Part 14 defaults. `randomFlowIncompressible` has `nu = 0` and this scheme has no
artificial viscosity term, so **every joule lost below is numerical**.

| mode | `KE(6)/KE(0)` | mean `\|rho-1\|` near wall | in bulk | max `rho` | **particles past the wall** |
|---|---|---|---|---|---|
| `positionShift` *(default)* | **0.807** | 1.186e-3 | 8.95e-4 | 1.004 | **0** |
| `positionAndVelocity` | 0.409 | **5.89e-4** | **5.33e-4** | **1.003** | **0** |
| `inStepVelocity` | 0.412 | 7.17e-4 | 6.02e-4 | 1.009 | **0** |

**1. Nobody penetrates.** `nOutside` is 0 at every one of 1201 steps for all
three modes. §6's legacy table — 4506 / 239 / 63 particles inside the wall — was
the whole case for the velocity modes, and it was taken at the legacy CFL where
`positionShift` was in the act of dying. At the published CFL with the Part 14
defaults **there is no penetration for any mode to prevent**, so the argument
that motivated them is not weakened but void.

**2. The velocity modes cost half the flow's energy.** They retain 41% of the
initial kinetic energy at t=6 against the default's 81% — **2.1x the loss**, on
an inviscid case where the exact answer is that energy is conserved. This is
§1.2's claim, confirmed on the case that can show it, and it is large: not a
3.3x factor on a fitted decay rate but 59% of the flow simply gone.

**3. What they buy is 2x lower density error** — 5.9e-4 against 1.2e-3 near the
wall, 5.3e-4 against 9.0e-4 in the bulk. Real, and small in absolute terms:
both are a tenth of a percent.

**So the trade is stated exactly: half the flow's kinetic energy for half the
density error, with the wall behaviour identical.** That is not a trade worth
taking. A scheme that dissipates 59% of an inviscid flow in six time units is
not better than one that dissipates 19% and carries 0.1% density error instead
of 0.06%. `positionShift` stays, and now on a measurement rather than on §6's
withdrawn 3.2x.

**Correction to Part 17.** Its bounded table reported `positionAndVelocity`
reaching `rho = 1.0367` against the default's 1.0071 — a "5x larger worst-ever
excursion", which was half the stated reason for keeping the default. **That was
the adaptive timestep, not the mode.** At pinned `dt` its max `rho` over the
whole run is 1.003, the *lowest* of the three. The mechanism is visible in the
`vMax` column: the velocity correction damps the flow, the CFL condition hands a
slower flow a *larger* `dt`, and the excursion came from that larger timestep.
Part 17 compared three modes at three different timesteps and read the
difference as a property of the modes. The probe's docstring warned about
exactly this; the warning was not heeded.

### A dam break — the incompressible scheme's first working free surface (Part 19, `probe_dambreakIncompressible.py`)

Every incompressible case in this document is periodic or wall-bounded except
`rotatingSquarePatch`, which is broken in a way [BK] §5 documents as a method
limitation and which is a hard free-surface test (four convex corners, and the
arms it grows are surface-tension-sensitive). A dam break is the easier one:
gravity-driven, one mostly-flat free surface. `dambreak` is a weakly-compressible
case and takes `--scheme divergenceFree` with no wiring, so this costs nothing
to ask.

**It works.** nx=64, 3000 steps to t=1.5, no divergence, fluid density in
[0.907, 1.004] for the whole run, and a recognisable dam break: the column
falls, spreads, and runs out along the floor. That is the first free surface
this scheme has done without breaking.

**But it is a different dam break from the validated one.** `deltaSPH` is the
control — same geometry, same resolution, same `dt`:

| t | DF KE | DF front | deltaSPH KE | deltaSPH front |
|---|---|---|---|---|
| 0.3 | 4.15 | -0.767 | 2.12 | -0.586 |
| 0.5 | **7.42** | -0.270 | 3.61 | 0.353 |
| 0.7 | 3.16 | 0.137 | 4.23 | **1.462** |
| 0.8 | 1.66 | 0.344 | 4.35 | 1.984 *(far wall)* |
| 1.0 | 1.11 | 0.809 | 3.69 | — |
| 1.5 | 0.84 | 1.676 | 1.32 | — |

Two things, and they are the same thing seen twice:

1. **The run-out is roughly half speed.** From its start at x=-1.359 the front
   has travelled 1.50 by t=0.7 against `deltaSPH`'s 2.82. `deltaSPH` reaches
   the far wall at t≈0.8; the incompressible run has not by t=1.5.
2. **The kinetic energy peaks 2x higher and then collapses.** 7.42 at t=0.5
   against 3.61, then **88% of it is gone by t=0.8** while `deltaSPH` is still
   gaining. The peak is the column *falling*; what does not happen is the
   turn — the vertical momentum that should become horizontal run-out is
   dissipated instead.

**It is not a timestep artifact.** `dambreak` has no incompressible `timestep`
hook, so it runs at the weakly-compressible `dt = 5e-4`. With `dx = 0.03125`
and `|v|` peaking near 5, [BK]'s condition would permit `2.5e-3` — the run is
**5x finer in time than the CFL requires**, not coarser. (It is also why it
costs 294s against `deltaSPH`'s 61s for the same 3000 steps; an incompressible
`timestep` hook would recover most of that, and is worth adding.)

**The free-surface density deficit disappears, and that is worth explaining.**
Under a summation density a particle at a flat free surface reads about
`0.5 rho0`, because half its kernel support is empty. It does, at first: the
minimum fluid density is **0.518 at t=0.02**. By t=0.2 it is 0.979, and it
stays near `rho0` for the rest of the run — while the fluid still has a free
surface. Measuring the geometry directly
(`probe_dambreakIncompressible.py --mode surface`: neighbour counts within the
support for the topmost particle in each `dx`-wide column, a definition that
uses no density at all and so cannot be circular):

| | surface nbrs | bulk nbrs | ratio | surface `rho` | min surface `rho` |
|---|---|---|---|---|---|
| `divergenceFree`, t=0.02 | 28.6 | 42.3 | 0.677 | 0.766 | **0.518** |
| `divergenceFree`, t=0.5 | 33.2 | 44.3 | 0.749 | **1.0009** | **0.9968** |
| `deltaSPH`, t=0.02 | 33.9 | 42.6 | 0.797 | 1.0001 | 1.0000 |
| `deltaSPH`, t=0.5 | 29.4 | 43.4 | 0.677 | 1.0005 | 0.9994 |

**The solid statement is the within-scheme one.** Over that interval
`divergenceFree`'s surface density goes from 0.766 mean / 0.518 worst to
1.0009 mean / 0.9968 worst — a full recovery to `rho0` — while its surface
neighbour count stays at 0.68-0.75 of the bulk's, i.e. while the surface is
still geometrically a surface. The neighbours it has must therefore be closer
together than the bulk's. Surface compaction by the constant-density solve is
the natural reading of that, and it is what §4 records the same solve doing at
the rotating patch's corners — here succeeding rather than breaking.

**The cross-scheme comparison does not support it, and should not be quoted as
if it did.** The `deltaSPH` ratio is 0.797 at t=0.02 and 0.677 at t=0.5 — it
*crosses* `divergenceFree`'s 0.677 and 0.749, so the ordering reverses between
the two times. The ratio is evidently dominated by how thin and spread the
sheet is at that instant, not by the scheme, which is unsurprising given the
two runs are at visibly different flow states by any fixed time (fronts 0.6
apart at t=0.5). It discriminates nothing here. (`deltaSPH`'s density column
says nothing either way — it integrates rather than sums.)

So the compaction reading rests entirely on the within-`divergenceFree`
recovery, which is real but is one observation. The clean test is DF against
itself with the shift disabled at surface particles: `divergenceFree.py` has
`pressureB[surfaceIndicators == 1] = 0.0` commented out for exactly this, and
§4's "Known-open" entry explains why it is untestable as written.

**Where this leaves the scheme.** It does free surfaces, on the easy case,
without breaking — but it delivers half the run-out and dissipates most of the
flow's energy at the moment the dam break is supposed to convert it. That is
the same over-dissipation Part 18 measured on the bounded case (19% of KE in 6
time units there, 88% in 0.3 here), in the regime that exposes it.

### Open items, ranked

1. ~~**Run the 2x2x2x2.**~~ ~~**Land `minShift`-on-bounded + `staticBoundary`
   as the defaults.**~~ **Both done — Part 13 and Part 14 above.** The
   defaults shipped are `minShift` wherever there is no free surface and
   `staticBoundary` on *both* solvers. `consistent` and `akinciBoundaryVolume`
   did not land, as planned: the first is inert against `static` and the
   second diverges.
2. ~~**Move `BoundaryOperatorTerms` to `RelaxedJacobiSolverConfig`**~~ —
   **done, and the split it proposed is rejected.** The setting is per-solver
   now, but `pressureSolver = staticBoundary` with `divergenceFreeSolver =
   full` measures 1.45x *worse* than both (6.49e-3 vs 4.48e-3), so both is the
   default. The reasoning that endorsed the split ("the win belongs to
   `solveIncompressible`") was a clamp-gauge result, like every other boundary
   ranking this document has had to invert.
3. **The divergence-free half-state's contraction collapse is unexplained —
   and, under the gauge fix, no longer fatal.** Under `staticBoundary` applied
   to the divergence-free solve alone *with the clamp*, each solve removes
   ~20% of its incoming residual against ~50% under `full`; the incoming
   residual creeps 2.4e-2 → 3.8e-2 over 250 steps and then detonates (max
   `|a_p|` 19.8 → 1.04e4 at step 276, NaN at 282). Each solve still converges
   internally. Three mechanisms tested and eliminated (§2). **Part 14 removed
   the observable**: the same half-state under `minShift` runs all 901 steps to
   t=6.30. What is unmeasured is whether the per-sweep contraction is still
   halved there — if it is, the mechanism is still live and merely survivable,
   and it is worth understanding before any third solve is added; if it is not,
   this item closes. `probe_boundaryOperatorTerms.py --mode diag` measures it.
4. ~~**Wire `rtol` into the relaxed-Jacobi path**~~ ~~**Then the
   one-sided-average vs floored-average criterion behind a flag.**~~ **Both
   done — Part 15.** Both are in and both are inert, and the measurement they
   enabled says neither was the defect: the constant-density solve does not
   converge in any norm, so no residual criterion can end it (new §1.7). What
   is left of this item is *documentation*, not code — `maxIterations` on
   `pressureSolver` should be named and described as the shift gain it is.
5. **Move `computeMdbcPressure` inside the solver iteration** ([B] Alg. 1
   recomputes `p_b` from the current iterate every sweep, so it is a pure
   function of `p_f` with no state and no lag) and add [B]'s **SVD safe
   inversion** on the MLS gradient system. This codebase falls back on a
   neighbour-count threshold (9) with no conditioning guard, and Part 2's
   worst offender had `numNeighbors = 22` with `|grad p| = 153` — a count does
   not detect a co-linear neighbourhood. *But see §4's note: if `consistent`
   lands, `mdbcMlsPressure` should be deprecated rather than repaired.*
6. **`relaxationFactor = 0.3` has a 4% stability margin on a bounded state**,
   not the ~15% `JacobiRelaxationMode`'s docstring quotes from the TGV family
   (`omega`/window = 0.957). Independent of everything else and probably the
   cheapest robustness win available. `probe_boundaryOperatorTerms.py --mode
   spectrum` measures it on demand.
7. **Test the mDBC hypothesis for `DensityEvolution.hybrid`.**
   `computeMdbcDensity` runs at the top of the step on the *carried* density,
   so under `hybrid` the boundary rows are extrapolated from a drifted field.
   The periodic case has no mDBC, which is exactly the difference. Cheap test:
   re-sum for the extrapolation only.
8. ~~**Port [C]'s shear-wave decay case**~~ **Done — Part 16.** `shearWave` is
   registered, swept and tested. It confirmed the half-viscosity explanation
   behind `tgv`'s 0.55x independently (0.49x, at two viscosities, on a flow
   with no pressure error to confuse it), reproduced §1.1's structural density
   bias as a **resolution-independent** volume error, and narrowed §1.2 — the
   three `ShiftApplication` modes dissipate identically here. What is left of
   this item is the comparison against [C]'s Fig. 3/Fig. 4 themselves, which
   needs the paper: nothing in this repository has its curves, and nothing here
   hard-codes numbers read off them.
9. **Warm start — on the divergence-free solve only.** [BK] does a full one
   (worth ~3x in iteration count), [I] and [C] do `0.5 p(t-dt)`, [B] does none;
   this codebase does none (`incompressible.py:167`). Part 15 splits the item:
   the divergence-free solve genuinely converges, so warm-starting it is the
   ordinary optimisation and is now unblocked. The constant-density solve is an
   integrator whose pressure grows linearly in the iteration count, so
   warm-starting *it* would carry that ramp across steps — the cold start is
   load-bearing there, and this is no longer a "wait and see" but a "do not".
10. **Rename `dfsph.py`/`dfsph_step` → `vdps.py`** (§1.3). Zero-risk, and the
    registered scheme name already needs no change.
11. **The scheme split.** Add a real `dfsph` scheme once 1–4 land. Fully
    specified by [BK] Alg. 1: density solve → integrate → divergence solve; no
    position shift; full warm start; tolerances 1e-4 / 1e-3; [B]'s MLS
    boundaries recomputed inside each iteration. Shares `computeAlpha`, both
    IISPH kernels, the Jacobi loop, and every case. **Do not build it before
    the stopping criterion and the boundary defaults land** — it is the
    formulation that puts the density solve into momentum, so it is the one
    most damaged by the current defects.
12. **`MINRES` without the non-negativity clamp** — 2.15x on periodic density
    error for 1.73x wall time. Needs (a) a free-surface test, since a shifting
    potential that may go negative can *pull* particles together, and (b) an
    equal-cost comparison against `relaxedJacobi` at `maxIterations=128`.
    Opt-in at best.

### Known-open, lower priority

- **`rotatingSquarePatch` corner density loss** (Part 3). Resolution-
  independent `rho ≈ 0.506` at the four convex free-surface corners, present
  within ~8 steps, under any `dt` or integrator; `--scheme deltaSPH` on the
  same geometry holds 0.9998. **[BK] §5 documents this as a known method
  limitation** ("the density near a free surface is underestimated which
  causes unnatural particle clustering") with a published remedy (ghost
  particles, Schechter & Bridson 2012) — so it is not an implementation bug.
  Two real, independent bugs on that case are ruled out as the primary driver
  but still want fixing: it has **no `Case.timestep` hook** (so `dt` is never
  adapted under any scheme) and it inherits `integrationScheme='rungeKutta2'`.
  The commented-out `pressureB[surfaceIndicators == 1] = 0.0` in
  `divergenceFree.py` is untestable as-is on this case (`detectFreeSurface`
  flags 96/100 particles at nx=32 on this thin patch) and, per Part 21, is not
  actually [BK]'s remedy anyway — it sits in the divergence-free solve, not the
  constant-density one, and it zeros the pressure outright rather than only
  its negative part. **Part 21 measured the real remedy's boundary on
  `dambreak` instead**, via the already-wired `forceShiftPressureGauge`: taking
  the clamp *away* (the opposite direction from strengthening it) NaNs the run
  in 4 steps, so the clamp is necessary, not merely damping. What is left of
  [BK]'s own remedy — Schechter & Bridson's ghost particles, now in
  `literature/` (`schechter2012`) — is the one structural option neither case
  has tried.
- **Nothing enforces `semiImplicitEuler`.** The PPE derivation is specific to
  it. All three incompressible cases set it explicitly, but `CaseSpec`'s
  default is `rungeKutta2` and no code path checks `scheme == divergenceFree`
  against `integrationScheme`. A multi-stage RK integrator would solve each
  stage as if it were final and then blend, so the blended velocity is not
  divergence-free. Candidate: a one-line assert/warn in `dfsph_step`.
- **`solveIncompressible` should raise on the Krylov path**, the way the
  `JacobiRelaxationMode.optimal` path already does, since it routes through
  `solvePressureKrylov(..., gauge='nonnegative')` — the clamped-solve
  combination both papers rule out.
- **Two mDBC slip defects** (`modules/mdbc/velocity.py`). `freeSlip` computes
  `u_f - 1*(u_f.n)n` while the comment above it says `u_f - 2*(u_f.n)n`;
  `noSlip` applies the same projection undocumented. Fixing measures *worse*
  (§2), but the comment and the code should be made to agree.
  Separately, **`noSlip`'s `2 u_wall` term is dead code**: it reads the *ghost*
  row's velocity, and `rigidBody/update.py:51-56` refreshes ghost velocities
  only for `BCType.constant`. Measured on `lidDrivenCavity`, ghost `|v|` is
  exactly 0 while boundary `|v|` reaches 1.0. It is latent today only because
  `enforceDirichlet` runs after `computeBoundaryVelocities` and re-imposes the
  lid. It would bite the first moving no-slip wall not backed by a Dirichlet.
- **Two-way coupling is absent.** [BWJ23] Eq. 35's `f_{k<-i}` is never applied;
  `integrateRigidBody(rigidBody, 0, 0, dt)`. Fine today (all walls static), but
  a moving-body case would silently get one-way coupling.
- **`DensityEvolution` + `BoundaryPressureMode.plain` is a trap** — `plain`
  skips the mDBC extrapolation and the non-summation modes skip the re-sum, so
  `kind != 0` rows would never be updated at all. Documented in the enum, not
  guarded in code.
- **Ghost particles (`kind == 2`) are lumped in with boundaries** by the
  `kj == 0` test. Correct for this scheme (`dfsph_step` freezes them too), but
  no measurement here separates them.
- **Two intermittent test flakes**, both pre-existing, neither a regression:
  `test_implicitShiftingComparison.py`'s `implicitShiftAutomatic` assertions
  (~1 run in 3; relative density std falls smoothly for six steps then jumps
  0.0145 → 0.217 on step 7) and
  `test_incompressibleKrylov.py::test_minresGivensMatchesDenseLstsq`.

---

## 5. The literature, and where this codebase deviates

**Sources read in full.**

- **[C]** Cornelis, Bender, Gissler, Ihmsen, Teschner, *An Optimized Source
  Term Formulation For Incompressible SPH* (TVCJ 2018/19) — VD+PS. **This is
  the paper this scheme implements.**
- **[BK]** Bender & Koschier, *Divergence-Free SPH* (SCA 2015) — DFSPH proper.
- **[I]** Ihmsen et al., *Implicit Incompressible SPH* (TVCG 2014) — IISPH;
  the solver this codebase's Jacobi loop discretises.
- **[B]** Band, Gissler, Peer, Teschner, *MLS pressure boundaries* (C&G 76,
  2018).
- **[BWJ23]** Bender, Westhofen & Jeske, *Consistent SPH Rigid-Fluid Coupling*
  (VMV 2023) — derives DFSPH from a density constraint; **this is the
  derivation behind `staticBoundary`.**
- Plus SPlisHSPlasH source (`~/dev/SPlisHSPlasH`) as reference implementation.

**Nothing is unavailable any more.** As of 08-29 all five above, plus the four
previously listed here as unavailable — Adami et al. 2012 (wall BC), Akinci et
al. 2012 (rigid-fluid coupling), Ihmsen et al. 2010 (adaptive timestep), Adami
et al. 2013 (transport velocity, the background-pressure question) — are in
`literature/`, along with 27 others. `literature/MANIFEST.md` maps the
shorthands above to bib keys and filenames and says what each newly-present
paper unblocks; `literature/ABSTRACTS.md` is the searchable index. Being
*present* is not being *read*: the claims below still carry whatever provenance
they carried before, and re-checking them against the documents is now possible
rather than done.

### The published CFL is calibrated against a metric a free surface silences

[BK]'s `dt <= 0.4 d/|v_max|` was validated alongside a convergence tolerance on
an *average density error*, and that average is computed compression-only.
SPlisHSPlasH, `TimeStepDFSPH.cpp:603-618`:

```cpp
const Real residuum = min(s_i - aij_pj, static_cast<Real>(0.0));   // r = b - A*p
density_error -= density0 * residuum;
...
avg_density_err = density_error / numParticles;
```

`min(..., 0)` keeps only compressive residuals; the sum is then divided by
**all** particles. A particle in expansion contributes exactly zero to the
numerator and a full unit to the denominator, so the reported error is diluted
in proportion to how much of the domain is expanding — and a free surface is
precisely the configuration that puts a large population there. Measured,
nx=128:

| case | fraction `rho < rho0` | honest `mean｜rho-rho0｜` | compression-only mean | dilution |
|---|---|---|---|---|
| `randomFlowIncompressible --bounded`, 200 steps | **21.7%** | 3.00e-3 | 2.66e-3 | **1.13x** |
| `rotatingSquarePatch --scheme divergenceFree`, 20 steps | **100.0%** | 3.29e-2 | **0.00e+00** | **total** |
| same, 60 steps | 93.9% | 4.53e-2 | 9.74e-5 | **465x** |

**On the free-surface case the metric reads exactly zero while the true mean
density error is 3.3%** — not approximately zero, zero, because after 20 steps
not one of the 1764 fluid particles is above rest density, so every term in the
numerator is clipped away. The bounded case is the other extreme: the metric is
within 13% of honest.

Two caveats. `rotatingSquarePatch` under `divergenceFree` is itself broken
(§1.9), so its 3-4% error is not what a healthy free-surface simulation would
show — but the 100%-below-`rho0` structure that silences the metric is a
property of sampling a free surface, not of that bug. And **this codebase's own
metric is worse than the published one**: `clamp(-residual, min=-threshold)`
lets an expanding particle *subtract* `threshold` rather than contribute zero.

The consequence is a calibration-scale mismatch of two to three orders of
magnitude between the scene type the constant was tuned on and the scene type
this codebase runs. It is a concrete reason to expect 0.4 to be permissive
here, and the reason §4 item 4 (the stopping criterion) is not cosmetic.

**The sweep that prediction called for has been run** (§10 has the table and the
cost analysis). The part that belongs here is the dilution column itself:
`frac rho < rho0` measures **5-13% on the bounded case**, against the ~100% a
free-surface scene shows in the table above. That is the calibration mismatch
quantified on the scene this codebase actually runs, and it is why the published
0.4 is permissive here without being wrong where it was tuned.

The sweep's own answer — a 2.83x gain for the first halving and ~1.25x for each
one after, i.e. a knee at 0.2 — was measured at the *shipped* boundary
configuration, where the near-wall band dominates the error. §4's factorial
then removed 40x of that band, and **the re-run at the landed defaults says the
knee was the boundary treatment, not the timestep** (§4, Part 14): halving 0.4
now buys 1.17x instead of 2.83x. The published constant is no longer
permissive on this case, because the error it was permitting is gone.

### Answers to the questions that drove the literature sessions

- **Q1, the unreachable setpoint.** [C]'s PS solve has exactly the same
  structural bias and does not avoid it — it avoids *integrating* it (§6: "we
  do not update the velocities using the solution of the PPE solver with the DI
  source term … it is just a resampling"). Not warm starting, not one-sided
  clamping, not rest-density recalibration. See §1.2.
- **Q2, which density.** Summation, in both papers, with a one-step divergence
  prediction on top ([C] Eq. 11, [B] Alg. 1). Neither integrates density, so
  `DensityEvolution.summation` is the paper-faithful default. One difference
  that **favours this codebase**: `finalize` recomputes the summation density
  at the *shifted* positions before the solve, where [C] predicts from the old
  positions. Do not "fix" that toward the paper. Rest-quantity calibration
  against the sampled configuration is real but **boundary-only** ([B] Eqs.
  6-7), and [B] §6.1 blames its planar-boundary coefficient `beta` for that
  variant's worst error.
- **Q3, shifting vs the constant-density solve.** No published variant does
  both at full strength — that is [C]'s whole point (§1: prior two-PPE
  combinations "typically result in inconsistent particle positions and
  velocities"). Measured here: doing both grows `tgv`'s kinetic energy 6.6x
  over 200 steps. `finalize` correctly drops the shift under `inStepVelocity`.
- **Q4, boundary coupling.** **There is no published VD+PS × mDBC/MLS pairing**
  — [C] ships Akinci one-layer boundaries and runs its headline test on a
  *periodic* domain "such that boundary handling does not influence the
  solution". This codebase's combination is novel, which is the context for
  every boundary-mode ranking here being formulation- and `dt`-dependent. The
  boundary-pressure feedback loop `mdbcPressureRelaxation` damps is a
  **documented** failure mode of holding boundary pressure as *state* ([B]
  §3.3, contrasting Pressure Boundaries' `omega_b` with its own
  recompute-in-loop MLS).
- **Q5, reference values.** Neither [BK] nor [B] reports a Taylor-Green decay
  rate, so `tests/test_physics.py`'s 0.55x has no published counterpart. [C]
  supplies the benchmark that was actually wanted — the shear-wave decay
  (§4 item 8).
- **Q6, timestep near boundaries.** **Neither paper carries a wall-proximity
  constraint.** [B]'s answer to bad wall behaviour is a better boundary
  pressure, not a smaller `dt` (§6.6: pressure mirroring "requires a time step
  that is half as large compared to our MLS extrapolation"). The real finding
  was the units mismatch — see §1.6. This supersedes Part 5's costed
  wall-aware `dt` proposal (~2.4x throughput) entirely.
- **Q7, background pressure.** **Open.** Two adjacent data points: [B] Eq. 3's
  Adami-style extrapolation carries an explicit hydrostatic term (published
  precedent for a background pressure that is *set*, not allowed to drift), and
  [C]'s PS solve is itself a de-clumping potential never applied to momentum.
  Needs a transport-velocity paper.

### Remaining deviations, with status

| deviation | status |
|---|---|
| `cflFactor` applied to `h`, not `dx` | **fixed in the working tree** (Part 12, uncommitted) |
| `computeAlpha` / operator include static-boundary reaction terms | `BoundaryOperatorTerms.staticBoundary`, **opt-in**; belongs per-solver (§4 item 2) |
| Boundary rows enter the solve at mDBC-extrapolated `rho_k` (1.3+), not `rho0` | `BoundaryPressureMode.consistent`, **opt-in** (§4) |
| MLS boundary pressure computed once per step and carried as under-relaxed state | **open** (§4 item 5) |
| No SVD guard on the MLS gradient fit | **open** (§4 item 5) |
| No warm start | **open** (§4 item 9), deliberately deferred |
| `omega = 0.3` against both papers' 0.5 | **explained, not a bug.** `INCOMPRESSIBLE_SOLVER_PLAN.md`: `omega < 2/rho(D^-1 A)` with `rho(D^-1 A) ≈ 5.636`, a degenerate high-frequency lattice cluster, `dt`-invariant. Candidate contributor: `n_h = 4` gives ~50 neighbours in 2D against [I] §2.2's "typically 30-40", and the IISPH operator reaches neighbours-of-neighbours. |
| Krylov routed at the clamped solve | should **raise** (§4, known-open) |
| Stopping criterion is absolute and floors negative contributions | **open**, §1.7 / §4 item 4 |
| Two-way rigid coupling absent | **open**, no case needs it |

---

## 6. Configuration surface

Every switch added by this work, with its default and what it measured.
All are round-tripped through `incompressibleConfigToDict` /
`dictToIncompressibleSPHConfig`.

| setting | default | notes |
|---|---|---|
| `boundaryPressureMode` | `mdbcDensity` | `plain` / `mdbcDensity` / `mdbcMlsPressure` / `consistent`. `consistent` was the best measured under the clamp gauge and is **inert** (0.007%) against `staticBoundary` once the gauge is fixed (Part 13), so it did not land; `mdbcMlsPressure` is the worst and should be deprecated. |
| `boundaryOperatorTerms` (per solver, on `RelaxedJacobiSolverConfig`) | **`staticBoundary`** on both | The published formulation, and one of the two defaults Part 14 landed: with `minShift` it is 40x the old default, and setting it on only one solver is 1.45x (PS only) or 16x (DF only) worse than on both. `diagonalOnly`/`operatorOnly` are deliberately-mismatched diagnostics — `diagonalOnly` runs the wall's Jacobi step 1.6x too large and NaNs in 47 steps. |
| `boundaryOperatorTerms` (on `IncompressibleSolverConfig`) | `None` | Bundle-level override: `None` means "each solver's own", any value forces both. Every A/B in this document sets it, so every recorded row still means what it says. |
| `akinciBoundaryVolume` | `False` | `consistent` only. Measured `m~/m_nominal` mean 1.102, max 1.456 on the five-layer band. Best row in the table *inside the operator*; fatal as actual mass (§2). |
| `shiftPressureGauge` | **`minShift`** | The Part 4 fix. `nonNegativeClamp` is the historical clamp and stays selectable. Scoped to solves with **no free surface** — Part 14 dropped the pinned-row half of that scoping, which is what makes it reach the bounded case at all. |
| `forceShiftPressureGauge` | `False` | Bypasses what is left of that scoping, i.e. the free-surface half. Its original target, the pinned-row half, is gone: half that guard's justification measured false (§1.5) and the other half's evidence was taken at 3x the CFL (§3.6). **The free-surface half is now measured (Part 21) and stays off**: forcing it on `dambreak` NaNs in 4 steps. |
| `shiftApplication` | `positionShift` | The paper-faithful default, and **settled in Part 18**: at a pinned `dt` the velocity modes buy 2x lower density error for **2.1x the kinetic-energy loss** on an inviscid case (41% retained against 81%), and the wall behaviour that used to justify them is identical — zero particles past the wall for all three modes. Its old justification (3.2x `tgv` dissipation) was withdrawn in Part 17 as unreproducible and resolution-dependent. |
| `densityEvolution` | `summation` | `continuity` (WCSPH standard) fails everywhere but `tgv`; `hybrid` matches `summation` exactly where support is complete, for ~21% less wall time on `tgv`, and dies at 286 steps at an mDBC wall (§4 item 7). |
| `mdbcPressureRelaxation` | `0.3` | Load-bearing for `mdbcMlsPressure` — at 1.0 it NaNs in 7-8 steps. Never swept; chosen to match the solver's own `relaxationFactor`. |
| `convergenceCriterion` (per solver) | `flooredOneSided` (PS) / `meanAbsolute` (DF) | Each solver's historical statistic, now one setting instead of two inline tests. `oneSided` is the published form. On the constant-density solve the swap is **bit-identical** (its criterion never fires); on the divergence-free solve it collapses the solve to 3.0 iterations for 1.53x the density error (§1.7). |
| `rtol` (relaxed-Jacobi path) | `1e-5` | Now a *disjunct* alongside the absolute test, same contract as the Krylov path. Inert at the default: `mean\|r\|/mean\|b\|` is ~0.97 at the last iteration on the bounded case. |
| `maxIterations` (`pressureSolver`) | `64` | **A gain, not a budget** (§1.7). The constant-density solve does not converge; its pressure grows linearly in the iteration count, so this sets the shift amplitude. Measured on the frontier: 128 buys 1.26x for 1.6x the time, 32 costs 1.46x. |
| `mdbcNoPenetrationShift` | `True` | Removing it is worse (§2). |
| `integrateRho` | `False` | Legacy alias; `resolveDensityEvolution` maps `True` → `continuity`. |
| `cflFactor` (incompressible cases) | **`0.4`** | Working tree only; multiplies `dx`. See §7. |

`ShiftApplication` comparison, bounded case, nx=128. **Superseded by Part 17 —
kept for the history, not for the ranking.** Every row is at the legacy CFL,
which Part 13 showed has no viable configuration at all, and predates both
Part 14 defaults. Part 17's table is the one to read:

| mode | near-wall `\|rho-1\|` | bulk | penetrating | `rho` range | outcome | `tgv` decay/analytic *(not reproducible — Part 17)* |
|---|---|---|---|---|---|---|
| `positionShift` (default) | 0.30 | 0.113 | 4506 | [0.139, 2.452] | NaN t=5.54 | **0.55x**, monotone |
| `positionAndVelocity` | 3.3e-2 | 1.2e-3 | 239 | [0.936, 1.147] | t=8.0 | 3.28x, non-monotone |
| `inStepVelocity` | **9.7e-3** | **6.6e-4** | **63** | [0.986, 1.140] | t=8.0 | 3.26x, non-monotone |

For comparison: `positionShift` at the **published** CFL reaches t=8.0 with
near-wall `|rho-1|` = 2.6e-2 — better than `positionAndVelocity` and without
its dissipation — for 4.2x the steps.

`DensityEvolution`, nx=128, 900 steps, published CFL. Note the split between
the **carried** density (what solvers and diagnostics see) and a fresh
summation on the same positions:

| case | mode | steps | `\|carried-1\|` | `\|true-1\|` | drift | DF resid | wall s |
|---|---|---|---|---|---|---|---|
| bounded | `summation` | 901 | 7.04e-3 | 7.04e-3 | 0 | 7.52e-2 | 114.6 |
| bounded | `continuity` | **63** | 7.06e-3 | 4.30e-2 | 4.74e-2 | 1.46e-1 | 3.9 |
| bounded | `hybrid` | **286** | 2.27e-2 | 2.52e-3 | 2.14e-2 | 1.67e-1 | 30.2 |
| periodic | `summation` | 901 | 1.93e-3 | 1.93e-3 | 0 | 8.79e-3 | 73.2 |
| periodic | `continuity` | **470** | 9.05e-3 | 3.66e-2 | 3.93e-2 | 7.25e-1 | 38.2 |
| periodic | **`hybrid`** | 901 | 3.52e-2 | **1.92e-3** | 3.33e-2 | **8.49e-3** | 72.6 |

---

## 7. Bugs found and fixed along the way

Landed, verified, and not worth re-litigating — but worth knowing about.

| bug | worth |
|---|---|
| The implicit shift computed its own kinematic velocity correction every step and then discarded it (`self.state.velocities -= proj_vel` commented out) | Fixed the original nx=128 step-720 divergence outright (commit `122d326`) |
| Boundary-row masking froze pressure at literal `torch.zeros_like` rather than at `currentState.pressures` | Made `mdbcMlsPressure` a **silent no-op** — `mdbcDensity` and `mdbcMlsPressure` were bit-identical at every step. Fixed in all four solver paths; Krylov got proper Dirichlet lifting (`b = source - A(boundaryOnly)`, re-pin at the end) |
| The exposed loop: with the projection actually live, boundary pressure doubled every step to NaN in 7 steps (`numNeighbors=22`, `\|grad p\|=153`) | Fixed with `mdbcPressureRelaxation` |
| [C] Eq. 17's velocity resample contracted `dx` over the **wrong axis** and applied it with the **wrong sign** (`einsum('nij,ni->nj')` and `-=`, i.e. `-A^T dx` where `+A dx` was wanted) | Verified against the compiled kernel with an asymmetric linear field; real bug, correctly fixed, but moves neither open metric — `dx = dt^2 dvdt` is second-order in `dt` |
| `drhodt` was evaluated on the **pre-projection** velocity, so an integrated density re-accumulated exactly the divergence the solve had just removed | **1800x** on the divergence-free residual (2.61e+2 → 1.46e-1); only reachable once `DensityEvolution` made the branch real |
| `shiftProperties.active` ran a full deltaSPH shifting solve every step and **discarded it** — `finalize` bound it to `dx` then shadowed `dx` | Any prior test of "the deltaSPH shift on top" was measuring a no-op. Fixed (`dxDeltaShift`), then the configuration was tested and rejected |
| `densityEvolution`, `boundaryOperatorTerms`, `forceShiftPressureGauge` were not serialised by `incompressibleSPHConfigToDict` | Silently reset to defaults on a TOML/HDF5 round-trip |
| `akinciBoundaryMass`'s `BoundaryToBoundary` sum is zero for fluid and ghost rows, and `rho0/0` is a simulation-ending mass | Never reached a neighbour sum, but is now a fallback rather than a landmine |
| `integrateRho`'s `True` branch was dead — `finalize` re-summed unconditionally, twice per step | Now real via `DensityEvolution` |
| `self.state.surfaceIndicator` assigned where the field is `surfaceIndicators` | `finalize`'s `detectFreeSurface` result is written to a typo'd attribute and discarded. **Not fixed.** |
| `finalize` recomputes plain summation densities and never re-applies mDBC, so the shifting solve sees systematically-low boundary rows | Real inconsistency; applying mDBC there changes nothing measurable (353 vs 349 penetrating). **Not fixed.** |

---

## 8. Tooling

All in `scripts/`, all confirmed working, none require source edits to use.

**Start here for anything about the current open items:**
- `probe_boundaryOperatorTerms.py` — `--mode diag` (exact diagonals binned by
  wall depth), `--mode spectrum` (`rho(D^-1 A)` and the `omega` margin),
  `--mode dfTrace` (per-solve first/last residuals), `--solvers` to scope
  `staticBoundary` to one solve.
- `probe_fourWayDefaults.py` *(new, Part 13)* — the `cfl x gauge x boundary`
  factorial. `--cfls/--gauges/--boundary` subset it; rows print as they
  complete. Ten of its cells are prior published rows, so it doubles as a
  regression check on this whole document.
- `probe_perSolverBoundaryTerms.py` *(new, Part 14)* — the two solvers'
  `boundaryOperatorTerms` crossed under `minShift`, which is what settled the
  landed default. Three of its five rows are Part 13 cells and it prints an
  explicit reproduction check against them, so it is the cheapest regression
  test for "did anything under the incompressible path move". `--rows` subsets
  it.
- `probe_stoppingCriterion.py` *(new, Part 15)* — `--mode trace` evaluates all
  three criteria **along one fixed iterate path** (early exit disabled, so the
  runs are bit-identical and only the reading changes) and prints each
  statistic and the pressure range per iteration; `--mode budget` sweeps the
  two solvers' `maxIterations` (`--budgets 64:32 128:16`); `--mode ab` crosses the two
  solvers' criteria end to end at the shipped tolerances, which is what "adopt
  [BK] Alg. 3" actually means. Point it at
  `--case kolmogorovIncompressible --extra` for the periodic contrast, which is
  where the non-termination claim breaks.
- `probe_dambreakIncompressible.py` *(new, Part 19)* — the dam break under
  both schemes, reporting surge front, column height, kinetic energy and the
  size of the population that reads as free surface under a summation density.
  `--schemes divergenceFree` for one of them.
- `probe_dambreakEnergyBudget.py` *(new, Part 22)* — the per-step
  kinetic-energy budget of `dambreak --scheme divergenceFree`, closing `dKE`
  exactly and splitting it the two ways Part 22 reports: the work form
  (first-order works + quadratic remainder, binned in x for the *where*) and
  the sequential form (exact KE change of adding each force in step order, for
  the unambiguous *which channel*). The `chain` column is the completeness
  check (captured forces vs the integrator's real update). `--tLimit/--bins/
  --interval/--fixedDt` subset it; the dissipation window runs from the KE
  peak to t=0.8 by default.
- `probe_shiftApplication.py` *(new, Part 17)* — the three modes across `tgv`,
  the bounded case and `shearWave` in one table. `--nx/--tgvSteps` matter more
  than usual here: the `tgv` ratio is resolution-dependent for two of the three
  modes, which is itself the finding.
- `probe_shearWave.py` *(new, Part 16)* — the shear-wave case's three
  questions: `--mode shift` crosses the `ShiftApplication` modes (add `--nu` to
  do it against a real analytic decay instead of a stationary one),
  `--mode resolution` checks what converges and what does not, `--mode viscous`
  grades the applied viscosity against the prescribed one.
- `probe_consistentCoupling.py` — [BWJ23]'s `consistent` mode end to end.
- `probe_cflCondition.py` *(new, Part 12)* — `--mode verify` checks that
  `dt |v_max| / dx == cflFactor` exactly whenever the advective term binds
  (run: 39/40 steps, 0.4000 against 0.4); `--mode sweep` runs each `cflFactor`
  to the same simulated time and reports the sub-rest-density fraction
  alongside the error, printing rows as they finish (run: 0.4 / 0.2 / 0.1, §10).
  It must not be given `--nSteps` -- `runner.py:246` only honours `tLimit` when
  `nSteps` is absent, so a step cap silently converts a time-matched sweep into
  a step-matched one, which is the comparison the mode exists to avoid.
- `probe_boundedIncompressibleBlowup.py` — step-by-step wall penetration,
  now with kinetic energy (`ke`) alongside `vMax`, since `vMax` is one particle
  and the `ShiftApplication` question is whether the *flow* is damped.
  **`--fixedDt` is mandatory for any A/B that changes the velocity field** —
  the CFL hands a damped flow a larger `dt`, and Part 17 read that as a
  property of the modes.
  per-step worst particle, wall-depth density profile. Knobs for everything
  that turned out not to matter, so they stay cheap to re-check: `--mode`,
  `--noPenShift`, `--shiftCap`, `--cflFactor`, `--case randomFlow` for the
  deltaSPH control.

**Solver and gauge:**
- `probe_shiftPressureGauge.py` — end-to-end A/B of the two gauges through the
  real solver, either case (`--extra=--bounded`).
- `probe_incompressibleGaugeDrift.py` — standalone reimplementation of
  `solveIncompressible`'s loop, verified byte-identical to the real solver on
  the baseline. `--gauge {clamp,center,center-clamp,minshift,quantile,none}`,
  `--project-source`, `--null-test`, `--no-clamp`, `--maxIters`, `--jitter`,
  `--setpointEps`. Has no `--tolerance` knob; add one if §4 item 4 is run.
- `probe_incompressiblePressureSolvers.py` — Krylov methods end-to-end inside a
  case (the prior plan's numbers are all single solves on a seeded state).
- `probe_operatorDiagonal.py` — exact diagonal extraction, one matvec per row.
- `probe_relaxedJacobiOmega.py` — the `omega` stability window.

**Density and sampling:**
- `probe_densityBiasVsDisorder.py` — the no-dynamics demonstration that the
  bias is structural (§1.1).
- `probe_densityEvolution.py` — `DensityEvolution` modes, carried vs true.
- `probe_initialSampling.py` — as-sampled uniformity and mass normalisation.
- `probe_densitySign.py` — signed vs unsigned bulk bias on the running case.

**Boundaries:**
- `probe_wallSupportCompleteness.py` — Shepard sums and `|A.1|/|A.rand|` by
  wall depth (§1.5).
- `probe_boundaryVelocityModes.py` — `BCType` A/B; `--mode verify` decomposes
  each condition against the wall normal and grades it against the published
  integer slopes.
- `probe_dfsphWallDensityProfile.py` — `|rho-1|` binned by signed wall distance,
  DFSPH vs deltaSPH.
- `probe_randomFlowIncompressibleBoundaryModes.py` — the three legacy boundary
  modes at matched physical time.
- `probe_mdbcMlsPressureInstability.py` — re-runs `computeMdbcPressure`'s
  internals per boundary particle on the steps before a NaN.

**Case-specific / historical:** `probe_tgvShiftGauge.py`,
`probe_pressureGaugeDrift.py`, `probe_kolmogorovIncompressible.py`,
`probe_kolmogorovAdjacencyRebuild.py`, `probe_kolmogorovContinuation.py`,
`probe_kolmogorovSpinup.py`, `probe_kolmogorovIncompressibleVelCorrection.py`.

**Always run after touching a `@wp.kernel`/`@wp.func`:** the `gradcheck` skill
(`scripts/gradcheck_incompressible.py`). Part 9 added a second case there
covering the `includeBoundaryReaction=False` branch with mixed `kinds` in one
launch — a conditional accumulation inside a neighbour loop is exactly the
shape that file is about.

---

## 9. Session index

One line each, for locating the full write-up in git history.

| part | date | outcome |
|---|---|---|
| 1 | 08-26 | Kolmogorov nx=128 step-720 divergence — the fix (the dropped velocity correction) was already in commit `122d326`; verified to 1600 steps. |
| 2 | 08-26 | mDBC boundary handling for the incompressible scheme: `BoundaryPressureMode`, solver-row masking in all four paths, `computeMdbcPressure`, and `randomFlowIncompressible` — the first case to sample `kind==1` under `divergenceFree`. Two bugs found and fixed. Its boundary-mode ranking is **retracted** (§3.1). |
| 3 | 08-26 | `rotatingSquarePatch` corner loss root-caused, not fixed; `integrateRho`'s dead branch found; the [C] Eq. 17 sign/axis bug fixed. |
| 4 | 08-26/27 | The periodic pressure-gauge drift: integrator wind-up against an unreachable setpoint. `ShiftPressureGauge.minShift` landed **as the default**. |
| 5 | 08-27 | Bounded DFSPH stability: the wall is late, not weak (§1.6). `ShiftApplication.positionAndVelocity` and `inStepVelocity` landed opt-in. |
| 6 | 08-27 | Scheme-naming/architecture question opened; the DFSPH × boundary-mode matrix measured. Nothing restructured. |
| 7 | 08-27 | Literature session ([C], [B]). Headline: **this is VD+PS, not a mis-named DFSPH** (§1.3). Its Q1 tolerance argument is retracted. |
| 8 | 08-27/28 | The originals ([BK], [I]). Setpoint and `alpha` hypotheses both falsified; **the published CFL constant landed as the one positive result**; MINRES-without-clamp measured; the wall-truncation premise corrected. |
| 9 | 08-28 | The terms the papers do not compute for static boundaries: `BoundaryOperatorTerms.staticBoundary`, 5.9x at the published CFL. Its boundary-velocity explanation is retracted. |
| 10 | 08-28 | Initial sampling (exact; the mass is not) and `DensityEvolution`. The `drhodt` pre-projection bug, worth 1800x. |
| 11 | 08-28 | [BWJ23] — the derivation behind Part 9. `BoundaryPressureMode.consistent`, the best configuration measured. |
| 13 | 08-28 | The factorial (§4) and the CFL sweep (§5). `minShift`+`staticBoundary` is 40x the default and 5.4x better than the two changes composing independently; `consistent` is inert and `akinci` diverges once the gauge is fixed; the legacy CFL has no viable configuration at all. Ten prior rows reproduced exactly. |
| 12 | 08-28 | The CFL condition rewritten in [BK]'s units (particle diameters) and **landed as the default**; verified per step and bit-for-bit against the old units. The compression-only error metric measured as diluted 465x on a free surface and 1.13x on the bounded case (§5). |
| 19 | 08-29 | A dam break under `--scheme divergenceFree` (§4). It works — 3000 steps, no divergence, a recognisable collapse, the first free surface this scheme has not broken. But the run-out is half `deltaSPH`'s speed and 88% of the kinetic energy is dissipated at the moment the fall should become run-out. The free-surface density deficit (0.518 at t=0.02) disappears while the surface stays geometrically a surface. |
| 20 | 08-29 | `dambreak`'s incompressible `timestep` hook, landed. The published CFL (0.4) diverges here, unlike every other incompressible case measured — bisected to a safe `cflFactor = 0.2`, which buys 1.7x fewer steps (not the ~5x guessed) at the cost of a worse `rho_max` (1.105 vs 1.004). First hint that the impact itself is the sharp event Part 19's dissipation traces back to. |
| 21 | 08-29 | The free-surface clamp ruled out as the dissipation's cause. [BK]'s own text (read from the PDF, now that `literature/` exists) says clamping negative pressure at the surface *is* their published remedy, not a gap; forcing it off (`forceShiftPressureGauge`, previously untested at the free surface) NaNs `dambreak` in 4 steps rather than reducing the loss. Narrows item 1 to the impact itself. |
| 22 | 08-29 | The energy budget ran and named the channel (§4). The per-step KE budget closes exactly (chain 3.3e-6, gap 0.0); the loss is the incompressibility cycle — the DF projection (−35.8) and the Eq. 17 resample (+27.3) net to −8.5 (85% of the loss), with Monaghan viscosity secondary (−6.4, 64%) and the no-pen shift negligible. Both channels are `divergenceFree`-only, which carries the cross-scheme gap. Item 1 now reduces to the mechanism question: an `nx` convergence of the cycle's net. |
| 18 | 08-29 | The tail measured, and `ShiftApplication` settles (§4). At a pinned `dt`: **zero** wall penetration for all three modes, so §6's whole case for the velocity modes is void; and the velocity modes cost **2.1x** the kinetic energy of an inviscid flow for 2x lower density error. The default stays, on a measurement. Part 17's excursion claim was my own adaptive-`dt` artifact and is retracted. |
| 17 | 08-29 | `ShiftApplication` re-measured at the current defaults (§4). The 3.2x that justified the shipped default does not reproduce — at `tgv`'s own nx=256 the three modes agree to 12%, and the ratio moves 2x with resolution, so it cannot be the resolution-independent residual §1.2 blames. `positionAndVelocity` is better on every sustained metric at equal cost; the default stays on tail behaviour and energy monotonicity. |
| 16 | 08-29 | [C]'s shear-wave case ported (§4). An exact solution with a constant pressure, so dissipation and volume error separate. Confirms `tgv`'s half-viscosity at 0.49x independently; the volume error is resolution-independent (§1.1 from a new direction); the three `ShiftApplication` modes dissipate identically, which narrows §1.2. |
| 15 | 08-29 | The stopping criterion (§1.7). It was the wrong suspect: the periodic cases terminate in 3 iterations, the floor changes nothing, and the constant-density solve does not converge in any norm — it integrates, so `maxIterations` is a gain. The criterion is now one configurable setting across all three loops; no default changed. |
| 14 | 08-29 | The landing (§4). `boundaryOperatorTerms` moved per-solver; the two solvers crossed under `minShift` says **both**, not the PS-only split Part 9 implied. `minShift` on bounded and `staticBoundary` on both are now the defaults, and the bounded case ships at 4.48e-3. The half-state's divergence turns out to belong to the clamp. Three prior rows reproduced exactly. |
| 23 | 08-30 | The three baseline cases landed (`staticBlob`, `impact`, `hydrostaticColumn`). Free space and the collision hold; the quiescent hydrostatic column diverges — the DF projection's source is exactly 0 for a uniform body force, and the position-shift support cycle that is left is an amplifier. Also lands the `relaxLattice` free-surface guard. |
| 24 | 08-30 | The hydrostatic-column failure root-caused (position shift can't sustain a body force) and a reference DFSPH scheme built (`dfsphReference`) that applies both corrections to the velocity: it holds the *exact* hydrostatic gradient for ~15 steps where `divergenceFree` NaNs by 6, confirming the mechanism. Not yet stable — the composed pressure primitives lack a faithful wall force / free-surface gauge. Warm-starting `solveIncompressible` and cold `inStepVelocity` both measured negative. Left as a five-step hardening track under item 3. |
| 25 | 08-30 | Harden-track step 1: the wall-adjacent `kappa` runaway on `hydrostaticColumn` removed. Not a kernel and not the Akinci volume the plan named — on the five-layer band `akinciBoundaryMass` returns the nominal volume, so the correction is inert here. The boundary term in `A p` is simply carried at ~half the weight it needs; a 2x `akinciBoundaryVolumeScale` (new config, default 1.0 = no-op, set to 2.0 by `dfsphReference` only) bounds `kappa_max` at 6.81 and holds `|v|max` < 1 for 25+ steps. The `dp/dy` target is half-met: steps 3 (free-surface gauge) and 4 (contractive divergence solve) are now the co-blockers — the DF solve does not converge and the surface compacts by ~step 40. |
| 26 | 08-30 | Harden-track steps 3 and 4 explored, nothing landed. Free-surface `kappa^v` mask: cleans the `dp/dy` fit but makes the slump faster — a bad trade. Linear optimal-step divergence solve: converges the DF solve for ~13 steps but regresses `staticBlob` hard (needs null-mode handling that §1.5 forbids at a free surface). The finding that redirects the track: the residual slump is driven by the *constant-density* solve's locally lumpy `a_p` (`|a_p|max` 17–45 vs `g ≈ 9.81`), not the divergence solve, so step 2 (faithful DFSPH factor / Akinci boundary force kernel) comes before steps 3–4. Two negative results recorded so they are not re-run. |
| 27 | 08-30 | Harden-track step 2 landed: the faithful DFSPH factor (`SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp::computeDFSPHFactor` — bare-mass `|Σ V_j ∇W|²` over fluid + `|Σ_fluid V_j ∇W + Σ_boundary V_k ∇W|²`, boundary in the vector term only, ghosts excluded) is now its own kernel (`wp_dfsph_factor.py`) and wired into `dfsphReference._factor`, replacing the IISPH `computeAlpha` diagonal. Verified two ways: it is `computeAlpha`'s diagonal / `ρᵢ` exactly (bulk ratio 1.049, wall 1.047 — the expected `1/ρ̄`), and the composed `a_p` is checked against a direct O(N²) torch reference of the standard SPH pressure acceleration to ~5e-7, so `a_p` was already faithful and needed no change. The `hydrostaticColumn` slump **survives** (modest gain: `|v|max` ~1.25→1.17, `rho_min` ~0.68→0.70 over 30 steps; DE still 2 iters, DI still the 32-iter cap). But the faithful factor — correct for SPlisHSPlasH's *linear* Jacobi — **regresses `staticBlob` harder** (`|v|max` 70.9→inf, 20 steps): `dfsphReference`'s *nonlinear* re-summed solve is far more step-size-sensitive, and the ~1/ρ larger step pushes the already-marginal blob over. The blocker moves from step 2 to step 4 (the solve structure). |
| 28 | 08-30 | Harden-track step 4, the linear solve, implemented in `dfsphReference._jacobiSolve` (fixed source from `vEnter` + `aij_pj = Drho/Dt(a_p)` recomputed each iteration + 0.5 relaxation + `max(p,0)`, replacing the nonlinear re-summed fixed point). The first draft **diverged** (DF pressure doubling per iteration, 2e-3→8e9 in 32 iters) and the root cause was a **sign-convention bug, fixed and verified against the reference source, not derived**: SPlisHSPlasH's `delta` operator (difference-form `V_i Σ (v_i−v_j)·∇W`) is the *negative* of the continuum divergence, this codebase's scatter Divergence (inside `_drhodt`) *is* the continuum one (probed: a `div=+1` field gives `_drhodt≈−1.0` in the bulk), their `factor = 1/(Σ|∇W|²·h^k) > 0`, and both solves iterate `p −= 0.5(s−aij_pj)·factor`. With all three signs (source, `aij_pj`, step) corrected to the reference convention, the **physics is now right** (under-compressed column → p=0; over-compressed particles → positive p; compressing flow → positive p) but the Jacobi **does not converge inside its budget**: CD oscillates at the 64-iter cap (err ~0.1), DF diverges by step 2, NaN at step 6. Also found: their convergence metric is one-sided (compression-only `min(s−aij,0)`, with a `<20`-neighbour guard), not the two-sided `mean|resid|` used here — on an under-compressed state the two-sided metric can never reach `tol`, so both solves run to their caps regardless. Blocker: iteration contraction, next is the one-sided metric + a spectral-radius study (omega sweep / iteration budget). |
| 29 | 08-30 | **Step 4 closed: the linear Jacobi now contracts.** Adopted the reference's one-sided compression-only convergence metric (`residuum = min(s−aij_pj, 0)`, `err = rho0·mean(−residuum)` over the fluid; the 2D <7-neighbour deficiency guard zeroes the DF source, warm start, and residuum — the CD solve has no guard; Part 28's "3D-only" note is corrected: the guard is two-sided in the reference, `<7` in 2D, on both the setup and the metric side of the DF solve) — the CD solve now exits in 2 iterations on the under-compressed step 1 instead of running to the 64-iter cap. Contraction study (`probe_dfsphReferenceContraction.py`, omega sweep × 256 iters, what-if trajectories re-driven from the exact production inputs inside the same coupling context): the reference's **omega = 0.5 is OUTSIDE this composed operator's Jacobi window** — step-1 DF grows ~1.2×/iteration asymptotically (→4e14 at 256 iters), step-2 DF →2.7e18; 0.4 is marginal (step-1 DF still grows, p→187); **omega = 0.3 decays in all four (step, mode) states** (step-1 DF 2.5e-2→6.4e-5, step-2 CD →2.4e-6, step-2 DF 42→1.1e-3); 0.1/0.05 decay first then **regrow late** (the clamp-limited fixed point / a weak mode). Window ≈ [0.2, 0.35] → it is a **matrix problem (the window), not a budget problem** — a bigger budget at 0.5 only grows more. Landed: omega = 0.3, both budgets → the reference's 100 (local override in `dfsphReference_step`, the `akinciBoundaryVolumeScale` pattern). Validated: `hydrostaticColumn` (nx=32) — the ratchet is gone; every solve converges (2-100 iters), pressures bounded (CD ≤ ~11, DF ≤ ~10), |v|max 0.01→1.3-1.7 bounded post-slump slosh over hundreds-to-~1100 steps; `staticBlob` A/B (nx=128, 30 steps) — **Part 27's regression is fixed**: max |v| 70.9/inf → 1.15 (alpha) / 1.28 (dfsph factor), centroidDrift ~1e-9 (the residual |v|~1.1 blob slosh is pre-existing — it was 70.9 before the factor change). 20/20 tests pass. Residual: a **late-time free-surface degradation** at t ≈ 1.1 s (step ~1150): 2 of 3 1500-step runs fail there (one degrades surface rho_min 0.6→0.31→0.21→0.14 over ~100 steps then blows up p→1.8e6, NaN at step 1160; the other collapses into a uniform rho-0.139 soup with inf velocities that the runner's NaN-based divergence check does not catch), 1 of 3 completes 1500 steps bounded — same code, same seed, so the failure details are GPU-non-deterministic. That failure mode is step 3's (free-surface gauge) territory — parked in Part 26 under the old nonlinear solver, now testable. |
| 30 | 08-30 | **Step 3 re-run under the linear solve: the free-surface gauge is a measured negative.** Part 26's gauge implemented under the Part 29 linear Jacobi: the divergence solve holds `kappa^v` = 0 on the rows the case's own (dilated) `detectFreeSurface` flags (124–177 of 465 fluid rows, 27–38%, matching Part 26's ~27%) — the gauge rows join the reference-deficient rows in the source / warm-start / metric-residuum zeroing, and the pressure is **pinned to 0 at every iteration** so the carried field (and the next warm start) is 0 there and the final acceleration sees no surface-row pressure; DF solve only, module flag `FREE_SURFACE_GAUGE` (default off = the Part 29 baseline), `--gauge` in both probes. Also landed the one-line runner fix: the divergence check is `~isfinite` instead of `isnan` (`runner/runner.py`), so Part 29's inf-velocity soup now reports `diverged=True` — verified: one soup run stops at step 1279 with `non-finite velocities detected`. A/B (`hydrostaticColumn` nx=32, 1500 steps, 2 runs per arm, sequential and uncontended): the degradation's **onset is the same in all four runs** (~step 300–400) — the gauge does not delay or prevent the late-time failure; the gauge-on surface degrades **deeper and never recovers** (rho_min 0.15–0.21 persistent, runs end 0.23–0.24) while the gauge-off survivor recovers (0.25 at step 600 → 0.49 at 1500), and the gauge raises the bounded slosh ~30–40% (|v|max 1.8–2.0 vs 1.3–1.5). Blow-up count 1/2 (off) vs 0/2 (on) is inconclusive at n=2 against Part 29's 2/3 baseline. `staticBlob` unaffected (1.12 on / 1.16 off vs 1.28 baseline); 20/20 tests pass. The sign reproduces Part 26 (worse slump): with the surface rows out of the unknowns the sub-surface layer loses the support even a noisy `kappa^v` was providing. The gauge stays in the tree as an A/B toggle, default off; the recorded next lever is the reference's damped warm start against the full-`kappa` carry. |
| 31 | 08-30 | **The reference's damped warm start against the full-`kappa` carry: null on onset and end-state, mildly favourable on surface depth, ~5x the CD iterations — and it exposed that the baseline's IC seed self-destructs.** Verified against `TimeStepDFSPH.cpp` (08-30; constants re-verified): the reference does not carry the solved pressure as-is — it stores `p·h²` (CD) / `p·h` (DF), dt-invariant, and seeds the next solve with `0.5·min(stored, cap)/h^k` GATED on the row being compressed (CD: `densityAdv > 1`; DF: clamped `delta > 0`; both are "the one-sided source is negative" in this code's sign convention), zero otherwise; caps in stored units CD 2.5e-4, DF 0.5. Landed as the `DAMPED_WARM_START` toggle (default off = the Part 29/30 full carry): the same dt-scaled carry, the `source < 0` gate evaluated after the exemption zeroing (deficient/pinned rows seed 0, as the reference's zeroed `densityAdv` does), step 1 seeds from 0 (the reference has no IC pressure); `--warmStart` in both probes. **Baseline defect the A/B exposed:** the full-carry arm's step-1 CD solve is seeded with the IC hydrostatic profile (carried max 6.15 at t=0, measured), but its two forced iterations (minIters = 2; the one-sided metric reports err = 0 on the under-compressed column) run the TWO-SIDED update `p = max(p − 0.3(s − aij_pj)·invDiag, 0)` with s > 0 everywhere, driving the seed to exactly 0 in 2 iters (DE line `it=2 err=0.00 p[+0.00,+0.00]`) — the baseline is effectively a cold start at step 1, the CD pressure is rebuilt from 0 over ~10 steps (DE p max 0 → 2.9 by step 10), and that is the initial slump's true origin; the gated damped seed is structurally immune (it exists only where the update adds). A/B (`hydrostaticColumn` nx=32, 1500 steps, 2 runs per arm, sequential and uncontended): **all four runs complete 1500 steps bounded — no blow-up in this batch, either arm** (0/4 vs Part 30's 1/4 and Part 29's 2/3; the blow-up face is batch-stochastic, not an arm effect). The degradation's **onset is the same in all four runs** (first rho_min < 0.50 at step 226–429) — the damped warm start does not delay or prevent the late-time failure. Surface depth is mildly favourable (rho_min low 0.259–0.260 vs 0.227–0.243; one damped run holds the mid-run surface at 0.685 at step 301 vs 0.52–0.57 for both full runs) but not conclusive at n=2; end-state comparable (damped 0.480–0.490 consistent; full 0.342/0.626 split); late slosh unchanged (|v|max 1.18–1.79 both arms). Cost: the CD solve runs ~5x more iterations (median 22 vs 4; 18–39 vs 2–18) because the capped/gated seed starts far from the standing field; the 100 budget still covers it (no CD budget hits). `staticBlob` (nx=128, 30 steps, faithful factor): max |v| 0.348 (damped) vs 1.08 (full), KE 0.0015 vs 0.0305, centroidDrift ~5–7e-9 — the damped seed tames the blob's residual slosh (Part 29's 1.15–1.28). 20/20 tests pass. Verdict: **not a fix** — the late-time degradation now survives three levers (Part 26 nonlinear gauge, Part 30 linear gauge, Part 31 damped warm start); the toggle ships off and the track is at a decision point (a targeted onset-mechanism study, or a return to the ranked plan items). |

---

## 10. Overview — where things stand

### What is shipped and stable

The incompressible path is **VD+PS** (Cornelis et al.), faithfully implemented,
registered as `divergenceFree`. Three of this work's changes altered a shipped
default, all three measured before landing; every other new switch is opt-in
and default-inert, and the bug fixes are behaviour-preserving on every case
that existed before them.

- **`cflFactor = 0.4` against the particle diameter** — [BK]'s published
  condition in [BK]'s units (Part 12).
- **`ShiftPressureGauge.minShift` is the default** (Part 4) **and now reaches
  wall-bounded solves** (Part 14). It turns `kolmogorovIncompressible` at
  nx=128 from a NaN at step 574 into a stable 1000-step run with density in
  [0.980, 1.015], and it leaves `tgv`'s analytic decay rate alone to 0.4%. It
  still falls back to the historical clamp where there is a free surface.
- **`BoundaryOperatorTerms.staticBoundary` on both solvers** (Part 14), the
  formulation [BK] §3.2 states and SPlisHSPlasH implements. It is a strict
  no-op on every case with no `kind != 0` particles, which is every case here
  except the bounded one.
- Those last two are one fix at two points: together they take the bounded
  `randomFlowIncompressible` at nx=128 from a density band of 1.78e-1 to
  **4.48e-3**, 40x, and 5.4x better than they compose independently.
- **Several real bugs are fixed** (§7), the largest being the Eq. 17 resample,
  the boundary-row masking, and the `drhodt` pre-projection evaluation.
- Full suite passes (241 passed, 1 skipped), `gradcheck_incompressible.py`
  passes, and `run_sweep.py` is 30/30. Two known intermittent flakes, both
  pre-existing (§4).

**Case status at the shipped defaults:** `tgv`, `kolmogorovIncompressible` and
`shearWave` (all periodic) are healthy. `randomFlowIncompressible --bounded` is the case that
exercises everything; it is now the best-behaved it has ever been, and is still
where all the remaining error lives.
`rotatingSquarePatch --scheme divergenceFree` (free surface) is broken and is a
known method limitation, not an implementation bug — and is untouched by the
Part 14 defaults, which are both no-ops there. **`dambreak --scheme
divergenceFree` runs** (Part 19) — the scheme's only working free surface — but
at half `deltaSPH`'s run-out speed and with most of the flow's energy
dissipated on impact. It also needs its own, tighter CFL: `--cflFactor 0.2`
(Part 20), not the `0.4` every other incompressible case in this document
ships — the published constant diverges on this case.

### Part 12, the CFL condition — landed and verified

`kolmogorovIncompressibleTimestep` now applies `cflFactor` to the particle
**diameter** `dx` rather than the support radius `h`, which is the form [BK]
state their condition in, and `kolmogorovIncompressible` /
`randomFlowIncompressible` ship `cflFactor = 0.4` — literally their constant.
The old form was wrong in two independent ways: it was `n_h` times larger (4x
here, so the legacy `0.3` permitted 1.2 spacings of travel per step), and it
silently rescaled with the neighbour count, so the same number meant different
physics at different `n_h`. The viscous limit stays in `h`; it is a diffusion
condition over the smoothing length, not an advection condition over the
spacing.

**Verified two ways.** `probe_cflCondition.py --mode verify` reports the
dimensionless travel `dt |v_max| / dx` per step: **39 of 40 steps are
advection-limited and the travel is 0.4000 against the configured 0.4** (the
fortieth is step 0, which takes the case's seeded `targetDt`). And the change
is *exactly* a relabeling — re-running the §4 baseline row under the new
default reproduces the pre-change run bit for bit:

| | steps | `rho` range | `｜rho-1｜` 2nd half | t_final | DF resid |
|---|---|---|---|---|---|
| pre-Part-12, `cflFactor=0.1` (h-based) | 901 | [0.90228, 1.24742] | 1.7782e-01 | 4.6895 | 7.5192e-02 |
| post-Part-12, `cflFactor=0.4` (dx-based) | 901 | [0.90228, 1.24742] | 1.7782e-01 | 4.6895 | 7.5192e-02 |

So every measurement in this document recorded at "the published CFL" is a
measurement at today's `cflFactor=0.4`. `SimulationConfig.cflFactor`'s
description now records that what it multiplies is scheme-dependent
(compressible and weakly-compressible still use `h`), `tgv` is untouched (no
`timestep` hook, `dt` pinned at 1e-3), and the suite passes.

**The sweep is now run, and §5's prediction is confirmed: there is a lot of
room below 0.4.** Three earlier attempts failed for a reason that turned out
not to be GPU contention — `--mode sweep` was passing `--nSteps`, and
`runner.py:246` only honours `tLimit` when `nSteps` is *absent*
(`timeLimited = spec.nSteps is None and case.timestep is not None`), so every
"run to t=3" was silently a 20 000-step run. With that removed, the same
simulated time on `randomFlowIncompressible --bounded`, nx=128, t=3.0, stock
configuration otherwise:

| `cflFactor` | steps | `rho` range | mean`｜rho-1｜` | mean`(rho-1)` | frac `rho<rho0` | wall s |
|---|---|---|---|---|---|---|
| **0.4** (published) | 457 | [0.902, 1.202] | 5.17e-3 | 4.84e-3 | 12.2% | 166.7 |
| 0.2 | 903 | [0.969, 1.091] | **1.83e-3** | 1.58e-3 | 13.1% | 276.2 |
| 0.1 | 1830 | [0.960, 1.044] | 1.37e-3 | 1.21e-3 | 8.2% | 511.5 |

**Halving the published timestep cuts the density error 2.8x for 1.66x the
wall time. Halving it again buys 1.33x for another 1.85x.** So the knee is at
about 0.2 and the published 0.4 is not near it — on this case 0.4 is a
permissive setting, not a safe one, exactly as §5's calibration-scale argument
predicts. The tail says the same thing more strongly than the mean: `rho_max`
falls 1.202 → 1.091 → 1.044, so at 0.4 the worst particles are still strongly
timestep-limited.

Two things worth keeping from the cost column. The scaling is **sub-linear** —
2x the steps costs 1.66x the time, because a smaller `dt` hands the pressure
solvers an easier problem and they terminate sooner — so halving `cflFactor` is
cheaper than it looks. And this is the first measurement in this document where
the two solvers' iteration counts are *not* pegged at their caps, which is a
second, independent way of seeing §1.7.

The upper side was already measured, repeatedly: at 3x the constant the bounded
case diverges at t=5.5 with `|rho-1|` 5.4e-1 against 1.8e-1 at the constant
itself. Both sides now agree that 0.4 is the edge of a plateau rather than the
middle of one.

*(An independent second sweep, run concurrently, reproduced this table to
three significant figures and extended it to `cflFactor=0.05`: 1.12e-3 for
3692 steps, i.e. a further 1.22x. Both runs also found the `--nSteps` defect
above independently.)*

**Superseded by Part 14, and worth reading as a warning.** Every number above
still reproduces, but every *conclusion* drawn from them was about the
boundary treatment wearing a timestep's clothes. Re-run at the landed defaults
the same sweep gives 9.91e-4 at 0.4, and halving buys 1.17x rather than 2.83x:
the knee at 0.2 was not firm, it was the near-wall band, and 0.4 turns out to
sit in the middle of a plateau after all (§4, "The CFL question, re-run at the
new defaults"). The one claim that survives unchanged is the negative one — 3x
the constant is broken in every configuration.

### Part 13, the factorial — run, and it settles the default question

**Done; Part 14 then landed what it endorsed.** Full tables and analysis
in §4; the short version:

- **`minShift` (forced on bounded) + `staticBoundary` = 4.48e-3 against the
  shipped default's 1.78e-1 — 40x** — and **5.4x better than the two changes
  composing independently would predict** (2.42e-2). Neither shows this alone.
  It is the best configuration measured in this document by 5.3x.
- **`consistent` is inert** against `staticBoundary` once the gauge is fixed
  (0.007%, against 4.7% under the clamp), and **`akinciBoundaryVolume`
  diverges** at step 137. Part 11's "best configuration measured" was an
  artifact of the clamp gauge — its own numbers reproduce exactly, so this is
  an interaction, not a contradiction.
- **At the legacy CFL, 0 of 8 configurations survive**, the shipped default
  included. "Better at 0.4, worse at 1.2" was never a trade-off: 1.2 was
  already broken, and the shipped default merely fails *later* there, with
  `rho` in [0.139, 2.452].
- **Ten of the sixteen cells are configurations this document had already
  published, and all ten reproduced to every recorded digit.**

### Part 14, the landing — done

**The defaults are in.** `minShift` on wall-bounded solves, `staticBoundary`
on both solvers, and `randomFlowIncompressible --bounded` at nx=128 now ships
the 4.48e-3 configuration with no flags. Full tables in §4; the short version:

- **The bundle-level knob moved per-solver**, and the split it was moved for
  is rejected by measurement: `staticBoundary` on both (4.48e-3) beats the
  constant-density solve alone (6.49e-3) and the divergence-free solve alone
  (7.11e-2). §4 item 2 had endorsed the PS-only split on clamp-gauge evidence.
- **Three of the five rows are prior published configurations and reproduced
  to every digit**, which also proves the per-solver refactor inert at the old
  values.
- **The unexplained half-state divergence belongs to the clamp** — under
  `minShift` that configuration runs to completion. The mechanism behind it is
  still not explained, but it no longer gates anything.
- **The CFL question closes with the published constant intact.** Re-running
  Part 12's sweep at the new defaults, halving 0.4 buys 1.17x rather than
  2.83x: the knee was the boundary treatment. The new defaults at 0.4 beat the
  old ones at 0.1 for a quarter of the steps.
- Suite 241/1 skipped, gradcheck passes, `run_sweep.py` 30/30, config
  round-trips.

### Part 15, the stopping criterion — measured, and retired as a mystery

**No default changed.** §1.7 has been rewritten around what the measurement
actually found; the short version:

- **The periodic cases converge in 3 iterations** at the shipped defaults, both
  solvers. "Never terminates under every gauge" was a clamp-gauge observation
  that outlived the clamp. Non-termination is a near-wall phenomenon, like
  everything else here.
- **The floor was the wrong suspect.** All three criteria read within 6% of
  each other on the constant-density solve and all sit a factor of two above
  the tolerance.
- **That solve does not converge in any norm.** 64 sweeps remove 2.7% of the
  residual and grow the pressure field 61x, linearly in the iteration count.
  It is an integrator, and `maxIterations` is a shift gain wearing a
  convergence budget's name. No criterion can end it, which is why every one
  tried has failed identically.
- **The published criterion buys the published iteration count, and it costs
  accuracy**: adopting it takes the divergence-free solve from 31.9 iterations
  to 3.0 — [BK]'s 4.5, near enough — for 1.53x the density error. On the
  constant-density solve the same swap is bit-identical.
- **The shipped iteration budget is on the frontier** and should stay.
- Landed in code: one configurable `JacobiConvergenceCriterion` across all
  three relaxed-Jacobi loops (they spelled two different tests inline), and
  `rtol` as a relative disjunct. Both inert — Part 14's end rows reproduce bit
  for bit.

### Part 16, the shear-wave case — ported

**The first incompressible case here that grades this codebase against
something other than itself.** Full analysis in §4; the short version:

- **An exact solution with a constant pressure.** `u_x = u0 sin(k_w y)`,
  `u_y = 0` makes both nonlinear terms vanish identically, so
  `u_x = u0 sin(k_w y) exp(-nu k_w^2 t)` holds for all time with `p = const`.
  Every pressure the solver produces is therefore an artifact, and dissipation
  and volume error separate cleanly — which is why [C] reports two figures on
  this case and why `tgv`, whose exact solution carries a real pressure field,
  cannot make that separation.
- **`tgv`'s 0.55x now has independent corroboration.** The viscosity is applied
  at **0.49x** its prescribed value here, at two viscosities an order of
  magnitude apart, on a flow where it cannot be pressure error — which is the
  half-of-the-pairs Monaghan-switch explanation `tests/test_physics.py` states.
- **The volume error does not converge**: `max rho` is 1.0036 ± 0.0003 across
  nx = 32…256, while the amplitude error falls 7x. §1.1's structural density
  bias, measured in a running simulation with a known answer.
- **§1.2 is narrowed.** All three `ShiftApplication` modes dissipate
  identically here (0.1% apart at `nu = 0.01`); at `nu = 0` the *default* is
  the most dissipative. The velocity modes' dissipation penalty is not
  intrinsic to them — on a flow with no pressure gradient it vanishes. The
  position shift instead carries 1.8x the volume error and 2.4x the wall time.

### Part 17, `ShiftApplication` — the argument changed, the default did not

Full tables in §4. The short version:

- **§6's 3.2x is withdrawn.** At `tgv`'s own default nx=256 the three modes'
  decay ratios are 0.615 / 0.693 / 0.674 — within 12%. The velocity modes'
  ratio moves 2x between nx=128 and nx=256; `positionShift` holds 0.58-0.62
  everywhere.
- **§1.2's mechanism is withdrawn with it.** A penalty that halves under
  refinement cannot be the permanent residual of an unreachable setpoint —
  that residual is resolution-*independent*, which §1.1 argues and Part 16
  measured flat across an 8x range.
- **What survives is narrower and real**: the velocity modes are non-monotone
  at every resolution, i.e. a decaying unforced flow gains energy on some
  steps.
- **The bounded comparison, redone at the published CFL and the Part 14
  defaults**, has `positionAndVelocity` 1.5x better on the sustained band at
  identical wall time and 23% more simulated time — but with a 5x larger
  worst-ever density excursion. `inStepVelocity` is out on both cost and error.
- **The default stays**, on tail behaviour and energy monotonicity rather than
  on the number that used to justify it. Flipping it on a 1.5x mean improvement
  that carries a 5x worse tail is not the trade this project has been making.

### Part 18, the tail — `ShiftApplication` settles

Full tables in §4. At a pinned `dt` (which Part 17 failed to use, and which the
probe's docstring has always required for this comparison):

- **Zero wall penetration for all three modes**, at every one of 1201 steps.
  §6's legacy table — 4506 / 239 / 63 particles inside the wall — was the whole
  case for the velocity modes and was taken at the legacy CFL. At the published
  CFL with the Part 14 defaults there is nothing for them to prevent.
- **The velocity modes cost 2.1x the kinetic energy**: 41% retained at t=6
  against the default's 81%, on an inviscid case with no artificial viscosity,
  so all of it is numerical.
- **What they buy is 2x lower density error**, 5.9e-4 against 1.2e-3 near the
  wall — real, and a tenth of a percent either way.
- **So `positionShift` stays**, and the reason is now a measurement rather than
  §6's withdrawn 3.2x: half the flow's energy is not worth half the density
  error.
- **Part 17's "5x worse excursion" is retracted as my own protocol error** —
  three modes compared at three different timesteps, because the velocity modes
  damp the flow and the CFL then hands them a larger `dt`.

### Part 19, a dam break — the scheme's first working free surface

Full tables in §4. `dambreak --scheme divergenceFree`, nx=64, 3000 steps:

- **It works.** No divergence, fluid density in [0.907, 1.004], a recognisable
  collapse and run-out. The only other free-surface incompressible case,
  `rotatingSquarePatch`, is broken.
- **The run-out is half speed.** The front travels 1.50 by t=0.7 against
  `deltaSPH`'s 2.82 on identical geometry, resolution and `dt`.
- **88% of the kinetic energy is dissipated between t=0.5 and t=0.8** — the
  moment the falling column should be turning into horizontal run-out — while
  `deltaSPH` is still gaining. Same over-dissipation Part 18 measured on the
  bounded case, in the regime that exposes it.
- **Not a timestep artifact**: the run is 5x *finer* in time than [BK]'s CFL
  requires, because `dambreak` has no incompressible `timestep` hook. Adding
  one is worth ~5x wall time.
- **The free-surface density deficit vanishes** (0.518 at t=0.02, ~0.98
  thereafter) while the surface keeps only three quarters of the bulk neighbour
  count, i.e. while it is still geometrically a surface. Surface compaction is
  the natural reading and is what §4 records at the rotating patch's corners,
  but this measurement does not isolate it.

### Part 20, `dambreak`'s timestep hook — landed, and the "~5x cheaper" guess corrected

Part 19 guessed that giving `dambreak` an incompressible `timestep` hook would
be "cheap" and buy roughly 5x. `dambreakTimestep` (`cases/dambreak.py`) now
does this — active only under `--scheme divergenceFree`, reusing
`kolmogorovIncompressibleTimestep` exactly as `randomFlowIncompressible` does,
and a strict no-op for `deltaSPH` (`Case.timestep` is one hook shared by every
scheme a case might run under; it returns `config.dt` unchanged when
`ctx.scheme` is not `divergenceFree`). Measuring it found two things the guess
got wrong.

**The published CFL constant is not safe on this case.** `randomFlowIncompressible
--bounded` ships `cflFactor = 0.4` under the Part 14 defaults; `dambreak` does
not have the option. Bisected on `dambreak --nx 64 --scheme divergenceFree`,
full run to t=1.5 (nx=64, `--integrationScheme semiImplicitEuler`):

| `cflFactor` | outcome | steps | `rho` range |
|---|---|---|---|
| 0.4 (published) | **NaN at step 30** (t≈0.2) | — | — |
| 0.3 | **NaN at step 76** (t≈0.3) | — | — |
| 0.25 | survives | 1960 | [0.507, 1.231] |
| **0.2** | survives | **1769** | **[0.507, 1.105]** |
| fixed `dt=5e-4` (Part 19 baseline) | survives | 3000 | [0.907, 1.004] |

0.2 is the recommended value: 0.25 is technically stable but its `rho_max`
(1.231) is markedly worse, so there is no reason to run it. The mechanism is
presumably §1.6 again — the falling column's impact is a sharper, more
localised event than `randomFlowIncompressible`'s gentle bounded shear, and the
CFL condition's `vMax` is read from the *previous* step, so a fast-developing
local spike at the point of impact is exactly what a lagged advective
condition sees latest. **Unmeasured**: whether this is the same mechanism
behind Part 19's over-dissipation, since both are about what happens at the
moment of impact — worth keeping in mind while doing item 1.

**The step-count win is real but far smaller than guessed, and it is not free.**
At the recommended `cflFactor = 0.2`, the full run to t=1.5 takes 1769 steps
against the fixed-`dt` baseline's 3000 — **1.7x fewer**, not ~5x. (The 5x
figure in Part 19 was `dt_adv` at the *initial*, near-rest state compared
against the shipped fixed `dt`; it was never a measurement of the adaptive
run, which spends much of its time at higher `vMax` once the column falls,
where the CFL condition hands back a smaller `dt` than that initial estimate.)
And it is not a win on every axis: `rho_max` over the whole run is 1.105
against the baseline's 1.004, i.e. adaptive stepping here is trading some
density accuracy for fewer steps, not dominating the fixed `dt` on both. Wall
time was not compared cleanly — this session's GPU had unrelated processes
resident throughout (an `nvidia-smi` check found four other python/llama.cpp
processes holding device memory), which the rest of this document has already
found is enough to swamp a per-step difference (§4, Part 14 point 3), so no
wall-time number is reported here.

**What shipped:** `dambreakTimestep` in `cases/dambreak.py`, and the case
docstring now says `--cflFactor 0.2`, not the published 0.4.
`scripts/probe_dambreakIncompressible.py`'s `runScheme` passes it for
`divergenceFree` runs so every number that probe reports from here on is at
the stable value. No default in `Case.defaults` changed — `cflFactor` is one
config field shared by every scheme a case can run under, `deltaSPH` still
gets its own `0.3` unchanged, and a `divergenceFree` run requires passing
`--cflFactor 0.2` explicitly, the same way it already requires
`--integrationScheme semiImplicitEuler`.

### Part 21, the dissipation is not the free-surface clamp — the literature and a measurement agree

Item 1's leading candidate was §1.10's compaction story: the constant-density
solve drives free-surface particles back toward `rho0` against what the
geometry allows, and that was flagged as the plausible (but unestablished)
cause of Part 19's 88% kinetic-energy loss at impact. Two things now weigh
against it, one from the literature and one from a measurement — read
together with `literature/` before touching any code, per this session's
brief.

**The mechanism the codebase already runs at the free surface is [BK]'s own
published remedy, not a gap.** `bender2015`'s discussion section (extracted
from the PDF, p.9): "In SPH simulations the density near a free surface is
underestimated which causes unnatural particle clustering artifacts. In our
implementation this problem is solved by clamping negative pressures to
zero." That is exactly `ShiftPressureGauge.nonNegativeClamp`, which
`solveIncompressible` already falls back to on any solve with free-surface
particles (§1.5) — so the shipped configuration already implements the
paper's fix, it does not omit it. The paper's own "better solution" is
`schechter2012`'s ghost particles, a structural addition (a sampled layer in
the surrounding air), not a one-line pressure edit — so the commented-out
`pressureB[surfaceIndicators == 1] = 0.0` in `divergenceFree.py` that §1.10
and the old Known-open entry pointed at is neither this codebase's own
workaround nor the paper's remedy; it is a third, unpublished idea that
happens to be sitting in the file, in the wrong solver besides (the
divergence-free solve, which does not target density, rather than
`solveIncompressible`, which does).

**Removing the clamp does not slow the dissipation down — it kills the run in
four steps.** `scripts/probe_dambreakSurfaceGauge.py` forces
`forceShiftPressureGauge = True`, which keeps `ShiftPressureGauge.minShift`
active at the free surface instead of falling back to the clamp — the
free-surface half of that guard, explicitly flagged as untested in
`solver.py`'s own field description. Same case, same `cflFactor = 0.2`, nx=64:

| gauge at the free surface | outcome |
|---|---|
| shipped (clamp) | 1327 steps to t=1.0, no divergence, KE peaks 11.72 at t≈0.46 then falls to 1.18 by t=1.0 — Part 19's dissipation, reproduced |
| forced `minShift` (no clamp) | **NaN at step 4** (t≈0.03); `nLow` (surface population) roughly doubles in the first 3 steps and `rhoMin` collapses to 0.19 |

So the clamp is load-bearing, not a source of drag to relax: without it the
surface does not merely stay under-dense, it destabilises immediately — a
signed shifting potential can pull particles together at a boundary with
genuinely truncated support (§1.5's own reasoning, and exactly the caveat §4
item 12 already carried for MINRES-without-clamp: "a shifting potential that
may go negative can pull particles together... needs a free-surface test").
This is that test, run for the first time because `dambreak` is the first
working free surface, and it settles `forceShiftPressureGauge`'s free-surface
half as unsafe rather than merely unmeasured.

**Consequence: item 1's search moves off the free-surface treatment and onto
the impact itself.** The compaction is real (§1.10 measured it) and the clamp
that produces it is necessary for the run to survive at all, but forcing it
off does not reduce the dissipation — there is no dissipation to observe once
it is off, because the run is already dead. That rules out "relax the
free-surface pressure handling" as a fix and leaves the moment of impact — the
falling column striking the floor — as the remaining candidate, which is also
where Part 19's own timing (loss concentrated at t=0.5-0.8) and Part 20's CFL
finding (this case cannot survive the published constant, unlike every other
bounded case measured) both point.

Landed: `scripts/probe_dambreakSurfaceGauge.py`, the free-surface A/B above.
No config or default changed — `forceShiftPressureGauge` stays `False`.

### Part 22, the energy budget — the dissipation is the incompressibility cycle, not the walls or the viscosity

Item 1's instrument, run. `scripts/probe_dambreakEnergyBudget.py` closes the
kinetic-energy budget exactly, per step, and decomposes it two ways. The
**work form** splits `dKE` into each channel's first-order work plus a
quadratic remainder; it localizes work to an x-bin, so it answers *where*. The
**sequential form** takes the exact KE change of adding each force to the
running velocity, in the order the step applies them (gravity, no-penetration,
viscosity, the projection last, then the resample); those five values telescope
to `dKE` by construction and are unambiguous about *which step* removes KE and
which returns it. Same run as Part 19/21: `dambreak --scheme divergenceFree`,
nx=64, `cflFactor = 0.2`, 1327 steps to t=1.0.

**The closure is exact.** The captured forces reproduce the integrator's real
update to `max |u4 − v*| = 3.3e-6` (float32 round-off at |v|~10), the work-form
closure to `max |dKE − ΣW| = 1.15e-7`, and the decomposition gap (the captured
sum of the five accelerations against the integrator's) to exactly `0.0`. Every
velocity-changing term is accounted for; nothing is lost to the probe.

**The loss is the incompressibility cycle.** The dissipation window (KE peak
t=0.444, KE=12.13, to t=0.801) loses 10.04 KE, 82.8% of the peak — the same
event as Part 19's "88% between t=0.5 and t=0.8", measured from the peak instead
of a fixed start. The sequential channels:

| channel | KE change | share of the loss |
|---|---|---|
| divergence-free projection (`d_DF`) | −35.80 | 357% |
| Eq. 17 position-shift resample (`d_resample`) | +27.30 | −272% (returns) |
| **incompressibility cycle (the two summed)** | **−8.50** | **85%** |
| Monaghan viscosity (`d_visc`) | −6.37 | 64% |
| mDBC no-penetration shift (`d_nopen`) | −0.009 | 0.1% |
| gravity (`d_grav`) | +4.84 | source, opposes the loss |

The projection removes 35.8 of KE at impact and the resample gives 27.3 of it
back; the 8.5 the cycle keeps is the dominant single contribution to the loss.
Monaghan viscosity is the secondary channel (6.4). The no-penetration wall
shift — the other wall-side suspect — is negligible (0.009), so Part 21's
ruling out of the free-surface clamp is now joined by a ruling out of the
no-pen shift: **neither wall-side treatment is the dissipator.**

**The work form's `W_DF` alone overstates the projection's cost.** Its
first-order projection work is −44.3 in [0.4,0.5] — read naively, "the
projection dissipates 44". The sequential form shows +25.6 is returned by the
resample in that same window, so the cycle's *net* there is +0.1, essentially
energy-conserving. The cycle's true dissipation is front-loaded one window
later, at [0.5,0.6] (net −3.8), and decays after:

| window | `d_grav` | `d_nopen` | `d_visc` | `d_DF` | `d_resamp` | cycle net | `dKE` |
|---|---|---|---|---|---|---|---|
| [0.3,0.4] | +2.52 | −0.002 | −1.58 | −7.77 | +13.10 | +5.33 | +6.30 |
| [0.4,0.5] | +2.57 | −0.002 | −3.50 | −25.50 | +25.60 | +0.10 | −0.84 |
| [0.5,0.6] | +1.60 | −0.003 | −2.26 | −13.10 | +9.31 | −3.79 | −4.48 |
| [0.6,0.7] | +1.05 | −0.002 | −1.23 | −4.70 | +2.95 | −1.75 | −1.93 |
| [0.7,0.8] | +0.83 | −0.003 | −0.87 | −2.71 | +1.72 | −0.99 | −1.04 |

So the loss is not one channel throughout: in [0.4,0.5] the cycle is
net-neutral and the KE drop is viscosity (−3.5) outrunning gravity (+2.6); in
[0.5,0.6] the cycle itself turns net-dissipative and leads. Over the whole
window the cycle (−8.5) still outweighs viscosity (−6.4). The net is the
residual of two large, nearly-opposing terms (−36 and +27), so the answer is a
seesaw balance, not a single gross loss — and both terms are exact, given the
closure.

**It is where the impact is.** The per-bin table concentrates the loss in the
left bins, x ∈ [−2.0,−1.0] — the falling column's footprint — where the jet
strikes the floor, and the surviving KE moves right into the run-out. The
projection's work is most negative in exactly those bins (W_DF −9.0 in
x∈[−1.73,−1.51] at [0.4,0.5]).

**Why this closes the cross-scheme gap.** Both incompressibility channels are
specific to the `divergenceFree` scheme: the DF velocity projection and the
Eq. 17 position-shift resample are both part of its `finalize`, and neither
exists in the weakly-compressible `deltaSPH` control (Part 19). The Monaghan
viscosity (−6.4) is common to both schemes. So the 88%-vs-less gap between the
two schemes is carried by the incompressibility cycle — the thing the two
schemes differ on. What the budget does *not* by itself settle is why the
discrete cycle is net-dissipative at all: that −8.5 is the residual of two
~30-magnitude terms, and whether it is a discretization error that vanishes as
`nx` grows or a structural cost of the incompressible constraint is the natural
next instrument (an `nx` convergence of the cycle's net on this same case).

Landed: `scripts/probe_dambreakEnergyBudget.py` (the per-step / per-window /
per-bin budget above). No config or default changed.

### Part 23, the baseline cases — the scheme holds free space and the collision, but cannot hold a quiescent hydrostatic column

Item 2 from "What is left, in order": the three baseline correctness cases
for `divergenceFree`, required before the dam-break mechanism question (item
1) is worth compute. All three are new for the scheme (the existing
`hydrostatic` case is the compressible density-jump test, not this). They are
run to their full `tLimit = 1.0` at `nx = 64` by
`scripts/run_baseline_cases.py`.

**A setup finding first: the constant-density pre-relaxation cannot be run on
a free-surface state.** `tgv` and `shearWave` settle their ICs with
`relaxLattice` before the run. On a state with a free surface that
pre-relaxation is actively harmful: it drives the surface layer toward the
constant-density solve's `0.9` clamp floor, i.e. it compacts the surface.
Measured directly (the `relaxLattice` loop stepped on a `staticBlob` state):
step 0 shifts the surface particles `0.11` — 11% of the `L = 1.0` domain,
`|a|max = 1.1e5`; step 1 the shift explodes to `1.24e12` and the positions
stay at `1.2e12`. The free-surface density deficit (`~0.45-0.62`) is below
the `0.9` clamp floor, so the solve sees a permanent compaction source at the
surface and there is no equilibrium to relax to. The two free-space baselines
(`staticBlob`, and the `hydrostaticColumn` IC) therefore use jitter-only ICs
(`shuffleParticles`, `shiftIters = 0`, no `relaxLattice`), and `relaxLattice`
now raises `ValueError` if the post-relaxation `minDensity < 0.9 * rho0` so a
caller cannot silently repeat this (see the `caseUtils/incompressible.py`
docstring).

**Case 1 — `staticBlob`: PASS.** A square blob in periodic free space, no
gravity, no forcing, jittered lattice. Over `tLimit = 1.0` (101 steps): max
velocity is exactly `0` at every step (the DF projection of a zero-source,
zero-divergence state is the zero field), centroid drift is numerically zero
(`~1e-9` to `1e-10`), and the max displacement is `9.25e-4` — the jitter
scale, i.e. the particles stay where the jitter put them. The density band
(`min 0.467` at the blob's sharp corners, `max 1.029`) is the free-surface +
corner kernel deficit, steady over the whole run. Nothing moves. The scheme
holds a quiescent free-space state.

**Case 2 — `impact`: PASS (IC reproduces the WC outcome).** The
weakly-compressible `impact` case (two blobs, initial gap `0.5625`, closing
velocity `0.5`) ported to `divergenceFree` via `configureScheme` (physical
viscosity, no artificial viscosity) and `impactTimestep` (Bender & Koschier's
advective CFL, active only under `divergenceFree`). IC run
(`semiImplicitEuler`, 101 steps): the gap closes monotonically
`0.5625 -> 0.322 -> 0.0615 -> 0.0223 -> 0.0006` (at `t = 0.99`), the blobs
merge, max velocity peaks at `1.413` and relaxes to `0.86`, the density band
stays `0.957-1.017`, and the COM drift is `~1.8e-7`. The WC reference
(`deltaSPH`, fixed `dt`, 2001 steps): the gap closes to `0.0023` at
`t = 0.525` (contact), then the blobs oscillate and rebound, the gap
oscillating `0 -> 0.02`, final `0.014`, max velocity `1.611`. The IC run
reaches contact earlier and more completely (it merges; the WC run rebounds
off the compressive spike) — the expected difference, incompressible has no
compressive rebound — but both are stable and reproduce the collision-and-
merge physics. The scheme handles a dynamic free-surface impact.

**Case 3 — `hydrostaticColumn`: FAIL (the scheme diverges).** A wall-bounded
column at rest under gravity (`fillRatio = 0.5`, `g = 9.81`), IC the exact
at-rest state (fluid *and* wall pressure the analytic hydrostatic profile in
the DF gauge, so the run grades whether the scheme *maintains* the balance).
It does not: the column falls and flies apart. At `nx = 64` (adaptive `dt`)
the run does not reach NaN/Inf by `t = 1`, but it is grossly unphysical by
step ~25 — max velocity `14` (peaking at `73`), displacements up to `1.35` in
a `1.0` domain, the fitted pressure slope reaching `280x` the hydrostatic
value, and the pressure field departing from a straight line by `638x` the
column's own pressure drop. At `nx = 32` (fixed `dt = 1e-2`) it hits NaN/Inf
within ~5-8 steps. This is the limitation the case exists to expose.

**Why the scheme fails it.** Two coupled mechanisms, both specific to the
quiescent, wall-bounded, free-surface-under-gravity state (per-step
`scripts/probe_hydrostaticColumn.py`, `nx = 32`):

1. **The DF projection cannot balance a uniform body force.** Its source is
   `-div(v*)`, and the mDBC operator's divergence of the gravity-driven
   `v* = dt*g` is *exactly* `0` — the `freeSlip` wall projects the normal
   component out, so it is velocity-transparent to uniform normal motion, and
   a uniform field is divergence-free. The solver's logged source term is `0`
   and a direct `|div v*|` measurement confirms it. So the DF solve's correct
   solution for this source is a flat pressure field (step-0 output slope
   `+0.14`, i.e. flat) — it enforces `div v = 0` and, for a uniform `v*`, has
   no source to act on, so it cannot produce the hydrostatic pressure that
   would oppose gravity. The column's support must come from the
   constant-density solve's fall-and-push-back cycle instead.
2. **That cycle is unstable here.** (a) The DF Jacobi carries a
   boundary-layer mode the mean-residual convergence test cannot see: the
   per-iteration mean residual decays toward the tolerance while the
   wall/surface-adjacent pressure amplitude grows monotonically (step 0:
   residual `0.125 -> 0.010` across the full 32-iteration budget, while
   pressure `min -2.3 -> -7.8`, `max 3.9 -> 6.1`), and the `0.75x` warm start
   of the large hydrostatic gradient feeds it every step — so each step's
   early exit is clean on the mean and locally divergent. (b) The
   constant-density solve's pressure drifts under the `nonNegativeClamp`
   gauge — the free-surface guard downgrades `minShift` to the clamp, which
   does not pin the constant null mode — compounding the free-surface
   compaction. `|div v'| / |div v*|` is `4.3` at step 1 (amplification, not
   damping) and `|a_p|max` grows `35 -> 35 -> 49 -> 79 -> 2549` over the
   first five steps.

**A/B: it is not a config or IC choice.** A zero initial pressure
(`--zeroIC`) diverges more slowly but still diverges (vMax
`0.01 -> 0.12 -> 0.35 -> 0.59 -> 0.82 -> 1.05 -> 1.25`); `inStepVelocity`
diverges *faster* (vMax `225` at step 4, NaN at step 5); `forceGauge`
(keeping `minShift` on the free surface) diverges (vMax `0.04 -> 8.6`, NaN at
step 8). The failure is a fundamental scheme limitation on this state, not an
artifact of the defaults or the initialisation. Contrast: `staticBlob` (no
walls, no gravity, quiescent) and the dynamic `impact` and dam break all
hold — it is specifically the quiescent + gravity + wall + free-surface
combination that fails.

Landed: `scripts/run_baseline_cases.py` (the three cases to `tLimit` with the
stability summary), `scripts/probe_hydrostaticColumn.py` (the per-step
divergence-free / constant-density / boundary-shift diagnostics), the three
cases `staticBlob`, `impact` (port), and `hydrostaticColumn` (kept as the
failing baseline), and the `relaxLattice` free-surface guard in
`caseUtils/incompressible.py`. No scheme config or default changed.

### What is left, in order

1. **Explain the dam break's dissipation** (Part 19). **The channel is
   identified (Part 22)**: it is not the walls — Part 21 ruled out the
   free-surface clamp and Part 22's budget measures the no-pen shift at
   negligible — and it is not viscosity alone; it is the incompressibility
   cycle, the DF projection plus the Eq. 17 position-shift resample, net −8.5
   over the loss window (85% of it) against Monaghan viscosity's −6.4 (64%),
   localized at the column's impact footprint. Both channels exist only in the
   `divergenceFree` scheme, which is what carries the cross-scheme gap against
   the `deltaSPH` control. **What remains is the mechanism**: that −8.5 is the
   residual of two ~30-magnitude terms, and whether it is a discretization
   error that vanishes as `nx` grows or a structural cost of the incompressible
   constraint is open. An `nx` convergence of the cycle's net on this same case
   is the natural next instrument — but the case's default `nx = 128` diverges
   mid free-fall (§2), so the resolution-dependence is a stability problem as
   much as a dissipation question; with the baselines in (item 2, Part 23),
   this is now the immediate next step.
2. ~~**Baseline test cases for `divergenceFree`**~~ **Done (Part 23)** — the
   three cases landed (`staticBlob`, `impact`, `hydrostaticColumn`) and were
   run to `tLimit = 1.0` at `nx = 64`. `staticBlob` (free space) and `impact`
   (collision, reproducing the WC outcome) **pass**; `hydrostaticColumn`
   (quiescent column under gravity) **fails** — the scheme diverges (Part 23),
   which surfaces item 3.
3. **The quiescent hydrostatic column diverges** (Part 23, new). The
   `divergenceFree` scheme cannot hold a quiescent, wall-bounded,
   free-surface-under-gravity state: the DF projection's source is exactly `0`
   for the uniform gravity velocity, so it cannot balance a body force, and the
   constant-density support cycle that must carry the load is unstable there
   (a DF Jacobi boundary-layer mode invisible to the mean residual, plus
   `nonNegativeClamp` gauge drift and free-surface compaction). The
   `inStepVelocity` / `forceGauge` / zero-IC A/Bs confirm it is a scheme
   limitation, not a config or IC artifact. **Answered by Part 24 (below):** no
   configuration of the two-solve structure holds it; a DFSPH-proper scheme
   does hold the gradient but needs faithful boundary/gauge kernels to be
   stable. Part 24 lists that as a five-step track.

   **Part 24 (unlanded, mostly negative).** The root cause is confirmed: the
   VD+PS density-invariance correction is a momentum-neutral *position shift*
   (§1.2/§1.3), which cannot sustain a body force. Three things were built and
   measured:

   1. **A reference DFSPH scheme** (`schemes/dfsphReference.py`,
      `IncompressibleSPHScheme.dfsphReference`, composed from the existing warp
      operators; `DFSPHReferenceSystem` in `systems/incompressible.py`) that
      applies *both* corrections to the velocity as warm-started pressure
      impulses, SPlisHSPlasH-`TimeStepDFSPH` order, one-sided constant-density
      solve with a per-iteration re-summed `Drho/Dt`, Akinci `rho0` boundary
      volumes. At nx=32 it holds the **exact** hydrostatic gradient (`dp/dy`
      ratio ~ 1.0, `|v| < 1`, constant-density solve converging in 2
      iterations) for the first ~10-15 steps, where `divergenceFree` NaNs by
      step 6 — so velocity coupling + warmstart **is** the mechanism this
      state needs. But it is **not a stable general scheme**: the composed
      pressure primitives do not reproduce SPlisHSPlasH's Akinci boundary
      pressure loop or a free-surface gauge, so wall-adjacent `kappa` still
      grows without bound (the §1.7 boundary-layer mode) and the free-surface
      baselines `staticBlob` / `impact` are unstable under it. Landing it
      needs dedicated `@wp.kernel`s for the boundary force/factor plus a
      free-surface gauge — the same order of work `divergenceFree` itself
      took.

   2. **Warm-starting `solveIncompressible`** (new `warmStartPressure` param;
      `warmStartConstantDensity` flag, `inStepVelocity` only, carrier =
      `soundspeeds`). **NaNs by step 13** on `hydrostaticColumn`: a warm start
      on the solver *as it stands* — two-sided source, 0.9 `rhoStar` clamp,
      linear operator, `nonNegativeClamp` gauge — feeds the linear operator's
      wall-truncation `kappa` inflation. Only helps paired with the one-sided /
      re-summed inner loop of (1). Kept as an off-by-default hook.

   3. **Cold `inStepVelocity` on `divergenceFree`** looked promising at nx=32
      (`pressureSlopeRatio` 0.41 vs `positionShift`'s -300, `|v|` recovering to
      ~2) but that was the chaotic non-determinism: at **nx=64 it diverges**
      (vMax 3e7), and it **regresses `staticBlob`** (a quiescent free-space
      case `positionShift` passes with `|v| = 0` exactly) to a step-40 blow-up.
      Part 23's assessment stands.

   So: nothing in the existing two-solve structure holds this state, and
   retrofitting velocity coupling onto `solveIncompressible` breaks free-space
   cases. **The reference DFSPH scheme is the vehicle.** Every remaining
   failure traces to a composed-primitive limitation, so the next steps are to
   replace those primitives with faithful `@wp.kernel`s, in this order (each is
   gradcheck territory — see `.claude/skills/gradcheck`):

   1. **Akinci boundary pressure force kernel.** `a^p_i` currently comes from
      `computePressureAccelIISPH`'s symmetric SPH gradient, which truncates at
      a wall: the boundary contribution `sum_bk (p_i/rho_i^2) psi_bk gradW_ik`
      is under-resolved, so `A p` under-estimates the relief a standing `kappa`
      provides and the one-sided constant-density drive never releases —
      wall-adjacent `kappa` inflates without bound (the §1.7 boundary-layer
      mode, measured here as `kappa_max` 5 -> 40 -> 400 -> 2000 over ~15 steps
      on `hydrostaticColumn`). Write the boundary loop the way
      `SPlisHSPlasH/TimeStepDFSPH.cpp::computePressureAccel` does: an explicit
      sum over **`kind == 1` (Boundary)** neighbours only, with Akinci volumes
      `psi_bk = rho0 / sum_l W_bl` (`akinciBoundaryMass` already computes this)
      and the `p_i` self-term only (no boundary pressure value, no reaction —
      [BWJ23] Eq. 33 / `staticBoundary`).

      **`kind == 1`, NOT `kind != 0`.** `ParticleType` is Fluid=0, Boundary=1,
      **Ghost=2**, and this case genuinely has ghosts (`hydrostaticColumn` at
      nx=32: 465 fluid / 720 boundary / **720 ghost**). Ghost particles are
      mDBC *evaluation points* — an MLS density/pressure fit is run at each
      ghost position and written back to its owning `kind == 1` boundary
      particle via `ghostIndices` (`modules/mdbc/density2025.py`) — they are
      **not** interacting particles and must never enter a fluid or boundary
      neighbour sum. The existing operators are safe today only because every
      `warpSPHCore` gradient/divergence/density kernel skips `kind == 2`
      neighbours under any `operationMode != TrueAllToToAll`, and the
      incompressible path runs the default `OperationDirection.AllToAll`
      (`checkDirectionality_j(kind, 9) == (kind != 2)`;
      `warpSPHCore/util/directionality.py`). A hand-written boundary loop that
      filters on `referenceKind != 0` would pull those 720 ghosts into every
      near-wall fluid particle's pressure sum and destabilise the fluid-wall
      interface exactly as feared. Filter on `referenceKind == 1`, or route the
      sum through `OperationDirection.FluidToBoundary` / `checkDirectionality`.
      (The "`kind != 0`" phrasing in Parts 9/13/14 is always about which
      particles are *pressure unknowns* / take reaction forces — a property of
      `i`, where excluding ghosts too is correct — not about which neighbours
      `j` to sum over.)

      Validate against `hydrostaticColumn`: `dp/dy` ratio must stay ~1.0 past
      the ~15-step mark it currently breaks at.

      **Done, in part (Part 25).** The mechanism is confirmed and the wall
      runaway is removed, but *not* by a new kernel and *not* by the Akinci
      volume the bullet names. On this codebase's five-layer `BOUNDED_BAND` the
      Akinci `psi_bk = rho0 / sum_l W_kl` comes out **numerically equal to the
      nominal particle volume** (measured, `hydrostaticColumn` nx=32:
      `akinciBoundaryMass` returns `m_k` to 4 digits), so the paper's
      correction is inert here and "write the boundary loop with Akinci
      volumes" would change nothing. What is actually short is the *weight* the
      boundary term is carried at: at 1x it under-resolves the wall by ~2x, the
      constant-density solve exits clean on the mean residual (2 iterations)
      while a few wall particles ratchet `kappa` up every step
      (`kappa_max` 5 -> 130 -> runaway), and `|v|max` reaches ~100 by step 25.
      Carrying the boundary apparent volume at **2x** (new
      `IncompressibleSolverConfig.akinciBoundaryVolumeScale`, default 1.0 =
      strict no-op, set to 2.0 by `dfsphReference` only) bounds it: `kappa_max`
      saturates at 6.81 (against a true floor value ~4.55), `|v|max` holds
      < 1 through 25+ steps, `rho_min` (free surface) holds ~0.7. A sweep
      (`scripts/probe_dfsphReferenceColumn.py --sweep`) puts the usable range
      at 1.5–3x; below 1.5 the runaway returns, above 3 the surface compacts
      faster. **The `dp/dy` ~1.0 target is only partly met**: the wall is
      controlled but the divergence solve still does not converge (32
      iterations, err ~5e-3) so the fitted slope is noisy, and by ~step 40 the
      free surface begins to compact (`rho_min` 0.7 -> 0.24 by step 55). Those
      are steps 3 and 4 below, now the co-blockers. A faithful *single-layer*
      Akinci sampling would presumably not need the 2x, but that needs the
      oversized domain the module docstring notes.
   2. **DFSPH-factor kernel.** `computeAlpha` returns the IISPH `a_ii` with
      apparent-volume (`m_j/rho_j`) weights; DFSPH's `alpha_i = 1 / (|sum_j m_j
      gradW_ij|^2 + sum_j |m_j gradW_ij|^2)` uses bare masses and puts the
      boundary (`kind == 1`) in the *first* sum only ([BWJ23] Eq. 32). §2
      measured `diag(A)/alphas ~ 1.0001`, so the two agree in the bulk — but at
      a wall, with Akinci `psi_bk` in the mix, they diverge, and the factor is
      the Jacobi step size. Match `computeDFSPHFactor` exactly. Same
      ghost caveat as item 1: `computeAlpha` today excludes `kind == 2` from
      both sums (first via `AllToAll`, second via its explicit `kj == 0`
      guard); a replacement kernel must keep both exclusions.

      **Done (Part 27).** Landed as its own kernel
      (`modules/incompressible/wp_dfsph_factor.py`) mirroring
      `SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp::computeDFSPHFactor` exactly: the
      `|V_j ∇W_ij|²` sum runs over fluid neighbours only, the boundary
      (`kind == 1`) enters only the `|Σ V ∇W|²` vector term, ghosts (`kind == 2`)
      are excluded from both, and the apparent volumes (`m_j/rho_j`) carry the
      Akinci boundary volume from the `applyConsistentCoupling` context.
      `dfsphReference._factor` now returns its negation (the `<= 0` convention
      the solvers iterate against), replacing `computeAlpha`. Verified it is
      `computeAlpha`'s diagonal / `ρᵢ` exactly — the `1/ρ̄` the algebra predicts
      (bulk 1.049, wall 1.047, `scripts/probe_dfsphFactorCheck.py`) — so the
      step size changes by ~1/ρ everywhere, not just at the wall. The composed
      `a_p` was checked against a direct O(N²) torch reference of the standard
      SPH pressure acceleration (`computePressureAccelIISPH` =
      `-Σ_j m_j (κ_i/ρ_i² + κ_j/ρ_j²) ∇W_ij`, ghosts excluded) to ~5e-7:
      `a_p` was already faithful and needed no change, so step 2's "and/or the
      Akinci boundary force as its own kernel" resolves to *no new force kernel*
      — the standard acceleration already is SPlisHSPlasH's physical `a_p`.
   3. **Free-surface gauge / mask.** The gauge-free divergence solve
      accumulates `kappa^v` on sampling noise where support is truncated, so a
      jittered quiescent `staticBlob` blows up (the constant mode is not null
      at a free surface, §1.5 — the same reason `solveIncompressible` downgrades
      `minShift` to a clamp there). Either detect free-surface particles and
      hold their `kappa`/`kappa^v` at 0 (SPlisHSPlasH's few-neighbours guard in
      `divergenceSolveIteration`), or gate the solve on a surface mask.

      **Tried, deferred (Part 26).** `detectFreeSurface` (the scheme's own
      dilated detector) flags ~27% of the column's fluid rows; holding their
      `kappa^v` at 0 in the divergence solve **cleans the `dp/dy` fit** — it
      tracks ~1.0 through step 40 instead of the raw -2..+3 the unmasked solve
      leaks into the bulk pressure — but makes the column **slump faster**
      (`|v|max` 23 by step 59 against Part 25's ~2 at step 55). Masking the
      *constant-density* solve's `kappa` the same way is much worse: `rho_max`
      2.5 in 20 steps, the sub-surface layer over-compresses without the
      surface rows as unknowns. So the mask is not free, and its benefit
      (fit legibility) is not the metric that matters (stability).
      SPlisHSPlasH's `< 20`-neighbour guard is inapplicable here regardless —
      at `n_h = 4` even a flat-surface particle keeps 53+ neighbours.

      **Re-run under the Part 29 linear solve (Part 30) — negative.** The
      gauge is implemented in `_jacobiSolve` (`surfaceMask`; the module flag
      `FREE_SURFACE_GAUGE` toggles it, `--gauge` in both probes): the
      flagged rows' DF source / warm start / metric residuum are zeroed and
      their pressure is pinned to 0 at every iteration, so the carried
      `kappa^v` (and the next warm start) is 0 there. Sequential 1500-step
      A/B (2 runs per arm, nx=32): the late-time surface degradation's
      **onset is the same in all four runs** (~step 300-400) — the gauge does
      not delay or prevent it; the gauge-on surface degrades **deeper and
      never recovers** (rho_min 0.15-0.21 persistent, runs end 0.23-0.24)
      while the gauge-off survivor recovers (0.25 at step 600 → 0.49 at
      1500); the gauge also raises the bounded slosh ~30-40% (|v|max
      1.8-2.0 vs 1.3-1.5). Blow-up count 1/2 (off) vs 0/2 (on) is
      inconclusive at n=2 against Part 29's 2/3 baseline. The sign
      reproduces Part 26 under the linear solve: the mask's cost (surface
      rows out of the unknowns → the sub-surface layer loses the support
      even a noisy `kappa^v` was providing) exceeds its benefit. Left as an
      A/B toggle, default off.
   4. **Contractive divergence solve.** Even masked, the per-iteration
      re-summed `Drho/Dt` form has no contraction guarantee. Use the linear
      `A p = dt * shift(accel(p))` operator with an in-window `omega`
      (`solveDivergenceFree`'s form, `omega < 0.355`, §divergenceFree.py
      docstring) for the divergence half, or clamp `|kappa^v|`.

      **Done (Part 29).** Both solves use the reference's fixed-source
      linear form (source from `vEnter` once, `aij_pj = Drho/Dt(a_p)/rho0 *
      opDt` recomputed per iteration, `max(p,0)`), with Part 28's corrected
      sign conventions, the reference's one-sided compression-only
      convergence metric + 2D <7-neighbour deficiency guard, and a
      measured-stable relaxation: the omega sweep showed the reference's
      0.5 is outside this composed operator's Jacobi window (~[0.2, 0.35])
      and omega = 0.3 decays in every (step, mode) state, so it is set to
      0.3 and both budgets carry the reference's 100. The ratchet is gone:
      `hydrostaticColumn` (nx=32) runs hundreds-to-~1100 steps with every
      solve converging (2-100 iters), pressures bounded (CD ≤ ~11, DF ≤
      ~10), and a bounded post-slump slosh (|v|max ~1.3-1.7); the
      `staticBlob` A/B (Part 27's regression) recovered (70.9/inf →
      1.15/1.28 max |v|). Residual, step 3's territory: a late-time
      free-surface degradation at step ~1150 (surface rho_min → 0.14, then
      blowup or a uniform rho-0.139 soup with inf velocities; 2 of 3
      1500-step runs fail there).

      **Earlier attempt, deferred (Part 26).** The linear SPD operator
      `A(p) = -dt·_drhodt(a_p(p))` with the exact residual-minimizing step
      (`solveDivergenceFree`'s `optimal` device) converges the divergence solve
      in 14–25 iterations for the first ~13 steps on `hydrostaticColumn` —
      against the re-summed form's permanent 32 — but **regresses `staticBlob`
      hard** (`|v|max` 19 by step 2): `solveDivergenceFree`'s version pairs the
      optimal step with per-iteration gauge re-centering to kill the constant
      null mode, which §1.5 forbids at a free surface, and without *some*
      null-mode handling `omega_k` blows up along the near-null directions a
      free surface creates. The re-summed form's fixed `omega` is ugly but
      does not have that failure mode. A `|kappa^v|` clamp (the bullet's
      fallback) was not tried. Neither path addresses the actual driver of the
      column slump, which is the constant-density solve (see below), so this is
      parked until step 5 clarifies whether the DF solve needs to be good at
      all on this state.
   5. **Then the DFSPH validation ladder** — `hydrostaticColumn` clean to
      `tLimit`, then a `dambreak` A/B against `divergenceFree` and `deltaSPH`
      (this also gives item 1 a second data point on the incompressibility
      cycle: a genuine DFSPH has no Eq. 17 resample, so the cycle's dissipation
      channel is structurally different).

   The `dfsphReference` scaffold (step, system, warm-start carriers on
   `pressures`/`soundspeeds`, Akinci coupling, no-op integrator wiring) is in
   place; items 1-4 are edits to `_jacobiSolve` / `_factor` / `_pressureAccel`
   and one or two new kernels. This is the same order of effort
   `divergenceFree` itself represents (~20 sessions), so it is a track of its
   own, not a one-session fix.

   **Part 25 — step 1's wall runaway removed, and it was a weight not a
   kernel.** The `hydrostaticColumn` failure at nx=32 has two coupled
   mechanisms (Part 23): the DF projection cannot balance a body force, and
   the constant-density support cycle is unstable at the wall. Part 25
   addresses the second. `dfsphReference` holds the exact gradient for ~7
   steps and then wall-adjacent `kappa` runs away — the constant-density
   solve exits at 2 iterations on a converged *mean* residual while a handful
   of wall particles ratchet their pressure up every step (`kappa_max`
   5 -> 130 -> clamp, `|v|max` ~100 by step 25). The cause is the boundary
   term in `A p`: it is the feedback that tells the one-sided drive a
   standing `kappa` is already relieving the compression, and at this
   codebase's five-layer `BOUNDED_BAND` it is carried at roughly half the
   weight it needs.

   The bullet above expected the fix to be Akinci volumes `psi_bk =
   rho0 / sum_l W_kl` in an explicit `kind == 1` loop. Measured, that is a
   near-no-op: on the band `sum_l W_kl` is large enough that
   `akinciBoundaryMass` returns the nominal particle volume `m_k` to four
   digits, so the paper's single-layer correction has nothing to correct
   here. What works is carrying the boundary apparent volume the pressure
   solve sees at **2x** — `IncompressibleSolverConfig.
   akinciBoundaryVolumeScale`, default `1.0` (a strict no-op; every A/B on
   file keeps its meaning and `divergenceFree` at shipped defaults never
   reads it — its bounded case runs `mdbcDensity`, not `consistent`), set to
   `2.0` by `dfsphReference` only. With it: `kappa_max` saturates at 6.81
   (true floor value ~4.55), `|v|max` holds < 1 through 25+ steps, the free
   surface holds `rho_min` ~0.7. A sweep
   (`scripts/probe_dfsphReferenceColumn.py --sweep`) puts the usable band at
   1.5–3x; below 1.5 the runaway returns, above 3 the surface compacts
   faster.

   **What this does not fix.** The DF solve still does not converge (32
   iterations, err ~5e-3 — gauge-free, accumulating on free-surface noise:
   steps 3/4), so the fitted `dp/dy` slope stays noisy even though the wall
   is controlled, and by ~step 40 the free surface begins to compact
   (`rho_min` 0.7 -> 0.24 by step 55) and the column slowly slumps
   (`dispMax` -> 0.46). So step 1's own validation target ("`dp/dy` ~1.0 past
   15 steps") is only half met: the wall mechanism is closed, the
   free-surface mechanism (step 3) and the non-contractive divergence solve
   (step 4) are now what remains. Landed: the config field + round-trip, the
   `akinciBoundaryVolumeScale` plumbing through `applyConsistentCoupling` /
   `akinciBoundaryMass`, `scripts/probe_dfsphReferenceColumn.py`. No
   `divergenceFree` default changed; 25 incompressible tests pass.

   **Part 26 — steps 3 and 4 explored, nothing landed, and the blocker moved.**
   With the wall runaway gone (Part 25), the residual `hydrostaticColumn`
   failure is a slow slump: `|v|max` creeps ~0.05/step from step 0 and the
   column visibly falls from ~step 15. Steps 3 (free-surface `kappa^v` mask)
   and 4 (contractive divergence solve) were both built and measured; details
   in the step bullets above. Neither lands:

   - The **surface mask** trades a cleaner `dp/dy` fit (it tracks ~1.0 instead
     of the raw -2..+3 the unmasked `kappa^v` noise leaks into the bulk) for a
     *faster* slump — a bad trade, since legibility of the fit is not what is
     failing. Applied to the constant-density solve instead it is much worse
     (`rho_max` 2.5 in 20 steps).
   - The **linear optimal-step divergence solve** converges the DF solve
     (14–25 iters vs a permanent 32) for ~13 steps but regresses `staticBlob`
     hard: the optimal step needs null-mode handling, and the only one
     available (`solveDivergenceFree`'s per-iteration mean-centre) is the
     spurious-force move §1.5 forbids at a free surface.

   **The slump's driver is the constant-density solve, not the DF solve.**
   `|v|max` climbs from step 0 while the DF solve is masked out entirely, and
   the CD solve — which *does* converge (`err` ~1e-4, `kappa_max` bounded at
   6.85) and holds `dp/dy` ~1 on the *mean* — carries `|a_p|max` ~17–45, far
   above `g ≈ 9.81`. It is globally balanced and locally lumpy: a few
   particles per step get a push several times gravity, and that velocity
   noise accumulates into the slump. This is Part 23's mechanism 1 (the DF
   projection's source is identically 0 for a uniform body force, so the CD
   fall-and-pushback cycle must carry the entire hydrostatic load) showing up
   as a discretisation-quality problem in the CD solve rather than an
   instability. That is step 5's territory — a faithful DFSPH factor (step 2)
   and/or the Akinci boundary *force* written as its own kernel so the CD
   `a_p` is not the symmetric-gradient approximation it is now. Steps 3–4 are
   parked behind it: there is no point converging a divergence solve on a
   state whose real error is upstream of it.

   Landed by Part 26: nothing in `src/`. The step bullets and this narrative
   record the two negative results so the next session does not re-run them.

   **Part 27 — step 2 landed: the faithful DFSPH factor, and the blocker moves
   to the solve.** Step 2 (the harden track's "faithful DFSPH factor and/or the
   Akinci boundary force as its own kernel") is done. The factor is now its own
   kernel, `modules/incompressible/wp_dfsph_factor.py::computeDFSPHFactor`,
   mirroring `SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp::computeDFSPHFactor` line for
   line: the `|V_j ∇W_ij|²` sum runs over **fluid** neighbours only, the
   boundary (`kind == 1`) enters only the `|Σ_fluid V_j ∇W_ij + Σ_boundary V_k
   ∇W_ik|²` vector term (a static boundary takes no reaction, [BWJ23] Eq. 32),
   and ghosts (`kind == 2`) are excluded from both via the `AllToAll`
   directionality plus the explicit `kj == 0` guard. The apparent volumes
   `V_j = m_j/ρ_j` carry the Akinci boundary volume, so the kernel reads the
   `applyConsistentCoupling`-modified `state.masses`. `dfsphReference._factor`
   now returns its negation (the `<= 0` convention both solvers iterate
   against), replacing the IISPH `computeAlpha` diagonal.

   **Two checks, both in `scripts/probe_dfsphFactorCheck.py`.** (1) The factor
   is `computeAlpha`'s diagonal **/ `ρᵢ` exactly** — the algebra says
   `alpha_IISPH = (V_i/m_i)|Σ V_j ∇W|² + …` collapses to `diag_DFSPH/ρᵢ` when the
   neighbour set is all fluid, so the two must agree in the bulk and diverge by
   `1/ρᵢ` wherever `ρᵢ ≠ 1`. Measured: bulk ratio 1.049, wall 1.047 — a uniform
   `1/ρ̄` (bulk `ρ̄ ≈ 0.95`), confirming the DFSPH formula rather than a bug, and
   showing the step size changes by ~1/ρ **everywhere**, not just at the wall.
   (2) The composed `a_p` is the standard SPH pressure acceleration:
   `computePressureAccelIISPH` (`-warpOperation(Symmetric gradient)/ρ`) is
   checked against a direct O(N²) torch double-loop
   (`-Σ_j m_j (κ_i/ρ_i² + κ_j/ρ_j²) ∇W_ij`, Wendland2 2D, `Scatter` support,
   ghosts excluded) and agrees to ~5e-7 (float32). So `a_p` was **already**
   faithful to SPlisHSPlasH's physical `a_p` — its `p/ρ²` (`p_rho2`) is an
   internally-scaled variable, the physical acceleration is the same — and
   step 2's "Akinci boundary force as its own kernel" resolves to **no new
   force kernel**: the standard acceleration already is the faithful `a_p`.

   **The `hydrostaticColumn` slump survives** (the re-check step 2 asked for).
   nx=32 × 30 steps, `scripts/probe_dfsphReferenceColumn.py`: `|v|max` climbs
   0.04 → a step-26 peak of 1.33 → 1.17 at step 30; `rho_min` 0.62 → 0.70
   (recovering); `rho_max` ~1.0; the CD solve stays at 2 iterations and the DF
   solve still hits the 32-iter cap (err ~2e-3 → 3e-2). Against the Part 25
   baseline (old `computeAlpha` factor: `|v|max` ~1.25, `rho_min` ~0.68) this is
   a **modest, consistent gain** — the faithful factor is a small improvement,
   not a fix. The slump is Part 23's mechanism 1 (the CD solve carrying the
   entire hydrostatic load as a fall-and-pushback cycle), now a
   discretisation-quality limit of the *solve*, not a formula-fidelity gap: the
   two composed pressure primitives (factor and `a_p`) are both faithful, and
   the lumpiness remains.

   **The regression that redirects the track.** `staticBlob` under
   `dfsphReference` (nx=128, 20 steps, `--factor` A/B in
   `scripts/probe_dfsphReferenceStaticBlob.py`): the **baseline** `computeAlpha`
   factor already regresses it hard (Part 26's known failure) but stays finite
   (`|v|max` 70.9, KE 0.243, `centroidDrift` ~2e-9); the **faithful** factor
   diverges it outright (`|v|max` → inf, `rho_max` → 569, KE → inf). The
   faithful factor is *correct* for SPlisHSPlasH — whose solve is a **linear**
   Jacobi (fixed source `1 − ρ_adv`, 0.5 relaxation, the factor scaled by
   1/h²) — but `dfsphReference._jacobiSolve` is a **nonlinear re-summed fixed
   point** (`v* = vEnter + dt·a_p`, `Drho/Dt*` re-summed each iteration). That
   structure is far more step-size-sensitive, and the faithful factor's ~1/ρ
   larger step pushes the already-marginal blob over the edge. So step 2 is
   done and correct, and it **moves the blocker to step 4**: until the
   divergence/constant-density solve is the linear form SPlisHSPlasH uses (or
   has a step-size that does not depend on this sensitivity), the faithful
   factor cannot be adopted on the nonlinear solve. Two negative results
   recorded (the `a_p` is already faithful; the faithful factor regresses
   `staticBlob`) so they are not re-run.

   Landed by Part 27: `src/warpSPH/modules/incompressible/wp_dfsph_factor.py`
   (new kernel) and the `_factor` rewiring in `src/warpSPH/schemes/dfsphReference.py`;
   two probes, `scripts/probe_dfsphFactorCheck.py` and
   `scripts/probe_dfsphReferenceStaticBlob.py`. The 20 incompressible tests
   (`tests/test_incompressibleKrylov.py`) still pass — `divergenceFree` is
   untouched, the new kernel is imported only by `dfsphReference`.

   **Part 28 — step 4, the linear solve, and the sign conventions hiding in
   it.** Step 4 (the harden track's "contractive divergence solve") is now
   implemented as SPlisHSPlasH's **linear** Jacobi in
   `dfsphReference._jacobiSolve`, replacing the nonlinear re-summed fixed
   point that Part 27 found step-size-sensitive. Both solves (constant-density
   and divergence) now use the reference's fixed-source form: the source `s`
   is computed **once** from `vEnter` (the post-non-pressure velocity, not
   re-summed each iteration), `aij_pj = Drho/Dt(a_p)/rho0 * opDt` (the density
   change the *current* pressure would cause, recomputed per iteration from
   the acceleration field, `opDt = dt²` CD / `dt` DF matching their
   `aij_pj *= h²` / `*= h`), the relaxation is their fixed 0.5, and the
   one-sidedness is the `max(p, 0)` clamp (their source is two-sided). The
   diagonal `invDiag = 1/(opDt * sum_grad_p_k)` is their `factor =
   (1/sum_grad_p_k) * invH^k` exactly (`sum_grad_p_k` is the Part 27 faithful
   factor, a sum of squared kernel-gradient norms; their source stores
   `factor = 1/sum_grad_p_k` then `*= invH²`/`*= invH`).

   **The first draft diverged, and the root cause was sign, not
   structure.** The initial implementation (source `1 - rho/rho0 +
   dt*Drho/Dt/rho0` and `aij_pj = -Drho/Dt(a_p)/rho0 * opDt`, both read
   straight off `_drhodt = -rho0 * div`) ran the DF pressure up by ~2x
   *every* iteration (2e-3 → 8e9 in 32 iters, NaN by step 2). Measuring, not
   deriving, against `TimeStepDFSPH.cpp` pinned three sign facts:
   (1) SPlisHSPlasH's `delta` operator — the difference-form `V_i
   Σ (v_i − v_j)·∇W` used by `computeDensityAdv` / `computeDensityChange` /
   `compute_aij_pj` — is the **negative** of the continuum divergence
   (the difference form `Σ (q_i − q_j)∇W` negates the continuum gradient);
   (2) this codebase's scatter Divergence (inside `_drhodt`) **is** the
   continuum one — probed by running a `div = +1` field through
   `computeMomentumIncompressible`: `_drhodt ≈ −1.0` in the bulk (the
   positive max is the known free-surface truncation bias); (3) their
   `factor > 0` and **both** solves iterate `p −= 0.5(s − aij_pj)·factor`
   (their comment: `alpha_i = −1/(a_ii ρ_i²)`, `a_ii = −ρ_i²/Σ|∇W|² < 0`,
   so the matrix is negative in their convention: positive p →
   `aij_pj < 0`). With (1)+(2), the first draft's source *and* `aij_pj`
   were each sign-flipped relative to the reference, so the residual was
   sign-flipped and the `p −=` step was the diverging direction (spectral
   radius > 1). An intermediate attempt (flip only the step sign to `p +=`)
   fixed the DF solve and broke the CD solve (p → 1e25) — the diagnostic
   that the source and `aij_pj` must flip *together*: the corrected, unified
   form is the reference convention for source (`density: 1 − rho/rho0 −
   dt*Drho/Dt/rho0`, `divergence: −Drho/Dt/rho0`) and `aij_pj`
   (`+Drho/Dt(a_p)/rho0 * opDt`), with the same `p −= 0.5(s − aij_pj)·invDiag`
   for both modes.

   **With the signs right, the physics is right.** Measured on
   `hydrostaticColumn` nx=32 (`scripts/probe_dfsphReferenceColumn.py`):
   step-1 CD — the initial column is under-compressed (source
   [+1.8e-2, +3.7e-1]) and the pressure stays exactly 0 (no tensile
   pressure, correct); step-1 DF — the post-gravity source is compressive
   (source min −0.275, `O(dt·g/h)`) and the pressure grows positive (correct);
   step-2 CD — the first over-compressed particles appear (`rho_max` 1.31)
   and get positive pressure (correct). The pressure now lands where the
   physics says it should, in all three regimes.

   **But the iteration does not contract inside its budget.** Step-1 CD runs
   to the 64-iter cap (err 7e-2 — see the metric note below); step-1 DF runs
   to the 32-iter cap (p max 124, err 6.9); by step 2 the CD pressure
   oscillates without settling (p max 12.9 → 7.5 → 11.9 → 10.1 → 12.6 →
   11.1 → 13.0 → 11.8, cap at 25.9) and the DF solve is actively diverging
   (p max 186 → 150 → 266 → 191 → 364 → 257 → 489 → 353, cap at 1.8e4). The
   ratchet compounds: `|v|max` 0.01 → 1.76 → 1732 → 8.9e4, `rho_max` 1.31 →
   1.66 → 2.01, NaN at step 6. The CD "non-convergence" at step 1 is partly
   a **metric artifact** — see next — but the step-2+ oscillation and the DF
   divergence are real.

   **The convergence-metric finding (from the reference source).**
   SPlisHSPlasH's stopping metric is **one-sided**: the CD solve accumulates
   `density_error −= rho0 * min(s_i − aij_pj, 0)` (only the *compression*
   part of the residual counts, averaged over all particles) and the DF
   solve uses the same `min(s − aij_pj, 0)` with a **particle-deficiency
   guard** (`< 20` neighbours in 3D / `< 7` in 2D → that particle's
   residuum is excluded; it does not skip the pressure update). The local
   metric here is the two-sided `mean|resid|`, which on an under-compressed
   state can *never* reach `tol` (the clamp holds p=0 while `resid = source
   > 0`), so both solves run to their caps regardless of the physics.
   Adopting the one-sided metric + guard is the first concrete next step:
   faithful, cheap, and it restores the early exit on clamped states — but
   it does not by itself fix the step-2+ oscillation (the over-compressed
   subset's residual genuinely does not settle).

   **Known remaining differences from the reference (the contraction
   study's levers).** (a) *Warm start*: this scheme carries the full
   previous-step `κ`; SPlisHSPlasH's `USE_WARMSTART` branch uses
   `0.5 * min(κ, 2.5e-4) * invH²` gated on `densityAdv > 1` (else 0), and
   the no-warm-start branch is the one-sided guess `max(0, −s * factor)` —
   a damped/capped warm start is a direct candidate for the ratchet.
   (b) *Budget*: CD 64 iters / tol 5e-4, DF 32 iters / tol 2.5e-3; the
   composed operator on this state may simply need more iterations, or a
   smaller `omega` (0.5 is their value, not a law). (c) *Operator
   conditioning*: the composed `aij_pj` (scatter Divergence of
   `computePressureAccelIISPH`) may lose diagonal dominance at the free
   surface and on the five-layer `BOUNDED_BAND` in a way the reference's
   dedicated kernels do not.

   **Next, concretely.** (1) Adopt the one-sided `min(s − aij_pj, 0)`
   metric + the neighbour-deficiency guard (faithful, and it changes the
   early-exit behaviour on clamped states). (2) A contraction study: sweep
   `omega` {0.5, 0.3, 0.1} and the iteration budget (128/256) and classify,
   per mode, whether the residual decays (slow — a budget problem) or grows
   (diverging — a matrix problem). (3) If diverging: inspect the composed
   operator's diagonal dominance at the free surface / five-layer band and
   try the reference's damped warm start. (4) Only then re-run the
   `staticBlob` A/B — the whole point of step 4 is to fix the Part 27
   regression, and that is untestable until the column is stable.

   Landed by Part 28: the linear `_jacobiSolve` in
   `src/warpSPH/schemes/dfsphReference.py` (sign notes in its docstring),
   and `scripts/probe_dfsphReferenceColumn.py` now prints `n_fluid` and
   handles an empty wall band. The 20 incompressible tests
   (`tests/test_incompressibleKrylov.py`) still pass — `divergenceFree` is
   untouched, the change is inside `dfsphReference` only. The scheme is a
   troubleshooting artifact (not a landed solver) and is **diverging on
   `hydrostaticColumn`** at the end of this part — do not treat it as
   usable until the contraction study above lands.

   **Part 29 — step 4 closed: the one-sided metric, the contraction window,
   and omega = 0.3.**

   **What landed.** Three edits in `dfsphReference`, all measured, all
   against the reference source:
   1. *The one-sided convergence metric* (Part 28's recorded next step).
      `residuum = min(resid, 0)` per particle, `err = rho0·mean(−residuum)`
      over the fluid — SPlisHSPlasH's `density_error −= rho0·min(s−aij_pj,0)`
      averaged over active particles. The two-sided `mean|resid|` is gone; on
      an under-compressed state the new metric reads 0 and the solve exits
      early (step-1 CD: 64-iter cap → 2 iters, err = 0 — the clamp holds
      p = 0 and there is nothing to correct, which is the physics).
   2. *The 2D <7-neighbour deficiency guard*, faithful on all three of its
      reference sites: the DF **setup** zeroes the source of deficient
      particles (`densityAdv = 0`, so both warm-start and guess branches
      start them from p = 0), the DF **metric** zeroes their residuum, and
      the **pressure update still runs** for them (their `aij_pj` relaxes
      toward 0, not their source). The CD solve has **no** guard in the
      reference — none here. Correction to Part 28's note: the guard is not
      3D-only; the DF kernel's metric guard is `<7` in 2D / `<20` in 3D, and
      the setup-side guard is the same two-sided split. The count is the
      fluid+boundary neighbour count (`countNeighbors` with `AllToAll`
      excludes ghost references, matching their per-point-set sum — ours
      re-evaluates `w > 0` while theirs reads the Verlet-list length, so
      ours is ≤ theirs); it is evaluated once per solve (the Verlet list and
      positions are fixed during the solve).
   3. *The relaxation and the budget.* omega = 0.5 → **0.3**; both
      maxIterations → **100** (the reference's value for both solves; the
      config ships 64/32 — local override in `dfsphReference_step`, the
      `akinciBoundaryVolumeScale` pattern). The minIterations floors already
      matched (2 CD / 1 DF).

   **Why omega 0.3 (the contraction study).**
   `scripts/probe_dfsphReferenceContraction.py` re-drives the production
   loop — same state, source, warm start, factor, same
   `applyConsistentCoupling` context — with a swept omega and a fixed
   256-iteration budget and no early exit, at each production solve call,
   so every trajectory is a what-if on identical inputs (the production
   solve still runs omega 0.5, so the step-2 states are the production
   ratchet's). Result (one-sided metric, `rho0·mean(max(0, aij_pj−s))`):

   | omega | step-1 DF | step-2 CD | step-2 DF |
   |---|---|---|---|
   | 0.5 (prod) | **GROW** 2.5e-2 → 4.4e+14 | GROW (plateaus 1.2e-2, p → 27) | **GROW** 45 → 2.7e+18 |
   | 0.4 | GROW (slow, p → 187) | DECAY → 2.3e-6 | STAGNANT (slow growth) |
   | **0.3** | **DECAY** → 6.4e-5 | **DECAY** → 2.4e-6 | **DECAY** 42 → 1.1e-3 |
   | 0.1 | DECAY (slow) | DECAY → 1.6e-5 | GROW (decays to 8e-3, regrows) |
   | 0.05 | STAGNANT | DECAY → 3.2e-5 | GROW (decays to 1.2e-3, regrows) |

   (step-1 CD is trivially FLAT0 in every run — under-compressed, p = 0.)
   The window is ≈ **[0.2, 0.35]**: 0.5 has spectral radius > 1 (≈1.2×/iter
   asymptotic on step-1 DF), 0.3 contracts in every state, and the low
   omegas decay first then **regrow late** (p stays small — a clamp-limited
   fixed point and/or a weak mode, not a blowup). It is a **matrix
   problem, not a budget problem**: at 0.5 a bigger budget only grows more,
   and 0.3 reaches the tolerances inside 100 iters in every state (step-1
   DF at ~iter 90-100, step-2 DF at ~iter 16). The reference runs 0.5 in
   3D, where the composed operator is better conditioned; the window
   narrowing here is consistent with the free-surface / five-layer-band
   conditioning hypothesis (Part 28, lever c) — now a note, not the
   blocker.

   **Validation (measured).**
   - `hydrostaticColumn` (nx=32), 300 steps: `diverged=False`; every solve
     converges (CD 2-8 iters, DF 4-74, err at the ~2.5e-3 tol); CD p ≤
     11.5, DF p ≤ 8.4, |a_p|max 12-75 (O(g), not the old 1e5-1e18);
     |v|max 0.01 → 1.29 (the slump) → bounded slosh 1.4-1.6; rho max
     1.02-1.07, min 0.61-0.78. The ratchet (|v|max 0.01→1.76→1732→NaN at
     step 6) is gone.
   - 1500 steps × 3 runs (same code, same seed): **run 1 completes
     bounded** (`diverged=False`, CD p 7-10, |a_p|max ≤ 47 at the tail).
     **Run 2: NaN at step 1160** — preceded by ~100 steps of free-surface
     degradation (rho_min 0.61 → 0.31 → 0.21 → 0.14, the surface layer
     diluting), then p → 1.8e6, |v|max → 5072. **Run 3: the same late-time
     failure, a different face** — the column collapses around step ~1150
     into a uniform rho-0.139 soup (rho min == max), |v|max → **inf**, both
     solves trivially "converge" (it=2, err=0, p=0) and the run continues
     degenerate — and the runner's **NaN-based divergence check does not
     catch it** (`diverged=False` with inf velocities; the tells are
     rho min == max and `pWall[empty]`). So the long-time behaviour is:
     stable to t ≈ 1.1 s (step ~1100-1150), then a free-surface degradation
     mechanism that in **2 of 3 runs** progresses to full failure; the
     details (NaN vs. inf-soup, exact step) are GPU-non-deterministic.
   - `staticBlob` (nx=128, 30 steps) A/B — the goal that unblocks Part 27:
     max |v| over the run **1.15 (alpha factor) / 1.28 (dfsph faithful
     factor)**, both `diverged=False`, centroidDrift ~1e-9, KE ~0.033.
     Before (Part 27, the nonlinear solve): **70.9 / inf**. The factor
     delta is gone — the regression was the nonlinear re-summed solve,
     exactly as Part 27's blocker analysis predicted. The residual |v|~1.1
     blob slosh is **pre-existing** (it was 70.9 before the factor change;
     the blob's initial state, like the column's, has no standing
     over-compression for the one-sided pressure to hold) — a separate,
     smaller limitation.
   - 20/20 incompressible tests (`tests/test_incompressibleKrylov.py`) pass;
     the change is inside `dfsphReference` only.

   **What remains (both are step 3 / step 5 territory, not step 4).**
   (a) The late-time free-surface degradation (t ≈ 1.1 s, step ~1150; 2 of
   3 long runs): surface particles dilute to rho_min ~0.14 and the column
   loses integrity (blowup, or the uniform-soup collapse with inf
   velocities). Part 26's gauge experiment (`kappa^v` held at 0 on
   free-surface particles) was run under the old nonlinear solver and made
   the slump faster — it is now **re-testable** under the linear solve, and
   the reference's own guard (<7 neighbours) already removes the most
   extreme surface rows from the DF source. A divergence check that also
   catches the inf-velocity soup (e.g. on the trajectory's `maxVelocity` or
   a rho min==max tell) is a one-line probe fix worth landing with it.
   (b) The bounded but non-zero slosh in both the column (|v|max ~1.3-1.7)
   and the blob (~1.1): the initial
   states are not hydrostatic rest for a one-sided pressure, and nothing in
   the scheme damps the slosh (no physical viscosity in the test config).
   Neither is a solve-contraction failure — every solve in every sampled
   step converged inside its budget.
   Tolerance calibration note (deferred): the reference's eta is
   percent-of-rho0 (1e-4·rho0 CD; 1e-3/h·rho0 DF) against the config's
   dimensionless 5e-4/2.5e-3; the DF solve sits right at its tol
   (err ~2.4-2.5e-3) — fine at this tol, but re-tune before tightening.

   Landed by Part 29: the one-sided metric + 2D <7 guard (source, warm
   start, metric) in `dfsphReference._jacobiSolve`, omega = 0.3, the
   100-iteration budget override in `dfsphReference_step`, the module
   docstrings, and the new probe
   `scripts/probe_dfsphReferenceContraction.py`. The scheme is a
   troubleshooting artifact (not a landed solver): `hydrostaticColumn`
   is now stable for hundreds-to-~1100 steps instead of NaN at step 6, and
   the Part 27 `staticBlob` regression is fixed — the next experiments are
   the step-3 free-surface gauge under the linear solve and (if the slosh
   matters) a damped warm start.

   **Part 30 — step 3 re-run under the linear solve: the free-surface gauge
   is a measured negative, and the runner now catches the inf-soup.**

   **What landed.** (1) The Part 26 gauge, implemented in
   `dfsphReference._jacobiSolve` under the Part 29 linear Jacobi: the
   divergence solve holds `kappa^v` = 0 on the rows the case's own
   (dilated) `detectFreeSurface` flags (124-177 of 465 fluid rows, 27-38%,
   matching Part 26's ~27%). Mechanically the gauge rows join the
   reference-deficient rows — their DF source, warm start, and metric
   residuum are zeroed — and the pressure is additionally **pinned to 0 at
   every iteration**, so the returned field (and the warm start it carries
   into the next step) is 0 on those rows and the final acceleration sees
   no surface-row pressure. DF solve only — Part 26 measured that masking
   the constant-density solve over-compresses the sub-surface layer. The
   module flag `FREE_SURFACE_GAUGE` (default False = the Part 29 baseline)
   and `--gauge` in both `dfsphReference` probes toggle the A/B. (2) The
   one-line runner fix recorded in Part 29: the divergence check is
   `~isfinite` instead of `isnan` (`runner/runner.py`), so a degenerate
   uniform-density collapse with inf velocities is reported
   `diverged=True` instead of running on degenerate.

   **Why the pin, and what it costs.** Pinning inside the solve (not just
   zeroing the final field) makes the interior iterate against a zero
   Dirichlet row set — the surface rows are genuinely not unknowns — and
   the metric zeroing keeps the pinned rows from holding the one-sided err
   above tolerance (which would cost the solve its early exit and push it
   to the 100-iter cap). The per-step `detectFreeSurface` cost (Barecasco
   detection + LambdaGrad normals + 1 dilation) is what `deltaSPH` /
   `divergenceFree` already pay every step; the 1500-step runs completed in
   comparable wall time with the gauge on.

   **A/B (measured).** `hydrostaticColumn` (nx=32), 1500 steps, 2 runs per
   arm, sequential and uncontended (a paired 300-step pre-run was
   concurrent and is not used for the conclusion):

   | run | gauge | outcome | \|v\|max (slosh) | rho_min 201 -> 1500 |
   |---|---|---|---|---|
   | 1 | off | **diverged @ step 1279** — the inf-velocity soup (uniform rho 0.139), now caught by the `~isfinite` check (Part 29's run 3 reported this same face `diverged=False`) | 1.26-1.41 | 0.62 -> 0.38 -> soup 0.14 |
   | 2 | on | completed 1500 | 1.79-2.04 | 0.60 -> 0.21 -> 0.24 |
   | 3 | off | completed 1500 | 1.29-1.53 | 0.68 -> 0.25 (step 600) -> **recovers to 0.49** |
   | 4 | on | completed 1500 | 1.77-2.06 | 0.51 -> 0.16 (step 800) -> 0.23 |

   Read: the degradation's **onset is the same in all four runs** (~step
   300-400) — the gauge does not delay or prevent the late-time surface
   failure. What it does: degrade the surface **deeper** (rho_min 0.15-0.21
   vs 0.25-0.38) and **block recovery** (gauge-on runs end 0.23-0.24; the
   gauge-off survivor recovers to 0.49), and raise the bounded slosh
   ~30-40%. The blow-up count (1/2 off vs 0/2 on) is inconclusive at n=2
   against Part 29's documented 2/3 baseline and its GPU
   non-determinism; the surface-integrity and slosh metrics, which compare
   cleanly, consistently favour the gauge off.

   **Why it is negative (the Part 26 mechanism, confirmed).** The mask's
   benefit (keeping sampling noise out of `kappa^v`) is real but small; its
   cost is structural — with the surface rows removed from the solve's
   unknowns, the sub-surface layer loses the pressure support that even a
   noisy `kappa^v` was providing, so the surface dilutes faster and never
   heals. Part 26 saw the same sign under the nonlinear solve (faster
   slump); Part 30 sees it under the linear solve (deeper, unrecovered
   surface + higher slosh). The gauge is not the lever for this failure
   mode.

   **Also measured.** `staticBlob` (nx=128, 30 steps, faithful factor):
   max |v| 1.12 (gauge on) / 1.16 (gauge off) against Part 29's 1.28 — the
   gauge does not regress the blob (its original motivation, Part 26's
   "jittered quiescent blob blows up", was already fixed by the linear
   solve). 20/20 incompressible tests pass. Step-to-time mapping at nx=32
   (for the record): dt is pinned at `maxDt = 1e-2` to ~step 30, then
   ~0.007, so 300 steps ≈ 2.3 s and 1500 steps ≈ 10 s — Part 29's "t ≈ 1.1
   s at step ~1150" was an estimate from the degraded-dt regime, not a
   step-to-time contradiction.

   **What remains / next.** The late-time surface degradation stands as the
   live blocker, now with a second negative (Part 26 nonlinear, Part 30
   linear) on the surface-mask lever. The recorded next lever is the
   reference's **damped warm start** against the current full-`kappa`
   carry: SPlisHSPlasH warm-starts `0.5·min(κ, cap)·scale` gated on
   `densityAdv` instead of carrying the full field (CD cap 2.5e-4·invH², DF
   cap 0.5·invH — exact constants to be re-verified against the source),
   which may cap the surface accumulation both mask experiments showed is
   the live driver.

   Landed by Part 30: `FREE_SURFACE_GAUGE` + the `surfaceMask` pin in
   `dfsphReference._jacobiSolve`, `--gauge` in `probe_dfsphReferenceColumn.py`
   and `probe_dfsphReferenceStaticBlob.py`, the `fsGauge` count in the
   probe DI line, the `~isfinite` divergence check in `runner/runner.py`,
   and this record. The gauge is left in the tree (default off) so the A/B
   can be re-run cheaply; it is not on by default.

   **Part 31 — the reference's damped warm start against the full-`kappa`
   carry: null on onset and end-state, mildly favourable on surface depth,
   ~5x the CD iterations — and a baseline defect it exposed.**

   **What landed.** The reference's `USE_WARMSTART` / `USE_WARMSTART_V`,
   verified against `TimeStepDFSPH.cpp` (08-30, constants re-verified): the
   reference does not carry the solved pressure as-is — it stores `p*h**2`
   (CD) / `p*h` (DF), dt-invariant (the `*= h**k` blocks at the end of each
   solve), and seeds the next solve with `0.5*min(stored, cap)/h**k` GATED
   on the row being compressed (CD: `densityAdv > 1`; DF: clamped
   `delta > 0` — both are "the one-sided source is negative" in this code's
   sign convention, `source = 1 - densityAdv` / `source = -delta`), zero
   otherwise; caps in stored units CD 2.5e-4, DF 0.5. Implemented as the
   `DAMPED_WARM_START` module flag (default False = the Part 29/30 full
   carry) in `dfsphReference`: the carry block stores the same dt-scaled
   field, `_jacobiSolve` seeds with `0.5*min(warmStart, cap)/opDt` where
   `opDt` is `dt**2` (CD) / `dt` (DF), gated on `source < 0` evaluated
   AFTER the exemption zeroing (deficient/pinned rows seed 0, exactly as the
   reference's zeroed `densityAdv` does), and step 1 seeds from 0 (the
   reference has no IC pressure at all; the case's raw hydrostatic profile
   is not a dt-scaled field). `--warmStart` in both probes.

   **Baseline defect the A/B exposed.** The full-carry arm's step-1 CD solve
   IS seeded with the IC hydrostatic profile (carried max 6.15 at t=0,
   measured by wrapping `dfsphReference_step`), but the solve's two forced
   iterations (minIters = 2; the one-sided metric reports err = 0 on the
   under-compressed column) run the TWO-SIDED update `p = max(p - 0.3(s -
   aij_pj)*invDiag, 0)` with `s = 1 - rho/rho0 > 0` everywhere, driving the
   seed to exactly 0 in 2 iterations (the DE line: `it=2 err=0.00
   p[+0.00,+0.00]`). The baseline is therefore effectively a cold start at
   step 1: the CD pressure is rebuilt from 0 over ~10 steps (DE p max 0 →
   2.9 by step 10), and that — not merely "the one-sided pressure
   responding to an initial state with no standing over-compression" as the
   Part 29 docstring had it — is the initial slump's true origin. The gated
   damped seed is structurally immune to the destruction: it exists only on
   compressed rows, where the update adds. (The A/B itself is unaffected:
   both arms share the identical step-1 state — CD p = 0, DF p built from 0
   in 65-66 iters.)

   **A/B (measured).** `hydrostaticColumn` (nx=32), 1500 steps, 2 runs per
   arm, sequential and uncontended (12.5 min for the four-run chain):

   | run | arm | outcome | rho_min low → end | onset (first < 0.50) | CD iters |
   |---|---|---|---|---|---|
   | 1 | full | completed 1500 | 0.243 → 0.626 (recovers) | step 368 | 2-17 (med 4) |
   | 2 | damped | completed 1500 | 0.260 → 0.480 | step 429 | 20-38 (med 22) |
   | 3 | full | completed 1500 | 0.227 → 0.342 (degraded end) | step 381 | 2-18 (med 4) |
   | 4 | damped | completed 1500 | 0.259 → 0.490 | step 226 | 18-39 (med 22) |

   Read: **no blow-up in this batch, in either arm** (0/4 vs Part 30's 1/4
   and Part 29's 2/3) — the blow-up face is batch-stochastic, not an arm
   effect. The degradation's **onset is the same in all four runs** (step
   226-429) — the damped warm start does not delay or prevent the late-time
   surface failure. Surface depth is mildly favourable (rho_min low 0.259-
   0.260 vs 0.227-0.243; the damped mid-run surface holds higher — 0.685 at
   step 301 in run 2 vs 0.521/0.572 for the two full runs — though run 4
   matches the full arm at 0.532), not conclusive at n=2; end-state is
   comparable (damped 0.480-0.490, consistent; full 0.342/0.626, split);
   late slosh is unchanged (|v|max 1.18-1.79 in the last 300 steps of every
   run, early-slump peak 1.92-2.00 in all four). Cost: the CD solve runs
   ~5x more iterations (median 22 vs 4; range 18-39 vs 2-18) because the
   capped/gated seed starts far from the standing field; the 100 budget
   still covers it (no CD budget hits in any run).

   **Also measured.** `staticBlob` (nx=128, 30 steps, faithful factor):
   max |v| **0.348 (damped) vs 1.08 (full)**, KE 0.0015 vs 0.0305,
   centroidDrift ~5-7e-9 — the damped seed tames the blob's residual slosh
   (Part 29's 1.15-1.28, Part 30's 1.12-1.16) without destabilising it.
   20/20 incompressible tests pass.

   **Verdict / what remains.** Not a fix: the late-time surface degradation
   now survives three levers (Part 26's gauge under the nonlinear solve,
   Part 30's gauge under the linear solve, Part 31's damped warm start).
   The toggle ships off (default = the Part 29/30 full carry); the
   mildly-favourable depth trend and the blob-slosh taming are real but do
   not justify the ~5x CD cost while the blocker stands. The harden track
   is at a decision point: either a targeted mechanism study of the
   degradation itself (what the ~step 200-450 onset does to the surface
   rows — which rows, what the rho/velocity structure is at onset — before
   another lever is tried), or a return to the ranked plan items (item 1:
   the dam-break dissipation mechanism).
4. ~~Give `dambreak` an incompressible `timestep` hook.~~ **Done (Part 20)** —
   landed as `dambreakTimestep`, active only under `--scheme divergenceFree`.
   Worth ~1.7x fewer steps at the case's own safe `cflFactor = 0.2`, not the
   ~5x guessed, and not free (`rho_max` 1.105 against the fixed-`dt`
   baseline's 1.004) — see Part 20 for why the published 0.4 diverges here.
5. **Grade `shearWave` against [C]'s Fig. 3 and Fig. 4** (§4 item 8's
   remainder). Blocked on the paper — `literature/MANIFEST.md`.
6. **Warm-start the divergence-free solve** (§4 item 9, split by Part 15).
   Unblocked; do **not** warm-start the constant-density solve.
7. **Re-measure the divergence-free half-state's contraction** under `minShift`
   (§4 item 3) — still the one mechanism observed and never explained.
8. **Then** the rename and the scheme split.

### What is next, concretely

Item 2 is done (Part 23) and item 3 is scoped (Part 24), started (Part 25),
its divergence-solve half explored and parked (Part 26), and its factor half
done (Part 27). The scheme cannot hold a quiescent wall-bounded column under
gravity — but a DFSPH-proper scheme (`dfsphReference`, scaffolded) holds the
exact hydrostatic gradient before the composed pressure primitives lose it.
**Step 1 is done (Part 25)**: the wall-adjacent `kappa` runaway is removed by
carrying the boundary volume at 2x (`akinciBoundaryVolumeScale`). **Step 2 is
done (Part 27)**: the faithful DFSPH factor (`computeDFSPHFactor`, SPlisHSPlasH
line for line) is its own kernel and wired into `dfsphReference._factor`, and
the composed `a_p` is verified already the standard SPH pressure acceleration
(O(N²) check, ~5e-7) — so *both* composed pressure primitives are now faithful,
and the "Akinci boundary force as its own kernel" resolves to no new kernel.
The re-check: the `hydrostaticColumn` slump **survives** (a modest gain,
`|v|max` ~1.25→1.17, `rho_min` ~0.68→0.70 over 30 steps) — confirming it is
Part 23's mechanism 1 as a *solve*-quality limit, not a formula-fidelity gap.

**Step 4 is done (Part 29).** The linear SPlisHSPlasH Jacobi (Part 28, signs
fixed) now **contracts**: the reference's one-sided compression-only metric +
2D <7-neighbour deficiency guard were adopted, the omega sweep
(`probe_dfsphReferenceContraction.py`) showed the reference's 0.5 is outside
this composed operator's Jacobi window (≈[0.2, 0.35] measured — 0.5 grows
~1.2×/iter, 0.3 decays in every (step, mode) state, 0.1/0.05 regrow late)
and it is set to **omega = 0.3** with both budgets at the reference's **100**.
The ratchet is gone: `hydrostaticColumn` (nx=32) runs hundreds-to-~1100 steps
with every solve converging (2-100 iters), pressures bounded (CD ≤ ~11, DF ≤
~10), and a bounded post-slump slosh (|v|max ~1.3-1.7); and the `staticBlob`
A/B — the goal that unblocked the Part 27 regression — recovered:
70.9/inf → max |v| **1.15 (alpha) / 1.28 (dfsph factor)**. Two residuals
remain, both outside step 4: a **late-time free-surface degradation** at
step ~1150 (surface rho_min → 0.14, then blowup or a uniform rho-0.139 soup
with inf velocities; 2 of 3 1500-step runs fail there) — now re-tested under
the linear solve (Part 30, below); and the bounded non-zero slosh in the
column and the blob (initial states with no standing over-compression for a
one-sided pressure).

**Step 3's re-run under the linear solve is done (Part 30) — a measured
negative.** Part 26's free-surface `kappa^v` gauge (hold `kappa^v` = 0 on
`detectFreeSurface`-flagged rows, now implemented as the `FREE_SURFACE_GAUGE`
toggle with the pressure pinned to 0 at every iteration of the linear solve)
does not delay or prevent the late-time surface degradation — the onset is
the same in both arms (~step 300-400) — it degrades the surface deeper,
blocks the recovery the gauge-off survivor shows, and raises the slosh
~30-40%. The blow-up count (1/2 off vs 0/2 on) is inconclusive at n=2. The
runner's `~isfinite` divergence check also landed, so the inf-velocity soup
now reports `diverged=True`.

**The damped warm start is done (Part 31) — also not a fix.** It was
implemented as the `DAMPED_WARM_START` toggle (default off, `--warmStart` in
both `dfsphReference` probes), with the constants re-verified against
`TimeStepDFSPH.cpp` (CD cap 2.5e-4, DF cap 0.5, in stored `p·h^k` units; the
carrier is now dt-scaled on this path and the step-1 IC pressure is zeroed,
matching the reference's zero-start field). The 1500-step column A/B: same
onset in all four runs (step 226-429), comparable end-state, mildly
favourable surface depth at n=2 (rho_min low 0.259-0.260 vs 0.227-0.243,
not conclusive), no blow-up in either arm (0/4 — the blow-up face is
batch-stochastic), at ~5x the CD iterations (median 22 vs 4). The A/B also
exposed a baseline defect: the full-carry arm's IC hydrostatic seed
(max 6.15 at t=0) is driven to 0 by step 1's two forced CD iterations, so
the baseline is effectively a cold start. The late-time surface degradation
now survives **three levers** (Part 26 nonlinear gauge, Part 30 linear
gauge, Part 31 damped warm start). The toggle ships off, and the harden
track is at a decision point — either a targeted mechanism study of the
~step 200-450 onset itself (what it does to the surface rows: which rows,
rho/velocity structure at onset) before another lever is tried, or a return
to the ranked items below.

The other immediate step is **item 1: the dam-break dissipation mechanism** —
the `nx` convergence of the incompressibility-cycle net on the dam break, now
also a stability study since the default `nx = 128` diverges and Part 23 shows
the scheme is resolution-sensitive on quiescent states too. Items 1 and 3 are
independent and can run in parallel. Items 5-7 stand as ranked; the rename and
the scheme split (8) stay last.
