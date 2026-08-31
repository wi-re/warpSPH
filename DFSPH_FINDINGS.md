# warpSPH — Incompressible (VD+PS / DFSPH) — Findings & Reference

Durable reference material for the incompressible SPH path
(`schemes/dfsph.py`, registered `IncompressibleSPHScheme.divergenceFree`;
plus the `dfsphReference` troubleshooting scheme). Extracted 2026-08-31 from
`DFSPH_IMPROVEMENT_PLAN.md`, which is now just the current state and the
actionable list.

- **Section numbers are kept from the original** so every internal `§N`
  cross-reference still resolves. There is no §4 or §10 here: those were the
  part-by-part investigation narratives (Parts 1–33), which live in
  `git log -p DFSPH_IMPROVEMENT_PLAN.md` and are indexed one line each in §9.
- **Units note.** `cflFactor` on the incompressible cases multiplies the
  particle **diameter** `dx` (Bender & Koschier's form, `dt <= 0.4 d/|v_max|`).
  Before Part 12 it multiplied `h = n_h dx`; with `n_h = 4` that is 4x per
  step for the same number, so **every `cflFactor` in git history before
  commit `9a9bfe7` means 4x more travel than the same number means now**.
  `cflFactor = 0.4` here == `0.1` in the old units == "the published CFL".

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
| **`dfsphReference` wall-XSPH `ε_b = 0.1`** (Part 33) | Part 32's n=1 "first lever to hold the late-time surface" **does not confirm at n=3** (`hydrostaticColumn` nx=32, 1500 steps): onset delay only ~50–100 steps, end-state (rhoEnd 0.14–0.64) inside the baseline batch spread (0.14–0.44), 1/3 to the inf-soup. Part 32's win was a lucky draw on both the wall-XSPH and the baseline single runs. Not a fix. |
| **`hydrostaticColumn` rest-density calibration** (`calibrateRestDensity` param, Part 33) | Normalises the fluid mass so the at-rest bulk reads `rho0` (the `n_h = 4` Wendland lattice integrates to ~0.95, a flat deficit floor to top — §1.1). It *does* kill the Part 31 IC-seed self-destruct (`s = 1 − rho/rho0 ≈ 0` at rest). But **with the divergence solve in the loop it detonates**: the surviving seed's `a_p_cd` feeds a non-uniform `vEnterDf` into the DF Jacobi (which is fine only for the near-uniform `v* = dt·g` it sees when the seed is dead), 3/3 immediate blow-up at step ~14. Paired with the damped warm start it survives 1500 steps but degrades the surface *earlier and deeper* than baseline (onset 5–28 vs 256–345). In `iisph` mode (no DF solve) it only speeds the gradient build, which plain IISPH reaches unaided. Param ships **off**. |
| **The Part 31 damped warm start under the single-solve (`iisph`)** (Part 33) | The gated/capped seed *starves* the accumulating `kappa` — the hydrostatic gradient never forms (`pressureSlopeRatio` late-run ~0 vs 0.92–1.06 for full-carry / cold). The single solve wants the full carry (or a cold start); the damped seed only made sense as a stability aid for the two-solve structure it was designed against. |
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
| `cflFactor` applied to `h`, not `dx` | **fixed and committed** (Part 12, commit `9a9bfe7`) |
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
| `cflFactor` (incompressible cases) | **`0.4`** | Committed (Part 12, `9a9bfe7`); multiplies `dx`. See §7. |
| `IncompressibleSPHScheme.iisph` (Part 33) | — | Plain IISPH ([I]): the CD Jacobi as a velocity impulse, no divergence pass, no VD+PS shift. `iisph_step` = `dfsphReference_step(..., skipDivergence=True)`. First scheme to hold `hydrostaticColumn`; also holds `staticBlob` where `dfsphReference` diverges. Reuses `DFSPHReferenceSystem` + the incompressible codecs. |
| `dfsphReference.SKIP_DIVERGENCE_SOLVE` (module flag, Part 33) | `False` | Drops the divergence-free pass. `False` = the two-solve DFSPH order. Superseded for production by the `iisph` scheme; kept for ablation in the probes. `dfsphReference_step`'s `skipDivergence` kwarg overrides it. |
| `hydrostaticColumn` `calibrateRestDensity` (case param, Part 33) | `False` | Normalise the fluid mass so the at-rest bulk reads `rho0` (the `n_h = 4` lattice integrates to ~0.95). **Measured negative** (§2): detonates with the DF solve in the loop, not needed under `iisph`. Kept as a documented dead end. |

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
- `probe_dfsphReferenceColumnBatch.py` *(new, Part 33)* — multi-arm ×
  multi-run 1500-step `hydrostaticColumn` batch under `dfsphReference` /
  `iisph`, no per-step trace. Arm tokens are `epsF:epsB[:flags]` with flags
  from `w` (damped warm start), `c` (`calibrateRestDensity`), `g`
  (`FREE_SURFACE_GAUGE`), `s` (`SKIP_DIVERGENCE_SOLVE` / plain IISPH).
  Reports per run: onset (first `rho_min < thresh`), blow-up (`~isfinite`
  **or** `|v|max > 1e4` — the inf-soup has finite components, overflowing
  norm), `rhoEnd`, run `rho_min`/`|v|max`, and `slopeLate` (late-run
  `pressureSlopeRatio` — 1.0 = the hydrostatic gradient). GPU
  run-to-run nondeterminism is real here; use `--runs >= 3`.
- `probe_dfsphReferenceColumnSurface.py` *(new, Part 33)* — the
  onset-mechanism study. Wraps the step and, every `--interval`, reports
  column geometry (base/surface y, height, count above the initial
  surface), the diluted-population structure (`rho <` thresh: count, mean
  height above the surface, mean `v_y`, mean neighbour count, persistence
  vs the previous sample), the original top-layer cohort's fate, and the
  KE split top-10% vs the rest. This is what showed the `minDensity`
  readings are transient ballistic spray, not structural collapse.
- `probe_dfsphReferenceContraction.py` *(Part 29)* — omega sweep × 256 iters,
  what-if trajectories re-driven from the exact production inputs; the tool
  that pinned `omega = 0.3` as inside the composed operator's Jacobi window.
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
| 32 | 08-30 | **The free-slip diagnosis, confirmed in code — the scheme family has no tangential stress anywhere — and the reference's XSPH wall drag is the first lever to change the late-time surface (the slosh still does not decay).** Trigger: the user's reading of the best-config video (1500-step column, velocity/pressure/density panels) — the post-slump motion is particles sliding freely down the walls, deflecting along the floor, and setting up two counter-rotating vortices (a dipole), the classic free-slip boundary effect. **Confirmed in `wp_viscosityDelta.py`:** the deltaSPH viscosity term projects each pair's relative velocity onto the pair axis and clamps it to the approaching case (`mu_ij > 0 → 0`) in BOTH the `alpha` and the `nu` branch — purely normal, compression-only — and the case runs `nu = 0.0` / `alpha = 0.01` (the documented stability floor, not a physical choice); so the wall is EXACTLY free-slip and the bulk vorticity is undamped: the pressure solve bounds the slosh, nothing decays it, and the three prior levers (26/30/31) were all pressure-solve levers that never touched the energy source. **Landed** the reference's XSPH (`XSPH.cpp`: `a_i -= (1/h)·eps·Σ_j (m_j/ρ_j)(v_i−v_j)W_ij` as a non-pressure acceleration, summed over fluid + static-boundary neighbours — the boundary's v = 0 is the wall drag — with separate `xsph`/`xsphBoundary` coefficients, both default 0 there too) as `XSPH_FLUID_EPSILON` / `XSPH_BOUNDARY_EPSILON` (default 0.0 = off), `--xspH` / `--xspHBoundary` in both probes, two `warpOperation(Interpolate)` calls (the per-kind epsilon folded into the reference values; `state.supports` is h itself, verified against `volumeToSupport`); the boundary takes no reaction (static, like the rest of the scheme). **A/B** (`hydrostaticColumn` nx=32, 1500 steps, sequential, 1 run per arm): **wallOnly (ε_b = 0.1) holds the late-time surface** — rho_min 0.52/0.51/0.58 @600/900/1200, ends **0.497** vs baseline **0.228** (run low 0.325 vs 0.226) — the first of the four levers to do so — **but does not decay the slosh** (|v|max flat ~1.35-1.6, run peak 1.991 vs 1.888); **fluidOnly (ε_f = 0.1) is a measured negative** — the surface degrades earlier and deeper (0.34/0.27 @600/900 vs 0.39/0.45) and the run diverges ~step 1200 in the Part 29/30 inf-soup mode (|v|max 7.5e17, rho uniform 0.139). Reading: the dipole's energy is injected by the initial collapse (the Part 31 IC-seed cold start) and persists in the tangentially-stress-free bulk; wall drag removes energy only where the overturning flow touches the wall. Caveat: the wallOnly arm is n=1 and its end sits inside the baseline's batch range (Part 31's full-carry runs ended 0.626/0.342; today's baseline 0.228) — a 2-run confirmation batch was in flight when this was written. 20/20 incompressible tests pass (the default-off path is unchanged). |
| 33 | 08-31 | **The late-time degradation resolved — it was the divergence-free Jacobi (instability) plus `minDensity` reading ballistic surface spray (FOM artifact); `IncompressibleSPHScheme.iisph` (plain IISPH, [I]) landed as the first scheme in the codebase to hold `hydrostaticColumn`.** Wall-XSPH `ε_b = 0.1` confirmation batch (Part 32's n=1 lead, now nx=32 / 1500 steps / 3 runs): **not confirmed** — onset delay ~50–100 steps, end-state inside the baseline spread, 1/3 to the inf-soup; Part 32's single-run win was a lucky draw both sides. Origin-fix attempt: a `hydrostaticColumn` `calibrateRestDensity` param (default off) that normalises the fluid mass so the at-rest bulk reads `rho0` instead of the ~0.95 the `n_h = 4` Wendland lattice integrates to — it stops the Part 31 IC-seed self-destruct (`s = 1 − rho/rho0 ≈ 0` at rest, so the two forced CD iters no longer drive the seed to 0) but **detonates with the DF solve in the loop** (the surviving seed's `a_p_cd` feeds a non-uniform `vEnterDf` into the DF Jacobi → 3/3 immediate blow-up; +damped warm start survives but degrades the surface *earlier/deeper* than baseline). Then the lever that worked: **`SKIP_DIVERGENCE_SOLVE` — run only the CD solve as a velocity impulse, i.e. plain IISPH.** Batch (`0:0` × {plain, damped, calib, calib+damped}, 1500 × 3): **0/12 blow-ups** (vs 1/3–3/3 with the DF solve), and plain IISPH builds the correct hydrostatic gradient (`pressureSlopeRatio` late-run 0.92–1.06) with no calibration — calibration only speeds it, the damped warm start *starves* it (`slope` ~0, the gated/capped seed under-supplies the accumulating `kappa`). Onset-mechanism study (`probe_dfsphReferenceColumnSurface.py`, plain IISPH, 2000 steps): column geometry flat (surface height 0.45–0.46 throughout), embedded-column min density 0.76–0.92, 5th-pct density 0.75–0.85 — **stable**; the `minDensity` → ~0.14 readings are 1–3 fluid particles thrown 1–3 dx above the surface by the bulk slosh (10–25 neighbours vs ~50, `rho` low by kernel deficiency), never the same two samples running (`persist` = 0), falling back — cosmetic spray. The bulk carries a **bounded undamped free-slip slosh** (Part 32's vortex pair; KE plateaus ~0.12, neither grows nor decays; `nu = 0`, no tangential stress) — not fatal, keeps the spray alive; decaying it is a viscosity/XSPH choice, not a pressure-solve lever. **Landed:** `IncompressibleSPHScheme.iisph = 2`; `schemes/dfsphReference.py::iisph_step` (calls `dfsphReference_step(..., skipDivergence=True)`; `dfsphReference_step` took a `skipDivergence` param overriding the `SKIP_DIVERGENCE_SOLVE` module flag); `schemes/builder.py::_iisph` (reuses `DFSPHReferenceSystem` + the incompressible codecs); `hydrostaticColumn` seeds the raw hydrostatic profile for `iisph` as for `dfsphReference`. Also holds `staticBlob` nx=64/60 where the two-solve `dfsphReference` diverges (not a regression — the two-solve baseline diverges there too). Full test suite green, `gradcheck_incompressible.py` green (no new kernel — the CD Jacobi already carries `computeAlpha`'s IISPH `a_ii`). No `divergenceFree` default changed. |
