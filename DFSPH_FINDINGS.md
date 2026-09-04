# warpSPH — Incompressible (VD+PS / DFSPH) — Findings & Reference

Durable reference material for the incompressible SPH path
(`schemes/dfsph.py`, registered `IncompressibleSPHScheme.divergenceFree`;
plus the `dfsphReference` troubleshooting scheme). Extracted 2026-08-31 from
`DFSPH_IMPROVEMENT_PLAN.md`, which is now just the current state and the
actionable list.

- **Section numbers are kept from the original** so every internal `§N`
  cross-reference still resolves. There is no §4 or §10 here: those were the
  part-by-part investigation narratives (Parts 1–38), which live in
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

**RESOLVED (Part 49, 2026-09-04): the `c637785` rewrite had dropped
`mdbcNoPenetrationShift`; it is re-wired and back on by default.** `dfsph_step`
had the call (`if schemeConfig.solverConfig.mdbcNoPenetrationShift: dvdt +=
nopenshift/dt`) commented out since the rewrite, so the shipped `divergenceFree`
scheme relied on the pressure projection alone — and a violent wall impact
penetrated the band. Re-graded A/B (`scripts/probe_dfsphWallPenetration.py`,
`dfsph.NOPEN_SHIFT` gates the restored call; folded into `dvdt` *before* the
pressure solves, matching the pre-rewrite step and `schemes/deltaSPH.py`):

| case (nx=64) | shift OFF | shift ON |
|---|---|---|
| `columnCollapse` | pen peak 6, deepest 1.21 dx, `\|v\|`max 9.1 | pen peak 1, deepest 1.10 dx, `\|v\|`max 6.2 |
| `dambreak --scheme divergenceFree` | pen peak 5, deepest 0.82 dx, `\|v\|`max 11.3 | **pen peak 0**, deepest 0.00 dx, `\|v\|`max 8.4 |
| `hydrostaticColumn` (quiescent) | pen 0, `\|v\|`max 0.35 | pen 0, `\|v\|`max 0.22 (no regression) |
| `randomFlowIncompressible --bounded` | BLOW step 23 | BLOW step 21 (unchanged — that is the near-wall CD operator, §1.7 / Active track, not penetration) |

So the shift removes `dambreak`'s wall crossing outright, cuts `columnCollapse`'s
6→1 and its impact `\|v\|` spike, is inert on the quiescent column, and does
**not** touch the `randomFlowIncompressible --bounded` divergence (that is a
different failure — the pure-Neumann near-wall constant-density solve, not
ballistic penetration). Consistent with §2's "removing it is strictly worse".
`dfsph.NOPEN_SHIFT = 'config'` (default) obeys
`schemeConfig.solverConfig.mdbcNoPenetrationShift` (itself `True`), so this is
back to the pre-rewrite behaviour; `True` / `False` force it. `dambreak`'s
`diagnostics` gained an `nPenetrating` / `maxPenetrationDx` watch (off
`ctx.scratch['interiorDomain']`) alongside `columnCollapse`'s. Incompressible
physics suite green.

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

**Part 42 — on a *closed* domain the non-convergence is a specific, fixable
thing: an incompatible source.** Captured `omniIncompressible`'s
constant-density system `A p = s` on `randomFlowIncompressible --bounded`
(a fully-walled box, **no free surface** → the pressure operator is
pure-Neumann, `A·1 ≈ 0`). Measured on step 1: `|s|_2 = 7.72e-2` but
`mean(s_fluid) = -1.2e-3` and `|s - mean(s)|_2 = 1.16e-5` — **the source is
99.98 % its own mean**, and that mean (the §1.1 `n_h = 4` lattice density
bias) is in `null(A)`, so `A p = s` has *no solution*. The Jacobi residual
floors at exactly the incompatible component (`|r|_2` flat at ~8e-2 for 256
iters) and `p` ramps linearly; MINRES / CG break down immediately on the
inconsistent system. **Subtracting `mean(s_fluid)`** (the textbook
pure-Neumann compatibility projection) makes the Jacobi *converge* —
`|r|_2` 1.05e-5 → 1.2e-6 over 256 iters, most of it in the first 8 — on the
resolvable spatial part, with `p` kept mean-zero (a closed box has no free
surface, so no `p ≥ 0` tensile guard). `mean(ρ) ≠ ρ0` is then a rest-density
calibration offset the solve correctly ignores. Landed as
`omniIncompressible.CD_SOURCE_PROJECT = 'auto'` (projects only when the
source is mean-dominated — `frac_uniform > 0.7` — so a strict no-op wherever
there is a free surface: `hydrostaticColumn` step 1 `frac_uniform ≈ 0.09`,
`dambreak` likewise). This is the closed-box half of "make the CD solve
converge"; the free-surface half (where the Jacobi still caps out but the
run holds) still wants a contractive solve / `band2018pb` — plan active
track.

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

### 1.12 The `hydrostaticColumn` failure is the free-slip side walls, not the density solve (Part 38)

Made a semi-periodic variant of the case — x periodic (the side walls gone, so
**zero tangential fluid↔boundary interaction**), a floor wall band in y, free
surface on top (`scripts/splishsplash_compare/semiperiodic.py`). With warpSPH's
native discretization (Wendland2, `n_h = 4`), `omniIncompressible` on it sits at
`|v|max ≈ 0.22`, KE ≈ 1e-4, `embeddedMinDensity` ≈ 0.99, `pressureSlopeRatio` ≈
1.00 for 500 steps — **the column is at rest density with the exact hydrostatic
gradient, no slump, no splash, no volume loss.** `iisph` also holds (`|v|max`
~0.6, `embeddedMinDensity` ~0.96).

So the constant-density velocity-impulse solve's **vertical** stability is
sound. Everything wrong in the fully-walled runs — the bounded free-slip slosh
(Parts 32/33/34), the ~30 % column drop the fully-walled `omniIncompressible`
video shows, the ballistic surface spray — is downstream of the free-slip side
walls: particles slide down them, deflect along the floor, and set up the
Part 32 counter-rotating vortex pair, which is what works the free surface. The
scheme family still has no tangential stress (§ Part 32); a wall no-slip is the
missing ingredient, not a pressure-solve lever. `XSPH_BOUNDARY ≈ 0.15` (a
penalty no-slip: near-wall fluid velocity dragged toward the static wall's
`v = 0`) cuts the walled slosh energy ~2/3; a proper viscous no-slip on the
`applyConsistentCoupling` static-wall path is untried (`BCType.noSlip` exists
only on the mDBC/ghost path).

### 1.13 SPlisHSPlasH holds the column; importing its exact state into warpSPH matches it for ~0.3 s, then the composed Jacobi loses it (Part 36)

SPlisHSPlasH's DFSPH holds a matched 2D hydrostatic column (mild slump peaking
`|v|max ≈ 2.4` at t ≈ 0.11, then monotonic decay, particles stay in the box).
The two libraries discretize the *same* physical case differently — warpSPH
runs `h = n_h·dx = 4·dx`, SPlisHSPlasH `h = 4·particleRadius = 2·dx` (**2×** the
smoothing length at the same spacing), a different kernel (Wendland C2 vs cubic
spline), and a different particle-volume factor (`1.11` vs `0.80 · rho0·dx²`);
warpSPH's initial summation density is mean 0.98 / min 0.64 where SPlisHSPlasH's
0.8 factor is tuned so a regular lattice ≈ rho0.

Importing SPlisHSPlasH's **exact** 8001 fluid positions + mass + `h = 0.015625`
+ cubic kernel + zero velocity + zero IC pressure into warpSPH
(`import_and_run.py`): all schemes start at SPlisHSPlasH's exact frame-1
numbers, and `iisph` / `omniIncompressible` **reproduce SPlisHSPlasH's slump
transient for the first ~0.3 s / ~150 steps** (`omniIncompressible` tracks it
better than `iisph` — `embeddedMinDensity` 0.98, slope 1.0). **Then warpSPH's
time evolution departs**: `dfsphReference` detonates by step ~50, the
`omniIncompressible` Jacobi at t ≈ 0.4, `iisph` doesn't detonate but its
surface thins (`embeddedMinDensity` 0.9 → 0.38 over 500 steps) where
SPlisHSPlasH holds ~0.56. **The setup and the physics match; warpSPH's
operator-composed divergence-free Jacobi is not contractive at the matched
`h = 2·dx` / cubic discretization** where SPlisHSPlasH's dedicated
`TimeStepDFSPH.cpp` at `omega = 0.5` is (cf. Part 29's `omega` window), and its
free-surface handling can't hold the surface. **Do not chase SPlisHSPlasH's
`h = 2·dx`** — it blows up even in the trivial semi-periodic case (§ 1.12);
warpSPH's operators/density estimator need `n_h ≳ 3`.

### 1.14 The free-slip slosh decays under a real viscous no-slip wall, not under XSPH or fluid-only viscosity (Part 39)

§1.12 diagnosed the `hydrostaticColumn` residual as the free-slip side walls
and §Part 32 as the scheme family having no tangential stress. The viscosity
path, graded on `iisph` / nx=128 / 1500 steps (Part 39, via the since-removed
`_physViscosity` toggles — see the follow-up at the end of this section for
the stock-machinery re-do):

- **The WCSPH / Adami 2012 no-slip wall works.** Each static boundary
  particle's velocity is set to the mirror `v_b = 2·v_wall − shepard_f(v) =
  −shepard_f(v)` (`shepard_f(v)_b = Σ_f V_f v_f W_bf / Σ_f V_f W_bf`, the
  Shepard-smoothed fluid velocity at `b`), and a Brookshaw Laplacian
  `nu·∇²v` is taken over the fluid↔boundary pairs only and added as a
  non-pressure acceleration. At `nu ≈ 0.01–0.02`: `|v|max` 1.7 → ~0.8, slosh
  KE from the ~0.04 plateau (creeping *up* over 1500 steps) to ~0.012 and
  *flat*, `embeddedMinDensity` 0.94–0.97 (baseline 0.94), `pressureSlopeRatio`
  ~1.0, `dispMax` drift roughly halved per unit time, bounded to 1500 steps.
  `nu = 0.1` over-drives — the mirror amplifies a noisy near-wall field and
  pumps (`|v|→4`).
- **Fluid-only `nu·∇²v` (`FluidToFluid`, walls left free-slip) roughens the
  free surface.** It decays the bulk KE and pins the column (`dispMax` end
  0.15–0.19 at `nu ≈ 0.03`) but drives `embeddedMinDensity` to ~0.72 at every
  coefficient tried (0.01–0.05): it is diffusing the interior while the
  free-slip walls keep feeding the surface-working motion. `physFW` (both
  terms) pins the column hardest but keeps the fluid term's ~0.82 surface.
- **XSPH is null-to-negative at nx=128 on `iisph`.** Wall drag
  (`XSPH_BOUNDARY_EPSILON` 0.05–0.2) raises `|v|max` (1.7 → 2.2) and KE with
  the coefficient; fluid XSPH (`XSPH_FLUID_EPSILON` 0.02–0.1) pumps KE
  (0.036 → 0.067 at 0.1) and degrades `embeddedMinDensity`. Consistent with
  Part 33 (`ε_b` not confirmed at n=3) and Part 37 (`omniIncompressible`
  wash); Part 38's "`XSPH_BOUNDARY = 0.15` cuts slosh ~2/3" was nx=64 /
  `omniIncompressible` and does not carry to `iisph` / nx=128.

The lesson is §1.12 from the other side: put the stress **at the wall**. A
momentum-diffusing bulk term without a wall condition trades the slosh for a
rough surface; a wall no-slip removes the energy where it enters.

**Part 40 caveat:** omniSPH, run through its own bindings, holds the matched
column *fully inviscid* (bounded, near-rest, no divergence) — so the viscosity
path is the smaller half of the story. See §1.15.

**Follow-up (post-Part-41 cleanup): Part 39's bespoke path was removed and
re-done through the stock machinery, and the result does not carry cleanly.**
Part 39 hand-rolled two things that already exist: (1) the Adami no-slip
*mirror* — that is `computeBoundaryVelocities` with `BCType.noSlip`
(`modules/mdbc/velocity.py`, the same call `schemes/deltaSPH.py` /
`schemes/dfsph.py` make); (2) a shear-carrying Brookshaw vector Laplacian
`ν∇²v` — which the module layer does **not** have (`computeVelocityDiffusion`'s
`inviscid=False` branch is `μ_ij·∇W`, normal-projected + approach-only, so it
applies no tangential stress; `computePi_actual` is the same). The
`PHYS_VISCOSITY_*` globals + `_physViscosity` were deleted; `dfsphReference_step`
(→ `iisph`) now calls `computeBoundaryVelocities` in step 2 and the physical
viscosity is the ordinary `computeVelocityDiffusion` / `schemeConfig.
diffusionParams.viscidNu` already in that step. `hydrostaticColumn` got a
`wallBC` param (default `freeSlip`). Re-graded (`iisph`, nx=128, 1200 steps,
tail = last quarter):

| arm | `\|v\|`mean | `\|v\|`max | KE | embMin | slope |
|---|---|---|---|---|---|
| `wallBC=constant` (≈ pre-change, wall v=0) | 1.72 | 1.94 | 0.042 | 0.940 | 0.99 |
| `wallBC=freeSlip` (**new default**) | 1.72 | 1.94 | 0.042 | 0.942 | 0.99 |
| `wallBC=noSlip` + `nu=0.01` | 1.07 | 3.73 | 0.0098 | **0.60** | 1.02 |
| `wallBC=freeSlip` + `nu=0.01` | — | 4.08 | 0.016 | **0.43** | 0.99 |
| `wallBC=extended` (MLS) + `nu=0.01` | 8.8 | 10.7 | 0.81 | 0.14 | 0.51 |

So: adding `computeBoundaryVelocities` at its `freeSlip` default is a strict
no-op on the landed `iisph` baseline (identical to `constant`, matches Part 34).
`noSlip` + `nu` through the projected `viscidNu` term bounds the slosh KE (4×
down) and keeps the hydrostatic gradient, but roughens the surface
(embMin 0.94 → 0.60) and spikes `|v|` — it does **not** reproduce Part 39's
`_physViscosity` numbers (embMin held 0.94–0.97), because that used a real
shear-carrying vector Laplacian + a mirror-velocity clamp. `extended` (MLS
velocity extrapolation onto the wall) is unstable here.

**A clean viscous no-slip wall for this scheme family needs a shear-carrying
laminar viscosity term (Morris 1997: full `v_ij` vector, no approach-only
clamp) in the module/config layer.** Neither `viscidNu` nor `viscosityParams`
has one. That is a TODO — see `DFSPH_IMPROVEMENT_PLAN.md` ranked queue. Until
it exists, `hydrostaticColumn` stays `freeSlip` and the bounded free-slip
slosh stays documented as non-fatal (§1.12, Parts 32/34/38).

### 1.15 omniSPH holds the column because of its analytic wall boundary, not its viscosity (Part 40)

The `omniIncompressible` port (Part 35) was transcribed from omniSPH's
`SPHSimulation::timestep`. omniSPH's own Python bindings (`omnySPH`, now built
for the warp env — §8) let the reference be run directly. On a matched
hydrostatic column (`scripts/omnisph_compare/column.yaml`, omniSPH's native
`n_h ≈ 1.8`):

- **omniSPH's shipped loop** (DFSPH + `XSPH` `viscosityConstant = 0.01` +
  `BXSPH` `boundaryViscosity = 0.50`, both run unconditionally every step —
  it is *not* an inviscid loop) **decays** the startup slump: `|v|max`
  0.6 → 0.05, `KE/n` 3e-3 → 8e-5, `ρmax` pinned at **1.002**, surface flat to
  ±0.003, `divergenceSolve` / `densitySolve` converge in 4 / ~11 iterations.
- **omniSPH fully inviscid** (`viscosityConstant = 0`, `boundaryViscosity = 0`,
  `ablate_xsph.py`): still **bounded and near-rest** — `|v|max ~0.45`,
  `ρmax 1.002`, `KE/n` ~flat 3e-3, no divergence. Turning off just `BXSPH`
  takes the residual `|v|max` 0.10 → 0.39; just `XSPH` → 0.17.

Against warpSPH's `omniIncompressible` / `iisph` at `n_h = 4`: sloshing at
`|v|max ~1.7` and creeping up, `ρ` band ~1.05, needing `OMEGA = 0.3` (not
omniSPH's 0.5) to not diverge. **The gap is the boundary model:**

| | omniSPH | warp `omniIncompressible` / `iisph` / `dfsphReference` |
|---|---|---|
| wall | analytic solid triangles, `interactTriangle(p,h,tri) → (k, gk)` | 5-layer Akinci **particle** band |
| density near wall | `density()` adds boundary kernel integral `k` → `ρ ≈ ρ0` at rest → CD source ≈ 0 at the wall | summation over fluid + band, uncalibrated → ~0.79 near the floor → standing spurious drive |
| solve operators | `computeAlpha` / `computeSourceTerm` / `computeAcceleration` / `updatePressure` each add the analytic wall gradient `gk` (`boundaryFunc`) | none — `applyConsistentCoupling` wraps the solve once |
| wall pressure | `computeBoundaryPressure` (MLS) recomputed **inside every Jacobi iteration** | not recomputed |
| relaxation | `omega = 0.5` stable | `OMEGA = 0.3`; 0.5 detonates |

So **Part 39's viscous no-slip wall is real** — omniSPH's `BXSPH` is a shipped
wall no-slip and the port had explicitly set `XSPH_BOUNDARY = 0.0`, dropping
it — but it is patching over the missing analytic boundary. omniSPH's inviscid solve *damps* the slosh (or at least
holds it bounded and small) because the analytic wall makes the near-wall
constant-density source ≈ 0 and gives the pressure solve the wall
kernel-gradient feedback it needs. The warp family, with a spurious near-wall
source and no wall `gk`, can only *bound* the slosh with a pressure solve and
needs an added viscous term to decay it. This partly walks back §1.12's "the
failure is *entirely* the free-slip side walls" — the free-slip walls are a
real factor, but they are downstream of a boundary discretization that omniSPH
does not share.

**Direction (user's call, after Part 40).** Do **not** port the analytic
triangle boundary — the interior fluid physics matter more first (the warp
composed Jacobi needing `OMEGA = 0.3` where omniSPH's runs at `0.5`, and only
*bounding* rather than converging, is an operator-level discrepancy the
bindings can now pin down substep by substep). When the boundary *is*
addressed, use **`band2018pb`** — Band, Gissler, Ihmsen, Cornelis, Peer,
Teschner 2018, *Pressure Boundaries for Implicit Incompressible SPH*, ACM TOG
37(2):14 (`literature/`). Its **extended PPE** solves for pressure at boundary
samples *as unknowns*, with their own source term and diagonal, iterated in
the *same* Jacobi loop as the fluid — it maps directly onto the DFSPH/IISPH
step this codebase runs, is volume-centric (not density-centric), and its
abstract promises exactly this codebase's open problems: "reduced pressure
oscillations, improved solver convergence, and larger possible time steps".
**Not** the triangle geometry, **not** the Akinci-volume
`applyConsistentCoupling`, and **not** the `band2018` / `[B]` MLS
extrapolation (which is what `mdbcMlsPressure` already is — the worst boundary
mode measured, §2). Adami 2012 §3.2 is the outside-the-loop extrapolation
`band2018pb` supersedes. omniSPH is also stable at `boundaryViscosity = 0.01`
(not only `0.5`), so any wall no-slip should be light.

### 1.16 On the current `divergenceFree` step the `hydrostaticColumn` residual is a bounded undamped free-surface limit cycle, and a light fluid XSPH decays it (Part 46)

Context: `schemes/dfsph.py::dfsph_step` is now the omniSPH two-solve loop
(`omniIncompressible._solve` for both the divergence and the constant-density
pass) with DFSPH placement, `calibrateRestDensity` on, fixed `dt = 1e-3`, and
the incoming `vEnter` re-passed through `computeBoundaryVelocities` before each
solve. In that configuration it **holds `hydrostaticColumn` at nx=64 and
nx=128** to `t = 1` — no divergence, `pressureSlopeRatio` 1.001, exact-gradient
column. What is left is a **bounded residual `|v|`** that neither grows nor
decays: nx=64 settles at `|v| ~ 0.07`, KE ~1.6e-5 flat for 1000 steps; nx=128
peaks `|v| ~ 1.0` on the startup transient then settles at `|v| ~ 0.26`, KE
~6.7e-4. Anatomy (`probe_hydrostaticColumnDfsphSurface.py`): the motion is in
the **top ~3 fluid rows only** (bulk `|v| ~ 3e-3`), the same skin where plain
summation density truncates to `rho ~ 0.69 rho0` (top row) / `~0.94` (second)
/ `1.000` (row 3 down). The density deficit itself is **static** — those rows
read the same low value at step 0 and step 1000 — so it is SPH free-surface
kernel deficiency (§1.1), not an instability.

Levers measured (`probe_hydrostaticColumnDfsph{SurfaceSource,Xsph,Tune}.py`,
nx=64/128, 250–1000 steps, fixed `dt`):

- **Reshaping the constant-density source at the surface is a wash.**
  `omniIncompressible._solve` gained a `surfaceSource` kwarg
  (`'full'` default = omniSPH; `'clamp'` = `min(1 - rho/rho0, 0)`, one-sided;
  `'mask'` = drop the term on `surfaceIndicators == 1`; `'shepard'` = use the
  0th-order density `rho_sum / sum_j (m_j/rho_j) W_ij`). `'shepard'` lifts
  `densityP05` 0.947 → 0.972 and `minDensity` 0.76 → 0.80 but pushes `|v|`
  0.066 → 0.20 and `embeddedMinDensity` 0.999 → 0.93; `'clamp'` moves nothing
  by 500 steps; `'mask'` spikes `|v|` to 0.23. They trade the error between
  the density axis and the velocity axis. `dfsph.SURFACE_SOURCE` ships
  `'full'`. (Consistent with §2's "one-sided source clamping" published
  negative — [I] §3.2.)

- **Raising the inner-solve iteration counts makes it worse.** The Jacobi is a
  smoother here, not a solver: `DIV_MIN_ITERS` 2 → 6 → 12 takes `|v|` 0.09 →
  0.44 → 1.15 (the extra sweeps amplify the near-singular boundary mode the
  mean-residual test cannot see — §1.7). `RHO_MIN_ITERS` 3 → 10: `|v|` → 0.18.
  Left at the omniSPH values.

- **A post-solve fluid XSPH decays the cycle cleanly, and this is new for the
  DFSPH path.** Folding omniSPH's `SPHSimulation::XSPH` filter into `dvdt`
  *before* the next step's divergence projection (so any divergence it adds is
  cleaned up), `dfsph.XSPH_SCALE` in units of omniSPH's own `XSPH_FLUID =
  0.05`:

  | | nx=64, 1000 steps | nx=128, 500 steps |
  |---|---|---|
  | `XSPH_SCALE = 0` | KE 2.97e-5, `\|v\|` 0.115, `\|v\|@half` 0.066 (flat), presRes 1.2e-3, `densityP05` 0.979 | KE 6.7e-4, `\|v\|` 0.264, presRes 1.2e-2, `densityP05` 0.999, `embMin` 0.992 |
  | `XSPH_SCALE = 1` | KE **5.4e-7** (55×), `\|v\|` **0.011**, `\|v\|@half` 0.025 (decaying), presRes 6.6e-4, `densityP05` 0.947 | KE **3.0e-7** (2250×), `\|v\|` **0.015**, presRes **3.4e-4** (36×), `densityP05` **1.000**, `embMin` **1.000** |

  At nx=64 the only cost is `densityP05` 0.979 → 0.947 (the filter smears the
  skin one row; `embeddedMinDensity` stays ~1.0). At nx=128 every axis
  improves. `XSPH_SCALE = 2` is marginally better on `|v|` at nx=64 but
  overshoots the nx=128 startup peak; `1.0` is the knee.

  Why this is not §1.14 / Parts 37/39 (XSPH "a wash / null-to-negative"):
  those were `iisph` (CD-only, energy-injecting on any vortical flow — Part
  42) and `dfsphReference` in its *diverging* late-time regime, where XSPH fed
  a marginal divergence Jacobi. Here the run is stable, the residual has no
  energy source, and the increment is re-projected next step — so the smoother
  monotonically removes energy.

**`XSPH_SCALE` ships `0.0`** (inert; the periodic cases the scheme is clean on
must stay clean — `tgv` KE at step 400 drops 33% at `XSPH_SCALE = 1`, the
same over-dissipation §1.10 / the plan's "`omniIncompressible` over-dissipative
from `XSPH_FLUID = 0.05`"). It is a per-case knob: turn it to ~1.0 for
`hydrostaticColumn`, leave it off elsewhere.

**The same rewrite regressed `tgv` — but this paragraph's "the projection
injects" reading is SUPERSEDED by §1.17 (it is the missing particle shift; the
divergence-solver A/B below stands on its own merits). Kept for the trail.**

**Original reading:** `tests/test_physics.py::test_tgvKineticEnergy{DecaysAtRoughlyTheAnalyticRate,IsMonotoneDecreasing}`
now fail — `tgv` nx=32 fluid KE *grows* ~6–8% over the first ~15 steps
(`|v|` rises from 22.6 to 23.4 monotonically) before it turns over. The
injector is the divergence-free projection itself, not the constant-density
pass (killing the CD `_solve` leaves the growth unchanged) and not the
pressure-mean gauge (per-iterate mean-centring the divergence `p` moves it
< 1%). It is that `omniIncompressible._solve`'s divergence mode is an
**under-relaxed Jacobi** (`OMEGA = 0.3`, cold start, hard 2-iteration cap):
two sweeps from zero do not fully project `vEnter`, and the semi-implicit
integrator does `+work` with the residual divergence over the cold-start
transient. Iterating it to `tol = 1e-5` still leaves ~4.5% — the under-relaxed
Jacobi plateaus on this operator. The convergent alternative,
`solveDivergenceFree`'s optimal step (`omega_k = <r,q>/<q,q>`) + 0.75x warm
start + per-iterate mean-zero gauge (`dfsph.DIVERGENCE_SOLVER = 'vdps'`), cuts
the `tgv` injection to ~0.8% — **but** the optimal step + mean-centre is the
spurious-force move §1.5 / Part 26 forbid at a free surface, so it cannot
sustain a body force and `hydrostaticColumn` blows up (`|v|` → 5, slope ratio
0.24), and with the position-shift path gone from
`IncompressibleSystem.finalize` it also no longer *decays* `tgv` at the
analytic rate. So the two projections each hold exactly one of {periodic KE,
wall-bounded column}. `DIVERGENCE_SOLVER` ships `'omni'` (hold the column, as
the rewrite intends); closing both at once is Part 23 / the active track, not
a default flip. `probe_dfsphXsphRegression.py` + `scripts/…Tune.py` cover it.

### 1.17 The `tgv` energy injection is a *missing particle shift*, not the projection; restoring it (gated by `gravityConfig.active`) fixes `tgv` + the bounded box with the column untouched — suite green (Part 47)

§1.16's first read blamed the under-relaxed divergence Jacobi. The fuller
picture: the tmp-commit rewrite of `dfsph_step` onto `omniIncompressible._solve`
also **gutted the "PS" of VD+PS** — the post-integrator particle shift in
`IncompressibleSystem.finalize` (`solveIncompressible` with `dvdt = 0`, plus
the Cornelis et al. Eq. 17 resample). Nothing then keeps the particle
*distribution* regular.

**It is a shift instability with a snap.** On `tgv` (`probe_hydrostaticColumnDfsphSurface.py`
lineage, `scripts/make_video_dfsphPsCompare.py`):

- With a **perfect lattice** (`jitter = 0`) there is *no* injection at all —
  KE ×0.9996, peak ×1.0000, monotone. The bump only appears once the jitter
  perturbs the lattice, and it scales *with* the jitter (0.01 → peak ×1.017,
  0.03 → ×1.043). The jitter is relaxed by `relaxLattice` against
  `solveIncompressible`, but the run's `_solve` operator has a different fixed
  point, so steps 1–15 re-snap the distribution — that snap is the KE jump.
- `jitter = 0` only **delays** it: nx=64 detonates near step ~800.
- Over the full second the baseline `tgv` injection *runs away*: KE 9.9 → 18.3
  (×1.86), **peak ×46**, `|v|max` → 3.1 in a unit-speed vortex.

**Restoring the shift resolves it.** `incompressible._RESTORE_PS_SHIFT = True`
with `dfsph.INSTEP_CD = False` (the true pre-tmp VD+PS: divergence projection
in-step, constant density *only* as the finalize position shift) →
`tgv` KE ×0.975, **peak ×1.0000, monotone for the whole second**, and stable
at nx=64 / 400 steps (no `jitter = 0` late detonation).

**It is the position move, not the resample (`_PS_POSITION_SHIFT` /
`_PS_VELOCITY_RESAMPLE`).** `_PS_POSITION_SHIFT` alone is bit-identical to
position+resample; the Eq. 17 `_PS_VELOCITY_RESAMPLE` alone does not
regularise the distribution (non-monotone at 400 steps, ≈ no shift). And the
shift must be a *position* move: `_PS_SHIFT_AS_VELOCITY` (route `dx` to
`v += dx/dt`) NaNs — the exact mirror of `INSTEP_CD = False` routing the CD
solve to a position shift and breaking the column.

**The double-count.** PS shift *and* the in-step CD velocity
(`INSTEP_CD = True`) both solve the same `1 − ρ/ρ0`: `tgv` KE ×2.6, column
NaN. `dfsph.SOLVE_ORDER = 'cd_then_div'` (the divergence pass projects the CD
impulse's non-div-free part the *same* step, rather than next step) *contains*
it — `tgv` bounded, column NaN → `|v| ≈ 2` — but does not fix it.

**Ordering (`SOLVE_ORDER`).** `div_then_cd` (shipped) vs `cd_then_div`:
**~neutral for `tgv`** (×1.061 vs ×1.064 at nx=32/20 — so the injection is
*not* an ordering artifact), but `cd_then_div` degrades the column
(`|v|` 0.067 → 4.5): running the divergence pass *last* projects out the CD
impulse that holds the column against gravity. CD applied **last and
unprojected** is load-bearing. (omniSPH's own loop is div-then-density.)

**The bounded box detonates.** `randomFlowIncompressible --bounded` nx=64,
`dt = 1e-3` fixed: the shipped scheme blows up — KE 0.35 → 20.7 (**×60**),
`|v|` → 10.7, peak KE ×116. **PS-restore fixes the fixed-`dt` case** — KE
×0.994, peak ×1.0000, `|v|` ~ 1.0, density band 8.5e-3
(`scripts/make_video_dfsphBoundedRandom.py`). **But the graded run (adaptive
`dt`, `cflFactor = 0.4`) still NaNs even with PS-restore** — the CFL's lagged
`vMax` does not see the near-wall impulse coming; `randomFlowIncompressible
--bounded` at its shipped defaults is a **regression** the tmp-commit rewrite
introduced and the suite does not catch (§9 Parts 46–47 / the plan).

**Ablation — the one thing PS-restore cannot hold is a free surface**
(`scripts/probe_dfsphAblation.py`, nx=48 / 300 steps / fixed `dt`). PS-restore
holds:

| test | geometry added | PS-restore |
|---|---|---|
| `tgv` | periodic, decaying | ✓ KE ×0.98 |
| `randomFlowIncompressible --bounded` | domain wall band | ✓ KE ×0.99 |
| periodic + **static obstacle** | solid internal boundary | ✓ finite (density rough near the obstacle, band 0.36 vs 0.003) |
| `drivenSquare` | **moving** solid boundary | ✓ finite (band 0.24 near the body) |
| periodic + **oscillating gravity** (`dfsph.GRAVITY_OSC`) | uniform mean-zero body force | ✓ **bit-for-bit as good as baseline** (KE ×19.5 both, band *better*: 1.8e-3 vs 3.1e-3) |
| **`hydrostaticColumn`** | walls + **free surface** + gravity | ✗ **NaN** |

So it is **not** walls (bounded box holds — and is the case baseline
*detonates*), **not** the body force ("position shift cannot sustain a body
force", Part 23, does *not* generalise — oscillating gravity in a periodic box
is fine), **not** solid/moving boundaries. **Mechanism** (named in
`relaxLattice`'s own docstring): the PS shift calls `solveIncompressible`,
whose source `ρ0 − ρ*` with `ρ*` clamped at `0.9` drives the free-surface skin
(legitimately ~0.5) toward that floor every step → surface collapse → NaN.
**Free-surface fixes that DID NOT work.** `_PS_SURFACE_MASK` (zero the shift on
`surfaceIndicators == 1`, hard or `surfaceLambdas`-tapered) NaNs
`hydrostaticColumn` *faster* — it just moves the `rho*`-clamp compaction down
one row and opens a discontinuity in the shift field at the surface.
`_PS_SHIFT_MODE = 'delta'` (a surface-safe Fickian shift) as the *only* shift
(`INSTEP_CD = False`) resolves `tgv` / the bounded box but still blows up the
column at every strength — not from surface compaction this time but because
with `INSTEP_CD = False` there is **no hydrostatic support at all** (the
divergence projection stops further compression but never pushes back against
gravity; a position shift cannot sustain a body force). `'delta'` *with*
`INSTEP_CD = True` (no double-count — different mechanisms) holds the column
and gets `tgv`'s decay-rate test to pass at `iterations ≈ 4`, but leaves a
peak ×1.008 non-monotone bump (the in-step CD's own unprojected impulse) and
degrades the bounded box past `iterations ≈ 8`.

**The resolution (Part 47): gate by case.** `dfsph.INSTEP_CD = 'auto'` and
`incompressible._RESTORE_PS_SHIFT = 'auto'` (both now the default) key off
`schemeConfig.gravityConfig.active`:

- **body-force case** (`hydrostaticColumn`): in-step CD **on**, VD+PS shift
  **off** — byte-identical to the pre-Part-47 holding behaviour.
- **everything else** (`tgv`, `randomFlowIncompressible` bounded/periodic,
  `kolmogorovIncompressible`, `shearWave`, `staticBlob`, `impact`,
  `squarePatch`): in-step CD **off**, VD+PS position shift (`'cd'` mode) **on**.

No case runs both, so the double-count never arises. Measured:
`tgv` nx=32/20 KE ×0.9996 / peak ×1.0000 / monotone — **passes both
`test_tgvKineticEnergy*`** (rate/analytic 0.56, in the 0.6 ± 0.45 band);
nx=64/400 the same. `hydrostaticColumn` nx=64 unchanged (`|v|` 0.066, slope
1.0007, `densityP05` 0.947). `randomFlowIncompressible --bounded` KE ×0.994
(was the ×60 detonation). **Full `tests/test_physics.py` + `test_runner.py`:
82 passed / 0 failed** — green for the first time since the `_solve` rewrite.

**Not covered by the suite, and broken by the rewrite:**
`randomFlowIncompressible --bounded` at graded defaults NaNs (above);
`kolmogorovIncompressible` is inert — `dfsph_step` zeroes the forcing
(`computeForcing(...) * 0`, an undocumented tmp-commit change), so the forced
case does nothing (`KE ≡ 0`). Neither case is in `test_physics.py` /
`test_runner.py`. Both are open items for the plan.

**Secondary:** baseline holds the static obstacle but detonates the walled box,
so **baseline's own instability is the domain wall band specifically**, not
solid boundaries in general. And PS-restore roughens the density near a solid
obstacle (band 0.36 vs 0.003) — a `--scheme divergenceFree` obstacle case is
still not clean, just finite.

Knobs: `dfsph.{INSTEP_CD ('auto'), SOLVE_ORDER, GRAVITY_OSC}`,
`incompressible.{_RESTORE_PS_SHIFT ('auto'), _PS_SHIFT_MODE, _PS_POSITION_SHIFT,
_PS_VELOCITY_RESAMPLE, _PS_SHIFT_AS_VELOCITY, _PS_SURFACE_MASK}`.

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
| **Zero-meaning the source** (textbook compatibility projection) | **For `divergenceFree` (VD+PS, position shift): negative.** Makes the solver *converge* (nIter 64→13.4) and the density *worse* (`mean\|rho-1\|` 3.9e-3 vs 2.9e-3, `rhoStd` 3.4x). Only part of the source's mean is unreachable; the rest is the real de-clump signal for the shift. **For `omniIncompressible` (velocity impulse) on a *closed* box: correct (Part 42).** There the whole source mean is genuinely in `null(A)` (pure-Neumann, `A·1 ≈ 0`), the Jacobi cannot touch it, and a mean-zero pressure is the right gauge for a velocity impulse. See §1.7 Part 42 / `CD_SOURCE_PROJECT`. The two schemes differ because the correction is a shift vs an impulse and because the domain here has no free surface. |
| **Capping the implicit shift** (`--shiftCap`) | 0.25 spacings/step still diverges (t=5.28); 0.05 diverges *much earlier* (t=1.45). This is [C] §6's argument about user-tuned shift magnitudes, measured. |
| **deltaSPH shift stacked on the implicit shift** | 4% (noise) at default magnitude, then monotonically worse, then NaN at `shiftProperties.CFL=3.0`. Same [C] §6 argument from the other side. Note it was also *never running* — `finalize` shadowed `dx`; fixed, then rejected. |
| **Confining the velocity correction to a wall band** | Diverges at t=4.17 vs t=8.0 unscoped; widening the band is worse (t=2.74). The correction's value is not wall-local — it improves the *bulk* 2x — and a shell edge injects a velocity discontinuity. Mode implemented, measured, removed. |
| **Scaling the velocity correction down** | No sweet spot. Bounded case stable at λ=0.25; `tgv` already 1.4x analytic and non-monotone there. |
| **Moving the MLS projection earlier in the step** (`--mlsBeforeSolve`) | 232 penetrating vs 228. It changed the *lag*, not the *statefulness* — the real fix is moving it inside the Jacobi iteration (§4, item 5), which is untried. |
| **Removing `mdbcNoPenetrationShift`** | Strictly worse on 2 of 3 crossing metrics (543 vs 441 crossing; worst depth 16.2dx vs 10.4dx). Not a crutch worth removing. Re-confirmed post-`c637785` (Part 49, §1.6 table): OFF vs ON, `dambreak` wall crossing 5→0, `columnCollapse` 6→1, impact `\|v\|` spike down ~30%, quiescent column untouched. |
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
| **omniSPH's faithful `omega = 0.5`** (Part 35) | The `omniIncompressible` port with omniSPH's hardcoded relaxation detonates the density Jacobi by t ≈ 0.06 / step 7 (`|v|max → 1e17`) on `hydrostaticColumn` — the same window failure Part 29 measured. `OMEGA = 0.3` (the module constant) is inside the window: holds nx=32 for 2000 steps, but nx=128 still spikes `|v|max` to ~107. The two-solve-on-one-neighbourhood structure is not the lever. |
| **Matching SPlisHSPlasH's discretization** (`n_h = 2` → `h = 2·dx`, cubic spline; Part 36/38) | Diverges every warpSPH scheme, even in the trivial semi-periodic case where the column physics is a non-issue (`iisph` 3.6e5, `omniIncompressible` 1.7e7). warpSPH's operators / summation-density estimator are calibrated for `n_h ≳ 3`; at `h = 2·dx` the initial density estimate is min ~0.45. Cubic spline is worse than Wendland2 at every `n_h`. The `importSP_*` "volume loss" is this same pathology. |
| **XSPH damping to close the DFSPH-path late-time failure** (Part 37) | `dfsphReference` + `ki==0` factor at nx=128 diverges at step ~940 (the Parts 26/30/31 late-time free-surface degradation); `XSPH_FLUID = 0.05` is a **wash** (KE unchanged ~0.005, `embeddedMinDensity` marginally worse), and `XSPH 0.1/0.05` (fluid+wall) brings the blow-up *forward* to step ~380. The failure is not slosh-driven (KE ~0.005 right up to it) — damping the bulk does nothing and coupling more XSPH into the marginal divergence Jacobi destabilises it. Wall drag helps the *free-slip slosh* in schemes that are otherwise stable (§ 1.12), not the divergence-Jacobi instability. |
| **XSPH to decay the `iisph` nx=128 free-slip slosh** (Part 39) | Null-to-negative. `XSPH_BOUNDARY_EPSILON` 0.05–0.2 *raises* vmax (1.7 -> 2.2) and KE with the coefficient; `XSPH_FLUID_EPSILON` 0.02–0.1 pumps KE (0.036 -> 0.067 at 0.1) and drops `embeddedMinDensity`. The Part 38 nx=64 / `omniIncompressible` `XSPH_BOUNDARY = 0.15` win (slosh ~2/3) does **not** carry to `iisph` / nx=128. A shear-carrying no-slip wall (§1.14 follow-up) is the direction instead. |
| **Fluid-only physical viscosity `nu*lap(v)` to decay the slosh** (Part 39) | Works on the KE and pins the column, but `embeddedMinDensity` -> ~0.72 at every `nu` 0.01–0.05 (baseline 0.94): with the walls left free-slip it diffuses the interior while the wall sliding keeps working the surface. Put the stress at the wall, not (only) in the bulk. **Re-confirmed in the Part 42 cleanup** via the stock path: `wallBC=freeSlip` + `viscidNu=0.01` drives `embMin` to 0.43. |
| **`wallBC=noSlip` + `viscidNu` through the stock `computeVelocityDiffusion` term** (Part 42) | Bounds the slosh KE (4× down at `nu=0.01`) and holds the gradient (`slope` 1.02), but `embeddedMinDensity` 0.94 → 0.60 and `|v|max` spikes to ~3.7 — the `inviscid=False` branch is normal-projected + approach-only, so a no-slip mirror only adds noisy *normal* wall damping, no tangential stress. Does not reproduce Part 39's `_physViscosity` (which used a real vector Laplacian + a mirror clamp). `wallBC=extended` (MLS) + `nu` is outright unstable (KE 0.81). A shear-carrying Morris term is the fix (TODO, ranked queue). |
| **`iisph` (plain IISPH, CD-only) as a general incompressible scheme** (Part 42) | `tgv --scheme iisph` nx=64 *injects* energy: KE 9.86 → 1876 by t=0.1, `|v|max` 1.0 → 29, then a wrong bounded plateau (KE ~1200, `|v|max` ~25) for 300 steps — while density stays near-perfect (`rho` [0.995, 1.007], `rhoStd` 1.3e-3). `divergenceFree` holds `tgv` (KE ratio 0.996, monotone); `dfsphReference` (= `iisph` + the divergence-free pass) is bounded (KE ratio 0.81). So the density-invariance solve alone controls `rho` but not the velocity field: on a vortical flow the unconstrained pressure impulses spin the vortices up. `iisph` is viable only near-quiescent (`hydrostaticColumn`, `staticBlob`); a general scheme built on it must add the divergence solve back (ranked queue item 4 / item 9). `randomFlowIncompressible --bounded` confirms it (`|v|max` → 1.3e6 by step 40 vs `divergenceFree` 1.06); `kolmogorovIncompressible` (forced) is milder — `|v|max` 3.78 vs 2.51 at matched KE, a rougher field not a blow-up. |
| **`omniIncompressible` as a general incompressible scheme** (Part 42) | Holds the *periodic* cases (`tgv` KE ratio 0.79 / bounded — over-dissipative from the `XSPH_FLUID = 0.05` default; `kolmogorovIncompressible` KE → 1.38) — its 3-iter divergence pass keeps it bounded where `iisph` blows up. **Diverged on `randomFlowIncompressible --bounded`** with `WALL_PRESSURE_MODE = 'mls'` (KE → ~1e31 within a few steps): the constant-density Jacobi hits its 256-iter cap without converging on step 1 (`errRho` 3.4e-2, `max｜p｜` ~4e4, `max｜a_p｜` ~3e5, `｜v｜max` ~84) — the divergence solve is fine (`errDiv` ~1e-10). `'mls'`'s linear `β·x+γ·y` term assumes a locally-linear near-wall pressure (exact for the hydrostatic column, Part 41), wrong for a sheared flow. **Fixed (Part 42): `WALL_PRESSURE_MODE = 'shepard'`** (0th order, no linear term) threads both — `hydrostaticColumn` nx=128 holds (`｜v｜max` ~0.5, exact gradient) and `randomFlowIncompressible --bounded` holds (`｜v｜max` decays 2 → 0.4). It makes the run *survive*, not the CD Jacobi *converge* (still caps out — §1.7); a contractive CD solve / `band2018pb` is the deeper fix. Default changed `'mls'` → `'shepard'`; `'mls'` kept as an option. |
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
| `mdbcNoPenetrationShift` | `True` | Removing it is worse (§2). Commented out by the `c637785` rewrite, re-wired Part 49 (§1.6); `dfsph.NOPEN_SHIFT = 'config'` obeys this flag, `True`/`False` force it. |
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
- `probe_hydrostaticColumnIisph.py` *(new, Part 34)* — the validation-ladder
  run of `--scheme iisph` on `hydrostaticColumn` at any `nx`. Tabulates
  `|v|max` / KE (the slosh), the three density FOMs side by side
  (`minDensity` = spray, `densityP05` and `embeddedMinDensity` = the column
  body), `maxDensity`, `pressureSlopeRatio` / `pressureResidual`, and
  `dispMax` at ten evenly-spaced samples plus a last-quarter summary.
  `--steps 0` (default) runs time-limited to `tLimit`; `--scheme` also takes
  `dfsphReference` / `divergenceFree` for the A/B. This is what confirmed
  Part 33's nx=32 result at the default nx=128.
- `probe_hydrostaticColumnViscosity.py` — *removed in the Part 42 cleanup*
  along with the `_physViscosity` / `PHYS_VISCOSITY_*` bespoke path it
  drove. The viscosity A/B is now just `probe_hydrostaticColumnIisph.py`
  (or a plain `run(...)`) with `params=dict(wallBC=..., nu=...)`.
- **`scripts/splishsplash_compare/`** *(new, Parts 35–38)* — the SPlisHSPlasH
  cross-check kit. `pysplishsplash` imports in both conda envs after Part 36's
  install fixes (see the Part 36 §9 row). `README.md` documents the driver
  gotchas (`setTimeStepCB` corrupts the sim; post-`run()` access segfaults).
  - `hydrostatic_2d.json` — matched 2D scene (L=1 box, spacing 1/128,
    bottom-half fill, DFSPH inviscid, `density0=1`, Bender2019 volume-map
    walls). `run_export.py` runs it head-less with VTK export and parses the
    frames (the only reliable stepped-trace path); `run_splish.py` reads the
    initial state.
  - `warpsph_initial.py` dumps warpSPH's `hydrostaticColumn` (nx=128) initial
    state; `compare.py` prints the side-by-side + the SPlisHSPlasH `|v|` series.
  - `import_and_run.py` — the controlled test: overwrites warpSPH's fluid rows
    with SPlisHSPlasH's exact positions / mass / `h` (from `splish_fluid8001.npz`),
    zeroes velocity + IC pressure, sets the cubic kernel, runs the requested
    scheme. This is what showed the setup matches for ~0.3 s (§ 1.13).
  - `warpsph_matched.py` — sweeps `n_h` × kernel × `calibrateRestDensity` on
    the native case (the "match the setup" negatives, § 2).
  - `semiperiodic.py` — the x-periodic / floor-wall-only variant that isolates
    vertical stability (§ 1.12, Part 38). `one_noslip.py` sweeps the
    `XSPH_BOUNDARY` penalty no-slip.
  - `make_videos.py` — renders every scheme (native + imported-SP state +
    semi-periodic) to `videos/<tag>/output.mp4`. (`make_videos_viscosity.py`
    was removed in the Part 42 cleanup with the rest of the `_physViscosity`
    path.)
- **`scripts/omnisph_compare/`** *(new, Part 40)* — the omniSPH cross-check.
  `build_omnysph.sh` builds `~/dev/omniSPH/omnySPH`'s `_core` for the warp env
  (the repo's `.so` is py3.14-only) — recompiles `omnySPH/src/main.cpp` with
  system gcc 13 against the pre-built `~/dev/omniSPH/build/lib/*.a`. Exposes
  `SPHSimulation.timestep` + every substep + all `fluid*` buffers.
  `run_omnisph.py` + `column.yaml` run a matched hydrostatic column;
  `ablate_xsph.py` toggles omniSPH's `XSPH` (`ptcl.viscosityConstant`) and
  `BXSPH` wall no-slip (`ptcl.boundaryViscosity`). Showed omniSPH holds the
  column fully inviscid and the warp gap is the analytic wall boundary
  (§ 1.15). `operator_diff.py` *(new, Part 41)* — the ranked-queue-item-0
  operator-by-operator diff: drives omniSPH substep by substep (`muffle()`
  redirects its C++ stdout; buffers are readable, **not** writable), extracts
  its exact `column.yaml` rest state, evaluates warpSPH's composed operators
  on the same positions / `h` / kernel and diffs on the deep-bulk rows
  (> 2.5 h from every wall, where omniSPH's analytic triangle boundary is
  inert). Result: rest-state interior operators clean (density ~5e-6, source
  exact, `alpha` matches modulo the `ρ0` 1-vs-998 convention, `a_p` sound to
  the linear-field gradient error); the density solve holds the column in
  *neither* code on the pristine rest state (`p ≥ 0` clamp vs a slightly
  under-rest bulk). Throwaway.
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
| 34 | 08-31 | **Validation-ladder item 1: `iisph` on `hydrostaticColumn` confirmed at the default `nx = 128`, and the case's density FOMs made spray-robust.** Part 33 measured `iisph` only at nx=32; the run at the shipped `nx = 128` **holds** — clean to `tLimit` (438 steps) and on to t = 2.9 / 1500 steps, `diverged = False`, `maxDensity` ≤ ~1.06, `pressureSlopeRatio` late-run **~0.99** (the exact hydrostatic gradient, built unaided from the raw hydrostatic IC seed), column body `embeddedMinDensity` **0.92–0.96** throughout. The plain `minDensity` still swings **0.2–0.6** — the ballistic surface spray Part 33 diagnosed, unchanged. **Landed** two spray-robust FOMs in `hydrostaticColumn.py::hydrostaticDiagnostics` (additive, no config): `densityP05` (5th-percentile fluid density) and `embeddedMinDensity` (min over fluid rows > 1 dx below the 95th-percentile surface height — the ballistic skin removed outright), the two metrics `probe_dfsphReferenceColumnSurface.py` used inline in Part 33. Two caveats: (a) a **startup transient** at t ≈ 0.14–0.22 as the raw hydrostatic seed relaxes to the nx=128 discretization — `embeddedMinDensity` dips to ~0.59, `pressureResidual` spikes to ~1.0 — deeper than at nx=32, fully recovered by t ≈ 0.25; (b) the bounded free-slip slosh does **not** flat-plateau at nx=128 as it did at nx=32 (KE ~0.12) — KE creeps ~0.025 → 0.050 over 1500 steps, still bounded, no blow-up (Part 33's open "decaying it needs a viscosity/XSPH choice" stands). New tool `probe_hydrostaticColumnIisph.py`. Physics + Krylov + runner suites green (102 passed), `gradcheck_incompressible.py` green (no kernel touched). No `divergenceFree` default changed. |
| 35 | 08-31 | **The omniSPH incompressible solver loop ported as `IncompressibleSPHScheme.omniIncompressible = 3` (`schemes/omniIncompressible.py`) — a literal transcription of omniSPH's (`~/dev/omniSPH/simulation/{SPH,fluidMechanics}.cpp`) `SPHSimulation::timestep` → `divergenceSolve` / `densitySolve`:** summation density, gravity, an **exactly-3-iteration** divergence Jacobi, a **min-3 / max-256** constant-density Jacobi warm-started from `0.5·p_prior`, every pressure acceleration accumulated into one `a` consumed by a **single** semi-implicit Euler step. Both solves run on the SAME neighbourhood and positions — omniSPH never advances `x` between them (contrast `dfsphReference`'s SPlisHSPlasH order, which does). Composed from `computeAlpha` (the IISPH `a_ii` bracket, `includeBoundaryReaction=False`; `alpha = dt²·computeAlpha(...) ≤ 0` = omniSPH's `fluidAlpha`), a scatter/difference `WarpOperation.Divergence` (= `computeMomentumIncompressible` minus its `−rho0`), and `computePressureAccelIISPH`, masked to fluid rows; `kind==1` walls enter both solves via `applyConsistentCoupling` (`consistent`, `akinciBoundaryVolume=True`) — the particle-boundary analogue of omniSPH's triangle `boundaryFunc`. Reuses `DFSPHReferenceSystem` + the incompressible codecs. **No** free-surface gauge, deficiency guard, damped warm start, or masking. **Result: the port does not rescue `hydrostaticColumn`.** omniSPH's fixed `omega = 0.5` detonates the density Jacobi by t ≈ 0.06 / step 7 (`|v|max → 1e17`) — the exact failure Part 29 traced to `omega` lying outside this codebase's composed-operator Jacobi window (~[0.2, 0.35]); the module constant `OMEGA = 0.3` holds nx=32 for 2000 steps (bounded slosh, `embeddedMinDensity` ~0.96, `maxDensity` ~1.01) but **nx=128 spikes `|v|max` to ~107 then sustains ~12** (grossly unphysical, not Inf). warpSPH's `iisph` at n_h=4 holds nx=128 with `pressureSlopeRatio` ~1.07 — so the two-solve-on-one-neighbourhood structure is **not** the lever; the divergence Jacobi is (Parts 26/28/29/33). `scripts/splishsplash_compare/{run_splish.py,warpsph_initial.py}`. |
| 36 | 08-31 | **SPlisHSPlasH's Python bindings installed both envs, and the cross-library hydrostatic comparison run.** *Install:* base env — the editable install only registered the empty `pySPlisHSPlasH` source package; the real `pysplishsplash*.so` (repo root) was off `sys.path`. Fixed with `miniconda3/.../site-packages/SPlisHSPlasH-local.pth → ~/dev/SPlisHSPlasH`. warp env (py3.13) — no matching `.so`; reconfigured `~/dev/SPlisHSPlasH/build` with `-DPython_EXECUTABLE=<warp python>`, built the `pysplishsplash` target (link needs `LIBRARY_PATH=$CONDA_PREFIX/lib` so `-ldbus-1` resolves), copied `build/lib/pysplishsplash.cpython-313-*.so` into warp site-packages. Driver gotchas recorded: `base.setTimeStepCB()` corrupts the sim in this checkout (shipped `DamBreakModel_2D` explodes at step 1 with a callback attached, runs clean without), and any `Simulation.getCurrent()` / field-buffer access **after** `base.run()` segfaults — so the working path is `init` → `setValueFloat(STOP_AT)` → `base.run()` with `enableVTKExport`, then parse the VTK frames (`run_export.py`). **Does SPlisHSPlasH hold the hydrostatic column? Yes.** Matched 2D scene (L=1 box, spacing 1/128, bottom-half fill, DFSPH, inviscid, `density0=1`, Bender2019 volume-map walls): a mild startup slump peaking `|v|max ≈ 2.4` at t ≈ 0.11, then **monotonic decay** (0.66 by t=0.30, `|v|mean` 0.59 → 0.10), **100 % of particles stay in the box** every frame, surface breathes ~8 dx and settles. **Initial-state diff (both ~8001 fluid, same spacing 1/128, same bottom-half extent):** warpSPH support `h = n_h·dx = 4·dx`; SPlisHSPlasH `h = 4·particleRadius = 2·dx` — **warpSPH's smoothing length is 2× SPlisHSPlasH's** at the same spacing (~4× the neighbour count in 2D; warpSPH mid-column = 45). Kernel Wendland C2 vs cubic spline. Particle mass `1.11·rho0·dx²` (warpSPH) vs `0.80·rho0·dx²` (SPlisHSPlasH's 2D volume factor). Net: warpSPH's initial summation density is mean **0.98**, min **0.64**; SPlisHSPlasH's 0.8 factor + tight kernel is tuned so a regular lattice ≈ rho0. **"Match the setup" negatives** (`warpsph_matched.py`, nx=128): setting `n_h = 2` (→ h = 2·dx) diverges every scheme (`iisph` 3.9e4, `omniIncompressible` 8.9e7); cubic spline is worse than Wendland2 at every `n_h`; `n_h < 3` diverges. **The controlled test — import SPlisHSPlasH's EXACT 8001 fluid positions + its mass + `h = 0.015625` + cubic kernel + zero velocity + zero IC pressure into warpSPH's `hydrostaticColumn`** (`import_and_run.py`): all three schemes start at SPlisHSPlasH's exact frame-1 numbers (`|v|max 0.0098`), and `iisph` / `omniIncompressible` **reproduce SPlisHSPlasH's slump transient** (peak `|v|max ≈ 2.4` at t ≈ 0.14) for the first ~0.3 s / ~150 steps — `omniIncompressible` tracks it *better* than `iisph` early (`embeddedMinDensity` 0.98, `pressureSlopeRatio` 1.0). **Then warpSPH's time evolution loses it:** `dfsphReference` (the DFSPH match) detonates by step ~50 (`|v|→10–23`), `omniIncompressible`'s pressure Jacobi detonates at t ≈ 0.4 (KE → 3.3, `|v|→18`), `iisph` doesn't detonate but the free surface slowly thins (`embeddedMinDensity` 0.9 → 0.38 over 500 steps) where SPlisHSPlasH holds ~0.56. **So the initialization + physics match — for ~0.3 s.** What fails is warpSPH's *evolution*: the operator-composed divergence-free Jacobi is not contractive at the matched (cubic, h = 2·dx) discretization where SPlisHSPlasH's dedicated `TimeStepDFSPH.cpp` at `omega = 0.5` is, and warpSPH's free-surface handling can't hold the surface. Unmatched piece: the wall (warpSPH 5-layer particle band vs SPlisHSPlasH volume-map; no importable boundary particles). `scripts/splishsplash_compare/` (`hydrostatic_2d.json`, `run_export.py`, `compare.py`, `import_and_run.py`; `splish_fluid8001.npz`, `warpsph_initial.npz`). |
| 37 | 08-31 | **`wp_dfsph_factor.py` back-reaction gate changed `kj == 0` → `ki == 0` (user's edit), and an XSPH dissipation term added to `omniIncompressible`; neither closes `hydrostaticColumn` nx=128.** *Factor:* the sum-of-squares (back-reaction) term of `computeDFSPHFactor` is now gated on the **query** kind (`ki`, constant across the neighbour loop) rather than the neighbour kind. Effect: a fluid query accumulates it over **all** non-ghost neighbours (fluid + boundary); a boundary/ghost query gets 0. This is a deliberate departure from SPlisHSPlasH `computeDFSPHFactor` / Bender-Westhofen-Jeske Eq. 32, which keep it fluid-neighbours-only (a truly static wall takes no reaction) — folding the wall in enlarges the near-wall denominator → smaller `alpha` → gentler near-wall pressure updates (same intent as `akinciBoundaryVolumeScale = 2.0`, Parts 24/25); bulk-identical to the reference. Neighbour iteration itself verified correct (Verlet list, `checkDirectionality_j(kind, 9)` drops ghosts only). Docstring + inline comments updated. *Dissipation:* omniSPH's `XSPH` + `BXSPH` ported into `omniIncompressible` as a post-solve velocity filter on the start-of-step velocity — `v_i += Σ_j c_j V_j W_ij (v_j − v_i)` with per-kind `XSPH_FLUID` / `XSPH_BOUNDARY` (wall enters at `v_j = 0` → drag), one `Interpolate` pair, masked to fluid; defaults `0.05` / `0.0` (0/0 = the faithful no-dissipation loop). *A/B:* `dfsphReference` + `ki==0` at nx=128 holds ~400 steps then **diverges at step ~940** — the *same* late-time free-surface degradation (`embeddedMinDensity` → 0.74, then `|v|→inf`) that survived Parts 26/30/31; adding `XSPH_FLUID = 0.05` is a **wash** (KE unchanged ~0.005, `embeddedMinDensity` marginally *worse*), and `XSPH 0.1/0.05` (fluid+wall) brings the blow-up *forward* to step ~380. The failure isn't slosh-driven (KE ~0.005 right up to it), so damping the bulk does nothing and feeding more XSPH into the marginal Jacobi destabilises it. `iisph` + `ki==0` holds nx=128 to 1000 steps (`embeddedMinDensity` 0.94, slope 1.0) with a persistent ~0.04 slosh KE that slowly creeps up. Verdict: the divergence-free Jacobi remains the thing that breaks the DFSPH path at real resolution; the factor tweak delays it, dissipation doesn't touch it. `scripts/splishsplash_compare/one_noslip.py`. |
| 38 | 08-31 | **The semi-periodic isolation test — with warpSPH's native discretization the vertical density-solve physics is essentially perfect; the entire `hydrostaticColumn` failure is the free-slip side walls.** `scripts/splishsplash_compare/semiperiodic.py` builds a `hydrostaticColumn` variant with x **periodic** (no side walls → zero tangential fluid↔boundary interaction) and a floor wall band in y, free surface on top — a `DomainDescription` with `periodic = [True, False]` and per-axis extents (x width L, y widened by the band), interior ceiling at +∞ so `domainBoundarySdf` cuts a floor band only. **Result (nx=128, 500 steps, Wendland2 / n_h=4):** `omniIncompressible` — `|v|max` **0.22**, KE **1e-4**, `embeddedMinDensity` **0.99**, `pressureSlopeRatio` **1.00** — the column sits at rest density with the exact hydrostatic gradient, no slump, no splash, no volume loss. `iisph` — `|v|max` settles 0.61, KE 0.011, `embeddedMinDensity` 0.96, slope 0.99 — also holds, slightly noisier. So the constant-density solve's **vertical** stability is sound; everything wrong in the fully-walled `omniIncompressible` video (the slosh, the ~30 % column drop, the splashing) is the **free-slip side walls** (particles slide down them → floor deflection → the Part 32 counter-rotating vortex pair → surface degradation). **Second finding: cubic-spline / `h = 2·dx` is unusable in warpSPH** — it blows up *even here* in the trivial semi-periodic case (`omniIncompressible` → 1.7e7, `iisph` → 3.6e5), independent of the column physics. So the `importSP_*` volume loss (Part 36) is a miscalibrated summation-density estimate at h = 2·dx (initial `embeddedMinDensity` ~0.45), not a hydrostatic-solve failure — warpSPH wants `n_h ≳ 3`; chasing SPlisHSPlasH's h = 2·dx is the wrong move. **Penalty no-slip** (walled `omniIncompressible`, nx=64, 400 steps): `XSPH_BOUNDARY = 0.15` (near-wall fluid velocity dragged toward the static wall's v = 0) cuts the free-slip slosh energy ~2/3 (late KE 0.0051 → 0.0018) and reduces the slump (`dispMax` 0.51 → 0.43), column body + gradient unchanged; `0.4` over-fights the pressure solve. A proper viscous no-slip (full, not normal-projected, fluid↔wall diffusion) is the principled version — `BCType.noSlip` exists but only on the mDBC/ghost path (`schemes/dfsph.py`), not the `applyConsistentCoupling` static-wall path these schemes use. Videos: `scripts/splishsplash_compare/videos/{native_*,importSP_*,semiPeriodic_*}` (per-scheme mp4/gif, via `make_videos.py` → `run(plot=True, video=True, plotBackend='matplotlib')`). |
| 39 | 08-31 | **The viscosity path — the WCSPH / Adami 2012 no-slip wall decays the `iisph` free-slip slosh with the surface and gradient intact; XSPH is null-to-negative and fluid-only viscosity roughens the surface (§1.14).** Three default-inert module toggles on the shared `dfsphReference` step body (`_physViscosity`, applied as a non-pressure acceleration in step 2 alongside gravity / the deltaSPH term / XSPH; `iisph_step` picks them up), all composed from the existing `WarpOperation.Laplacian` (Brookshaw) + `Interpolate` operators — **no new kernel**, `gradcheck_incompressible.py` + physics + Krylov suites green (one pre-existing Krylov flake, §4). Graded on `hydrostaticColumn` / `--scheme iisph` / nx=128, `scripts/probe_hydrostaticColumnViscosity.py` (`--visc {none,xsphF,xsphW,xsphFW,physF,physW,physFW}`, `--sweep`, `--trace`; flips the `ref.*` module globals like the Part 32 XSPH probes). **`PHYS_VISCOSITY_WALL_NU` (`physW`) — the fix.** Each `kind==1` particle's velocity is temporarily set to the Adami mirror `v_b = 2·v_wall − shepard_f(v)` (static wall → `−shepard_f(v)`; `shepard_f(v)_b = Σ_f (m_f/ρ_f) v_f W_bf / Σ_f (m_f/ρ_f) W_bf` via a `FluidToBoundary` `Interpolate` pair), a `BoundaryToFluid` Brookshaw Laplacian `nu·∇²v` is taken over the fluid↔wall pairs, and the boundary rows are restored — never leaks into the solves. Mirror speed capped at `PHYS_VISCOSITY_WALL_VCLAMP = 4×` the near-wall fluid speed. `nu ≈ 0.01–0.02` (1500 steps, `diverged=False`): vmax 1.7 -> ~0.7–1.0, slosh KE from the ~0.04 plateau *creeping up* to **~0.010–0.013 and flat**, `embeddedMinDensity` **0.94–0.97** (baseline 0.94), `pressureSlopeRatio` ~1.0, `dispMax` drift ~halved per unit time; `nu = 0.01` has the gentlest startup, `nu = 0.02` a rougher startup transient (`embeddedMinDensity` dips ~0.68 at step ~167) then a tighter steady state; `nu = 0.1` over-drives (vmax -> 4, KE pumps). **`PHYS_VISCOSITY_NU` (`physF`) — fluid↔fluid `nu·∇²v` (`FluidToFluid`), the "normal viscosity path, no boundary velocity".** Decays bulk KE and pins the column (`dispMax` end 0.15–0.19 at `nu ≈ 0.03`) **but `embeddedMinDensity` → ~0.72 at every `nu` 0.01–0.05** — diffuses the interior while the free-slip walls keep feeding the surface; `slope` also drifts (0.93–1.13) at low `nu`. `physFW` (both) pins hardest, keeps the fluid term's ~0.82 surface. **XSPH (`XSPH_*_EPSILON`, Part 32, already present) — null-to-negative here:** wall drag 0.05–0.2 raises vmax 1.7 -> 2.2 and KE with the coefficient; fluid XSPH 0.02–0.1 pumps KE (0.036 -> 0.067) and degrades `embeddedMinDensity`. Confirms Part 33 (n=3) / Part 37 (`omniIncompressible` wash); Part 38's nx=64 `omniIncompressible` `XSPH_BOUNDARY = 0.15` win does **not** carry to `iisph` / nx=128. Verdict: the wall is the right place for the stress (§1.14) — matches Part 38's prediction. Not promoted to a config default; open items in `DFSPH_IMPROVEMENT_PLAN.md` ranked queue item 0. New tools `probe_hydrostaticColumnViscosity.py` and `scripts/splishsplash_compare/make_videos_viscosity.py` (nx=128 / 700-step mp4+gif for `visc_{none,physW_0.01,physW_0.02,physF_0.03,physFW}` in `videos/`: `none` shows the undamped bulk dipole, `physW_0.01` the calmed column, `physF_0.03` the pinned-but-rougher surface). No `divergenceFree` default changed. **[Superseded by the Part 42 cleanup: the `_physViscosity` + `PHYS_VISCOSITY_*` bespoke path was removed; the Adami mirror is `computeBoundaryVelocities`/`BCType.noSlip` and the physical viscosity is the stock `viscidNu` term, but that term is normal-projected so `noSlip`+`nu` bounds the slosh only noisily — a clean result needs a shear-carrying Morris term, now a TODO. §1.14 follow-up.]** |
| 41 | 09-01 | **Ranked-queue item 0, first pass: the rest-state interior operators are clean — no warpSPH-vs-omniSPH discrepancy at rest.** `scripts/omnisph_compare/operator_diff.py` drives omniSPH substep by substep through its bindings (buffers read-only, so the diff runs one direction: omniSPH on its native lattice, warpSPH's composed operators on the same positions / `h` / kernel) and diffs on deep-bulk rows (> 2.5 h from every wall → omniSPH's analytic triangle boundary inert). Kernels verified identical first: both Wendland2, `C_d = 7/π`, `W = C_d/h²·(1−q)⁴(1+4q)`, `q = r/h`, cutoff `q>1`, and `gradW` sign/scale (`−20q(1−q)³·C_d/h³`, from i toward j). Findings: (1) **density** — warp `computeDensities` == omniSPH `density()` to `~5e-6`. (2) **alpha** — same bracket, but warp is `998×` omniSPH's: omniSPH carries `fluidRestDensity = emitterDensity = 998` (`SPH.h`) and `fluidArea = πr²` (geometric, not `m/ρ0`), warpSPH's `hydrostaticColumn` runs `ρ0 = 1`. `alpha`, `a_p` and the solved `p` all scale with `1/ρ0`, and `a_p = −∇p/ρ` is `ρ0`-invariant → a **unit convention, not a bug** (recomputing omniSPH's bracket from its own buffers + neighbour list gives the warp value; the ratio to omniSPH's reported `fluidAlpha` is exactly 998). (3) **constant-density source** `(1−ρ/ρ0) + dt·div(v*)` == omniSPH `computeSourceTerm(true)` **exactly** on the bulk (`v* = v + dt·g` is uniform → divergence 0 in both). (4) **`a_p` operator** — `computePressureAccelIISPH` on the analytic hydrostatic `p = ρ0 g (H−y)` returns bulk `a_p_y ≈ +9.37` (target `9.81`; the ~4.5 % is the SPH gradient of a linear field near a cut surface), `a_p_x ≈ 0` — sound. (5) **the density solve holds the column in *neither* code on the pristine rest state** — bulk `ρ/ρ0 ≈ 0.999` (slightly *under* rest, same uncalibrated `n_h`-lattice deficit warpSPH has, §1.1) → source `> 0` → the `p ≥ 0` clamp zeros the bulk pressure. warpSPH's density Jacobi sits at `<p>bulk = 0`, `resid = −9.2e-4` flat for 256 iters at `omega` 0.3 **and** 0.5; omniSPH's `densitySolve` exits in 4 iters with bulk `a_p_y ≈ 0.04`, `dp/dy ≈ 0`. The hydrostatic gradient is a **transient build-up** (compression from the floor up over hundreds of steps — Parts 33/34), not a rest-state fixed point, in omniSPH too. **Conclusion: no interior-operator bug at rest; the `OMEGA` 0.3-vs-0.5 gap (§ deviations table) is not visible in the rest-state operators.** *Transient A/B* (`scripts/omnisph_compare/transient.py`, `omniIncompressible._solve`/`_step` instrumented for per-step/per-solve `nIter`/`err`/`max｜p｜`/`max｜a_p｜`/`｜v｜max` + the worst particle's location; three arms — warp walled nx=128, warp semi-periodic nx=128 (Part 38's x-wrap floor-only, isolates the interior), omniSPH): **warpSPH's composed constant-density Jacobi does not converge on this case.** omniSPH: `nDiv = nRho = 4` every step, `errDiv`/`errRho` ~1e-4–1e-3, `｜v｜max` 0.15–0.72 for 120 steps, rock-solid. warpSPH **walled**: `nRho` 3 → 21 → **256 (the cap)** by step 4, `errRho` **rising** through +7.8e-4 → +2.1e-3 (above tol), then a cascade **at the bottom corners** (`｜v｜max`/`max｜p｜`/`max｜a_p｜` all at the FLOOR band, `x ≈ ±0.5`; particles ~2.5 dx *below* `y = −0.5`) — `｜v｜max` 1.6 → 9 → 47 → Inf by step ~10 (or bounds badly at `｜v｜max ~21` — GPU-nondeterministic, cf. Parts 29/33). warpSPH **semi-periodic**: `nRho` **= 256 (cap) on essentially every step**, `errRho` oscillates 1.0–2.1e-3, **never converges** — *yet the column holds* (`｜v｜max` 0.15–0.38, `rmin` recovers 0.64 → 0.99, matches Part 38). So: **(a)** the Jacobi's non-contraction is a real solver/operator property, present with the side walls *removed* — the interior finding the user asked for — and the residual is *positive* (`aP > source`, an overshoot at the floor band, not a deficit); **(b)** non-convergence alone is survivable (semi-periodic), it turns fatal only when the **bottom corners** amplify the un-converged floor field (walled) — the boundary-model gap of Part 40, stacked on top. Also: `nDiv` is hard-fixed at 3 and warpSPH's 3-iter divergence residual (~5e-3) is ~20× omniSPH's 3-iter one (~2e-4). **Why the composed Jacobi does not contract at the floor band — read from the two loops, not swept.** omniSPH's `densitySolve` inner iterate is `computeBoundaryPressure(true)` → `computeAcceleration(true)` → `updatePressure(true)`, and `computeBoundaryPressure` (i) zeroes `fluidPredAccel`, (ii) adds a **wall-pressure acceleration** `−ρ0(p_i/(ρ_iρ0)² + p_b/ρ0²)·gk` where `p_b = calculateBoundaryPressureMLS(i, …)` is **re-extrapolated from the *current* `fluidPressure2` every iterate** (MLS: Shepard mean of neighbour `p` + a linear `β·x+γ·y` term). Then `computeAcceleration` adds the fluid–fluid `a_p` on top, and `updatePressure`'s Laplacian probe carries its own wall term `kernelSum += dt²·fluidPredAccel_i·gk` (density mode). So omniSPH's near-wall row has the wall in **all three** of `α` (`computeAlpha`'s density-mode `kernelSum1 += gk`), the operator `A·p` (`updatePressure`'s `gk` term), and `a_p` (the `p_b` gradient) — a full Robin closure, recomputed per iterate. **warpSPH's `omniIncompressible._solve` has the wall in `α` only** (`computeAlpha`'s first sum runs over the Akinci band; `includeBoundaryReaction=False`), while `computePressureAccelIISPH` is BWJ23 Eq. 33 — **no boundary-pressure term, boundary `p ≡ 0`** — and `_divergence(a_p)` sees `a_p = 0` on the (masked) boundary rows with no analytic `a_i·gk` self-term. `applyConsistentCoupling` wraps the whole solve **once** and only sets boundary `ρ_k ← ρ0` / Akinci `ψ_k`. So warpSPH's near-wall iteration matrix is **inconsistent** — `D` (via `α`) carries a wall term that the operator `A·p` and `a_p` do not — which is the classic recipe for a non-contractive Jacobi (cf. §1.8 "the operator wants to be the same on both sides"; §2's `diagonalOnly` "runs the wall's Jacobi step 1.6× too large and NaNs"). It manifests as the *positive* `errRho` overshoot at the floor band. **This converges item 0's "interior first" onto the boundary closure:** the bulk fluid–fluid operators are correct (rest-state diff above) and contract fine where there is no wall (the periodic / bounded cases, §1.7); the column's non-contraction is the missing per-iterate wall-pressure closure — exactly what `band2018pb` (boundary samples as solve unknowns in the same Jacobi loop, ranked-queue item 0 boundary sub-item) and ranked-queue item 5 ("move `computeMdbcPressure` inside the solver iteration") prescribe. **Fix hypothesis tested and CONFIRMED.** Added `WALL_PRESSURE_MIRROR` to `schemes/omniIncompressible.py` (module flag, default `False`): in the density-mode `_solve` loop, each iterate recompute a **zero-order (Shepard) wall pressure** `p_b[k] = Σ_f V_f p_f W_kf / Σ_f V_f W_kf` (clamped ≥ 0, only where the fluid weight is real) on the `kind == 1` rows via a `FluidToBoundary` `Interpolate`, and pass `p_all` (fluid `p`, boundary `p_b`) into `computePressureAccelIISPH` — so the symmetric gradient picks up the `−Σ_k m_k(p_i/ρ_i² + p_b/ρ0²)∇W_ik` wall term (omniSPH's, minus the MLS linear `β·x+γ·y` correction); the divergence still sees `a_p ≡ 0` on the masked boundary rows, which reproduces omniSPH's `a_i·gk` self-term. `_wallPressureMirror` composes only the existing `warpOperation(Interpolate)` — no new kernel. **Result on `hydrostaticColumn` nx=128** (`transient.py` A/B + a 400-step run): **mirror OFF** — 256-iter cap, `errRho` rising, blow-up at the bottom corners by step ~6–10 (`|v|max` → 1e5 / Inf, or bounds badly at ~35). **Mirror ON** — runs **400 steps / t = 2.48 clean, `diverged = False`**, `|v|max` peak 0.70 settling ~0.55 (the undamped free-slip bulk slosh, Parts 38/39), `embeddedMinDensity` **0.99**, `densityP05` **1.000**, `maxDensity` 1.009, `pressureSlopeRatio` **0.99–1.00** (the exact hydrostatic gradient), `pressureResidual` ~0.065; the Jacobi **starts to contract** — `nRho` 256 → ~180, `errRho` 2.3e-3 → 5e-4 (was pinned above tol). Semi-periodic + mirror: `|v|max` ~0.05, `rmin` → 0.998 (true rest). This matches / slightly beats `iisph`'s Part 34 hold of the same case, and `omniIncompressible` was previously *diverging* here (Part 35). **Mechanism proven: the non-contraction is the missing per-iterate wall-pressure closure.** Then, on the user's steer, replaced the hand-rolled Shepard with the **existing `modules/liu` MLS** — `WALL_PRESSURE_MODE ∈ {None, 'shepard', 'mls'}`; `'mls'` calls `interpolateLiuLiu` (the same first-order value+gradient fit `computeMdbcPressure` uses: evaluate at each `kind == 2` ghost point, Taylor-correct to the owning `kind == 1` row, Shepard fallback), i.e. omniSPH's `p_b = α + β·x_b + γ·y_b` in full, no new math. A/B on `hydrostaticColumn` nx=128 (`transient.py`, 100 steps + 400-step confirm): both `'shepard'` and `'mls'` **hold 400 steps / t ≈ 2.3–2.5 clean**, `pressureSlopeRatio` 0.99–1.00, `densityP05` 1.000, `|v|max` ~0.56 (the undamped slosh), Jacobi contracting `nRho` 256 → ~170–200; `'mls'` recovers `embeddedMinDensity` closer to rest mid-run (~0.96 vs `'shepard'` ~0.92 at step 60–100) — the linear `β·x+γ·y` term captures `∂p/∂n` at the wall so the near-wall pressure is more accurate. `'mls'` **is** the `band2018pb`-lite: the wall pressure is extrapolated, not solved as an unknown, and there is no relaxation/lag (unlike `computeMdbcPressure`, which carries it as under-relaxed state across steps — the §3.1 / `mdbcMlsPressure` instability). The residual gap to omniSPH's 4-iter convergence is now just the composed-operator conditioning at `n_h = 4` (the `omega` window, § deviations table) + `p_b` also entering `α`. **`omniIncompressible.WALL_PRESSURE_MODE = 'mls'` landed as the default** (user's call — the scheme was *diverging* at its own default nx=128 without it, Part 35). The extrapolation is factored into **`modules/incompressible/wallPressure.py`** (`wallPressureExtrapolation`, `mode ∈ {'shepard','mls'}`, no relaxation / no carried state) and shared by `omniIncompressible` and `dfsphReference` (so `iisph` too). Full suite (102) + Krylov + runner + `gradcheck_incompressible` green. Two follow-ups: **(a)** `omega = 0.5` (omniSPH's) still detonates `hydrostaticColumn` nx=128 even *with* the mls wall pressure — so the `OMEGA = 0.3` window is a **separate, pre-existing** bulk-operator conditioning issue (`n_h = 4` → ~50 nbrs → `ρ(D⁻¹A) ≈ 5.6`, § deviations table), not the wall, and it is what leaves the residual `nRho ~180` (vs omniSPH's 4). **(b)** On the shared `dfsphReference` body: `WALL_PRESSURE_MODE` (default `None`) + `WALL_PRESSURE_ON_DIVERGENCE` (default `False`) — wiring the wall pressure into the DF (divergence) Jacobi **detonates** `hydrostaticColumn` nx=128 at step ~110 (`|v| → 1e14`); omniSPH's `divergenceSolve` has no wall pressure (`computeBoundaryPressure(false)` just zeroes and `continue`s), so this is faithful and a clean negative — wall pressure is a **density-solve-only** closure. **`iisph` / `dfsphReference`-CD A/B** (`wallp_ab.py`, nx=128, 1300 steps ≈ t 2.5): the wall pressure is **decisive only for `omniIncompressible`**. `iisph` (CD-only, already holds nx=128 — Part 34): `'mls'` cuts the end slosh (`|v|end` 1.7 → 0.9) but *worsens* the startup (`|v|pk` 4.3 → 8.4), KE (0.022 → 0.032), `embeddedMinDensity` (0.65 → 0.58) and `pressureResidual` (0.41 → 0.57) — **wash-to-negative**, stays `None`. `dfsphReference` (two-solve): **both arms held 1300 steps this run** (n=1; Part 37's ~step-940 divergence is GPU-stochastic, ~1/3 survives), `'mls'` lower `|v|end` (0.32 → 0.17) / KE, slightly lower `embeddedMinDensity` (0.95 → 0.91), `pressureSlopeRatio` ~0.70 either way (the two-solve path's separate gradient-underbuild) — **inconclusive at n=1**, and not worth a 3-run batch now (item 9 "stays last"). Both flags ship `None`; `omniIncompressible` keeps `'mls'`. |
| 40 | 08-31 | **omniSPH's Python bindings built for the warp env; omniSPH holds the matched hydrostatic column — the warp port's real gap is the analytic wall boundary, not the viscosity (§1.15).** User's push-back: omniSPH (compiled, working) runs this case cleanly, the port "is barely able to not diverge to inf", there are fundamental issues, and omniSPH has Python bindings. *Build:* `~/dev/omniSPH/omnySPH` ships a `_core.*.so` for Python 3.14 only (empty `dir()` / `__file__ None` when imported in the py3.13 warp env). `scripts/omnisph_compare/build_omnysph.sh` recompiles just `omnySPH/src/main.cpp` with the system gcc 13 (matching the pre-built `~/dev/omniSPH/build/lib/lib{simulation,tools,imgui}.a`) and links `--start-group` with `libyaml-cpp.a` / `libboost_atomic.a` / the GL static libs; imports in the warp env, exposes `SPHSimulation.timestep` + every substep (`computeAlpha`, `computeSourceTerm`, `computeAcceleration`, `updatePressure`, `divergenceSolve`, `densitySolve`, `predictVelocity`, `XSPH`, `BXSPH`, …) + all `fluid*` buffers (pybind11/Eigen → numpy). *Result (`run_omnisph.py` + `column.yaml`, a 1×1 box / bottom-half fill / DFSPH-incompressible column at omniSPH's native `n_h ≈ 1.8`, 7020 ptcls):* omniSPH's **shipped** loop (DFSPH + `XSPH` `viscosityConstant = 0.01` + `BXSPH` `boundaryViscosity = 0.50`, both run every `timestep()` — the shipped loop is *not* inviscid) **decays** the startup slump — `vmax` 0.6 → 0.05, `KE/n` 3e-3 → 8e-5, `ρmax` pinned at 1.002, surface flat to ±0.003, solves converge 4 / ~11 iters. **Fully inviscid** (`ablate_xsph.py` sets both to 0): still **bounded and near-rest** — `vmax ~0.45`, `ρmax 1.002`, `KE/n` ~flat 3e-3, no divergence — where warp's `omniIncompressible` / `iisph` at `n_h = 4` slosh at `vmax ~1.7` and creep up, or need `OMEGA = 0.3` not omniSPH's 0.5. Ablation (`vmax` last quarter / KE-per-particle q1→q4, 1000 steps): inviscid 0.45 / 3.9e-3→3.0e-3 (flat); `bv=0.01` 0.29 / 2.8e-3→8.0e-4; `bv=0.05` 0.15 / →3.9e-4; `bv=0.5` (default) 0.10 / →2.2e-4 — a *light* wall no-slip already decays it, monotone in `bv`, all `ρmax ~1.002` bounded. `XSPH` off (bv=0.5) → 0.17. *The gap:* omniSPH walls are **analytic solid triangles** — `density()` adds the boundary kernel integral `k` (near-wall `ρ ≈ ρ0` at rest → constant-density source ≈ 0 at the wall), `computeAlpha` / `computeSourceTerm` / `computeAcceleration` / `updatePressure` each add the analytic wall gradient `gk` via `boundaryFunc`, and `computeBoundaryPressure` (MLS extrapolation of fluid pressure onto the wall) runs **inside every Jacobi iteration**. The warp incompressible family (`omniIncompressible` / `iisph` / `dfsphReference`) has none of this — a 5-layer Akinci **particle** band wrapped once by `applyConsistentCoupling`, near-floor `ρ ~ 0.79` uncalibrated → a standing spurious near-wall drive. So Part 39's `PHYS_VISCOSITY_WALL_NU` (= omniSPH's `BXSPH`, which the port had zeroed via `XSPH_BOUNDARY = 0.0`) is real but patches the smaller half. Partly walks back §1.12's "*entirely* the free-slip side walls". **Direction (user's call):** do **not** port the analytic triangle boundary — the interior fluid physics come first (compare the warp operators against omniSPH's substep by substep with the bindings; the boundary, later, via **`band2018pb`** — Band et al. 2018's extended PPE with boundary samples as solve unknowns in the same Jacobi loop, not triangles, not Akinci volumes, not the `band2018` MLS extrapolation). omniSPH is also stable at `boundaryViscosity = 0.01`, not only `0.5`. `scripts/omnisph_compare/` (`build_omnysph.sh`, `run_omnisph.py`, `column.yaml`, `ablate_xsph.py`) — throwaway comparison tooling. |
| 42 | 09-01 | **Part 39's bespoke viscosity path removed and folded into the stock machinery; `iisph` shown to fail `tgv` (it is CD-only, not a general scheme).** *Viscosity cleanup (user's steer — "there is an existing physical viscosity in the modules, why is this not being used?" / "the wcsph scheme extends the velocity field by mirror or MLS for free/no-slip").* Part 39's `_physViscosity` helper + `PHYS_VISCOSITY_NU` / `PHYS_VISCOSITY_WALL_NU` / `PHYS_VISCOSITY_WALL_VCLAMP` module globals on `schemes/dfsphReference.py` hand-rolled two things that already exist: the Adami no-slip *mirror* = `computeBoundaryVelocities` with `BCType.noSlip` (`modules/mdbc/velocity.py`, the call `schemes/deltaSPH.py` / `schemes/dfsph.py` already make); and the physical viscosity itself = the stock `computeVelocityDiffusion` / `schemeConfig.diffusionParams.viscidNu` term **already in `dfsphReference_step` step 2**. **Removed** all three globals + `_physViscosity` + `scripts/probe_hydrostaticColumnViscosity.py` + `scripts/splishsplash_compare/make_videos_viscosity.py`; **added** a `computeBoundaryVelocities` call to `dfsphReference_step` step 2 (before `computeVelocityDiffusion`, so the viscous term sees the extended wall velocities; a strict no-op with no `kind == 1` particles → periodic cases unaffected) and a `wallBC` param to `hydrostaticColumn` (`BCType` name, default `freeSlip` = the historical no-op). *Re-grade* (`iisph`, nx=128, 1200 steps, tail = last quarter): `wallBC=constant` (≈ pre-change, wall v=0) and `wallBC=freeSlip` (new default) are **identical** — `|v|mean` 1.72, KE 0.042, embMin 0.94, slope 0.99, matches Part 34, so the `computeBoundaryVelocities` addition is a **strict no-op on the landed `iisph` baseline**. `wallBC=noSlip` + `viscidNu=0.01`: slosh KE 0.042 → 0.0098 (4×), `|v|mean` 1.72 → 1.07, gradient held (slope 1.02) — **but** embMin 0.94 → 0.60 and `|v|max` spikes to ~3.7. `wallBC=freeSlip` + `viscidNu=0.01` (fluid-only): embMin → 0.43. `wallBC=extended` (MLS) + `nu`: unstable (KE 0.81). **Cause:** `computeVelocityDiffusion`'s `inviscid=False` branch (`wp_viscosityDelta.py`) is `μ_ij·∇W` — the pair velocity normal-projected onto `x_ij` + zeroed for separating pairs — so it carries **no tangential shear stress**; a no-slip mirror only adds noisy *normal* wall damping. Part 39 got a clean result because `_physViscosity` used a real Brookshaw *vector* Laplacian `ν∇²v` + a mirror-velocity clamp. `dissipation/pi.py`'s `computePi_actual` is the same story (all Monaghan-family, projected). **So the module/config layer has no shear-carrying laminar viscosity — a TODO** (Morris 1997: full `v_ij` vector, no approach-only clamp; add as a `DiffusionParameters`-wired `ViscosityTerms` option with a `deltaSPH` regression pass, and fix `wp_viscosityDelta.py`'s "Morris-style Laplacian" docstring). Until then `hydrostaticColumn` stays `freeSlip` and the bounded free-slip slosh stays documented as non-fatal (§1.12, Parts 32/34/38). §1.14 got a follow-up block; §2 got two rows. *`iisph` on `tgv` (validation-ladder item 3, user's ask "run the tgv case through this solver path").* `tgv --scheme iisph` nx=64 **injects energy**: KE 9.86 → 1876 by t=0.1 (step 100), `|v|max` 1.0 → 29, then a wrong bounded plateau (KE ~1200, `|v|max` ~25) for 300 steps — while **density stays near-perfect** (`rho` [0.995, 1.007], `rhoStd` 1.3e-3). `divergenceFree` holds `tgv` (KE ratio 0.996, monotone); `dfsphReference` (= `iisph` + the divergence-free pass) is bounded (KE ratio 0.81, non-monotone). The *only* structural difference from `iisph` is the divergence pass, so: **plain IISPH (CD-only velocity-impulse) is not a general incompressible scheme** — the density-invariance solve controls `rho` but not the velocity field, and on a vortical flow the unconstrained pressure impulses spin the vortices up (§1.2 from the other side). `iisph` is viable only near-quiescent (`hydrostaticColumn`, `staticBlob`). `randomFlowIncompressible --bounded` confirms it (`iisph` `|v|max` → 1.3e6 by step 40 where `divergenceFree` holds at 1.06); `kolmogorovIncompressible` (forced, from rest) is milder — `iisph` `|v|max` 3.78 vs `divergenceFree` 2.51 at matched KE, a rougher field not a blow-up. This makes the divergence pass **non-optional** for any general scheme built on `iisph` (ranked queue items 4 / 9). *`omniIncompressible` on the same cases (its viscosity path was never bespoke — a faithful omniSPH XSPH/BXSPH `_xsphFilter`, untouched here):* **holds `tgv`** (KE ratio 0.79 over 200 steps — bounded, over-dissipative from the `XSPH_FLUID = 0.05` default, vs `divergenceFree`'s 0.996) and `kolmogorovIncompressible` (KE → 1.38 vs 2.51), because its 3-iter divergence pass does what `iisph` lacks; but **diverged on `randomFlowIncompressible --bounded`** with `WALL_PRESSURE_MODE = 'mls'` — the constant-density Jacobi caps out on step 1 (`errRho` 3.4e-2, `max｜p｜` ~4e4 → `｜v｜max` 84; the divergence solve is fine, `errDiv` ~1e-10). `'mls'`'s linear extrapolation term assumes a locally-linear near-wall pressure (Part 41's quiescent column), wrong for a sheared flow. **Fixed: `omniIncompressible.WALL_PRESSURE_MODE` default `'mls'` → `'shepard'`** (0th order, no linear term) — threads both regimes: `hydrostaticColumn` nx=128 holds (`｜v｜max` ~0.5, exact hydrostatic gradient, 350 steps), `randomFlowIncompressible --bounded` holds (`｜v｜max` decays 2 → 0.4, 300 steps). It makes the run survive, not the CD Jacobi converge (still caps out, §1.7). `'mls'` kept as an option. `dfsphReference` / `iisph` flag unchanged (`None`). *Then chased the CD solve itself (user's steer "chase the cd solve"):* captured `omniIncompressible`'s constant-density system `A p = s` on `randomFlowIncompressible --bounded` step 1 offline — the Jacobi **stalls** (`|r|_2` flat at ~8e-2 = `|s|_2`, never drops), because the box is **fully closed / no free surface** → the pressure operator is pure-Neumann (`A·1 ≈ 0`) and `s` is **99.98 % its own mean** (`mean(s) = -1.2e-3`, `|s - mean(s)|_2 = 1.16e-5`), so the constant part (the §1.1 lattice bias) is in `null(A)` and `A p = s` has no solution; MINRES/CG break down immediately. **Fix: the pure-Neumann compatibility projection** — subtract `mean(s_fluid)`, iterate on the spatial residual with `p` mean-zero (no `p ≥ 0` clamp — a closed box has no tensile-instability free surface): the Jacobi then *converges*, `|r|_2` 1.05e-5 → 1.2e-6 over 256 iters. **Landed: `omniIncompressible.CD_SOURCE_PROJECT = 'auto'`** — projects only when `s` is mean-dominated (`1 - |s - mean(s)|/|s| > 0.7`), a strict no-op wherever there is a free surface (`hydrostaticColumn` step 1 `frac_uniform ≈ 0.09`, `dambreak` likewise). Full matrix (nx=64/128, 120–150 steps): `randomFlowIncompressible --bounded` holds with the CD Jacobi now *converging* (KE 0.34 → 0.10 decays, `rho` 0.996–1.007); `hydrostaticColumn` `slope` 0.995; `dambreak` / `tgv` / `kolmogorov` unchanged. §1.7 got a Part 42 block, §2's "Zero-meaning the source" row split (negative for VD+PS, correct for `omniIncompressible` on a closed box). The free-surface CD solve still caps out (§1.7) — contractive solve / `band2018pb` is the remaining work. `gradcheck_incompressible` + tgv/shearWave/dambreak physics green; no `@wp.kernel` touched; no `divergenceFree` default changed. |
| 43 | 09-01 | **The free-surface CD solve characterised: near-singular, not slow — `CD_TIKHONOV` (opt-in) restores the Jacobi's iteration budget; non-symmetric Krylov ruled out.** *(DFSPH_IMPROVEMENT_PLAN.md active track "Next" item 1.)* Captured `omniIncompressible`'s density-mode `A p = -dt²·div(a_p(p))` offline from a *healthy* run (`scripts/probe_omniIncompressibleCDSystem.py`, wraps `_solve`, rebuilds the Krylov operator alongside the real jacobi solve at a chosen density-solve index). On `hydrostaticColumn` nx=64 step 30: the source is 99 % spatial (`|b−mean|/|b| ≈ 0.99` → `CD_SOURCE_PROJECT` correctly does not fire), but the *clamped* Jacobi leaves `|r|₂/|b| ≈ 0.94` while `|p|max ≈ 23`; drop the `p ≥ 0` clamp and raw BiCGStab/GMRES drive `|r|₂/|b|` to ~1e-2 **with `|p|max ≈ 640` and `|p|₂ ≈ 1500`** — the operator is **near-singular** (the free surface pins the pressure constant only weakly), the minimum-residual solution blows up along the near-null space. On `hydrostaticColumn` nx=128 the running scheme's CD Jacobi hits its 256-iter cap every step (omniSPH's floored `mean(max(·,−1e-3))` metric reads "converged"); `randomFlowIncompressible --bounded` nx=64 is the same even *with* the compatibility projection. **Landed (default-inert): `omniIncompressible.CD_TIKHONOV`** (`0.0`) — a uniform absolute diagonal shift `tik·median(|α_fluid|)` (solving `(A − eps|D|) p = s`, a nearby slightly-compressible problem), applied only for the density solve where `CD_SOURCE_PROJECT` did not fire. Uniform, *not* per-row `∝ α` — the kernel-deficient near-surface rows have tiny `|α|` and that is where the near-null space lives. On the Jacobi path, `tik = 0.1`: `hydrostaticColumn` nx=128 density solve **mean ~210 → ~75 iters** (off the cap), holds 400 steps, `embeddedMinDensity` 0.984 → 0.99, `slope` 0.999 → 1.005, `maxRho` 1.012 → 1.027, KE band unchanged ~2e-3; **strict wash on `dambreak`** (its CD solve already converges in the 3-iter minimum). `tik = 0` bit-identical to pre-Part-43. **Negative: `CD_SOLVER ∈ {'bicgstab','gmres'}`** (also landed, as scaffolding + a reject guard: matvec from the existing `accel`/`_divergence` closures, `1/α` Jacobi preconditioner, zero warm start, `wallPressureExtrapolation(clampNonNeg=False)` for linearity). Offline at step 30 BiCGStab + Tikhonov 0.1 converges beautifully (`|r|₂/|b|` 1e-6 in 38 iters, `|p|max ≈ 24`) — but **in the closed step-by-step loop it detonates on step 1** (`|p| ~ 1e9`), on every wall-bounded case, free surface *and* compatibility-projected closed box, with or without Tikhonov. A reject guard (`|r| ≥ |s|`, `|x|₂ > 1e3·|s|`, `|x|max > 1e5`, or non-finite → the step takes no density correction) stops the 1e9 blow-up but the run then loses density control (`maxRho ~ 1.4`, `slope 0`). MINRES already breaks down here (`status −13`); BiCGStab/GMRES do too — the composed `−dt²·div(a_p_IISPH(p_with_wall))` is near-singular *and* strongly non-symmetric near the wall, so a matrix-free Krylov solve is not the path. **The real fix (ranked-queue item 0 / "Next" item 1): `band2018pb`** (boundary samples as PPE unknowns → consistent + symmetric near-wall block → symmetric Krylov applies) or an explicit symmetrisation of `A` via `computePressureShiftIISPH`. `'jacobi'` stays the default and only usable `CD_SOLVER`. New tools: `scripts/probe_omniIncompressibleCDSolver.py` (end-to-end A/B across `CD_SOLVER` × `CD_TIKHONOV` on `hydrostaticColumn`/`dambreak`/`randomFlowBounded`), `scripts/probe_omniIncompressibleCDSystem.py` (offline linear-system capture). `wallPressureExtrapolation` got a `clampNonNeg` kwarg (default `True` = unchanged). `gradcheck_incompressible` + tgv/shearWave/dambreak/incompressible physics green; no `@wp.kernel` touched; no `divergenceFree` default changed; `omniIncompressible` defaults unchanged (`CD_SOLVER='jacobi'`, `CD_TIKHONOV=0.0`). |
| 44 | 09-02 | **The "symmetrise `A` + MINRES" interim probe (active-track "Next" item 1) is a dead end — `A` is *already* symmetric; the blocker is rank deficiency, and only full `band2018pb` addresses it.** `scripts/probe_omniIncompressibleCDSymmetry.py` captures the healthy density-mode system *inside* `applyConsistentCoupling` (the existing `probe_omniIncompressibleCDSystem.py` evaluated its captured closure *after* the coupling context had exited — stale boundary state; this one does all analysis at capture time) and, on that one system, measures the symmetry defect of three operator forms and runs relaxed Jacobi / MINRES / CG / BiCGStab on each. Cases: `hydrostaticColumn` nx=64 s30 (free surface + walls), `randomFlowIncompressible --bounded` nx=64 s5 (closed box), `dambreak` nx=64 s30. **(1) `A` is already symmetric.** `A_plain` (boundary `p ≡ 0`, BWJ23 Eq. 33) and `A_krylov` (`krylov.buildIISPHMatvec`, `staticBoundary`, `dt²`) have relative symmetry defect `|⟨Ax,y⟩−⟨x,Ay⟩| / (‖Ax‖‖y‖)` = **2.7e-5** (closed box) to **8e-3** (dambreak) — fp32 + kernel-asymmetry noise. The per-iterate `wallPressureExtrapolation` Robin closure (`A_wall`, the current operator) adds only ~4e-3 more defect and a ~10 % operator perturbation `‖(A_wall−A_plain)x‖/‖A_plain x‖`. **Symmetrising `A` via `computePressureShiftIISPH` buys nothing that is not already there.** **(2) MINRES / CG / BiCGStab all diverge on the un-regularised system, symmetric form included** — `|x|` → 1e4–1e7, status −14 (max-iter, residual growing) / −16 (CG indefinite) / −10 (rho breakdown) — on every wall-bounded case. The operator is **rank-deficient** (near-null space from kernel deficiency at the free surface *and* the wall corners; `median|alpha_fluid|` as low as **2.5e-5** on `dambreak`), so a symmetric Krylov has nothing to converge to. Only a uniform Tikhonov shift bounds `|x|`: on the closed box `tik = 0.1` gets every method to `|r|/|s| ≈ 0.1` in 20–70 iters; on the free-surface column it bounds `|x|` (~23, physical) but still `|r|/|s| ≈ 2.8` — the shifted (slightly-compressible) problem, not the PPE. **(3) `dambreak`'s CD "converges in the 3-iter minimum" (Part 43) is the *floored omniSPH metric*, not a solved system** — the captured `A p = s` there has `|r|/|s| = 1.000` after 2000 relaxed-Jacobi iterations (`Ax ≈ 0`; tiny diagonal). Same mechanism as `hydrostaticColumn` (Part 43), even more kernel-deficient. **Consequence:** `computePressureShiftIISPH`-symmetrisation struck from "Next" item 1; the operator's symmetry was never the blocker, its **near-singularity at the wall** is, and the only listed fix that removes it is *full* `band2018pb` (boundary samples as PPE unknowns → near-wall rows get their own non-trivial equation + diagonal → `A` no longer rank-deficient there; the free-surface near-null space stays handled by the `p ≥ 0` clamp / `CD_TIKHONOV`). Item 1 rewritten around the paper's §2.3 discretization (unified Eq. 8 accel over fluid+boundary neighbours, fluid Eq. 9 / boundary Eq. 10 rows, volume-centric Eqs. 11–17, diagonals Eqs. 19–20, per-sample `ω_i`, all-row volume-error convergence). New tool `scripts/probe_omniIncompressibleCDSymmetry.py`. No code changed, no defaults touched — analysis only. |
| 45 | 09-02 | **`band2018pb` (Band et al. 2018, *Pressure Boundaries for IISPH*) implemented as a fresh operator module + a distinct scheme — the wall rank-deficiency is removed; `hydrostaticColumn` nx=32 holds, nx≥64 not yet.** *(DFSPH_IMPROVEMENT_PLAN.md active-track "Next" item 1 — the only path left after Part 44.)* **New: `src/warpSPH/modules/incompressible/pressureBoundaries.py`** — the whole extended-PPE operator set: `bandRestVolumes` (`V0 = m/ρ0`), `bandActualVolumes` (`V = V0 / Σ_j V0_j W_ij`, fluid + boundary alike), `bandBoundaryUnknownMask` (`kind == 1` rows with `Σ_bf V0_f W_bf > MIN_FLUID_CONTACT = 0.02` — the interface layer only), `bandVelocityDivergence`, `bandPressureAccel` (Eq. 8 `a^p_f = −(V_f/m_f) Σ_j V_j (p_f+p_j) ∇W` via `warpOperation(Gradient, Summation)`), `bandApplyOperator` (`(Ap)_i = −dt²·div_diff(a^p)` — Eqs. 9 & 10 at once, since `a^p ≡ 0` on boundary rows makes boundary-boundary pairs vanish), `bandDiagonal` (`computeAlpha(apparentVolumes=V, includeBoundaryReaction=False)` = Eq. 19; on boundary rows the first term `(V_b/m_b)‖Σ V_j ∇W‖²` is added back → Eq. 20), `bandRelaxation`, and a `bandInjectVolumes` context manager that sets `state.densities := state.masses / V` so every `warpOperation`'s `m_j/ρ_j` weight becomes the band volume `V_j` (the `applyConsistentCoupling` trick). **Composed entirely from existing `warpOperation` primitives + `computeAlpha` — no new `@wp.kernel`, no gradcheck burden.** **New: `src/warpSPH/schemes/band2018pb.py`** + **`IncompressibleSPHScheme.band2018pb = 4`** + `builder._band2018pb` — a distinct IISPH-style single relaxed-Jacobi solve over the concatenated `p = [p_f ; p_b]`, `p ≥ 0` clamp on both, floored convergence metric (`max(residual, −1e-3)`), reusing `DFSPHReferenceSystem`. **Paper→codebase adaptations:** the one-layer cubic-spline-`2h` constants `γ` (Eq. 12) / `β` (Eq. 14) are **dropped** — measured, they inflate `V_b` to ~1.4× `V0_b` on this codebase's Wendland2 / `n_h = 4` / five-layer band, injecting a spurious **+0.3** boundary source; boundary rows instead take the nominal `V0` and the fluid's full-neighbour `V` formula, and only the fluid-facing interface layer (130 of 720 boundary rows at nx=32) is an unknown (the deep layers have a zero Eq. 20 diagonal). **Verified (`scripts/probe_band2018pbSystem.py`):** (1) Eq. 8 accel on the analytic hydrostatic `p` → bulk `a^p_y ≈ +9.805` (cancels gravity — correct scale + sign), `a^p_x ≈ 0`; (2) `A` symmetric to ~3e-3; (3) **the wall interface rows now have healthy non-singular diagonals** — `dt²·a_bb` median 5.9e-4, *all* correct sign (0/130 positive), vs `omniIncompressible`'s `median|α| → 2.5e-5` near-null wall (Part 44). **The rank deficiency this item exists to remove is removed.** **The nx≥64 Jacobi divergence was the near-surface diagonal, fixed by a Tikhonov floor.** The relaxed Jacobi diverged at nx≥64 even at step 1 (`|x| → ∞`); a numerical probe (`A(e_i)·e_i` vs `bandDiagonal`) confirmed the band diagonal is `computeAlpha`-**exact** (ratio 1.000, fluid *and* boundary), so the cause is not a wrong diagonal but kernel-deficient near-surface rows with `|dt²·a_ii| ~ 1e-6` where the step `omega/a_ii` (× `s > 0`) detonates. **Landed: `band2018pb.DIAG_TIKHONOV`** (default `0.3` — non-zero, unlike `omniIncompressible.CD_TIKHONOV` which is opt-in): solve `(A − eps·median|dt²a_ii|·I) p = s`, applied to both the diagonal and the operator. **Holds now** (`OMEGA_FLUID = 0.1`, `DIAG_TIKHONOV = 0.3`): `hydrostaticColumn` **nx=32 (600+ steps)** — `|v|max` 0.18, `KE` 4e-4, `embeddedMinDensity` 0.997, `maxDensity` 1.02 — and **nx=64 (400 steps)** — `|v|max` 0.10, `KE` 2e-4, `embeddedMinDensity` 0.999, `maxDensity` 1.04 — both quiescent, near rest; `tgv` (periodic) clean. **Does not hold:** **nx=128 still sprays** — bounded (`maxDensity` 1.008, no `inf`) but `|v|max` ~4 and a thick ballistic surface layer (`embeddedMinDensity` → 0.14); lowering `OMEGA`/`TIK` makes it worse. **Inherited unchanged:** the *free-surface* source pollution — `1 − V0/V → +0.32` at the column surface (`n_h = 4` kernel deficiency, §1.1 in volume-centric form) makes `s` **~99 % positive**, so the Jacobi / MINRES / CG still do not converge the linear system there; the `p ≥ 0` clamp + floored metric are what hold nx≤64 (as for `omniIncompressible`, Parts 42/43). So `band2018pb` fixes the wall but not the free surface — as Part 44 / the plan predicted. Next: the nx=128 surface spray (per-iterate surface closure, source floor `s ← min(s, ε)`, or resolution-scaled `OMEGA`/`TIK`), then whether a symmetric Krylov now applies on the consistent `A`, then re-grade nx=128 / `randomFlowIncompressible --bounded`. New tool `scripts/probe_band2018pbSystem.py`. Suite impact: a new enum member + scheme; existing schemes/defaults untouched; no `@wp.kernel` touched. |
| 46 | 09-03 | **`divergenceFree` `dfsph_step` was rewritten (c637785, "tmp commit") from the VD+PS `solveDivergenceFree` + `finalize`-shift path onto `omniIncompressible._solve` for both the divergence and constant-density passes — it now *holds* `hydrostaticColumn` (nx≤128) but the periodic KE tests regressed. Part 46: characterised the `hydrostaticColumn` residual + a per-case XSPH knob.** The quiescent column now holds to `t=1` (`pressureSlopeRatio` 1.001, exact gradient) with a **bounded, undamped free-surface limit cycle** — motion in the top ~3 fluid rows only (the kernel-truncated skin: `ρ≈0.69ρ0` top row, static). `dfsph.XSPH_SCALE` (default 0.0; per-case via `schemeConfig.xsphFilterScale` — `hydrostaticColumn`'s `xsphScale` param) folds omniSPH's `SPHSimulation::XSPH` into `dvdt` *before* the next divergence projection: `scale=1.0` (= `XSPH_FLUID 0.05`) takes nx=64 KE 3e-5→5e-7 (55×), nx=128 KE 7e-4→3e-7 (2250×) with `densityP05`/`embeddedMinDensity`→1.000; the cycle then decays instead of persisting. Dissipative (tgv KE −33% at scale 1.0) so it stays per-case, not a scheme default. `SURFACE_SOURCE` (`'clamp'`/`'mask'`/`'shepard'` on the CD source's `1−ρ/ρ0` at the surface) — measured a wash. §1.16. Scripts: `probe_hydrostaticColumnDfsph{Surface,SurfaceSource,Tune,Xsph}.py`, `probe_dfsphXsphRegression.py`. |
| 47 | 09-03 | **The `tgv` energy injection the rewrite introduced is a *missing particle shift* (the "PS" of VD+PS the tmp-commit gutted from `finalize`), not the projection — resolved by gating the in-step CD / VD+PS shift on `gravityConfig.active`; full physics + runner suite green (82/0) for the first time since c637785.** With a perfect lattice there is no injection at all; it appears only once the jitter perturbs the lattice and scales with it, because `relaxLattice` relaxes against `solveIncompressible` while the run's `_solve` has a different fixed point → the first ~15 steps re-snap. Over t=1.0 the baseline `tgv` runs away (KE ×1.86, peak ×46, `|v|`→3.1); the shipped scheme also *detonates* `randomFlowIncompressible --bounded` at fixed dt (KE ×60) — the adaptive CFL clamp hides it. **The two cases each need the CD solve applied a different, incompatible way:** `hydrostaticColumn` as an unprojected in-step velocity impulse (its only hydrostatic support); `tgv` / the bounded box as a post-integrator *position* shift (Q1: the position move does 100 % of the work, the Eq. 17 resample nothing; `_PS_SHIFT_AS_VELOCITY`→NaN). Applying both double-counts the density error → blow-up; `SOLVE_ORDER='cd_then_div'` contains it but degrades the column (the divergence pass must run *after* / not project the CD impulse — that impulse is the column's support). **Landed: `dfsph.INSTEP_CD='auto'` + `incompressible._RESTORE_PS_SHIFT='auto'`** (both key off `schemeConfig.gravityConfig.active`): gravity-active → in-step CD, no shift (byte-identical to the prior holding behaviour); otherwise → no in-step CD + VD+PS `'cd'`-mode position shift. No case runs both. `tgv` nx=32/20 KE ×0.9996 / peak ×1.0000 / monotone — passes both `test_tgvKineticEnergy*`; `hydrostaticColumn` unchanged; `randomFlowIncompressible --bounded` fixed-dt KE ×0.994. **Cost:** the shift adds a `solveIncompressible` + adjacency rebuild per step on non-gravity cases. **Still open:** the bounded box's *graded* (adaptive-dt) run still NaNs even with PS-restore, and `kolmogorovIncompressible` is inert (`dfsph_step` zeroes the forcing, `computeForcing(...)*0` — an undocumented tmp-commit change); neither is in the suite. Rejected: `_PS_SURFACE_MASK` (NaNs the column faster), `'delta'`-mode shift alone (no support with `INSTEP_CD` off), `'delta'`+CD (holds the column, `tgv` decay-rate passes at iterations~4 but a peak ×1.008 bump remains). New: `scripts/probe_dfsphAblation.py`, `scripts/make_video_dfsph{PsCompare,BoundedRandom}.py`. §1.17. Ablation (nx=48/300, fixed dt): PS-restore holds `tgv`, the bounded box, a static obstacle, an oscillating obstacle (`drivenSquare`), and a uniform oscillating body force (`dfsph.GRAVITY_OSC`) — 5/6 — and the *only* case it cannot hold is the free surface (`solveIncompressible`'s `ρ*` 0.9 clamp compacts the surface skin). Refactored the two `_solve` passes into an ordered `_omniPass` helper (`SOLVE_ORDER`), default path behaviour-neutral. |
| 48 | 09-03 | **`c637785` "Immediate" fallout: A2 fixed, A3 landed, A1 investigated + confirmed unfixable without the near-wall CD work; Parts 34–48 validation run. Suite 86 → 89 passed / 1 xfail.** *A2 — `kolmogorovIncompressible` inert:* `dfsph_step`'s `forcing = computeForcing(...) * 0` was a tmp-commit mid-experiment leftover (`f401e4a` had it un-multiplied). Dropped the `* 0`. `computeForcing` returns an all-zero tensor for every incompressible-path case with no `forcingFunctions` (only `kolmogorovIncompressible` has one), so bit-identical everywhere else; `hydrostaticColumn`/`dambreak` take gravity via `computeGravity`. Verified: KE 0 → 2.06 / 40 steps (nx=32), density [0.99, 1.003]. *A1 — `randomFlowIncompressible --bounded` divergence: characterised, NO stopgap works.* The blow-up is **a density excursion that builds near the wall over many steps** (ρ spreads [0.98,1.02] → [0.2,3.4] while KE / `\|v\|` stay flat), then the pressure response detonates `\|v\|`. Onset time scales with `nx`: nx=32 t≈0.025, nx=48 t≈0.16, nx=56 t≈0.15, **nx=64 t≈0.95**, nx=96 t≈0.9, **nx=128 (default) step 4**. nx=64 merely lasts longest → a `maxDt = 1e-3` cap "held" it 400 steps there (KE ×0.994) and NOWHERE else — a false positive; the `boundedMaxDt` param from the first pass was **reverted**. **Isolated (probe toggling `_RESTORE_PS_SHIFT` / `INSTEP_CD` / `DIVERGENCE_SOLVER`):** it is the constant-density `solveIncompressible` **inside the VD+PS shift** (`systems/incompressible.finalize`), not the divergence `_solve` — B (`_RESTORE_PS_SHIFT=False`) delays it (nx=64 → step 90) but cannot hold it and can't be global (tgv needs the shift, §1.17); C (shift, no position move) detonates *fastest* (step 91); D (shift, no velocity resample) silently rots ρ to [0.14, 4.97]. Nothing else moves the needle: `--cflFactor 0.2` (identical blow step), `DIVERGENCE_SOLVER='vdps'` (**worse** — step-1 blow, cap off; the pre-`c637785` convergent projection is not a drop-in here), `boundaryPressureMode` ∈ {mdbcDensity, mdbcMlsPressure, plain} (identical), `nu = 0.01` (worse). Same near-singular pure-Neumann near-wall CD operator as Parts 42–45 — `iisph` / `omniIncompressible` fail this same case (Part 42). **Folded into the Active track / `band2018pb` (this case is its acceptance test).** *A3 — regression tests* (`tests/test_physics.py`): `kolmogorovIncompressible` forcing drives the flow + stays incompressible; `randomFlowIncompressible` **periodic** does not diverge + stays incompressible; `randomFlowIncompressible --bounded` `@pytest.mark.xfail(strict=True)` at nx=96 (blows in a few steps) — flips to a suite failure the moment it holds. *Validation:* `run_tests.sh` green bar 2 pre-existing failures (the `implicitShiftAutomatic` flake — passed on retry — and `test_incompressibleKrylov.py::test_optimalStepRejectedForConstantDensitySolver`, which **fails on a clean tree too**: `solveIncompressible` + `JacobiRelaxationMode.optimal` no longer raises `ValueError('optimal')` — the known-open "should raise on the Krylov path" guard has regressed at some earlier point); `gradcheck_incompressible` green; `run_sweep.py --nSteps 25` **33/33**; a deeper nx=48 / ~200-step pass of every `divergenceFree` case holds with the gravity gate routing correctly (`inStepCD` for dambreak/column, `PSshift` for the rest). **A4 (the `_RESTORE_PS_SHIFT` per-step `solveIncompressible` cost) still open.** No `@wp.kernel` touched. |
| 49 | 09-04 | **The `c637785` rewrite's dropped `mdbcNoPenetrationShift` re-wired into `dfsph_step` + back on by default; wall-crossing metrics re-graded (§1.6, §2).** The rewrite left `computeMdbcNoPenShift`'s call commented out, so post-`c637785` `divergenceFree` relied on the pressure projection alone for no-penetration — and a violent wall impact crossed the band. Restored the call at the pre-rewrite spot (fold `nopenshift/dt` into `dvdt` *before* the pressure solves, as `schemes/deltaSPH.py` does), gated by new module flag **`dfsph.NOPEN_SHIFT`** (`'config'` default → obey `schemeConfig.solverConfig.mdbcNoPenetrationShift`, itself `True`; `True`/`False` force). A/B (`scripts/probe_dfsphWallPenetration.py`, nx=64): `dambreak --scheme divergenceFree` wall crossing **5 → 0** (deepest 0.82 dx → 0), `columnCollapse` **6 → 1** (1.21 → 1.10 dx), impact `|v|`max down ~30 % on both (9.1→6.2, 11.3→8.4), `hydrostaticColumn` quiescent **untouched** (`|v|`max 0.35→0.22, pen 0 either way), `randomFlowIncompressible --bounded` **unchanged** (still BLOW ~step 22 — that is the near-singular pure-Neumann near-wall CD solve, §1.7 / Active track, a different failure from ballistic penetration). Confirms §2's "removing it is strictly worse". Also: `dambreak.diagnostics` gained an `nPenetrating` / `maxPenetrationDx` watch (off `ctx.scratch['interiorDomain']`) matching `columnCollapse`'s; `c637785`-era unconditional `print`s in `dfsph_step` (density stats) and `omniIncompressible._solve` (`[mode] N iters` — new `VERBOSE_SOLVE` flag) gated behind verbose. Incompressible physics suite green (14 pass / 1 xfail). No `@wp.kernel` touched. |
| 50 | 09-03 | **`band2018pb`'s two remaining blockers were both bugs in the port, not physics — `hydrostaticColumn` nx=128 and `randomFlowIncompressible --bounded` (nx=64 & nx=128) both hold now.** *(Active-track "Next" item 1; the bounded box is that item's acceptance test.)* **Bug 1 — `bandRelaxation` scaled only the fluid rows.** The paper's per-sample `ω_i = 0.5·V0_i/h^d` applies one base constant to *every* row, a boundary sample stepping smaller only because its rest volume is smaller. The port detuned the fluid base to `OMEGA_FLUID` for `n_h = 4` but left boundary rows at a hardcoded `0.5`; since `bandRestVolumes` returns `V0 = m/ρ0` for both kinds, `V0_b = V0_f` and the boundary kept a flat 10× larger ω — on rows whose Eq. 20 diagonal is itself ~10× *smaller* than the fluid's Eq. 19 one (Eq. 20 lacks the first term), i.e. a Jacobi step `ω/a_ii` ≈ **100×** the fluid's. Measured (nx=128, step 20): `p_b` max **96** against a peak fluid `p` of 11.8, Eq. 8 turning that into `|a_p|` **1346** on the near-wall fluid; the column is kicked apart, the source goes **96 % positive**, every row clamps to `p = 0`, and the floored convergence metric then reads `err < 0` so the solve **exits at the 8-iteration minimum doing nothing** — a self-reinforcing trap. Fixed to `ω_i = OMEGA_FLUID·V0_i/V0_f` for all rows (reduces to a uniform ω today; stays correct if the paper's Eq. 12 boundary rest volume is ever restored). **Bug 2 — no gauge for a closed domain.** Eq. 8 is a *summation* gradient, so a uniform `p` produces no acceleration wherever the kernel support is complete: in a fully enclosed domain the constant is a null mode of `A`, and the Eq. 18 `max(·,0)` clamp — which can only push a row *up* — ratchets it away instead of pinning it (`randomFlowIncompressible --bounded` step 1 solved to `p ∈ [1.6e3, 2.6e3]`, a pure **offset**, `|a_p| ~ 2e3`; KE reached **1.3e8** by step 200 with the density still at [0.992, 1.007] — Part 42's "a solve overshoots" signature). **This is a different mechanism from Part 42's incompatible source and its test does not detect it:** `fracUniform ≈ 0.03` here against `CD_PROJECT_THRESHOLD = 0.7`, because the *source* is well-conditioned and it is the *solution* that is unpinned. **Landed: `band2018pb.CLOSED_DOMAIN_GAUGE`** (`'off'`/`'auto'` default/`'always'`) keyed on a new direct null-mode test **`bandConstantModeRatio` = `rms|A·1| / rms|a_ii|`** over the solve rows (one extra operator application per step, not per iteration) — nx=64 step 1: closed box **0.032**, free-surface column **1.29**, a 40× separation, `NULL_MODE_THRESHOLD = 0.25`. When it fires, each iterate is projected to zero mean over the solve rows and the non-negativity clamp is dropped (a closed domain has no free surface, so no tensile instability for it to suppress — the same pairing Part 42 reached for its projection). **Retuned on the fixed operator:** `OMEGA_FLUID` 0.05 → **0.1** (0.2 and 0.3 detonate nx=128, `|v|max` ~1e3); `DIAG_TIKHONOV` 0.3 → **0.05** (0.05 and 0.1 both hold nx=32/64/128, 0.3 loses nx=128 at `embMin` 0.38; 0.05 over-compresses least — bulk ρ 1.0035 vs 1.0063 at nx=64 and 1.013 vs 1.023 at nx=128, `pressureSlopeRatio` correspondingly closer to 1 — and the paper carries no Tikhonov term at all). `tik = 0` still fails (`embMin` 0.44 / 107 spray at nx=64), so the floor is genuinely needed; both earlier values were tuned *against bug 1*. **Results.** `hydrostaticColumn` 200 steps: **nx=128** bulk ρ 0.934 → **1.013**, `embeddedMinDensity` 0.14 → **0.9998**, `pressureSlopeRatio` 0.020 → **1.015**, `|v|max` 2.70 → **0.16**, spray 1695 → **2**; nx=64 also improves (bulk ρ 1.0066 → 1.0035, `slope` 1.004); nx=32 a wash. Column results are **bit-identical with the gauge on and off** — it correctly does not fire at a free surface. **`randomFlowIncompressible --bounded` HOLDS under `band2018pb`:** nx=64 KE decays 0.446 → 0.213 monotonically / 200 steps, `|v|max` ~1.0 (peak 1.37), ρ ∈ [0.991, 1.017]; nx=128 KE 0.322 → 0.302 / 150 steps, `|v|max` 1.0–1.36, ρ ∈ [0.995, 1.018]. **nx=96 is the one exception and it is a case-IC artifact, not the scheme** — its step-0 state is already `KE 9.23` / `|v|max 10.2` versus ~0.3 / ~1.0 at nx=64 and nx=128, i.e. the random-flow initial condition is anomalous at that resolution *before any solve runs*; the run then stays bounded but shows recurring KE injection (0.36 → 35 → 26 → 38). Worth chasing in `cases/randomFlowIncompressible.py` separately. **`test_randomFlowBoundedDoesNotDiverge` stays xfail** — it runs the case's *default* scheme (`divergenceFree`), which is untouched. `tgv` under `band2018pb` stays bounded; the gauge does fire there (periodic ⇒ complete support ⇒ same null mode) at a mild energy cost (KE ratio 1.016 → 1.038 / 100 steps) and a slightly tighter density band — kept, since gauging a periodic domain is mathematically right and `divergenceFree` is the graded `tgv` path. **New tests:** `test_band2018pbColumnStaysQuiescent` (nx=64, 200 steps — quiescence + rest density + hydrostatic gradient) and `test_band2018pbBoundedRandomFlowDecays` (nx=64, 120 steps — decay + no velocity spike + tight density) lock both fixes; first suite coverage `band2018pb` has had. `gradcheck_incompressible` green; no `@wp.kernel` touched (the module is still composed from existing `warpOperation` primitives). **Also: `literature/band2018pb_…pdf` was corrupt on disk** — header `%PDF-1.7` followed by `EF BF BD` bytes, i.e. the file had been round-tripped through a text decode and every binary stream destroyed; `pdftotext` and `mutool` both extracted 0 bytes and all 11 pages rendered blank. Re-fetched from the authors' copy (`cg.informatik.uni-freiburg.de/publications/2018_TOG_pressureBoundaries.pdf`, 12 pages incl. appendices). It was the only corrupt file in `literature/` and is gitignored, so nothing was committed. The plan's equation transcription was re-checked line by line against the clean text and **was correct**. |
| 51 | 09-04 | **[SUPERSEDED BY PART 52 -- the results table in this row is INVALID.** Four of the 14 experiments (`dambreak`, `sloshingTank`, `impact`, `squarePatch`) are weakly-compressible-first cases whose defaults are `Wendland4` + `KernelMeanSymmetric`, and the harness overrode only the integrator, so they ran under the wrong discretisation for an incompressible scheme. The grading was also field-only and could not see particle pairing or voids. Both fixed in Part 52; re-read the numbers there.]** **A per-scheme validation harness (`scripts/validate_scheme.py`), and `band2018pb` graded across 14 cases: 13 pass, `staticBlob` is the one real failure. `sloshingTank` runs the full 7 s SPHERIC TC10 with Sensor 1 in the measured band.** *(Answers the question `run_sweep.py` does not: take ONE scheme, push it through the cases it should handle, grade each on physics rather than "did not crash".)* **The harness.** 14 experiments, each in its own subprocess (one divergence cannot take the sweep); two profiles — `smoke` (coarse, minutes, run first) and `full` (graded resolution + `--video`); per-experiment generic guards (ran to completion, no divergence, finite KE, density band) plus its own physics check; a check whose metric is missing reports `n/a`, never a silent pass. Output is a timestamped `sweeps/validate-<scheme>-<profile>-<stamp>/` with `summary.md`, `summary.json`, per-experiment JSON/log, and an mp4 each. Scheme-agnostic: `--scheme divergenceFree` grades the shipped path and the two can be diffed. **Prerequisite fix — 7 sites assumed `divergenceFree` was the only incompressible scheme.** `dambreak` (1), `impact` (2) and `sloshingTank` (4) tested `ctx.scheme is IncompressibleSPHScheme.divergenceFree` where they meant "is an incompressible scheme", silently routing `band2018pb` / `iisph` / `omniIncompressible` down the weakly-compressible branch (wrong `dt` hook, wrong sensor-pressure path). Added **`enumTypes.isIncompressibleScheme()`** and converted all 7. Also added **`sloshingTank.sensorPressureWall`**: `band2018pb` solves pressure *at* the wall sample, which is the paper's own Sec. 3.3 tank scenario ("Pressure Boundaries enable the analysis of computed pressure values at the boundary"), so Sensor 1 can be read directly instead of Shepard-probing nearby fluid. **Four grading corrections, all of which had `band2018pb` failing cases it actually passes** — the first smoke run said 7/14 fail and most of it was the harness: (1) **raw `minDensity` fails every free-surface case, including correct ones** — it reads 1-3 ballistic spray particles, the Part 33 FOM artifact that `densityP05` / `embeddedMinDensity` exist to replace; added **`densityP05` (5th-percentile fluid density) to the shared `weaklyCompressibleDiagnostics` (11 cases) and to `dambreak` / `columnCollapse` / `staticBlob` / `sloshingTank`**, and graded the low side on it. (2) A single last-sample `pressureSlopeRatio` is not gradeable — it is a least-squares fit through the pressure column and swung **-1.1e5 .. +248** on one nx=32 run whose density profile was healthy; all watched series now also emit a `_tailMedian` (median of the final fifth) and the checks use it. (3) The generic low-side guard used the **global** minimum, which catches `hydrostaticColumn`'s documented Part 34 startup transient; it now grades the converged state, since blow-up is already caught by the divergence / finite-KE / max-density guards. (4) The three `randomFlowIncompressible` variants are the **same case** and were graded with inconsistent bands (0.97/1.03 vs 0.95/1.05); unified at 5%. A fifth, `gradeDensityLow=False`, drops the low-side density guard entirely for the small free-space bodies `staticBlob` and `impact`, where the case docstring already says so ("the surface *deficit* `minDensity` ~ 0.5 is sampling geometry, not a defect ... only the upper bound is a health check") — these have a large surface-to-volume ratio, so even p05 is dominated by legitimate surface deficit (`impact`: p05 0.29 on a run with KE ratio 0.98 and `maxDensity` 1.032). Clumping, the real failure mode, is on the upper bound, which is always graded. **Results (full profile, 14/14 videos rendered).** PASS: `tgv` (KE x1.010 / 300 steps at nx=64 — note it *fails* at nx=32/60, so the Part 50 injection is a coarse-grid artifact), `kolmogorov`, `shearWave` (x0.921), `randomFlow` periodic / bounded / obstacle (KE x0.687 / x0.478 / x0.377), `hydrostaticColumn` nx=64 (**slope 1.002**) and nx=128 (**slope 1.016**), `dambreak` (600 steps, **nPenetrating 0**, `maxRho` 1.018), `columnCollapse` (**nPen 0**), `impact`, `squarePatch`, and **`sloshingTank` — the full 8500 steps / 7 s, Sensor-1 peak 7.0 kPa, inside the measured 2.2-13 kPa SPHERIC band.** The new `sensorPressureWall` channel peaks at **25.8 kPa**, ~3.7x the smoothed fluid probe: the boundary sample sees sharp impulsive spikes the probe averages out, so that channel needs its own calibration before it is a validated figure. **FAIL: `staticBlob`** — a free-space blob that should sit at `|v| = 0` reaches `|v|max` 1.02 with `dispMax` > 0.1, while `centroidDrift` < 0.02 **passes**: local surface distortion, not a spurious net force. Coherent with the paper, which is a *boundary-pressure* method with no free-surface treatment at all (every scenario in it is a wall-bounded tank) — `band2018pb` fixes walls and inherits the free-surface problem, which on `hydrostaticColumn` is masked by the `p >= 0` clamp + `DIAG_TIKHONOV` and on a pure free-space body is not. **Correction to Part 50:** that entry reported `randomFlow --bounded` holding at `rho in [0.991, 1.017]`; that came from sampling every 12th step. Over all 200 steps the true range is **[0.952, 1.044]** — the KE trace is unchanged and still decays cleanly, but the peak density excursion is ~4.4 %, not ~1.7 %, i.e. far looser than the paper's 0.01 % average volume-error convergence target. Recorded rather than tuned away. |
| 52 | 09-04 | **The `sloshingTank` startup destabilisation is a LATTICE-DENSITY artifact, now computable in closed form and corrected; Part 51's results table is withdrawn.** *(User-reported: a sweep run delaminated into horizontal sheets by `t ~ 0.066` where `export/16-sloshingTank-dfsph_2026-09-03_14-30-29` (nx=200) and `..._14-29-45` (nx=60) are both clean.)* **Two wrong hypotheses, killed by measurement, recorded so they are not re-tried:** (1) *"the harness ran it under the WCSPH kernel"* -- true and worth fixing (`dambreak`, `sloshingTank`, `impact`, `rotatingSquarePatch` are weakly-compressible-first cases defaulting to `Wendland4` + `KernelMeanSymmetric`, and the harness overrode only the integrator; `INCOMPRESSIBLE_PRESET` now applies `Wendland2` + `SuperSymmetric` + `semiImplicitEuler` per scheme) -- but it did **not** fix sloshing. (2) *"nx=100 is too coarse to resolve the 0.093 m depth"* -- **false**: the user's nx=60 export is coarser still and clean. The effect is non-monotonic in `nx`, so it is not resolution. (3) *"the IC asserts rho = rho0 so the solver sees false compression"* -- **false**: the scheme recomputes the summation density at the top of every step; seeding it changed the result by 0. **What it actually is.** The first-step summation density is a *perfectly uniform* offset (identical at floor, bulk and free surface -- not a boundary or surface effect) that varies non-monotonically with `nx`: nx=60 -> 1.0039, nx=70 -> 1.0069, nx=80 -> 1.0022, nx=90 -> 1.0044, **nx=100 -> 1.0153**, nx=110 -> 1.0024, nx=200 -> 1.0016; the step-0 `|v|max` tracks it exactly (0.31 / 0.62 / 0.11 / 0.37 / **1.74** / 0.15 / 0.06). It decomposes **exactly** into two independent factors: `rho_measured = [m / (rho0 * prod_i s_i)] * L(h/s)`. **`L` is the lattice quadrature factor** -- an SPH density is a midpoint quadrature of `int W dV = 1` sampled on the lattice, so a defect-free lattice carrying `m = rho0 s^d` integrates to `rho0 L` with `L != 1`, depending only on `h/s`. **New tool: `scripts/lattice_density_offset.py`** computes it by the exact `n_h`-box direct sum (every point with `W != 0` has `|k_i| <= ceil(h/s)`, so the finite box misses nothing), using the repo's own kernel definitions transcribed from `warpSPHCore/kernels/kernelFunctions/`. Wendland2 2D: `L` = 1.1698 at `h/s` 1.5, 1.0376 at 2.0, 1.0051 at 3.0, **1.0012 at 4.0**, 1.0004 at 5.0 -- monotone, and **continuous across shell crossings** (the (4,0) shell enters at `h/s` 4.01 and the (4,1) at 4.13 with no jump, because entering neighbours arrive at `q -> 1` where `W -> 0`). `--validate <case>` reads the achieved per-axis spacing, `h`, mass and measured density back out of a live run: on `sloshingTank` the prediction matches to **1.1e-6 / 1.7e-6 / -8.9e-8** at nx=60/100/200. Per-axis spacing is required -- the sampler fits each direction independently and the lattice comes out slightly anisotropic (nx=60: sx 0.009375 vs sy 0.009400), and an isotropic `s^d` mispredicts by exactly that (2.7e-3). **Which factor dominates:** `L ~ 1.0012` is essentially constant (h/s is pinned near 4.01 at every `nx`); the nx-dependent part is `m/(rho0 * cell)` = 1.0004 (nx=200) to **1.0142** (nx=100), i.e. the sampler's own block fit. **The correction works:** scaling fluid masses to `rho0 * cell / L` takes nx=100's first-step density **1.015345 -> 0.999999** and its step-0 `|v|max` **1.738 -> 0.131**. It does *not* stop nx=100 pairing later, so that is a separate defect. `sloshingTank` never relaxes its lattice, unlike `tgv` / `shearWave` / `staticBlob` (`relaxLattice`) or `hydrostaticColumn` (`calibrateRestDensity`) -- giving it one of those is the durable fix. **Also corrected: `pairedFraction` was normalised by `config.dx`, which is NOT the achieved spacing** (measured ~0.6 dx at every `nx` on this case), so a "< 0.5 dx" test really fired at 0.83 of the true spacing and reported a spurious ~6 % pairing on known-good runs -- the basis for the (now withdrawn) claim that "the validated reference sits at a flat 6 % equilibrium". Self-normalised by the run's own median nearest-neighbour distance, the metric separates cleanly: `sloshingTank` nx=60 **0.0000** and `staticBlob` **0.0000** (both known good) against nx=100 **0.235** with `voidFraction` 0.198 (the reported failure); nx=200 reads 0.057. **Process:** `scripts/validate_scheme.py` gained a "DO NOT RUN CASUALLY" header -- the full profile is ~50 min/scheme, and a change must be validated by re-running the ONE affected case against a known-good export before any sweep. Two sweeps were burned on unvalidated changes here. Its `sloshingTank` entry is pinned to the reference configurations (smoke nx=60, full nx=200) with the measurement table inline. |
| 53 | 09-04 | **The lattice-density offset is FIXED at source: a shared `calibrateRestDensityMasses` corrects the particle mass at fixed spacing and fixed `h`. `sloshingTank` nx=100 goes from delaminating to matching nx=200. Also: how omniSPH solves the same problem.** *(Follows Part 52, which diagnosed the offset and built `scripts/lattice_density_offset.py`.)* **Why mass and not spacing.** `dx` is deliberately chosen to divide the domain evenly -- that is what places the domain bounds a clean `dx/2` from the real surface, which the boundary / ghost-particle representation depends on. Re-sampling at a spacing that happens to give a nicer lattice factor perturbs the initial geometry discontinuously, and picking a "good" `nx` is dodging the problem. The two legitimate levers are the **mass** (free, exact for an ideal lattice, but a real change to the fluid mass) and the **support** (keeps mass exact, improves the operators too, but costs neighbours as `(h/s)^d`). **The support route is bounded, not free:** on a square lattice `L(h/s) > 1` and decreases strictly monotonically (verified at 0.005 resolution over `h/s` in [3,6], zero non-monotone steps), so there is no `h` with `L = 1` and no interior optimum -- only an accuracy/cost curve. Wendland2 2D, `|L-1|` targets vs required `h/s` / lattice neighbours: 1e-2 -> 2.618 / 20; 3e-3 -> 3.410 / 36; **1e-3 -> 4.252 / 60**; 3e-4 -> 5.408 / 96; 1e-4 -> 6.645 / 136 (3.1x the neighbours of `h/s = 4`). warpSPH already runs `h/s ~ 4.0` (`L-1 = 1.2e-3`), so the quadrature term is NOT the problem here -- the `nx`-dependent part is the sampler's block fit (`m/(rho0 * cell)` 1.0004 at nx=200 vs **1.0142** at nx=100). **How omniSPH handles it (checked in `~/dev/omniSPH`; identical Wendland2 kernel `7/pi/h^2 (1-q)^4 (1+4q)`, `simulation/2DMath.h:68`).** It takes the mirror-image convention: it fixes the RATIO and snaps the geometry to the lattice. `SPH.h`: `packing_2D = 0.399200743165053487`, spacing `s = packing_2D * h`, so `h/s = 2.505005` is a hard constant everywhere; particle area is `pi r^2` with `h = r sqrt(targetNeighbors)`, `targetNeighbors = 20`. `config.cpp:428`: emitter bounds are snapped **outward** to an integer multiple of `packing` (it prints "Changed emitterMax from ... to ..."), then `domain.min/max = emitterExtent -/+ packing/2` -- the domain is *derived* from the lattice, half a spacing outside the outermost particles. So omniSPH earns the clean half-spacing boundary offset by fixing `s` and moving the domain; warpSPH earns it by fixing the domain and letting the sampler pick `s`. That is precisely why the offset is one constant there and varies with `nx` here, and it is why omniSPH's route is closed to this codebase. **omniSPH's calibration is the mass route in disguise, and lands at 1e-3, not 0:** its ideal lattice measures `areaRatio * L` = `0.985683 * 1.013588` = **0.999076**, i.e. 0.09 % below rest density -- it absorbs a 1.4 % quadrature error into a 1.4 % smaller assigned area. The packing giving exactly 1 is `0.399007431645928`; the shipped and exact values share the digit run `0074316` offset by one place (`0.3992|0074316|5053` vs `0.399|0074316|45928`), which is what a single inserted digit looks like -- recorded as a curiosity, not a claim. Its commented-out earlier value `0.39960767069134208` gives 0.997137, so the live one is the better of the two regardless. **Landed: `cases/weaklyCompressible.calibrateRestDensityMasses(ctx, system, quantile=0.75, fluidOnly=False)`.** Measures the sampled state's summation density and rescales mass so the **interior** reads `rho0`; "interior" is the upper `quantile` tail of the fluid density rather than a hard-coded geometric margin, so it is case-agnostic (a free surface reads low by kernel deficiency and falls in the lower tail, while every fully-supported particle shares one plateau). This replaces `hydrostaticColumn`'s inline y-margin version (which carried a `# the above code is garbage` comment) and finally implements `columnCollapse.calibrateRestDensity`, a parameter declared in Part 33 and never read. **`sloshingTank` now calibrates by default, auto-gated to incompressible schemes** (`calibrateRestDensity=None` -> on iff `isIncompressibleScheme`): `deltaSPH` evolves density through the continuity equation from `rho = rho0` and never forms a summation density, so it never sees the offset (its step-0 density is 1.000000 at every `nx`) -- **which is why the WCSPH SPHERIC validation was always clean** -- and calibrating there is a no-op that still perturbs the fluid mass ~1.5 % and the sensor peak ~1.6 %. **Result (`divergenceFree`, 60 steps).** step-0 `|v|max` becomes **0.0098 at every resolution**, killing the `nx` lottery: nx=60 0.306 -> 0.0096, nx=100 **1.738 -> 0.0098**, nx=200 0.0637 -> 0.0098; measured density 1.0039 / 1.0153 / 1.0016 -> **0.999999** in all three. nx=100 is fully repaired -- `pairedFraction` **0.235 -> 0.0000**, `voidFraction` **0.198 -> 0.0000**, `|v|max` 1.19 -> 0.040 -- and now behaves like nx=200. nx=200's residual `pairedFraction` ~0.057 is **unchanged** (0.0571 -> 0.0573), so that is a separate phenomenon and not the density offset. |
