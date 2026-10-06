# Resolved problems

Items that used to be in [OPEN_PROBLEMS.md](../../OPEN_PROBLEMS.md), moved here
once resolved, with their original section number so older references still
find them. Newest first. The original text is kept as it was when resolved.

## OPEN_PROBLEMS §16 — Monaghan host did not conserve total energy on a strong shock — RESOLVED 2026-09-30

Found 2026-09-30 by the AV_PLAN M0 report (`scripts/av_report.py`, `sedov`). Not investigated yet.

`scheme='Monaghan'`, Sedov, `E(t) = sum m (u + v^2/2)` (E0 = 1): **1D nx 400: 1.0 -> 2.33**
(NoneSwitch); **3D nx 24: 1.0 -> 1.23** (NoneSwitch) / 1.25 (Cullen-Dehnen); 3D nx 40 (full profile): +47 %.
`scheme='CompSPH'` on the same case holds **1.0 exactly**. On Sod and Gresho the Monaghan drift is
0.2-0.7 % (already over the `tests/test_physics.py` 5e-3 budget on some), so the defect grows with shock
strength / dynamic range (Sedov's cold background has `u ~ 1e-25`).

**Not time integration:** 1D nx 400 with `cflFactor` 0.3 / 0.15 / 0.075 gives E_final 2.3273 / 2.3151 /
2.3113 -- a 4x smaller dt moves it by 0.7 %. It is a formulation-level inconsistency between the
pressure force and `dudt`, or the density update, of `schemes/monaghan.py` (Owen adaptive supports,
`KernelMeanSymmetric` force vs `Gather` density, `computeDudtMonaghan` vs `computePressureForceSymmetric`).

**Why it matters for AV_PLAN:** Monaghan is the plan's AV laboratory host and Group E gates on energy
drift; every detector comparison on a strong shock (Sedov, Noh) would be confounded by an energy source.
First checks: A/B `adaptiveSupportScheme` (Owen vs none) and `adaptiveSupportCorrections`; compare the
`dudt` and `dvdt` pair terms for a symmetric two-particle test; check whether `dEdt` (the scheme's own
energy-rate output) sums to zero.

**Update 2026-09-30 (M0 baseline, `docs/av/av_baseline_2026-09-30.md`):** not caused by the missing
quadratic AV — `C_q = 2` makes Sedov worse (3D nx 40: +68 % vs +47 %) and **Noh (1D) also gains energy,
+42-65 % at `C_q = 2`**. So it is a general strong-shock property of the Monaghan host, not Sedov
specific. (At `C_q = 0` Noh is degenerate: cold gas, `c = 0`, zero viscosity, no interaction at all.)
CompSPH is exact on Sedov (1.000).

**Resolution (2026-09-30, AV_PLAN S5).** `scripts/probe_monaghanEnergy.py` evaluates each piece of
`schemes/monaghan.py`'s RHS on a mid-run state and reports the net energy it injects,
`E_dot = sum m (v . dvdt + dudt)`, which must vanish pair by pair. On Sedov (3D nx 24, 120 steps):

| piece | kinetic | heat | net |
|---|---|---|---|
| pressure force + work | +0.861 | -0.861 | -1e-7 (conserves) |
| **viscous force + heating** | **-0.5525** | **+1.1050** | **+0.5525** |
| conductivity | 0 | +3e-8 | ~0 |

The heating was **exactly 2x** the kinetic energy the viscous force removes, so each dissipation event
*added* one dissipation's worth of energy. It matches the baseline numbers directly: the integrated AV
energy (`avEnergyLinear`) equalled the energy drift on Sedov (0.467 vs 0.468) and on Sod (0.0107 vs
0.0115 absolute), and `C_q = 2` made it worse because it dissipates more. The kernel
(`computeThermalDissipation_Func_i`) accumulated `-V_j Pi u^2 lap W` without the 1/2 of Monaghan
(1992)'s `du_i/dt = 1/2 sum m_j Pi_ij v_ij . gradW_ij` (each pair's dissipation is shared between the two
particles). The heating kernel is used only by the Monaghan host, which is why CompSPH and CRKSPH
conserved exactly. Fix: the factor 1/2; the viscous force (and so the dynamics the coefficients `C_l`,
`C_q` define) is unchanged. After the fix every piece conserves to round-off (Sedov 1.0000 -> 0.9995 by
step 120, where it had been 1.133; Noh and Sod likewise) and the full suite and gradchecks pass.
Regression test: `tests/test_monaghanEnergy.py` (fails by 6-7 % of the energy scale without the fix).
**Consequence:** every Monaghan-host compressible result before this date carried the spurious heating --
the `tests/test_physics.py` Monaghan energy-drift budget of 5e-3 was sized to it. The AV baseline was
regenerated (AV_PLAN M0c).

## OPEN_PROBLEMS §5 — englishWedge concave-corner residual (and the fourtakas2019 sign bug) — RESOLVED 2026-09-28

- **`_gridSnapGhostOffsets`'s concave-corner ghost collapse**: ~40 boundary
  particles at a concave corner (e.g. englishWedge's base corner) collapse
  their ghost node onto one interior point, a degenerate near-coplanar
  stencil no fit handles well. Not rechecked as of 2026-09-18 (deferred, see
  `WCSPH_DEFAULT_CLOSEOUT_PLAN.md` item H).

  **TODO:** re-run `englishWedge` dp=0.01 under the current default combo
  (`english2025`+`fourtakas2019`+`symplecticEuler`) to check whether the
  base-corner residual (RMSE 0.0431 last measured) has moved at all, before
  deciding whether this still needs its own fix.

  **Re-run 2026-09-28** (default combo, dp = 0.01, t = 4 s, video:
  `scripts/out_looseEnds/baseline/englishWedge/`): stable, bulk / near-wall /
  wedge-face bands pass, but the **base corners got worse — RMSE 0.0674 (max
  0.096)**, against 0.0431 under the old defaults; the apex just fails too
  (0.0312 vs 0.03). The frames show a weak spurious circulation along the
  wedge faces (|v| up to 0.04 m/s) and a disturbed lattice around the apex.
  So this still needs its own fix; which part of the default flip moved it
  (english2025 / fourtakas2019 / symplecticEuler) is not isolated.

  **Correction (2026-09-28):** 0.0431 was not the last good number. On
  2026-09-12 (`DELTASPH_VALIDATION_PLAN.md` §5.13, hybrid ghost placement)
  this case measured base corners **0.0178**, faces 0.012, apex 0.008 — so
  today's 0.0674 / 0.0393 / 0.0312 is a 3-4x regression since then. Knob A/B
  (revert one of today's defaults at a time to the 09-12 configuration —
  `ramped` mDBC density, `deltaSPH` DDT, RK4, `constant` walls — then all
  four): `scripts/run_wedgeKnobAB.sh`. dp = 0.01, t = 4 s, RMSE / rho g H:

  | run | faces | apex | base corners | near wall/bed | settled KE |
  |---|---|---|---|---|---|
  | today's defaults | 0.039 | 0.031 | **0.067** | 0.057 | 7.9e-6 |
  | `ramped` mDBC density | 0.043 | 0.031 | 0.078 | | |
  | **`deltaSPH` DDT** | **0.014** | **0.013** | **0.020** | **0.016** | 3.0e-6 |
  | RK4 | 0.037 | 0.037 | 0.054 | | |
  | `constant` walls | 0.057 | 0.022 | 0.091 | | |
  | all four reverted | 0.015 | 0.011 | 0.020 | 0.016 | 2.3e-5 |

  **The DDT flip alone (`fourtakas2019`, default since 2026-09-18) explains
  the regression** — at the wedge and in the whole near-wall band. Today's
  code with the 09-12 configuration reproduces the 09-12 numbers. The DDT
  already runs fluid-to-fluid only (the memory note saying otherwise is
  stale). Leading hypothesis: Fourtakas diffuses toward an *analytic*
  hydrostatic profile, and next to a wall — a one-sided, fluid-only stencil —
  any departure of the real density from that profile becomes a net flux;
  the δ-SPH term subtracts a renormalised local gradient instead and
  vanishes for any locally linear field. Next (user suggestion): the same
  runs x-periodic (`--semiPeriodic`, no side walls) to separate the wedge
  from the wall/free-surface corners, where circulation shows even under
  `deltaSPH`.

  **x-periodic (no side walls):** `deltaSPH` faces 0.006 / apex 0.003 /
  corners 0.008 / near wall 0.009 — the side walls' free-surface corners
  account for about half of its remaining error; `fourtakas2019` stays at
  0.048 / 0.037 / 0.075 / 0.061 without them. **Root cause: a sign bug in the
  `fourtakas2019` hydrostatic correction** (`wp_densityDelta.py`) — with
  x_ij = x_i - x_j it *added* rho^H_j - rho^H_i instead of subtracting it.
  Invisible in the bulk (the Laplacian of a linear field is zero), twice the
  plain-difference error on every truncated stencil. On an exact hydrostatic
  block: edge rows 5.33 = 2x `densityOnly` before the fix, 1e-4 after.
  Fixed 2026-09-28 (`68a9a6d`, `tests/test_fourtakasHydrostatic.py`).
  **Consequence:** `fourtakas2019` has been the default DDT since 2026-09-18
  (`WCSPH_DEFAULT_CLOSEOUT_PLAN.md` item E, chosen as the tightest dam-break
  match) — every validation on it since then used the buggy term, so the
  default-combo decision needs re-checking with the fixed term (user).

(The H/Δx≈72-160 resolution hole that used to share this item was resolved
2026-09-18: [RESOLVED_PROBLEMS.md](docs/historic_plans/RESOLVED_PROBLEMS.md).)

  **After the fix (dp = 0.01, t = 4 s, RMSE / rho g H):**

  | run | near wall | faces | apex | base corners |
  |---|---|---|---|---|
  | buggy `fourtakas2019` (default until the fix) | 0.057 | 0.039 | 0.031 | 0.067 |
  | fixed `fourtakas2019`, walled | 0.017 | 0.015 | 0.013 | **0.018** |
  | fixed `fourtakas2019`, x-periodic | 0.008 | 0.005 | 0.002 | 0.008 |
  | `deltaSPH` DDT, x-periodic | 0.009 | 0.006 | 0.003 | 0.008 |

  Every check passes; the base corners are back at the 2026-09-12 level
  (0.0178). About half of the walled residual comes from the side walls'
  free-surface corners (the circulation there, the §1/§8 wall/free-surface
  family), not from the wedge. **Resolved.**

## OPEN_PROBLEMS §11 — probe_contactLine delta-SPH toys: pinned dt is overridden — RESOLVED 2026-09-28

**What it is (minor, probe-level):** the probe sets `soundSpeed` and
`targetDt = 0.3 dx / c0` for the WC toys, but the per-step adaptive
`computeTimestep` hook overrides dt (the runs still step at ~5.9e-4,
acoustic Courant ~0.8). Stable with `symplecticEuler`, so results stand, but
the toys do not run at the Courant number the probe claims.

**Next step:** make the toys' timestep hook return the pinned dt (or report
the achieved Courant number) before relying on dt-sensitive toy results.

**Fix (2026-09-28, `c72e1ea`):** the toys' timestep hook returns
`min(adaptive, targetDt)`. Checked: film toy, delta-SPH, nx=32 runs at
dt = 2.117e-4 = 0.3 dx / c0 every step (was ~5.9e-4, Courant ~0.8).

## OPEN_PROBLEMS §12 — Render thread + grid-interpolated plots crash or hang CRKSPH cases — RESOLVED 2026-09-28

**What it was:** with `--plot` and no live window, the runner renders frames on
a worker thread (`asyncPlot`, SMALL_PROBLEM_PERFORMANCE.md §9.4/§12.4).
`kelvinHelmholtz`, `rayleighTaylor` and `triplePoint` crashed within 10 steps
(exit 139, CUDA "operation not supported on global/shared address space",
raised at the next event record) or hung; `yeeVortex` did not. All three have
a `gridResolution=1024` density panel: the SPH interpolation onto that grid
runs warp kernels inside the plot hooks, on the render thread, concurrently
with the step's own warp work. Warp's current stream and module loading are
per process, not per thread. Same failure on clean `main` (pre-existing; found
by the OPEN §6.3 re-runs). `--no-asyncPlot` ran clean (40 steps, 5.8 s).

**Fix:** `cases/plotting.py:particlePlot` marks hooks with a grid panel
`usesWarp`; `runner.py:_setupPlot` keeps those on the loop thread. Pure-GL
plots keep the render thread. `tests/test_runner.py::
test_gridInterpolatedPlotsStayOffTheRenderThread`. The headless loop-thread
path still renders through EGL (the default vispy backend opened a black
window).

**Sibling, same day:** Marrone 3.1 δ-SPH crashed (exit 139) with "operation
not permitted when stream is capturing": warpSPHPlotting's `filterState` did a
boolean-mask index (a sync) on the render thread while the loop thread was
capturing a CUDA graph. `utils/cudaGraph.py:CAPTURE_LOCK` is now held by every
capture and by every render job (`0e15883`).

## OPEN_PROBLEMS §10 — Runner divergence detection misses bounded-NaN-free blowups — RESOLVED 2026-09-28

**What it is:** `run()` reported `diverged=False` for delta-SPH toy runs that
had reached vmax 5e4 with rows at rho = 0 and pressures ~1e20
(`scripts/out_contactLine/*_a0`, `*_u0`, 2026-09-24): the check only fires on
non-finite values. `stallDtSteps` only fires on dt pinned exactly at minDt,
and `stallProgress` (sim-time progress) only when time stops advancing -- an
explosive but finite run passes all three.

**Next step:** a relative velocity bound (e.g. vmax against the case's own
`referenceVelocity` / sound speed -- a multiple of c0 is physically
impossible in a WC run) or a kinetic-energy growth check, reported as
diverged; decide whether it stops the run or only flags it.

**Implemented 2026-09-28 -- flags, does not stop** (user decision: a person
watching gets a warning, checks the dynamics, and stops the run if it stalls).
`runner/velocityAlarm.py`: when `max |v|` exceeds `velocityAlarmFactor`
(default 100) x the expected velocity, the runner prints a
`[warpSPH] VELOCITY ALARM` line (again at every further 10x, and when it
clears), records the event in `RunResult.velocityAlarms`, exposes it on
`ctx.scratch['velocityAlarm']`, and calls `run(..., onVelocityAlarm=fn)`
(returning `True` stops the run). Two declarative reactions:
`velocityAlarmPlotInterval` (frames every N steps while active) and
`velocityAlarmStopRatio` (hard ceiling); probes get all of it, plus
`stallProgress 1e-3`, from `scripts/_runWatch.py`. Usage: README "Watching a
run"; run rule: `CLAUDE.md` item 3.

**Checked on the real failure (2026-09-28)**, ACSPH Marrone 3.1 nx=70 (the
contact-line runaway of §8, `probe_deltaSPHMarrone.py --scheme
artificialCompressible --nx 70 --tLimit 1.0`, video +
`scripts/out_velocityAlarmDemo/`): raised at step 1211, t = 0.6893 (105x,
particle uid 334 already below the floor, y = -0.138) -- the step the flow
fails at; escalated at 1222 (1.2e3x, dt 1.7e-6), 2125 (1.3e4x, dt 3.5e-8) and
2135 (1.9e5x, dt at the 1e-8 floor); from step 1211 a frame every step.
`stallProgress 1e-3` stopped it at step 2204 (9.2e-4 s of sim time over 1000
steps), 487 s wall in all. Why 1e-3 and not 1e-4 as the probe default: in a
first try (1e-4, stopped by hand at ~step 1980) the run sat at dt 2.6e-7 for
~750 steps, a rate at which 1e-4 never trips (~60 h to tLimit had it stayed
there); here dt kept falling, so 1e-4 would have tripped too, only later.
1e-3 trips on any hover below ~1e-6 s of dt at tLimit 1, while healthy runs
advance >= 1e-2 of tLimit per 1000 steps. Expected velocity: `spec.velocityScale`, else
the case's `ctx.velocityScale` (`dambreak`: `referenceVelocity` or
sqrt(2 g H); `hydrostaticColumn`: sqrt(g H)), else the largest of initial
`|v|max`, compressible initial sound speed, WC `c0/10`, ACSPH `uChar`; the
banner's `watch` row says which. Same single host read as the non-finite check
(`max` propagates NaN/inf), so no extra sync per step and the simulation is
unchanged. Still open: a case run under `divergenceFree` from rest without a
declared scale (e.g. `columnCollapse`, `staticBlob`) has the alarm off, and
stopping remains the stall watchdogs' job (`stallProgress`, `stallDtSteps`).

## OPEN_PROBLEMS §5 (part) — the H/Δx≈72-160 Marrone 3.1 resolution hole — RESOLVED 2026-09-18

**RESOLVED, 2026-09-18** (moved out of this list): the **H/Δx≈72-160
resolution hole** (`DELTASPH_VALIDATION_PLAN.md` item 4b) — every Marrone
3.1 config used to blow up specifically at nx=120 (H/Δx=72), in t*≈5.5-6.3,
"regardless of scheme." Rechecked under the new default combo
(`english2025`+`fourtakas2019`+`symplecticEuler`) at the exact same
resolution, run to t*≈7.68 (past the old failure window):
`diverged=False`, maxVel peak 7.41 (moderate, decaying), density
[0.982, 1.039] — no divergence. Not root-caused *why* (which of the
combo's components fixed it, or whether it's a combination, wasn't
isolated), but the practical symptom is gone. `WCSPH_DEFAULT_CLOSEOUT_PLAN.md`
item H has the full result.

## OPEN_PROBLEMS §17 — Read-Hayfield's entropy dissipation did not conserve total energy — RESOLVED 2026-10-01

**Symptom** (found 2026-09-30 by the AV_PLAN M0c baseline): with the §16 heating fix in, `none` and Cullen-Dehnen conserved
energy to round-off, but `ReadHayfield2012` drifted 1.8e-3 (Sod), 6e-3 (Sod 2D), 1.5e-2 (Sod 3D), 1.3e-2 (Sedov), 1.6e-3 (KH),
and **+23 % on Noh at `C_q = 2`** (post-shock density -16 %).

**Cause.** Read & Hayfield (2012, MNRAS 422, 3037) Eq. (33) is

    dA_i/dt = sum_j (m_j / rho_ij) alpha_ij v_sig^p_ij L_ij [ A_i - A_j (rho_j/rho_i)^(gamma-1) ] K_ij

with the density ratio *inside* the bracket (checked on the rendered page, 2026-10-01). With `u = A rho^(gamma-1)/(gamma-1)` the
pair energy transfer `m_i rho_i^(g-1) dA_i + m_j rho_j^(g-1) dA_j` is then exactly antisymmetric
(`[A_i rho_i^(g-1) - A_j rho_j^(g-1)]` against its negative). The code had `(A_i - A_j) * rho_ratio * K_ij`, which cancels only
when `rho_i = rho_j`. A second, smaller asymmetry: `K_ij` used `h_i` alone; the paper's `K_ij` is a symmetric kernel gradient,
now the mean over both supports.

**How it was found.** `scripts/probe_monaghanEnergy.py --switch ReadHayfield2012` now reports the net `sum m dudt_diss`
(extension of the §16 probe): Sod -4.9e-4 and Noh +0.26 before, -1e-9 and +4e-9 after. First check was the paper's own
statement ("explicitly conserves energy"), which pointed straight at the equation.

**Result** (baseline M0d, `docs/av/av_baseline_2026-09-30.md`; `none` / C&D bit-identical to M0c): energy drift Sod 9e-7, Sod 2D 1.6e-5,
Sod 3D 2.6e-5, KH 3e-7, Sedov 5.9e-4 (the same as `none` / C&D, so not R&H's), Noh `C_q = 2` 1.7e-6 with post-shock density
-0.11 %. 14 R&H pairs re-run: bit-identical. Regression test `test_read_hayfield_entropy_dissipation_conserves_energy`.
**Follow-up the same day (user review):** the pair loops themselves were host torch code taking raw `x_i - x_j` (wrong across a periodic boundary) with a
hand-copied kernel derivative that silently fell back to Wendland2; moved to warp kernels (`wp_readHayfield.py`) using the core distance / kernel-gradient
functions (baseline M0e). Note for later: the bracket form dissipates `A rho^(g-1)` (a u-like variable) rather than `A`, so in an isentropic flow with a
density gradient the term is not zero by itself; the pressure limiter `L_ij` is what keeps it quiet there.

## OPEN_PROBLEMS §5 follow-up — re-validation of the default combo after the `fourtakas2019` sign fix (2026-09-29)

**Re-validation done 2026-09-29 (recommendation: keep the default; the decision is the user's):** the default combo (english2025 +
fourtakas2019 + symplecticEuler) re-run with the fixed term against the
`deltaSPH` DDT, everything else default (batch
`scratchpad/run_overnight_batch_2026-09-28.sh`, tables in
`scripts/out_overnight_2026-09-28/SUMMARY.md`, videos looked at):

- **Marrone 3.1**: P1 plateau (0.46 delta+, 0.83-0.88 PST-off — the PST-off
  value is the old on-wall-probe extrapolation, In1/Shepard read 0.62) and P2
  peak (0.23-0.33) are **identical between the two DDTs and to the pre-fix
  09-26 baseline**; the sign bug never moved them. Checks passed: delta+ nx67
  x3 seeds fourtakas 27/27 vs deltaSPH 24/27 (fails are P2 "quiescent"
  flags = a flier crossing the probe); nx134 8/9 vs 7/9; PST-off 8/9 vs 7/9.
  Bulk density: PST-off [0.992, 1.007] vs [0.977, 1.023]. **One signal against
  fourtakas:** delta+ nx67 max|v| 16.1 / 11.4 / 19.7 vs deltaSPH 10.3 / 10.1 /
  10.2 (buggy baseline 10.9) and wider rho extremes ([0.81, 1.20] vs
  [0.93, 1.08] worst seed); the fast ones are ceiling-pinned isolated
  particles at t\*~5.3-5.7 (the §1 kick, rhoMin/rhoMax and vmax coincide in
  time), present in both DDTs' frames. Opposite at nx134 (23.0 vs 30.1) and
  PST-off (10.2 vs 21.4). n=3 seeds: not a separation.
- **Marrone 3.4** nx256 (delta-SPH and delta+): all 6/6 PASS, both DDTs;
  penetration 0.64-0.76 dx; fixed fourtakas indistinguishable from the buggy
  baseline (vmax 40.9 vs 39.5, same rho extremes 0.63/1.8). Pointwise rho max
  is higher with fourtakas (1.79 / 1.58 vs 1.40 / 1.33; P99 bulk band equal).
- **sloshingTank** t=7: both DDTs survive; Gaussian-10 ms wall-sensor peaks
  2.9 / 3.8 / 7.3 kPa (fourtakas, fixed) vs 3.9 / 4.8 / 5.3 (deltaSPH) vs
  2.4 / 3.5 / 5.0 (buggy baseline), all inside the measured 2.2-13.1 kPa band;
  rho extremes [0.82, 1.32] (fourtakas) vs [0.73, 1.50] (deltaSPH). Raw
  sensor spikes 25-50 kPa in both (known).
- **Gallery**: 33/34 ran; weakly-compressible examples healthy (impact,
  LDC, open-flow, moving obstacle etc. look at least as good as the
  2026-09-13 stored images, which pre-date the current defaults); the one
  failure is compressible and unrelated, see §15.
- **Verdict for the user:** no evidence to change the default; fixed
  `fourtakas2019` is equal or slightly better on the Marrone/sloshing
  checks and density band, with one mild counter-signal (nx67 delta+ kicks
  in 2/3 seeds) that belongs to §1, not to the DDT. A 10-seed nx67 delta+
  A/B would settle that counter-signal if wanted.
  **2026-09-29: counter-signal explained** ([CEILING_STICKING_PLAN.md](CEILING_STICKING_PLAN.md) §3):
  those kicks are symplecticEuler's explicit-midpoint (ρ, v) update
  amplifying the wall-contact acoustic mode of a ceiling-pinned cluster;
  with `timeCentredContinuity` the three seeds read max|v| 10.1 / 9.9 / 9.9
  and ρ [0.96, 1.03], tighter than either DDT before.

## OPEN_PROBLEMS §6 item 3 — sampler mass fix on the other compressible cases (closed 2026-09-28)

**Done 2026-09-28** (`scripts/run_looseEndsBaseline.sh baseline`, video,
   `scripts/out_looseEnds/baseline/`): all eight reach their tLimit, no
   velocity alarm, total energy conserved to the printed digits in the
   compressible ones (kidder is driven, RT's total excludes potential energy).
   Frames match the stored references where comparable (squarePatch vs the
   2026-09-18 gallery, kelvinHelmholtz, rayleighTaylor plumes at t≈3.8). The
   first pass crashed three CRKSPH cases — not the sampler fix, the render
   thread (§12, resolved). Item 3 closed; items 1 and 2 stay optional.

## OPEN_PROBLEMS §7 — missing shear-carrying laminar viscosity term (Morris et al. 1997) — RESOLVED 2026-09-28

**What it is:** the stock velocity-diffusion term
(`computeVelocityDiffusion`'s `inviscid=False` branch, `wp_viscosityDelta.py`)
is `mu_ij * gradW` with `mu_ij = (v_ij . x_ij)/|x_ij|^2` — a scalar built by
contracting the relative velocity along the separation vector `x_ij`, not the
full `v_ij` vector. That makes it a normal-projected diffusion: it damps the
approach/separation component of relative velocity but carries no
tangential/shear stress at all. A real Morris et al. (1997) laminar viscosity
term needs the full vector Laplacian, which neither `viscidNu` nor
`viscosityParams` currently has.

**Why it matters:** `hydrostaticColumn`'s free-slip side walls leave a
bounded, undamped limit-cycle slosh (documented as non-fatal — DFSPH_FINDINGS.md
§1.12/§1.20). Switching to `wallBC=noSlip` + `viscidNu` through the existing
(normal-projected) term does bound the slosh KE (~4x down) and hold the
hydrostatic gradient, but roughens the free surface
(`embeddedMinDensity` 0.94 -> 0.60, `|v|max` spikes to ~3.7-4.1) — a no-slip
mirror through a normal-only diffusion term adds noisy *normal* wall damping
with no tangential component, not the physically-correct shear stress a real
no-slip wall should apply. A hand-rolled full-vector Brookshaw Laplacian
(DFSPH_FINDINGS.md Part 39, since removed in the Part 41 cleanup) *did* hold
the surface (embMin 0.94-0.97) while damping the slosh at the same time,
confirming the mechanism — the module layer just doesn't have it as a real,
supported option today.

**Why it's not a quick fix:** it needs a new kernel term (the full `v_ij`
vector, not the scalar `mu_ij` reduction), wired as a `DiffusionParameters`
option, gradcheck'd (it touches a `@wp.kernel`), and given its own `deltaSPH`
regression pass so it doesn't silently change WCSPH's diffusion behaviour
too. Half of the groundwork already landed (2026-09-05, `ACSPH_PLAN.md` step
5 — `computeVelocityDiffusion(approachOnly=False)` lifts the approach-only
clamp, turning the `inviscid=False` branch into the Monaghan & Gingold
(1983) Laplacian, De Courcy et al. 2024 Eq. (25), gradchecked in all four
`inviscid` x `approachOnly` combinations, default unchanged) — what remains
is the actual full-vector term, a new kernel, not a flag flip.

**Concrete next step:** implement the full `v_ij` Morris Laplacian as a new
`DiffusionParameters`-wired option alongside the existing `viscidNu` scalar
term, gradcheck it, and re-run the `hydrostaticColumn` `wallBC=noSlip` A/B to
confirm it reproduces Part 39's numbers (embMin held, KE damped) through the
stock machinery instead of the since-removed bespoke path.

**2026-09-28 — implemented, A/B pending.** `diffusionParams.viscousTerm =
ViscosityTerm.morris1997` (opt-in; default `monaghanGingold` unchanged),
gradchecked, `tests/test_morrisViscosity.py`. **Correction to "carries no
tangential/shear stress at all" above:** on a divergence-free shear wave
`v = (sin ky, 0)` in the bulk, the projected term reproduces `nu lap(v)` too
(4.8 % / 4.3 % error at n = 32 / 64, vs Morris 2.6 % / 1.8 %) — its
`K = 2(d+2)` normalisation makes it a consistent Laplacian away from
boundaries. So whatever Part 39's term did better has to come from the
truncated supports at walls and the free surface, which is what the
`hydrostaticColumn` A/B (`scripts/probe_morrisNoSlipColumn.py`) measures.

**A/B, 2026-09-28** (`scripts/probe_morrisNoSlipColumn.py`, `divergenceFree`
— `iisph` no longer holds even the free-slip arm, §13). First pass: projected
+ no-slip was already clean on `divergenceFree` (embMin 0.956, slope 1.003 —
Part 41's embMin 0.60 was `iisph`-specific); Morris + no-slip showed a
near-wall band at |v| ~0.17 whose velocity **flipped sign every step** while
the particles stayed put — an explicit-diffusion instability. Cause: the
shared DFSPH timestep (`kolmogorovIncompressibleTimestep`) divided the viscous
limit by `kernelScale` once instead of squared, allowing nu dt / hs^2 = 0.24
(hs = smoothing length) instead of Morris' 0.125. Fixed; at the corrected
limit Morris + no-slip settles the column to rest (|v|max <= 0.01, near-wall
and bulk rows ~1e-4 by t = 0.2). The projected term tolerated the old limit
because it couples more weakly. Final table (`scripts/out_morrisNoSlipColumn/SUMMARY.md`,
`divergenceFree`, nx=128, 1200 steps, tail = last quarter, corrected dt):

| arm | \|v\|max mean | KE mean | embMin | slope |
|---|---|---|---|---|
| free-slip, nu = 0 | 0.541 | 3.1e-4 | 0.889 | 0.966 |
| no-slip, projected | 0.0127 | 1.5e-7 | 1.000 | 1.004 |
| no-slip, Morris | 0.0055 | 4.3e-8 | 1.000 | 1.006 |
| free-slip, Morris | 0.171 | 5.3e-5 | 0.982 | 1.006 — see §14 |

**Verdict:** with a no-slip wall both terms now bring the column to rest;
Morris damps ~3.5x harder (KE) with smaller peaks. The Part 39 gap is closed
through the stock machinery. Morris stays opt-in (`viscousTerm`); whether to
make it the default for viscous no-slip cases is a separate decision (user).
**Resolved** except that decision — the free-slip + viscosity row is §14. `DFSPH_IMPROVEMENT_PLAN.md`'s
ranked-queue item 1, `DFSPH_FINDINGS.md` §1.14 (both now retired to
`docs/historic_plans/` — the incompressible/DFSPH track itself reached a
stable, documented recommendation (`divergenceFree` default, `band2018pb` as
a deliberate trade-off) and this is the one item that survived it).

## OPEN_PROBLEMS §15 — CRKSPH lattice-shock blow-up — RESOLVED 2026-09-30 (residual and follow-ups stay in OPEN_PROBLEMS §15)

**Status: blow-up RESOLVED 2026-09-30 (cause found, fixed, regression test); a benign residual and follow-ups
below.** The mechanism sections further down were written before the cause was known; the "what it is not" list
there stays valid (kernel, `n_h`, limiter constants, gradB, CUDA graphs were all innocent).

**Cause (2026-09-30).** The blow-up is the CRK **artificial viscosity** on a near-coincident approaching pair,
not the pressure force. `modules/crk/accel.py` / `dudt.py` regularised the switch as
`mu = v_hat . eta / (eta . eta + 1e-7 h^2)` with `eta = x/h` **dimensionless**: the regulariser was ~1e-11, so
`mu ~ v/|eta|` was unbounded as a pair closed and `Q = rho (-C_l c mu + C_q mu^2)` gave a pair acceleration
~1e4-6e4 in one right-hand-side evaluation (trapped: the second stage of step 163, pair at r = 0.0026 dx,
|a_visc| = 6.2e4 vs |a_pressure| = 0.12; then u -> -5e5). Frontiere et al. 2017 Eq. (69) has
`eps^2 = 1e-2` in eta^2 units (their standard parameter set). Fix: exactly that (`scalar_t(1.0e-2)`), so the
viscosity vanishes smoothly as `eta -> 0`.

**Result.** Sod 2D same-lattice: blow-up at step 164 -> clean to t = 0.6 (268 steps, total energy 0.35385
exact); the shipped `triplePoint_equalSpacing.py`: blow-up at step 88 -> t = 10.0 in 4700 steps, energy exact
(48.203), vortex roll-up as in the paper's Fig. 25 (videos in `scripts/out_crk2d/examples/export/`); a regression
test (`tests/test_crkCoincidentPair.py`) fails on the old code (non-finite state) and passes now; gradcheck
(`scripts/gradcheck_crk.py`) and `tests/test_physics.py` etc. pass (110). A/B of the change over the 13
compressible cases of the limiter-plan harness (old vs new, `scripts/out_crk2d/sweep/`): Sod L1 rho -0.1 %,
Sedov 0.0 %, Noh -1.7 %, Kidder -2.7 %, linear wave -3.6 %, Yee -0.9 %, hydrostatic max|v| +0.5 %, KH rebound
-34 %, RT +0.7 %, nothing diverges; the one worse number is Gresho (+12-15 % velocity error, KE spin-up
+7.3 % -> +8.2 %), a case CRKSPH_LIMITER_PLAN O4 shows is a knife-edge (2 % threshold shift = 7x spin-up).

**Tested and refuted on the way.** (a) The sampling ratio is not the cause (it only decides when). (b) Frontiere
Eq. (76)'s multi-material mass rule for the density (`m_ij = m_i` across materials; our kernel always uses `m_j`,
`warpSPHCore/crk/crk_density.py`; the core has no material tag) was built by linearity and tried: it makes Sod 2D
blow up *earlier* (step 83), and the user reports it degraded the hydrostatic 2D box, so it stays out. (c) The
equal-spacing hydrostatic box (static mass jump at uniform pressure) is essentially perfect under CRKSPH
(max|v| 1e-5 over t = 3), so the density treatment across a jump is not the problem; the mass-jump trouble is
dynamic, at start-up.

**What it is.** `compressible/14-triplePoint/triplePoint_equalSpacing.py`
(CRKSPH, nx 256, shipped preset, `dev` @ a28aeb1) is clean for 86 steps
(max|v| 1.2, energy exact) and one step later |v| ~ 1e5, dt collapses, NaN
in the hash-map build (gallery `F_gallery` rc=1). The same thing happens in
the **Sod 2D slab under CRKSPH on the same-lattice sampling** once it runs
past the shipped tLimit (0.15): a 4000-particle reproducer.

**Replicate** (`scripts/probe_triplePointPileup.py`, prints dt / max|v| /
extrema per step and the smallest nearest-neighbour spacing; dumps the last
clean + blown snapshot to an npz and stops at the blow-up):

```
python scripts/probe_triplePointPileup.py --nnEvery 8                    # triple point, cfl 0.3: blows at step 88
python scripts/probe_triplePointPileup.py --case sod2d --nnEvery 20      # Sod 2D, CRKSPH, --no-equalMass: blows at step 164
python scripts/probe_triplePointPileup.py --case sod2d -- --equalMass    # control: clean to t = 0.6 (269 steps)
python scripts/probe_triplePointPileup.py --case sod1d                   # control: clean to t = 0.6 (2157 steps)
python scripts/probe_triplePointPileup.py --nnEvery 8 -- --cflFactor 0.05 --nSteps 600   # survives, same pile-up
```
Everything after `--` goes to the case CLI. The shipped-preset views:
`examples/compressible/14-triplePoint/triplePoint_equalSpacing.py` and
`scripts/out_sodCRKSPH_2026-09-29/run.sh` (Sod 1D / 2D equal-mass / 2D
equal-spacing under CRKSPH at the shipped tLimit 0.15 — all three finish
there: 539 / 67 / 67 steps, too short to reach the collapse; videos under
`export/`, git-ignored).

**Mechanism (measured).** The nearest-neighbour spacing on the light side of
the contact starts shrinking around step 16 (triple point) and roughly halves
every 8 steps; by step 87 there are 228 coincident pairs (456 particles), one
pair per row, both particles *light* (m = 6.9e-5, same v, different u), at
x = 1.3125 and its periodic mirror 12.6875. The blow-up is the step a pair
reaches r = 0 (kernel gradient -> 0; the CRK-corrected gradient keeps a
W(0)(grad A + A B) term). CRK terms are benign until then (cond(m2) <= 2.3,
|B|h <= 2.9, no singular-matrix warning). Sod 2D shows the same thing:
coincident light pairs (m = 2.5e-5) at the contact, x = +-0.7303, with the
dense side (m = 1e-4) expanded behind it.

**Where it forms / what drives it.** Always at the contact discontinuity, on
the light side. The particle-spacing jump across the contact (in the
compressed direction) is what the sampling controls: with the same lattice
on both sides the dense-side gas is 4x (Sod) / 8x (triple point) heavier per
particle, so its x-spacing at the contact ends up ~2.5x the light side's;
with equal-mass sampling (light side coarser from the start) it is ~1.6x and
the run is clean. Triple-point A/B on rho_II: 0.5 / 0.25 / 0.125 blow at step
194 / 136 / 89 -- **the sampling ratio sets *when*, but even rho_II = 1.0
(equal masses everywhere) still blows (step 208, pairs at the entropy
contact x ~ 1.41)**, so the ratio is an amplifier, not the whole cause.

**What it is not (A/B'd, 5-10 s each):** kernel or n_h (B7, Wendland4,
QuinticSpline, n_h 2-5 all blow, at different steps); the CRK viscosity limiter
(`enableCRKLimiter=0`, eta_crit 0 / 0.1 / 1 -- none cure it, eta_crit = 1 is
earlier); the 09-26 gradB fix (reverted in a scratch core: same); CUDA graphs /
pipelined outputs; `calibrateNormalization`.

**Not a clean regression.** The 08-13 code (stored gallery image) and the
08-31 / 09-04 core pairs develop the same pile-up as a hot cluster (rho 8,
umax 10-16, h down to 0.014, c_s 5) that shrinks the adaptive dt to ~3e-4 and
keeps pairs from touching; the 09-04 pair blew at t = 0.248 in one run and
survived to t = 1.9 in a re-run. From 09-19 on it blows every time.

**A smaller dt only hides it.** `--cflFactor 0.05` survives to t = 1.87 and
its state through t = 0.29 matches cfl 0.3 (rho_max 3.37 vs 3.35, umax 3.07,
h_min 0.050), but it carries the same pile-up (484 particles < 0.1 dx from a
neighbour, min separation 0.009 dx vs 0). cfl 0.15 blows at step 168. The
compressible dt is `cfl * h_min / (xi c_s)`: acoustic only -- no bulk/approach
velocity, no separation term -- and h adapts to density, so it cannot see a
column collapse.

**Next, when picked up:** the pair force between the first two light columns
at r -> 0 (CRK gradient at small separation) and the first-light-column
acceleration (a_i ~ m_heavy (P_i/rho_i^2 + P_j/rho_j^2)); then test a
mass-jump-smoothing remedy on the Sod 2D reproducer (cheap: 4000 particles).
Interim for the example (user's call): leave failing, ship `--cflFactor 0.05`
(runs but carries the pairs), or drop the variant (equal-mass is the case's
shipped default and passes). No videos of the collapse itself (crash probes,
~90-160 steps); the frames to compare are the equal-mass gallery output vs a
cfl-0.05 run.

## OPEN_PROBLEMS §18 — Monaghan Sedov energy drift 5e-4 — closed 2026-10-01 as expected behaviour

Found while closing §17: on `sedov` (3D, nx 40) `none`, Cullen-Dehnen and Read-Hayfield all drifted 5.2e-4 / 5.3e-4 / 5.9e-4, flagged
over the Monaghan budget 1e-4 by the report's Group E. Measured (`results/sedovE_*`): the energy is lost in the first ~12 of 864 steps, while
the point-like deposit is released (K 0.002 -> 0.10), and is flat afterwards; the instantaneous RHS is exactly conservative (probed at steps 4 /
12 / 40: pressure force + work cancels to 0, viscous pair to 1e-7, conductivity ~1e-6); the drift converges at second order in dt
(`cflFactor` 0.3 / 0.15 / 0.075: 5.22e-4 / 1.31e-4 / 3.27e-5); CompSPH on the same case, dt and RK2 conserves to 4.8e-7. So it is the
O(dt^2) energy defect of an RK2 step on a non-compatible `u`/`v` pair, largest when the dynamics are fastest. Closed as expected behaviour of
a converging method (user, 2026-10-01). Why CompSPH's compatible form is immune was not pinned down.

## OPEN_PROBLEMS — the 2026-09-28 batch (done)

**Batch of 2026-09-28 — done** (on branch `dev`; outcomes in each item):
1. §5 / §6.3 / §11 re-runs as a baseline, in the background;
2. §7 Morris shear viscosity (opt-in; DFSPH `hydrostaticColumn` no-slip A/B
   against Part 39, δ-SPH unchanged when off);
3. §8's `detectIsolated` in the shared surface detector (changes a default:
   shifting and the Antuono switch everywhere) — then repeat the step-1
   re-runs plus Marrone 3.1 and sloshingTank as the before/after check;
4. §1 step 1, frozen Antuono mask across RK sub-stages — dev work: the
   flip-rate diagnostic has to be rebuilt first.

## OPEN_PROBLEMS §6 — lattice-density kernel calibration — closed 2026-10-01

The core design (`calibrateNormalization`: scale the kernel by `1/L` so a defect-free lattice measures `rho0`) landed and is verified; the
sampler mass/cell mismatch that made it look necessary was fixed at the source (`sample/regular.py:62`), so the residual it still corrects is
<= 0.04 % on sloshingTank. Two optional loose ends were left; both are now closed:

1. **`calibrateRestDensity`'s `onResidual='raise'` path** was untested. `tests/test_restDensityCalibration.py` now covers raise (default, and the
   refused calibration is not applied), warn, ignore, the tolerance, and an invalid `onResidual`. Writing it showed the residual check reads the
   *median* fluid mass, so only a bulk mass/cell error trips it, not a minority of overridden particles; the docstring now says so.
2. **Wiring the ~131 remaining `OperationProperties(...)` call sites** (gradient / divergence / Laplacian / curl / interpolation) so the `1/L`
   correction applies to every kernel sum, not just the summation density: **decided not worth it** (user, 2026-10-01). It is the theoretically
   consistent thing to do, since the lattice-quadrature offset touches every kernel sum, but the offset is tiny after the sampler fix and the
   change would touch the core operators for no measurable gain. Documented on `SimulationConfig.calibrateNormalization`; if it is ever wanted,
   build the `propsFromConfig` helper (`LATTICE_DENSITY_PLAN.md` §3.9) rather than editing the sites by hand.

Third item (the sampler mass fix on the other compressible cases) is in the entry above ("§6 item 3").

## OPEN_PROBLEMS §20 — R&H Sod contact test fails since the adaptive-support clamp (013a22f) — RESOLVED 2026-10-06

Found 2026-10-06. Passes at d0f4774, fails from 013a22f on (`_clampSupportsToVolume` in
`modules/adaptiveSupport/optimalSupport.py`, `h <= n_h (m/rho)^(1/d)` on fluid rows): R&H's contact entropy overshoot
3.43% vs NoneSwitch 3.24% (the pressure overshoot criterion still passes). The clamp is still needed by the
compressible walls: without it the shockReflection outermost row is -10% in p again (frozen 1.8x h, a coincident
pair) vs -0.9% with it. Options: keep the clamp and re-baseline the test (the margin is 6% of a 3% overshoot), or
find why the clamp shifts the contact. Your call.

**Wider than the one test (2026-10-06, AV_PLAN Phase 2):** the clamp moves *every* compressible number in the AV
baseline. The smoke profile at HEAD vs the M0e reference (`results/av_smoke_ref_M0e`): 30/30 pairs differ, including
`none` (Sod contact P spike up to 90 % relative, Sedov E0 recovery, Noh post-shock rho 11 %). So the M0e baseline in
AV_PLAN no longer describes the code. If the clamp stays, AV_PLAN needs a re-taken baseline (M0g); a fresh
none / C&D reference at HEAD is being taken for Phase 2 (`results/av_M2_ref_*`).

**Resolution (2026-10-06).** Measured with a new `SimulationConfig.supportVolumeClamp` ('always' / 'walls' / 'off'):
- clamp off at HEAD is bit-identical to the pre-clamp M0e baseline (smoke 30/30, full 28/28 pairs for none / C&D / Q), so
  the clamp is the only change to pure-fluid runs since M0e;
- clamp on vs off: shocks better (Sod contact P spike -46..-71 %, Sedov 3D peak none 2.53 -> 2.75), smooth flow worse
  (inviscid linear wave error 0.00053 -> 0.0033, Gresho L1 +11-14 % and angular-momentum loss +31-85 %, C&D KH amplitude
  -18 %; with alpha fixed at 1 Gresho / KH move <= 2 %);
- why (`scripts/probe_supportClampBinding.py`): in 2D the clamp binds on ~100 % of rows by ~2 % (Owen's equilibrium sits
  above the volume bound -- a table bug, fixed in SUPPORT_SOLVER_PLAN), i.e. it turned h into h(rho) everywhere; in 1D only on
  over-grown rows (31 %, up to 1.32x) -- OPEN_PROBLEMS §21. (Rho is re-summed at the final h by every scheme, so this was
  the h definition at work, not a rho-h mismatch.)
- shockReflection near-wall rows: 'walls' identical to 'always' (outermost -0.9 % in p), 'off' -10.2 %.
User rule (2026-10-06): if the clamp degrades fluid behaviour it is off for pure-fluid cases. Default is now 'walls'; the
R&H test passes again; the AV baseline M0e describes the code again.

## OPEN_PROBLEMS §21 — Owen adaptive support: frozen / over-grown h, 2D offset — RESOLVED 2026-10-06

Found 2026-10-06 while resolving §20 (`scripts/probe_supportClampBinding.py`, clamp off). Two properties of the Owen
support solve (`modules/adaptiveSupport/optimalSupportOwen.py`) that `supportVolumeClamp` papers over:
1. **One-way ratchet** (COMPRESSIBLE_WALLS_PLAN 2026-10-05): the relaxation grows h while a kernel is under-sampled but
   never shrinks a well-sampled one, so h freezes over-large after a transient. Wall outer rows: 1.8-2x (shockReflection
   -10 % in p without the clamp). Pure fluid, 1D Sod at t 0.056: 31 % of rows above the bound, up to 1.32x at the shock /
   contact -- which is why the clamp halved Sod's contact pressure spike (none 0.060 -> 0.026) and raised the Sedov 3D peak
   (2.53 -> 2.75).
2. **2D offset:** Owen's well-sampled equilibrium is h = 1.02 n_h (m/rho)^(1/d) on ~100 % of rows (Gresho, KH), so a clamp
   at the plain volume bound caps every row -- in effect h = h(rho) everywhere. (Correction 2026-10-06: the schemes re-sum
   rho at the final h, so it is not a rho-h mismatch; the cause of the offset was a table bug, fixed in SUPPORT_SOLVER_PLAN,
   and h = h(rho) itself is what costs shear flows there.) Inviscid linear wave error 6x, Gresho angular-momentum loss +30-85 %, KH growth
   -18 % (C&D) with the clamp on.
The principled fix is in the solve (let h shrink, e.g. a two-sided relaxation or a Newton step on the h-rho constraint),
not a clamp; with that, `supportVolumeClamp` could go. **Being worked: [SUPPORT_SOLVER_PLAN.md](SUPPORT_SOLVER_PLAN.md) (2026-10-06).**

**Resolution (2026-10-06, docs/historic_plans/SUPPORT_SOLVER_PLAN.md).** No ratchet on a lattice (h returns from 0.5x / 2x). The 2D / 3D
offset (h 2 % / 0.9 % too large) was `wp_psi.py`'s psi_H table: a continuum shell approximation, exact only in 1D, while the lattice
sum it also computed went unused; `owenTable='lattice'` (default) fixes it, new AV reference M0g (moves <= 6 %). Owen's
over-growth on the dense side of a density jump (+12 % vs h(rho)) is a property of neighbour-count h (Price 2012 §2.3), kept
(user): tying h to rho everywhere (Newton, per-side grad-h, or the clamp everywhere) improves shocks but costs shear/contact
flows. The wall-only clamp stays for the walls' outer row.

