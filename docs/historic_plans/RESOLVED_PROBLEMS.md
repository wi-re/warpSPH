# Resolved problems

Items that used to be in [OPEN_PROBLEMS.md](../../OPEN_PROBLEMS.md), moved here
once resolved, with their original section number so older references still
find them. Newest first. The original text is kept as it was when resolved.

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
