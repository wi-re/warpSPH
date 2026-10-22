# Classic fluid comparison, 2026-10-21

## Preservation and future comparisons

This is a retained improvement baseline. **Do not overwrite or remove the original runs during routine cleanup.** New solver versions should write to a new dated directory and compare against this baseline, retaining failures, the numerical-source snapshot and the terminal-step correction flags.

The report, static [ranking PNG](figures/case_ranking.png), [vector SVG](figures/case_ranking.svg), [ranking values/definitions](evidence/ranking_2026-10-22.json), [ranking CSV](evidence/case_ranking.csv), [quantitative metrics CSV](evidence/comparison_metrics.csv), manifest, reproduction scripts and harness regression checks are preserved in Git. Reading the figure and tables does not require a GPU or daemon. Other numerical working-tree changes are not part of the report-preservation commit; measured code is defined by the archived source snapshots.

The complete raw dataset, including videos, particle HDF5, original/final states, logs, source bundles, experimental reference data, runtime package versions and report-generation scripts, has a separate versioned archive:

`/home/lu26029/dev/warpSPHDevelopment/core/archives/fluid-matrix-2026-10-21-v1.tar.zst`

Its `.sha256`, `.inventory.json` and `.verification.json` sidecars record the archive hash and every payload file's SHA-256. The verification receipt is also retained in [archive_2026-10-22.json](evidence/archive_2026-10-22.json). Archive files are read-only. Full streamed verification covered **16,625 files / 11,657,141,225 payload bytes**; the archive is **7,212,926,353 bytes**, SHA-256 `edc06911a3c1054d21a4d4c12f004ccb84a4bb3d85ce6e1273b2fa3617b94921`. A separate `fluid-matrix-2026-10-21-v1-source-context.tar.zst` supplements it with 933 verified installed-source files, including the boundary package at its actual `curvatureBoundaries` location. This supplement is labeled as source context at preservation time; the measured launch/per-run snapshots remain authoritative. This is a **local archive** in the existing manually backed-up `warpSPHDevelopment` tree; no cloud upload or automatic synchronization is claimed.

Restore into an empty directory with `tar --zstd -xf <archive>`, then open the extracted `comparison.html`; its data/video links are relative. `preservation/RESTORE.txt` and `preservation/inventory.json` provide instructions and integrity information. Verify independently without a GPU:

```text
python scripts/preserve_fluidMatrix.py --verify --archive /path/to/fluid-matrix-2026-10-21-v1.tar.zst
```

The verifier reads the full archive and checks every decompressed payload hash. Its valid-archive and deliberately corrupted-payload checks pass; the seven harness regressions and runtime import of 514 frontend modules also pass. The broad static scanner additionally encounters 30 existing Python-2 syntax files in the ignored third-party Spheral build tree and three obsolete imports in ignored `.tmp` probes; those unrelated files were not modified. The exact same command works using the archived copy under `preservation/repository/scripts/`. Future baselines must use a distinct archive version; the preservation script refuses to overwrite an existing archive.

## Final results — 2026-10-22

Finished **03:41 Europe/London on 22 October**. All 86 runnable combinations were attempted: **63 reached their target time, 23 stopped on numerical alarms, and 34 matrix entries were unsupported**. There were no wall-budget timeouts or process crashes. About 6 h 40 min elapsed overall; summed simulation wall time was 6.43 h over **985,966 timesteps**. No numerical defaults were changed and no failed run was tuned or retried.

[Full comparison, videos and data](/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21/comparison.html) · [Completion matrix](/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21/completion_matrix.png) · [All quantitative metrics CSV](/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21/comparison_metrics.csv) · [Evidence audit](/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21/final_audit.json) · [Curated summary](evidence/final_summary_2026-10-22.json)

### Quantitative ranking figure — 2026-10-22

[Single ranking figure, PNG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21/ranking/case_ranking.png) · [Vector PDF](/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21/ranking/case_ranking.pdf) · [SVG](/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21/ranking/case_ranking.svg) · [Ranks and exact metrics CSV](/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21/ranking/case_ranking.csv) · [Definitions and values JSON](/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21/ranking/case_ranking.json)

The upper panel ranks case-specific physical agreement or energy retention; the lower panel independently ranks total wall time for completed runs. Each case/boundary column has its own best-to-worst color scale, based on log loss. Ranks and actual values are printed. Stopped and unsupported cases are categorical, and a single completed candidate is neutral blue. No overall quality/cost score is formed.

**User-selected vortex criterion:** less energy decay is better, but energy growth is penalized. Non-growing completed runs rank by `100*(1-Efinal/Einitial)`. A maximum energy excursion above initial energy by more than 0.1% puts a run behind all non-growing runs; its cell reports maximum gain. Band has +30.7% periodic and +43.6% bounded excursions and ranks last in both. This is an energy-retention criterion, not agreement with exact viscous dissipation.

TGV and shear use time-integrated relative L2 error of the analytic energy-decay curve. Marrone uses the existing coarse Buchner P1/P2 digitization and a common 0.1-t* median; close ranks are not significant. Sloshing uses relative L2 error of case-native Sensor 1, with matching 10 ms Gaussian smoothing and no time shifting; pressure estimators can differ. Hydrostatic and wedge use the worst of fractional bulk pressure-gradient error and tail max-speed divided by sqrt(gH); Couette uses the worst of profile and both torque errors. These exact, inspectable choices are saved beside the figure. Already-audited near-zero terminal samples remain excluded. No simulations were rerun.

### What works under the tested presets

- **Delta-SPH has the broadest tested usable coverage.** All 13 supported runs reached the end; hydrostatic and wedge runs remained quiet, and it was the only tested Couette configuration to reach t=25.
- **Sun delta-plus best matches TGV energy decay.** E(5)/E(0)=0.818910 versus 0.818731 analytic, an error of +0.0219%. It finished all eight supported entries. Its analytic-wall combinations remain unsupported by the named-scheme guard.
- **VD+PS is the most broadly completing projection path:** 12/13 endpoints. TGV dissipates too quickly (−6.85% final energy error), still-water residual motion remains, and this Couette preset stops.
- **Band is useful as a particle-wall pressure comparison, not a viscous-flow accuracy result.** It reaches all eight endpoints and has the quietest native-column projection result, but its source assembly omits physical viscosity. Its shear mode consequently does not decay.
- **Omni and IISPH remain case/boundary dependent.** Analytic walls improve several endpoint outcomes, but an IISPH analytic still pool has tail max-speed near 9.18 m/s, and the wedge near 11.14 m/s: these are poor static-water solutions despite completion.
- **The local reference DFSPH path is unreliable in this matrix:** ten of twelve runs stop; its completed TGV run loses far too much energy. This is a result about the local implementation/preset, not published DFSPH in general.
- **ACSPH finishes TGV, periodic random vortices, Marrone and English wedge, but stops on bounded vortices, sloshing (t=4.148) and hydrostatics (t=3.347).** These are the recorded matrix configurations; historical successes under other presets are not disproved.

| Scheme | Reached end | Stopped | Unsupported |
|---|---:|---:|---:|
| VD+PS | 12 | 1 | 2 |
| Local reference DFSPH | 2 | 10 | 3 |
| IISPH | 6 | 6 | 3 |
| Omni | 10 | 3 | 2 |
| Band pressure boundaries | 8 | 0 | 7 |
| Delta-SPH | 13 | 0 | 2 |
| Sun delta-plus | 8 | 0 | 7 |
| ACSPH | 4 | 3 | 8 |

**Reached end is not an accuracy grade.** In particular, four raw shear endpoints are invalid because of the harness roundoff defect described below. Coverage also differs: 13/13 and 8/8 are not the same test set.

### TGV: 65,536 particles, t=5, ν=0.01

The analytic energy ratio is exp(−0.04×5)=0.818730753. Final energy error below is relative to that ratio. Wall timings include initialization, video, diagnostics and particle exports; timestep counts differ between schemes.

| Scheme | E(5)/E(0) | Relative energy error | Wall s | Wall ms/step |
|---|---:|---:|---:|---:|
| VD+PS | 0.762673 | -6.847% | 136.5 | 27.31 |
| Local reference DFSPH | 0.151328 | -81.517% | 136.1 | 27.22 |
| IISPH | stopped at t=0.113 | — | 8.2 | 72.57 |
| Omni | stopped at t=1.780 | — | 24.4 | 13.70 |
| Band pressure boundaries | 0.974020 | +18.967% | 894.6 | 178.93 |
| Delta-SPH | 0.813676 | -0.617% | 35.6 | 11.06 |
| Sun delta-plus | 0.818910 | +0.022% | 37.7 | 11.71 |
| ACSPH | 0.800102 | -2.275% | 734.6 | 537.34 |

Band has no physical-viscosity source, so its curve is not a matched viscous baseline. The failed traces and full-range energy excursions remain in the full comparison. No resolution or timestep convergence claim is made.

### Hydrostatics and the English wedge

Pressure-gradient ratios should approach one; tail max-speed should approach zero. Values are medians over the last 10% of recorded steps. Particle projection/AC hydrostatics use the native column; WCSPH and analytic runs use the established full-width still-pool adapter. Thus these are preset comparisons rather than isolated wall-treatment A/Bs. The slope fit is a bulk diagnostic and does not certify sharp-corner pressure accuracy.

| Hydrostatic scheme / wall | Tail pressure-slope ratio | Tail max-speed [m/s] |
|---|---:|---:|
| VD+PS / particles | 0.9227 | 0.22488 |
| VD+PS / analytic | 0.9058 | 0.81963 |
| IISPH / particles | 1.0093 | 1.85085 |
| IISPH / analytic | 0.7771 | 9.17788 |
| Omni / particles | 0.9982 | 0.72585 |
| Omni / analytic | 1.0400 | 0.96826 |
| Band pressure boundaries / particles | 0.9984 | 0.06647 |
| Delta-SPH / particles | 1.0060 | 0.00366 |
| Delta-SPH / analytic | 1.0046 | 0.00309 |
| Sun delta-plus / particles | 1.0059 | 0.00343 |

Delta-SPH wedge tail max-speeds are 0.01333 m/s (particles) and 0.01254 m/s (analytic), with bulk pressure-gradient ratios 1.0053 and 1.0045. Sun on particle walls gives 0.00986 m/s and ratio 1.0057; PST is deliberately disabled in both still-water controls. In contrast, omni on particle walls reaches the endpoint with a 172 m/s transient peak and tail max-speed 2.21 m/s. A good bulk pressure slope alone is insufficient.

### Sloshing: t=7 and measured Sensor 1

Nine supported runs reach t=7. Reference DFSPH stops for both wall types; particle IISPH stops at t=3.210; ACSPH stops at t=4.148. Analytic IISPH reaches t=7 but costs 1,007 s / 61,069 steps and retains substantial residual motion.

Peak pressures below use the same **10 ms Gaussian** for simulated and measured traces. Values are kPa @ seconds. These are the case-native Sensor-1 estimators; particle and analytic estimators can differ. A separate common fluid-probe series and peak table are exported where available. The first impact window is 2.2–2.9 s, then 3.9–4.6 and 5.4–6.1 s.

| Run | Impact 1 | Impact 2 | Impact 3 |
|---|---:|---:|---:|
| Measured, same smoothing | 1.800 @ 2.390 | 2.322 @ 4.066 | 1.171 @ 5.697 |
| VD+PS / particles | 1.911 @ 2.787 | 4.631 @ 4.173 | 3.779 @ 5.795 |
| VD+PS / analytic | 3.091 @ 2.374 | 3.732 @ 4.050 | 4.170 @ 5.653 |
| IISPH / analytic | 2.805 @ 2.458 | 1.519 @ 4.294 | 4.396 @ 6.044 |
| Omni / particles | 1.784 @ 2.374 | 1.711 @ 4.045 | 2.086 @ 5.652 |
| Omni / analytic | 2.255 @ 2.370 | 3.227 @ 4.028 | 3.546 @ 5.644 |
| Band pressure boundaries / particles | 1.018 @ 2.685 | 1.618 @ 4.011 | 1.491 @ 5.662 |
| Delta-SPH / particles | 4.447 @ 2.357 | 5.449 @ 4.016 | 4.363 @ 5.632 |
| Delta-SPH / analytic | 6.196 @ 2.367 | 5.797 @ 4.029 | 4.279 @ 5.647 |
| Sun delta-plus / particles | 4.877 @ 2.343 | 5.157 @ 4.005 | 5.277 @ 5.557 |

No scheme matches all three measured peaks here. For example, omni/particle is close on the first peak, 26% low on the second and 78% high on the third. Delta variants reach t=7 but substantially overshoot measured impact pressures. Particle VD+PS and Band have notably late first peaks. The full overlay and peak CSV retain all estimators and incomplete-window exclusions.

### Taylor–Couette

Only delta-SPH reaches t=25 under the tested nx=96 presets. Its tail velocity-profile amplitude is **0.98499** of the steady analytic solution (−1.50%); inner torque is **1.01536** of exact (+1.54%), and outer opposing torque **0.95059** (−4.94%). VD+PS stops at t=0.998 and omni at t=0.216. These projection runs used the declared fixed dt=0.002 and n_h=2.505; failures do not establish timestep-independent method failure. Other scheme/case combinations, including spinning particle-wall adapters, remain unsupported.

### Endpoint defect and evidence qualification

The original matrix harness clamped each next dt to the exact remaining time. Floating-point summation left 13 nominally completed runs with an extra residual step of roughly 10⁻¹³–10⁻¹¹ seconds. Four shear pressure-map endpoints were corrupted (local reference DFSPH, IISPH, omni and Band). The runner also checks completion before its terminal velocity alarm, so those rows still report completed.

All raw NPZ/CSV, tensors, HDF5 and videos are unchanged. Derived curves and metrics now consistently exclude the negligible final sample in all 13 affected runs, using the preceding sample already at target within rounding error; fields `roundoff_terminal_sample_excluded`, `endpoint_corrupted` and `raw_*` make this explicit. The prior valid IISPH shear solution is already poor (energy ratio about 681), so filtering does not turn it into a successful shear test. Band correctly shows no physical viscous decay in the valid trajectory.

The harness has been repaired for future runs by snapping only the rounding-sized endpoint time label, with seven CPU regression checks passing. No physics trajectory was rerun and no solver was modified.

Final audit: **86/86** have ffprobe-readable videos, diagnostic CSV/NPZ, initial/final tensor snapshots, FP32 particle HDF5, and matching terminal daemon time/step/status. There are no missing records or partial process failures. All measured fluid-mass drifts are zero. The 81 per-run numerical source inventories match the launch snapshot; the first five runs retain the shared launch source bundle. Thus no numerical source drift was detected during the queue.

Quantitative raw comparisons, all per-run timings and pressure/profile data remain linked from the HTML report. This pass ends with the recorded capability findings; numerical failures, local corner-pressure accuracy and convergence questions are not chased.

Tracked by [SCHEME_CLEANUP_PLAN.md](../../SCHEME_CLEANUP_PLAN.md).

User scope: weakly compressible and incompressible schemes, including ACSPH as
a comparison. No compressible gas solvers. Run established stable setups;
record failures, do not debug or tune solvers during this sweep. Both wall
representations are included wherever the existing case and scheme support
them. Unsupported combinations remain explicit matrix entries.

Eight schemes: VD+PS (`divergenceFree`), local reference DFSPH, IISPH, omni,
Band pressure boundaries, plain delta-SPH (PST disabled), Sun 2017 delta-plus
(PST enabled and its named frozen-diffusion default), and ACSPH. The still-water
and isolated shear-viscosity controls explicitly disable PST for both delta
variants, as required by their established test setup.

| Test | Resolution | End time | Boundaries |
|---|---:|---:|---|
| Taylor–Green | 256² = 65,536 | 5 s | periodic |
| Decaying divergence-free random vortices | 192² = 36,864 | 5 s | periodic |
| Viscous shear mode | 192² = 36,864 | 2 s | periodic |
| Decaying vortices in a box | nx=128 | 3 s | particle walls |
| Marrone 2011 §3.1 dam break | nx=100, dx=0.01 m | 2.45 s, t*≈9.91 | particles / analytic |
| SPHERIC TC10 sloshing | nx=200, dx=0.0045 m | **7 s** | particles / analytic |
| Hydrostatic column / still pool | nx=128, dx=1/128 m | 4 s | particles / analytic |
| English 2022 wedge | nx=120, dx=0.01 m | 4 s | particles / analytic |
| Taylor–Couette | nx=96, dx=1/96 m | 25 s | existing rotating analytic cylinders |

All simulations use FP32, VisPy video, progress, daemon publishing and HDF5
particle trajectories (every 100 steps, plus normal final export and exact
initial/final tensor snapshots; early stops retain the last available HDF5
sample). Every diagnostic timestep is also streamed to JSONL; completed or
normally stopped runs export CSV, NPZ and trajectory JSON. Initial and effective
case specs, source hashes and source archive are retained. One GPU child runs at
a time in a fresh process. The queue resumes without repeating terminal rows.
The declared per-run wall budget is two hours; velocity >100 times the declared
physical scale, nonfinite velocity and the established timestep/progress stall
guards stop a row. These are recorded stopping criteria, not accuracy grades.

## Historical setup sources

- [Existing validation battery](../../scripts/validate_scheme.py): Wendland C2,
  SuperSymmetric supports and semi-implicit Euler for projection schemes.
- [Sloshing example](../../examples/sloshingTank/run_sloshingTank.py) and
  [corrected omni TC10 results](../analytic_boundaries/omni_sloshing_fixed_2026-10-09/README.md):
  calibrated analytic lattices; omni n_h=2.57, other projections n_h=4;
  dt≤0.002 s. Case defaults retain WCSPH continuity/contact and ACSPH
  low-cost authors-code alignments and wall-bounce ramp.
- [Marrone probe](../../scripts/probe_deltaSPHMarrone.py): H=0.6 m,
  width=3.2196 m, reservoir width=1.2 m, free slip, 90 mm pressure discs.
- [English wedge probe](../../scripts/probe_englishWedge.py) and
  [omni wedge probe](../../scripts/probe_omniEnglishWedge.py): 2.4×1.2 m tank,
  depth 0.5 m, calibrated 0.24 m wedge, no PST, hydrostatic initialization.
- [Couette probe](../../scripts/probe_taylorCouette.py): Morris viscosity,
  no-slip moment closure, calibrated lattice, projection closed preset and
  n_h=2.505. Prescribed rotating analytic walls are currently supported only
  for delta-SPH, omni and VD+PS. No new moving particle-wall adapter is added.
- Native periodic TGV/random-flow and hydrostatic-column cases supply their
  established initialization and diagnostic hooks. TGV physical wave number is
  `k/2=1`; analytic energy ratio is **exp(-0.04 t)** at ν=0.01.

## Comparison limits

This is a capability survey using existing stable presets, not a controlled
single-parameter discretization study. Projection and WCSPH lattice relaxation
differ. Particle hydrostatics use the native column (with rest-density
calibration) for projection/AC; the weakly compressible and analytic legs use
the existing full-width still-pool adapter. Effective specs expose all choices.
Band omits physical viscosity in its current source assembly, so its viscous
decay results cannot establish viscosity accuracy. ACSPH has no existing
periodic-channel viscosity adapter and is explicitly skipped there.

The named Sun scheme remains excluded by the existing analytic-wall allowlist;
no guard is bypassed. Analytic and particle pressure estimators can differ.
TC10 pressure curves must be compared against measurements with **the same
smoothing**; the historical corrected report uses a 10 ms Gaussian. Couette
nominal viscosity can differ from realized viscosity: shear-mode data are
included, with torque and profile diagnostics exported by the native case.
The Morris shear/Couette legs target the same nominal realized long-wave
ν=0.0185: the weakly compressible alpha coefficient is derived at each
resolution from the native case's `alpha c0 h / (8 xi)` relation. Residual
realized-viscosity differences remain measurable in the shear data.
GPU timing is contextual throughput with video, diagnostics and exports; other
existing daemon workers are logged through `nvidia-smi pmon` per run. No claim
of isolated kernel benchmark timing is made.

## Commands and output

Use `/home/lu26029/miniconda3/envs/warp/bin/python`:

```text
scripts/run_fluidMatrix.py plan
scripts/run_fluidMatrix.py run
scripts/run_fluidMatrix.py report
scripts/analyze_fluidMatrix.py
```

The immutable plan is [matrix_2026-10-21.json](matrix_2026-10-21.json).
Raw output, videos, growing quantitative `comparison.html`, queue `index.html`,
plots, `comparison_metrics.csv`, `summary.csv`,
`queue_status.json` and per-run logs are outside Git at:

`/home/lu26029/dev/warpSPHDevelopment/core/experiments/fluid-matrix-2026-10-21`.

## Launch qualification (historical)

The queue ran independently of the chat, with one GPU child at a time.
There are **120 matrix entries: 86 runnable and 34 explicitly unsupported**.
All 86 runnable configurations passed `prepareContext` on CPU without sampling
particles or advancing a simulation. Both scripts compile and `git diff --check`
passes. Rendered TGV frames and the quantitative comparison plot were visually
inspected; ffprobe, HDF5 and daemon metadata checks confirm the first terminal
runs have playable videos, particle data, diagnostic CSV and correct terminal
time/status in the daemon.

Initial results are observations, not adopted fixes: VD+PS and reference DFSPH
finish TGV at t=5; IISPH reaches its declared velocity stop at t≈0.113 and omni
at t≈1.78. Those failures retain videos and diagnostics and are not retried.
VD+PS E(5)/E(0)=0.76267 against 0.81873 analytic (−6.85%); the reference path is
substantially more dissipative. The growing comparison contains the remaining
results as they arrive. No solver code or numerical default was changed for
this matrix.
