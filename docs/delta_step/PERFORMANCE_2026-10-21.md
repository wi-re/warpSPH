# Delta / delta-plus staged-module sweep, 21 October 2026

Tracked in [SCHEME_CLEANUP_PLAN.md](../../SCHEME_CLEANUP_PLAN.md). The user's ordering is correctness,
modularity/structure, general performance, then focused optimization. This is
an implementation-preservation pass; it does not reopen the long-time
free-surface/contact model investigations listed in OPEN_PROBLEMS.md.

## Structure and numerical contract

Delta-SPH and `sun2017DeltaSPH` continue to use the same RHS and system.
Delta-plus remains a configuration preset: indexed-stage diffusion freezing
and Sun Eq. (7)'s existing shifting constants. No duplicate solver or new time
integrator is introduced. RK/Verlet arithmetic stays in warpSPHIntegrators.
Unindexed evaluations still recompute diffusion, preserving the current
symplectic/Verlet behavior. An explicit RK4 selection remains explicit.

| Responsibility | Owner |
|---|---|
| Adjacency, ordered RHS operations and boundary callback sites | `schemes/weaklyCompressible/deltaSPH.py` |
| Per-step diffusion/surface memory, shared by stage clones | `modules/deltaSPH/stages.py`, system `initialize` |
| Surface stage policy and legacy time/index adapter | `modules/surfaceDetection/stages.py` |
| Completeness/bulk gating of pressure renormalization | `modules/pressure/renormalization.py` |
| Analytic wall stage evaluation and load booking | `modules/analyticBoundary/stages.py` |
| Pointwise update assembly and post-callback fluid masks | `modules/deltaSPH/updates.py` |
| Post-integration shifting transport | `modules/shifting/transport.py` |
| Existing contact impulse/replacement policies | `modules/mdbc/corrections.py` |
| Prescribed/free body advancement | `rigidBody/stages.py` |
| Lazy acceleration/contact diagnostics | `modules/diagnostics/weaklyCompressible.py` |

The weakly-compressible system is now a lifecycle adapter over these parts.
Its finalization order remains: copy stage diagnostics; evaluate shifting
transport; restore boundary density/positions and protect density; apply
shifting transport; optional lone-density reset; contact correction; body
advancement; diagnostics and pinned velocity. Shifting transport is deliberately
evaluated before density restoration/flooring, matching the old calculation.

Normal integrator stages use a system-owned holder, renewed at each real step
and shared explicitly by stage clones. They cannot consume a foreign or legacy
configuration cache. The surface diagnostic dictionary on the configuration
is retained as a compatibility view; owned physics never reads it. Direct
legacy helper calls retain their old adapter. Old scheme-level wall/surface
imports remain compatibility shims.

Boundary callbacks still run between update assembly and the final fluid mask.
Captured native assembly/masking uses the caller's Torch/Warp stream and
disables floating-point contraction. Eager execution and reverse/JVP or other
dtype/device inputs retain the ordered Torch arithmetic, avoiding per-call
native view marshalling on the eager path. No EOS, diffusion sign, all-pair viscosity,
pressure/surface policy, shifting strength/cap, density floor, no-penetration
placement or temporal default is changed.

## General performance changes

- Remove the unused neighbour-count evaluation from every shifting iteration.
- Remove an unused support-minimum reduction from the delta shifting law.
- Compute the same shifting-velocity divergence once when density and velocity
  transport corrections both request it.
- Fuse pointwise assembly and final masking around the existing callback boundary.
  Neighbour kernels and their traversal policy are unchanged. Native eager
  assembly was measured and rejected as a default: repeated array conversion
  erased its benefit and slightly regressed one sample. The final route uses
  Torch eagerly and native fusion during capture only.

No caching load policy, broad kernel specialization or register cap is promoted.
Existing numerical concerns such as the shifting tensile reference value
(OPEN_PROBLEMS.md §25), implicit-shift boundary behavior (§26) and long-time
free-surface/contact dynamics remain unresolved by this preservation sweep.

## Qualification and known limits

**35 supported source pairs are bitwise identical** across FP32/FP64, periodic
TGV, dam break and moving-gravity sloshing, eager/graph modes, RK4, frozen
surface masks, projected pressure correction, shifting transport corrections
and the supported delta-SPH analytic-wall path. All saved tensor fields are
present and finite. The **12 updated requested-graph/eager pairs** are also bitwise
identical. Existing graph eligibility and eager fallbacks are retained; an
indexed frozen-diffusion RHS is still ineligible for independent RHS capture. The additional delta-plus analytic-wall combination is the
unchanged expected refusal described below.


Focused FP32 contracts: **136 passed**. Focused FP64 lifecycle/surface/continuity
contracts: **28 passed**. These include independent contact-placement algebra,
interleaved owner/legacy cache isolation, reverse/forward derivative fallback,
nondefault-stream/external-capture ownership, diffusion cancellation,
hydrostatic corrections, surface stages, acoustic continuity, pinned bands,
Morris/wall viscosity, analytic body loads and graph/runner checks.

The existing runner refuses analytic walls for the named delta-plus preset,
although delta-SPH supports them. This restriction is preserved and checked as
a refusal, not counted as a numerical state pair. Automatic approval review
rejected expanding that boundary-policy contract during this cleanup. No
workaround or wall model change is included.

A kernel code-generation failure in the initial native mask prototype was fixed
before the final gates; its superseded log is retained. The analytic-cylinder
inner-RHS validation warning also occurs in the isolated old-source test;
whole-step graph/state/load contracts pass on both sources. This is retained
as an existing fallback, not silently attributed to this change.

## Measured throughput

RTX PRO 6000 Blackwell; dam break `nx=432`, **72,816 particles**; ten warm-up
and 100 timed steps. Three alternating trials per source, precision and
execution mode. Entries are medians of trial means; renders are off.
Both sources use the same current core and integrators, so the separately
completed core register pass is not included in the claimed change.

| Scheme | Precision | Mode | Wall ms/step before → after | Integrator ms before → after |
|---|---|---|---:|---:|
| delta | FP32 | eager | 17.088 → 16.731 | 13.578 → 13.236 |
| delta | FP32 | graph | 13.661 → 13.505 | 13.304 → 13.151 |
| delta-plus | FP32 | eager | 17.174 → 17.370 | 13.619 → 13.811 |
| delta-plus | FP32 | graph | 13.720 → 13.481 | 13.358 → 13.132 |
| delta | FP64 | eager | 66.254 → 65.759 | 53.892 → 53.244 |
| delta | FP64 | graph | 71.863 → 70.925 | 70.029 → 69.141 |
| delta-plus | FP64 | eager | 66.428 → 65.816 | 53.731 → 53.294 |
| delta-plus | FP64 | graph | 71.750 → 70.958 | 69.962 → 69.179 |

The result is **mostly neutral throughput with modest measured gains**, not
a large solver speedup. FP32 graph medians improve 1.1% (delta) and 1.7%
(delta-plus); delta eager improves 2.1%. Delta-plus FP32 eager has a **1.1%
slower median** with overlapping/noisy ranges: before 17.019–17.510 ms, after
16.764–18.177 ms. FP64 improvements are 0.7–1.3%. The evidence preserves every
trial rather than treating small differences as firm universal wins.

At this FP64 workload, graph mode remains slower than eager despite the
small within-mode improvement. No execution or precision default is changed.

The three-step FP32 delta trace has 798→796 host kernel launches per nominal
eager step, with the same synchronization count. In graph mode host launches
remain 54 plus two graph launches per nominal step; **GPU kernel events in
the complete window fall 2,877→2,784** (about 31 fewer per nominal step),
including common diagnostic/rebuild work. The four native update kernels per
step are present in captured execution and absent in eager execution. These
short-window counts are not universal per-step constants; rebuilds affect them.

The rejected eager-native pilot used 22,256 particles. Its delta-plus FP32
median increased from about 15.09 to 15.48 ms; array-view marshalling erased
the launch reduction. That prototype, its measurements and the interrupted
remaining trial are retained separately. The final choice keeps Torch eager
arithmetic and capture-only native fusion. Focused neighbour/register/cache
optimization is deferred.

## Recorded physical controls

Four old/new FP32 dam-break controls at `nx=64` (5,751 particles), with VisPy
video, progress, trajectory/measurement recording and the existing daemon,
reach **t=0.15 in 300 steps**. All four daemon records are verified completed
at the exact final step/time. Final video frames were inspected.

| Updated control | Fluid density min / max | Relative density RMS | Particle mass change | Wall seconds, with recording/setup |
|---|---:|---:|---:|---:|
| deltaSPH | 1.000028 / 1.005642 | 0.002465 | 0 | 8.497 |
| sun2017DeltaSPH | 1.000028 / 1.005596 | 0.002465 | 0 | 8.382 |

Every saved field of each updated control is **bitwise identical to its old
source counterpart**. Total mass and each particle mass are unchanged.
Final kinetic energy is 0.6479644102 (delta) / 0.6478621241 (delta-plus),
also exactly preserved against the corresponding old source. Weakly
compressible stabilization is dissipative; these checks do not claim total
energy conservation. Wall averages with setup/rendering/recording are
28.32 / 27.94 ms per step; the warmed table above is the throughput measure.

These are short preservation/positivity controls, not long-time impact or
resolution-convergence qualification. Independent diffusion/continuity/contact
and wall/body oracles remain the correctness basis. Existing physical-model
questions and graph policy restrictions stay explicit.

## Configuration preservation

The legacy exporter/importer dropped `sun2017Eq7Shift`, transport corrections,
shift caps and other `ShiftProperties`, plus explicit contact/continuity
policies. The new regression tests failed on both delta and delta-plus before
the fix. A reusable codec now lives alongside `ShiftProperties`; the weakly
compressible serializer uses it and retains the additional explicit policies.
The saved selector/options round-trip through JSON without changing the chosen
method. Missing keys retain the historical constructor/decoder defaults;
older exports cannot recover options that were never written.

**104 configuration/schema checks pass**, plus four idempotence/legacy-default
checks. Four post-fix default-trajectory spot checks (both schemes, eager and
requested graph) are bitwise against the original states. This serialization
fix followed the timed runs; all numerical hot-path source is unchanged.

## Evidence and reproduction

[Curated timings, profiles, controls and daemon IDs](evidence/performance_2026-10-21.json).
Raw states, traces, all trial logs, passing and superseded qualification logs,
source snapshots, command records and reproduction scripts are archived at:
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/delta-cleanup-2026-10-21`.
The archive has a SHA-256 manifest. It preserves the exact measured benchmark
script and pre-codec configuration source in addition to the final tree.
No raw traces or binary state files are added to Git.

```bash
/home/lu26029/miniconda3/envs/warp/bin/python scripts/bench_delta.py \
  --scheme sun2017DeltaSPH --case dambreak --nx 432 --precision float32 \
  --graph --warmup 10 --steps 100 --out /tmp/delta-plus-timing

/home/lu26029/miniconda3/envs/warp/bin/python scripts/bench_delta.py \
  --scheme sun2017DeltaSPH --case dambreak --nx 64 --precision float32 \
  --no-graph --warmup 10 --steps 100 --end .15 --video --daemon \
  --plot-interval 25 --out /tmp/delta-plus-control
```

Run GPU jobs sequentially. Use the archived old frontend src on PYTHONPATH
for the reference and the shared current core/integrators on both sources.
The sloshing roll-history asset and source-relative layout are preserved in
the archive. Tiny A/B/timing windows omit video; the physical controls retain it.

The system adapter shrank from roughly 650 lines to **212**. The prescribed/free
body loop now delegates from its module using the existing body-integration
API and its historical once-per-fluid-step placement. No new fluid temporal
implementation or body-coupling model is introduced. The scoped cleanup is
complete; AC, long-time model investigations and focused optimization remain
separate work.
