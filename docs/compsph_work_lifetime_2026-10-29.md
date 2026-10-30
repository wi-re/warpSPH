# CompSPH pair-work recomputation and stage lifetimes (2026-10-29)

User authorizes applying the CRK-style work-storage optimization to CompSPH and investigating overlapping allocation lifetimes. Baseline is commit `c187532`; artifacts are under `.tmp/comp-work-lifetime-20261029/`. The two changes have independent baseline/operator-only/lifetime-only/combined frozen runtimes. No neighbour format, support policy, physical formula, source, stage reuse or integration coefficient is changed.

## Implementation

CompSPH's existing mean-gradient pair force and gather heating now share one Warp pair function. The expanded operator still returns pressure/viscous pair arrays. The native memory route instead sums particle acceleration/heating without those arrays, then recomputes each pair to write or accumulate the weighted compatible-work vector directly. The entropy partition uses the same existing `crkEntropyFraction` as expanded CompSPH. It is evaluated only after the complete stage acceleration, including native forcing and enforced updates, is available for the half-step velocity. Zero-weight stages omit work; changed-list replay reduces directly into particle energy increments using the native replay velocity.

`compPairStorage=recompute` is the normal qualified single-gas eager CRK-partition route. `expanded` retains the reference implementation. Walls, materials, other energy partitions, non-compatible energy, differentiation/late differentiable forcing and graph/grid/in-place settings retain expanded work. The storage option round-trips through configuration and is omitted from the physical restart signature. This preserves old checkpoint semantics and permits changing only storage on continuation.

The runner now releases the previous `IntegrationResult` before entering the next ordinary integrator call. Its accepted system remains live, and the most recent result remains available for final stage/checkpoint output and early stops. The pipeline releases consumed stage data after its output dependency has been registered; terminal stages remain retained. This removes obsolete ownership without dropping the current step's public stage results or changing neighbour reuse.

## Controls and scope

At32³ and96³, compare the original baseline, refactored expanded operator, recomputation alone, lifetime cleanup alone and both changes. At96³ also compare lifetime-only against baseline for CRKSPH, Monaghan, GSPH, InutsukaGSPH, MFM and MFV. All are from rest for36 accepted steps, independent three-step warmup and first six discarded for steady timings. Five additional36-step allocation-history/ownership runs retain matching recordings and final stored fields. Unlike the earlier six-step captures, these span the full measured windows.

The physical setup is unchanged from the previous memory sweep: periodic finite-gamma driven3D gas, seed1234, band1–2 OU acceleration0.3/tau0.5/knots0.05, mass-mean subtraction, u0.9 relaxation tau0.2, B7/n_h4, native SPH midpoint/CFL0.15 and meshless SSPRK/CFL0.1. Every native RHS and accepted state is checked; no extra RHS, positivity, signed energy ledger, progress/watchdogs and source pins are required. GPU overlap with unrelated compute is explicitly user-authorized. Own-process Torch memory is reported separately from sampled process residency and contended timings. No per-step cache flushing or allocator tuning is introduced.

Numerical screens for startup windows are predeclared `rtol=3e-6, atol=3e-7` for stored fields; lifetime-only and refactored-expanded changes additionally require bitwise agreement. CompSPH CPU controls compare full live W, not uninitialized padded tails, including grad-h, multi-weight RK4/Heun, changed-list replay and restart. Native Sod nx400/t0.15 and Gresho nx64/t1.5 runs compare original/combined metrics: energy drift below1e-5, at most1% worsening plus1e-7 absolute floor for continuous metrics, shock width no more than one extra sampled element. These are storage nonregression checks against CompSPH's own baseline, not application of CRK-specific accuracy bounds to another method.

The candidate measurement snapshot predates a final restriction of recomputation for inherited grid/in-place graph settings. Native measured settings have both flags false, so their route is unchanged; the force/work kernel functions are identical. The final graph/grid restrictions have separate current-source tests. Cosmetic whitespace cleanup also leaves the compiled force functions unchanged. Exact measured source pins and final-source differences remain archived.


## Results

All30 finite GPU campaign jobs pass:27 memory windows/captures, a CUDA
regression suite and two original/combined physical-control jobs. All1,944
observed startup RHS are valid, no extra RHS, signed relative energy-budget
residual at most5.33e-8. All4,816 pinned source/input/test/driver files remain
unchanged. All five capture endpoints agree bitwise with their ordinary windows.
No solver failure or GPU trajectory retry occurred.

|96³ CompSPH change|Peak live GiB|Peak reserved GiB|
|---|---:|---:|
|Original|28.791|29.971|
|Refactored expanded reference|28.791|29.971|
|Recomputation only|10.276|13.213|
|Lifetime cleanup only|28.616|29.992|
|Both changes|10.100|13.182|

Together the changes save18.692GiB peak live allocation (64.9%); at884,736
particles,34.12→11.97KiB/particle. At32³ the live peak is1.097→0.404GiB.
The expanded refactor alone is bitwise identical to the original.96³ direct
work fields and sampled W are also bitwise. At32³ recomputation gives bounded
FP32 changes: velocity at most3.35e-8, u5.96e-8, density1.19e-7, divergence
1.73e-7 and sampled W3.73e-9. Every stored field passes the unchanged3e-6/3e-7
screen. Sampled W lacks a complete pair-ID archive; full live-W parity is
separately checked by the operator/trajectory tests.

CompSPH's retained accepted tensors are almost unchanged: one5.601GiB W and
3.781GiB adjacency plus small state/source buffers. Post-run Torch allocation
is9.550GiB before and after. The large reduction comes from removing expanded
ap/av/f and weighted pair-vector intermediates during each stage, rather than
reducing the accepted work vector or changing the pair representation.

|Other96³ scheme|Before live GiB|After lifetime cleanup GiB|
|---|---:|---:|
|CRKSPH|10.333|10.159|
|Monaghan|8.333|8.125|
|GSPH|8.316|8.124|
|InutsukaGSPH|8.316|8.124|
|MFM|10.533|10.391|
|MFV|10.533|10.391|

All lifetime-only fields are bitwise identical across all seven schemes.
The live saving is0.14–0.21GiB; obsolete stage results mainly retain particle
buffers at the high-water point. The larger within-step neighbour overlap
remains. For example, Monaghan's new first accepted step holds8.032GiB with
an8.125GiB cumulative peak; at steps2/3 it holds4.285GiB, but the first step
already established the high-water. Its two stage lists are created inside
that first step, not retained solely through the previous runner result.
Meshless geometry/search lists likewise coexist within a native step. Removing
those needs a separate validated stage/list ownership or reuse design; this
patch preserves the returned stage data and existing list validation/reuse.

Reserved memory responds differently: CRK reserved peak13.396→10.328GiB with
only0.174GiB less live memory, reflecting changed allocation lifetimes and pool
layout. Across all five captures there are no pending free requests or observed
non-adjacent request/completion events. Native Torch allocation timelines and
allocator-state HTML remain local. A boundary ownership inventory is not a
complete inventory at the true allocating instant; foreign storage and released
transients may be absent. Use the allocation stacks and counters together.

## Performance and physical checks

All timings are contended observations; other GPU compute was allowed throughout.
No robust speedup/regression claim follows from single matched windows.
Whole-process samples cover the entire child, including warmup/setup, whereas
Torch measurement counters reset after warmup; their different maxima must not
be treated as simultaneous or subtracted for exact attribution. CompSPH sampled
process peak falls30.949→16.137GiB in these invocations. Startup windows do not
bound a later developed/clustered96³ run.

Full27-row wall/ms-per-step/reserved table: `analysis/report.md`; exact observer,
source, allocation and contention records: `analysis/comparison.json`.

|96³ ordinary window|Run wall s|Steady wall ms/step|
|---|---:|---:|
|CompSPH-96-before|32.564|826.90|
|CompSPH-96-operator-expanded|32.224|818.23|
|CompSPH-96-operator|26.226|660.19|
|CompSPH-96-lifetime|32.316|826.80|
|CompSPH-96-combined|26.291|657.32|
|CRKSPH-96-before|40.226|1032.81|
|CRKSPH-96-lifetime|39.625|1026.66|
|Monaghan-96-before|37.296|960.89|
|Monaghan-96-lifetime|37.177|962.44|
|GSPH-96-before|27.091|679.56|
|GSPH-96-lifetime|27.557|697.63|
|InutsukaGSPH-96-before|24.134|592.57|
|InutsukaGSPH-96-lifetime|23.732|592.90|
|MFM-96-before|13.872|307.88|
|MFM-96-lifetime|13.810|305.03|
|MFV-96-before|13.913|313.24|
|MFV-96-lifetime|13.712|309.10|

Recorded native physical controls use their actual endpoints (Sod0.1497103,
Gresho1.4994251); requested maxima are0.15/1.5. Sod's resolution parameter is400
(the mirrored unequal-density sampler has500 particles,250 per tube); Gresho64² has4,096.
Sod metrics match exactly: density L1 error0.0044200, plateau-velocity std0.0045133,
entropy error0.0062189, shock-width count10. Gresho velocity L1 changes
0.1387174→0.1387156 and final relative KE changes−0.5290352→−0.5290350.
All declared1% nonregression screens pass. CompSPH's existing Gresho solution
is strongly dissipative (~52.9% KE loss); this does not qualify vortex accuracy
or improve that physical limitation. The regression diagnostic's energy drift
is reported as0 at its output precision, not a claim of exact real-arithmetic
energy conservation.

|Physical control|Original wall s / ms per step|Combined wall s / ms per step|Steps|
|---|---:|---:|---:|
|sod|25.306 / 94.08|38.980 / 144.91|269|
|gresho|178.977 / 170.29|149.804 / 142.53|1051|

Final controls:27 FP32 CPU checks;49 FP64 checks including scientific heating
and driver stage contracts;4 forced replay/thermal checks;2 inherited-grid
routing checks; CUDA regression suite14 passes and actual graph replay/wall/
two-material EOS suite5 passes. The full expanded CompSPH gradient canary reports
ALL PASSED. Early test-setup failures remain in their original logs/XML:
expanded-only replay injector, unsupported Heun-in-driven-case setup, and an
invalid test filename. Corrected controls keep the original physical assertions
and tolerances. The current-source graph replay test verifies more than pipeline
ownership alone; the pipeline ownership control can still use its existing
capture fallback.

All27 startup recordings were inspected through first/last contact sheets;
all four physical recordings through first/middle/final sheets. Before/after
flow and shock profiles agree visually; Gresho's substantial baseline diffusion
is visible in both. Density slice colour scales magnify tiny startup variations.
This is a representative slice/frame review, not full-volume accuracy evidence.

Verdict: keep native CompSPH recomputation and the runner lifetime cleanup.
Retain expanded compatibility routes, native neighbour/stage policies and the
recorded limitations. No new long turbulent trajectory, drag, forcing adjustment
or neighbour-format change was introduced. Plan/roadmap updated in place.

Artifacts: `analysis/completion-review.json`, `analysis/report.md`,
`analysis/comparison.json`, native Torch HTML and five raw allocator snapshots,
source-pinned runtime trees/commands, all recordings and successful/failed test
logs under `.tmp/comp-work-lifetime-20261029/`.
