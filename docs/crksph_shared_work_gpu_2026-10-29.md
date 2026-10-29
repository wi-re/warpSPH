# CRKSPH shared-work short GPU comparison — 2026-10-29

Follow-up: the obsolete RunResult owner has now been removed. Matched short-window
96³ peak live allocation is 10.333 GiB; the historical measurements below retain
their original scope. [Ownership fix and Torch memory-viz captures](crksph_retention_2026-10-29.md).

The recomputation path cuts peak allocated memory by **53.7% at96³**
and is **4.4% faster in these startup windows** than refactored expanded storage.
At32³ from an evolved checkpoint it uses43.2% less allocated memory but is9.8%
slower. These short windows do not establish a universal speedup. Following the
user’s preference and further qualification, recomputation is now the default
for native eager single-gas compatible-energy steps. Expanded remains an
explicit reference and automatic fallback. No full t4 performance run was repeated.

## Matched finite windows

Artifact root: `.tmp/crk-shared-gpu-20261029/`. GPU: RTX PRO6000 Blackwell.
Compare frozen original, shared-helper expanded, and recompute, with two36-step
repeats per path at each size. Each process first does a separate3-step warmup;
exclude the first6 steps of each measured run from steady timing. All runs use
FP32, native fresh midpoint/CFL0.15, B7/n_h4, original buffered neighbour policy,
OU band1–2/seed1234/acceleration0.3/tau0.5/knot0.05, mean removal and the u0.9/tau0.2
bath.32³ resumes the native saved t4 checkpoint and reaches4.1368613243;
96³ starts from rest and reaches0.05232479423. Inputs, driver, sources and
checkpoint hashes are archived. Output policy and guards are identical.

Slice video/progress/watchdogs are enabled. The independent3-step warmups omit
video. Wall intervals include native integration, accepted diagnostics, progress
and amortized video frames. Native times are the runner's CUDA-event step timers
and include the common raw-RHS guards; they are not isolated-kernel timings.
No allocator/section profiling runs inside these timed windows. Initial/final
particle dumps happen outside the steady interval. Allocated/reserved peaks are
read before the final CPU dump, and cover the whole measured run.

| Size | Path | Steady wall ms/step | Native step ms | Peak allocated GiB | Peak reserved GiB | Both loop walls, s |
|---|---|---:|---:|---:|---:|---:|
| 32³ | legacy | 32.72 | 28.25 | 1.589 | 1.766 | 6.081 |
| 32³ | expanded | 32.27 | 27.96 | 1.589 | 1.766 | 2.297 |
| 32³ | recompute | 35.43 | 31.05 | 0.903 | 1.070 | 2.521 |
| 96³ | legacy | 551.61 | 547.04 | 34.545 | 35.584 | 39.791 |
| 96³ | expanded | 559.99 | 555.39 | 34.545 | 35.584 | 40.399 |
| 96³ | recompute | 535.20 | 530.62 | 16.006 | 21.639 | 38.663 |

The two expanded/recompute wall means are32.21/32.33 versus35.34/35.52ms at32³,
and558.22/561.76 versus534.00/536.41ms at96³. Compared directly with the frozen
original, recomputation is8.3% slower at32³ and3.0% faster at96³. The helper-only
refactor shifts measured timing by−1.4%/+1.5% at32³/96³ while preserving outputs
bitwise. Fixed path order and only two repeats limit interpretation of the small
speed differences. No unrelated compute was observed in the GPU telemetry;
desktop graphics and GPU clocks were not controlled.

All12 measured loops together take129.752seconds; measured run walls including
setup/recording take157.678seconds. Full child process wall times, including
warmup/imports/compilation/output, are retained in supervisor status. These short
startup/evolved windows should not be substituted for the earlier full-t4 costs
or the historical42.10GiB allocated high-water over the whole96³ trajectory.

## Separate memory ownership probes

Two3-step96³ startup probes with region synchronization, allocator traces and
full tensor ownership confirm what disappears:

| Allocation/region | Expanded | Recompute |
|---|---:|---:|
| Maximum stage ap/av/f storage |13.070GiB |0 |
| X6 transient above larger region boundary |11.203GiB |0.073GiB |
| Largest region allocated peak |34.503GiB |15.895GiB |

Both peaks occur at stage1 pair-work accumulation on the third accepted step.
The profiling clocks are instrumentation costs:6.112s/2037ms per step expanded,
6.079s/2026ms per step recompute; do not use those for throughput. The measured
36-step allocator peaks above are slightly larger because their physical windows
are longer. Peak reserved memory is distinct from live allocated memory.
The separate obsolete first-step W retained through RunResult remains present,
as does the current W; this comparison does not include that ownership fix.

The initial copied profiler could not locate memoryAudit.py and exited before
physical stepping in both attempts. Preserve those logs/status/manifests. The
recovery changes only an explicit helper-path argument and reruns the two short
probes; physical runtime sources are unchanged. Recovered status is in
`memory-recovery/status.json`; the original status correctly retains failures.

## Numerical/provenance audit

- 864 measured raw RHS checks and12 memory-probe RHS checks are finite and
positive; all432 measured steps plus6 profiled steps have valid accepted states,
no extra RHS, and maximum relative energy-budget residual8.61e-9. Warmup steps
are separate from these raw-check counts.
- All timing initial states match bitwise. Refactored expanded matches original
endpoints at both sizes bitwise, including its sampled live W.96³ recompute also
matches every saved endpoint field and sampled W bitwise. Both repeats agree.
- 32³ recompute has small FP32 differences after36 steps: maximum velocity
4.17e-7, position1.19e-7, internal energy2.38e-7, density3.58e-7 and accepted
source-budget2.15e-11. Primary endpoint fields pass the declared rtol3e-6 /
atol3e-7 comparisons.
- The auxiliary divergence maximum difference3.219e-6 (3.452e-6 relative to
its field maximum) fails both the strict pointwise screen and the recorded
scaled-field screen. Retain that failure; do not describe the entire endpoint
comparison as passing or claim bitwise parity at32³.
- Two32³ recompute W samples are explicitly incomparable: live buffered-edge
counts differ by2 after the small state differences, so linear sample offsets
refer to different pairs. Do not subtract unrelated W entries or label them a
work error. The96³ W samples are directly comparable and bitwise identical.
- Complete forcing checkpoints, including knots/generator/parameters, match
bitwise across all12 measured runs.2,357 original campaign source pins and2,359
recovery pins verify unchanged; per-run inventories also remain stable.
- All12 timing videos' initial/final sheets and both memory-probe recordings
were inspected. Patterns agree, with no observed instability. Density is
strongly autoscaled in the nearly uniform96³ startup frames; these short flows
are performance controls, not developed-turbulence validation.

`comparison.json` retains full field errors, strict failed screens, incomparable
samples, both repeat timings and telemetry. Physical formulas are unchanged. Subsequent default qualification below changes
storage routing on native eager single-gas steps; other contexts retain the
reference path automatically.

![Timing and memory comparison](../.tmp/crk-shared-gpu-20261029/analysis/performance.png)


## Default adoption after the short comparison

User decision2026-10-29: the approximately3ms small-case cost does not justify
opt-in. Recompute is now normal for the qualified path, with expanded as an
explicit reference override. Timing does not establish launch overhead as the
small-case cost: this path adds pair arithmetic while removing balance/Torch
launches. No hardware-counter attribution is claimed.

The config constructor and loading configs without `crkPairStorage` now default
to `recompute`. Automatic reference fallbacks cover walls, graph/auto mode,
material tables, experimental CRK policies, derivative-bearing state, missing
compatible-work context and late differentiable external-force/dt inputs.
Native single-gas thermal Dirichlet projection/replay retains its tested behavior.

Evidence in `.tmp/crk-default-20261029/`:77 distinct CPU controls pass (2
unavailable-reference skips),26 GPU controls pass, and recorded existing Sod
and Gresho accuracy regressions pass unchanged thresholds. This includes fresh
midpoint/RK4, forced replay, Heun, three neighbour-support policies, checkpoints,
thermal projection/replay and differentiable external-force fallback.

Recorded accuracy costs include initial compilation and video/setup: Sod269
steps,31.548s/117.28ms per step, actual endpoint0.1497103; Gresho1051steps,
87.984s/83.71ms per step, endpoint1.4994251. Endpoints follow the existing CI
setup. Sod passes ringing/total-variation/entropy/density-error/shock-width/
energy bounds. Gresho passes kinetic-energy change/rebound, velocity error
and energy bounds. Recordings were inspected. Sod’s custom native1D profile
plot ignores the requested VisPy flag and uses Matplotlib; its recording cost
and that harness limitation are retained. Gresho uses VisPy.

The32³ auxiliary-divergence strict-screen failure and incomparable W samples
remain recorded above and in comparison.json. Adoption rests on primary-state
parity, unchanged formulas, fallbacks and the accepted physical accuracy gates;
it does not relabel that strict screen as passing. No physics, timestep,
precision or RunResult ownership change accompanies the storage default.


## Current96³ memory terms after default adoption

The measured36-step startup window uses16.006GiB peak live Torch allocation,
21.639GiB peak reserved allocator memory and22.479GiB sampled process memory.
At884,736particles the live high-water is18.97KiB/particle. These distinct
peaks need not occur simultaneously; reserved includes allocated storage.
The historical full-t4 maximum is not a new-path bound.

The separate3-step ownership probe peaks at15.895GiB. Its dominant tensors
are current X6 W5.601GiB; obsolete first-step W retained through RunResult
5.601GiB; buffered adjacency/metadata3.781GiB; one stored particle state
0.065918GiB; remaining live tensors/workspaces/allocator effects0.846GiB.
The last term is aggregate remainder, not exact attribution to a single routine.
State/stage copies, CRK coefficients, gradients and rates are included there.
Both large W allocations have distinct audited addresses; aliases do not
explain the duplicate.

There are224,755,007physical directed edges (~254.0/particle),455,639,040
buffered edges (515/particle), and501,202,944padded slots (566.5/particle).
The2^(1/3) radius buffer roughly doubles3D neighbour volume; capacity adds10%.
Each W slot is3 FP32 components (12bytes), so501,202,944*12=5.601GiB.
Two int32 COO indices cost8bytes/slot (3.734GiB), plus CSR rows and stored
geometry/hash metadata to reach3.781GiB. The particle fields themselves
are80bytes/particle plus a16byte source ledger (67.5MiB once).

Removing the obsolete W owner is the next independent target: subtracting
one W from the short-window live peak suggests ~10.40GiB, before neighbour
representation changes. That is a capacity estimate, not an implemented or
remeasured result. Evidence: `analysis/current-memory-budget.json`.
