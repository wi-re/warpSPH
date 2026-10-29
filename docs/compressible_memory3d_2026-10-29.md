# Native 3D compressible memory scaling (2026-10-29)

Follow-up: native CompSPH work recomputation and runner lifetime cleanup are now
qualified. CompSPH96³ live peak is10.100GiB; other schemes save0.14–0.21GiB.
Historical sweep values below are unchanged. [New implementation/results](compsph_work_lifetime_2026-10-29.md).

User requests committing the completed turbulence/CRK work, then measuring memory at scale across the other compressible schemes. The completed work is committed as `931aca9`. This campaign changes instrumentation only.

Seven supported driven-flow schemes: CRKSPH (recompute control), CompSPH, Monaghan, GSPH, InutsukaGSPH, MFM and MFV. PESPH is outside the current driven case’s supported-scheme contract; no new scheme/source support is inferred by this memory test.

Each scheme has independent-from-rest 32³/64³/96³ windows of 36 accepted steps, separate three-step warmup, first six steps discarded for steady timings. Then a separate six-step 96³ allocator/ownership capture, with matching accepted-step6 primary stored-field hashes to check observation equivalence. Smaller-resolution failures block dependent larger jobs. Other schemes continue; failures are retained without automatic retries or numerical tuning.

All use the retained periodic3D box, gamma5/3, rho1, c1, seed1234, band1–2 OU acceleration0.3/tau0.5/knots0.05, mass-mean subtraction and relaxation bath u0.9/tau0.2. Native support, density, stage and energy policies are preserved. SPH uses native midpoint/CFL0.15; MFM/MFV use the supported stage-consistent SSPRK path/CFL0.1, no neighbour-skin optimization. The common configured kernel is B7/n_h4; Inutsuka additionally evaluates its Gaussian convolution and native extended pair-list reach. GSPH uses its native shockSwitchC0 default; Inutsuka uses C3. Resolved configurations are archived. These are current driven-cohort defaults, not an equal-error or matrix-preset accuracy comparison.

Report peak Torch live/reserved and sampled own-process residency separately; compute KiB/particle and actual neighbour edges/padded slots. Read-only inventories deduplicate storage aliases and separate stored particle state, stage pair arrays, adjacency, compatible W, meshless geometry and other system data. Allocation stacks/regions expose transient peaks not reachable at observer boundaries. Global high-water marks are not sums of independently measured group peaks. Captures are explicitly instrumented and are not throughput benchmarks. No per-step empty_cache or allocator tuning is introduced.

GPU jobs launch serially through a persistent tracked supervisor, only with no unrelated compute user visible. Physical progress/stall/wall guards, every-RHS positivity/finiteness, no-extra-RHS, accepted positivity, signed energy ledger, stable source hashes and recorded vispy slices are required. Own-process telemetry records later contention without changing other sessions. Current startup windows do not bound memory for evolved/clustered turbulence or certify training-data quality.

Tiny CPU controls pass for all seven schemes. Original MFM/MFV observer failures (attempted finiteness check of an unset derived velocity field) are retained; the observer now reads their stored conserved vector and native stage primitives. CRKSPH/MFM six-step capture-path CPU controls also pass, including neighbour accounting. Native solver code is unchanged. Tiny CPU controls omit video; GPU windows record slices.

Artifacts: `.tmp/compressible-memory3d-20261029/`. `manifest.json` seals 28 finite jobs and exact runtime/input/driver hashes. `status.json` and `supervisor.pid` track the GPU queue. `analysis-manifest.json`/`analysis-status.json` track a finite CPU-only companion that automatically analyzes terminal results and preserves failures. `analysis/summary.json`, `analysis/report.md`, scaling plot and native memory-viz HTML are its outputs. Raw snapshots remain local. The initial GPU launch waited for unrelated ACSPH compute observed during setup. The finite campaign and CPU companion have now completed successfully.


## Completed measurements and review

All28 jobs pass:21 ordinary36-step windows and7 separate six-step96³ captures.
All1,596 observed raw RHS are valid, no extra RHS, accepted state remains finite
and positive, maximum absolute relative signed-budget residual5.33e-8.
All seven capture/native stored-field hashes match bitwise at accepted step6.
All1,216 runtime/input/supervisor pins and both analysis-worker pins remain
unchanged. No solver modifications or trajectory retries were needed. All28
recordings were reviewed via first/last contact sheets at32³/64³/96³ and for
the captures; smooth common startup velocity response, no visible blow-up.
Density colour limits amplify tiny lattice/interpolation and scheme differences;
this is not a full-volume or developed-flow inspection.

|Scheme|32³ live peak GiB|64³ live peak GiB|96³ live peak GiB|96³ reserved peak GiB|96³ sampled process peak GiB|96³ live KiB/particle|
|---|---:|---:|---:|---:|---:|---:|
|CRKSPH|0.413|3.088|10.333|13.396|14.186|12.25|
|CompSPH|1.097|8.557|28.791|29.971|30.986|34.12|
|Monaghan|0.339|2.492|8.333|9.797|10.555|9.88|
|GSPH|0.338|2.487|8.316|9.801|10.557|9.86|
|InutsukaGSPH|0.338|2.487|8.316|9.803|10.559|9.86|
|MFM|0.420|3.147|10.533|10.711|11.465|12.48|
|MFV|0.420|3.147|10.533|10.711|11.465|12.48|

The96³ runs contain884,736 particles. Near-constant KiB/particle from64³ to96³
shows approximately linear live-memory scaling over these startup cases,
with pair occupancy and padded capacities nearly fixed per particle. Full21-row
runtime table, native step timers and each job's wall time are saved in
`analysis/report.md` and `analysis/summary.json`. No equal-accuracy performance
ranking follows from these windows.

|Scheme at96³|36-step run wall seconds|Steady wall ms/step|Final simulated time|Six-step capture wall seconds (instrumented)|
|---|---:|---:|---:|---:|
|CRKSPH|21.276|533.97|0.052325|5.653|
|CompSPH|17.366|426.14|0.052325|4.882|
|Monaghan|20.647|509.07|0.052325|5.578|
|GSPH|14.425|344.12|0.052325|4.504|
|InutsukaGSPH|12.669|295.62|0.052322|4.250|
|MFM|7.069|139.97|0.037457|3.227|
|MFV|7.073|140.64|0.037457|3.194|

The actual GPU queue ran11:02:29–11:10:19 UK time, after the initial resource
wait; sum of whole-child job wall times468.56s. Capture timings include history/
ownership overhead and are not throughput. Only CRKSPH32 had unrelated compute
observed in sampled telemetry; its timing is contended. Other timing rows still
represent single short windows with observer overhead, not robust repeated
benchmarks. The user subsequently authorizes concurrent accuracy/memory probes:
future sweeps may overlap other GPU work; retain per-process measurements,
label contended timings and distinguish shared-headroom OOMs from intrinsic
scheme limits. No completed trajectory was rerun for this policy update.

## Where the memory goes

CRKSPH and CompSPH retain the same5.601GiB compatible-work vector and3.781GiB
native padded adjacency, plus0.0659GiB particle state. Their post-run reachable
groups total about9.448GiB, while Torch owns about9.550GiB. CompSPH's much higher
28.791GiB peak occurs with expanded ap/av/f and pair-work expressions. Allocation
stacks show5.601GiB pressure and viscous3-vectors,5.601GiB pair-vector temporaries,
a1.867GiB pair scalar, and W coexisting. The CRK optimization has not been
applied to CompSPH. This makes CompSPH's expanded force/work path a concrete
next optimization candidate, subject to its own native-formula and energy gates.

Monaghan/GSPH/Inutsuka retain one3.781GiB list and about0.066–0.073GiB particle
fields. Their8.316–8.333GiB peak includes four distinct1.867GiB COO buffers:
two neighbour lists coexist. Ownership paths include the accepted/current
system and the previous runner stepResult's stage data. This is evidence of
overlapping list lifetimes, not evidence that one can safely remove a list
without a lifecycle and numerical check. Post-run Torch allocation is3.948–
3.954GiB. Inutsuka's Gaussian reach does not require more padded slots than
the existing native buffer in these startup windows: all five SPH methods use
455,639,040 live directed candidates /501,202,944 padded slots. This equality
is specific to this support/spacing regime.

MFM/MFV post-run Torch allocation is2.173GiB; reachable groups include0.0758GiB
primary particle state,1.935GiB geometry including its list, and0.0758GiB other
system data, about2.087GiB total after alias deduplication. Each accepted
geometry has about224.75million physical edges and247.22million padded slots,
roughly half the SPH buffered capacity. Native geometry and neighbour-search
allocations coexist transiently: the separate six-step traces contain several
pairs of roughly0.93GiB index arrays. Their six-step peak8.645GiB is lower than
the36-step peak10.533GiB; do not use the short trace as an exhaustive attribution
of the later maximum. Shorter captures preserve physics and step6 hashes but
cannot characterize every later allocation. State-owned geometry and runner/
stage lifetimes are candidates to inspect before changing neighbour formats.

All seven captured endpoints contain zero active-awaiting-free bytes and no
pending free requests; completed frees have adjacent request/completion events.
Much of reserved memory is inactive cache. Exact block-state snapshots and
native Torch timeline/segment views are retained per scheme. A Python ownership
inventory at a region boundary is not an inventory at the true global peak:
short-lived temporaries may have gone and meshless/foreign containers may not
be recursively traversed. In particular, `peakRegion` in the initial automatic
summary names an observer that noticed the global high-water, not necessarily
the allocating region. Use allocation stacks/traces for attribution;
`analysis/trace-peaks.json` provides an approximate event-size reconstruction
(the small allocator rounding differences remain distinct from Torch counters).

The sweeps establish that all seven native implementations fit at96³ in these
startup windows and expose differing memory costs. They do not validate long
3D turbulent dynamics, stationary data, all-volume accuracy, alternative Godunov
matrix presets, or later clustered-state maxima. No further GPU run or solver
optimization is included in this review.

Evidence: `analysis/completion-review.json`, all source-pinned runs, native
Torch HTML/snapshots, and `analysis/contact-{32,64,96,96-capture}.jpg`.
