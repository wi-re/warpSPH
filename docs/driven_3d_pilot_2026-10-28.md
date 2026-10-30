# Driven3D performance pilot (2026-10-28)

CRKSPH from rest, seed1234;32³,64³,96³ to t4; full accepted-state exports every0.25. Unchanged finite-gamma OU/bath/B7/n_h4/native fresh midpoint model. Runs launch serially only after unrelated GPU work is terminal and GPU compute is free.

|Resolution|Particles|Steps|Loop seconds|ms/step|Seconds/simulated second|Torch peak GiB|Sampled process peak GiB|Snapshot MB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|32³|32,768|985|35.12|35.65|8.78|1.590|2.557|26.29|
|64³|262,144|2,011|376.19|187.07|94.05|12.504|14.990|209.59|
|96³|884,736|3,040|1919.04|631.26|479.76|42.097|48.010|705.44|

3/3 endpoints audited. Configuration/source/common-forcing-knot parity: {'configurationEqualExceptResolutionOutput': True, 'sourceInventoriesEqual': True, 'commonForcingKnotsBitwiseEqual': True}.

Loop wall includes setup plotting, accepted/RHS diagnostics, sparse exports and final video/checkpoint handling after hydro construction. Setup/first compilation and warmup are separate from this loop measurement. Per-step medians include the stated observers. Whole-device/process memory samples can miss transient peaks; exact Torch high-water marks do not measure all device allocations.

Every retained particle frame has been checked for finite fields, positive mass/support/density/pressure/u, actual accepted timestamps and shape. Raw RHS and accepted guards, no-extra-RHS and source stability are required. Compression and export cost are in each performance.json; recording cadence is not a training-data temporal-resolution qualification.

The first32³ run also saves native initial/final restart artifacts; complete3D restart-continuation parity remains a separate gate. Slice video shows z=0 interpolated density and speed, not a projection or the full volume. Initial/midpoint/final contact sheets for all three runs and the history plot were visually reviewed2026-10-29. No visible blow-up/ejection in these representative slices; this is not full-volume inspection. Per-frame colour limits are automatic, and the initial density panels magnify small interpolation/lattice variations.

![Histories](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-3d-pilot-20261028/analysis/histories.png)

No stationarity, broad inertial range, native viscosity or production training qualification is claimed. Evolved-state throughput and peak neighbour memory can exceed these startup measurements.

Evidence: /home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-3d-pilot-20261028


## Completed review (2026-10-29)

All three simulations completed overnight; no trajectories were rerun. The
supervisor then failed before CPU analysis because dict() received duplicate
thread-count keyword arguments when merging the job environment. Preserve
original manifest/supervisor/log/status; a corrected mapping-unpack version
is archived separately. The sealed CPU analyzer was run directly with
CUDA_VISIBLE_DEVICES empty. The final queue status now records successful
offline recovery. All1,178 original executable/input pins remain unchanged.

All51 particle snapshots pass shape/finite/positive checks; all12,072 raw RHS
evaluations are valid, with no extra RHS. The largest relative energy-ledger
residual is2.29e-8. Native mass is unchanged throughout each run. Maximum
mass RMS Mach is0.372/0.415/0.430, and final values are0.338/0.391/0.406.
These are startup values, not stationary means. Source inventories and physical
configuration match across resolution; common OU knot prefixes are bitwise equal.

At96³, the17 compressed snapshots occupy705.44MB. Transfer/compression/write
cost is22.80seconds,1.19% of the1919.04second loop, so this sparse output is
not the dominant cost. Sampled process residency peaks at48.01GiB, whole-device
used memory at53.82GiB (including other non-compute allocations), and Torch
allocated/reserved peaks at42.10/47.22GiB. No unrelated compute was observed.
The device has95.59GiB total VRAM; this startup pilot fits with useful headroom,
without establishing a later clustered-state peak bound.

From64³ to96³, particles increase3.375×, mean step cost increases3.375× and
the number of accepted steps increases1.512×: loop time grows5.10×. Step costs
include guards, accepted-power measurement, video and sparse output.32³ has
notable fixed startup/compilation cost: total child wall87.98seconds versus
35.12seconds measured loop.64³ total child wall384.74seconds versus376.19loop;
96³ total1931.99seconds versus1919.04loop. These are separate measurements.

96³ currently costs about8wall minutes per simulated second. A linear t40
projection is about5.3hours, conditional on unchanged throughput/CFL; this is
not a measured long-run forecast. Evolved turbulence may require smaller steps.
The case is feasible at96³, but longer-run stationarity, dissipation/cascade
resolution and3D restart-continuation checks remain open. No larger run started.


## Memory concern: initial source/storage audit (2026-10-29)

At884,736 particles,42.10GiB peak live Torch allocation is49.89KiB/particle;
47.22GiB peak reserved is55.97KiB/particle; sampled process48.01GiB is
56.90KiB/particle. Peaks need not be simultaneous. This is substantial live
allocation, not explained solely by allocator caching. The exported state
contains64bytes/particle; this is not a count of all runtime per-particle fields.

Source inspection finds per-directed-edge pressure/viscosity3-vectors, partition
fractions and compatible-work weighted vectors, including Torch intermediates,
at padded adjacency capacity. The32³ native final checkpoint has18,169,368
directed candidate edges/18,563,072 capacity (554.5/566.5 perparticle), with
int32 indices. The Verlet radius factor2**(1/3) doubles spherical search volume
relative to physical support. At similar occupancy,96³ would have roughly
490million directed candidate edges; a single FP32 pair3-vector would then
occupy about5.5GiB. That96³ edge count is an extrapolation, not measured.

Pair-sized buffers and temporary liveness are concrete targets for profiling
and implementation optimization. No allocator-by-site breakdown, memory leak
verdict or verified reduction is claimed. No GPU job was started for this
read-only audit; preserve native compatible energy, fresh stages and support.


The source-only memory estimate is superseded by the measured
[CRK ownership audit](crksph_memory_2026-10-29.md): particle fields80bytes each,
large per-pair outputs/temporaries, and an additional first-step X6 work vector
retained through RunResult.state. The startup probe has not reproduced every
late full-run high-water event. No optimization was applied.
