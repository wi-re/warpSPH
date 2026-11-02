# Native solver layouts and graph integration — 2026-11-02

Fixed-count Morton layouts now run through the existing WCSPH, ACSPH, IISPH
and DFSPH runner paths, including the predictor and shift compositions. The
input-order default remains selected. No numerical formula or coefficient was
changed. Population events and physical inlet/outlet conditions remain deferred.

## Application boundary and integrator contract

`NativeMortonLayout` prepares the complete next system/configuration between
accepted steps. Initial adoption sorts the state; later adoptions happen only
when its current Verlet list is stale and the Morton permutation changes. Every
native stage sees its actual input order. Native search may rebuild a list within
a step without changing the order. The adapter drops old search and row caches
when adopting a layout, so indices never refer to a different row basis.

The existing explicit/map integrator and native finalization run unchanged.
ACSPH remains one exact outer Euler map around its internal pseudo-time solve;
its six position/velocity history fields and previous dt travel with the system.
DFSPH retains all three physical pressure/impulse warm-start fields. Delta-SPH
surface-mask memory and lazy diagnostic particle axes are declared explicitly.

CompSPH/CRKSPH graph requests use this same accepted-boundary route for one-step
explicit methods. Their existing eager `MortonStepper` route continues to carry
UID-mapped multistep history, implicit seeds and FD-JFNK continuation. An implicit
or multistep request cannot silently enter the native graph route. Particle-valued
tolerances, arbitrary solver contexts, exact JVP and implicit differentiation
remain guarded until their respective adapters are qualified.

## Bodies, inputs and observations

Ghosts receive fresh nonnegative UIDs in Morton runner initialization. Legacy
input-order initialization keeps its old UID convention. Reordering conjugates
the reciprocal ghost/boundary relation. Rigid-body geometry retains compact local
order; global body and ghost scatter rows are joined by persistent UID. Empty
analytic body rosters may use the boundary library's int64 label type alongside
int32 fluid labels.

Analytic providers retain their pointwise geometry and rebind owned body objects.
The standard analytic boundary provider, declared UID-valued callbacks, planes,
all-particle regions and pinned-band geometry are admitted. Arbitrary stateful
callbacks or custom providers require an explicit adapter. Four kinematic body
steps through repeated permutations, analytic distance permutation and finite /
unbounded analytic body pose restarts have regression fixtures. These are adapter
checks, not new moving-wall physics validation.

Particle frames and native stage states/derivatives export in ascending UID
order. Stage CSR rows, columns and build anchors follow that state. Trajectory
headers also rebind body scatter rows, preserve fractional ghost offsets and
declare particle/global axes. The native delta-SPH bundle now names the actual
weakly compressible update type, fixing its stage-import mismatch. ACSPH's named
dynamic position tensors import separately from constructor fields.

## Numerical restart memory

The atomic `nativeMemory` group adds ACSPH BDF history (incremental and absolute
position variants), previous/next dt, delta surface masks, rigid-body poses and
loads, and the adaptive pseudo-time convergence schedule. The last item matters:
ACSPH's remembered first-converged iteration determines subsequent host stopping
checks. Losing it changed the next accepted pressure even with identical particle
and BDF fields; preserving it restores continuation.

The spatial seed explicitly labels compute-order UIDs and numeric list-build
positions/supports. Restart reconstructs the previous list generation from these
anchors, without serializing CSR, pointers, graph objects or executable search
caches. Physical state and history remain UID ordered. Static numerical settings,
body-local geometry and supported analytic representations are fingerprinted;
callbacks/geometry are reconstructed from the same configured case. Infinite
mass/inertia of an unbounded static analytic wall are admitted only as those
declared constants. Particle, pose and numerical history values must be finite.

## CUDA graph behavior

Adoption invalidates captured row addresses, clears the private RHS graph and
forces warm-up/validation before recapture. Stable layouts retain replay. Native
fixed host dt inputs become device scalar inputs to the existing whole-step
wrapper. Pipelined outputs retain the preceding step's config/body view while
the next step uses its newly adopted layout.

A host boundary-kind read in the viscosity reference helper prevented capture,
including on a fluid-only case. It now uses the existing layout-scoped boundary
flag and cached ghost rows. This preserves the calculation and removes repeated
host reads. Existing graph eligibility restrictions still apply; storing full
stages continues to use eager whole steps, while ACSPH can retain iteration graphs.

Retained validity flags share the runner's existing readback. Whole-step graphs
capture the read-only final-envelope observation **after** the numerical step;
its flag travels through the existing deferred-decision packet. A stale endpoint
requests preparation of the next attempt, rather than rejecting the completed
step. Post-step position/support changes invalidate that observation and compute
a fresh flag. Native transitions preserve declared constant UID buffers; later
hook writes/replacements undergo a value check. Both mutation paths have tests.
Liu interpolation now uses a private support view instead of replacing the
physical state's support tensor during observation, preserving cache ownership.

The forced-resort graph fixture records **two captures, fourteen replays and zero
fallbacks**, both sequentially and with pipelined output. Graph/eager particle
results are bitwise equal when evaluated in the same adopted layout. Sampled
walls retain replay and reciprocal ghost relations. CompSPH and CRKSPH each
record one capture and eleven replays in the short runner fixture.

## Qualification and precision

Runs use the normal `warp` conda environment and the RTX PRO 6000 Blackwell GPU
outside the sandbox, one GPU run at a time. Sources are frozen for GPU runs.
These are short repeated numerical fixtures and microbenchmarks, so video/viewer
registration is deliberately omitted. Cold Warp compilation is included in the
first qualification wall times; those times are not throughput measurements.

Recorded checks:

| Selection | Result | Wall time |
| --- | --- | --- |
| CPU solver/layout/continuation/runner/graph regressions | 454 passed, 9 CUDA skips | 73.82 s |
| FP64 CPU native/boundary/I/O/solver-input/runner selection | 92 passed | 10.79 s |
| CPU native/runner/graph follow-up | 43 passed, 8 CUDA skips | 7.89 s |
| Float32 GPU native/history/graph/wall, adaptive-H/compatible 2D/3D and existing default graph regressions | 18 passed | 12.06 s, warm caches |
| FP64 GPU native/history/graph/wall and adaptive-H/compatible 2D/3D | 12 passed | 10.79 s, warm caches |

The evolving GPU fixtures compare four accepted steps against native physics
with adaptive supports and compatible-energy replays, force a list rebuild and
periodic translation, and check positive internal energies in 2D and 3D for both
CompSPH and CRKSPH. Native solver fixtures also compare physical state and resumed
history. This extends the earlier fixed-H, compatible-energy-off large timing
workload; it does not establish long-time flow accuracy for every combination.

Reordering can change floating-point reductions. On the coarse 8×8 periodic
IISPH fixture, float32 Morton ordering changes velocity by at most 2.16e-5 and
pressure by 3.63e-3 after three translated steps. A deterministic random-row
control changes velocity by 2.76e-5 and pressure by 5.47e-3. FP64 uses the strict
comparison. Sampled-wall float32 pressure checks account for the acoustic EOS
amplification of a few density ulps by c0². These are explicit precision bounds,
not a claim of bitwise input-order equivalence or an alteration to the solver.

## Retained-layout timing

Three paired repetitions use a fixed-support, unshifted periodic delta-SPH TGV,
24 steps per run, light KE diagnostics and sequential outputs. The first five
steps are excluded. Values below are medians of per-run steady **whole iteration
wall time**, including layout flags, readback and bookkeeping. The runner's
integrator-only timer would hide checks performed after the numerical step and
is not used for the performance conclusion. All 24 timing runs total 2.606 s,
including setup; individual wall times and step timings remain in the JSON.

| Particles | Execution | Input order, ms/iteration | Morton, ms/iteration | Median paired extra cost, ms |
| --- | --- | ---: | ---: | ---: |
| 4,096 | eager | 3.769 | 4.433 | 0.657 |
| 4,096 | whole-step graph | 0.493 | 0.736 | 0.246 |
| 16,384 | eager | 3.803 | 4.487 | 0.638 |
| 16,384 | whole-step graph | 0.618 | 0.889 | 0.272 |

Capturing the observation reduced the small graph path's additional iteration
cost from about 0.52–0.58 ms to 0.25–0.27 ms. It retains actual replay, but these
small cases remain slower with sorting. This is a measured limitation, and the
input-order default stays selected. Further reductions need packet/bookkeeping
profiling or a size-dependent policy; the earlier million-particle CompSPH/CRKSPH
benefits do not establish a native WCSPH crossover.

Input/Morton timing controls have maximum differences of 0 in position,
2.38e-7 in velocity, 1.19e-7 in density and 1.19e-5 in pressure. These short runs
qualify the adapter and cost, not long-time physical accuracy.

Evidence lives under `.tmp/particle-layout-2026-11-02/native-solvers/`; source
manifests, final qualification logs and timing results are retained there.
The final frozen source/test manifest contains 780 files and hashes to
`889ae92912fb5c33239b415fe87e99093343d1d0fe8b260b5ba8fbc81dbdc357`.
The first cold FP64 GPU pass also passed all 12 new cases in 462.19 s, including
Warp kernel compilation; the final-source rerun uses those caches.
