# Operator, full-step and multi-GPU performance evaluation

Index: [PLANS.md](PLANS.md).

User decision, 3 November 2026: after the C++ native backend is integrated,
implement a reproducible benchmark suite for operators and complete simulation
steps, strong/weak scaling, and synchronization overhead relative to useful work.
User clarification: **CPU and single-GPU performance are the first priority**.
Distributed scaling follows only after those baselines are understood.
The user subsequently requested implementation and publication of the scripts
now, for comparison on an RTX 3090. The initial Warp suite is in
[benchmarks/performance](benchmarks/performance/README.md), with source-tree
workers in the core repository. Input/Morton storage and compute flags are
explicit sweep axes. Short local CPU/CUDA qualification is authorized; the stopped
Isambard run stays stopped. C++ and distributed scaling remain later phases.

## Questions the suite must answer

1. How much does the same work cost on one workstation GPU and one GH200 GPU?
2. How do workstation and Grace CPUs perform with one thread, matched thread
   counts, and each machine's supported parallel capacity?
3. Which operators are limited by arithmetic, device-memory traffic, irregular
   access, allocation, host dispatch, communication, or exposed synchronization?
4. How much of an accepted simulation step is useful device work, and how much
   is coordination, validation, transfers, imbalance and observation/output?
5. After single-device performance is understood, how do one, two and four GPUs
   change cost, and where do strong/weak scaling and replication limit efficiency?

The FP32 and memory-bandwidth differences between GPU families are hypotheses
about likely limits, not predictions of application speed. Record the exact GPU
variant, measured bandwidth, achieved traffic/throughput and arithmetic policy.
Separate ordinary FP32 work from TF32/tensor-core paths and FP64 accumulations.
A one-GPU GH200 baseline is essential: workstation-to-four-GPU timing alone
cannot distinguish hardware differences from distributed overhead.

## First gate: CPU and single-GPU performance

This phase comes before distributed scaling and does not wait for a distributed
C++ port. Begin once the relevant native CPU and single-GPU paths pass numerical
qualification. Keep single-device results usable independently of the later
multi-GPU suite. The initial report must explain the observed hardware comparison
operator by operator before treating a slower FP32 result as expected.

For genuinely bandwidth-saturated FP32 kernels, reduced peak FP32 arithmetic
alone does not explain halved application throughput. The expected direction
must be assessed from sustained bandwidth and measured bytes per operation.
Conversely, memory-related stalls may reflect latency/dependencies and insufficient
concurrency rather than saturation of aggregate memory bandwidth. Establish which
case applies instead of labeling every low-arithmetic-intensity kernel bandwidth
bound. A roofline-style lower bound uses measured work/traffic and effective
compute/bandwidth ceilings; whole-step host waits remain separately measured.

**GPU hardware controls and placement:** benchmark one workstation GPU against
one GH200 GPU, each with device-resident inputs/outputs and a working set larger
than the relevant cache. Measure sustainable streaming read/write/copy bandwidth,
then indexed gathers/scatters and neighbor traversal representative of our actual
layouts, degree distribution, vector widths and reuse. Record logical requested
bytes and profiler-measured DRAM bytes separately to expose wasted transactions,
reloads and cache effects. Check GPU clocks/power/throttling, architecture flags,
register pressure, occupancy, cache behavior and FP32/FP64/TF32 instruction mix.
Measure whether production particle/pair buffers reside in GPU HBM; identify any
host-mapped, managed-memory, migration or spill traffic explicitly.

The GH200 GPU's HBM3/HBM3e and Grace CPU's LPDDR5X are distinct memory tiers.
CPU bandwidth and GPU bandwidth require separate controls. CPU/GPU coherence or
addressability does not make host-memory access an HBM-bandwidth operation.
See [NVIDIA Grace tuning guide](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/).

**CPU matrix:** compare scalar/single-thread and matched thread counts first,
then thread-scaling through each CPU's available cores. Record physical-core/SMT
policy, affinity, NUMA placement/first touch, compiler/vectorization target and
sustainable memory bandwidth. Use the same neighbor sets, fields, precision,
operator output and allocation policy. Benchmark direct native C++ work and its
Python/Torch adapter separately. Distinguish CPU operator execution from CPU
control of a GPU workload; both can influence the observed GH200 step time.

**Measurement levels:** device/CPU operator execution; completed public-call
cost; full RHS; complete accepted step with retained neighbors; explicit neighbor
rebuild step; and ordinary evolution with measured renewal frequency. On GPU,
report device-event durations together with completed wall time and host gaps.
Compare standalone allocation/reuse, native/Torch and the Warp reference where
supported. Do not mix JIT/setup with steady-state timing or infer kernel speed
from the duration of a long adaptive trajectory.

For each large FP32 operator, report time, measured DRAM GB/s, fraction of the
machine's measured sustainable bandwidth, effective edges/s, traffic per edge
and the evidence for its limiting resource. Include FP64 as a diagnostic axis,
with identical numerical semantics and explicit accumulation precision. A result
near half the workstation's throughput despite higher available HBM bandwidth
requires a recorded explanation (traffic, latency, compute, host cost or another
measured limit) before moving on to multi-GPU tuning. Report unresolved cases
plainly rather than forcing an arithmetic-peak explanation.

## Integration gate and existing tools

Start implementation after the relevant native backend paths are integrated and
numerically qualified. Inventory actual backend dispatch for every measured
operation. Native/Torch, standalone native, Warp and any mixed route must be
labeled separately; unsupported distributed/native combinations are explicit
skips, not silently measured Warp fallbacks. Core primal integration alone does
not establish native CRK compatible-work or distributed full-step support.

Reuse the existing core `scripts/benchmarks/bench_backends.py`, its native CMake
bridge/analyzer, and operator inventory in the development archive. Reuse
frontend `benchmarks/common`, `benchmarks/particle_layout/bench_cuda.py` and the
existing memory/step profilers where their measurement contracts fit. Existing
native reports already distinguish completed public-call cost from standalone
allocation/reuse and exclude independent numerical verification from operators.
Extend that machinery into one versioned experiment manifest and result format.

Proposed frontend entry points under `benchmarks/scaling/` are `plan.py`,
`run.py` and `analyze.py`, with adapters to the core benchmarks and Slurm. These
are planned interfaces, not implemented commands. Plan generation should work
without a GPU. One manifest specifies fixture, backend, placement, policies,
repetitions, timing mode and resource/time limits; outputs retain raw samples,
JSON/CSV summaries, traces and source/build provenance.

## Workload matrix

| Layer | Timed work |
|---|---|
| Hardware controls | Device-memory bandwidth, bounded host/device copies, peer copies, collective latency/bandwidth across the actual message-size range |
| Neighborhood | Index/key generation, sorting, counts, scan, allocation, collection, filtering, support renewal, layout adoption and UID gathers |
| Operators | Density/interpolation/differential operators; CRK volume, moments, corrections, small matrix solves, pressure/viscosity and compatible-energy work; source terms and reductions |
| RHS and integrator | Complete native RHS, each stage, state updates/history, adaptive-support preparation and accepted-state finalization |
| Runner | Accepted step including timestep selection and mandatory distributed checks; diagnostics, checkpoints and visualization measured as additional modes |

Use uniform and evolved nonuniform particle distributions, at least 2D and 3D,
small launch-bound cases through large memory/edge-bound cases. Begin with FP32
CRKSPH/B7/n_h4 and the saved 128³ turbulence states. Extend to representative
CompSPH, weakly compressible and projection/implicit routes as supported. For
iterative/implicit routes report iterations, residuals and RHS/JVP counts so
changed convergence work cannot masquerade as faster kernels.

Hold particle state, UID set/order, support, physical neighbor sets, precision,
solver/integrator, forcing clock and timestep policy fixed within a comparison.
Pin actual physical and Verlet edge counts, degree distribution and padding;
particle count alone does not describe work. Include input/Morton order,
retained-list/filter-only steps, explicit full rebuilds and ordinary renewal
frequency. Chunk size, row caps, lanes/block size, index widths and allocation
reuse are declared axes, with comparable defaults before any per-device tuning.

Use saved initial and evolved states (including the retained Isambard states)
for bounded replay windows. Setup, state restoration and correctness checks are
outside a pure-step timing window; ordinary accepted evolution is also measured
separately so history/list reuse is represented honestly. Do not time different
flow ages or permit warmup to evolve the compared starts differently.

## Scaling contract

**Strong scaling:** keep the complete global problem and numerical work fixed;
measure one, two and four GPUs on the same GH200 node. Include both an ordinary
single-process/one-GPU baseline and the one-rank distributed route to expose the
process-group/framework entry cost. Use identical global snapshots and report
any change in per-step work or solver iteration counts.

**Weak scaling:** keep owned particles, physical neighbor degree and pair work
per GPU approximately constant while increasing ranks. Prefer deterministic tiled
fixtures with global volume proportional to GPU count, fixed particle spacing,
support and timestep. Refining a fixed box changes the CFL timestep and cannot
serve as the only weak-scaling experiment. If the ordinary case CLI only supports
cubic counts, use a dedicated fixture builder or report actual N/rank deviations.
For evolving cases also control local forcing/physical scales explicitly.

The current replicated-state design stores global particle fields per rank.
Therefore constant owned rows do not imply constant total storage or traffic per
GPU. Report owned rows/edges alongside replicated particles/bytes and actual
payload/control bytes. This should reveal the architecture's weak-scaling limits
rather than hiding them in an N/GPU label.

For elapsed time T and p GPUs report:

- Strong speedup S = T(1,N) / T(p,N), efficiency S/p.
- Weak efficiency = T(1,N1) / T(p,p*N1), with actual work counts alongside it.
- Particles/s, directed physical and candidate edges/s, RHS/s, accepted steps/s,
  time per simulated unit, memory/GPU and GPU-hours per unit work.
- Per-rank min/median/max work, elapsed time, payload and memory; slowest-rank
  and whole-step makespan are primary distributed timing results.

A missing one-GPU capacity point is a capacity limitation, not an extrapolated
strong-scaling baseline. Begin with one node; inter-node strong/weak scaling is
a subsequent separately qualified matrix with explicit topology and fabric data.

## Timing and synchronization attribution

Measure both completed wall time and device execution. Warm compilation/import,
context creation and clock ramp separately; retain cold-start costs as their own
records. Batch small operations and synchronize at measurement-window boundaries,
not after every operator in the throughput run. All ranks participate, with
per-rank elapsed durations and a coordinated whole-window makespan. Device event
clocks from different GPUs must not be compared as a common absolute timeline.

Use a separate instrumented pass with ranges for operators, integrator stages,
renewal, field exchange, control handshakes, checks and I/O. Collect:

- Device kernels/memcpy and GPU idle gaps; per-stream dependencies and launch gaps.
- Host launch/dispatch, allocation/free and Python/Torch/C++ wrapper costs.
- Explicit/implicit host waits: device/stream/event synchronization, scalar reads,
  `.cpu()`/NumPy transfers, hashing/serialization and polling for completed work.
- Collective and peer-transfer sizes, call counts, latency, bandwidth and actual
  overlap with computation; fixed control handshakes separately from payloads.
- Rank arrival skew and imbalance, including waits induced by a slower peer.
- Required algorithmic reductions versus optional diagnostics or validation.

Do not subtract summed kernel durations from wall time and call the remainder
synchronization. Kernels, communication, CPU work and waits can overlap. Report
critical-path exposed communication/waits and overlap separately, using a
CPU/GPU-correlated timeline; also retain cumulative busy/wait durations with their
non-additive meaning explicit. A blocked CPU thread can coexist with busy GPUs.

Use Nsight Systems for representative multi-rank timelines and Nsight Compute
for selected isolated operators, where supported. Quantify profiler/instrumentation
perturbation against ordinary runs. Use measured DRAM traffic and operation mix
for bandwidth/arithmetic analysis; label estimated traffic and FLOPs separately.
Synchronization can expose launch latency or imbalance as well as transport cost.

Benchmark production checks at their actual cadence. Independently qualify each
fixture before timing and check it afterward. Separately label measurements of
supported lighter-check policies or isolated validation cost; do not silently
remove required correctness/ordering checks or alter failure semantics to improve
results. Keep diagnostics/output off for compute-only runs, then add them back
at matched physical cadence to quantify their actual contribution.

## Experiment control and reports

Record source/dirty hashes, native compiler/toolkit/architecture flags, container
and package versions, GPU variant/UUID/memory, power/clock/throttling observations,
CPU type/thread count/affinity, rank/device/NUMA mapping, peer connectivity and
transport selection. Preserve precision, TF32/fast-math and accumulator defaults.
Run source-pinned matching builds for sm_90 and the workstation GPU; verify actual
backend/kernel dispatch rather than inferring it from a requested label.

Use bounded short runs, serial repetitions and repeated samples with distributions
(median/spread and sufficient independent repetitions for uncertainty estimates).
Keep desktop/other-job contention visible. Predeclare the sample/window and
resource budgets; calibrate short windows first rather than launching another
multi-hour trajectory. Numerical validation and working-set readiness precede
large allocations. The benchmark must report failed/skipped/capacity-limited
points and retain logs without retrying changed physics automatically.

Reports must include scaling/efficiency curves; operator versus full-step cost;
rebuild versus retained-list cost; per-rank imbalance; achieved bandwidth and
communication traffic; and timelines/fractions for exposed coordination. The
result should identify the limiting term and its work/GPU dependence, not merely
rank backend wall times. Preserve plain-machine-readable results and standalone
figures with exact experiment provenance.

## Implementation checkpoint — 3 November 2026

The initial suite is implemented for the existing Warp backend and documented in
[benchmarks/performance/README.md](benchmarks/performance/README.md). It includes
fresh-process precision/thread isolation, memory controls, all 17 existing core
operator/matrix stages, build/query/rebuild search phases, production accepted
steps and a separate instrumented RHS path. Input/Morton storage, CSR/chunked
topology, lanes, execution and convergence-check flags are explicit, recorded
choices. Numerical/finite-state checks, raw samples, capacity skips, timeouts,
source digests and CSV/Markdown/JSON comparisons are included. Profiling is a
separate pass; core hooks are disabled during the base timing.

Current limitations are explicit: the single-device driven Morton/source adapter
requires chunked lists; MFM/MFV Morton is unqualified; per-call RHS instrumentation
requires input/eager execution. Restart fixtures, actual DRAM-counter analysis,
C++ harness integration and distributed scaling remain subsequent work. The
stopped Isambard run's configuration and run manifest confirm Morton/eager with
CUDA graphs disabled. No remote run has been resumed.

## Execution sequence

1. Initial Warp manifest/timing schema and CPU/CUDA smoke controls are implemented.
   Extend the same reporting contract to qualified native CPU and single-GPU paths
   when the separate C++ backend is integrated.
2. Measure CPU memory/thread-scaling and GPU streaming/indexed-access controls;
   compare matched operator/RHS/accepted-step fixtures on workstation and GH200.
3. Profile the single-device cases and publish achieved bandwidth, traffic,
   device versus host time, and explanations for material FP32 performance gaps.
   This is the first performance gate; distributed work does not replace it.
4. Once the relevant distributed native paths are qualified, measure one-rank
   framework overhead and transport/host-copy/control baselines.
5. Run bounded one/two/four-GPU strong and weak scaling, including rebuild and
   evolved-state windows, and validate all endpoints/work counts.
6. Account for exposed synchronization, communication, imbalance and observation
   costs without double counting overlap; publish raw data and bottleneck findings
   before selecting tuning changes or authorizing longer/multi-node experiments.

The stopped 0.7.1 run is motivation and retained evidence, not a controlled scaling
benchmark: the local run used different runtime/build, chunk sizes and observation
policies. Keep its partial diagnostics/checkpoints available for fixture selection.
