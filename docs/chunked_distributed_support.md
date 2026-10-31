# Eager row-chunk distributed support

This is an opt-in, bounded proof of concept using replicated particle fields and
owned whole CSR rows. The implementation uses PyTorch c10d: Gloo for CPU debugging
and NCCL for CUDA. Ordinary single-device runner defaults are unchanged. The
[canonical plan](../CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md) records the implementation
sequence, qualification evidence and remaining capacity/site gates.

## Supported scope

| Context | Locally qualified scope | Limits |
| --- | --- | --- |
| Monaghan | Eager 2D/3D particle operators and solver | No chunked walls/materials/AD/graphs |
| CRKSPH, CompSPH | Eager native compatible work; chunk-owned W; fresh-stage replay after a list change | Native uniform-material controls; no expanded global pair fallback |
| GSPH | Eager 2D/3D self queries; first/second-order reconstruction | Global self-pair identity is supplied separately from local query indexing |
| InutsukaGSPH | Eager 2D/3D Gaussian density plus first/second-order force/energy | Compact and Gaussian lists remain distinct; Gaussian reach uses an agreed eager scalar |
| PESPH | Eager periodic 2D/3D, both variants, number-density supports/grad-h, native RK2/RK4 and OU driving | Pressure-energy supports the existing relaxation bath; pressure-entropy driving requires adiabatic mode; no walls/materials/AD/graphs |
| MFM/MFV | Eager native conserved stages, global canonical flux identity, projected CG/MG closure and owned face weights, OU sources | Native temporal defaults preserved; driven sources require SSPRK; no chunked walls/materials/AD/graphs |
| deltaSPH, Sun2017 | Eager periodic native density stages, diffusion, pressure-sign switch and delta shifting in 2D/3D; managed 2D tgv-wc case | Sun2017 frozen diffusion and full shifting strength retained; no walls/free surfaces/materials/AD/graphs or other shifting methods |
| IISPH, DFSPH / predictor / shift | Eager periodic native C/F/diagonal, dependent pressure passes and post-drift resampling; managed 2D tgv and separate native 3D controls | Native budgets, warm starts and stopping norms retained; no walls/free surfaces/materials/external shifting/Band geometry/AD/graphs |
| Prepared operators | Shared opted-in operators, native work, PESPH pressure passes, MFM geometry/closure/fluxes, delta operators, GSPH, Gaussian density and Inutsuka | Explicit query/reference/chunk/dummy roles; primal eager execution |
| Neighbour lanes | Fixed scalar and global 32-lane CUDA configurations | Existing tiled kernels are used where available; Godunov/Gaussian kernels use scalar fallback |
| Native checkpoint | Accepted-step drivenTurbulence for eight compressible schemes; periodic 2D tgv-wc for delta/Sun2017 and tgv for four native projections | Native particle/time/search history; driven cases retain FourierOU/source ledgers; delta uses its own density/mask-history schema and never stores frozen diffusion or pair arrays; projections save native pressure impulses and nominal/next timestep clocks |
| Legacy checkpoint | Qualified version-1 CSR/COO import for Monaghan/CRKSPH/CompSPH | Godunov legacy import explicitly rejects; use the native format |
| Actual wide-edge addressing | Up to2,500,000,000 real neighbours across38 bounded chunks; dedicated J/work; eager/prepared density; filtering and owner transfers | Dense debug geometry, one GPU, standalone/two NCCL ranks, both local offset widths; per-chunk extents remain bounded |
| Process/rank changes | Gloo 1–5 ranks and tiny two-process/one-GPU NCCL controls | Physical device-count changes and multi-node topology remain unqualified |

The dedicated launcher currently accepts `drivenTurbulence` with these eight
compressible schemes. Qualified PESPH pressure-entropy driving uses `solver.variant=PressureEntropy`,
`solver.dissipation=false` and `params.thermalModel=adiabatic`; no entropy-bath
policy is selected automatically. Native checkpoint recovery additionally covers bounded 2D/3D CRKSPH,
CompSPH, PESPH, MFM/MFV and delta/Sun2017 allocation failures; a bounded164³ failure/restart with more than two billion neighbours also passes.
Full-duration large-flow stability remains a separate gate. GSPH/Inutsuka operator controls
include reconstructed inputs and the existing optional time-centred energy
second pass. This adds no new time-integration or time-centred RHS method.

## Selection and execution

Use an explicit `RowChunkPolicy`/`buildChunkedVerletList` Python neighbourhood
hook, with an optional `RowChunkProcessGroup`. Budget J and retained W together
for native compatible work. The communicator is explicit and caller-owned; the
library does not initialize a global process group. The dedicated entry point is
`scripts/run_distributed.py`; a minimal CPU debugging command is:

```bash
PYTHONPATH=/path/to/warpSPH/src:/path/to/warpSPHCore/src \
python -m torch.distributed.run --standalone --nproc-per-node=3 \
  scripts/run_distributed.py --configuration driven.json --backend gloo \
  --output /fresh/shared/output --edge-budget 1024 --max-rows 3 \
  --checkpoint-every 1 --replica-checks exact --offset-dtype int64
```

The configuration must select eager execution and disable automatic
store/plot/video, CUDA graphs and output pipelining. Checkpoints and rank-zero
diagnostics are managed by this launcher. Use a fresh shared output directory;
existing outputs reject rather than silently mixing runs. The manifest records
actual package source hashes, precision, offsets, backend, rank placement,
loaded NCCL library/runtime, GPU UUIDs and diagnostic timings. Pin both source
checkouts explicitly; precision and offset width are chosen before importing
core. Inspect `--help` and the canonical plan for configuration and site templates.

For physical CUDA execution, use one process per visible GPU and `--backend nccl`;
the launcher selects `LOCAL_RANK`. The local debugging exception uses
`--shared-gpu`, at most two local processes, a runtime with shared-GPU rank support
and `NCCL_MULTI_RANK_GPU_ENABLE=1`. The qualified isolated runtime is NCCL 2.32.3,
under `.tmp/nccl-shared-gpu-20261030/with-nccl-2.32.sh`; it leaves the installed
Torch/NCCL stack unchanged. This is a correctness/stability fixture on one device.
No multi-GPU speedup or transport-bandwidth conclusion follows from it.

## Consistency, storage and failures

`exact` is the default replica-check mode. It hashes replicated operator inputs.
`contract` checks operator layouts/scalars and the effective kernel/lane choice,
while exact checks remain at search/rebuild, initial/accepted-state and checkpoint
boundaries. It relies on deterministic replicated stage inputs between those
boundaries. Differing effective reduction settings reject before computation;
requests that both resolve to the same scalar fallback remain compatible.

Each rank owns only its J and native W chunks. Dependent particle results are
exchanged before the next pass reads them. Filtering counts old owned rows,
agrees the new plan and transfers bounded J slices directly from source owner
to target owner. It generates optional global I locally. One source chunk of
filtered J/local offsets is temporary; old and new resident generations coexist.
Old source leases/J/W remain valid. A changed list identity triggers the existing
fresh-stage compatible-work replay, preserving Butcher stages and equations.

Replicated particle fields, counts, the int64 prefix and output fields remain
O(N). Logical accounting also includes overlapping lists and a fixed-capacity
particle packing cache (default 4 MiB per process). Bounded per-chunk addressing
and scratch do not bound total process memory or guarantee a problem will fit.
Checkpoint startup currently overlaps a seed list with the restored list.

Local validation/allocation exceptions are coordinated before payload operations.
The tests exercise retry after recoverable work allocation and transfer-contract
errors, and launcher termination after sender/receiver death. Device faults,
process loss or failure during a collective require process-group timeouts and
the launcher to terminate peers; this is not communicator fault recovery.

## PESPH sources and restart

PESPH now participates in the driven runner/CLI and native managed checkpoints.
Both variants retain authoritative E/A and all saved switch history. The source
adapter integrates native stage-power work and the pressure-energy bath without
applying a mass-density EOS or overwriting E/A at finalization. Source objects
survive stage cloning. Forcing/cooling bounds remain combined with the native
PESPH signal-speed timestep. Managed restart retains validated accepted fields
and exact search history; it does not insert an extra support Newton solve.

Two/three Gloo ranks qualify managed save/resume with changed chunk budgets,
local offset widths and rank count, plus injected force-output and neighbour
allocation failure. Two NCCL2.32.3 ranks sharing one GPU qualify both variants
in 2D/3D, with ordinary and failure-recovery continuation against independent
four-step references. Accepted fields, stage clocks and forcing histories are
exact; the existing source-budget roundoff tolerance remains2e-12. Allocation
failures are injected, not deliberate device exhaustion. No long-flow or
performance claim follows from these controls.

## Unsupported contexts

MFM/MFV and other unported scheme families reject at distributed startup.
Meshless geometry, closure reserves and iterative exchanges require their own
ports and qualification. Wall closure and material paths also remain gated.

Chunked reverse/forward AD, JVP/HVP, deferred search and CUDA graph execution are
unsupported. Search rejects dual tensors before detached snapshots can discard
a tangent. Actual CUDA capture rejects locally before control callbacks, host
readbacks or collectives. All ranks must agree on execution mode before entering
capture; arbitrary mismatched capture contexts are not recoverable by a handshake
inside the capture. Existing flat Godunov reverse-AD checks continue to pass.

MPI-enabled and ROCm-enabled Torch builds are deferred. The storage/operator
boundary permits future transport adapters, but backend capability, placement and
failure tests are still required. MPI launching NCCL ranks is distinct from an
MPI tensor backend. No MPI/ROCm support is claimed by these Gloo/NCCL results.

## Qualification and next gates

Durable regressions live in `tests/test_distributed.py`,
`tests/test_chunkedCheckpoint.py`, their `tests/distributed/` workers and the
paired core's search/operator/distributed suites. They compare chunked results
with the existing flat path, including fresh RHS stages, owned W, accepted fields,
time/dt, OU history and source budget. Tiny repeated A/B runs omit video while
retaining progress and stall/velocity controls. Exact field comparisons retain
only the existing 2e-12 tolerance for the globally reduced source budget.

L4 qualifies redistribution/filter replay; L5 adds seeded 4/5-rank stress,
64-step NCCL native replay and IO/failure controls; L6 adds the matrix above,
Godunov prepared launches and save/resume. The integer global-row ABI change
preserves normalized floating Warp ASTs and passes all 16 existing flat Godunov
gradchecks. See the canonical plan for exact test counts and source variants.

L1–L6 CUDA evidence used one GPU under ACSPH contention; L7 ran after the GPU
was released. All CUDA evidence still uses one physical device, and timings are
diagnostic.
L7 now qualifies real total-edge access above INT_MAX:47,000/50,000 coincident
particles, up to2,500,000,000 stored neighbours and three-component work buffers are fully
written/read, with explicit device probes across2^31. Production density matches
an independent one-row flat reference. Filtering preserves old J/work and
redistributes changed ownership through NCCL. The largest standalone allocated
peak is46.58 GiB. Entire chunks starting above INT_MAX also pass. The durable core worker is
`tests/distributed/_wideRowChunkWorker.py`; its small CPU/Gloo oracle controls run
in the ordinary process suite. Run the hardware gate explicitly, for example:

```bash
# From the paired core checkout, using the selected Python environment.
PYTHONPATH="$PWD/src" warpSPHCore_OFFSET_DTYPE=int32 \
python tests/distributed/_wideRowChunkWorker.py \
  --device cuda:0 --particles 47000 --edge-budget 67108864 --filter \
  --require-wide --memory-budget-gib 64 --reserve-gib 8 --output /fresh/output
```

Use the isolated NCCL wrapper and torchrun with two processes to repeat on the
shared GPU; set the shared-rank environment described above. Both ranks select
`cuda:0` in this debug worker. Check headroom immediately before every large job.
Physical multi-GPU/site topology, bandwidth/scaling, developed-flow capacity,
full wide-edge solver trajectories and very large individual chunk extents
remain open. This debug fixture has no flow integration or video. Timings mix
JIT, lanes and filtering choices and remain diagnostic.

## Allocation failure recovery

For an eager chunked driven flow, an allocation failure during an integrator
attempt raises `warpSPH.runner.RunAllocationError`. Its `result` retains the
last accepted particle state and completed-step count; `checkpointPath` names
the native `allocationFailure.h5` if publication succeeded. The original error
is preserved as `__cause__`. `recoveryError` records a secondary sync/write
failure without replacing the allocation evidence. Recovery does not accept
the failed stage or change the timestep/integration method.

The runner retains particles while saving small forcing-clock metadata at each
accepted boundary. After failure it releases trial frames/partial buffers,
synchronizes CUDA, restores the accepted forcing history and atomically writes
particles plus the neighbour reconstruction recipe. J/W are never exported.
Recovery needs at least one qualified accepted step, a working CUDA context and
writable storage. Device faults, process death, graph replay and failures in
partially completed post-step hooks do not establish a recoverable boundary.
If saving fails, the exception still retains the accepted result for inspection.

A coordinated local-phase allocation error is propagated to every live rank as
`DistributedAllocationError` (also a `DistributedContractError` and `MemoryError`).
This classification uses the existing fixed control packet. Canonical native
publication stays collective; process death or failed communication is not a
checkpoint-recovery mechanism. Automatic output remains disabled for distributed
proof-of-concept runs.

Core bulk J/pair-field allocation failures include requested batch capacity,
completed chunks and CUDA live/reserved/free memory in exception notes. Failed
partial allocations are released. Guards must distinguish live memory from
Torch's cached reservation: raw device-free memory alone can reject a request
that the allocator can satisfy from its cache. Cache plus free bytes are an
admission estimate, not a fragmentation guarantee; actual allocator failure
still follows the recovery path. No per-step cache flush is introduced.

## Scalar launch geometry

`OperatorSpec.blockDim` optionally selects scalar CUDA threads per block;
`None` keeps Warp's256-thread default. Values must be Python integers1–1024.
The setting is forwarded by ordinary and prepared launches and is part of
distributed effective-launch agreement. CPU uses one thread and cooperating
tiled kernels keep their lane block size.

The CRK compatible-work sweep selects128threads. Its register-heavy3D FP32
kernel admitted only one resident256-thread block per SM on the local GPU;
128threads admit three and remove most of the small-chunk utilization penalty.
Matched64³ controls retain bitwise particle values while work falls from51.47
to30.47ms/step. FP32/FP64 compatible-work/replay and nondefault-stream controls pass, as do
CUDA block/prepared/reverse tests and NCCL launch agreement/recovery.
This is a local launch qualification, not a physical multi-GPU
performance claim; arithmetic, neighbour traversal and fresh stages are fixed.


The authorized continuation order is now MFM/MFV geometry/closure/flux
admission, then delta-SPH/Sun2017, then IISPH and DFSPH variants. Preserve each
scheme's thermodynamic variables, native integration, pressure iteration and
history. ACSPH is excluded. MFM/MFV already accumulate canonical face fluxes by
row; their closure weights, CSR matvec shortcut and multigrid/preconditioner
selection need explicit owned storage and dependent particle exchanges. Turning
off closure or changing the selected preconditioner is not a compatibility port.

## MFM/MFV closure and continuation

Meshless execution keeps its native conserved Q, support policy and closure warm
start. The default unprojected faces remain the default; opt-in projection keeps
CG and auto/explicit multigrid, coarse stencils, stopping norms and bounded outer
passes. Each owner retains dedicated scalar face-weight buffers and, for capped
closure, an additional scalar scale field. Budget these pair fields with J/I;
one bounded weight-output scratch chunk can coexist during construction. Matvec
and stencil outputs are exchanged as particle fields before dependent reads.
The chunk path runs eagerly; the flat CSR graph optimization is unchanged.

Native MFM SSPRK and MFV Hancock defaults pass separate controls. The driven case
already requires explicit SSPRK for its stage sources, including MFV. Restart
rebuilds moments from the edge-free search recipe and supplies saved state.lam to
the usual start-of-step closure, retaining finite-tolerance warm-start history.
Neighbour and weight allocation failures preserve the accepted conserved state.
Gloo/NCCL debug continuation changes rank count, row budget and local offset width.
These short controls establish correctness, not physical multi-GPU performance.

## Delta-SPH / Sun2017

The launcher additionally accepts `case=tgv-wc` with `solver.scheme=deltaSPH` or
`sun2017DeltaSPH`. The bounded case adapter is 2D, periodic and regularly sampled;
it retains the native initial relaxation and its 128-iteration default. Optional
`params.shuffleSeed` makes the jitter reproducible. Without a seed, rank0's
jitter is replicated. Initial relaxation retains the original pair-list-free
grid density pass on every replica and owns only the shift neighbour rows.
Native 3D operator/step controls are separately qualified on Gloo/NCCL.

Sun2017 retains its frozen diffusion within each RK step and fresh diffusion at
stage 0 of the next step. Native shifting and its transport corrections remain
active under their existing defaults. Weakly compressible pressure may be
signed; accepted density must stay positive and all particle fields finite.
Restart saves accepted density/particle fields, clock, edge-free search recipe
and accepted mask history when present. It never saves a mid-step frozen RHS.
RHS and post-step shifting allocation failures preserve the accepted input and
publish an atomic checkpoint. Wall-only no-penetration statistics with no
samples are JSON null in launcher diagnostics.

The existing CPU `test_theUnrenormalizedVariantCancelsToo` threshold fails with
RMS ratio 0.2218983918428421 against 0.2 on both pre-port and current source.
This numerical validation issue is recorded in the canonical plan; the formula
and tolerance were left unchanged. New storage/stage/restart controls pass.
The broader wall/free-surface/other-shift/AD/graph and case-adapter gates remain.

## Native incompressible projections

Use `case=tgv` with `solver.scheme=iisph`, `dfsph`, `dfsphPredictor` or
`dfsphShift`. The managed adapter is eager, periodic, regularly sampled, 2D and
one fluid material. Initial relaxation owns only its neighbour rows; optional
`params.shuffleSeed` makes jitter reproducible, otherwise rank0 supplies it.
The four native integration/projection orders, pressure signs, iteration
budgets, residual norms and warm-start policies are retained. Each owned
particle result is exchanged before a dependent pressure read; replicated
means/norms are not summed again across ranks.

Accepted checkpoints contain the native physical pressures, independent DF/CD
impulses, exact edge-free search history and both nominal and next timestep
clocks. They can rebuild different row ownership/local offset widths on resume.
Restore matches the native next RHS I-storage contract to retain saved J
reduction order; a disposable seed cannot force a fresh spatial search.
Bulk allocation failures inside pressure iterations or after shift's drift keep
the preceding accepted state dumpable. Device/communicator faults remain outside
this recovery promise. Use the explicit chunked target to load a native managed
file; an ordinary flat resume is rejected.

These are storage/execution qualifications. Existing native numerical limits
remain: the coarse nx8 absolute-source/default-relaxation stress hits pressure
budgets, and CUDA FP64 IISPH stops at the same timestep guard on pre-port source.
The separate existing mean-free periodic control exercises converged nonzero
pressure and warm starts; it does not replace defaults. See the canonical I1
receipt for exact controls and retained failures. Long-time pressure accuracy,
wall/free-surface coupling, experimental Band geometry, other case adapters,
AD/graphs and physical multi-GPU/site performance remain separate gates.
