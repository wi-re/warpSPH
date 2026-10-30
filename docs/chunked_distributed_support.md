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
| Prepared operators | Shared opted-in operators, native work, GSPH, Gaussian density and Inutsuka | Explicit query/reference/chunk/dummy roles; primal eager execution |
| Neighbour lanes | Fixed scalar and global 32-lane CUDA configurations | Existing tiled kernels are used where available; Godunov/Gaussian kernels use scalar fallback |
| Native checkpoint | Accepted-step drivenTurbulence, all five schemes, FourierOU/ConservedSources | Canonical particle/time/source/search recipe; rebuild owned J, never serialize W or a global pair tensor |
| Legacy checkpoint | Qualified version-1 CSR/COO import for Monaghan/CRKSPH/CompSPH | Godunov legacy import explicitly rejects; use the native format |
| Process/rank changes | Gloo 1–5 ranks and tiny two-process/one-GPU NCCL controls | Physical device-count changes and multi-node topology remain unqualified |

The dedicated launcher currently accepts `drivenTurbulence` with these five
schemes. Native checkpoint qualification is the small 2D driven case; 3D solver
controls do not establish 3D restart qualification. GSPH/Inutsuka operator controls
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

## Unsupported contexts

MFM/MFV/PESPH and other unported scheme families reject at distributed startup.
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

All CUDA evidence is from one contended physical GPU. Timings are diagnostic.
Physical multi-GPU/site topology, bandwidth/scaling, developed-flow capacity,
large populations and real total-edge access above INT_MAX remain open. L7 needs
an explicit free-memory/headroom check after the competing ACSPH work permits it.
