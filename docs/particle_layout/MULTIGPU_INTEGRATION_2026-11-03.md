# Multi-GPU / particle-layout integration — 2026-11-03

The qualified owned-row/multi-GPU infrastructure is integrated with the opt-in
fixed-count Morton layout on frontend `dev` and core `main`. The resulting core
baseline is also merged into the native C++ development branch. Default input
ordering and single-device execution remain unchanged. The accepted Isambard
image/profile is frozen; no site job or new long-flow campaign is part of this
integration.

## Checkpoints and merge direction

- Frontend pre-integration checkpoint: `825f9e0`, preserving layout `6a067df`.
- Multi-GPU inputs: frontend `2b57b59`, core `179df45`.
- Core/main merge: `a2c85b5` (parent/main `04e126e`).
- Integrators/main: `cdbbc52`, pinned explicitly for all new checks. The prior
  multi-GPU cohort used pre-layout `1fea2a3`; that result was not treated as a
  substitute for checking the current integrator.
- Native worktree checkpoint: `eaa75fb`, preserving backend `430aef7` and the
  existing operator survey. The survey is source/evidence, not generated binaries.

Core integrates first, frontend resolves the shared runner/stage/export contracts,
then the validated core baseline feeds `codex/native-cluster-pairs`. No dependency
editable installation, remote branch or accepted simulation image is replaced.

## Combined contract

`ChunkedMortonLayout` uses the accepted-boundary native path for the audited
CompSPH/CRKSPH and delta/projection system families. Each attempt keeps one basis
through all stages, compatible-work replay and pressure iterations. Current
admission is eager explicit one-step/maps; multistep, implicit, chunked AD and
CUDA graph execution remain guarded. Existing flat multistep/FD-JFNK and graph
adapters remain available within their separate supported scope.

Staleness is agreed across ranks. A changed basis permutes declared particle and
history fields and rebuilds only the newly owned chunks, retaining the policy,
communicator, support mode and store-I choice. A valid ordering/list stays resident.
Old per-edge W is discarded; there is no global J/W gather or inferred edge axis.
All ranks agree on the layout transition and check exact replicated UID order.
The explicit driven wrapper and ConservedSources/FourierOU inputs are admitted;
the two-element work/heat ledger is global, even if N equals two. Arbitrary
closures and source providers do not acquire sorting support through this merge.

Restart preserves the saved compute basis and numeric resident-search recipe;
changed chunk budgets/world sizes repartition that recipe before further sorting.
Chunk checkpoints remain compute-order numerical restart files. Automatic flat
pair export remains rejected for chunked lists; the dedicated checkpoint path
never serializes global pair buffers. The existing flat trajectory/export path
continues to present UID order.

The runner retains both native layout/history restoration and distributed resume,
accepted-step callbacks and allocation-failure cleanup. After a boundary resort,
allocation recovery binds the checkpoint token and surface history to the new
accepted basis before entering the numerical attempt.

## Native development branch

The relocated compatibility modules remain aliases. Row-chunk eligibility and
query/reference roles were applied to their real Warp implementations under
`coreOperations/native/wp_*`, including paired CRK interpolation inputs. Search
exports and relocated tests follow the native tree's backend/suite boundaries.

**This is source integration, not a C++ multi-GPU implementation.** Owned-row
execution still uses the qualified Warp backend. The pure native backend retains
its own public API and import isolation. Native C++ chunk dispatch, distributed
adjoints and native frontend/JFNK integration remain separate ports.

## Validation

All tests use `/home/lu26029/miniconda3/envs/warp/bin/python` and explicit paired
source paths; CUDA jobs run sequentially. These tiny repeated controls omit video
and viewer registration. Progress and per-run timing remain in logs and receipts.
Results are correctness gates, not performance or physical scaling measurements.

| Selection | Result |
| --- | --- |
| Core/main planner/search/recipe/operator CPU | 334 passed, 273 expected device/precision skips; 2.48 s |
| Core/main separate-process ownership/failure controls | 4 passed; 23.80 s |
| Existing frontend solver/history/boundary/export/ownership CPU selection | 157 passed, 3 skipped; 8 stale test-injection locations found, corrected and requalified below; original run 385.84 s |
| Compatible-work/replay CPU after injection-location repair | 8 passed, 2 CUDA skips; 7.21 s |
| First frontend layout/native/runner contract selection | Passed (one CUDA skip); raw log retained |
| New combined CPU sorting/repartition/restart controls | 10 passed; 37.45 s |
| New combined controls plus existing native CUDA graph controls | 21 passed; 58.71 s |
| New single-GPU FP64 sorting/restart controls | 9 passed; 37.33 s |
| Two ranks sharing one GPU, exact replicas, sorting and empty→active owner restart | All 8 schemes passed on both ranks; 31.41 s cohort wall time |
| Native worktree Warp search/operator CPU | 334 passed, 2 skipped, 203 device deselections; 24.23 s |
| Final frontend combined/recovery/budget controls using native worktree Warp backend | 14 passed; 46.16 s |
| Native worktree C++ primal/corrected/field-linearization CPU | 1,751 passed, 97 skips; 20.38 s |
| Native package Python/import isolation | 21 passed; 0.74 s |
| Frontend complete runtime/static import audit | Passed |

The combined solver matrix includes CompSPH, CRKSPH, deltaSPH, Sun2017, IISPH,
DFSPH, predictor and shift. It forces repeated accepted-boundary changes, compares
uninterrupted and changed-partition restart, preserves source knots/RNG/clock,
pressure and surface histories, and checks empty-owner J/W storage. Particle
fields/time/next-dt are exact in the same adopted basis; source-budget comparisons
retain the established 2e-12 reduction tolerance. Dedicated tests inject an
allocation failure after sorting for CompSPH/deltaSPH/DFSPH and verify the accepted
particle/surface basis in the recovery checkpoint.

Shared-GPU checks use the existing isolated NCCL 2.32.3 overlay, not a shared
package upgrade. Both receipts report NCCL runtime 23203 and the same GPU UUID.
This does not renew physical multi-device or inter-node qualification.

Two environment setup failures were diagnosed without solver changes: sandboxed
Warp could not write its normal kernel cache, and native JIT initially lacked the
existing Ninja tool directory on PATH. Successful runs use the normal cache and
`/tmp/warpsph-build-tools/bin`, with the prior native JIT cache. Native compiled
sources are unchanged by the wrapper merge.

Raw logs, commands and shared-GPU source/runtime receipts are under
`.tmp/layout-multigpu-integration-20261103/`. Final precision/CUDA follow-up results
and source digests are recorded below.

## Remaining qualification

Keep sorting opt-in. Larger/longer boundary cases, native crossover measurements,
physical GPU topology, large implicit/multistep history costs and the unsupported
chunked wall/material/graph/AD contexts retain their existing separate gates.
Population changes and physical inlet/outlet models remain deferred.

## Final gates

| Final selection | Result |
| --- | --- |
| CPU FP64 combined + existing runner/native history | 54 passed; 65.60 s |
| Native C++/CUDA primal/corrected/field-linearization | 1,752 passed, 96 skips; 179.15 s including JIT compilation |
| Frontend combined CUDA against native tree Warp backend, including recovery | 13 passed; 31.29 s |
| Actual distributed launcher, Morton CompSPH/CRKSPH, 1→2→3-rank reference/save/resume | 2 passed; 62.66 s |
| Final full frontend import audit | Passed; 6,723 static first-party imports |

The launcher gates verify rank-owned diagnostics, manifests, source pins, native
checkpoints and exact resumed state/forcing/clock; CRK also checks failure-report
publication and preserving an existing output directory. Runtime/source receipts
from the shared-GPU cohort are retained under `nccl/`.

`source-manifest.json` hashes all Python/C++/CUDA source and test files across
frontend, main core, current integrators and the native development tree. Its
SHA256 is `d5a2db8b92a6eef5e07f824c456470c0a4d315cf6c5f916e4000e8f5fcd487c0`.
The manifest records pre-merge-commit HEADs alongside actual file hashes; Git
merge commits preserve the resulting source. Documentation-only completion does
not alter those source hashes. All local gates are complete; physical topology,
throughput, long flows and C++ distributed execution remain separate.
