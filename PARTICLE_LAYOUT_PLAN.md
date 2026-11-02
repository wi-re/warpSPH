# Particle layout, persistent identity and changing particle counts

Progress and priorities: [PLANS.md](PLANS.md).

2026-11-02 user direction: execute the next phases in order **1, 2, 4, 3**:
commit the qualified frontend adapters (done as `24d7a92`), persist numerical
continuation, extend boundary/ghost/source and remaining integrator-input adapters,
then qualify on the machine's GPU outside the sandbox using its normal Warp
environment. Changing particle counts/emitters/outlets remain the following phase.

Current implementation (2026-11-02): numerical continuation checkpoints,
declared Adams/NDF contexts, implicit warm/preconditioner inputs, pure UID forcing
and boundary/ghost/body adapters are implemented and qualified. See the
[restart/input and GPU report](docs/particle_layout/RESTART_AND_INPUTS_2026-11-02.md).
Final gates: 547 CPU passes / 46 skips (ten hard-CUDA cases deselected), 283 FP64
passes, 52 GPU passes, 44 GPU FP64 passes; all ten CUDA configuration cases pass
outside the sandbox in the normal `warp` conda environment. Paired 2D GPU timing
at 1,048,576 particles improves CompSPH from 79.435 to 58.237 ms/step and CRKSPH
from 125.321 to 94.668 ms/step. Small problems pay adapter overhead; input-order
defaults remain selected. CPU-pinned paired native controls show no consistent
regression beyond observed variation. Frontend remains on `dev`; integrator
main/worktree remain aligned at `cdbbc52`, core at `04e126e`.

Next: accepted-boundary fluid population events, history reset and variable-roster
output before emitters/outlets. Stateful source transactions, automatic WCSPH
runner/export integration, analytic providers, graph execution and exact JVP
remain separate gates. Low-level ghost/body utilities do not enable those paths.

Previous checkpoint: fixed-count continuation adapters were CPU-qualified;
see [the latest report](docs/particle_layout/CONTINUATION_ADAPTERS_2026-11-02.md)
and the preceding [renewal/persistence checkpoint](docs/particle_layout/RENEWAL_AND_PERSISTENCE_2026-11-02.md).
User direction (2026-11-02): commit that checkpoint, then implement continuation
adapters. Committed locally as `ffcd152` on `dev`, without a push. The next slice
adapts fixed-step Adams/BDF history and explicit FSAL input/output by UID, plus the
generic runtime's accepted `Continuation`. Qualification is complete on CPU. Raw
derivatives must survive SPH's step-average diagnostic replacement; physical
hook changes restart history, and unsupported method-owned contexts fail closed.
Final frontend gates: 493 passes / 25 skips (ten hard-CUDA configuration cases
deselected), 231 FP64 passes; archived old-backend default-route gate 128 passes /
six skips. Raw thermal derivatives, every admitted Adams/BDF order, accepted
runtime memory/dense/rejected-step behavior and in-memory checkpoint continuation
are covered. The isolated integrator quiet-reuse formatting fix `cdbbc52` passed
759 tests / 415 skips and is fast-forwarded into main; no push. CPU FSAL fixtures
save about 7–13%, while UID/history adaptation adds about 0.7–0.8 ms to small
AB3 steps. GPU throughput remains unqualified. Next: persisted numerical history,
additional method-owned contexts, CUDA, then boundary/ghost/source/topology adapters.
The following audit introduction records the original scoping baseline.

Original audit: scope and proposed implementation sequence complete, 2026-11-01.
The 2026-11-01 scoping session changed only planning documents; the separate
integrator implementation checkpoint is recorded below.
Revised after the user's follow-up: keep compute order for a neighbor list's
lifetime; return RHS updates in input order with optional layout-change metadata.
Separate stage preparation/resource maintenance from equation evaluation;
adaptive support can require a list rebuild without requiring a new ordering.
User priority: establish the integrator-library contract first, especially
multi-stage/coupled schemes, JFNK probes and implicit differentiation. Frontend
stage-preparation interfaces below remain proposals until that gate passes.

2026-11-02 checkpoint: the user authorized implementation in a separate
integrator worktree. Initial generic resource/continuation contract and synthetic
qualification are complete on `codex/integrator-resource-lifecycle`, based on
`1fea2a3`, at `/home/lu26029/.codex/worktrees/integrator-resource-lifecycle/warpSPHIntegrators`.
See the [implementation and results](/home/lu26029/.codex/worktrees/integrator-resource-lifecycle/warpSPHIntegrators/docs/development/evaluation-resources.md).
Broad CPU selection: 2,986 passes/471 skips; final targeted 143 passes; installed
consumer tests 47 passes, including 30 new synthetic cases. No frontend solver
change, production neighbor manager, automatic coverage-renewal policy or GPU
qualification. The integrator main checkout remains unchanged; work is uncommitted
in the isolated regular Git worktree (the app cannot attach it as managed).

User request: permit physical particle-state sorting by Z-order for locality,
restore a persistent order for exports, and relax fixed particle counts for
emitters/outlets. Preserve neighbor-list and ghost/boundary relationships and
account for the time-integrator library and its inputs.

## Recommendation

**First workstream: integrator evaluation/resource/acceptance contract.** Do not
start by wiring sorting into frontend RHS callbacks. Audit and qualify the
generic library behavior below with synthetic systems, then implement the
particle-specific spatial adapter against that contract.

Make **persistent particle identity independent of storage row**, and make a
**layout change an explicit operation on the whole live simulation context**.
Implement fixed-count sorting first, then changing populations. Keep the
**integration-visible input order** and population fixed throughout an attempt,
including stages, subflows, nonlinear iterations and internal reruns. Allow a
separate **compute order**, selected when building a neighbor-list generation
and retained for that generation's lifetime. Return every derivative in the
substep's input order and carry the compute layout with adjacency in simulator
auxiliary data. The integrator's numerical algorithms need not interpret sorting.

At finalization, finish all old-order physics/work first, then prepare adoption
of the latest compute layout for subsequent steps. A fixed-step one-step path
with no subsequent old-order consumers can commit there. In general commit via
a simulator-owned result/acceptance adapter, after rejection/rerun/error/history
handling is safe. This distinction is necessary: existing integrators can create
history or use the error estimate after calling the system's `finalize`.

This replaces the initial rebuild-on-reorder-first proposal. Build new lists
directly against sorted compute data; adopting that layout does not require an
O(E) remap of a list already in that layout. Generic live-list CSR remapping
remains a utility for independently requested reorders and exported index spaces,
not a prerequisite or the normal hot path. Do not change a live list's ordering
merely because particles move; rebuild only under the existing validity rules.

This is coordinated work in **warpSPH and warpSPHCore**, with
**warpSPHIntegrators lifecycle/result compatibility** to qualify. Keep any needed
generic adapter hook solver-agnostic; avoid sorting logic in integrator algorithms.
Plotting, the remote viewer and data consumers need compatibility qualification.
The reported up-to-2x locality penalty is the motivation supplied by the user;
this audit does not independently measure it or promise a 2x whole-step gain.

## Evidence and present contracts

Read-only source baseline: frontend `dev` at `5534db1`, sibling core at
`04e126e` (0.6.3), sibling integrators at `1fea2a3` (0.7.0).
These are source-checkout observations, not a claim about any running worker's
imports. Implementation must record actual import provenance for all packages.

| Area and source | Current assumption / fallout | Required treatment |
|---|---|---|
| [Particle states](src/warpSPH/systems/), [identity](src/warpSPH/systems/identity.py), [initialization](src/warpSPH/initializers/weaklyCompressible.py) | Integrated and `constant` fields both carry particle rows. Identity currently separates material, source and body labels, but not layout lifetime. | Declare layout semantics independently of integration behavior; gather all particle fields, including metadata and history. Preserve source/material/body meaning. |
| [Core state fields](../warpSPHIntegrators/src/warpSPHIntegrators/state/fields.py) | `constant()` means not integrated and cloned for stages, not immutable allocation or global data. `copied()` and `ephemeral()` describe temporal lifecycle. Tensor shape alone does not identify an axis. | Add orthogonal axis/remap metadata or an explicit adapter registry; cover frontend state subclasses and update objects. |
| [Compact-hash sorting](../warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/sort.py), [index kernel](../warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/wp_index.py), [query-order contract](docs/compute-neighbor-search.md) | Spatial query scheduling leaves all public rows unchanged. The active compact-hash path stably sorts **linear cell indices**, via `getLinearIndex64`; a [Morton helper](../warpSPHCore/src/warpSPHCore/radiusSearch/compactHash/morton.py) also exists. | Add physical Morton ordering explicitly. Do not treat the current `sortIndex` or `neighborQueryOrder` as an existing Z-order state sort. Hash-table cell encoding can remain linear. |
| [Adjacency ABI](../warpSPHCore/src/warpSPHCore/dataTypes/adjacency_t.py), [Verlet construction](../warpSPHCore/src/warpSPHCore/radiusSearch/verlet/build.py), [validity](../warpSPHCore/src/warpSPHCore/radiusSearch/verlet/validCheck.py) | Counts/offsets identify query rows, J contains reference rows, optional I is redundant. Validity compares current rows with build-time positions/supports. Shape checks catch count changes, not a same-size replacement or permutation. | Tag query/reference compute layouts; transform stage inputs into those layouts before traversal/validity. Build directly in the new sorted layout. Reserve CSR remapping for independent layout changes/export. Preserve padding and integer widths. |
| [Stage list selection](src/warpSPH/utils/pairList.py), [pair work](src/warpSPH/modules/compSPH/pairWork.py) | `listFitsParticles` checks shapes. Compatible energy accumulates edge-aligned work on one list across stages and may rerun a whole step. | Check list/layout generations. W stays with its adjacency in compute order; transform half-step velocities into that order and resulting particle energy rates back to input order. A generation change still needs the existing inconsistency/rerun handling. |
| [Ghost construction](src/warpSPH/rigidBody/ghostParticles.py), [mDBC helpers](src/warpSPH/utils/syncFree.py) | `ghostIndices` is a bidirectional row relation; fluid rows use -1. Offsets are row values. Ghost UID is currently `-boundaryUID`. | Gather relation rows and remap their valid targets. Preserve -1 and offset signs. Resolve UID uniqueness before treating UIDs as universal entity keys: a boundary UID of zero produces a ghost UID of zero too. |
| [Rigid-body build](src/warpSPH/rigidBody/build.py), [update](src/warpSPH/rigidBody/update.py), [stages](src/warpSPH/rigidBody/stages.py) | Bodies retain full-length boolean masks plus compact body-local positions, normals, masses and UID sequences. Mask selection assumes corresponding boundary and ghost sequences. | Preserve body-local ordering and map its persistent IDs to current rows; use explicit scatter indices and ghost relations. Permuting only masks would attach local geometry to different particles. |
| [Kind-index cache](src/warpSPH/utils/syncFree.py), [boundary-presence cache](src/warpSPH/modules/mdbc/_util.py) | Kind indices are cached by kind/device/N; boundary presence once per config. A same-N reorder silently invalidates indices. A population change may invalidate presence. | Cache by particle-set identity and appropriate layout/topology generations, or maintain counts in the layout owner. Avoid new per-step `nonzero`/host reads. |
| [ACSPH system](src/warpSPH/systems/artificialCompressible.py) | `positionsPrev*`, `velocitiesPrev*`, `positionStep`, `positionLow` live outside the particle state and are shared by stage copies. | Remap the whole history coherently with the accepted state; preserve dt/scalar history. Never mutate snapshots still owned by a stage/output. |
| [CompSPH system](src/warpSPH/systems/compSPH.py), [DFSPH state](src/warpSPH/systems/incompressible.py) | Entropy/divergence/Wadsley switch history, CRK gradient memory and pressure-impulse warm starts are row-aligned. Pair arrays and global budgets coexist with those rows. | Gather particle memory; classify pair work separately; preserve global budgets. Birth initialization must be scheme-specific. |
| [Meshless state](src/warpSPH/systems/meshless.py), [geometry cache](src/warpSPH/modules/meshless/geometry.py), [driven particles](src/warpSPH/modules/meshless/boundary.py), [inflow reservoir](src/warpSPH/modules/meshless/reservoir.py) | Conserved Q and closure warm start are particle arrays. Driven particles retain full-row masks/reference positions/masses. Reservoir `owners` indexes physical rows; geometry concatenates physical and virtual points. | Remap external inputs and owner indices, invalidate geometry tickets, preserve physical/virtual index spaces. Existing analytic reservoirs are not general particle emitters. |
| [Integrator history](../warpSPHIntegrators/src/warpSPHIntegrators/state/history.py), [Adams](../warpSPHIntegrators/src/warpSPHIntegrators/drivers/history/adams.py), [variable Adams](../warpSPHIntegrators/src/warpSPHIntegrators/drivers/history/variable_adams.py), [BDF](../warpSPHIntegrators/src/warpSPHIntegrators/drivers/history/bdf.py) | Optional `StepHistory` UID guard is pointer+shape, runs on push, and is not uniformly supplied by drivers. Some drivers inspect lowercase `uid`, while frontend exposes tagged `UIDs`. Histories include derivatives, states, aux and method context. | Validate before history is consumed through a shared protocol; remap or reject explicitly. A pointer check or adding a `uid` alias alone cannot fix this. |
| [Runner](src/warpSPH/runner/runner.py), [graph adapters](src/warpSPH/utils/cudaGraph.py), [deferred decisions](src/warpSPH/utils/stepDecisions.py) | Runner passes state/f/dt/config/schemeConfig directly; its current calls do not thread generic `IntegrationResult.history` or `priorStep`. Pipeline overlaps old outputs with the next step. Graph keys use shapes, adjacency addresses and optional scheme keys. | Add one boundary coordinator to both runner loops; integrate history ownership when enabled. A graph-safe row change also needs coherent captured inputs/mappings; same addresses do not prove semantic validity. |
| [Trajectory export](src/warpSPH/io/export.py), [load](src/warpSPH/io/importIO.py), [dataset](src/warpSPH/io/dataset.py) | Most identity/material/kind/mass metadata is written once; frame arrays assume those rows. Several UID datasets explicitly narrow to int32. Dataset masks and temporal samples assume a fixed roster. | Canonical export projection for fixed populations; versioned variable-roster format and UID joins for changing counts; retain int64 IDs. |
| [State/stage HDF5](src/warpSPH/io/hdf5.py), [driven restart](src/warpSPH/caseUtils/drivenTurbulence.py) | State/stage dumps can include adjacency and edge data. Driven restart retains native adjacency but validates dimensions, not the particle layout. Missing UIDcounter is reconstructed from surviving IDs. | Persist layout association, real allocator high-water mark and source state. Canonicalizing particles requires canonicalizing all indexed structures too. |
| [Inlet/outlet regions](src/warpSPH/configurations/region.py), [inlet](src/warpSPH/regions/inlet.py), [outlet](src/warpSPH/regions/outlet.py) | Region enum entries exist; generic runtime injection/removal modules are placeholders. `emitters` currently records provenance. | Introduce explicit accepted-step population events; do not infer runtime emitter behavior from provenance or region enums. |

This is a subsystem-wide contract change, not a local sort-kernel change. The
table names concrete seams, not an exhaustive claim that every user hook or
external consumer has been audited.

## Proposed data contract

1. **Identity:** an entity has a persistent int64 ID for its lifetime, independent
   of row. IDs are never recycled after deletion. Keep `materials`, `emitters`
   and `bodyIDs` as separate labels. Give new ghosts their own IDs and keep the
   parent relation explicit. Preserve existing real-particle IDs on migration;
   assign legacy ghosts fresh IDs with a recorded compatibility mapping.
2. **Canonical export order:** capture the original combined order before any
   sort, using an ordinal/identity registry. Existing particles retain their
   ordinal; births append in deterministic event/local order. Deletions leave
   gaps in this registry, not inactive rows in every export. Sorting raw signed
   legacy UIDs does not recover the current fluid/boundary/ghost append order.
3. **Layout descriptor:** carry a particle-set token, topology generation,
   order generation and active count. Topology generation changes for membership
   changes even when N stays the same; order generation changes for row changes.
   Track storage/capacity changes separately for graph addresses. Clone/stage
   operations retain logical layout identity despite allocating new tensors.
4. **One owner:** the simulator owns spatial contexts and prepares transitions;
   the runner supplies the outer acceptance boundary when needed. State/system,
   core adjacency, sources, bodies and integrator memory provide typed adapters.
   Configuration stays configuration; avoid hiding the sole authoritative
   permutation in a scheme config cache.
5. **Field semantics:** distinguish particle values, particle references with
   target index space/sentinel, pair values, body-local values, global values,
   and disposable derived caches. These are orthogonal to
   `integrated/constant/copied/ephemeral`. Do not gather every tensor with first
   dimension N: an edge count or small global array can coincidentally equal N.
6. **Unknown extensions:** require custom states, hooks and source objects to
   declare remap behavior; fail clearly when enabling sorting on an unsupported
   adapter. Preserve the existing unsorted route for compatibility.

Suggested interfaces, to agree across repositories before implementation:

```python
# Names illustrate responsibilities; not an implemented API.
prepared = simulator.prepare_stage(stage_input, spatial_context)
compute_update = scheme.evaluate_equations(prepared)
input_update = prepared.to_input_order(compute_update)
return input_update, prepared.aux  # context persists; optional transition proposal
```

A transition must publish a coherent context after all adapters complete.
Respect tensor ownership: validate before mutation and do not overwrite old
accepted/output/history storage while it is still in use. Start with owned
scratch and explicit stream events; qualify in-place buffer reuse separately.

## Integrator-first workstream: speculative evaluations and resource ownership

The user's latest direction makes this the first implementation gate, before
the frontend locality prototype. The integrator algorithms can remain agnostic
to Morton sorting, but the library needs an explicit contract for execution
resources used during repeated/speculative function evaluation. Returning an
optional permutation in aux alone does not establish that contract.

### Source-backed findings

| Current implementation | Implication for the proposed extension |
|---|---|
| [updateStep](../warpSPHIntegrators/src/warpSPHIntegrators/state/lifecycle.py) invokes preprocess, f and postprocess for every evaluation; postprocess receives the initial system as its receiver. | These hooks are not once-per-accepted-stage/step operations. Do not let them advance emitters, accepted history, bodies or a globally published layout. State cloning alone cannot isolate mutations of the initial system or external resources. |
| [DIRK](../warpSPHIntegrators/src/warpSPHIntegrators/drivers/implicit/dirk.py) closes over a stage equation, passes it to a solver, then re-evaluates the converged stage for JFNK. | Preserve that final materialization: the last solver probe's aux is not automatically the converged stage's aux. Generic stage/evaluation identity must be explicit; time alone is not an evaluation ID. |
| [Coupled implicit RK](../warpSPHIntegrators/src/warpSPHIntegrators/drivers/implicit/coupled.py) evaluates every stage on each block residual, using a substate-major [BlockState flat layout](../warpSPHIntegrators/src/warpSPHIntegrators/state/fields.py). | A single global latest spatial context is insufficient. Different stage geometries need separate associated contexts; input field/stage/particle coordinate meanings remain fixed across the entire block solve. |
| [JFNK](../warpSPHIntegrators/src/warpSPHIntegrators/solvers/krylov.py) evaluates the base map, captures G(Y), constructs FD/JVP matvecs, runs GMRES, and optionally evaluates line-search trials. | There is currently no explicit RHS resource scope distinguishing base evaluation, linearization, probe and rejected trial. Adding stateful search/sorting needs isolation and a controlled invalidation policy. |
| FD matvec computes `(G(Y+eps*v)-G(Y))/eps`; JVP matvec seeds dual integrated fields and re-enters the same stage map. | Probe calls must preserve the residual's meaning. Cache/reduction/support-solve differences can be amplified by division by eps; dual paths must preserve tangents. |
| JFNK preconditioning adapts `preconditioner(v, Y, context)` once per Newton iterate, but context is an ordinary dict. | Vectors, residual action and preconditioner must agree on flat layout and linearization resources. A reference to mutable geometry is not an immutable linearization snapshot. |
| [Structured RHS](../warpSPHIntegrators/src/warpSPHIntegrators/adapters/rhs.py) distinguishes combined/split functions and linear actions; a linear action can receive arbitrary vector-valued integrated fields via unflattening. | Physical-state preparation cannot run blindly for every callable. A Krylov vector's position-shaped block is not a physical geometry. Linear actions need a captured base geometry/operator context. Cover IMEX/ARK and Rosenbrock/semilinear routes, not only ordinary f(Y). |
| [State snapshots](../warpSPHIntegrators/src/warpSPHIntegrators/state/ownership.py) clone state-owned resources and preserve aliases, but share functions; closure/external effects require a transaction. Stage clone handlers have their own ownership rules. | Define which resource handles are immutable/shared and which scratch is private. Do not deep-copy multi-GiB adjacency on every matvec, or mutate shared in-place buffers held by earlier evaluations/backward graphs. |
| [Runtime solve](../warpSPHIntegrators/src/warpSPHIntegrators/runtime/run.py) snapshots attempts, rolls back external transactions on rejection, and calls `transaction.commit(token, snapshot(result))` on acceptance. It then takes continuation state/history/prior from the original result and computes requested dense samples. | Existing transactions isolate attempt-level external effects, but their commit callback cannot transform the actual continuation result by mutating its snapshot. Acceptance is also not the end of dense-output consumption. A result/layout transition requires its own defined ordering and real memory transformation. |
| [Implicit differentiation](../warpSPHIntegrators/src/warpSPHIntegrators/solvers/krylov.py) re-evaluates the converged map to construct parameter and VJP paths, then runs an adjoint solve during backward. | Resources needed by those evaluations/graphs must remain coherent after forward completion and later time steps. "Solve finished" does not mean a mutable adjacency buffer can be overwritten safely. |

These are compatibility obligations for adding resource/layout changes, not a
claim that the current library already produces wrong results on its qualified
existing systems. The 2026-11-01 audit changed no numerical solver behavior;
the separate 2026-11-02 implementation checkpoint adds opt-in generic scopes
and continuation handling with the qualifications recorded above.

### Required scopes and invariants

1. **Attempt:** owns candidate state, pending resource changes and event/history
   proposals. Rejection, event truncation, compatible-work rerun and graph
   validation must not publish physical changes. External resources have explicit
   fork/discard/promote semantics; numerical scratch and physical side effects
   must be classified separately.
2. **Stage/block:** owns a physical stage geometry and its derived fields.
   Multiple evaluations at the same time are not identical states; coupled stages
   cannot share one mutable "last geometry" slot. A converged stage is explicitly
   materialized for its retained update/aux. Layout seen by arithmetic stays fixed.
3. **Linearization:** associates the base residual, vector layout, candidate
   search coverage, deterministic reduction order and preconditioner with one
   token for the duration of a GMRES solve. Probe scratch is isolated; it cannot
   replace the base/converged stage's accepted context. Ordinary GMRES requires
   a fixed linear operator and fixed linear preconditioner within that solve.
4. **Accepted continuation:** transforms the actual next state and retained
   memory coherently, while preserving previous-order snapshots needed for error
   assessment, event localization, dense output and backward differentiation.
   This is separate from system finalize and external transaction notification.
5. **Backward/adjoint:** owns or reconstructs the converged operator context and
   mappings without relying on whatever mutable search cache the simulator most
   recently used. Release resources only when their actual consumers are done.

Start by pinning compute layout across an entire nonlinear solve; permit only
coverage-preserving resource maintenance there. Per-linearization resource
renewal is a separate, explicit operation. A finite-dimensional vector layout
cannot change midway through flatten/unflatten, Arnoldi bases, block residuals
or preconditioner application, even if all returned derivatives are input-ordered.

For JFNK, aim to retain a valid candidate superset and its traversal order across
the base residual and its FD/JVP probes. Continue evaluating kernel distances,
supports, density, gradients and other numerical fields at the actual probe state.
**Freezing a candidate list does not freeze the physical support or equations.**
If coverage is exceeded, request resource renewal at a controlled solver boundary,
recompute the base residual under the new context and restart the affected
linearization. Do not swap lists/preconditioners silently halfway through GMRES
or continue with stale coverage. Line-search trials can require their own valid
contexts and must not overwrite the base token on rejection.

The resource contract should allow a stateless evaluator that independently
constructs correct coverage, too. A coverage rebuild is not inherently a
mathematical operator change if the same exact residual is evaluated. Pinning
coverage/order is the preferred first SPH policy for numerical reproducibility,
not permission to drop newly required neighbors or a universal restriction on
all applications of the integrator library.

Adaptive support adds a numerical derivative obligation. An FD residual probe
must solve h for that probe state with the declared convergence policy. An exact
JVP must include dh/dY when the chosen residual includes h(Y), through a supported
solver derivative or implicit differentiation. Freezing h for probes changes the
Jacobian approximation and needs an explicit inexact/lagged policy and separate
qualification. Search coverage and initial guesses may be cached; they must not
make the residual depend on the order in which probes were evaluated. Support
solve tolerances/noise must be assessed against the FD perturbation scale.

### Smallest generic library design to settle

- An optional execution-resource adapter/protocol usable by a callable and by
  solver scopes. It distinguishes physical residual evaluation, vector operator
  action and derivative/adjoint probes, without adding SPH terms to the library.
  Plain stateless callables retain the existing interface and behavior.
- Explicit stage/block identity and a linearization context lifetime, including
  resource-renewal signaling and restart behavior. A per-evaluation mode flag
  alone is insufficient without ownership and promotion rules.
- A continuation transformation hook that operates on the real result and
  memory (state, history, reuse/warm starts and their aux associations), at a
  defined point relative to error/event/dense-output consumers. Do not overload
  the existing snapshot-only external transaction commit callback.
- Consistent snapshot/clone/AD contracts for resource handles: immutable sharing,
  private scratch and explicit promotion; no blanket tensor cloning or arbitrary
  shape-based remapping. Resource invalidation should be observable in diagnostics.

These are responsibilities to agree, not a commitment to a large new public API.
First determine what wrappers can provide and where a small lifecycle hook is
necessary. Integrator formulas/tableaux, fresh evaluations, tolerances and solver
termination behavior stay unchanged in this infrastructure work.

### Integrator-first implementation/qualification order

1. Write the minimal evaluation/resource/continuation protocol and source map;
   inventory ordinary, split and linear-action entry paths. Establish a toy
   particle-ID state and resource adapter before connecting SPH neighbor code.
2. Qualify explicit multi-stage and embedded RK, reuse/history, rejection and
   dense/event output with forced pending permutations. Verify that finalize
   proposals cannot change old-order arithmetic or publish speculative effects.
3. Qualify DIRK/JFNK FD base/probe/line-search isolation and controlled coverage
   renewal. Test repeated G(Y) before/after unrelated probes; compare finite-
   difference actions with an independent small dense-Jacobian oracle, checking
   convergence and residuals as well as final values.
4. Add exact-JVP and coupled block cases: tangent maps, per-stage resources,
   block flattening, split/linear actions and fixed preconditioner context. Include
   linear actions whose vector blocks are deliberately invalid physical positions
   to ensure no geometry is inferred from them.
5. Qualify implicit backward gradients after subsequent contexts have been built
   and old caches would normally be recycled. Verify no probe/adjoint increments
   event budgets or overwrites another evaluation's retained aux.
6. Connect SPH stage preparation/support solves and the selected-list-build
   sorting path only after these generic gates pass. Then resume the frontend
   eager, boundary, graph and population phases below.

Extend existing integrator suites (`test_jfnk.py`, `test_fullyimplicit.py`,
`test_preconditioner.py`, `test_rhs.py`, `test_run.py`, `test_run_outputs.py`,
`test_gradients.py`, `test_state.py`) with resource-sensitive synthetic cases.
Ordinary numerical tests with stateless RHS functions do not exercise these new
failure modes. The 2026-11-01 audit specified these gates. The 2026-11-02
checkpoint implements and tests the initial generic subset, updates the isolated
integrator execution queue, and records the remaining application/driver limits
above.

## Stage preparation, list lifetime and input-order results

The user proposes keeping compute ordering fixed for a neighbor list's lifetime,
returning dxdt/dvdt/etc. in the order supplied to the substep, and carrying an
optional updated sort with the adjacency through aux into simulator finalization.
Adopt this as the normal path. It removes the need to permute a live list merely
to adopt the compute order in which that list was already built.

Use a persistent spatial context containing the layout ID, adjacency, its build
anchors and the input/compute maps. A one-off permutation field only on the stage
that rebuilt is insufficient: later stages still need the context, and the last
stage may not itself have rebuilt. An optional transition proposal can indicate
that adopting the context would change the next step's storage layout.

For a step with input layout A and compute layout B, define `p[B_row] = A_row`,
`q[A_row] = B_row`. Stage preparation gathers needed fields as `valueB=valueA[p]`,
maintains support/geometry in B, evaluates equations there, and returns
`updateA=updateB[q]`. All particle-valued updates, derived fields consumed by
old-order finalization, diagnostics and reductions must obey this association;
returning only dxdt/dvdt in A while silently returning unlabelled supports in B
would still corrupt the state.

If a stage rebuilds again into C, its mapping must resolve against that stage's
actual input layout A (or carry an explicit composable chain), not be mistaken
for a B-to-C permutation. Prior aux snapshots must not change meaning when a
shared "latest context" pointer advances. Pair W stays attached to its own list
generation; map half-step velocity into that list's layout and particle energy
increments back to A. Reuse the existing fallback when generations change.

Finalize physics and consume stage work in consistent declared layouts before
adopting B/C. Remap simulator-owned history, ghost relations, bodies and external
particle inputs on adoption. On the next ordinary step, input and compute order
match, so the mapping is an identity fast path. Mapping overhead is normally
confined to the transition step rather than every step of the list lifetime.
Measure this cost separately from equation-kernel locality gains.

**Finalize is not universally an acceptance boundary.** In the current RK driver,
`stages` and an embedded `error` are assembled before finalize; generic history
can be pushed afterward from the original ks/rs. BDF also creates history after
finalize. The integrator runtime computes error norms/acceptance after the entire
driver returns. Reordering only the final state therefore cannot be invisible to
all these consumers. A fixed-step one-step path can commit at the end of finalize
when no such consumers remain. Otherwise finalize prepares a pending change and
a simulator-owned result adapter commits after acceptance, mapping retained
state/update/history/error context as required. Composed substeps and reruns do
not independently publish changes. Integrator numerical formulas stay unchanged;
any general lifecycle bridge is a small separate concern.

### Separate numerical stage geometry from storage/cache management

Neighbor-list construction is a resource operation; adaptive support is a
scheme-dependent numerical stage operation. Put their coordination in simulator
stage preparation, then pass a prepared stage to equation kernels. Do not move
all support/search work to once per full step: stage positions and supports can
invalidate coverage during RK/implicit stages.

Conceptually, preparation supplies a candidate search envelope, solves the
stage's support/density constraints, validates coverage of required support
iterates and the converged supports, enlarges/rebuilds and repeats if needed,
then supplies pair geometry valid for the actual RHS operator. This need not
mean redundant builds/checks: preserve the existing optimized list/grid paths.
In [solveStageSupport](src/warpSPH/modules/adaptiveSupport/stage.py), list mode
already returns a list validated against the solve, while grid mode performs
the support solve on a grid and subsequently obtains the pair list. The Owen
solver's [validateOnce path](src/warpSPH/modules/adaptiveSupport/optimalSupportOwen.py)
tracks peak supports and restarts from incoming supports when coverage fails.
Preserve that retry semantics; a truncated support solve cannot be fixed merely
by rebuilding the force list after it has returned.

Distinguish three operations: **recompute stage geometry**, **rebuild a list**,
and **choose a new layout**. Support changes within the skin need no list rebuild.
A coverage failure requires a rebuild but does not require sorting. Several list
generations can share a layout epoch. Keep an existing compute order on support-
driven rebuilds by default; choose a new Morton order only when the locality
policy warrants it, preferably coalesced with a rebuild. An invalid list can
never be retained until a convenient reorder/finalize boundary.

An integrator can continue calling the same f(stage) interface: that callable is
the simulator's stage adapter, which runs preparation, equation evaluation and
input-order conversion. The equation kernels receive an explicit valid context
and do not allocate/sort/rebuild themselves. Cache/resource changes must not
commit physical emitter/body/source changes during speculative evaluations.

## Exact permutation rules

Let `p[new] = old` and `q[old] = new`, so `q[p] = arange(N)`.

- Particle values: `new_value = old_value[p]`, including UIDs, kind, material,
  source, body labels, supports, all integrated fields and particle histories.
- A full-row ghost relation:
  `new_g[i] = -1 if old_g[p[i]] == -1 else q[old_g[p[i]]]`.
  Offsets use only `old_offsets[p]`. Verify ghost/boundary reciprocity and
  ownership. Never apply `q[-1]` and subsequently try to repair it.
- A compact list of referenced rows, such as reservoir owners, maps its values
  through q. Its own compact axis is not the physical particle axis.
- A full-length membership mask gathers through p. Body-local arrays retain
  their local order and resolve to global rows by explicit ID-based indices;
  boundary and ghost storage order need not match after sorting.

The following CSR utility is needed only when actually changing an existing
list's layout (including an exported adjacency view), not when adopting the
sorted compute layout that already owns a list. Query and reference
permutations can differ (`pQ`, `qR`).
For each new query row `i`, copy the old row `pQ[i]`:

```text
new_counts[i] = old_counts[pQ[i]]
new_offsets  = exclusive_prefix_sum(new_counts)
old_edge     = old_offsets[pQ[i]] + local_neighbor_slot
new_edge     = new_offsets[i]    + local_neighbor_slot
new_J[new_edge] = qR[old_J[old_edge]]
new_I[new_edge] = i                         # only if stored
new_edge_value[new_edge] = old_edge_value[old_edge]
```

Preserve the order *within* an old row initially. Preserve numEdges, capacity
rules, optional-I representation and -1 tail padding. The row prefix scan and
edge addresses use configured offset types, independently of particle ID width.
Do not reconstruct a full COO I array merely to remap a CSR-only list.

Also permute **build-time** query/reference positions and scaled supports on
their respective axes, preserving self-query aliasing where applicable. A
pure permutation must preserve Verlet usage and rebuild decisions; replacing
anchors with current positions would incorrectly reset the skin budget.
Invalidate/rebuild retained hash grids or update all their `sortIndex` and
sorted buffers deliberately. Tag the result with both new layouts. A mismatch
must be rejected/rebuilt **before** any consumer runs, even when N matches.

Pairwise ap/av/f, CRK W, cached pair geometry and any retained edge diagnostics
follow the same old-edge-to-new-edge mapping or are explicitly discarded and
recomputed. Accumulated work already needed by a step cannot be discarded.
This is a principal reason to sort outside the entire step. Moving only J
indices is insufficient: CSR row ownership would still be wrong.

Births/deaths are not bijections. Initially compact survivors, append initialized
births, fix surviving relations and **rebuild all affected neighbor lists**.
Do not try to reuse an old list that cannot contain new particles. Retain build
anchors only for a genuine permutation, never to disguise membership changes.

## Time integrator contract and its inputs

The integrator should remain solver-agnostic: it need not know Morton keys,
SPH materials or the emitter geometry. It needs a protocol for logical layout
identity and a caller-supplied transformation of stored state/update/aux data.
Use tagged field discovery consistently rather than assuming `uid` versus
`UIDs`. A no-layout system retains its existing behavior.

Keep the entire input bundle coherent:

- Accepted system/state, derivative update objects and all their particle axes.
- `referenceState`/typed `StateBlend.reference_state`, initial-step states,
  cached stage/FSAL/`priorStep` values and any retained aux state/adjacency.
- `StepHistory` entries (`update`, `state`, `aux`) and method-owned context,
  including NDF/variable-order caches or flattened vectors where present.
- Per-particle inputs reachable through positional/keyword arguments, config,
  source/boundary objects, driven reference data, constraints, tolerance arrays
  and user callbacks. Global time/dt, tableau coefficients, OU mode state and
  other global scalars remain global.
- Implicit solver vectors, block-stage layouts and preconditioners: keep them
  fixed within an attempt and invalidate any persistent cache at a transition.

Do not change integration-visible order in `preprocess`, equation evaluation,
nonlinear residual/JVP, substep initialize or an inner composed finalize.
Simulator stage preparation may create a different private compute order on a
rebuild, provided all returned numerical values respect the input-order contract
and derivative paths preserve the explicit mapping. Coverage-driven rebuilding
can also retain the same compute layout.

**Pure reorder:** remap every supported retained particle history without
changing method, dt history, coefficients or stage reuse policy. Clear only
disposable caches. Aux objects with undeclared semantics require a clear
unsupported/restart policy; do not recursively permute arbitrary tensors by
shape. Validate layout before reading history, not merely when pushing a new
entry. The current optional pointer fingerprint is insufficient and is not
uniformly wired into the drivers.

**Membership change:** invalidate cached derivatives/FSAL, even for survivors,
because the interaction operator can change. Initially restart generic multistep
methods through their registered startup path, clear incompatible solver memory,
and record the event. ACSPH needs a separate adapter for system-owned BDF data
and compensated position fields. Never manufacture a newborn's physical past
by copying its birth state into all previous time levels.

Repeated insertion every step would repeatedly trigger startup/BDF1 and can
permanently reduce accuracy or erase the benefit of a history method. Therefore
first qualify emitters/outlets on supported one-step RK paths. ACSPH/history
methods need an explicit, qualified policy for frequent events before claiming
production support; survivor remapping/newborn startup masks alone do not prove
accuracy near changing neighborhoods. Do not silently change an integrator.

The frontend's ordinary runner currently does not thread generic `history=`.
Add that ownership only with the corresponding feature gates; it must not make
an unrelated promise that all library methods already run warm in the frontend.
The distinction between a **spatial Verlet neighbor list** and the
**velocity-Verlet time integrator** should be explicit in API/docs/tests.

## Sorting policy and performance

- Add a storage-order setting separate from `neighborQueryOrder`, initially
  opt-in (`input`/`morton`), plus selected-rebuild/adoption and export-order settings.
  Retain input order as the control and for unsupported custom adapters.
- Use a stable sort of quantized Morton keys; ties retain prior storage order.
  Initialize from canonical order. A UID secondary key is a separate optional
  determinism policy, not needed every time. Define key bits, 1D/2D/3D encoding,
  periodic wrapping, signed/open-domain coordinates and out-of-range behavior.
  Never silently alias overflowing keys. Existing Morton helpers need dedicated
  validation before reuse.
- Use a reproducible quantization origin and scale over a layout epoch. If scale
  follows a changing maximum support, changing it can reorder much of the set;
  record that as a layout-policy change. Sorting bin size need not equal the
  compact hash's cell size.
- Exploit temporal coherence first by **sorting infrequently**, coalescing with
  planned neighbor rebuilds and skipping identity permutations. A stable sort
  preserves ties but is not automatically an incremental sort. Even when few
  particles move, a general gather still streams most persistent fields.
- Compare initialization-only, selected rebuilds and later disorder-triggered
  policies; interval checks can request a later suitable reorder rather than
  changing a live list's layout. Avoid a host read every step just to discover
  that the permutation is almost unchanged. Do not force an extra neighbor
  rebuild every step in the name of sorting.
- Measure key generation, sort, transition-stage gather/inverse gather,
  history/body/source remap, list rebuild and any separately requested CSR remap,
  graph invalidation/recapture, export projection and complete-step
  throughput. Include changing and already-local flows, small and large cases,
  2D/3D, and peak live memory. Keep the physical solver, stage count, tableau,
  lanes, support policy and precision fixed in comparisons.
- Budget scratch explicitly. Permutations/inverses cost O(N); gathering state
  and history needs temporary ownership. CSR relocation costs O(N+E) and can
  need an extra edge buffer, significant at the existing multi-GiB list sizes.
  Reuse valid scratch only when ownership permits. Remapping may cost more than
  rebuilding; choose from measurements rather than assuming it always wins.

Pure data gathers/inverse gathers should be exact. Full trajectories need
precision-aware tolerances: rebuilding in the new layout can change per-row
neighbor summation order and floating-point reductions. Preserve the legacy
bitwise unsorted control, and separate indexing correctness from roundoff and
physical accuracy. No artificial reuse of derivatives to recover timing.

For differentiable runs, a pure permutation needs the corresponding inverse
scatter in backward/JVP paths; do not detach evolved fields just to reorder them.
Keep each compute context's chosen permutation fixed for that context, keep
integration-visible order fixed during an attempt, and preserve all associations
for backward/replay. Key selection and birth/death decisions
need an explicitly documented differentiation policy. Keep unsupported topology
events out of AD paths until that policy is implemented and qualified.

## Export, restart and external consumers

For fixed populations, create an immutable canonical-order snapshot at export
cadence. Apply it to every particle-valued output, not just positions, including
extra fields, budgets with particle axes and stage outputs when requested.
Remap exported ghost/neighbor references into the snapshot's index spaces.
Leave live computation in spatial order. Persist initial canonical metadata in
that same order; the existing static-metadata trajectory route can then remain
compatible for fixed-roster runs.

For changing populations, add a versioned trajectory schema with per-frame
active int64 IDs and matching particle metadata (initially write the complete
active roster; optimize with birth/death catalogs later). Canonical order means
surviving original ordinals followed by births; frame lengths can differ.
Explicitly define identity joins, missing particles and birth/death masks in
dataset time windows, diagnostics and analysis. A constant array index is no
longer a particle trajectory. Do not resurrect deleted rows just to keep shapes.

Checkpoints should preserve a coherent native layout by default, with all state,
adjacency, system/integrator history, layout association, ID allocator high-water
mark, emitter schedule/RNG/accumulators and outlet state. Keep analytical exports
canonical. If a checkpoint is canonicalized, its entire indexed context must be
transformed too. A restart with equal dimensions is not evidence that saved
adjacency matches its particles. Once deletion exists, `max(surviving_UID)+1` is
not an adequate allocator restore rule; persist the real counter.

Legacy files remain readable. Migrate legacy signed ghost IDs explicitly and
stop narrowing new persistent IDs to int32 in HDF5. Ghost removal in dataset
conversion must invalidate/remap references, not retain indices into removed
rows. Keep state export and per-frame metadata formats distinguishable.

Audit plotting GPU buffers, tracked-particle selections and per-particle colors
for both row changes and count changes. Selections should use persistent IDs;
buffers may need resizing. Qualify the remote viewer's completed-step adapter,
variable-length frame handling and recordings before enabling changing counts
for remote runs. These external APIs were not inspected in this source audit.

## Emitters/outlets: initial scope

Start with **fluid births/deaths** on the qualified one-step path. Boundary/body
creation, removal and ghost lifecycle are later extensions. Until supported,
reject such events clearly rather than leaving dangling links.

At an accepted time boundary, evaluate deterministic removal/injection events,
compact survivors, allocate fresh IDs, append initialized births and optionally
sort once. Initialize all scheme-required fields: primitive or conserved data,
supports, material/source/body/kind labels, EOS consistency, viscosity-switch
memory, warm starts and cumulative source fields. Retain survivors' budgets;
book inlet/outlet mass, momentum and energy as external exchange, including
removed kinetic/internal energy. Reservoir Riemann fluxes and explicit births
must not double-count the same boundary exchange.

Event decisions and emitter accumulators advance once per accepted interval,
never during an RK stage, graph warm-up, failed capture, rejected adaptive step
or compatible-energy rerun. Define event-time splitting and rate-accumulator
semantics before implementing fractional-rate injection. Checkpoint those
semantics, not just particle arrays. Test duplicate/crossing events and combined
insertion/deletion with unchanged total N.

Use dense active arrays first, with explicit shape changes/reallocation outside
graphs. Existing kernels commonly launch by tensor shape; a capacity-sized
particle pool would require a much broader active-count/mask ABI. Add pooled
capacity later only if event/allocation measurements justify it. Define all-empty
fluid behavior at the scheme/runner level; core empty-search support alone does
not make every timestep reduction, EOS or diagnostic empty-safe.

## Implementation sequence and exit gates

| Phase | Deliverable | Gate before proceeding |
|---|---|---|
| 0 — integrator contract first | Audit/qualify evaluation-resource scopes, JFNK linearization/probe isolation, split/vector actions, real continuation transformation and AD resource lifetime using synthetic systems. Agree field/identity adapters and import provenance alongside this. | Explicit/embedded, DIRK FD/JVP, coupled block, rejection/dense/history and backward resource controls pass before frontend sorting is wired. Plain callables remain compatible; no SPH-specific integrator algorithms. |
| 1 — fixed-N eager prototype | Simulator stage preparation and explicit spatial context; build directly in Morton compute order on selected rebuilds; return input-order updates and carry context/proposed adoption via aux. Start with periodic one-step RK, canonical export and safe end-of-step adoption. | No-sort control unchanged; generation ordering remains fixed; transition-stage derivatives match after inverse mapping; the next step takes the identity-map path; mid-stage rebuilds and support retries preserve physics. |
| 2 — context adoption and histories | Input/compute-layout handling for aux/derived fields/pair reductions, simulator-owned result acceptance adapter, integrator state/update/aux history and frontend system-history adapters. CSR remap utility only for explicit independent reorder/export needs. | Final state, error, reuse and warm history remain coherent; failed/replayed attempts never publish a transition; no generic sort logic in integrator algorithms; list validity/anchors unchanged by adoption. |
| 3 — boundaries and scheme coverage | Ghost maps, body-local scatter maps, ACSPH BDF/compensated fields, DFSPH impulses, switch histories, meshless Q/geometry/driven/reservoir inputs and relevant source objects. | Mixed materials and moving-body/ghost cases pass forced reorder checks; scheme-native work/conservation gates pass; all supported entry paths use the coordinator. |
| 4 — graph/pipeline and locality qualification | Correct capture invalidation and recapture, snapshot lifetimes, bounded sorting cadence, export/viewer checks and whole-step performance evidence. | Eager/graph and sequential/pipelined controls agree within declared tolerances; forced graph redo does not apply a transition twice; amortized gain and peak memory measured. |
| 5 — changing populations | Variable-roster IO/restart/dataset schema, dense survivor/birth transactions, fluid emitter/outlet events, budgets and one-step support. | Variable and same-N membership changes, rejected-step replay, empty cases and restart across events pass; measured external mass/momentum/energy balances close. |
| 6 — production policy | Frequent-event multistep/ACSPH policy, broader sources/callbacks, plotting/remote compatibility, distributed integration and optional capacity pools. | Explicit method-accuracy qualification, supported-combination matrix and performance evidence before defaults change. |

Phases 2 and 3 may be developed incrementally per family, but topology changes
must not precede the identity, boundary transaction and versioned IO foundations.
Sorting must not be advertised as universal while a scheme's history/input
adapter is missing. Graph enablement follows eager correctness.

## Verification matrix

1. **Algebra:** random permutations and inverse round trips over heterogeneous
   particle fields; None/scalars/global and edge arrays; custom state extensions;
   IDs beyond int32; stable ties, periodic seams and coordinate/key limits.
2. **Relations:** reciprocal ghost links with -1 fluid sentinels; independent
   boundary/ghost ordering; multi-body local geometry, normals, accelerations,
   force/torque and motion invariant under permutation.
3. **Neighbors:** self and independently permuted query/reference sets (including
   equal sizes), gather/scatter/symmetric modes, empty sets, variable supports,
   padding/capacity and CSR-only/COO, configured ID/offset widths. Compare physical
   ID pairs against a small direct-search oracle, not raw row numbers. Compare
   skin usage just below/above the rebuild threshold before/after permutation.
   Test several same-layout list generations caused by support growth, support
   retry after insufficient coverage, two compute-order changes in one attempt,
   and later stages carrying a context whose permutation was proposed earlier.
4. **Integrators:** multi-stage RK, velocity-Verlet drift fields, composed steps,
   priorStep/FSAL, warm Adams/BDF and adaptive/implicit paths where supported.
   Warm histories must follow particles without incidental restart on sorting;
   stale context must be rejected before use. Test immutable old snapshots and
   explicit topology-startup behavior. Exercise ACSPH system-owned history apart
   from generic library history. Check gather/scatter gradients and a small
   forward-mode/implicit fixture with a frozen permutation on supported AD paths.
5. **Scheme memory/work:** CompSPH/CRK compatible-energy work and rerun paths,
   viscosity switches, DFSPH pressure warm starts, meshless Q/closure/source
   budgets, driven masks and virtual-reservoir owner references. Test reorder
   after several accepted steps, not only before the first RHS.
6. **Execution/outputs:** CPU then sequential CUDA; eager/graph, in-place rebuild,
   forced capture/redo, sequential/pipelined output. Canonical frame and stage
   exports, full checkpoints and resumes before/after several sorts/events.
   Include old-file loading, ghost-filtered datasets and ID-based time joins.
7. **Population:** births, deletion of highest-ever ID, same-N replacement,
   all-fluid deletion, deterministic injection cadence, duplicate event defense,
   outlet crossings and budget conservation. Restore the allocator and event
   clocks exactly on resume; repeat through replay/rejection.
8. **Performance:** initially-local and deliberately shuffled/developed flows;
   sorting off vs chosen cadences, rebuild vs remap; matching solver/precision/
   stage/lane settings. Report complete-step ms, distribution of reorder/rebuild
   costs, memory, graph recaptures and export overhead. Qualify physical accuracy
   and conservation as well as speed. Use existing runner video/progress rules
   for simulation validation; no long campaign is part of this scoping task.

Extend existing coverage in `test_particleIdentity.py`,
`test_compressibleIntegratorContracts.py`, `test_meshlessNeighbors.py`,
`test_drivenTurbulence.py`, `test_compSPHGraph.py`, `test_crkGraph.py`,
`test_cudaGraph.py` and the boundary suites. Add meaningful permutation/event
oracles rather than tests that only repeat the implementation's gather calls.

## Relationship to chunked neighbors and multi-GPU work

[The chunk/multi-GPU plan](CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md) requires all ranks
to agree on particle order for a generation. Add logical layout/topology to that
generation contract before interfaces harden. First qualify single-device sorting;
do not expand this request into implementing a distributed backend.

At a distributed reorder boundary all ranks must apply the same permutation,
then recount/repartition whole query rows as necessary. Stable Morton ties must
agree across ranks. Chunk ownership follows storage rows, whereas physical
identity survives ownership changes. J remains an index into the current global
reference storage, not an immortal UID. W follows its edge/row owner.

Initially rebuild/repartition distributed chunks after a sort, rather than
assembling a global E-sized remap buffer. Preserve the existing rule that no GPU
gathers the complete J/W list. Later distributed CSR remapping must move complete
rows and their retained work together and reconcile capacities deterministically.
Topology events also require consistent ID allocation and exactly-once exchange
budgets across ranks. This is an interface dependency on the existing plan,
not a change to the user's current Focus or authorization for new runs.

## Immediate next implementation slice

The first eager fixed-count runner phase is CPU-qualified: whole-step Morton
order, accepted adoption, bounded structured coverage renewal, UID64 trajectory
axes and physical checkpoint/cache rebuilding. Integrator main contains the
resource/continuation APIs at `28d8bdd`; the simulator runner work remains on dev.
Next adapt history/prior/FSAL continuation before enabling reuse or multistep
methods. CUDA eager qualification remains blocked by the unavailable driver.
Boundary/ghost relations, sources, population events and graph capture follow
those gates. See the [current report](docs/particle_layout/RENEWAL_AND_PERSISTENCE_2026-11-02.md).

## Simulator preparation checkpoint 2 November 2026

User direction: fix default callback identity/serialization and quantify disabled
scope overhead before proceeding up the stack. Those fixes are applied in the
isolated integrator worktree. Real SPH inputs also exposed a pre-existing inactive
optional integrated-field residual error; matching `None` fields now remain
inactive, and asymmetric activation is rejected explicitly.

The next application slice is implemented: a shared CompSPH/CRK wall/support/list
preparation seam, optionally run by system preprocessing before the equation RHS.
The opt-in spatial manager isolates implicit scratch and PairWork and pins
coverage per stage within each JFNK linearization. Support remains stage-dependent.
Coverage replacement raises `ResourceInvalidated` before physical equations;
caller renewal requires a complete retry. Converged materialization retains the
existing aux/finalize contract. Default equations and support retry order are
preserved. Existing source observers and forced-replay tests follow the moved seam.

[Design, usage, measurements and limits](docs/particle_layout/STAGE_PREPARATION_2026-11-02.md).
Qualification (overlapping selections, not additive totals):

- Integrator explicit/implicit/RHS/runtime resource selection: **599 passed**,
  before the inactive-field fix; final state/resources/RHS/JFNK **922 passed,
  44 skipped**. Installed isolated wheel: **49 consumer tests passed**, plus
  existing all-module/135-driver smoke. Validator: 1,892 imports, 1,112 local links.
- Frontend broad targeted gate: **94 passed, 24 skipped**, 38.60s. Includes real
  FD-JFNK backward Euler and coupled Gauss stage equations, private probe scratch,
  independent fresh-list oracles, support-growth abort/retry, compatible-energy
  replay/storage tests and existing support/graph eligibility checks.
- Final new adapter suite: **21 passed**, 4.12s, after adding actual unsupported
  exact-JVP/implicit-AD failure/cleanup checks and an explicit once-per-stage
  support-call counter. No runtime-code changes after the broad frontend gate.
- FP64 spatial/IdealH/entropy selection: **27 passed**, 10.12s (before the two
  additional unsupported-capability checks). This includes the moved replacement
  support probe and actual implicit stage equations.
- Against the unchanged integrator 0.7.0 checkout: **42 passed, 24 skipped**,
  4.55s; ordinary frontend imports/steps need no optional resource API.
- Three-step default frontend comparison against original frontend + unchanged
  integrators: exact SHA256 match over particle fields and all stage derivatives
  for both CompSPH/CRKSPH. Tiny CPU full-step benchmarks: disabled +0.47%/+0.52%,
  overlapping sample variation. Interleaved integrator CPU measurements span
  −1.58% to +0.34%; no resolved default-path slowdown at these sizes. No GPU or
  sorting-performance claim.

Local evidence: `.tmp/particle-layout-2026-11-02/` in this checkout and
`.tmp/evaluation-resources/default-compatibility/` in the isolated integrator
worktree. Includes JUnit, raw benchmark samples/source, wheel report and provenance.
An earlier wider selection found an outdated equation-module monkeypatch; the
hook was moved and its forced-replay tests passed in the final gate. A duplicate
longer CPU pair-work sweep was interrupted and is not counted as qualification.

Remaining gates: application coverage-renewal/envelope policy, private fixed-count
compute layouts and IDs/epochs, input-order derivative mapping, safe frontend
accepted adoption and canonical export. Exact-JVP/implicit-AD support preparation,
implicit compatible-energy quadrature, external source transactions and CUDA graph
lists are explicitly unsupported by this first adapter. Sorting and population
changes are not yet implemented. The next locality fixture should use the prepared
stage seam with sorting only on selected list builds, then qualify adoption and
export separately. No topology change precedes these fixed-count contracts.

## Fixed-count Morton checkpoint 2 November 2026

User continued implementation after the stage-preparation checkpoint. The first
fixed-count eager adapter is now implemented and qualified on CPU:
[MortonStepper report and usage](docs/particle_layout/MORTON_STEPPER_2026-11-02.md).

The implementation uses a private whole-step compute order. RK stages,
compatible-energy work/replays and FD-JFNK vectors all share that order, and the
simulator boundary maps public state, stage fields/derivatives and error back to
input order. This is a coarser first boundary than changing compute order inside
individual RHS calls. It keeps native finalization intact and leaves a later
per-RHS optimization independent of this correctness gate. Sorting occurs only at
initial admission and selected stale boundary-list builds; support-driven rebuilds
inside a step retain the layout.

Explicit accepted-result adoption preserves the list and build anchors, moves
error fields with state, and retains historical stage snapshots in their evaluated
order. The following identity path performs no particle/update mapping or extra
particle-buffer copies. Same-count membership/order changes, unknown field axes,
unadapted histories/reuse, external inputs, preconditioners and row-valued
tolerances are rejected. Fluid-only periodic eligibility excludes unadapted
boundary/ghost relations. No integrator numerical formula was changed in this slice.

Self-contained HDF5 frame/stage export obtains an owned UID-ordered view. Requested
CSR export remaps rows, reference indices, anchors and retained edge values
consistently, preserving widths, capacity padding and optional-I storage in memory.
Particle-only export skips edge remapping. A persistent marker keeps exports
canonical if an adopted state leaves the private cache for an ordinary driver.
Unlabelled export extras and the existing fixed-roster trajectory writers are
guarded until their explicit axis/roster adapters exist.

Final gates (overlapping selections, not additive totals):

- **146 passed, 24 skipped**, 35.69s: layout/resources and existing compressible
  integrator, shared/replayed work, support clamp, graph eligibility, IdealH and
  entropy controls. Includes **50 new layout cases**.
- **73 passed**, 9.55s, FP64: all 50 layout cases, 21 prior resource cases and
  the two moved IdealH checks. Explicit reverse-gradient comparisons use fixed
  support; frozen implicit/support AD remains unqualified.
- **42 passed, 24 skipped**, 4.05s, with unchanged integrators 0.7.0: ordinary
  frontend behavior needs no optional new API. Integrator main remains clean;
  its isolated resource worktree is unchanged by this frontend slice.
- Independent oracles cover Morton 1–3D/3- and 21-bit keys and periodic stable
  ties; all registered particle columns joined by UID; rectangular CSR with
  separate query/reference orders, both offset widths, optional I and empty rows;
  physical neighbor ID pairs and edge values; unchanged skin/anchors; actual
  backward-Euler/Gauss stage equations; compatible-energy replay and conservation;
  support retry with one fixed layout; stale-boundary composition, failed-attempt
  isolation, embedded errors, viscosity history and pre/post-adoption frame output.
- CUDA qualification was unavailable: `nvidia-smi` could not communicate with
  the NVIDIA driver. No GPU, graph or large-flow performance claim.

The reproducible CPU boundary probe is
`benchmarks/particle_layout/bench_morton.py`: float32 periodic shuffled 1D RK4,
compatible energy, one numerical-library thread, CPU 0 pinned, five interleaved
samples of four calls. At N=32 the option costs roughly 0.3–0.6ms per step. At
N=4096 retained shuffled/Morton medians are 140.613/133.404ms (CompSPH) and
171.940/160.213ms (CRKSPH). Adoption costs about 0.5ms at N=4096; canonical
state-only views about 0.14ms without adjacency / 0.60ms with CSR, excluding file
I/O and stage views. Samples/native-sorted controls and limitations are retained;
these modest CPU results do not establish the motivating 2× large-scale benefit.

Local evidence: `.tmp/particle-layout-2026-11-02/morton-stepper/`, including final
JUnit/logs, pinned interleaved benchmark JSON/source, source provenance and diff.
No simulation campaign/video was run; fixtures and timing loops are small CPU
probes. No default, package publication, commit or push was made.

Next: application renewal envelopes, runner/trajectory UID/axis schema and
acceptance bridge, then persistent checkpoint/history/prior adapters. Qualify
CUDA eager behavior and host-control overhead before larger locality cases or
production policy. Boundaries/ghosts, external sources, topology events and graph
capture remain separate gates. The prototype is available through the direct
step adapter; it is not yet a runner/CLI option.


## Runner acceptance bridge 2 November 2026

User instruction: commit the qualified state, then start the next phase. Both
checkpoints were committed locally; no branches were pushed. The new work stays
on frontend `dev` and uses the isolated integrator branch explicitly in tests.

Implemented and CPU-qualified: `CaseSpec.particleOrder` defaults to
`input`; `morton` admits only eager periodic fluid-only CompSPH/CRK with registered
one-step tableaus and native RHS functions. The adapter is selected before output
creation. Initial and later state files use UID order, with Python/NumPy scalar
metadata allowed and unlabelled arrays still refused. Hooks/timestep/diagnostics,
frame consumers and stop checks consume the current input order before accepted
adoption. Exposed `lastStageUpdate` follows the adopted state's rows; historical
stage snapshots retain their actual evaluated order. Reported step time includes
layout adoption. The CFL runner does not perform embedded-error rejection;
resource invalidation still unwinds the whole attempt rather than auto-retrying.

Trajectory output, checkpoint restart, graphs/pipelining and multistep/reuse
histories remain guarded. Case hooks that retain particle columns must join by UID
when opting in. This is data lifetime work; no numerical formula/default changes.


Final qualification: 90 passes in each precision for the runner/layout/resource
suite, including 19 runner cases. The broader selection produced 317 passes,
25 skips and 10 existing hard-CUDA initialization failures (driver unavailable).
Ordinary runner/contract controls with unchanged integrators 0.7.0 passed 87 tests
with 24 skips. Final tiny CPU runner medians include adoption: CompSPH input/Morton
7.934/8.501ms (N=32), 39.389/38.935ms (N=1024); CRK 10.616/11.012ms and
49.661/48.569ms respectively. The smaller case has overhead and larger CPU gains
are modest/variable; CUDA and the motivating large-scale benefit remain unproven.
Small repeated fixtures omit video and retain streamed progress, not a campaign.

[Runner usage, ownership, measurements and outstanding gates](docs/particle_layout/RUNNER_2026-11-02.md).
Evidence: `.tmp/particle-layout-2026-11-02/runner/`. Current runner work is
uncommitted; the requested preceding checkpoints are committed. Next implement
structured renewal envelopes/whole-attempt retry, then trajectory UID axes and
checkpoint cache rebuilding before history/reuse or topology changes.


## Integrator main synchronization 2 November 2026

The user requested merging the isolated integrator branch because the simulator
work on `dev` otherwise expected APIs absent from the active dependency.
`/home/lu26029/dev/warpSPHIntegrators` was clean on `main` at `1fea2a3`; it was
fast-forwarded to `28d8bdd`. The isolated branch/worktree is retained. The warp
conda environment already installs the main checkout in editable mode, so its
resource/continuation APIs now update with the merge without reinstalling.
No remote push or package publication was performed. Earlier old-release gates
above refer to pre-resource source at `1fea2a3`, rather than the now-updated main
checkout. Frontend runtime verification uses the normal environment with
`PYTHONPATH` removed.

Post-merge normal-environment validation (no `PYTHONPATH` overrides): **90 passed**,
covering runner, layout and spatial resources. Imports resolve to the main
integrator checkout. Evidence: `.tmp/particle-layout-2026-11-02/integrator-main-merge/`.


## Renewal and UID persistence implementation 2 November 2026

User direction: return to the next implementation steps after aligning integrator
main. Bounded whole-step coverage renewal is implemented outside integrator
scopes. Per-stage envelopes own UID order, anchor positions and candidate radii;
FD/line-search/materialization and support-iteration peak checks cannot replace
an active candidate. Insufficient reach carries structured evidence and unwinds
the full step; retry starts from private incoming state with no retained Krylov
basis. Two renewals are allowed by default, configured with coverageRetries or
MortonStepper.max_coverage_retries. Generic ResourceInvalidated exceptions remain
unhandled by this policy. No equation, support convergence rule or default
particle-order policy was changed.

Fixed-roster trajectory files now canonicalize particle rows by UID and label
particle/global axes, preserving UID64 and scalar/array precision. Frame UID
links refer to the header roster; counts/membership and extra-field axis changes
are rejected. Stage export has independently canonical particle/update/CSR views.
Scalar extras cannot replace layout metadata. Unknown row arrays remain guarded.

Sorted state checkpoints record layout version, scheme, domain and precision.
Restart checks roster/static fields and physical state before output creation,
restores time/history/counters, and discards every list/grid/pair-work/layout
cache. It then builds fresh compute order/resources on the first continuation.
These checks do not replace case-specific parameter/forcing restoration; external
source/boundary/history/reuse eligibility remains restricted.

Qualification in progress: renewed real FD-JFNK across all four support modes,
runner retry-budget behavior, UID64 trajectory/stages and global E=N fields,
checkpoint continuation with viscosity history and renewed implicit solves,
ordinary runner/default controls and small paired CPU overhead probes.


Final renewal/persistence qualification: **403 passed / 25 skipped** on the broad
CPU selection (10 hard-CUDA tests deselected); **138 FP64 passes**; **123 passes /
24 skips** against archived pre-resource integrators `1fea2a3`. There are 31
coverage and 17 persistence cases. Exploratory FP64 failures from float32-hardcoded
legacy support fixtures and a tight-tolerance FP32 benchmark are retained, with
final supported gates and independent controls described in the report. No
numerical equation, physical convergence tolerance or default order was changed.

Three paired CPU timing processes show small differences (-5% to +3.6% across
controls/routes), including comparable variation in unchanged native controls;
no large-scale/GPU promotion. `nvidia-smi` remains unavailable. Small fixture/A/B
loops omit video and keep streamed progress. Sources/results are frozen in
`.tmp/particle-layout-2026-11-02/next-phase/`.

[Renewal policy, UID schema, restart checks and qualification](docs/particle_layout/RENEWAL_AND_PERSISTENCE_2026-11-02.md).
Next: history/prior/FSAL adapters, CUDA eager qualification, then external/boundary
and changing-population contracts. This simulator slice remains uncommitted on dev;
integrator main stays clean at the previously merged commit.
