# Fixed-count Morton stepping — 2 November 2026

This continues the [stage preparation checkpoint](STAGE_PREPARATION_2026-11-02.md)
and [particle layout plan](../../PARTICLE_LAYOUT_PLAN.md). It is an opt-in eager
prototype for periodic, fluid-only CompSPH/CRKSPH. Ordinary drivers keep their
existing behavior. Integrator numerical algorithms are unchanged.

## Boundary and ownership

`MortonStepper` runs the complete one-step integrator in a private compute order.
That order is fixed through all stages, compatible-energy replays, Newton
linearizations, FD probes and materialization. An RHS returns derivatives in the
order its integrator supplied. After numerical finalization, the simulator adapter
maps the state, all returned particle derivatives, stage fields and embedded error
back to the caller's input order.

This first implementation chooses the full-step boundary for the mapping. It
avoids moving accumulated compatible-energy work between layouts inside a step.
A support-driven list rebuild retains the current compute order. Initial sorting
and later sorting at stale boundary lists build directly in the new order.
`sort_on_rebuild=False` retains the initial order across subsequent generations.
A finer per-RHS transition remains possible later, with additional work ownership.

`ParticleLayout` owns p[compute] = input, q[input] = compute and both UID sequences.
Its handle identity denotes a layout epoch; adjacency identity denotes a list
generation. Several generations can share one layout. Every later stage carries
its actual layout/list context, including stages which did not rebuild. Re-sorting
composes against the current input order. Same-N membership/unannounced order
changes are rejected before cached lists are used.

The field schema is explicit. Particle provenance, integrated values, support,
material/UID fields and per-particle viscosity histories follow particle rows.
Counters, scalar timestamps and the boundary budget are global. ap/av/f and
compatible work are edge data, including when E equals N. Unknown schema fields
fail before sorting. The prototype supports logical IDs beyond int32 independently
of configured neighbor index/offset widths.

Before adoption, a public adjacency is `None` when its private compute rows differ
from input rows. `system.spatialContext` owns that compute list. Stage aux is
`(input_order_adjacency_or_None, input_order_state, SpatialStageContext)`; the
third entry owns compute adjacency/layout and any retained edge fields. No
unlabelled compute indices are passed off as input-order indices.

## Acceptance and continuation

```python
from warpSPH.evaluation import MortonStepper, adopt_spatial_layout
from warpSPHIntegrators import get_scheme

step = MortonStepper(get_scheme('RK4'))
proposal = step(state, dt, config=config, schemeConfig=schemeConfig)
# Fixed-step owner: inspect status/diagnostics and accept before adoption.
if not proposal.step_status.successful:
    raise RuntimeError('step failed')
accepted = adopt_spatial_layout(proposal)
state = accepted.state
```

For an embedded/adaptive method, perform the application's error test and other
acceptance consumers first, in input order. `step_status.successful` alone is not
adaptive error acceptance. Adoption refuses numerical failure and unadapted
continuation memory. It transforms the final state and embedded error together;
historical stage records remain in their evaluated input order with explicit IDs.

Adoption keeps the exact compute list and its build anchors. It does not spend or
reset the Verlet skin. On subsequent identity calls the adapter performs no
particle/update gathers or extra particle-buffer copies. The numerical integrator
retains its existing stage cloning. Declining adoption is supported for checks,
but requires transition mappings on each later call.

The input system is unchanged on success or failure. No accepted cache lives on
the stepper itself. Discarding a proposal and retrying the same owned input does
not publish a layout. Coverage invalidation still unwinds the complete implicit
solve; application renewal/envelope policy remains separate. A plain driver can
leave the adapter safely: it drops the private cache and rebuilds in its supplied
row order when needed. A small system marker preserves canonical frame export
after an adopted state leaves that cache.

## Export views

`exportSimulationSystem` automatically obtains a UID-ordered view for adapter
states, including independently ordered historical stages after adoption. The
live state, retained stages and cached adjacency are not modified.

For requested adjacency export, `remap_adjacency` copies complete CSR rows and
remaps reference indices. It preserves row-local neighbor order, scaled build-time
supports/positions, edge count, capacity padding and index/offset dtypes. Query
and reference permutations can differ. CSR-only lists remain CSR-only in memory;
hash grids are discarded. Retained ap/av/f follow the same old-edge map. The
legacy HDF5 adjacency writer still serializes explicit live I/J arrays.

```python
from warpSPH.io.export import exportSimulationSystem
exportSimulationSystem(folder, tag, 'CompSPH', accepted.state,
                       stages=accepted.stages, exportAdjacency=True)
```

For particle-only output, set `exportAdjacency=False` and, if exporting stages,
`exportStagesAdjacency=False`. The sorted export path then performs no edge remap;
unkeyed pair arrays are omitted with their adjacency. `canonical_view` exposes
the same owned view explicitly, with separate state/stage adjacency choices.

Unlabelled `extraData` is rejected by automatic sorted export. Use a canonical
view and explicitly transform/declare those extra axes before handing it to the
writer. The existing fixed-roster trajectory writer cannot safely mix its initial
UID roster with later storage orders; it rejects adapter states until its schema
adapter is implemented. Self-contained per-frame HDF5 output is qualified here.
A frame view is not a general history/source/portable continuation checkpoint.

## Scope and current limits

- Supported: audited CompSPH state/update schema; periodic fluid rows; eager
  list-mode explicit one-step RK, including existing compatible energy and
  internal replay. DIRK/coupled FD-JFNK use compatibleEnergy=False and summation
  CRK evolution, under the prior resource contract.
- Stable Morton order uses a fused key kernel, the core 64-bit Morton encoder,
  1–3 dimensions and 1–21 bits per coordinate. Periodic copies and equal-key
  stability are checked independently; coordinates/bounds must be finite.
- No automatic runner/CLI enablement. The caller owns direct-step acceptance.
  No warm history, FSAL/priorStep reuse, warmStart, custom RHS, composed/partial
  stepping or non-tableau drivers are admitted by this adapter.
- External sources, boundary/body inputs, particle-valued tolerances and
  preconditioners need explicit layout adapters and are rejected. Population
  changes, ghost relations and boundary sorting remain later gates.
- Exact-JVP/implicit-AD support preparation and graph/in-place lists retain their
  explicit unsupported errors. Explicit reverse-gradient comparisons here use
  fixed support; broader support/implicit derivative qualification remains open.
- Domain pointer/version, precision/device and Verlet settings bind the cache.
  Domain resources are immutable while a context is retained; replacement or
  ordinary in-place updates require explicitly discarding/rebuilding the cache.
- Membership and eligibility checks still read device flags on a CUDA path.
  GPU control-flow/performance qualification is required before production use.

## Verification and CPU measurements

Final counts and local evidence are indexed in the
[Morton checkpoint](../../PARTICLE_LAYOUT_PLAN.md#fixed-count-morton-checkpoint-2-november-2026).
Coverage includes independent Morton bit interleaving, every particle field joined
by physical ID, rectangular query/reference CSR remapping, optional COO and both
offset widths, empty rows, old anchors/skin usage, retained edge ownership,
canonical frames before/after adoption, compatible replay, support retry, stale
boundary resorting, repeated transitions and rejected proposals. Real backward
Euler/coupled stage equations, explicit gradients and viscosity history are
checked along with default-family regressions and old-integrator compatibility.

The [CPU probe](../../benchmarks/particle_layout/bench_morton.py) uses a shuffled
periodic 1D linear-wave fixture, float32, RK4 and compatible energy. Python
3.13.14, Torch 2.13.0+cu130, Warp 1.17.0, Core Ultra 9 285K; one thread in Torch /
OMP / BLAS, pinned to CPU 0 (reported maximum 5.7 GHz). Five samples of four calls;
mode order is deterministically shuffled within every sample. Each call evaluates
the same small fixture state; these are tiny repeated probes without video.

| Scheme / N | Cold input / Morton ms | Retained shuffled / Morton ms | Adoption ms | Canonical particles / with CSR ms |
|---|---:|---:|---:|---:|
| CompSPH / 32 | 9.027 / 9.552 | 8.069 / 8.478 | 0.233 | 0.046 / 0.135 |
| CompSPH / 4096 | 143.636 / 137.221 | 140.613 / 133.404 | 0.511 | 0.142 / 0.592 |
| CRKSPH / 32 | 11.386 / 11.839 | 10.632 / 10.940 | 0.237 | 0.049 / 0.132 |
| CRKSPH / 4096 | 174.828 / 163.559 | 171.940 / 160.213 | 0.524 | 0.145 / 0.603 |

The opt-in adapter adds about 0.3–0.6 ms on these small 32-particle cases. At 4096,
retained sorted medians are about 5–7% below the shuffled controls. Raw samples
and a native already-sorted control are retained; cache/frequency variation remains.
These measurements do not establish a large-scale gain or the motivating 2× GPU
claim. Canonical timings are state-only view construction, excluding stage views and HDF5 I/O. CUDA validation
was unavailable because the NVIDIA driver could not be reached. No default or
production performance promotion is made.


The subsequent [runner acceptance bridge](RUNNER_2026-11-02.md) exposes the opt-in
policy through CaseSpec/CLI/configuration and permits scalar frame metadata.
The direct-adapter checkpoint above records the earlier qualification boundary.
