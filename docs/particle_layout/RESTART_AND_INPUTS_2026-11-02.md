# Particle continuation, solver inputs and GPU qualification — 2026-11-02

Requested order: commit the existing adapters, persist numerical continuation,
extend the input/boundary adapters, then qualify on CUDA. The first checkpoint
is `24d7a92` on frontend `dev`. Integrator main and its worktree are both at
`cdbbc52`; core remains `04e126e`. No push or release is involved.

## Restart contract

Morton state exports now carry a versioned `integration` group. It records the
method, a portable numerical-configuration digest, last/next timesteps, complete
UID-ordered derivative history, BDF state snapshots and admitted method-owned
memory. An eligible FSAL prior stores the raw endpoint derivative rather than
the finalizer's step-average diagnostic. No search list, edge cache, process
pointer or executable callback is stored in this group. Existing configuration
serialization is separate and unchanged.

Writes use a sibling temporary file and atomic replacement. A failed write
preserves the preceding checkpoint. Reads validate the method, configuration,
frame time, roster, precision, schemas, history capacity, snapshots and context
axes before returning memory. Search resources are rebuilt from the restored
physical state. Nonzero-time multistep state-only restarts are rejected rather
than silently repeating a bootstrap. The input-order runner keeps its existing
state-only format and behavior.

Direct callers can pass `continuation=Continuation(state, history, prior_step)`,
`integrator=method`, `config=config`, `schemeConfig=scheme_config`, `last_dt=dt`
and `next_dt=next_dt` to `exportSimulationSystem`. Restore the physical frame
with `restoreCheckpoint`, then use `readContinuation` on its integration group.
The Morton runner does this automatically. A physical post-step hook restarts
history; observations do not.

## Additional numerical inputs

Alongside the existing fixed-step AB/ABM/AM/BDF and RK tableaus, `MortonStepper`
admits the library's `VariableAdams`, `AdaptiveAdams`, `NDF` and `AdaptiveNDF`
objects. Their contexts declare integrated field blocks explicitly; UID joins
transform flattened current vectors, NDF difference rows and Adams order
defects. Leading difference rows and coupled stage blocks are independent axes.
The method-owned context carries `particle_uids` at the simulator boundary.
It is portable through the same numerical checkpoint format.

`as_scheme()` also handles these non-tableau method objects for generic
`runtime.solve`. Acceptance transforms numerical continuation while retaining
callback/result stage observations in their evaluated order. NDF quadrature's
known state is an explicitly UID-bearing observation. NDF remains restricted to
fluid-only evolution without boundary projections and with compatible energy
disabled; those restrictions preserve its existing finalize requirements.

DIRK methods with an implicit first stage accept one UID-bearing `warmStart`
stage; coupled methods accept one per stage. Seeds are remapped into the entire
attempt's compute order, including after a coverage retry. They change guesses,
not equations. Raw arbitrary preconditioners remain rejected. Wrap a pure
preconditioner as `UIDPreconditioner(ids, action)` to declare its input basis.
The action receives its vectors and physical system states in that order;
coupled systems preserve stage-major blocks. The live evaluation/resource token
is separately exposed as `compute_evaluation_context`, retaining its real owner.
Unlabelled tensor-valued preconditioner contexts and particle-valued tolerances
remain unadmitted.

## Boundaries, forcing and ghosts

Morton keys now wrap periodic axes and clamp nonperiodic axes. Stable ties keep
incoming order. CompSPH/CRKSPH admit solid wall rows as well as fluid rows. Fixed
wall volumes travel in the explicit particle field `wallVolumes`, preserving
their initial physical meaning through sorting, list rebuilds and checkpoint
restart. A native cache also records its UID owner so entering the spatial path
after an independent row change joins the correct volumes. A bare legacy cache
without that owner requires explicit `state.wallVolumes`. The ordinary path
still uses its previous config cache when that optional state field is absent.

The native boundary dispatcher accepts explicit pure adapters:

```python
from warpSPH.evaluation import UIDValues, AllParticles, PlaneRegion

force = UIDValues(state.UIDs, particle_forces)
bc = BoundaryCondition(BoundaryConditionType.forcing, AllParticles(),
                       forcingFunctions=[force])
scheme_config.boundaryConditions = [bc]
```

`UIDValues` owns constant per-particle values; native force units are preserved.
`AllParticles` and `PlaneRegion` declare pointwise geometry. Velocity/internal
energy Dirichlet values and declared update fields are also supported. Geometry,
rosters, target shapes and precision are checked before sorting. These adapters
do not infer a row axis from tensor length. Boundary projections and forcing
disable FSAL reuse. Opaque closures, time-advancing sources and `sourceTerms`
transactions still require separate adapters. The global two-entry work/heat
vector `sourceEnergyRates` is now explicitly global, including when N equals two.

`permute_boundary_state` declares complete row/global/index schemas for WCSPH,
incompressible, DFSPH and ACSPH particle states. It gathers row values and
conjugates the reciprocal ghost/boundary map, preserving -1 sentinels. Body-local
geometry retains its compact order; global scatter rows are rebuilt by particle
and ghost UIDs. `reorder_boundary_system` returns a matching next system,
private config with cleared row caches, and adapted body config for the native
one-step WCSPH route. The caller commits all three after acceptance. It discards
search/diffusion caches and retains explicitly permuted surface-mask history.
Opaque boundary callbacks and analytic providers are rejected. It does not adapt another family's system history, a CUDA
graph or a retained multistep history.

Ghost construction has an explicit `persistent_ids=True` option. It allocates
fresh nonnegative IDs from the counter and advances it, avoiding the legacy
`-boundaryUID` collision at zero. The default constructor retains its legacy
identity behavior. Existing negative-ID ghost states must be resampled with
that option before using the new UID layout utilities. Automatic runner Morton
stepping remains limited to CompSPH/CRKSPH; WCSPH ghost/body adaptation is an
explicit accepted-boundary utility. It does not enable UID trajectory/checkpoint
export for those other state families; callers must handle their observation
order explicitly before adopting it in a trajectory-producing application.

## Qualification and limits

Evidence and frozen sources are under
`.tmp/particle-layout-2026-11-02/restart-inputs/`. Small repeated numerical and
one-step timing fixtures deliberately omit video/viewer registration. CUDA runs
are sequential in `/home/lu26029/miniconda3/envs/warp/bin/python`, outside the
sandbox. Hardware is an RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB,
driver 610.43.02. The available conda environment is named `warp`.

| Gate | Result | Wall time |
|---|---:|---:|
| Broad CPU layout, continuation, runner, configuration, support and ordinary solver contracts | 547 passed / 46 skipped; 10 hard-CUDA cases deselected | 100.397 s |
| FP64 layout, resources, continuation, runner and all new input/schema cases | 283 passed | 38.302 s |
| Final CUDA method/implicit/restart/boundary/ghost matrix | 52 passed | 17.569 s |
| CUDA FP64 method/implicit/restart/boundary matrix | 44 passed | 289.870 s |
| Configuration and initial GPU matrix | 139 passed; four NDF control-fixture failures subsequently corrected | 355.755 s |
| Final native-wall cache handoff and affected CPU contracts/layout/restart | 143 passed | 14.349 s |
| Final boundary and native-wall cache handoff on CUDA | 35 passed | 6.553 s |
| Final boundary/restart and native-wall cache handoff in FP64 | 43 passed | 6.447 s |

The configuration/initial-GPU row includes all ten cases that require CUDA and cannot
run inside the sandbox. Its four failures were in the independent NDF control:
`initializeNewState` dropped that test's dynamic `ndf_compatible` flag before the
native solver call. The simulator adapter already sets it privately. Corrected
controls pass on both GPU precisions. First-use compilation is included in test
wall times; the subsequent GPU FP32 gate ran with compiled kernels cached.
Earlier logs, including these failures and the expected sandbox CUDA failures,
are retained. Implicit checks establish coordinate/continuation equivalence;
they do not claim tight-tolerance convergence for FP32 precision-floor exits.

The timing matrix uses 2D periodic smooth states, RK4, fixed supports,
`NoneSwitch`, `compatibleEnergy=False`, Wendland2 and `KernelMeanSymmetric`.
Each timing is the median of ten sample averages, eight synchronized calls per
sample, across two processes; route order is shuffled within samples. Current
and archived frontend `24d7a92` native controls alternate in process order. All
use installed integrator main `cdbbc52` and core `04e126e`. The repeated one-step
fixtures isolate locality and adapter cost; they are not evolving long runs.

| Fluid | Particles | Native input order (ms/step) | Morton with retained layout (ms/step) | Speedup |
|---|---:|---:|---:|---:|
| CompSPH | 1,024 | 8.222 | 8.976 | 0.916× |
| CompSPH | 16,384 | 8.503 | 9.115 | 0.933× |
| CompSPH | 65,536 | 10.176 | 9.788 | 1.040× |
| CompSPH | 262,144 | 21.363 | 17.541 | 1.218× |
| CompSPH | 1,048,576 | 79.435 | 58.237 | 1.364× |
| CRKSPH | 1,024 | 10.515 | 11.146 | 0.943× |
| CRKSPH | 16,384 | 10.931 | 11.440 | 0.955× |
| CRKSPH | 65,536 | 13.344 | 13.617 | 0.980× |
| CRKSPH | 262,144 | 31.886 | 26.722 | 1.193× |
| CRKSPH | 1,048,576 | 125.321 | 94.668 | 1.324× |

At one million particles, a native call on already sorted data takes 57.562 ms
(CompSPH) / 93.885 ms (CRKSPH); the retained-layout adapter adds 0.676 / 0.784 ms.
Cold Morton steps, including sorting and input-order result mapping, take
64.813 / 101.253 ms against cold native 84.268 / 129.865 ms. Initial accepted
adoption costs 1.359 / 1.350 ms; a canonical particle-only view costs 0.261 /
0.256 ms. These last two are per operation, not additional per-step costs once
the layout is adopted. HDF5 transfer/write time is excluded.

The initial unpinned sweep showed small CompSPH native controls up to 7.1%
slower, with sizable between-process variation. Three subsequent paired native
controls pinned to CPU 0 (5.7 GHz maximum; GPU runs still sequential) resolve
that check: median paired changes span -0.83% to +0.20% across both fluids and
the three small sizes; individual pairs span -2.71% to +3.11%. Pooled sample
medians span -0.93% to +1.52%. There is no consistent default-route regression
outside that observed variation. Morton still has the explicit small-case
overhead shown above, so no automatic selection threshold or default promotion
is made.

Every benchmark row compares UID-joined sorted/native physical fields before
timing. Across the sweep, maximum absolute differences are below 9.90e-6,
including density; at one million particles velocity differences are below
4.12e-7, internal energy below 2.39e-7, and support is unchanged. The native
one-step relative energy change is at most 4.24e-9 and momentum-change norm at
most 3.67e-7. These are short-step checks, not shock/long-time validation. The
earlier SuperSymmetric timing setup is retained separately under
`super-symmetric-diagnostic/`; it is not the CRK qualification baseline.

Raw per-sample timings, per-row wall times, native controls, progress logs,
XML, source manifests and both successful/failed probes are retained with the
evidence. `qualified-sources`/`manifest.json` retain the timed source;
`final-sources`/`manifest-final.json` retain the final checked source, including
the final checkpoint dtype guards and native-wall UID-owner handoff. Those
follow-ups do not affect the fluid-only timed execution path. The archived
baseline source is also retained. No integration coefficients, SPH equations or default execution
selection were changed. Changing particle counts, emitters/outlets, arbitrary
source transactions, additional simulator families' histories/export adapters,
graph execution and exact implicit JVP remain subsequent work. Large-scale
history/preconditioner costs have not been benchmarked.

## Next phase

Implement accepted-boundary population events for the fluid CompSPH/CRKSPH
route first. A step attempt keeps a fixed dimension and roster, including every
JFNK probe and rejected attempt. After acceptance, one event transaction creates
and removes all declared fields, allocates fresh UIDs, updates relations and
invalidates search resources. Initially restart multistep/FSAL memory after a
population change. Introduce variable-roster output frames with explicit IDs
and event provenance before enabling emitters/outlets in the runner. Preserve
the existing fixed-roster format and input-order default.
