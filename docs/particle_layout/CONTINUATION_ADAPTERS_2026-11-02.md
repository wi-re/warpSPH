# Particle continuation adapters — 2026-11-02

The renewal/UID persistence checkpoint is committed as `ffcd152` on frontend
`dev`. This following slice adds simulator-owned continuation adapters against
the generic lifecycle introduced in integrator `28d8bdd`. Profiling also found
eager prior-stage formatting in quiet reuse. That fix is committed in the isolated
integrator worktree as `cdbbc52` and fast-forwarded into its installed main
checkout. Core remains unchanged at `04e126e`. No remote push or release.

## Contract and API

`MortonStepper` now admits the registered fixed-coefficient AB2–5, ABM2–5,
AM2–4 and BDF1–5 drivers, alongside its existing explicit RK, DIRK and coupled
tableaus. Multistep calls require `compatibleEnergy=False`. The periodic,
fluid-only, fixed-count, eager/native-RHS restrictions still apply. A complete
attempt, including every bootstrap stage, FD probe and renewed retry, keeps one
compute order. The caller's history is an input value and is never advanced by
a failed attempt.

`history=` accepts the library's `StepHistory` with no method-owned `context`.
Each entry's derivative is joined by its auxiliary state's UIDs; BDF state
snapshots are transformed independently and retain their own times. Contexts
used by variable-coefficient/NDF/other drivers are rejected because their
internal axes have not been audited. History capacity, membership, state/update
schemas, snapshot order and edge-field declarations are checked before sorting
or equations. A same-size foreign UID set is an error.

Returned state, errors, stages and history use the current call's input order.
`adopt_spatial_layout(result)` transforms accepted state, error and continuation
history into compute order. Retained result stages keep their evaluated order
and explicit UIDs. It still rejects numerical failure, quadrature and transition
payloads. Lists and pair fields retain their compute owner; a native historical
auxiliary tuple in another basis gets a real CSR/edge remap. Already matching
history uses read-only views without particle gathers or CSR rewriting. A real
layout transition allocates new row values, preserving old observations.

For direct multistep use, thread `accepted.history` into the next call:

```python
stepper = MortonStepper(get_scheme('ab3'))
result = stepper(state, dt, config=config, schemeConfig=scheme_config,
                 history=history)
# Error acceptance, hooks and output consume result first.
accepted = adopt_spatial_layout(result)
state, history = accepted.state, accepted.history
```

The generic integrator runtime can use its existing acceptance hook:

```python
from warpSPH.evaluation import MortonStepper, adopt_spatial_continuation
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.runtime import solve

stepper = MortonStepper(get_scheme('dormandPrince'))
run = solve(stepper.as_scheme(), state, t_end, native_rhs, dt=dt,
            config=config, schemeConfig=scheme_config,
            continuation=adopt_spatial_continuation)
```

`as_scheme()` preserves registered tableau/error/dense-output metadata and
requires the matching native RHS, keyword configuration and
`compatibleEnergy=False`. The runtime sees ordinary numerical state/derivatives;
its accepted `Continuation(state, history, prior_step)` is transformed together
after error control, events, dense samples and callbacks. Callback/result stages
are not rewritten. Numerical history and raw FSAL memory survive the runtime's
**in-memory** checkpoints, including its owned snapshots. This is distinct from
the frontend HDF5 state format.

## Raw derivatives and FSAL

Legacy SPH finalization replaces the last weighted `dudt` with a step-average
diagnostic rate. That is unsafe as a retained Adams derivative; for Adams it can
even overwrite an older weighted history term. The opt-in
`CompSPHSystem.retainStageDerivatives` flag prevents that replacement and follows
stage clones. The multistep adapter and runtime-scheme view set it privately.
The default input-order runner and ordinary direct tableau diagnostics retain
their previous behavior. Endpoint-only implicit finalization is normalized into
the same SPH last-stage carry/switch hooks, with compatible energy rejected when
there are no RK weights. No integration coefficients or SPH equation change.

Explicit FSAL `priorStep=` is an additional opt-in. Both `StageResult` and legacy
`(update, aux)` input forms are accepted; non-FSAL and implicit reuse are rejected.
Actual reuse requires a spatial endpoint stamp for the same method, time, exact
dt and configuration, fixed support, `NoneSwitch`, native thermal integration,
current CRK viscosity gradient, and unchanged endpoint particle/global fields.
Configuration tensors use identity/version/shape/dtype/device; the two admitted
Warp parameter structs are described by all their declared scalar fields.
Unknown mutable configuration disables reuse. Physical edits, changed controls,
dt changes (including terminal truncation by an ulp), support solves, viscosity
switches and compatible energy trigger a fresh first-stage evaluation.

The raw endpoint derivative is captured before finalization in separate metadata,
with its own UID owner. Reuse never mistakes a step-average diagnostic `dudt` for
the endpoint thermal derivative. Historical stage observations stay intact.
Supplying `history=` alone never enables reuse. No automatic FSAL reuse was added
to the frontend runner or CRK/Spheral comparisons.

The integrator's `unpack_prior_step` previously formatted the entire prior stage
before discarding its message when `verbose=False`. For 32 untruncated particle
rows, a CPU profile spent about 16 ms per call under profiler instrumentation in
that unused representation. Formatting now runs only inside the verbose branch;
both StageResult/legacy tuple forms and verbose messages have regression checks.
This removes wasted work without changing numerical values or emitted messages.

## Frontend runner and persistence

Only the Morton route now threads multistep history after accepted adoption;
`ctx.scratch['stepHistory']` exposes its next continuation. An ordinary post-step
hook edit/replacement of particle state or numerical configuration restarts
history conservatively; an observation-only hook does not. Detection uses host
identity/version metadata without new device reads. Unsafe `.data` writes that
bypass tensor versioning are outside that mutation contract. Constant-coefficient
drivers retain their existing dt-change startup policy.

The current HDF5 state checkpoint contains particle state, not integrator history.
The Morton runner rejects multistep `resumeFrom` explicitly rather than silently
claiming uninterrupted history. Multistep history serialization and additional
method-owned contexts remain separate work. One-step UID state restarts remain
supported. Boundaries, ghosts, sources, topology changes, particle-valued
tolerances/preconditioners, graph execution and implicit differentiation still
need their own adapters/qualification.

## Qualification

Evidence is retained under `.tmp/particle-layout-2026-11-02/adapters/`.
Tests use small CPU fixtures; video/viewer registration is omitted for these
short repeated checks. CUDA remains unavailable on this host. Numerical tests
are coordinate/continuation checks, not a new long-run physical-model study.

Coverage includes every admitted multistep order on CompSPH and CRKSPH, arbitrary
UID64 labels independent of geometry, forced boundary cache eviction/re-sorts,
adoption and non-adoption trajectories, raw Adams thermal derivatives, immutable
incoming history, CSR/anchor ownership, unknown axes/foreign membership, FSAL
cost and invalidation, generic adaptive rejection/dense sampling, accepted
callback ordering, in-memory restart, and runner history/reset/state-only resume
guards. Implicit input-order controls use the same resource preparation/probe
isolation and bounded renewal as Morton; legacy unscoped support scratch is not
a coordinate-order control. High-order low-skin implicit starters intentionally allow 16 complete
coverage retries; the production default stays two.

Exploratory failures are retained: ten hard-CUDA configuration tests cannot run
without a driver; an initial implicit FP64 control without probe isolation drifted
in supports by about 1e-7 while physical integrated fields agreed much more
closely. Matching the preparation policy corrects that comparison. Small
fixed-step FP32 implicit fixtures may report a precision-floor termination;
runtime continuation tests explicitly permit stagnation and do not claim a
tolerance-converged solve for those cases.

Final installed-checkout gates:

| Selection | Result | Wall time |
|---|---:|---:|
| Frontend layout/resource/renewal/persistence/runner plus ordinary compressible/support/configuration contracts | 493 passed / 25 skipped; 10 hard-CUDA configuration cases deselected | 92.261 s |
| FP64 continuation/layout/resource/renewal/persistence/runner/ideal-H | 231 passed | 32.352 s |
| Isolated integrator reuse/groundwork/evaluation-resource suite | 759 passed / 415 skipped | 161.839 s |
| Default frontend runner/ownership/contracts/support/case-spec with archived pre-resource integrator `1fea2a3` | 128 passed / 6 skipped | 17.209 s |

The 93 continuation cases include the full admitted-order matrix. All final
frontend gates import the installed editable integrator main without a dependency
override. The backend gate deliberately imports the isolated worktree; its tested
commit is now also main. Frontend source/test AST parsing and `git diff --check`
pass. Earlier unsuccessful controls, retry-budget probes, unavailable-CUDA
failures and pre-optimization timing/profile evidence are retained separately.

## CPU timing

Three alternating baseline/current process pairs pin one CPU with one BLAS/OMP
thread. Each route warms three calls, then times six calls including adoption;
the table uses the median of the three process medians, in ms/step. Baseline
frontend is `ffcd152`; both final variants use installed integrator `cdbbc52`.
Native AB3 controls explicitly retain raw derivatives and thread history. DP
routes use fixed support, NoneSwitch and native thermal integration; RK4 uses
Owen support and compatible energy. Input rows are reproducibly shuffled.

| Fluid | N | Comparison | Before | After |
|---|---:|---|---:|---:|
| CompSPH | 32 | checkpoint/current Morton RK4 | 8.569 | 8.593 |
| CompSPH | 1024 | checkpoint/current Morton RK4 | 39.383 | 39.059 |
| CRKSPH | 32 | checkpoint/current Morton RK4 | 10.927 | 11.046 |
| CRKSPH | 1024 | checkpoint/current Morton RK4 | 49.308 | 47.495 |
| CompSPH | 32 | Morton DP fresh/guarded FSAL | 13.366 | 12.371 |
| CompSPH | 1024 | Morton DP fresh/guarded FSAL | 42.891 | 37.467 |
| CRKSPH | 32 | Morton DP fresh/guarded FSAL | 17.412 | 15.644 |
| CRKSPH | 1024 | Morton DP fresh/guarded FSAL | 53.501 | 46.728 |
| CompSPH | 32 | native/Morton AB3 with history | 1.730 | 2.477 |
| CompSPH | 1024 | native/Morton AB3 with history | 5.988 | 6.685 |
| CRKSPH | 32 | native/Morton AB3 with history | 2.295 | 3.091 |
| CRKSPH | 1024 | native/Morton AB3 with history | 7.805 | 8.492 |

The quiet-reuse fix removes the earlier small-N slowdown: pre-fix CompSPH/CRK
32-particle reuse cost 19.663/22.676 ms. Final eligible FSAL fixtures save one RHS
and about 7–13% of step time. History adaptation still adds roughly 0.7–0.8 ms
to these AB3 fixtures, visibly expensive when the native step is only 1.7–2.3 ms.
This cost must be checked against locality gains on large GPU problems before
default adoption. Lists need no remap on accepted layout adoption; historical
auxiliary lists can require an O(E) remap on an actual coordinate transition.

RK4 checkpoint/current differences span -3.7% to +1.1%; native controls also
vary. Clocks are unlocked and these are short CPU fixtures, so neither a causal
small regression/speedup nor the motivating large-scale 2x gain is established.
Raw timings, measured hot-loop wall totals/counts, import paths, scripts,
source fingerprints and unavailable-driver evidence are retained in the artifact
directory. CUDA and large-problem throughput remain unqualified.
