# Coverage renewal and UID persistence — 2 November 2026

The opt-in eager CompSPH/CRK runner now supports bounded complete-step coverage
renewal, fixed-roster UID trajectories and physical state checkpoint restart.
Integrator `main` remains at the merged `28d8bdd`; this slice changes the simulator
on `dev`. The default remains `particleOrder='input'`.

## Coverage and the nonlinear solver

`MortonStepper` selects one compute order before the attempt loop. For implicit
steps, each attempt gets fresh copies of incoming physical fields and fresh
integrator/solver/stage scopes. The previous Newton iterate, residual, GMRES basis,
line-search state and pair work are never retry inputs.

An optional list builder is threaded through stage preparation and the existing
Owen, Monaghan, NumberDensity and NoScheme support dispatch. The native callback
is retained when this option is absent. The support equations, convergence tests
and physical supports are unchanged. With resources enabled, incoming geometry
is checked before support iteration, and the support solver's existing peak
checks use the same candidate builder. A list admitted for an evaluation remains
fixed; insufficient coverage raises before equation evaluation/materialization.

A proposal owns its stage path, UIDs, anchor positions, previous radii, required
supports, motion budget and evaluation purpose. It retains search evidence rather
than the failed numerical state. Each coupled stage has its own envelope. For
anchor positions a and current positions x, let delta be the largest particle
motion under the domain's minimum-image metric. Candidate radii satisfying

    R_i >= h_i + 2 delta

contain every required super-symmetric pair by the triangle inequality. Existing
radii preserve all previously covered evaluations. Renewal builds from the same
anchors with `1.25 * max(previous_radii, required_supports + 2 delta)`; 25% is
search headroom, not a physical support multiplier. Physical h is recomputed for
each residual/probe/trial. Candidate radius data is held separately from h.

Only `CoverageInvalidated` carrying such a proposal triggers automatic renewal.
An unrelated `ResourceInvalidated` still propagates. Invalid/non-finite probe
geometry cannot generate an envelope. Exhaustion raises `CoverageRetryExhausted`
with the last proposal and number of complete attempts. Failed attempts publish
no input cache/state and perform no runner hooks or output.

The default budget is two renewals, configured by `CaseSpec.coverageRetries`,
`--coverageRetries`, versioned `simulation.coverageRetries`, or direct
`MortonStepper(max_coverage_retries=...)`. Zero performs one attempt and reports
coverage loss. The accepted spatial context records `coverage_renewals`; runner
rows expose `coverageRenewals`. Adoption preserves that count and the list/anchors.
Retry time is inside the integrator-step timer. Layout adoption is separately
measured and also included in `stepTime_ms`.

This is complete-step restart, including earlier stages, rather than restarting
only the affected linearization. Differentiable implicit support/exact JVP,
preconditioners, custom RHS and persistent solver memory remain excluded.

## Trajectories and state files

`storeMode='trajectory'` now accepts the Morton option. Its version-1 UID schema:

- Stores an ascending, fixed `combinedUIDs` roster in int64, plus full-width UID
  subsets, provenance IDs and the original UID counter. Float data keeps the
  configured precision. The legacy input-order writer retains its prior format.
- Gives each frame a hard link from `frameUIDs/frame_...` to the roster, without
  duplicating that array. Membership/count changes are rejected before frame data
  is created; topology events need a later schema.
- Labels frame field groups as `particle` or `global`. Declared particle columns
  follow UID order; global vectors stay global even when their length equals N.
  Extra field axes/order cannot change after the header. Changes to static
  masses/materials/kinds/provenance require an explicitly declared frame column.
- Canonicalizes every requested stage independently, including updates, state,
  CSR rows/reference indices, anchors and edge values. Observation readers retain
  UID64, declared field axes and numeric stage order beyond stage 9.

Scalar extras are allowed; unlabelled arrays/nested extras and attempts to replace
layout metadata are rejected. Full state files also record the layout version,
scheme, domain, precision and UID counter. The general frame reader understands
these metadata arrays instead of interpreting them as user scalar sentinels.

Trajectory columns remain selected observations. They are not complete solver
continuation checkpoints; use `storeMode='states'` checkpoints for restart.

## Checkpoint continuation

`resumeFrom` in Morton mode accepts version-1 UID-layout state files. Before output
creation it checks the format, scheme, domain, precision, unique roster, sampled
static particle fields, positive supports, finite particle data and UID counter.
It restores all saved physical/history fields and time, then clears adjacency,
layout, grids, pair work and rerun state. The first resumed step sorts and builds
fresh resources. No pointer/version fingerprints or Krylov memory are restored.

The normal case configuration must still describe the same physical problem;
these checks do not compare every numerical parameter. Case-owned metadata still
uses `restoreCheckpoint`, with UID-aware hooks. External source/boundary inputs
remain unsupported. Old unmarked state files and trajectory files are not admitted
as Morton continuation checkpoints by this adapter. Legacy input-order resume
behavior is unchanged.

## Qualification and cost

Final gates use the active editable integrator main checkout without a worktree
PYTHONPATH override, except the explicit old-release control:

- **403 passed, 25 skipped**, 63.713s: renewed JFNK, layout/resources/runner/IO,
  support modes and shared/compatible work, graph eligibility, viscosity history,
  ordinary runner/ownership, number-density support and configuration/CLI controls.
  Ten existing tests explicitly requiring CUDA were deselected because the driver
  is unavailable; their initialization failures were retained in the prior gate.
- **138 passed**, 13.704s, FP64: all renewal, IO, layout, resource, runner and IdealH
  cases, including all four support modes under renewed backward-Euler/Gauss solves.
- **123 passed, 24 skipped**, 13.419s: ordinary runner/support/contracts using
  archived pre-resource integrator source `1fea2a3`, leaving main unchanged.
- There are **31 coverage cases** and **17 persistence cases**. Independent checks
  include periodic all-pairs containment in 1–3D; a transient support peak followed
  by shrinkage; fresh nonlinear tokens and fixed order across real FD failures;
  budget exhaustion/isolation; Butcher stage equations and fresh physics controls;
  UID64/provenance/global E=N axes and pre/post-adoption stages; forced boundary
  resorts; restarted viscosity history and renewed implicit stepping; and rejection
  of mismatched checkpoints before output.

An exploratory expanded FP64 selection included `test_numberDensitySupport.py`,
which hardcodes float32 tensors/configuration: 33 kernel dtype-mismatch failures
are retained. That file passes in the final FP32 gate. The new NumberDensity
renewal cases use configured precision and pass in both gates above. No numerical
fix or tolerance change was made to hide those fixture incompatibilities.

The CPU A/B below compares this source with a frozen prior runner implementation,
using three alternating process pairs pinned to CPU 0 (reported max 5.7 GHz), one
Torch/BLAS thread, seed-17 shuffled periodic 1D data, and medians of four hot calls
after one warm call. Direct-step timings include adoption. Native/Morton RK4 use
Owen support and compatible energy; implicit Morton uses backward Euler, fixed
support, skin 1.5, and timing-only JFNK tolerance 1e-6. These runs need no renewals.

A first tight-tolerance (1e-10) FP32 timing probe failed the old CRK N=1024 solver
status check; its log is retained. The final comparison uses the same 1e-6 timing
tolerance for both implementations. Accuracy/renewal gates above use independent
controls and the tighter small-fixture solver settings.

| Scheme | N | Route | Prior ms/step | Current ms/step | Change |
|---|---:|---|---:|---:|---:|
| CompSPH | 32 | native | 8.053 | 8.018 | -0.43% |
| CompSPH | 32 | morton | 8.525 | 8.561 | +0.42% |
| CompSPH | 32 | implicit | 10.838 | 10.754 | -0.78% |
| CompSPH | 1024 | native | 40.234 | 41.072 | +2.08% |
| CompSPH | 1024 | morton | 39.167 | 40.585 | +3.62% |
| CompSPH | 1024 | implicit | 79.623 | 78.028 | -2.00% |
| CRKSPH | 32 | native | 11.027 | 10.470 | -5.05% |
| CRKSPH | 32 | morton | 11.465 | 11.266 | -1.73% |
| CRKSPH | 32 | implicit | 14.167 | 14.458 | +2.05% |
| CRKSPH | 1024 | native | 50.429 | 49.450 | -1.94% |
| CRKSPH | 1024 | morton | 48.818 | 47.970 | -1.74% |
| CRKSPH | 1024 | implicit | 71.800 | 72.451 | +0.91% |

Native controls also vary by roughly -5% to +2%; the small timing differences do
not isolate a policy regression or establish a performance benefit. This does not
qualify CUDA or large-scale locality. I/O is not included in this timing probe.
Video/viewer recording was omitted for these tiny repeated CPU fixtures; progress,
whole test-gate durations and per-sample timing data are retained.

Local evidence: `.tmp/particle-layout-2026-11-02/next-phase/`, including final and
exploratory logs/JUnit, paired timing scripts/JSON, frozen baseline sources,
source fingerprints, snapshots and diff. CUDA was rechecked: `nvidia-smi` still
cannot communicate with the driver. No remote push or package publication.

Next: adapt history/prior/FSAL continuation coherently before enabling reuse or
multistep methods, then qualify CUDA eager lifetimes and host-control overhead.
Boundaries/ghost relations, external sources, population events and graph capture
remain separate gates.

Source/test/benchmark Python fingerprint: `8d7295222501613304cb9735053da48234965464fe2b64a048d76e3bc48510f5` (761 files).
The current simulator work remains uncommitted; integrator main and core are clean.
