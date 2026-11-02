# Stage preparation and integrator compatibility — 2 November 2026

This implements the next slice of the [particle layout plan](../../PARTICLE_LAYOUT_PLAN.md).
The generic integrator changes remain in the isolated integrator worktree on
`codex/integrator-resource-lifecycle`. The frontend changes are on `dev`.

## Default compatibility fixes

Structured RHS construction now retains the original linear callback, preserving
identity and standard pickle for previously pickleable RHS objects. Drivers resolve
scoped linear accessors only inside an enabled resource session. Resolve accessors
inside that session; an accessor resolved before enabling resources remains the
original callback. Semilinear synthesis scopes its internal operator calls at
invocation. Explicit combined callbacks remain responsible for their own internals.

The disabled RHS/stage path bypasses the additional scope context manager. No
resource tensor operations, coefficient changes or extra equation evaluations are
introduced when management is disabled.

Real CompSPH implicit inputs also exposed a pre-existing `None - None` error in
integrated-field residual subtraction. Optional inactive integrated fields now
remain `None`, consistent with flattening and norms. Asymmetric activation raises
a field-specific layout error; changing the active vector layout within a solve
is unsupported. Ordinary active-field subtraction is unchanged.

## Simulator preparation seam

`modules/compressibleStage.py` owns the existing order of wall setup, stage-list
selection, support solve and pair-list validation. Both CompSPH and CRKSPH use it.
Without a resource session it runs where the old RHS preparation ran, including
the original support retry and compatible-energy replay behavior.

With `CompressibleSpatialResources`, `CompSPHSystem.preprocess` prepares the stage
before the equation callback. That callback consumes the result from this same
input exactly once. Support is recomputed for every numerical evaluation; it is
not cached by time or stage number. Wall support clamping and physical equations
remain downstream, in their original order. Aux remains `(adjacency, state)`.

For JFNK, residuals, FD probes and line-search trials clone particle scratch and
use independent PairWork, restoring the nonlinear input on scope exit, including
exceptions. A linearization pins its list independently for each stage path.
If support/position coverage requires a replacement list, preparation raises
`ResourceInvalidated` before evaluating physical equations. No GMRES basis is
continued after renewal. A later converged-stage materialization supplies the
aux/state used by the existing finalizer.

```python
import torch
from warpSPH.evaluation import CompressibleSpatialResources
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.solvers import JFNKSolver

# Eager list mode; implicit compatible-energy quadrature is not supported.
schemeConfig.compatibleEnergy = False
resources = CompressibleSpatialResources()
with torch.no_grad(), resources.session():
    result = get_scheme('Backward Euler (implicit)')(
        state.initializeNewState(), dt, stepFunction,
        solver=JFNKSolver(matvec='fd'), config=config, schemeConfig=schemeConfig)
```

On `ResourceInvalidated`, the caller must discard the complete numerical attempt,
renew coverage and retry from owned input. This adapter deliberately does not
advance time, automatically retry, infer acceptance or publish physical effects.
An optional `observe(request, prepared)` callback is diagnostic and reads
speculative evaluations, not accepted steps.

The adapter requires the integrator worktree's optional resource API. Imports are
lazy so ordinary frontend simulations still work against integrators 0.7.0.

## Explicit limits

This first adapter supports eager fixed-layout CompSPH/CRKSPH and FD-JFNK.
Implicit compatible-energy quadrature, non-summation CRK evolution and external
source transactions are rejected. Grid support solves/in-place graph lists are
rejected in the resource session. Exact-JVP preparation and implicit AD/frozen
support derivatives remain unqualified and fail explicitly. Explicit RK retains
its usual compatible-energy shared work and replay route.

Particle sorting, ID/layout epochs, ghost/boundary remapping, accepted adoption
in the frontend runner, canonical export, population changes and graph enablement
remain subsequent gates. The generic integrator's synthetic AD/history/adoption
checks do not establish those application capabilities.

Research support observers/replacement probes and forced list-replay tests now
patch the shared preparation module. The IdealH/dimensional source manifests
include that dependency so old frozen evidence cannot masquerade as this source.

## Measurements

CPU: Intel Core Ultra 9 285K, Python 3.13.14, Torch 2.13.0+cu130, Warp 1.17.0;
CUDA hidden, OMP/OpenBLAS/MKL one thread. These are small CPU fixtures, not a GPU
locality benchmark. No long simulation/video campaign was run.

Integrator baseline: clean `1fea2a3`. Three interleaved process pairs, nine timing
samples per process, RK4 180 calls/sample, JFNK 60 calls/sample, float64. Table
shows the median of process medians, including the full step/solve:

| Fixture | Original, µs | Fixed, µs | Change |
|---|---:|---:|---:|
| RK4, 16 values | 172.86 | 172.24 | −0.36% |
| JFNK, 16 values | 223.34 | 224.10 | +0.34% |
| RK4, 4096 values | 190.43 | 188.46 | −1.04% |
| JFNK, 4096 values | 260.62 | 256.50 | −1.58% |

No resolved default-path slowdown is established at this scale. Timing noise and
uncontrolled CPU frequency remain; this is not a claim of a speed improvement.

Frontend: periodic 32-particle float32 linear wave, RK4, existing compatible energy,
seven samples of 24 steps against a fresh input. Original frontend plus unchanged
integrator release versus updated frontend plus integrator worktree:

| Scheme | Original, ms/step | Fixed, disabled, ms/step | Resource session, ms/step |
|---|---:|---:|---:|
| CompSPH | 8.883 | 8.925 | 9.014 |
| CRKSPH | 11.552 | 11.612 | 11.541 |

The disabled +0.47%/+0.52% differences overlap sample variation. SHA256 hashes of
all positions, velocities, internal energies, supports, densities, UIDs and every
stage's dxdt/dvdt/dudt over three consecutive steps match exactly for both schemes.
These tiny repeated CPU fixtures omit video. They do not measure sorting gains.

## Verification

Final counts and retained local artifacts are indexed in the
[implementation checkpoint](../../PARTICLE_LAYOUT_PLAN.md#simulator-preparation-checkpoint-2-november-2026).
The new real-SPH tests cover preparation before RHS, exactly one support solve per
stage, bitwise explicit RK parity with/without compatible energy and adaptive
support, private speculative scratch, independent fresh-list derivative oracles,
support-growth invalidation/retry, separate coupled-stage coverage, actual FD-JFNK
backward-Euler/Gauss stage equations and converged materialization. CUDA graph
tests are included as existing CPU skip/eligibility checks, not GPU qualification.

The subsequent [fixed-count Morton stepper](MORTON_STEPPER_2026-11-02.md) adds
private whole-step compute layouts, input-order results, explicit accepted
adoption and canonical per-frame export. Its narrower application qualifications
and costs are recorded separately.
