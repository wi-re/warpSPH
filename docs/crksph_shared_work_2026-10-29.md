# CRKSPH shared force/work implementation — 2026-10-29

Status: recomputation is now the default on qualified native eager single-gas
compatible-energy steps. `crkPairStorage='expanded'` remains a reference
override, with automatic reference fallbacks elsewhere. Short GPU windows show
43%/54% less peak allocated memory at32³/96³, with9.8% slower/4.4% faster steps.
77 distinct CPU controls,26 GPU controls and recorded native Sod/Gresho accuracy
gates pass. The strict32³ auxiliary-divergence mismatch remains recorded.
See the [GPU/default report](crksph_shared_work_gpu_2026-10-29.md).

## Dependency and implementation

The native stage partition estimates half-step velocities as
`vhat_i = v_i + 0.5*dt*update.dvdt_i`. Its estimated pair work uses
`m_i*dt*dot(vhat_j-vhat_i, ap_ij+av_ij)`. The fraction therefore needs the
completed acceleration sum for both particles, rather than just this pair's
contribution. A GPU thread cannot safely consume a neighbour's partly accumulated
acceleration. The two sweeps remain inside one fresh native RHS evaluation:

1. Shared pair physics sums acceleration and raw heating, without storing the
   expanded pressure/viscosity pair vectors.
2. After native forcing/enforceUpdates and switch bookkeeping, the work kernel
   recomputes the same pair physics in registers, computes the native entropy
   fraction, and writes/adds `b_stage*f_ij*(ap_ij+av_ij)` directly into X6 W.

Zero-weight stages omit the second sweep. Multi-weight tableaus accumulate W in
place. A changed-list replay consumes the known final half-step velocity and
reduces energy by CSR row without allocating W. No extra RK stage, stale RHS,
CheapRK2 reuse, force formula, partition formula, source ordering, or timestep
policy is introduced. The case-local OU/bath wrapper still executes after the
native RHS's partition; it is not silently moved into the partition estimate.

Expanded remains the fallback for walls, material tables, experimental CRK
policies, missing compatible-work context, graph/auto mode, derivative-bearing
state and differentiable external-force/dt inputs. These fallbacks preserve existing behavior; the work does
not establish new AD, wall, graph, or multi-material qualification. The separate
RunResult.state obsolete-first-step-W retention issue remains unchanged, so its
possible benefit cannot be attributed to this kernel.

The [CRKSPH paper, section 3.3](https://arxiv.org/html/1605.00725#S3.SS3)
describes retaining or recomputing pair accelerations and specifically discusses
GPU memory versus recomputation. This implementation preserves the local native
entropy/sign estimator. Pinned Spheral `116c71f2ea559d569456ba694194e5a6de3c75f7`
uses completed accelerations for its half-step velocities in
`SpecificThermalEnergyPolicy.cc`; its active node-dudt partition differs from the
local native entropy policy. Matching that different policy is not part of this
storage change.

## CPU evidence

Artifacts: `.tmp/crk-shared-20261029/`.

- Seven controls pass in FP32 and FP64: patterned 2D midpoint, 3D midpoint,
  2D RK4, forced list-identity replay, config/legacy roundtrip, specialized-energy
  expanded fallback, and native checkpoint continuation with changed storage.
  Live W entries are compared; padded, uninitialized entries are excluded.
- Independent frozen-runtime A/Bs use three fresh steps with patterned velocity
  and thermal fields, forcing/bath enabled, on the same three dimension/tableau
  combinations. Both new-expanded and new-recompute match all 17 saved fields
  bitwise in FP32, including the accepted source budget. Maximum budget residual
  across these controls is 8.27e-9. This checks the shared-helper extraction
  against pre-change source rather than comparing only paths sharing new code.
- Existing CRK work/energy/regression/coincident-pair/terms/material/entropy suites: 52 controls pass, 11 unavailable-reference/CUDA controls skip.
- Frozen source audit verifies all 1,178 pins and all 348 current dependency files
  corresponding to the pinned Core/Integrators/Plotting sources, with no mismatch.
- Original compiler failure, FP64 scalar-argument ABI failure and a test fixture
  rejected by the driven case's native-only policy are retained. The scalar ABI
  now uses configured `scalar_t`; the specialized-policy fixture uses Sod without
  relaxing the driven case's physics contract.

The tiny CPU A/B loops omit video. Their wall/ms-per-step records are in each
`*-cpu/summary.json`: total wall times across the three reference runs are
13.64 s frozen, 8.83 s new-expanded and 13.80 s new-recompute. These include
compilation and CPU contention and are not throughput comparisons.

## Short GPU gate completed; broader adoption remains open

The authorized short serial CUDA controls are complete (report linked above):
source-paired evolved 32-cubed checkpoint and 96-cubed startup controls, raw RHS
and accepted-state positivity/source/energy checks, live-W/state comparisons,
then separately instrumented allocator ownership and uninstrumented timing.
Archive exact sources, inspect recordings, and report the memory/runtime tradeoff.
Keep the RunResult retention fix separate. The native eager single-gas default is adopted after the additional accuracy
and fallback gates in the linked report; other contexts retain reference storage.
