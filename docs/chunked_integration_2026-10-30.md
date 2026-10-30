# Paired turbulence/core integration — 2026-10-30

Plan: [whole-row neighbours and multi-GPU execution](../CHUNKED_NEIGHBORS_MULTI_GPU_PLAN.md).

User authorized integration after the ACSPH checkpoint and clarified that row
chunks and multi-GPU support are sequential phases of one effort. Use one paired
implementation workspace: a frontend checkout plus a sibling core checkout,
retained for both phases. This integration implements neither backend.

## Integrated baseline

- Core shared `main`: fast-forwarded `98081c1` to `f6c5900`, including CSR-only
  storage/core0.6.3 and the handoff. Documentation clarification then committed
  as `04e126e`; production core code is unchanged by that follow-up.
- Frontend shared `dev`: merge of ACSPH checkpoint `3d80e5e` and turbulence/memory
  branch `20de462`, in the merge commit containing this report. Frontend `main`
  is unchanged; normal main/dev promotion remains separate.
- Both shared checkouts and feature sources were clean before integration. No
  active simulation worker or unrelated GPU allocation was seen at preflight.
  Existing physics worktrees and ignored campaign artifacts were preserved.

Only `PLANS.md` and `docs/roadmap/driven_compressible_turbulence.md` conflicted.
The former retains the updated turbulence row, separate chunk-plan row, current
ACSPH rows, both decision histories and the user's exact Focus/follow-up text.
The latter retains the ACSPH overnight authorization and newer turbulence results.
No solver source needed conflict resolution or an integration repair.

A blob audit verifies all110 ACSPH-only paths against `3d80e5e`, and all136
feature-only paths against `20de462`, unchanged before the documentation update.
The two shared documents are the complete changed-path overlap. Import checks
resolve the shared frontend/core source directories explicitly and core reports
version0.6.3; no editable installs were changed.

## Focused integration validation

|Suite|Passed|Skipped|Wall seconds|
|---|---:|---:|---:|
|Core CPU|111|101|2.55|
|Core int64-offset CPU|36|22|2.11|
|Core CUDA|12|0|2.59|
|Frontend CPU + ACSPH checkpoint|162|6|54.31|
|Frontend CUDA-enabled + actual graph controls|100|0|117.35|

All selected checks completed with zero failures/errors. Skips are expected CPU
runs of CUDA-only controls. The CUDA-enabled frontend suite includes actual
CRK/CompSPH graph/rebuild controls plus short CSR/work/source/runner fixtures;
some ACSPH fixtures intentionally run on CPU. Known negative capture/fallback
warnings remain in the log; passing these tests does not certify all graph routes.
CPU ACSPH gates cover fixture geometry, fused terms/adjoints, increments,
reference audit/export/shifting/solver, motion, Michel transport and ablations.

These are short automated A/B/unit fixtures, without videos; no new long flow or
physics qualification was run. Existing ACSPH convergence/pressure limitations
and turbulence stationarity/large-resolution limitations are not declared solved.
Fresh int64 tests select the offset dtype before imports; ordinary integration
runs retain the existing int32-offset default. CUDA jobs ran serially; CPU tests
could overlap GPU work. Timings above are test wall times, not solver benchmarks.

Exact pytest invocations were issued against the listed suites with explicit
`PYTHONPATH=/home/lu26029/dev/warpSPH/src:/home/lu26029/dev/warpSPHCore/src`,
`OMP_NUM_THREADS=2`, CPU `CUDA_VISIBLE_DEVICES=''` or CUDA fixtures
`WARPSPH_TEST_DEVICE=cuda:0`. Core-only processes used the core source path.
Int64 CPU additionally set `warpSPHCore_OFFSET_DTYPE=int64` before import.

Retained local evidence: `.tmp/chunked-integration-20261030/` contains five
JUnit XMLs and matching logs, `validation-summary.json`, `merge-source-audit.json`
and `import-provenance.json`. Torch2.13.0+cu130, Warp1.17.0; plotting/integrator
source paths/versions/SHAs are recorded there. The provenance frontend HEAD is
its pre-merge parent because the checks ran before the merge commit; merged
source content and both parents are recorded separately.

## Next step

Prepare one paired implementation workspace from these integrated commits,
verify isolated imports, then start the count/prefix/chunk planner. Native
chunked execution precedes multi-GPU work in the same checkouts. Leave the
existing turbulence checkout available for smaller single-GPU physics through
128^3 after readiness checks; no turbulence campaign is launched here.
