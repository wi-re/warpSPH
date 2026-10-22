# Projection-family reorganization — 21 October 2026

The five projection compositions now separate source assembly, pressure
operators, iteration policy, boundary models, load accounting and temporal
updates. The [inventory](INVENTORY_2026-10-21.md) compares the actual local
formulas with seven primary papers and pinned SPlisHSPlasH/omniSPH source.
All four inspected reference files match their recorded commits. AC remains
deferred. This pass preserves the pressure models; their known correctness
limitations remain visible below.

## Architecture and the integration-contract correction

- `modules/incompressible/sources.py` names five source conventions, including
  different dt/rho0 scalings and the original predicted-density arithmetic
  grouping. Row restrictions, clamps, deficiency rules, gauges and regularizers
  stay with their respective solvers.
- `operators.py`, `reference.py`, `omni.py`, `band.py` and immutable policies
  separate shared geometry/forces from the three distinct pressure iterations.
  The existing Core `runStationary` scheduler is reused. Band boundary pressure
  unknowns, Akinci coupling and Shepard/MLS extrapolation remain distinct.
- Common force assembly explicitly selects body force and viscosity; it
  preserves each scheme's existing term order and omissions. Load/viscosity
  accounting, update BCs and field carry are reusable stages. Legacy XSPH
  increment/acceleration variants live in `modules/xsph/interpolated.py`.
- IISPH has its own scheme module. VD+PS post-integration shifting/resampling
  lives in `modules/shifting/projection.py`. Pose updates use an explicitly
  unforced body stage, preserving the historical coupling.
- `warpSPHIntegrators.schemes.ProjectedStep` evaluates each full projected map
  once through initialize/preprocess/postprocess/finalize. Kick/drift and
  pre-drift shift recurrences live in its mechanics driver. `ProjectedSystem`
  carries the advanced state; `DFSPHReferenceSystem` remains a compatibility
  alias. Ordinary VD+PS continues to use the user's selected RHS integrator.
- The new optional `SchemeBundle.integrationMethod` preserves legacy
  seven-member unpacking. The runner reports the actual package map driver.
  Direct legacy multistage calls fail clearly. Existing single-evaluation
  Euler calls remain supported.

**Adopted correctness correction:** the reference/omni/Band callbacks already
advanced a complete dt state. Running them repeatedly under an ODE RK wrapper
could repeat, blend or half-step that map. The package driver now executes the
intended full map once, regardless of the requested legacy ODE enum. Its
independent constant-acceleration/lifecycle/gradient controls pass, and all
single-evaluation old-source comparisons below are bitwise. A previous run
using a multistage wrapper is therefore not an equivalence baseline.

Configuration preservation also improved: all shared shifting fields, both
pressure-check schedules, and the existing region/body codecs now round-trip.
Old missing-key defaults are retained. Those established region/body codecs
still do not constitute a complete serialization of runtime boundary providers
or every case-added analytic/projection option; that schema debt is documented
in the inventory.

## Verification

Python 3.13.14, Torch 2.13.0+cu130, Warp 1.17, RTX PRO 6000 Blackwell.
All GPU jobs ran sequentially. Tiny state/counting/timing probes omitted video;
physical controls used VisPy, video, progress and the local daemon.

| Check | Result |
|---|---|
| Five schemes × periodic TGV, hydrostatic column, dam break, impact, bounded random flow; FP32 and FP64 | 50/50 state pairs bitwise, strict tensor-field sets matched |
| Analytic hydrostatic/MLS walls, damped carry, legacy XSPH enabled, body force, VD+PS route, changed legacy policies, particle MLS, experimental Krylov and unregularized Band | 19 non-diverged FP32 pairs bitwise |
| Compact projection + fixed shift on the calibrated periodic channel with the supported no-slip moment viscosity closure | 2/2 pairs bitwise, positive/finite over four steps |
| 65,536-particle TGV, 70 steps, three trials per source and scheme | 15/15 state pairs bitwise |
| Five recorded column old/new pairs | 5/5 state tensors bitwise, including the reproduced DFSPH failure; stopping time and status match |
| Final FP32 frontend/configuration group | 203 passed, 1 skipped, 1 verified baseline failure |
| FP64 frontend/operator group after precision-correct wall fixtures | 47 passed, 1 skipped, 4 verified baseline failures |
| Full integrators CPU gate, four workers, numerical threads=1 | 8,797 passed, 639 skipped, 601.9 s |
| Daemon terminal state | 10/10 runs have matching final status, step and time |

Thus **91 source-state pairs are bitwise identical**. This is a structural
preservation result, not a claim that all these numerical models are sound.
The matrix and large-run comparisons use the supported historical
`semiImplicitEuler` map as the before baseline.

The unchanged FP32 failure is analytic-wall IISPH's 150-step still tank:
`maxVelocity=3.037971019744873`, exceeding its 3.0 limit. The saved source fails
at exactly the same value; its threshold was retained.

An initial FP64 test invocation exposed seven wall-fixture dtype errors
(`vec2f` tensors passed to `vec2d` kernels). The original saved tests reproduce
all seven. Fixtures now follow the selected runtime precision; all eight wall
oracles pass in FP32 and FP64. Four other FP64 failures reproduce against the
saved solver source with identical numerical values: BiCGStab residual
0.01350645 against .01, MINRES residual 0.01015688 against .002, cross-Krylov
agreement, and the explicitly FP32 Jacobi fingerprint. Native FP64 random
velocity input differs from FP32; those budget/fingerprint expectations are
not a universal FP64 qualification. Their thresholds and solver behavior were
preserved. The original failed logs and JUnit reports remain archived.

Two additional compact/fixed-shift TGV probes diverge on both sources at the
first step. They are retained as negative evidence and excluded from the 91
bitwise comparisons. The mirror/Morris channel combination also refuses the
same way on both sources; the supported `noslipMoment` channel above is the
valid compact-path comparison. No boundary guard was expanded.

## Physical controls and limitations

Each recorded column requested 150 steps at nx=32 (1,992 total particles),
using the same case/timestep logic before and after. Actual elapsed physical
time is reported, rather than assuming a fixed dt from the initial setting.
All per-particle masses and total mass match exactly before/after.

| Scheme | Steps / final time | Fluid density range | Final max speed | Old/new state |
|---|---|---|---|---|
| VD+PS | 150 / 1.491 | 0.96630–1.00233 | 0.37572 | bitwise |
| DFSPH reference | 37 / 0.270834 | 0.14617–1.72383 | overflowed speed norm; diverged | bitwise, same failure |
| IISPH | 150 / 0.635809 | 0.20613–1.05415 | 2.82644 | bitwise; degraded surface |
| Omni | 150 / 1.491 | 0.94850–1.00724 | 0.82655 | bitwise |
| Band | 150 / 1.491 | 0.58620–1.00413 | 0.42016 | bitwise |

Final frames were inspected: the reference's explosive particle disorder and
IISPH's surface spray remain; they were not hidden by scalar finiteness or a
successful process exit. Finite component tensors can still produce an
overflowed speed norm. These are preservation controls, not hydrostatic-
accuracy or convergence studies. Wall/body energy and total-energy
conservation are not inferred from mass preservation.

The normalization audit makes another limitation quantitative. Scaling both
mass and density by 1,000 preserves the reference's volume-only factor
(ratio 1.00000007) while its physical-pressure operator diagonal scales by
0.00100000005. The wrapper's old fluid-only back-reaction description also
contradicted the actual kernel, which includes static-boundary neighbor
squares. Both differences are now documented explicitly. Rest/apparent-volume
choice, pressure normalization, boundary sources and load evaluation time
need separate numerical adoption experiments. No reference-fidelity claim is
made for the current local defaults.

## General performance

65,536-particle 2D TGV, Wendland C2, n_h=4, rho0=1, FP32, eager. Ten warmup
steps plus 60 measured steps; three alternating old/new trials. The case's
32-step initialization relaxation precedes timing. Values are median reported
integrator ms/step; raw trial distributions and whole-loop costs are retained.

| Scheme | Before | After | Median change |
|---|---:|---:|---:|
| VD+PS | 11.694 | 11.470 | −1.91% |
| DFSPH reference | 44.325 | 44.814 | +1.10% |
| IISPH | 27.550 | 27.352 | −0.72% |
| Omni | 7.571 | 7.585 | +0.18% |
| Band | 10.302 | 10.373 | +0.69% |

Performance is mostly neutral; the small slower medians are retained, within
the overlapping measured distributions. This is not a broad speedup claim.
A six-step nx=64 call audit (two warmups excluded) verifies the bounded general
optimizations: VD+PS loses an unused velocity-reduction diagnostic, reducing
819→811 Torch ops and 21→19 host syncs per step, with 36 Warp launches
unchanged. Reference CD skips an unused neighbor-count/deficiency calculation:
5,542→5,535 Torch ops, 221→220 Warp launches, host syncs unchanged at 21.
Pressure iteration recurrences and stopping policies are preserved.

Whole-step CUDA graphs remain unavailable for this family as before; these
host-checked pressure loops were not relabeled as capture-safe. Register,
cache-load policy and deep loop fusion remain deferred behind the numerical
questions identified in the inventory.

## Evidence and reproduction

[Machine evidence](evidence/qualification_2026-10-21.json) contains strict field/
byte comparisons, every timing trial, tests, source pins, control diagnostics,
operator scaling and daemon verification. Raw states, logs, videos, JUnit,
source snapshots and exact measured source variants live outside Git at:

`/home/lu26029/dev/warpSPHDevelopment/core/experiments/incompressible-cleanup-2026-10-21`

The archive README gives source-prefix mappings and commands. Reusable probes
are `scripts/bench_projection.py` and `scripts/audit_projectionOperators.py`.
The latter evaluates operators without advancing a simulation. The former
supports short state comparisons, timed windows and `--video --daemon`
controls. Source manifests identify the loaded frontend, Core and integrator
files for every timed/recorded run. Three intermediate source variants are
preserved and hash-verified; existing dirty work was preserved and no commit
or publication was made.
