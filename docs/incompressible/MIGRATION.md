# Native incompressible runtime — 27 October 2026

The supported methods are `dfsph`, `dfsphPredictor`, `dfsphShift` and `iisph`.
They use the native IISPH spatial operators and static sampled/analytic wall
setup. The three DFSPH compositions share one constraint implementation,
configuration, initialization, weak-factor cutoff and sampling-recovery rule.
This consolidation changes supported entry points, not their numerical defaults.
Pressure-boundary investigation is deferred; the opt-in full-band experiment
remains experimental.

## Select the cycle explicitly

Here “forces” means the nonpressure predictor, including viscosity. “Drift”
means the physical position advance. Density correction has two distinct uses.

| Entry | Cycle | Replaces the old composition family |
|---|---|---|
| `dfsph` | divergence → forces → density velocity correction → kick/drift | `dfsphReference`; classical DFSPH after moving the frame cut |
| `dfsphPredictor` | forces → divergence → density velocity correction → kick/drift | `omniIncompressible` |
| `dfsphShift` | forces → divergence → drift → density position shift → velocity resampling | gravity-free VD+PS branch of `divergenceFree` |
| `iisph` | forces → density projection → semi-implicit Euler kick/drift | native IISPH, retained |

`divergenceFree` used to choose its density correction **by gravity and module
switches**. There is no gravity-dependent switch now: select the intended cycle.
These are composition replacements, not bitwise replacements for the legacy
physics. The prior reference's nonpressure → density → drift → divergence cycle
can be described with the classical frame cut; force evaluation, history and
boundary details still differ.

Use `warpsph-run <case> --scheme dfsphPredictor`, for example. All DFSPH entries
select `ProjectedStep`, even when an ODE RK integrator was requested. IISPH
selects standard semi-implicit Euler. One physical drift occurs per step;
VD+PS adds a separately identified auxiliary displacement on rebuilt geometry.
The existing explicit `schemeConfig.dfsph.ordering` option is retained for saved
native configurations and prepared manifests. The public entries set its default.

## More changed than the iteration driver

The old implementations already defaulted to **relaxed Jacobi**, with optional
Krylov or residual-minimizing relaxation on some paths. The new implementation
uses fixed relaxed Jacobi through the shared stationary driver. It is not a
Jacobi-versus-Krylov controlled comparison.

| Aspect | Current native contract |
|---|---|
| Spatial operator | Rest-volume continuity `C`, density-dependent symmetric pressure acceleration `F`, exact diagonal of `CF`, including selected static contact interpolation |
| Units | Physical pressure fields; iterative unknown `p/rho0`; CD operator scaled by `dt²`, DF by `dt` |
| Histories | Independent `p_CD dt²` and `p_DF dt`, half carry, correct variable-step scaling; no soundspeed reuse or arbitrary physical-pressure ceilings |
| Sources | Explicit density/velocity constraints, with known wall acceleration assigned once; optional `periodicMeanFree` is a declared signed periodic experiment |
| Stopping | Separate stage budgets, minima and percentage thresholds; DF threshold scales as `1/dt`; compression-only clamped residual or signed absolute residual |
| Weak support | Rest-volume factor cutoff `1e-5` before timestep scaling; reference 7/20-neighbour warm-start/statistic exemption retains raw source/update |
| Sampling recovery | Clamped DF unknown pinned to zero below `rho/rho0=0.9`, with neighbour forces retained; signed DF and density solve unaffected |
| Walls | Static self/contact-Shepard closure, analytic integral completion and separately accounted gravity extension; fixed-support restrictions checked explicitly |
| Failures | Nonfinite solves fail; finite budget exits remain reported as budgets, not convergence; runner retains physical step limits and velocity/stall watchdogs |

Retired: legacy actual-volume injection/automatic Akinci scaling, gravity-selected
finalize shifts, soundspeed pressure carriers, source/gauge switches, old
floor-clipped residual tests, operator/diagonal mismatch switches, old compact CG
projection and its closed preset, legacy XSPH/friction/MLS projection options,
and the old Band driver with median-diagonal regularization/automatic gauges.
These switches cannot be copied into native policies as if their units and
operators were unchanged. Analytic per-body pressure/viscous load booking from the legacy loops has not
been ported. Native channel/array diagnostics report flow but omit wall-load
and drag fields; their old load-balance qualifications are retired. Rotating
Couette dynamics remain unsupported by the native static-wall setup. Native
Morris viscosity with analytic no-slip walls requires the explicit
`wallViscosityClosure=noslipMoment` adapter; the generic mirror Laplacian
does not implement Morris viscosity. Channel/array case hooks now defer the
wall mass correction to common initialization, preventing a duplicated factor.
Generic Krylov solvers remain available for shifting
and other users. Shared ACSPH wall-pressure/residual primitives also remain.

The native experimental `dfsph.pressureBoundary='band2018'` option retains its
own exact coupled operator and static sampled-wall restrictions. It is a local
full-band extension, **not** equivalence to the retired `band2018pb` method or a
qualified default. See [pressure-boundary evidence](PRESSURE_BOUNDARIES.md).

## Compatibility and evidence

Old scheme names fail with migration guidance; their numeric enum IDs 0, 1, 3
and 4 are not reused. A legacy-only `solverConfig` export cannot become a native
configuration silently. Earlier **native** exports containing both native policy
blocks can load with a warning that the unused legacy block is ignored. Native
state fields and HDF5 pressure histories retain their names.

Cases and executable Python examples now select native schemes. Exploratory
notebooks retain historical code; use the Python entry points for current runs. New matrix plans
default to the four-method prepared profile. The normal comparison composer
selects that cohort plus dated WCSPH controls. Historical metric readers retain
legacy labels so old reports can still be regenerated. A retired manifest cannot
be executed with new physics under its historical name.

Checkpoint `18d069d` preserves the old runtime, probes and tests. Restore it in
an isolated checkout for historical replay. [RETIRED_PROBES.txt](RETIRED_PROBES.txt)
lists removed one-off legacy experiments. Generated raw data, source snapshots,
videos, numerical stops and published local matrices were not deleted or rerun.

## Qualification boundaries

Existing dense 2D/3D C/F/diagonal tests, pressure-unit and timestep-scaling
checks, signed Fourier-mode projection/momentum tests, finite-failure checks,
weak-factor and density-gate regressions remain. New coverage checks public
routing, retired names/IDs/options, independent configuration/round trips,
common mass initialization, all three nontrivial stage orders, static wall
ownership, pressure-history scaling and physical-versus-shift pressure reporting.
Analytic lattice calibration now covers every native entry.

This is structural and short-step numerical qualification. It does not erase
the recorded long-time IISPH TGV stop, Jacobi conditioning limits, sampled-density
drift, capped tighter DF controls, VD+PS hydrostatic cost, or pressure-boundary
wall-pressure spread. See [the TGV investigation](TGV_2026-10-27.md),
[DFSPH](DFSPH.md), and [sampling recovery](SAMPLING_RECOVERY.md).


## Retirement verification

The focused suite passes 406 FP32 checks with two skips. The FP64 run passes
213 checks; two transport assertions initially constructed FP32 expected
vectors, were corrected to the state dtype, and pass on rerun (215 FP64 checks).
The 131-check adapter subset overlaps those counts. Whole-suite collection
succeeds; the entire unrelated physics suite was not run.

Twelve short GPU FP32 source A/Bs compare checkpoint `18d069d` with current
code: four methods × periodic/particle-wall/analytic-wall setup. All 12 initial
and final tensor sets match bitwise; all 12 current HDF5 continuations match
exactly at changed dt when both paths rebuild adjacency. The three-step timed
windows total 0.628s before / 0.600s after (per-window 4.30–69.45 / 4.35–67.34
ms/step). These tiny, unrendered contracts include cold/cache variation and are
not a performance ranking. The retained startup-scaling benchmark is unchanged.

The stock periodic-flow smoke suite has nine passes and two **strict expected
failures**: raw absolute-density TGV injects kinetic energy and fails decay/
monotonicity. The assertions are retained unchanged. A seeded 20-step replay
reproduces the energy/time series **bitwise** in the pre-retirement native solver
and the new runtime (measured/analytic decay ratio −18.4228; 1.439/1.420s wall,
71.95/70.99ms per step including runner overhead). This is a known raw-case
initialization/source limitation, not a new solver fix or a qualified TGV
baseline. Prepared matrix rows retain their separately declared
`periodicMeanFree` policy; their long-time limitations also remain open.

Raw comparison scripts, logs, failed/interrupted test passes, XML receipts,
HDF5 states and tensors are retained locally under
`.tmp/native-retirement-2026-10-27/`. Tiny contract loops omit video; no long
matrix or new physical qualification campaign was launched. Prior matrix data,
plots, videos, manifests and metrics remain unchanged.
