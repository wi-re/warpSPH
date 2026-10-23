# warpSPH — pressure-based SPH (PESPH) plan

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

> **Part of [COMPRESSIBLE_SPH_PLAN.md](COMPRESSIBLE_SPH_PLAN.md) (2026-10-11), Track D.** That plan consolidated the AV, CRKSPH-limiter,
> Godunov and MFM/MFV plans (the first three are in `docs/historic_plans/`); this file stays the PESPH design and keeps its own
> row in PLANS.md. §7 (the MFM outlook) is now Track E there; the text below stays as the rationale.

Earlier follow-up: [support, cold-shock and cost report](.tmp/pesph/docs/validation/pesph_followup_2026-10-22/index.html), with shared-scale figures, movies, metrics, configurations and provenance. The original 57-row comparison remains separately preserved.

Earlier: [source, initialization and efficiency report](.tmp/pesph/docs/validation/pesph_efficiency_2026-10-22/index.html). Guarded defaults remain; measured improvements and remaining accuracy/cost gates are recorded at the end of this plan.

Earlier: [adiabatic Hamiltonian/timestepper qualification](.tmp/pesph/docs/validation/pesph_hamiltonian_2026-10-22/index.html). A passes the tested force/potential, canonical-map and temporal-order checks; automatic AD/split lifecycle admission remains open.

Latest: [numerical closeout and limitations](.tmp/pesph/docs/validation/pesph_numerical_closeout_2026-10-22/index.html). Support closure and hydrostatic stage-budget fixes validated. The historical stop was superseded by the user's 2026-10-23 authorization to merge into dev.

## Integration into dev — 2026-10-23

The user explicitly authorizes merging the completed PESPH worktree into `dev`,
superseding the historical stop-unmerged instruction below. The integrated scope
retains both working variants, guarded defaults and opt-in experimental policies;
this integration does not change numerical formulas or promote those experiments.
Shared runner changes retain the newer incompressible initialization and first-step
physical timestep cap. The pre-existing primary-checkout scope audit is retained
in the original-audit sections, with its stale progress claims superseded.

[Maintained usage and limitations](docs/pesph.md) describe the shipped contract.
Generated reports, movies and exact source snapshots are preserved locally under
`.tmp/pesph/docs/validation/`; `.tmp/pesph/index.json` locates the original drivers
and raw runs in the retained worktree. A SHA-256 manifest covers the copied report
files. Investigation drivers and generated products are excluded from Git under
current CONTRIBUTING.md policy. Original worktree evidence remains intact.
The reusable evaluation/probe commands and lasting regression checks are included.

Integration verification on the combined primary `dev` tree: CPU float32 selection
307 passed / 12 skipped, with ten CUDA-only configuration cases deselected;
CPU float64 selection 200 passed / 1 skipped. All 22 subprocess kernel gradient-check scripts also pass. The selections overlap. Float32
covers runner/schema/catalogue, shared support/dissipation/reconstruction, PESPH,
projection modules, pressure gauge and the initialized first-step viscous cap.
Float64 covers pressure/grad-h, variational force, Hamiltonian maps, E/A power,
accepted-state closure, restart and case adapters. The import audit imports all
531 modules and resolves 5,732 first-party imports across 1,215 source files.

An initial over-broad float64 invocation also included 33 support checks whose
fixtures explicitly construct float32 tensors; those fail the expected Warp dtype
contract and pass in the float32 selection. An initial float32 invocation included
six sloshing calibration checks whose fixture enters a CUDA-stream pipeline;
no GPU device/driver is available on this host, so those checks cannot run here.
Both initial logs are retained alongside the corrected selections. No CPU-only
workaround is introduced into production code. This integration does not claim a
fresh CUDA or full physical-matrix rerun; earlier qualified CUDA/trajectory evidence
remains preserved. Small regression loops intentionally omit video.

A byte-level source audit confirms the PESPH source/tests match `410a6e6` and
newer dev source/tests match `867b6d5`, except the shared runner, which contains
both changes. Generated artifacts are outside the merge. Test XML/logs and the
pre-integration primary plan backups are in `.tmp/pesph/integration-2026-10-23/`.

## 0. Scope audit — 2026-10-22

**Both variants now run through the solver; dissipative A remains guarded.** `PESPH` here means the pressure-energy baseline
(PESPH-E); `PESPH-a` means pressure-entropy (PESPH-A). The enum currently contains
one `CompressibleSPHScheme.PESPH` plus `PESPHVariant.{PressureEnergy,PressureEntropy}`.
Keep one shared numerical implementation with a variant selector. The original
scope audit is retained below; implementation has now started in the isolated
worktree, as recorded in the checkpoint.

For the requested evaluation matrix, A can explicitly opt into
`entropyHeating='SwiftLocal'`. This is a reference-style approximate local
conversion using the native heat terms; its energy error and cold-shock
positivity are evaluation outcomes. It does not lift the default safety gate.

### Worktree isolation — 2026-10-22

User decision: implement PESPH in the managed worktree on `codex/pesph`, starting
from primary checkout `dev` at `699d6c2`, while incompressible work continues there.
Merge the validated PESPH branch back into the primary checkout when ready. Do not
copy unrelated incompressible changes or merge over an uncommitted primary checkout.
Reconcile shared runner/configuration/PLANS changes at integration time and rerun the
affected compressible and incompressible checks.

Worktree: `/home/lu26029/.codex/worktrees/pesph/warpSPH`. Run Python from this root
with `PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"` and the existing `warp` conda
interpreter. Do not reinstall the worktree as editable into the shared environment:
that would redirect the concurrent primary checkout's imports. In this sandbox,
set `WARP_CACHE_PATH=/tmp/warpsph-pesph-scope-cache`,
`NUMBA_CACHE_DIR=/tmp/warpsph-pesph-numba-cache`,
`PYTHONPYCACHEPREFIX=/tmp/warpsph-pesph-pycache` and
`MPLCONFIGDIR=/tmp/warpsph-pesph-matplotlib`; pass pytest
`-o cache_dir=/tmp/warpsph-pesph-pytest-cache`.

Isolation verified: `warpSPH.__file__` resolves inside this worktree, and all six
`tests/test_numberDensitySupport.py` tests pass here on CPU. The primary checkout's
incompressible source edits and shared editable installation were left untouched.

### Implementation checkpoint — 2026-10-22

**Shared foundation and runner integration implemented; validation is in progress.**

- `modules/pesph/operators.py`: one fused Gather pass for number density, pressure
  weight sum, both support derivatives and mass density; one shared per-side force
  and conjugate pdV-work pass for E and A. Uses the core `SPHContext`/`OperatorSpec`
  launcher, existing kernels, periodicity and neighbour structures. Only O(N)
  stage data; no compatible-energy or stored pair arrays.
- `modules/pesph/thermodynamics.py`: variant weights from authoritative E/A,
  thermodynamic volume/density, correct number-density grad-h coefficient. Reuses
  `idealGasEOS` at the thermodynamic density, with explicit finite/positive checks.
- `configurations/pesph.py`: variant-aware config/dict round trip and rejection of
  unsupported material EOS/background pressure/support models. Existing diffusion
  and switch configs are inherited; no new dissipation preset is promoted here.
- `systems/pesph.py`: separate E/A integrated-field contracts and a shared system
  adapter using existing position/velocity/component update helpers. Accepted-state
  closure preserves the integrated variable instead of copying the last stage.
- `modules/pesph/stage.py` and `schemes/compressible/pesph.py`: shared support/closure
  refresh and an explicitly adiabatic `pesphConservativeStep`, reusing
  `solveStageSupport`. `pesphStep` is now registered by `buildScheme`; the original
  `pesphConservativeStep` remains an explicit adiabatic test entry point.

**Model evidence:** weights and force factors implement Hopkins (2013) Eqs. 18-22;
pressure-energy also matches Frontiere Appendix G. Cross-checked the current local
Spheral source at `116c71f2e`, `src/SPH/computePSPHCorrections.cc` and `src/SPH/PSPH.cc`:
its common-gamma `Fij=1-Fcorri/(m_j*u_j)` agrees with our `1-b_i/w_j` (Spheral's
stored correction differs by a factor gamma-1). Its positive conjugate thermal-work
term agrees with the pairwise energy balance used here, resolving the conservative
work-sign question. This is a source cross-check, not compiled Spheral parity.
Spheral's density scheduling/material replacement is not adopted; this implementation
retains the published Gather mass density independently of thermodynamic density.

**Verification:** `tests/test_pesphOperators.py` independently calculates the cubic
kernel sums and the two published force coefficients; checks support derivatives,
1D/2D adaptive-support force versus thermal-energy finite differences, 1D/2D/3D
momentum/energy and angular momentum where applicable, uniform zero-force startup,
authoritative-field metadata, invalid inputs, and accepted-state closure under RK2,
RK4 and velocity Verlet. A fresh float64 subprocess also checks kernel autograd in
positions/supports/weights/velocity, including grad-h. The foundation/support/per-side
selection reports **58 passed, 2 skipped** in 21.30 s; those two float64-only checks
run in the successful float64 subprocess instead. Runtime imports (521 modules)
and static imports (1,195 files / 5,540 first-party imports) pass. All checks ran
on CPU; no CUDA device is visible
in this environment, and no GPU performance or full benchmark claim is made.

### Runner/dissipation checkpoint — 2026-10-22

`buildScheme('PESPH')` now selects `PressureEnergyState` by default; its optional
`configured(config)` resolves `PressureEntropyState` after solver overrides, keeping
legacy seven-member bundle unpacking intact. The typed configuration, Sod and
linear-wave cases accept PESPH. NumberDensity/grad-h defaults survive case setup.
Example: `solver={"scheme":"PESPH","variant":"PressureEntropy","dissipation":false}`
selects adiabatic A; E uses `"PressureEnergy"` with dissipation enabled by default.
The supported slice is one ideal gas/shared gamma, positive thermal quantities,
eager list-based execution and forcing-only boundaries. Walls, mixed-gamma gases,
background pressure, density post-corrections and graph capture are rejected.

**Reuse:** `modules/dissipation/rates.py` composes the existing AV, conjugate viscous
heat, conductivity, reconstruction/gradient and viscosity-switch operators. It
returns acceleration and separate specific heating terms so PESPH-E forms a single
total-E RHS including external-force work. No compatible-energy pair storage or new
dissipation model was introduced. The initial native configuration uses the existing
Cullen-Dehnen switch with corrected velocity gradients and existing diffusion
parameters; it is not an exact Frontiere/Hopkins reproduction preset. Paper-specific
and Monaghan-default comparison rows remain separate future experiments.

**Dissipative A verdict: unresolved, guarded.** An independent autograd calculation
of `U(A)=sum m*u(A,y(A))` on the 2D scattered fixture, with a unit heat source at
particle 13, gives desired power `0.047132600761879154` but
`dU/dA . [(gamma-1) rhoThermo^(1-gamma) q] = 0.046473901829215215`:
a **1.3975% energy deficit**. Changing one A changes its neighbours' pressure-volume
energies. The naive local heat-to-A conversion is therefore rejected. No clipping,
global rescaling or dense Jacobian solve was added to hide it. Adiabatic A is usable;
`PressureEntropy` with `dissipation=True` fails validation. Derive/check a published
conservative heating treatment before lifting that guard.

**Initialization and timestep:** masses are preserved. Prescribed initial pressure
sets u/A at geometric volume `1/n`; the final pressure is the weighted kernel sum.
This exactly balances constant pressure on regular unequal-mass lattices. A foreign
sampler neighbour cache is discarded once: the core cache does not record support
mode, and a cached mean-support list missed union-support force pairs in the Sod
restart check. PESPH owns SuperSymmetric lists thereafter. Linear-wave sampling
uses the physical `rho=rho0+deltaP/c_s^2`, `v=deltaP/(rho0*c_s)` and regular volume
weights; it bypasses the legacy iterative mass rescaling. Shared EOS conversions,
support solve and initial samplers are reused. CFL uses the existing approach signal
speed; fixed requests are physically capped too, and previous dt is checkpointed
for adaptive growth limits. A shared fixed-dt helper now falls back to `config.dt`
when a sampler passes `dt=None`.

**Restart:** store the active thermal discriminator, authoritative E/A, supports and
all optional/scalar switch histories. Full-state HDF restores declared scalar
attributes; trajectory HDF preserves float64 static fields/time for PESPH and
records per-frame None histories. Loading bypasses the legacy mass-density EOS
reconstruction and rejects a variant mismatch. CPU round trips and continuation
pass for both variants and both formats, including Rosswog history. On CUDA float32,
rebuilding a neighbour cache changes summed entropy by 1.19e-7 and support by
2.98e-8 in the measured fixture; after two steps Rosswog's entropy time-difference
amplifies this to `delta alpha0 = .02358`, while `max delta velocity = 2.33e-10`.
An independent cache rebuild without I/O reproduces the sensitivity. The test
therefore requires exact checkpoint recovery, physical continuation agreement and
strict continued switch/history agreement in float64; it does not assert bitwise
float32 Rosswog switch continuation. The CPU float64 gate passes; the additional
CUDA float64 run **fails the 2e-11 alpha0 continuation tolerance** for Rosswog E
after a cache rebuild (15 passed before that failure, 190.83 s). Exact checkpoint
recovery and physical fields pass up to that assertion, and the default
Cullen-Dehnen restart cases pass in that run. Thus GPU strict switch continuation
is still an open acceptance gate; do not claim complete float64 GPU restart parity.
Quantify the amplification at float64 and set a justified precision/error bound or
cost a canonical reduction before resolving it. The failing check remains in the
test suite. No thresholds, precision or numerical model
were silently changed to pass this test. Canonical reductions or checkpointing the
neighbour cache would be separate costed work if bitwise float32 restarts are needed.
The shared progress display also now guards its
CUDA-only memory-pool query so CPU progress works.

**CPU checks:** focused pressure/support/operator/integration selection: **75 passed,
2 skipped in 17.38 s** (the float64-only checks run in a fresh subprocess); explicit
float64 integration/restart selection, including both Cullen-Dehnen and Rosswog:
**21 passed in 4.72 s**. Shared runner/schema
selection: **120 passed, 10 CUDA-only checks deselected, 22.48 s**. Imports pass:
524 runtime modules, 1,200 scanned source files, 5,597 first-party imports. CUDA is
hidden in the sandbox but visible to the host validation process; GPU checks are
run sequentially. **CUDA regressions:** pressure/operator/support/integration plus
the full shared runner/schema/catalogue selection: **205 passed, 2 skipped in
253.16 s**; fresh float64 pressure/autograd subprocess succeeds, accounting for
those two dtype-specific skips. The four added default Cullen-Dehnen restart cases
also pass on CUDA (**4 passed, 7.08 s**). The first foundation checkpoint above
records the earlier CPU-only evidence.

### Initial validation battery — 2026-10-22

**Evaluation requested 2026-10-22:** user requests both variants versus CRKSPH
on the paper/general compressible battery, including KH and Sedov, visually and
quantitatively. Queue GPU work behind the running incompressible matrix (explicit
user choice); do not interrupt it. Reuse existing cases/metrics, common physical
ICs and fresh-stage RK integration, recording configuration, status/failures,
conservation, positivity, accuracy/mixing measures and video/performance.
For dissipative A, SWIFT `PressureEntropy/hydro.h::hydro_end_force` explicitly
uses the local `(gamma-1)*rhoThermo^(1-gamma)*q` conversion; its setter warns that
thermal energy cannot be accurately set independently. Add an opt-in
`entropyHeating='SwiftLocal'` evaluation policy, retaining the guarded default.
This is a reference implementation's approximation, not a newly conservative
model; quantify its energy defect rather than requiring exact irreversible
conservation. Sources downloaded for inspection from SWIFTSIM/SWIFT master,
`src/hydro/PressureEntropy/{hydro.h,hydro_iact.h}` on 2026-10-22. Spheral PSPH
evolves energy and supplies no matching entropy-source implementation.

Evaluation harness: `scripts/evaluate_pesph.py`; 19 cases x 3 schemes = 57 rows.
Current full matrix: `scripts/out_pesph_suite_20261022_v2/`. All 57 configurations
validate, and 19 two-step CPU adiabatic A case-adapter checks pass. The full matrix
enables dissipation for E/A, with SwiftLocal selected explicitly for A. Coarse CPU
Noh A with heating lost positivity during preflight; this is retained as failure
evidence, not cured by clipping or arbitrary extra heating.

### Completed comparison and readiness — 2026-10-22

**57 main rows finished: 53 reached target, four failed before accepting a step.**
The incompressible matrix finished before this sequential GPU queue started.
Final evidence: [comparison overview](.tmp/pesph/docs/validation/pesph_evaluation_2026-10-22/index.html),
[all fields/movies/diagnostics](.tmp/pesph/docs/validation/pesph_evaluation_2026-10-22/main/comparison.html),
[metrics CSV](.tmp/pesph/docs/validation/pesph_evaluation_2026-10-22/main/metrics.csv).
`scripts/package_pesph_evaluation.py` packages lightweight evidence, audits final
snapshot times/source fingerprints and makes ignored hardlinks to the movies.
All physical sources remain as evaluated. Raw NPZ, original/replayed Vispy frames,
movies and logs stay in `scripts/out_pesph_*`; preserve them before archiving the
worktree. Replays use periodic display wrapping and shared temporal color ranges.
The final fresh report uniformly recomputes analytic/entropy metrics; it supersedes
the interim numeric checkpoints below.

- E conserves energy to roundoff in every completed closed-system case, but native
  contact, vortex damping, cold-shock positivity and cost prevent general acceptance.
- A explicitly opts into SwiftLocal; initial Sedov energy is 14.37%/20.90% low in
  2D/3D, then loses 7.88%/8.12% of its own initial energy. Woodward loses 2.40%.
  Its default heating guard remains. No compensating clipping or rescaling was added.
- Sharp KH final mode CRK/E/A = .199472/.150116/.153899, kinetic-energy retention
  .977167/.923734/.927837, wall 128.09/795.63/745.30 s. Both PESPH presets are more
  diffuse and cost about six times CRK wall time. This includes diagnostics/video.
- Gresho speed L1 = .025608/.082946/.084222; E/A retain ~61.6% of kinetic energy,
  while CRK gains 3.46%. Sod mass-density L1 = .005231/.007733/.006684. Wave
  amplitude-normalized velocity L1 = .007189/.007367/.004515.
- Native contact t=7 pressure errors: CRK 1.32e-10, E .080512, A .095045.
  The shared conductivity operator ignores `thermalConductivity` because its active
  return precedes the multiplier. The saved CPU probe gives identical rate L1
  19.8306205333 for strengths 0/.5/1. The main suite therefore uses unscaled native
  conductivity. The nominal-zero controls are **not source-off physics**.
- Two explicit evaluator-only source-off contact controls completed to t=1 with
  AV/viscous heat retained. E/A pressure errors 6.46e-14/9.13e-14 and maximum speeds
  1.54e-13/1.75e-13 pass the declared <1e-6 diagnostic target. This identifies
  conductivity as the drift source for this contact. Root:
  `scripts/out_pesph_contact_sourceoff_20261022/`. The production multiplier fix
  and a contact-compatible conductivity treatment require tests and fresh runs;
  the current matrix does not validate either future change.
- Noh/shearing Noh E/A fail the first RK2 step (u/A positivity). Noh startup controls
  still fail at CFL .015; .0015 accepts only two steps to t≈1.11e-5. This is timestep
  sensitivity, not a completed Noh test. CRK completes but records negative minimum u.
- Kidder interior errors are small for all; CRK's four final negative-u particles
  are in prescribed bands, with no negative interior values. Yee's first-step band
  thermal projection needs initialization synchronization. Triple-point initial
  PESPH energies are 4.21%/4.56% below CRK, precluding matched-energy rankings.
- All 19 cases, including smooth KH, multidimensional Sod, RT, blob and shear, have
  fields, histories, costs and diagnostic interpretation in the final report.
  Native support/dissipation presets differ; no exact paper reproduction or isolated
  pressure-formulation superiority claim is made. Eight diagnostic rows are separate
  from the 57 main rows.

**Next gates:** tested conductivity multiplier and contact-source treatment; full
cold-shock positivity/CFL convergence; conservative A heating and blast initialization;
Yee band synchronization; performance; strict float64 GPU Rosswog alpha0 restart
tolerance; matched-dissipation and resolution refinement. Keep `codex/pesph` isolated;
no primary-checkout merge until these readiness issues are resolved.

Follow-up investigation (user, 2026-10-22): explain unexpected immediate Noh
failure and the large cost relative to CRKSPH before treating them as inherent
scheme limitations. First inspect RK-stage thermal/heat-source positivity,
total-energy thermal recovery, support iteration counts, repeated refreshes and
host/launch overhead. Preserve the frozen battery; any diagnostic or fix must be
separate and checked against the existing baseline. Monaghan cost was not measured
in that battery, so no new measured ratio to it is claimed.

**Follow-up evidence:** `.tmp/pesph/scripts/diagnose_pesph_startup.py` is a CPU float64
operator/RK-algebra probe, not an accepted simulation (no GPU/video required).
[Budgets](.tmp/pesph/docs/validation/pesph_startup_diagnosis_2026-10-22/diagnosis.json) and
[figure](.tmp/pesph/docs/validation/pesph_startup_diagnosis_2026-10-22/diagnosis.png) retain
eight evaluations: E/A × limited at CFL .15/.015/.0015 and raw at .15.

- Support solver: `h_ratio=h_new/(h_prev+1e-6)` cannot satisfy tolerance 1e-5
  for a stationary h below .099999. All initial KH/Gresho/wave supports fall in
  that range. The Noh operator probe executes all 16 iterations despite true
  relative changes <=6.7e-16; the implemented test reports 3.20e-5. RK2 calls
  support/closure for both RHS stages and accepted state. This is avoidable
  implementation cost, not evidence of intrinsic 6x PESPH complexity. A proper
  relative-change test and validated support reuse/fusion should be profiled
  before any reduced-iteration or looser-accuracy numerical preset is considered.
- Noh: first invalid particles are at x≈±.9961, the outer periodic seam, away
  from the central implosion. The periodic IC includes a separating velocity
  jump there. With limited reconstructed AV, initial seam acceleration is
  ~7.60 while local viscous heat is zero. At dt=.000561296, E's midpoint
  kinetic quadratic term is 2.2741e-6 against u0≈1e-8; recovered u becomes
  -2.2643e-6. This is explicit midpoint/thermal recovery, not mere roundoff.
- A's midpoint remains positive, but the second RHS has seam viscous heating
  -0.025234 and proposed final min A=-9.6126e-6. Conduction there is only
  ~1.5e-13: this failure is distinct from the contact conductivity issue and
  does not establish an intrinsic pressure-entropy shock limitation.
- Changing only the existing pair-velocity policy to Raw removes that initial
  seam acceleration and both proposed first steps remain positive at CFL .15
  (E min u≈9.65e-9, A min A≈6.67e-9). This isolates the reconstruction/viscous
  coupling; **it is not a full Noh pass or a reason to promote Raw by default**.
  Audit pairwise sign/limiter behavior and the periodic test boundary, then run
  separate full endpoint checks. Physical sources and original matrix are unchanged.

### Follow-up correction protocol — 2026-10-22

User authorized continuing the convergence/viscosity corrections and follow-up
validation. Preserve the original 57-row battery and pre-change sources under
`.tmp/pesph/docs/validation/pesph_followup_2026-10-22/as_evaluated_sources/`.
First correction: a true relative support change, with no absolute length bias;
unchanged tolerance, Newton updates, support constraint and fresh RK stages.
CPU support regressions: 10 passed, including converged small-support solves at
two length scales with Monaghan and NumberDensity.

Viscosity research: Frontiere (2017) Eqs. 48–55 and local Spheral
`LimitedMonaghanGingoldViscosityViewInline.hh` at 116c71f2e use the same projected
gradient ratio as the current native Limited policy. That formula does not by
itself bound reconstructed pair jumps by the sampled jump. Do not label a new
bound as a transcription fix or silently replace the published limiter. Any
admissibility/monotone reconstruction variant stays explicit and experimental.
First separate full Noh/shearing-Noh Raw controls at the original CFL/resolution;
the support-only correction must also receive native KH timing/quality checks.
Declare full endpoint, positive thermal fields and conservation/entropy diagnostics
as the validation targets; retain failures and do not add energy/entropy floors.

Candidate to test after the support-only queue finishes: explicit
`VelocityPairPolicy.MonotoneLimited`, default off, reusing native Limited first.
For the pair unit normal n, project its reconstructed normal jump onto the
interval joining zero and the sampled normal jump. Equivalently, minmod the
normal reconstruction correction against the sampled jump. Tangential
reconstruction, pressure forces, conductivity and conjugate AV work are unchanged.
The shared policy feeds both force and heating so pair energy/momentum balance
is preserved. This is a **local experimental composition**, not Frontiere's
published limiter or Spheral PSPH parity. Spheral's GSPH MinModLimiter/LimiterBase
and RiemannSolverBase provide a cross-check for limiting against the sampled
normal jump; their complete formula is different. Garcia-Senz/Cabezon (2026,
arXiv:2510.20534v2, introduction/§5.2) explicitly warn that SLR need not produce
positive dissipation. That supports retaining a separate native control rather
than claiming the original reconstruction is mistranscribed.
Predeclared candidate checks: local AV heat nonnegative on radial kernels with
native positive viscosity coefficients, pair momentum/energy balance, smooth
linear reconstruction preserved, no first-step failures, full Noh/shearing Noh
endpoints at CFL .15 plus Noh CFL .075, Sod profile/energy comparisons and native
KH timing/quality. E closed-system energy target <1e-10; A energy defects remain
measured and guarded independently. No adoption based only on completion.

### Historical progress notes from the matrix (superseded by final report above)

Live checkpoint: the incompressible matrix completed before this GPU queue
started. Wave/Sod and both Sedov dimensions are complete; sharp/smooth KH are
next. The completed 3D Sedov A run starts at E=0.79098516 for requested E0=1
and loses another 8.1208% of that initial energy (2D: E=0.85633052, loss
7.8764%). These are distinct initialization and evolution defects, so A's
profile errors are not a matched-energy acceptance result. E and CRK retain
roundoff energy balance. Per-row source/dependency fingerprints, actual times,
videos and raw snapshots remain available; report-only improvements do not
change the physical model during the matrix.

Sharp KH checkpoint at t=2.5, N=16384: CRK/E/A mode amplitudes
0.199472/0.150116/0.153899; KE retention 0.977167/0.923734/0.927837;
wall time 128.09/795.63/745.30 s. E conserves total energy to roundoff;
A's maximum relative change is 1.95362e-4. Shared-scale final plots show
roll-up in all three, more diffuse/interpenetrating original particle labels
in PESPH and less pressure scatter than CRK. This is not an isolated-formulation
ranking: native support/dissipation presets differ. Initial positions, masses
and velocities are exactly identical for the completed case triples.

Evaluator corrections: some native CRK samplers leave the initial entropy
cache zero. Postprocessing consistently derives its entropy from saved u/rho,
using PESPH's pressure-volume A for PESPH; raw histories remain retained.
Particle coordinates are unwrapped in the integrator. Wrap only the displayed
coordinates using the recorded domain, preserving raw trajectories. Early KH
movies require `evaluate_pesph.py replay` **after** the sequential GPU matrix;
this regenerates Vispy frames/video from the saved states, retaining originals.
Final publication requires a fresh `report` process (the running supervisor
loaded an earlier report implementation).

Contact follow-up recorded before investigation: both schemes start with
P=1 to roundoff, but the full native PESPH-E contact run develops ~5% maximum
pressure error by t=1.55, whereas CRK remains at rest through t=7. Existing
initialization tests independently check near-zero conservative force for an
unequal-mass constant-pressure contact. The reused native Price-2008
conductivity is a candidate coupling to test, not a diagnosed cause yet;
CRK's native energy kernel does not use that conductivity operator. After the
57-row GPU matrix, run separate matched E/A contact controls to t=1 with only
`thermalConductivity=0`, comparing against the retained default-0.5 trajectories.
Keep RK2/CFL/resolution/supports fixed, record video and pressure/velocity/shape
errors, and preserve the main matrix. Declared diagnostic target: pressure and
velocity errors below 1e-6 if this source is responsible. No default change or
baseline tuning without that evidence and a documented verdict.
**Ablation validity finding:** the nominal `thermalConductivity=0` controls
do not disable the component. The shared `computeConductivityWarp` returns
`launchOperator(...)` before its intended multiplication by
`conductivityParams.thermalConductivity`; the scaling return is unreachable.
Preserve the two knob-control runs as implementation-contract evidence, not
as conductivity-off physics. Verify this with a CPU operator scaling check,
then run a separately labelled evaluator-only source-off control that replaces
only the conductivity contribution by zero. Keep the solver source frozen so
the 57-row matrix and CFL controls retain the same model. The shared scaling
bug needs a tested fix and fresh physical validation before a default change;
no inference that conductivity is irrelevant can be made from the broken knob.
At that checkpoint controls were prepared/validated in
`scripts/out_pesph_contact_conductivity_20261022/` (two E/A rows, N100 to t=1).
Completed native E contact at t=7: pressureError=0.0805118,
shapeDisplacement=0.0329094, maximum velocity over history=0.0370976,
energy change <=4.44e-16, wall=927.36 s. CRK's pressureError is 1.32e-10
and max velocity 1.44e-12 at t=7. Both start at P=1; this is an evolution
quality failure despite E conservation, and blocks treating the native PESPH
configuration as generally validated.

Kidder native CRK checkpoint: reaches t=0.02699155 (0.95 tau) in
500.46 s / 50270 steps. Interior density/velocity L1 are 5.50e-7 / 3.18e-6,
but accepted internal energies reach -8653.93. Final negative values occur
only at indices 0, 1, 198, 199 in the prescribed bands; interior min u=129.71.
This is a boundary-state consistency issue to retain separately from the
very small interior analytic errors. The comparator reports band/interior
counts and excludes the ten-point bands from analytic L1. Native CRK
finalization is unchanged in this comparison; do not call this an intrinsic
CRK failure or discard its interior accuracy evidence.

Cold Noh native E/A both fail before accepting their first RK2 step at
N256 / CFL0.15 / uAmbient1e-8. CUDA assertion messages identify non-positive
specific internal energy (E) and non-positive A (A); only t=0 is recorded.
No floor or clipping is added. CRK reaches t=0.600263 with 0.0253% plateau
density error but minimum accepted u=-2.37e-4, so conservation does not imply
positivity there either. Record a timestep-sensitivity follow-up before
investigation: after the main matrix, run two-step E/A startup controls at
CFL0.015 and 0.0015, same N256/IC/presets, with video. These test immediate
failure sensitivity only; passing two steps is not a completed Noh shock test.
No other coefficient or physical state changes. Keep them separate from the
57 main rows and retain all failures.
At that checkpoint the four controls were prepared/CPU-validated in
`scripts/out_pesph_noh_startup_cfl_20261022/`. Shearing-Noh E/A also fail
before their first accepted step with the same u/A assertion messages;
its CRK row reaches t=0.600354 but records min u=-2.95e-4. Do not remove
the PESPH checks to make these appear complete.

Yee energy histories expose a startup band projection: E=199.23235 ->
192.82641 on its first step, then 192.82739 at t=2; A=198.81959 ->190.80165,
then 190.80149. CRK starts at 192.85266 and ends 192.85233. These are not
closed-box energy errors, but the initial thermal mapping and prescribed-band
synchronization need separate validation. Interior speed L1 is .0071693 /
.0016971 / .0025077 (CRK/E/A); do not claim an isolated formulation ranking
from that number while these startup states differ.

Cold Sedov/Noh/shearing-Noh cases use a common explicit `uAmbient=1e-8` for all
solvers (legacy case default stays zero); exactly zero pressure makes the
pressure-weighted fractional/division closure undefined away from the blast.
The helper adds this initial thermal energy and recomputes EOS consistently.
No per-step temperature/entropy floor is introduced. The first Gather-mode CRK
wave preflight emitted the existing conservation warning; it is preserved in
`scripts/out_pesph_suite_20261022/` and excluded from comparison. The v2 baseline
pins the published KernelMeanSymmetric pair mode and uses the existing CRK
acoustic-CFL hook each step. PESPH uses its own approach-signal CFL. Fresh-stage
RK2 and scalar n_h/kernel settings are common. Native dissipation presets remain
distinct and are recorded; results compare usable solver configurations, not
an isolated pressure-discretization ablation. Driven fluid-band BCs for
Kidder/RT/Yee reuse existing dispatch and translate prescribed rho/u/v into E/A;
their energy exchange makes closed-box conservation claims inappropriate.

**Early full-run evidence (matrix still running):** N128 acoustic crossing:
normalized velocity L1 CRK .007188, E .007367, A .004515; median step 6.62,
48.45, 51.44 ms respectively. Different native dissipation presets contribute
to those costs. Wave density/pressure include a CRK background quadrature
offset; report mean offset separately from mean-removed mode error.
N400 classic Sod: density L1 CRK .005231, E .007628, A .006547; A's maximum
energy change is 1.15e-4, versus roundoff for CRK/E. 2D Sedov N=9409: E reaches
the physical target, with relative energy drift 5.46e-14 and particle-sampled
density L1 .45269 (CRK .50226). A starts at U=.8563305 rather than requested E0=1
and loses another 7.8764% of that initial E during evolution. Its lower nominal
density L1 (.37129) is **not a matched-energy accuracy win**. This identifies
an initialization gap separately from the irreversible-source defect; no energy
renormalization or per-step correction was inserted during the experiment.
Classify large initial-energy mismatches explicitly in the report and preserve
the raw outputs. The radius diagnostic is the peak of a radial density bin, with
finite bin width, and should not imply sub-bin accuracy.

Reproducer: `scripts/probe_pesph.py` (float64, CPU, RK4, normalised kernel,
number-density supports). All six runs stream progress, use the stall/velocity
watchdogs, and save Vispy frames/video. Quantitative data:
[`.tmp/pesph/docs/validation/pesph_2026-10-22.json`](.tmp/pesph/docs/validation/pesph_2026-10-22.json).
Local regenerated videos/frames: `scripts/out_pesph/2026-10-22/` (ignored by Git).
These are analytic comparisons; no unchanged baseline solver comparison is claimed.

| Case | N / CFL | Density / pressure / velocity L1 | Relative E drift | Entropy change | Wall / median step |
|---|---|---|---|---|---|
| acoustic E, t=0.502 | 48 / .15 | .02945 / .03206 / .02563 | 1.23e-16 | -7.75e-9 | 8.82 s / 54.78 ms |
| acoustic E, t=0.500 | 96 / .15 | .007657 / .008320 / .006501 | 0 | -2.02e-9 | 17.90 s / 60.52 ms |
| acoustic A, t=0.502 | 48 / .15 | .02485 / .02485 / .01470 | -2.47e-16 | 0 | 8.44 s / 56.65 ms |
| acoustic A, t=0.500 | 96 / .15 | .006450 / .006445 / .003759 | 1.23e-16 | 0 | 17.42 s / 61.56 ms |
| classic Sod E, t=0.1000 | 144 / .10 | .01761 / .02117 / .03468 | -1.61e-16 | +.01186 | 18.25 s / 86.85 ms |
| classic Sod E, t=0.1002 | 144 / .05 | .01771 / .02126 / .03480 | -3.23e-16 | +.01192 | 36.25 s / 87.81 ms |

Wave errors are divided by the analytic perturbation amplitudes (pressure amplitude
1e-3), Sod errors are absolute particle-sampled L1 on x>=0 against the mirrored
Riemann solution. Comparisons use each run's actual final time; runner overshoot
differs slightly. Mass drift is zero in all rows, momentum changes <=1.72e-16,
pressure/u/mass-density/support stay positive. Wave refinement reduces errors by
3.8–3.9x (velocity orders E 1.98, A 1.97). Halving Sod CFL changes the reported
L1 errors by <=.55% and entropy growth by .51%, with similar per-step cost. Sod
has visible smoothing and small overshoots; this is a smoke/accuracy check, not
a strong-shock, mixing or default-preset acceptance. Videos inspected at the Sod
final profile. There was no previous same-case PESPH timing to compare; timings
are CPU eager baselines, not GPU performance claims. Wall times include setup/video.

**Verdict:** keep the shared conservative implementation and PESPH-E runner slice;
initial acoustic convergence and Sod timestep sensitivity pass. Still open:
strict CUDA float64 Rosswog switch continuation, conservative A heating, Kidder and hydrostatic
contact, unchanged-baseline comparisons, Sedov/Noh and multidimensional
vortex/mixing tests. No merge back to the primary checkout until those integration
and intended benchmark gates are met.

### 0.1 Original audit snapshot (before the implementation checkpoint)

| Ingredient | Current evidence | Remaining work |
|---|---|---|
| Scheme and variant enums | `enumTypes.py`; exported by `warpSPH.__init__` | `schemes/builder.py:buildScheme` deliberately raises `NotImplementedError`; no PESPH config, step, or module package exists |
| Number-density support solve | `modules/adaptiveSupport/optimalSupportMonaghan.py`, dispatched by `optimalSupport.py`; `solveStageSupport` requests Gather | Reuse it, but recompute number density and its derivative at the returned supports. Iteration history is not a final-state cache |
| Kernel support derivative | `modules/adaptiveSupport/wp_omega.py` calls `sphKernelDkDh` | New weighted sums for PESPH grad-h; existing scalar mass-density Omega is insufficient |
| Total-energy integration | `systems/compressibleMonaghan.py` already has `integrated('dEdt')`, the quantity-update call, and no finalizer overwrite of E | Done since the original plan. PESPH-E must make E authoritative and derive u at every stage; the existing state also integrates u |
| Entropy integration | `entropies` is still `constant`; `CompressibleSystemUpdate` has no `dAdt` | Entropy state/update contract and quantity update; avoid copying last-stage A over integrated A in `finalize` |
| Ideal-gas conversions | `modules/eos/idealGas.py` and `materials.py` already support A as input | PESPH closure. `refreshEOS` currently always reads u and overwrites A and P; do not call it unchanged |
| Dissipation and gradients | Existing AV, conductivity, viscous heating, switches and reconstruction modules | Reuse operators with explicit density/pressure conventions and consistent heat conversion; verify conservation |
| Cases, configuration and I/O | Existing shock, vortex, contact and mixing cases; generic bundle and HDF5 machinery | PESPH is absent from case allowlists and the typed runner configuration; trajectory loader currently reconstructs A/E from u |

**Fresh verification:** all six tests in `tests/test_numberDensitySupport.py` pass
on CPU, using the `warp` conda environment and
`WARP_CACHE_PATH=/tmp/warpsph-pesph-scope-cache`. These cover equal/unequal-mass
support behaviour, the enum-only failure, and the ideal-gas A round trip. They do
not validate a PESPH force or trajectory. The initial test attempt could not write
Warp's default cache; redirecting it resolved that. No GPU physics, timing, or
full-suite result is claimed by this audit.

### 0.2 Design constraints and corrections

1. **Fix the grad-h parentheses.** The old G.7 transcription put the leading
   `1 -` inside the numerator. Hopkins (2013), Eq. 18/22, and Frontiere Appendix G
   instead give `fH_ij = 1 - C_ij / OmegaN_i`, not `(1 - C_ij) / OmegaN_i`.
   Section 1 below is corrected. Use `fH`/`gradH`, never the compatible-energy `f_ij`.
2. **PESPH-A needs its own force coefficient.** It is not G.4 with only a new
   pressure sum: the factors are `(A_i A_j)^(1/gamma) Pbar_i^(1-2/gamma)`
   (Hopkins Eq. 21), and grad-h differentiates `y=Pbar^(1/gamma)` (Eq. 22).
   A shared weighted-volume implementation can express both variants.
3. **Keep three density concepts distinct.** The h constraint uses `n=sum W`;
   ordinary mass density is `rhoMass=sum m_j W`; pressure-entropy thermodynamics
   use `rhoThermo=(Pbar/A)^(1/gamma)`. In unequal-mass flow, the support solver's
   returned `m_i*n_i` is generally not `rhoMass`. For PESPH-A, derive
   `u=A*rhoThermo^(gamma-1)/(gamma-1)` and the matching sound speed, rather than
   applying the ordinary EOS at the mass-summation density. Retain `state.densities`
   as mass density for existing output consumers and carry/pass thermodynamic
   density explicitly. Audit each AV, switch, gradient and timestep consumer.
4. **Choose authoritative integration fields deliberately.** PESPH-E should advance
   positions, velocities and E; PESPH-A should advance positions, velocities and A.
   Internal energy is derived in both designs. Reusing the current Monaghan state
   unchanged leaves redundant integrated thermodynamic variables, including in
   implicit/JFNK packing. Prefer variant-specific internal state/system adapters
   sharing one RHS. Since `SchemeBundle` fixes its state class before config
   overrides, resolve the variant's adapter after config validation and before case
   sampling, or add an explicit factory mechanism; a config switch alone cannot
   change dataclass metadata. Test generic and component-update integrators.
5. **A plain RHS still needs lifecycle plumbing.** There is no compatible-energy
   step-map, but the inherited finalizer copies last-stage A/P/c and refreshes the
   ordinary EOS for material tables. Implement PESPH closure at the actual accepted
   state and preserve integrated E/A. Do not claim support for every integrator
   before testing it.
6. **Dissipative energy accounting is a separate gate.** Existing viscosity returns
   acceleration, `computeThermalDissipation` returns viscous heat, and conductivity
   transfers internal energy. PESPH-E's E rate must include all three consistently,
   plus external-force work when enabled. PESPH-A heating changes neighbours'
   pressure sums as A changes, so the usual local `du -> dA` conversion needs a
   conservation check; `dA=0` adiabatic tests are insufficient. The conservative work
   sign is now checked against pairwise energy balance and Spheral (checkpoint
   above); the dissipative extension remains a separate gate.
7. **Bound the first scope.** One ideal gas with shared gamma, positive A/u/P,
   periodic or existing open-domain benchmark setups, eager execution first.
   Explicitly reject unsupported mixed-gamma/material EOS, nonzero background
   pressure and solid/analytic walls initially. Recent material-EOS work does not
   establish a valid common `1/gamma` pressure sum across different gases. Graph
   capture, splitting integrators, cooling and new wall closures are follow-ups.
   The AV M8 tag is not a blocker (existing user decision, 2026-10-09).

### 0.3 Shared kernel design

Accumulate `n`, `y`, `dn/dh`, `dy/dh`, and `rhoMass` using Gather kernels at each
particle's own support. Include the kernel self term and derivative. Use the repo's
support-radius convention consistently with `sphKernelDkDh`.

```
                       PESPH-E                      PESPH-A
w_i                    (gamma-1) m_i u_i            m_i A_i^(1/gamma)
y_i = sum_j w_j W_ij    Pbar_i                       Pbar_i^(1/gamma)
Pbar_i                 y_i                          y_i^gamma

OmegaN_i = 1 + h_i/(dim*n_i) * dn_i/dh_i
b_i      = h_i/(dim*n_i) * (dy_i/dh_i) / OmegaN_i
fH_ij    = 1 - b_i/w_j

dv_i/dt = -sum_j (w_i*w_j/m_i) * [
             (Pbar_i/y_i^2)*fH_ij*grad_i W_ij(h_i)
           + (Pbar_j/y_j^2)*fH_ji*grad_i W_ij(h_j)] + aDiss_i
```

Store particle sums/coefficient `b_i`; form the pair coefficient inside the force
loop. No persistent O(N*neighbours) `fH_ij` storage is needed. Build a small
direct-sum reference first, then Warp kernels using the current operator API.
The variants need not agree on an arbitrary disordered finite-particle state;
test agreement on a consistently normalized uniform state and test each separately
at contacts. Shared infrastructure does not imply identical discrete thermodynamics.

### 0.4 Work packages and acceptance gates

| Order | Deliverable and main integration points | Acceptance |
|---|---|---|
| 1 | Close equations/state contract; PESPH config and variant adapter; reference sums (`configurations/pesph.py`, `systems/pesph.py`, `modules/pesph/`) | Exact integrated-field sets, config round trip, invalid-option rejection, uniform-state thermodynamics, unequal-mass density distinction |
| 2 | Weighted pressure/grad-h kernels and conservative PESPH-A RHS (`schemes/compressible/pesph.py`) | Float64 scattered-state finite differences for support derivatives; pair momentum/angular-momentum balance; force agrees with derivative of `sum m*u(q,A)` with supports re-solved under position perturbations |
| 3 | Register usable adiabatic PESPH-A; initial/accepted-state closure, CLI/schema and case allowlists | Stock RK2/RK4 smoke, then acoustic convergence and Kidder entropy/energy versus dt and resolution; static contact has no systematic pressure blip |
| 4 | PESPH-A dissipation and PESPH-E total-energy RHS; common reconstruction/switch path | AV kinetic loss equals heat gain; conductivity conserves total energy; entropy/positivity checks; E-derived u and diagnostics agree; forcing includes work |
| 5 | Complete runner, export/import, diagnostics and comparisons (`runner/configuration.py`, `cases/common/parameters.py`, `cases/compressible/common.py`, per-case adapters, `io/importIO.py`, reporting harness) | Case defaults cannot silently replace NumberDensity/grad-h with Owen/off; restore authoritative A/E and support/switch history needed for continuation; uninterrupted versus restarted trajectory agreement |
| 6 | Both variants in compressible validation matrix with unchanged baselines | Sod, Sedov/Noh, Kidder, acoustic wave, Gresho/Yee, hydrostatic contact, KH/blob; quantitative accuracy/conservation/positivity, timestep sensitivity and runtime evidence |

Some runner work from package 5 is needed for the early vertical slice in package 3.
Use paper-specific Cullen-Dehnen/Cullen-Hopkins + conductivity settings for a
Frontiere/Hopkins reproduction, and a separately labelled row for this repo's
Rosswog/reconstruction default. Do not conflate a new discretization with a changed
dissipation preset. Compare with Spheral where its PESPH implementation is available
and applicable; otherwise use the direct reference and published tests. Full
simulations retain the repository's video/progress/watchdog conventions.

**Revised sizing, not a commitment:** about 8-12 focused working days to an
adiabatic PESPH-A vertical slice, and 15-22 total for both variants with runner/I/O
and validation. Entropy heating/conservation, state-adapter selection and strong
shocks are the main uncertainty. CUDA graph work beyond a measured eager baseline
is additional scope. The old Phase 2 Kidder-before-force gate was circular:
derivative checks precede force kernels; Kidder follows a runnable adiabatic
force/closure. No dependency on new splitting or MFM work is required.

**References checked:** [Hopkins (2013), Eqs. 18-22 and section 3.4](https://arxiv.org/html/1206.5006),
[Hopkins (2015), Appendix F2](https://arxiv.org/abs/1409.7395), and the local
`literature/frontiere2017_crksph.pdf`, Appendix G. Sections 1-6 retain the original
rationale with central equation/gate corrections; this audit supersedes older
completion claims, file paths and estimates where they differ.

## Why this exists

The original reference comparison has `monaghan.py`, `compSPH.py` and
`crkSPH.py` (now under `schemes/compressible/`, alongside GSPH and MFM/MFV). The fourth scheme in that comparison —
Frontiere, Raskin & Owen (2017), *CRKSPH*, JCP 332:160-209 — is **PESPH**, the
pressure-based formulation, and it is missing. Adding it closes the comparison set,
gives the compressible branch two strong astro-side baselines, and (in one of its two
variants) provides a **clean
`f(state) -> update`** — no compatible-energy step reconstruction, no pairwise arrays
outliving a substep.

That last property is why this plan is sequenced **before** the integrator work in
`../warpSPHIntegrators/SPLITTING_PLAN.md` rather than after it. See §6.

Two variants exist and both are worth having:

- **PESPH-E (pressure-energy)** — pressure by summation over neighbours' *thermal
  energy*, evolving *total* energy. This is what Frontiere Appendix G and Hopkins (2015)
  Appendix F2 actually implement, and what GIZMO/FIRE run in production.
- **PESPH-A (pressure-entropy)** — pressure by summation over neighbours' *entropic
  function* `A`, evolving `A`. Hopkins (2013), MNRAS 428:2840. This is the variant that
  makes the scheme separable and variational.

They share roughly 90% of their code and should be **one scheme with a switch**, not two
schemes (§3.1).

Papers are in `../warpSPHIntegrators/literature/`:
`1-s2.0-S0021999116306453-main (1).pdf` (Frontiere et al.) and `stv195.pdf` (Hopkins
2015).

---

## 1. The equations

### 1.1 PESPH-E (pressure-energy), Frontiere Appendix G / Hopkins F2

```
n_i  = Σ_j W_ij(h_i)                                                        (G.3)
ρ_i  = Σ_j m_j W_ij(h_i)                                                    (G.2)
P̄_i  = Σ_j (γ−1) m_j u_j W_ij(h_i)                                          (G.1)

Dv_i/Dt = −Σ_j m_j (γ−1)² u_i u_j [ (f_ij/P̄_i) ∂_α W_ij(h_i)
                                  + (f_ji/P̄_j) ∂_α W_ij(h_j) ] + q_acc_ij   (G.4)

DE_i/Dt = m_i v_i·Dv_i/Dt
          + Σ_j m_i m_j (γ−1)² u_i u_j (f_ij/P̄_i) v_ij·∂_α W_ij(h_i)
          + v_ij·q_acc_ij                                                    (G.5)

q_acc_ij = ½ (ρ_i Π_i + ρ_j Π_j) (∂_α W_i + ∂_α W_j)/(ρ_i + ρ_j)             (G.6)

f_ij = 1 − [(h_i / (ν(γ−1) n_i m_j u_j)) ∂P̄_i/∂h_i] / [1 + (h_i/(ν n_i)) ∂n_i/∂h_i]  (G.7)
∂n_i/∂h_i = −Σ_j h_i⁻¹ [ν W_ij(h_i) + η_ij ∂W/∂η|η_ij]                       (G.8)
∂P̄_i/∂h_i = −Σ_j (γ−1) m_j u_j h_i⁻¹ [ν W_ij(h_i) + η_ij ∂W/∂η|η_ij]        (G.9)
```

with `η_ij ≡ |x_j − x_i|/h_i` and `ν` the dimension. `Π` is Monaghan-Gingold (Eq. E.7)
under the Cullen-Dehnen switch (Appendix F). `u_i = E_i/m_i − v_i²/2`.

### 1.2 PESPH-A (pressure-entropy), Hopkins (2013)

The pressure sum and integrated variable change, together with the force coefficient
and the derivative entering grad-h (Hopkins Eqs. 19, 21-22):

```
y_i = Σ_j m_j A_j^(1/γ) W_ij(h_i),    P̄_i = y_i^γ
ρThermo_i = (P̄_i/A_i)^(1/γ)
u_i = A_i ρThermo_i^(γ−1)/(γ−1),    c_{s,i} = √(γ P̄_i/ρThermo_i)

fH_ij = 1 − [h_i/(ν n_i m_j A_j^(1/γ)) * ∂y_i/∂h_i] / [1 + h_i/(ν n_i) * ∂n_i/∂h_i]
Dv_i/Dt = −Σ_j m_j (A_i A_j)^(1/γ) [fH_ij P̄_i^(1−2/γ) ∇_i W_ij(h_i)
                                  + fH_ji P̄_j^(1−2/γ) ∇_i W_ij(h_j)] + aDiss_i

DA_i/Dt = 0                              for adiabatic flow
DA_i/Dt = (γ−1) ρThermo_i^(1−γ) (du_i/dt)|_diss  is the local conversion to audit
```

Grad-h uses `∂y_i/∂h_i`, obtained with the weight `m_j A_j^(1/γ)`, not the
outer-power derivative `∂P̄_i/∂h_i`. Dissipative heating also affects neighbouring
pressure sums; check total-energy conservation explicitly (section 0.2).

### 1.3 Naming hazard — read this before writing code

**Frontiere uses `f_ij` for two completely unrelated quantities:**

- Eq. (67) — the compSPH/CRKSPH **energy-partition** factor, constrained by
  `f_ij + f_ji = 1`.
- Eq. (G.7) — the PESPH **grad-h correction**, the analogue of compSPH's `Ω_i`.

`CompressibleState.f_ij` in this repo already means the first
(`modules/compSPH/balance.py`). Pick a different name for the PESPH term —
`gradH_ij`, or `fH_ij` — before any of this is typed, or the two will be conflated in
review.

---

## 2. Audit: what this repo already has

Checked 2026-09-15. Substantially more than expected:

| PESPH ingredient | Status |
|---|---|
| Cullen-Dehnen viscosity switch (App. G specifies it) | ✅ `ViscositySwitch.CullenDehnen2010` **and** `CullenHopkins` (`enumTypes.py:43`, `modules/shockCapturing/`) |
| Artificial conductivity (App. G requires it) | ✅ `modules/dissipation/wp_conductivity.py` |
| `∂W/∂h` primitive for grad-h terms | ✅ `sphKernelDkDh`, live in `modules/adaptiveSupport/wp_omega.py:81` (not just spike scripts) |
| Linearly-corrected velocity gradient (E.4/E.6) | ✅ used by compSPH; App. G says PESPH uses it too |
| Entropic-function EOS `A ↔ (u, P, c_s)` | ⚠️ **exists but is dead code** — `modules/eos/gas.py`, `EOSSource.specificEntropy`, with all four conversions written. Its own docstring: "Not wired into `eos/__init__.py` or any scheme/case in this repo." |
| `entropies` / `totalEnergies` on the state | ⚠️ present but `constant()`, not `integrated()` (`systems/compressibleMonaghan.py:37-38`) |
| `dEdt` on the update | ✅ `CompressibleSystemUpdate.dEdt`, and all three schemes already compute it |
| Number density `n_i = Σ_j W_ij` | ✅ effectively — CRK's `V_i⁻¹ = Σ_j W_i` (Frontiere Eq. 75) is exactly this kernel |
| Scheme registration | ✅ ~10-line `SchemeBundle` (`schemes/builder.py:97`) + one `CompressibleSPHScheme` enum entry |
| Validation cases | ✅ `caseUtils/compressible/`: sod, sedov, noh, kidder, gresho, yeeVortex, kelvinHelmholtz, rayleighTaylor, triplePoint, hydrostatic, blob — **Frontiere's own test battery** |

### 2.1 Two state-field bugs found during the audit — **FIXED 2026-09-30 (AV_PLAN S2 step 2)**

> `entropies` is now tagged `'entropy'` and `pressures` `'pressure'` in both `systems/compressibleMonaghan.py` and `systems/compSPH.py`; nothing consumed the old tags. The text below is the original finding.

`systems/compressibleMonaghan.py:36-39`:

```python
totalEnergies : torch.Tensor = constant(tags=('energy',), default=None)
entropies     : torch.Tensor = constant(tags=('soundSpeed',), default=None)   # <-- wrong tag
pressures     : torch.Tensor = constant(tags=('damping',), default=None)      # <-- wrong tag
soundspeeds   : torch.Tensor = constant(tags=('soundSpeed',), default=None)
```

`entropies` and `soundspeeds` carry the **same** tag, and `pressures` is tagged
`'damping'`. Any tag-based lookup (`find_tagged_field` / `get_tagged_attr`) resolves
`soundSpeed` ambiguously and `pressure` not at all. Harmless today because nothing does
tag lookup on these — but PESPH-A promotes `entropies` to a first-class *integrated*
field, so fix the tags as part of Phase 0 rather than inheriting the ambiguity.

### 2.2 What is genuinely new

Short list:

1. **Pressure-by-summation kernel** (G.1 / §1.2) — structurally a density-summation
   variant, low risk.
2. **The grad-h `f_ij` kernel** (G.7-G.9) — needs `∂n_i/∂h_i` *and* `∂P̄_i/∂h_i`. This is
   the risk item, see §2.3.
3. **Momentum kernel** (G.4) — pattern-match `modules/compSPH/accel.py`.
4. **Energy kernel** (G.5 for PESPH-E, the entropy source for PESPH-A).
5. **State plumbing** — promote `totalEnergies` (PESPH-E) or `entropies` (PESPH-A) to
   `integrated()`, add `dAdt` to the update for the latter.

### 2.3 The risk item: number-density-driven `h`, and the grad-h derivatives

Hopkins is explicit that PESPH sets the smoothing length from **particle number
density, not mass density** — "reducing errors when there are particles of different
masses in the same kernel" (2015, App. F2).

This repo's `evaluateOptimalSupport` is mass-density driven. Both backends return
`(densities, supports, adjacency, …)` and neither exposes the `h`-derivatives PESPH
needs (`modules/adaptiveSupport/optimalSupportMonaghan.py:124`,
`optimalSupportOwen.py:159`; `computeH(rho, m, targetNeighbors, dim)` is the Monaghan
relation).

So Phase 2 needs a number-density h-solve variant plus two new derivative
accumulations in a `computeOmega`-shaped kernel. **This is where the estimate can
slip.** A wrong grad-h term produces roughly 1% energy-conservation drift that presents
as a mysterious bug rather than an obvious failure, and it can survive every test that
does not specifically probe it. Budget a dedicated check against the Kidder isentropic
case (`caseUtils/compressible/kidder/`), which is exactly the case sensitive to it, and
run that gate after the conservative force and closure exist (section 0.4).

---

## 3. Design decisions

### 3.1 One scheme, one switch

PESPH-E and PESPH-A differ in their authoritative thermodynamic variable (`E` vs
`A`), pressure-summation weight, matching weighted derivative, force coefficient,
and thermodynamic/heating closure (see the 2026-10-22 audit, section 0).
Everything else — momentum kernel, viscosity, conductivity, grad-h machinery, h-solve,
orchestration, config, registration — is shared. Build one `schemes/compressible/pesph.py` with a
`PESPHVariant` switch on the config. Two scheme files would duplicate four warp kernels
for no gain.

### 3.2 PESPH-A is the better first target

Not merely because it is less work. Three properties coincide only in this variant:

- **It is a clean `f(state) -> update`.** `DA/Dt = 0` adiabatically plus a viscous
  source. No compatible-energy step-map, so no `ap_ij`/`av_ij`/`f_ij` pairwise arrays,
  no `compSPH_deltaU_multistep`. It still needs a PESPH state/system lifecycle adapter
  to preserve A and refresh accepted-state thermodynamics (section 0.2); compatibility
  with registered integrators must be tested.
- **It is separable** in the Hamiltonian sense — `A` is the Lagrangian constant that
  keeps the SPH potential a pure function of position. See
  `../warpSPHIntegrators/SPLITTING_PLAN.md` §2.3.1.
- **It is variational**, so it is genuinely symplectic under a suitable integrator —
  unlike CRKSPH (§4).

PESPH-E, by contrast, is the one scheme in the comparison set that **cannot** be made
separable: because it integrates total energy, `u_i = E_i/m_i − v_i²/2` is
velocity-dependent by definition, and `P̄_i` (G.1) is a kernel sum over the neighbours'
`u_j`. The conservative force therefore depends on velocity before any viscosity enters.
Build it as a baseline — it is what GIZMO and FIRE run — but do not expect it to feed
the integrator work.

### 3.3 Known weakness of PESPH-A, stated up front

Pressure-entropy handles **non-adiabatic** physics awkwardly: heating must be converted
back into entropy, and the `A ↔ u` round-trip degrades with strong shocks plus
conductivity. This is precisely why Hopkins' own GIZMO implementation uses
pressure-*energy*. Guard `A > 0` explicitly (the `A^(1/γ)` summation is undefined
otherwise) and expect PESPH-E to be the better choice on shock-dominated problems. The
two variants are complementary, which is a second argument for the switch design.

---

## 4. CRKSPH: observations and concerns from the same reading

Recorded here because they came out of the same pass over the paper and the code, and
because they change how `crkSPH.py` should be described — not because they require
action now.

### 4.1 CRKSPH is not variational, and the repo should not imply it is

Appendix B is explicit. Replacing `W` with the reproducing kernel `W^R` breaks
`∂_α W_ij = −∂_α W_ji`, so the usual Lagrangian derivation of a conservative momentum
equation does not work. Eq. (64) is instead obtained by **conservative differencing**
(MLSPH-style integration by parts), which buys exact pairwise antisymmetry — and hence
machine-precision linear momentum conservation — at the cost of a consistency error that
vanishes only for uniformly spaced particles.

The consequence worth recording: **there is no potential `V(q)` behind CRKSPH's
pressure force.** A kick substep `(q,v) ↦ (q, v + τF(q))` has Jacobian
`[[I,0],[τ ∂F/∂q, I]]`, which is always volume-preserving but is *symplectic* only when
`∂F/∂q` is symmetric, i.e. only when `F` is a gradient. So CRKSPH is order-preserving and
volume-preserving under a symplectic-style integrator, but has **no shadow Hamiltonian**
and no bounded-energy-error guarantee. compSPH, being Lagrangian-derived (its `Ω` terms
in Eq. E.5 *are* the grad-h corrections of the variational formulation), does have one.

### 4.2 The compatible energy update is not an ODE right-hand side

Frontiere Eqs. (65)-(68), implemented here as `modules/compSPH/balance.py` +
`modules/compSPH/multistep.py` + the `finalize` override at `systems/compSPH.py:118`:

```
u_i(t+Δt) = u_i(t) + Σ_j Δu_ij Δt
Δu_ij = (f_ij/2)[v_j(t) + v_j(t+Δt) − v_i(t) − v_i(t+Δt)]·(Dv_ij/Dt)
```

This is a **step-level algebraic map**, explicitly `Δt`-dependent, implicit in
`v(t+Δt)`, and tableau-aware (`compSPH_deltaU_multistep` consumes `butcherTerms` and the
per-stage pairwise arrays). It is outside the `f(state) -> update` contract that
`warpSPHIntegrators` is built on, which is why `compSPH`/`crkSPH` need a `finalize` hook
where PESPH-A will not.

Two concerns follow:

- **`f_ij` (Eq. 67) is piecewise constant.** It branches on `sign(Δu_ij)` and
  `sign(s_i − s_j)`, so it is non-differentiable. Anything that differentiates through
  the step — the gradcheck scripts, any future implicit solve — will be wrong at the
  branch points. `f_ij` should be frozen (evaluated once, held) rather than re-derived
  inside an iteration.
- **`Δt`-dependence blocks operator splitting.** Sub-flows run at different effective
  step sizes, so a `Δt`-dependent update has no consistent meaning across them.

`../warpSPHIntegrators/SPLITTING_PLAN.md` §2.7.1 gives a substep-local rewrite that
removes the pairwise storage entirely and restores exact conservation per substep. It is
not needed for PESPH-A and is not scoped here — noted so the connection is not lost.

### 4.3 Memory

`ap_ij` / `av_ij` / `f_ij` are `O(N · neighbours)` arrays carried across the whole step.
Frontiere flags the alternative — recomputing the pairwise accelerations — as possibly
preferable "on architectures such as GPU accelerated machines with limited memory and
significant FLOPs to burn," which is the Warp target. Worth revisiting independently of
this plan.

---

## 5. Implementation phases

### Phase 0 — groundwork (~1 day) — **DONE 2026-10-09**

- ~~Wire `modules/eos/gas.py` into `eos/__init__.py`.~~ **Not needed (2026-10-09 finding):** `gas.py` is a dead duplicate;
  the live `idealGasEOS(A, u, P, rho, gamma)` (`modules/eos/idealGas.py`, used by every case builder) already converts
  `A ↔ (u, P, c_s)` in every direction. Round trip tested (`tests/test_numberDensitySupport.py`, float64, 1e-12).
  `gas.py` left in place; deleting it is the user's call.
- Fix the `entropies` / `pressures` tags (§2.1) — done 2026-09-30 (AV_PLAN S2).
- `CompressibleSPHScheme.PESPH = 5` and `PESPHVariant.{PressureEnergy, PressureEntropy}` added; `buildScheme` raises
  `NotImplementedError` for PESPH until Phase 4.

**Gate.** Existing schemes unchanged; full suite green. `idealGas(..., EOSSource.specificEntropy)`
round-trips `A → (u,P,c_s) → A` to roundoff.

### Phase 1 — pressure by summation — implemented 2026-10-22

- `modules/pesph/pressure.py`: `P̄_i` for both variants (G.1 and the `A^(1/γ)` form).
- Number density `n_i` — reuse the CRK `V_i⁻¹` kernel shape.

**Gate.** On a uniform lattice with uniform `u` (resp. `A`), `P̄_i` matches the analytic
EOS pressure to kernel-normalization accuracy; the two variants agree on a state where
`A = P/ρ^γ` holds exactly.

### Phase 2 — grad-h and the number-density h-solve — kernels implemented; Kidder open

- [x] Number-density-driven `evaluateOptimalSupport` variant — **done 2026-10-09**: `AdaptiveSupportScheme.NumberDensity`
  = the Monaghan Newton iteration with every mass set to 1, so `rho` is `n_i = Σ_j W_ij` and the constraint is
  `h = η n^(-1/d)`; the existing `computeOmegaWarp` is then exactly the Newton derivative (quadratic convergence
  observed, residual 1e-2 → 2e-4 → 0 in three iterations on a jittered 2D lattice). Equal masses reproduce the
  mass-density solve to 1e-4; unequal masses leave `h` and `n` untouched (h differs from the mass solve by up to
  33 % on masses spread 1-4x). Returns `rho_i = m_i n_i` and the per-iteration `n_i`. Finding: **Owen's solver was
  already mass-independent** (its statistic is a kernel sum without masses) but its fixed point is a lattice-calibrated
  gradient statistic, not `n h^d = const`, so it would not be consistent with G.8/G.9; hence the Newton route.
  Gather mode (`n_i = Σ W(r_ij, h_i)`, as G.2/G.3) is what the PESPH scheme will request.
- `∂n_i/∂h_i` (G.8) and `∂P̄_i/∂h_i` (G.9) in a `computeOmega`-shaped kernel.
- Assemble `f_ij` (G.7) — under its new non-colliding name (§1.3).

**Gate.** Finite-difference `∂n/∂h` and `∂P̄/∂h` against the analytic kernels to 1e-8 on
a scattered state. **After Phase 3's conservative force and closure are runnable,
the Kidder isentropic case** (§2.3): energy conservation to the level Frontiere
reports, not merely "no blow-up." The derivative gate precedes Phase 3; the Kidder
gate follows it — a wrong `f_ij` is the failure mode that hides.

### Phase 3 — momentum and energy kernels — E implemented; dissipative A guarded

- G.4 momentum, `q_acc` viscosity (G.6) under the existing Cullen-Dehnen switch.
- PESPH-E: G.5 total-energy rate; `totalEnergies` is already `integrated('dEdt')`
  (2026-10-22 audit). Make E authoritative and derive u at every stage.
- PESPH-A: integrated A with its own update contract; `dAdt=0` is implemented.
  Conservative irreversible entropy heating remains open (see measured defect above).
- Artificial conductivity via the existing `wp_conductivity.py`.

**Gate.** Linear momentum conserved to machine precision (pairwise antisymmetry). Total
energy conserved to integrator tolerance on an adiabatic case. Acoustic wave reaches the
expected convergence order.

### Phase 4 — scheme assembly and registration — implemented 2026-10-22; strict CUDA Rosswog restart gate open

- `schemes/compressible/pesph.py` with the variant switch; `configurations/pesph.py`;
  `SchemeBundle.configured` + dict round-trip in `schemes/builder.py`; runner,
  Sod/linear-wave initialization, timestep, diagnostics and restart adapters.

**Gate.** Round-trips through `compressibleConfigToDict`/`dictToCompressibleConfig`.
Runs end to end on sod with a stock integrator.

### Phase 5 — validation against the paper — comparison run; acceptance gates open

2026-10-22: the requested 19-case CRK/E/A matrix and eight separate diagnostic rows
have finished; see section 0's completed comparison. Several acceptance gates fail.
This is a repository case battery at recorded resolutions/native presets, not exact
paper reproduction. Further comparisons below remain future acceptance work.

Run Frontiere's own battery, already present in `caseUtils/compressible/`: sod, sedov,
noh, kidder, gresho, yeeVortex, kelvinHelmholtz, rayleighTaylor, triplePoint,
hydrostatic, blob. Compare against the published figures, and against this repo's
`monaghan`/`compSPH`/`crkSPH` on the same cases.

**Gate.**
- Kidder: PESPH-A holds entropy to the isentropic solution better than `monaghan` —
  the property the variant exists for.
- Hydrostatic box: PESPH shows the contact-discontinuity behaviour of Frontiere Fig. B.2
  (PESPH degrades *less* than CRKSPH there — a discriminating check that the pressure
  formulation is actually doing its job).
- KH / blob: the mixing behaviour PESPH is known for, visibly better than `monaghan`.
- Sedov / Noh: shock capture no worse than `compSPH`. If PESPH-A is visibly worse here,
  that is §3.3 showing up, not a bug — record it and prefer PESPH-E for those cases.

> Run validation cases with `--video` (vispy) — see the note at the top of
> `BOUNDARY_DENSITY_PLAN.md`. A live window is the only way to see *where* a compressible
> scheme fails, not just *that* it did.

**Original estimate: ~3-4 weeks for both variants.** Superseded by section 0.4's
2026-10-22 sizing and integration scope.

---

## 6. Sequencing: why this comes before the integrator work

`../warpSPHIntegrators/SPLITTING_PLAN.md` proposes conservative/dissipative operator
splitting, and its Phase A names **an entropy-carrying, variational, velocity-independent
scheme** as the prototype target. PESPH-A is that scheme. But the dependency runs one
way, and this plan should land first, for four reasons:

1. **It has no new integrator dependency.** PESPH-A is a plain `f(state) -> update`;
   establish the correct E/A state adapters and test the stock explicit and
   component-update paths before extending that claim to implicit schemes.
   Nothing here waits on the split.
2. **Debugging isolation.** If PESPH has bugs, find them under RK4, where the integrator
   is not in question. Bringing up a new discretization and a new integrator
   simultaneously means every discrepancy has two candidate causes. This is the main
   practical argument.
3. **It produces the measured baseline the split is judged against.** Order, energy
   drift, and cost per case under ordinary integrators, before splitting is introduced —
   so step 2's benefit is *measured* rather than asserted. That is the standard the rest
   of `IMPLICIT_ROADMAP.md` holds itself to.
4. **It converts a hypothetical gate into a real one.** `SPLITTING_PLAN.md`'s
   separability certificate (`classify_separable`) currently has only toy problems to be
   right about. PESPH-A gives it a real SPH state — and, usefully, PESPH-E gives it a
   real *negative* case, since the two differ only in the thermodynamic variable yet land
   on opposite sides of the separability test.

So: **PESPH first (this plan), splitting second** (`SPLITTING_PLAN.md` Phases A-B wrap
around the finished scheme). The one thing worth doing early on the integrator side is
`classify_separable`, since PESPH-E vs PESPH-A is its sharpest test case and it costs
little.

---

## 7. Outlook: MFM, and the move from "an SPH code" to a meshless solver

> **Status 2026-10-11: no longer an outlook.** The MFM / MFV backend exists in warpSPHCore (`warpSPHCore.mfm`, torch reference
> plus warp kernels, GIZMO-parity Riemann chain) and its migration into the frontend is planned in
> [COMPRESSIBLE_SPH_PLAN.md](COMPRESSIBLE_SPH_PLAN.md) Track E. This section is kept as the rationale. Out of date: the §7.3 table
> row "Riemann solver ❌" (the frontend has `modules/riemann`, from AV_PLAN Phase 7b, and the core has its own HLLC chain), and the
> "not a commitment" framing in the Status section.

This is an **orthogonal axis** to everything above and to `SPLITTING_PLAN.md`: those are
about the *timestepper* and the *thermodynamic discretization*; this is about the
**spatial discretization substrate**. Recorded here as a direction rather than a plan,
because PESPH is the natural last scheme to add before the question "should this be an
SPH code at all?" becomes worth asking.

### 7.1 Where the field actually is (2026)

Three families, and the 2015-era expectation that one would win has not materialized:

- **Moving mesh (AREPO)** — still the accuracy reference for cosmology (IllustrisTNG,
  Auriga, TNG-Cluster), extended to GR for neutron-star mergers around 2024.
- **Meshless finite mass / volume (MFM / MFV)** — the adoption winner since Hopkins
  (2015). Spread well beyond GIZMO: SWIFT implements both, OpenGadget3 added MFM in 2023.
- **Modern high-order SPH** — did not die, improved substantially. MAGMA2 (Rosswog 2020)
  with matrix-inversion gradients and slope-limited reconstruction; REMIX (2024) for
  mixing; SPHENIX in SWIFT. Cost is roughly 1.5-2.5x bare SPH, while MFM/MFV land
  comparable to or slightly faster than bare SPH.

Adoption specifics worth pinning down, because they are easy to misremember: **FIRE-1
used P-SPH; FIRE-2 and FIRE-3 both use MFM**, not MFV. The reason is not accuracy —
MFM forbids mass flux between cells, so a particle stays a fixed parcel of mass, which is
what keeps metal tracking, stellar-population bookkeeping and particle IDs tractable
under galaxy-formation subgrid physics. MFV's mass fluxes make it behave more like a
moving mesh and break that. **SPHERAL still develops CRKSPH** (releases through 2025.12.0)
and has added FSISPH for fluid-structure interaction — a trajectory pointing at
multi-material and solid coupling, not cosmology.

The intellectual shift worth noting: the 2010-2015 "SPH is broken" framing drove the
migration, but the specific failures (surface tension at contact discontinuities,
suppressed mixing, E0 errors) were largely diagnosed and fixed *within* SPH — pressure-based
formulations like PESPH being a substantial part of that — while MFM/MFV turned out to
have their own pathologies (particle disorder, slope-limiter sensitivity). The 2026
position is that discretization choice matters **less** than it appeared in 2015, and the
real differentiators are the subgrid physics you need, the conservation properties you
cannot give up, and cost.

### 7.2 The structural finding: CRKSPH and MFM share a substrate

This is the part that makes MFM a much shorter step for *this* codebase than a cold
estimate would suggest.

Frontiere et al. derive CRKSPH by assuming a generic interpolant `ψ` subject to

```
Σ_j ψ_j ≡ 1   ⇒   Σ_j ∇ψ_j = 0                                    (Eq. 24)
```

— that is, **a partition of unity**, satisfied in their case by `ψ_j = V_j W^R_ij`. That
is precisely the object MFM is built on. Carrying the derivation through gives

```
m_i DU_i/Dt = ½ Σ_j (F_i + F_j) (V_i ∂_α ψ_j − V_j ∂_α ψ_i)        (Eq. 35)
```

and MFM's update is

```
dU_i/dt = −Σ_j F̃_ij · A_ij ,    A_ij ~ V_i ∇ψ_j(x_i) − V_j ∇ψ_i(x_j)
```

**The effective face vector is the same object.** The difference is entirely in what sits
between: CRKSPH uses the *central average* `(F_i + F_j)/2`, MFM solves a **Riemann
problem** across the face for an upwind flux `F̃_ij`. Stated compactly:

> MFM is CRKSPH's conservative differencing with the flux upgraded from averaged to
> Godunov.

This is a structural correspondence, not an equivalence — the two use different `ψ`
(Hopkins follows Lanson & Vila; CRKSPH uses the RK-corrected kernel) and MFM adds
slope-limited reconstruction of the left/right states to the face. But it means the
face-construction machinery is not new work here.

### 7.3 What the repo already has toward MFM

| MFM ingredient | Status |
|---|---|
| Partition-of-unity interpolant with `Σψ = 1` | ✅ CRK's `ψ_j = V_j W^R_ij` (`modules/crk/`) |
| Effective face `V_i ∂ψ_j − V_j ∂ψ_i` | ✅ implemented as the symmetrized `gradw_ij · V_i V_j` in `modules/crk/accel.py` |
| Slope limiters | ✅ van Leer + eta limiter in `modules/crk/limiter.py` |
| Linearly-corrected gradient estimates | ✅ `M_i^{αβ}` (Frontiere E.6), used by compSPH |
| Neighbour/adjacency, adaptive support, periodicity | ✅ mature |
| **Riemann solver (HLLC in the face frame)** | ❌ **the genuinely missing piece** |
| Reconstruction of left/right states to the face | ⚠️ partial — CRK already reconstructs pair velocities (Eq. 71) with the same limiters |

So the missing core is a Riemann solver plus the boost into and out of the face frame,
not a new discretization substrate.

### 7.4 The architectural argument, and the hard caveat

**The argument for doing it:** MFM turns warpSPH from a code that implements SPH schemes
into a code that implements *meshless conservative discretizations*, of which SPH is one
family. The `ψ`/effective-face layer becomes a substrate that can host both, and the
module structure (`modules/{momentum,pressure,density,internalEnergy,dissipation}`),
currently kernel-sum-centric, would need to grow a flux-based sibling. That is a real
generalization of capability rather than another scheme in the registry.

**The caveat, and it is hard:** MFM is **permanently outside** the geometric-integration
work in `../warpSPHIntegrators/SPLITTING_PLAN.md`. The force comes from solving a Riemann
problem using left/right states that include velocity, so `F_cons` depends on `v` by
construction, there is no potential behind it, and there is no Hamiltonian structure at
all — not separable, not symplectic, not conformally symplectic. Every result in that
plan applies to the SPH family only.

These are therefore **two distinct bets**, and they do not compose:

| Bet | Scheme branch | What pays off |
|---|---|---|
| Geometric integration | PESPH-A / compSPH-entropy | Separable Hamiltonian, symplectic composition, viscous-CFL relief, conformal-symplectic diagnostics — all of `SPLITTING_PLAN.md` |
| Astro fidelity / generality | MFM | State of the art, broad adoption, Riemann-quality shock capture — and the integrator work becomes irrelevant to it |

Worth choosing deliberately rather than drifting into MFM and finding the splitting work
stranded. Both are defensible; doing PESPH first (§6) keeps the option open either way,
since PESPH is a strong baseline under both bets.

### 7.5 The shared prerequisite worth doing regardless

**MAGMA2-style gradient estimation** — integral-approximation / matrix-inversion gradients
rather than plain kernel gradients — improves `monaghan`, `compSPH`, `crkSPH` and PESPH
across the board without adding a scheme, and is *also* a prerequisite for MFM (which
needs accurate gradients for its reconstruction step). It is independent of both bets
above, cheaper than either, and the natural thing to do after PESPH lands regardless of
which direction is chosen next. `modules/crk/` already has the matrix machinery
(`M_i^{αβ}`, gradient renormalization `L_i`), so this is an extension rather than a build.

---

## Relationship to `AV_PLAN.md`

> **2026-10-11:** `AV_PLAN.md` is archived at [`docs/historic_plans/AV_PLAN.md`](docs/historic_plans/AV_PLAN.md); its open items live
> in [COMPRESSIBLE_SPH_PLAN.md](COMPRESSIBLE_SPH_PLAN.md) Track A. The bake-off ran 2026-10-09 and the default AV configuration
> is decided (Rosswog 2020 + limited reconstruction + coupled β); the M8 *gate* below was waived as minor on 2026-10-09 (user).

[`AV_PLAN.md`](docs/historic_plans/AV_PLAN.md) is the artificial-dissipation roadmap, and this plan is
its Phase 8. Three concrete couplings:

- **This plan is gated on AV_PLAN's M8** (the AV bake-off). The point of the gate
  is that PESPH is then evaluated as `old SPH + known AV` vs `new PE-SPH + same
  AV`, so the hydrodynamic change is isolated — §8.5 of that plan's predecessor
  sketch, made enforceable.
- **§8's dissipation comes from AV_PLAN's Phase-1 `DissipationOperator`,
  unchanged.** PESPH does not port a new AV; it consumes the M8 recommended
  configuration.
- **The §2.1 state-tag bugs below are fixed in AV_PLAN Phase 1** (its §2.5),
  together with the `limitXi` duplicate and the `correctXi`/Ξ naming collision.
  Mark them closed here when that lands.

AV_PLAN's Phase 6 work (Cullen & Dehnen 2010 wired into `schemes/compressible/monaghan.py`,
Read & Hayfield 2012 added) is complete as of 2026-09-19, which satisfies the §2
audit row that assumes a validated C&D switch.

---

## Status

**Historical status, superseded by the 2026-10-22 checkpoints above:** Phase 0 and Phase 2's number-density solve done 2026-10-09 (user: the AV gate is minor; MFM/GIZMO parity is the longer-term goal and is tracked elsewhere; MHD is only a thought, not planned). The next steps at that point were pressure summation and grad-h. Plan originally written 2026-09-15 from Frontiere et al. (2017) Appendix G and
Hopkins (2015) Appendix F2, cross-checked against the existing `compSPH`/`crkSPH`
implementations. The §2 audit and the §2.1 tag bugs are verified against the code as of
this date; the §5 estimates are not yet tested against any implementation work.

§7 is an outlook, not a commitment. Its adoption claims (FIRE-2/3 on MFM, SPHERAL's
continued CRKSPH + FSISPH development, MFM in SWIFT and OpenGadget3) were verified
2026-09-15; the §7.1 synthesis about the SPH/MFM gap narrowing is a reading of the
literature rather than a citable consensus. The §7.2 correspondence between Frontiere
Eq. (35) and the MFM face vector is checked against the paper directly (Eqs. 24, 35) and
against `modules/crk/accel.py`.

### Reading

- Hopkins (2015), MNRAS 450:53 — MFM/MFV derivation, `../warpSPHIntegrators/literature/stv195.pdf`
- Rosswog (2020), MNRAS 498:4230 — MAGMA2, the modern high-order SPH reference
- Rosswog (2026), *SPH methods in the modelling of compact objects*, arXiv:2607.14828 —
  current review, the place to check §7.1's framing
- Groth et al. (2023), MNRAS 526:616 — MFM in OpenGadget3, a worked port into an
  existing SPH code, i.e. the closest thing to a template for §7

### Support correction results — 2026-10-22

Separate support-only GPU root: `scripts/out_pesph_supportfix_20261022/`.
All six rows reached their endpoint with positive accepted thermal fields.
Raw Noh E/A both reach t=.6: E energy change -1.53e-14, A -3.9386%;
Raw shearing Noh E/A also reach .6: E -1.34e-14, A -.2183%. Raw shear profiles
are poor (plateau diagnostic about 65% below the native reference); Raw is a
control, not a blanket remedy. Noh E's plateau error is -.1210%, A +3.8692%.
The original CRK Noh saved snapshots also have negative-u seam particles
(min saved -2.2068e-4 at x=-.98027,t=.007854), consistent with the seam
interpretation and stricter PESPH rejection rather than an isolated central-shock verdict.

Native Limited KH after only the support correction, N16384,t=2.5:
E wall 795.63→340.62 s, median integrator 144.927→60.960 ms; mode
.150116→.150533 (+.278%), KE retention .923734→.924105.
A wall 745.30→341.39 s, median 139.288→60.928 ms; mode .153899→.153929,
KE retention .927837→.927331. Same tolerance and stage semantics, no preset
change. Residual cost requires profiling rather than relaxed accuracy.

Experimental MonotoneLimited implemented as an additional shared pair policy,
not the default. Selected CPU support/reconstruction/PESPH integration suite:
53 passed; explicit float64 new-policy conservation/local-heat tests: 2 passed.
The default native Limited path remains unchanged. Separate roots prepared:
`scripts/out_pesph_benchmark_20261022/` (four 256-step matched-size native
KH timing probes and three 8-step call audits; startup/call-count evidence only),
`scripts/out_pesph_monotone_20261022/` (10 full Noh/shearing Noh/CFL/Sod/KH
rows at recorded settings). Both queues are sequential, video recorded, and
wait for any new incompressible matrix work before beginning.

Startup call audit (six measured steps after two warmup steps, dispatch overhead;
use counts rather than audit timings): native PESPH-E 909 Torch ops, 71 Warp
launches and 18 host syncs per integrator step, versus CRK 418/23/2.5 and native
Monaghan 421/24/3.83. PESPH support solves alone use ~37 launches and 13 syncs
per step (three refreshes; repeated convergence/list-validity reads). Matched
256-step KH startups, median after 32 warmup steps: native Monaghan/Rosswog/Limited
31.03 ms, CRK 18.72 ms, E 56.72 ms, A 56.81 ms. This modern Monaghan preset is
not the user's classic raw/constant-AV baseline, so prepare that separately in
`scripts/out_pesph_classic_benchmark_20261022/` (two recorded timing/audit probes).
Further support/list batching and launch reduction are justified by these counts;
do not remove accepted-state closure or reuse stale RK derivatives to gain speed.

Experimental-policy endpoint checkpoints (before KH finishes): both Noh variants
complete at CFL .15 and .075; both shearing Noh variants complete at .15. E stays
positive and energy-conservative. Noh E plateau errors are -1.847e-5 (.15) and
+6.237e-5 (.075); particle density L1 is still a separate whole-profile metric.
A remains nonconservative (~4.94% Noh energy loss), including at half CFL; positivity
does not lift its heating guard. MonotoneLimited shearing-Noh plateau errors remain
about -44.8% (E) / -45.3% (A) at N64 versus Raw ~-65% and original CRK ~-51.8%.
These are accuracy warnings, not acceptance based only on reaching t=.6.

Record one separate existing-option control before interpreting that shear error:
`scripts/out_pesph_shear_gradient_20261022/`, E/A at identical N64/CFL .15 with only
`correctReconstructionGradient=True`. The saved native preset has this flag false;
the switch's corrected velocity gradient is a distinct setting. Keep initialization,
pressure force, limiter, conductivity and RK stages fixed. Compare plateau/front,
transverse-flow retention, energy/entropy/positivity and cost; do not silently change
the default or call it a published-preset fix. No resolution-convergence claim.

### Completed support/viscosity follow-up — 2026-10-22

[Follow-up report](.tmp/pesph/docs/validation/pesph_followup_2026-10-22/index.html).
All 18 full endpoint controls, six startup timings and five call audits completed;
two earlier corrected-gradient configuration failures remain separately preserved.
The runner now exposes the existing `correctReconstructionGradient` boolean through
its typed schema. Successful gradient controls use the fresh `_v2` root, without
altering the native default. The queues and provisional checkpoints above are now
superseded by these completed results.

Keep the scale-independent support convergence fix: the full native KH wall cost
falls 795.63→340.62 s for E and 745.30→341.39 s for A, with small changes in the
recorded flow diagnostics. Matched N16384 startup medians after 32 warmup steps:
classic raw/constant AV with conductivity explicitly off 19.51 ms; CRK 18.72 ms;
modern native Monaghan 31.03 ms; raw/constant AV retaining native conductivity
25.08 ms; PESPH E/A 56.72/56.81 ms. These are distinct saved presets and current
modular implementations, not a universal algorithmic ranking. Support/list batching,
within-stage reuse and launch/synchronization reduction remain worthwhile; preserve
fresh RK derivatives and accepted-state closure.

Keep `MonotoneLimited` opt-in. It bounds only the reconstructed normal jump against
the sampled jump, shared by viscosity force and conjugate heat. Noh E completes
positively at CFL .15 and .075 with energy at roundoff, density L1 .07705/.07559
and plateau relative errors -1.847e-5/+6.237e-5. A also stays positive but loses
4.94–4.95% energy at either CFL: its independent approximate heating remains guarded.
The native original Noh failures were at the periodic seam; original CRK also saved
negative-u seam states. Positive completion alone is not shock acceptance.

Monotone KH is more damped than support-fixed native Limited: E final mode
.15053→.13161 and KE retention .92410→.91450; A .15393→.14251 and
.92733→.92045. No exact growth reference is claimed. Shearing Noh profiles remain
poor (E plateau about 45% low); corrected reconstruction gradients improve this only
slightly (about 44% low). Transverse KE retention improves, but is not an accuracy
gate by itself. The 95th-percentile density-threshold position is a biased proxy:
an ideal uniform plateau already gives approximately -5%, so it is not literal
shock lag. Sod E/A density L1 remains .00770/.00664.

Verification logs are in the report: 53 selected CPU regressions; 61 float64 schema
checks; 44 operator checks plus two precision-dependent skips covered by the
successful float64 subprocess; four float64 GPU policy/schema checks. Counts overlap.
The GPU integration selection has 23 passes and the same two strict Rosswog E alpha0
continuation failures; tolerances are unchanged. Source fingerprints, exact original
source copies, failures, raw histories and recorded movies are retained. No primary
checkout merge. Remaining readiness work: shared conductivity multiplier/contact
source treatment, conservative A heating and blast initialization, shear refinement,
Yee bands, residual performance and strict Rosswog restart continuation.

### Next authorized phase — 2026-10-22

User: proceed with residual support/list overhead, shared conductivity/contact
treatment and conservative A heating. First isolate implementation-only support
work: honor the existing computeDensity=False contract, remove duplicate list
checks and validate speculative Newton iterates against their maximum support,
with a safe eager fallback when the incoming list is too short. Preserve every
fresh RK stage and accepted-state refresh. Compare against the recorded support-fixed
KH startup and full profiles before changing conductivity. Baseline files and hashes:
`.tmp/pesph/docs/validation/pesph_efficiency_2026-10-22/`. No A model promotion without the
published/reference derivation and a measured conservation/accuracy gate.

### Separate numerical-source candidates — 2026-10-22

Restore the documented shared conductivity multiplier, which currently sits after
an unconditional return. Keep its public default value; evaluate fresh controls.
Add opt-in Hopkins2015 conductivity (Hopkins 2015 Appendix F2 Eq. F18), using the
existing pair coefficient/mean-kernel path with its half-factor accounted for.
Cross-check: Spheral PSPH.cc lines 500–518 at 116c71f2e; alpha_C=.25, switched
alpha, linear pressure contrast and positive signal cutoff. Existing Price2008
uses sqrt pressure contrast and remains a separate control. No preset promotion.

Experimental A candidates are separately named: NonlocalPower uses the exact
positive discrete G=dU/dA (transpose gather via core scatter interpolation), with
dA=m*q/G. It matches continuous total source power and positive viscous heat
increases A; it does not enforce each particle's local du/dt or finite-step exact
conservation. Conductivity must explicitly be off until heat transport/entropy is
validated. This is a local experimental allocation derived from U(A), not the
published SWIFT source or finite-event feedback iteration. Borrow/Schaller/Bower
(2021), arXiv:2011.11641 §4 independently explains the nonlocal coupling; Spheral
PSPH evolves energy and supplies no entropy-source counterpart. Dense independent
autograd matches G to 4.6e-16 in the diagnostic fixture.

ThermalEnergy is a separate opt-in initialization that solves every prescribed
specific u target in A rather than globally rescaling injected energy. Its log
Jacobi map is a contraction with factor gamma-1 for 1<gamma<2; restrict that option
to this interval and reject failure to converge. Smoothed pressure is then an
outcome, unlike the historical PressureVolume policy. Check cold blast energy,
profiles and timestep sensitivity independently of the heating candidate. Both
historical initialization and the default dissipative-A guard remain unchanged.

### Completed source/efficiency phase — 2026-10-22

[Recorded report](.tmp/pesph/docs/validation/pesph_efficiency_2026-10-22/index.html): 57 attempts,
47 completed goals (39 full endpoints and eight intentional fixed-step probes),
eight preserved schema preflight rejections and two cold Sedov-E startup failures.
All 49 packaged movies are readable; all 57 histories match their raw bytes after
gzip decoding. Seven cohorts retain fingerprints and four exact evaluated source epochs.
The mixed full/fixed-step report classification is tested separately; short timing
probes are not claimed to reach the physical endpoint.

Implementation-only support reuse saves another 6.8–7.6% on full native KH
(E 340.62→317.33 s, A 341.39→315.34 s). The final audit is 867 Torch operations /
44 Warp launches / 8.83 counted host synchronizations versus 909 / 71 / 18.
Fresh stages and accepted closure remain. CPU/float32 time deferral was rejected
for switch-history rounding; the retained deferral is CUDA float64 only.

The shared conductivity multiplier is repaired: nominal default 0.5 now actually
scales the source, so its effective behavior changes from previously unscaled 1.
Opt-in Hopkins2015 at alphaC=.25 preserves the tested contact to roundoff through
t=1, but does not establish KH/shock superiority. The two conductivity KH timing
rows overlapped another user's GPU job and are explicitly excluded from isolated
performance claims. That job was left untouched and later launches queued.

NonlocalPower (conductivity-off only) cuts Noh A energy error from 4.057% to
0.02133%, then 0.005786% at half CFL, with positive accepted thermal states.
Density L1 changes .16190→.07836; all pressure/velocity errors remain in the report.
ThermalEnergy recovers Sedov's intended initial energy (.85633→.99999782).
At matched initialization, source drift is 2.794% local versus .1223% nonlocal,
then .03562% at half CFL. These are explicit experimental policies, not a published
SWIFT replacement. Spatial errors are mixed: coarse N48 CRK density L1 .37454,
A .48325/.47783, E .54362. Strong pressure/velocity scatter remains in the blast.
No spatial-convergence or general accuracy claim.

Conductivity-off Sedov E fails startup at CFL .15 and .075. The measured explicit
midpoint bound u_mid=u0-dt²|a|²/8 explains the intermediate-state violation;
quarter CFL .0375 completes at t=.480879 (target .480792), positive, with
5.57e-14 maximum relative energy error. No floor or rescale was introduced.

Focused checks pass: 59 CPU support/restart, 15 f64 source, 91 f32 source/schema
and adapter checks, 42 CUDA f64 source/integration, fresh eight-case restart repeat,
19 typed-policy and one actual loaded-config continuation, 45 shared dissipation,
and 25 final source/schema/evaluation-goal checks. Counts overlap. The previously
failing strict Rosswog E fixtures pass this revision twice; earlier failures stay
archived. This is fixture-level evidence, not universal bitwise continuation.

Keep dissipative A Guarded, historical PressureVolume initialization and the new
numerical policies opt-in. Open: heat-transport/entropy validation, shear accuracy,
wider paper/boundary bands, spatial convergence and residual cost. Work remains
isolated/uncommitted in codex/pesph, with no primary-checkout merge.

### Adiabatic Hamiltonian qualification — started 2026-10-22

User confirms separability for timestepper work is the primary A motivation. Check
fixed-A potential U=sum w*y^(gamma-1)/(gamma-1), exact force–potential consistency
with number-density adaptive supports, velocity independence/passive A and
canonical force-Jacobian symmetry. Use E at fixed authoritative total energy and
A with omitted grad-h as negative controls. Exercise the installed integrator's
classify_separable without treating unsupported AD as a successful certificate.
Then qualify stock fixed-step geometric integration and reversibility on an
adiabatic smooth case. Keep this investigation in codex/pesph; no external
integrator-repository edits, new dissipation models or default changes.

### Adiabatic Hamiltonian qualification — completed 2026-10-22

The user confirms that timestepper separability is A's primary purpose. The
[Hamiltonian report](.tmp/pesph/docs/validation/pesph_hamiltonian_2026-10-22/index.html)
qualifies the existing **adiabatic autonomous A operator**, with no production
solver equation or default changes and no external integrator edits.

For fixed masses and A, w=m*A^(1/gamma), y=W(q,h(q))*w, and
U=sum(w*y^(gamma-1))/(gamma-1). H=sum(|p|²/(2m))+U(q;A), p=m*v.
An independent dense cubic-kernel autograd derivative includes the implicit
constraint C_i=n_i*h_i^d-constant=0 via U_q-sum(U_hi/C_i,hi)*C_i,q.
Production m*a matches -dU/dq to 2.82e-15 relative error in six unequal-mass,
nonuniform 1D/2D/3D fixed/adaptive fixtures. Directional energy differences,
canonical force-Jacobian symmetry and warm-start independence pass. Removing
grad-h causes 12.5–27.9% force-gradient error and 1.0–7.3% projected skew.
These checks distinguish a gradient force from pairwise momentum conservation.

Direct A velocity perturbations give zero force change and dA/dt=0; varying dt
also leaves acceleration unchanged. E held at authoritative total energy is the
negative control and is velocity-dependent with evolving per-particle E.
Stock Velocity Verlet's finite-difference canonical (q,p) map defect is
3.4–3.8e-11; forward/back error is below 7e-18. Omitting grad-h still reverses
but gives canonical defect .00261: reversibility alone is not symplecticity.

Eleven recorded CPU/f64 smooth periodic N48 runs all finish: matched q/v/m/A/h,
Wendland4, A-perturbation 1e-3, support threshold 1e-12/64 iterations,
no dissipation/forcing, stock fresh-stage methods and audited fixed cadence.
Against refined RK4 dt=.000625 at t=2, Verlet order is 1.99991/1.99998 and
PEFRL 4.00244/4.03424. The coarser RK4 reference differs by 2.84e-13 in
phase-space RMSE; the conditional fourth-order Richardson estimate for the
fine reference is 1.89e-14. Over 20 crossings, Verlet max |dH/H| is
1.30e-11 (dt=.02) / 2.40e-12 (.01), PEFRL 1.78e-15 (.02). A remains bitwise
constant, states positive, and observed energy error oscillates within a finite
envelope. This is a small wave on a thermal background, not general nonlinear
or infinite-time stability. Wall/step costs and all 11 movies are in the report.

The initial cadence preflight caught maxDt=.01 capping intended dt=.02 rows;
preserve that cohort. All ranked rows reran separately with maxDt=.1 and an
inactive CFL cap. No frozen support/stage or detached force surrogate was used.
Default support tolerance 1e-5/16 iterations changes force by up to 3.39e-11
across tested warm starts; tight solves reduce the floor for geometric checks.
No claim applies to unconverged support roots or state-dependent adaptive dt.

**Mathematical/stock-stepper gate passes locally; automatic split admission stays
open.** classify_separable reports unsupported because torch.func.jvp reaches
wp.from_torch/data_ptr on a storage-less tensor in neighbour-grid construction.
Unsupported AD is not a certificate. The current system lacks composition_compatible
lifecycle support and a registered conservative/dissipative RHS capability pair.
The report contains a concrete timestepper handoff. Next: support that derivative
boundary honestly and declare a composition-aware lifecycle before dissipative
Strang/RKN use. Heat transport, bands, GPU/f32 geometric qualification and
state-dependent adaptive time remain separate. No primary-checkout merge.

Verification closeout: all fifteen focused CPU/f64 Hamiltonian/variational-force checks pass (eleven new plus four existing). Eleven movies/histories, fixed cadence, identical ICs, stable dependency fingerprints and relative report links audited; production source hashes match the preceding phase exactly.

### Numerical closeout — 2026-10-22, no merge authorized

User explicitly cancels merging: complete numerical work, document limitations,
then stop in the worktree. Focus on concrete numerical-contract defects and
validation rather than inventing a new dissipative method: enforce the support
constraint required by grad-h/Hamiltonian derivation, prevent hydrostatic sampling
from clobbering PESPH's stage iteration budget, extend adaptive viscous energy/
entropy checks, and retain the measured cold-state midpoint/CFL limitation.
Thermal transport for A, shear accuracy and new timestepper interfaces remain
separately identified research/interface limitations. No primary checkout edits,
commit, merge, branch advancement or worktree cleanup.

### Numerical closeout — completed 2026-10-22; stop unmerged

User explicitly requested numerical work and limitations documentation, then
stop without merging. [Closeout report](.tmp/pesph/docs/validation/pesph_numerical_closeout_2026-10-22/index.html).

Two concrete fixes completed. (1) Hydrostatic sampling's legacy 16-then-1 support
iteration assignments now operate on a temporary PESPH configuration; the actual
stage iteration budget is preserved. Other schemes retain their legacy convention.
The previous evaluation wrapper already restored the configured budget, so those
archived physical results remain valid. (2) Every adaptive PESPH closure now checks
that its number-density support constraint converged, rather than accepting an
exhausted Newton solve. The check reuses the closing pressure sum, removes its
constant lattice calibration to match the shared solver's raw-kernel target, and
uses a 32-epsilon precision floor. No extra neighbour sum or force equation change.
Under-budget solves fail explicitly; increasing iterations cannot repair an invalid
neighbourhood or unsupported state. Exact before/evaluated source trees retained.

New independent adaptive-h viscous tests close total continuous kinetic+thermal
power and nonnegative entropy production for Raw/MonotoneLimited reconstruction
with None, Cullen-Dehnen, Rosswog, Sphenix and Wadsley switches. CPU f64: 181 pass,
one skip. CPU f32 shared selection: 272 pass, 12 skip, ten hard-coded CUDA tests
fail only because the GPU was hidden; all ten pass on the GPU rerun. CUDA f64:
49 pass, including source, guard and restart checks. Counts overlap. Existing
incompressible work was left untouched; no such GPU job was active at launch.

Four full recorded controls complete positively with converged support checks:
N32 contact A (adiabatic) and E (explicit Hopkins conductivity) stay at roundoff
pressure/speed/energy through t=1; Noh experimental NonlocalPower A reproduces
rho L1 .0783628 and max energy error .000213339 at t=.600014; N48 quarter-CFL
Sedov E reaches t=.480879 in 1163 steps, min u=9.99654e-9, energy error 5.57e-14,
rho L1 .543620. Sedov's GPU final fields/time match the previous qualified control.
Two slow CPU Sedov execution-cost attempts are preserved separately as interrupted,
not numerical failures or accuracy winners. Their display processes were cleaned
up. CPU costs overlap regression work and are not isolated performance claims.

A signed-heat source-allocation counterexample establishes the conductivity limit
more concretely: S=sum m log(A)/(gamma-1) has conjugate temperature
T_i=(gamma-1) A_i (dU/dA_i)/m_i. A connected pair's u and T order can differ;
one unit of u-ordered energy transfer has zero continuous total power but
entropy rate -.0127047. This is not a completed conduction-kernel failure. It
shows that the current continuous power allocation alone is insufficient to
qualify general thermal transport; NonlocalPower remains conductivity-off.

Numerical limitations retained explicitly: sound-signal CFL does not ensure
cold E midpoint positivity (the saved .15/.075 Sedov failures and .0375 success);
no automatic controller/floor/rescale introduced. Historical SwiftLocal A has
percent-level conservation defects. NonlocalPower has finite-step error and is
experimental. ThermalEnergy initialization is opt-in, only 1<gamma<2; historical
PressureVolume can underinject a blast. Native shock/shear failures, KH damping,
coarse Sedov scatter, spatial convergence and relative cost remain open accuracy
issues, not solved by conservation alone. Shared-gamma ideal gas/eager scope;
wall/mixed-EOS/graph and driven-band qualification are not promoted. Adiabatic
fixed-step Hamiltonian evidence does not license state-dependent adaptive dt.
Automatic AD/split capability/lifecycle work remains the separately documented
timestepper handoff, not part of this numerical closeout.

**Stop here.** No commit, branch advancement, primary-checkout write, merge or
worktree removal. Keep the working schemes, guarded defaults, raw evidence and
limitations in the isolated codex/pesph worktree. Any integration is a later,
explicitly authorized task.
