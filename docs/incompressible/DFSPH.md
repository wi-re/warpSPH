# DFSPH constraints, operators and integration

`dfsph` is the native DFSPH scheme, built on IISPH's rest-volume operators,
exact diagonal and static boundary coupling. Select it with `--scheme dfsph`.
`dfsphPredictor` and `dfsphShift` select the other native compositions. Legacy
runtime paths are retired; see [migration and policy differences](MIGRATION.md). See
[IISPH.md](IISPH.md) for the shared spatial API and boundary restrictions.

The native default now includes the reference weak-factor cutoff and the
selected 0.9ρ₀ sampling-recovery gate in all three DFSPH orderings. Matched
particle/analytic-wall tests recover bulk dynamics and remove the persistent
sparse surface layer. Pressure transients, packing, containment and broader
qualification remain open. See [Sampling recovery](SAMPLING_RECOVERY.md) for
the current rule, configuration, common initialization and curated evidence.

The investigations below retain their experiment-time defaults and failures,
including the period when the gate was opt-in. They are a chronological
record; the linked guide defines the current shared policy.

The velocity-divergence projection is independently responsible for pressure
response: it must remove compressive velocity modes while retaining
solenoidal motion, without the density stage. A small correction in a coupled
run is not a qualification of this property. The periodic Fourier-mode
regression exercises that standalone response, momentum conservation and
density/timestep units. Diagnostic TGV runs with density correction removed
isolate the density-stage distortion; they still accumulate density drift and
do not qualify a divergence-only production scheme. Tightening the divergence
threshold alone does not resolve the coupled signed-pressure failure.

## Configuration and state

`IncompressibleSPHConfig.dfsph.density` and `.divergence` hold independent
`DFSPHProjectionConfig` policies. Both default to relaxation .3 and a cap of
100 iterations. Density requires at least 2 iterations and defaults to .01
percent error; divergence requires at least 1 and defaults to .1 percent,
converted to a rate threshold by dividing by dt. Clamped solves use the
compression-only residual; signed solves use its absolute mean. The statistic
belongs to the input iterate, as in the reference implementation.

Static boundary representation, self/contact-Shepard pressure and gravity
extension use `schemeConfig.iisph`'s boundary settings. Its standalone IISPH
iteration policy does not control DFSPH. Periodic TGV and shear-wave cases
select signed pressure in both native DFSPH stages. Lattice relaxation uses
the same native density policy and one-time mass calibration as the timestep.

`DFSPHState` stores physical density pressure in `pressures`, physical divergence
pressure in `divergencePressures`, and independent `densityPressureImpulse`
(`p_rho*dt²`) and `divergencePressureImpulse` (`p_div*dt`) histories. These
declared state fields clone, finalize and checkpoint with the particle state.
Rigid-body particle-state reconstruction preserves them for both sampled
and empty-mask analytic bodies.
Warm starts use half the previous impulse divided by the current stage's time
factor; compression gating applies when pressure is clamped. There are no
borrowed numerical caps on a differently normalized pressure. `warmStart=False`
disables both seeds. Sound speeds retain their original meaning.

`deficiencyGuard=True` implements the literal reference rule for clamped
divergence: below 7 neighbours in 2D / 20 in 3D, suppress the seed and error
statistic, but retain the raw source and pressure update. Signed solves do not
apply that free-surface rule. Both projections additionally disable pressure
unknowns whose base rest-volume factor denominator is at most
`dfsph.weakFactorThreshold` (default1e-5, in inverse simulation length squared).
The cutoff is applied before density-ratio and timestep scaling, excludes the
contact-interpolation diagonal contribution, and zeroes seeds and every update
on disabled rows. The remaining operator/diagonal and residual statistic are
unchanged. Setting the threshold to0 restores the historical exact-zero-only
policy. Convert the threshold when changing coordinate units; it is not a
universal dimensionless constant.


### Historical pre-cutoff defect: deficient-neighbourhood pressure spikes

The 26 October consistency replay stops at 0.818580 s: fluid UID478 reaches
DF pressure 100,007,768 Pa, and maximum fluid speed reaches 135.864 m/s.
An observer replay now reproduces the initial/final tensors and physical/solver
histories bitwise and captures the exact failing stage. UID478 first enters
two wall samples at the kernel fringe, with no other fluid neighbour. Its
assembled diagonal is -2.564641e-11 (basis-column check agrees bitwise), and a
source of only -3.282894e-6/s generates pressure 1.000078e8 at relaxation .25.
Previous ten captured stages had exactly zero diagonal, source and pressure.

The explosive kick is transmitted through **contact-Shepard wall pressure**:
UID478 contributes contact pressure 12,524.997 Pa at UID119, producing a
136.791 m/s kick. Keeping only UID478 pressure reproduces that entire kick;
removing its contact interpolation makes its kick on UID119 exactly zero.
The fluid particles are outside each other's direct support. UID119 has no
other fluid neighbour either, but wall samples raise its count to31, so the
count-based deficiency guard does not flag it. This is not a stale-adjacency
or incorrectly assembled diagonal explanation.

The compression-only mean error hides the expansion kick. Forcing100 sweeps
reduces the reported error to0.0008348/s while increasing the maximum kick to
547.167 m/s. A returned-iterate compression statistic alone is insufficient.

The paper (Section3.2, after Eq.11) explicitly clamps the mass-weighted
factor denominator below1e-6. Pinned SPlisHSPlasH instead disables the inverse
factor when its rest-volume denominator is<=1e-5, in scalar and AVX paths.
The pre-cutoff native exact-zero-only validity test omitted this protection.
On the identical frozen stage, a translated paper floor reduces UID478
pressure to82.157 Pa and its remote kick to0.0001124 m/s; the reference cutoff
sets both to zero. The largest remaining correction is0.119771 m/s elsewhere.
At100 sweeps the floor still accumulates pressure8215.619 Pa and a0.01124 m/s
remote kick; the cutoff keeps this pressure and remote kick zero. A floor used
only in the Jacobi denominator changes the preconditioner, not the exact
system or its fully converged pressure mode. These frozen-stage results
establish remedy candidates, not qualified production regularization. Mass/rest-volume thresholds require density-unit
conversion, and both need explicit length/resolution scaling.

User requirement: isolated or deficient particles must not acquire explosive
pressure corrections or transmit explosive kicks. This remains a solver
robustness blocker for a wider comparison sweep. Next validate a bounded
weak-factor policy against sparse wall contacts, bulk compression and
hydrostatic support, with units/resolution checks. The user subsequently authorized the reference-style cutoff; it is now
implemented in both native projections. The full GPU FP32 same-IC replay
completes2.45s/5628steps without the pressure explosion (vmax7.097 m/s).
It is bitwise identical to the pre-cutoff run through0.81826003s. Integrated
kinetic energy is99.14% of Delta-SPH, but pressure error remains42.245%
versus28.209% Delta and32.705% Sun;up to4simultaneous wall penetrations remain.
142focused checks pass with12CUDA-only skips; input/runtime/video audit1/1
passes. See the [cutoff replay](../../.tmp/dfsph-cutoff-dambreak-2026-10-26/README.md).
The cutoff does not impose zero pressure on every deficient row. Exact
snapshots, reference code, counterfactuals,
video and assertions are in the
[failure investigation](../../.tmp/dfsph-deficiency-investigation-2026-10-26/README.md).


### Persistent layer / sampling recovery

Saved cutoff dam-break data show a separate DF free-surface defect: particles
can remain near their isolated C4 self-density0.178974935rho0 while travelling
above the bulk. UID1438 stays between0.178978 and0.179076rho0 over1.918–2.45s,
moving0.740m horizontally but only0.037m downward. Its endpoint CDpressure
and CDacceleration are zero; reconstructed DFpressure acceleration is47.315
m/s² upward, almost entirely from its own DFpressure. Its factor denominator
0.007871 is787times the cutoff, so this is not a disabled-factor drift problem.

Its nearest neighbour is3.385dx away (support4dx), and its non-self kernel
contribution is only0.0001015rho0 despite10 counted neighbours. Sparse paths
through other almost self-density particles transmit pressure to bulk across
3.6–3.9dx links. This can support several loose layers before ordinary spacing
is restored. The raw divergence constraint resists the positive C(v) required
to recover summation density as returning particles acquire neighbours. That
sampling recovery is distinct from physical compression of a fully sampled
fluid, but the present rate constraint treats both as compression to remove.

The frozen endpoint force attribution uses exact saved pressures/densities
and reconstructs pre-drift positions with FP32 roundoff. Selected rows have
zero wall gradient, so the demonstrated mechanism does not require wall
pressure interpolation. It supports the user's explanation of re-merging
failure; no corrected policy or sloshing force attribution is qualified yet.
The user-requested native dam break now also reproduces the symptom:
cubic2dx completes2.45s, while UID719 stays near its own0.454485rho0
self-density for0.53s and travels0.478m over the bulk. It has one other
fluid neighbour and positive DFpressure/upward force, with CDpressure0.
Selected native gap is1.91dx versus local3.89dx, both near their support edge.
The declared native cubic4dx control fails at0.007s during density projection
with unmodified.5relaxation. Inputs match; kernel/boundary/viscosity differences
remain explicit. See the [native comparison](../../.tmp/dfsph-native-dambreak-2026-10-26/README.md).

The next discriminating control is re-entry into a quiet free surface, with
sampling restoration and bulk divergence measured separately. No change to
mass, density fields or factor cutoff was made for this diagnostic. See
[surface-layer evidence](../../.tmp/dfsph-surface-layer-2026-10-26/README.md).

The .3 relaxation is a native DFSPH choice, not an alteration of standalone
IISPH. On a periodic lattice at n_h=4, independent dense C/F algebra and actual
kernel products give largest Jacobi eigenvalues 5.6700 (C2) and 4.2737 (C4),
so the stability limits `2/lambda_max` are .3527 and .4680. The reference .5
exceeds both limits. At n_h=2.57 the measured limits are .7827 and .9008.
These measurements motivate .3 for the default support 4; they do not prove
stability for arbitrary kernels, larger supports or boundary geometries.

## Reference identities

The applicable reference is SPlisHSPlasH, local commit
`eccce86155776f6ac52d5080b1f720a52bf29450`, particularly
[TimeStepDFSPH.cpp](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/eccce86155776f6ac52d5080b1f720a52bf29450/SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp)
and its header (both warm-start macros enabled). This is a pinned source audit,
not a claim about every released build. The scalar implementation is at lines
1106–1450; its AVX counterparts must also be considered when comparing unequal
particle volumes.

The published method is Bender & Koschier, *Divergence-Free SPH for
Incompressible and Viscous Fluids*, TVCG 2017, Sections 3.1–3.3, Algorithms
1–3, Eqs. 9–12. A local author PDF is
`literature/bender2017_divergence-free-sph-viscous.pdf`.
Section 3.3 explicitly permits combining IISPH's density solver with the
divergence solver. The authors' density solver allows reuse of their factor;
the two constraints do not require unrelated spatial discretizations.

## What the incompressibility term solves

Use the native IISPH convention: `V_i=m_i/rho0`, `s_i=rho_i/rho0`,
`q_i=p_i/rho0`, `g_ij=grad_i W_ij`. For static self-pressure walls, define

```
G_i       = sum_solid_b V_b g_ib  (or the solid gradient integral)
S_i       = sum_fluid_j V_j g_ij + G_i
C(v)_i    = sum_fluid_j V_j (v_i-v_j).g_ij + G_i.v_i
F(q)_i    = -S_i q_i/s_i^2 - sum_fluid_j V_j q_j/s_j^2 g_ij
A(q)      = C(F(q))
diag(A)_i = -(|S_i|^2 + V_i sum_fluid_j V_j |g_ij|^2)/s_i^2
```

`C` is the normalized density-change rate, positive for compression; it is
the negative of the approximate velocity divergence, with units 1/time.
Nonfluid pressure-acceleration rows are zero. Static walls contribute to
`S_i` but have no neighbour back-reaction square. Contact Shepard pressure
adds a linear term to `F` and a corresponding contribution to the exact
diagonal; it must be included in both, as native IISPH already does.

At fixed positions, density projection solves

```
v* = v + dt a_nonpressure
dt^2 A(q_rho) = 1-s-dt C(v*)
v_rho = v* + dt F(q_rho)
```

For the contact/gravity wall extension, native IISPH separates the known
acceleration `a_wall` from the linear map. Its density source also contains
`-dt^2 C(a_wall)`, and the final pressure acceleration includes `a_wall`.
This separation preserves a linear pressure operator.

Divergence projection instead solves

```
dt A(q_div) = -C(v_enter)
v_projected = v_enter + dt F(q_div)
```

There is no `1-s` term: this stage corrects the density-change rate, not
existing density error. These equalities describe the fixed-geometry systems;
nonnegative pressure and excluded rows turn the actual free-surface solve
into a constrained correction. The density and divergence pressure fields
are separate multipliers.

Particle regularization is a separate numerical role. Cornelis et al.,
[*An Optimized Source Term Formulation for Incompressible SPH*](https://www.animation.rwth-aachen.de/publication/0559/),
use velocity divergence for the physical pressure response, then use a
density-invariance solve to shift sample positions. Their Eq. 17 resamples
the velocity as `v(x+deltaX) = v(x) + grad(v) deltaX`; the position correction
does not become an additional `deltaX/dt` momentum impulse. The local VD+PS
route implements that shift/resample distinction in
`modules/shifting/projection.py`, with its own legacy sources and case gates.
Native DFSPH currently applies its density correction to velocity instead.
Converting that stage into a position shift would be a VD+PS-style variant
requiring independent qualification. Directional particle stretching also
occurs under exact Lagrangian advection, so sampling anisotropy and error in
the physical velocity field must be measured separately.

This explains why native IISPH is a suitable basis: reuse `C`, `F`, their
boundary terms, the exact diagonal, and Core's stationary driver; change the
source, timestep scaling and solve policy explicitly. Calling the current
`solveIISPH` twice would retain the density source, wall forcing and stopping
policy in the divergence stage and would implement the wrong constraint.

## How the SPlisHSPlasH carrier relates to native IISPH

For one phase, interpret the effective carrier used by the reference force as
`chi_i=q_i/s_i^2`. This specifies the algebraic mapping; upstream names it
`pressure_rho2`, which should not be copied as a declaration of physical units.
Its force is `-sum_fluid V_j(chi_i+chi_j)g_ij-G_i chi_i`.
Thus it has the same self-wall force as native IISPH after changing unknowns.
For equal fluid rest volumes, its positive stiffness denominator

```
K_i = |S_i|^2 + sum_fluid_j |V_j g_ij|^2
```

equals `-diag(A)_i*s_i^2`. For unequal rest volumes, the native exact term
`V_i sum_j V_j|g_ij|^2` differs from `sum_j V_j^2|g_ij|^2`. The reference
scalar continuity and operator routines also explicitly assume equal fluid
volumes, multiplying a fluid sum by `V_i`; the AVX routines use `V_j`.
Equal-volume, single-phase parity is the appropriate first comparison.
Contact pressure interpolation and analytic hydrostatic closures are separate
boundary models, not part of this equivalence. The reference's small-value
force cutoffs also lie outside the exact linear algebra above.

## Position in the integration chain

| Route | Ordered operations |
|---|---|
| Paper Algorithm 1 | Nonpressure forces and predictor; density solve; velocity correction; position drift; rebuild neighbours/density/factor; divergence solve; final velocity correction |
| Pinned SPlisHSPlasH `step()` | Neighbours/boundary maps/density/factor; divergence solve and velocity correction; nonpressure forces; timestep-size update; nonpressure velocity kick; density solve and velocity correction; position drift |
| Local `dfsphReference_step()` | Neighbours/density; configured boundary/filter processing; nonpressure forces and kick; density solve/kick; drift; rebuild neighbours/density; divergence solve/kick |
| Native `iisph_step()` | Neighbours/density/Dirichlet data; nonpressure forces; one density pressure solve; return derivatives; standard semi-implicit Euler kicks velocity then drifts positions |
| Native `dfsph_step()` | Neighbours/density/Dirichlet data; shared native pressure geometry; divergence projection/kick; nonpressure forces on corrected velocity; density projection; update BCs; one nonpressure+density kick and position drift |

The local reference's CD/drift/DF order agrees with the paper. Moving DF to
the next step's beginning gives the reference-code phase convention. In an
uninterrupted fixed-step loop with matching inputs and no intervening velocity
changes, these are cyclic placements of the same sequence, not evidence that
one is incorrect. Startup, output/checkpoint velocity phase, boundary/filter
processing between stages, and variable-step warm starts/stopping still differ.
The paper's geometry after drift may serve the following density solve;
SPlisHSPlasH shares the precomputed factor between its two same-geometry solves.
Do not reuse coefficients across a position change.

Adaptive-step parity needs a further source-level qualification: this pinned
`step()` captures `h` before calling `Simulation::updateTimeStepSize()`, then
uses that captured value for the nonpressure kick, position drift and clock.
`pressureSolve()` reads the TimeManager value again after the update.
`Simulation.cpp` lines 401–498 and `TimeManager::setTimeStepSize()` confirm that
CFL selection can change the value immediately. Thus the pinned code can use
different timestep values inside one step. This is a source observation, not
a measured adaptive-run failure; start reference parity with fixed dt and
define one explicit dt contract for the new local path.

Both native and legacy DFSPH use the package `ProjectedStep` full-map driver and suppress
ordinary state updates. Native IISPH uses a normal derivative system with
`semiImplicitEuler`. Merely adding divergence acceleration to that one IISPH
RHS would miss the velocity phase and force-evaluation ordering. A native
DFSPH stage map uses the package's kick/drift primitives, and the driver
evaluates that map once without integrating its diagnostic update again.
Neither solve is a
post-integrator particle shift or a velocity resampling operation.

## Differences in the current local reference

| Choice | Pinned SPlisHSPlasH | Local `dfsphReference` |
|---|---|---|
| Continuity weights | Rest volumes `m/rho0`; scalar equal-volume assumption | Apparent fluid volumes `m/rho`; scatter/difference density-rate wrapper |
| Diagonal/preconditioner | Fluid back-reaction squares plus total-gradient square | Includes sampled static-wall squares; approximate factor for the composed physical-pressure operator |
| Relaxation | 0.5 | Hardcoded 0.3, independently of solver-config relaxation |
| Weak factors | Disable factor if denominator <= 1e-5 | Floor denominator magnitude at 1e-8, then invert |
| Stored fields | Separate density/divergence carriers scaled by `dt^2`/`dt` | Physical density pressure in `pressures`; divergence pressure in `soundspeeds`; full raw carry by default |
| Damped warm start | Half capped scaled carrier; active-compression gate | Optional module switch, off by default; same numeric caps applied to different pressure units |
| Density threshold | `rho0*0.01*maxErrorPercent`, default percent 0.01 | Absolute error tolerance 5e-4 by default |
| Divergence threshold | `rho0*0.01*maxErrorVPercent/dt`, default percent 0.1 | Absolute rate-error tolerance 2.5e-3 by default, no percent or `1/dt` conversion |
| Iterations | CD min 2, DF min 1; each max 100 by default | CD/DF at least configured minima (both 2 by default); silently raises caps below 100 to 100 |
| Sampled walls | Static self-pressure term, boundary volumes | Temporarily replaces wall state with Akinci data and automatically changes scale 1 to 2 |
| Analytic walls | Density/volume-map boundary branches | Legacy mirrored/hydrostatic force, potentially affine/nonlinear; separate analytic factor correction |

Both use pressure floors and a compression-only residual statistic, but native
IISPH's pressure-active signed stopping statistic is different again. Neither
statistic is a residual norm. A new DFSPH stage must select its statistic and
units explicitly rather than inherit IISPH's stopping rule. Capped carrier
values cannot be transferred between `chi`, `q` and physical `p` without
conversion, especially when `rho0` or density ratios differ.

### Low-neighbour guard: literal source behaviour

The reference setup at lines 411–455 copies the stored density-change rate
into a local `densityAdv`, clamps it and zeroes that local variable below
7 neighbours in 2D / 20 in 3D. That affects the warm-start gate/initial guess.
It does **not** write the clamped value back into simulation data. Iteration
at lines 665–697 uses the original stored source. The guard zeroes the error
statistic, but the pressure update still runs with that original source.

Local `modules/incompressible/reference.py` instead zeroes the source on
deficient rows, in addition to suppressing warm starts and the statistic.
That is a real port difference; its docstring's claim of literal source-guard
agreement is inaccurate. Neither version pins the deficient pressure iterate
throughout the solve. The optional local free-surface gauge does pin pressure
and is another, distinct policy. These choices require separate measurements;
the upstream comments alone do not establish which should be adopted.

## Other incompressible paths

- `omniIncompressible`: nonpressure predictor, divergence then density at the
  same geometry, one combined kick/drift. This puts divergence after the
  nonpressure predictor, unlike the pinned DFSPH code. It uses the omni
  discretization/policies, fixed three divergence sweeps, optional viscosity,
  signed divergence pressure and a floor-clipped signed density statistic.
- `divergenceFree`: a VD+PS/omni hybrid returning an ordinary RHS. Gravity's
  automatic policy may apply density correction to velocity; otherwise
  finalization uses a position shift and velocity resampling. It is not a
  second spelling of DFSPH.
- `band2018pb`: one extended density solve with fluid and selected boundary
  pressure unknowns, plus local regularization/gauge choices. Its wall-pressure
  system is distinct from static self pressure or contact interpolation.
- Compact CG is an optional projection/operator choice in the omni family;
  replacing Jacobi with CG alone does not add DFSPH's two-constraint schedule.

## Implementation and acceptance gates

Native DFSPH shares `computeIISPHProjectionGeometry` with standalone IISPH;
the latter's equations, policies and defaults are preserved. Density and
divergence sources, policies, histories and diagnostics are distinct. The
homogeneous divergence correction excludes known hydrostatic acceleration;
the density stage includes it in both the source and the applied acceleration.

The implemented placement is reference-start divergence, including an initial
projection of the supplied velocity. Output velocities have undergone the
density correction and drift; their next divergence projection occurs at the
next step's start. Both constraints use the same pre-drift geometry. Static
bodies and fixed common support remain required; analytic solid integrals
currently support 2D. There is no legacy band-volume scaling, mDBC velocity
extension, XSPH filter, VD+PS shift or velocity resampling in this path.

Validation requirements and limits:

1. Fixed-state `C`, `F`, exact basis-column diagonal and Jacobi iterates against
   independent algebra; equal-volume parity with the pinned reference after
   converting pressure variables, including deficient/free-surface rows.
2. Density-unit and changing-timestep invariance of velocity corrections and
   warm starts; separate pressure-floor and stopping-statistic checks.
3. One-step trace of force evaluation, both projections, drift and coefficient
   rebuilding; checkpoint/output phase and pressure-load geometry checks.
4. Matched calibrated periodic compression/expansion and vortex/shear cases;
   hydrostatic columns, disordered sampled/analytic walls, dam break and
   sloshing. Measure density and divergence errors separately, pressure,
   energy/dissipation, wall penetration/reaction, timestep/resolution sensitivity,
   iteration counts, wall time and ms/step. Preserve videos and failure evidence.

Source/algebra agreement establishes the operator relationship; it does not
establish long-time stability, dissipation or GPU performance.

### Current validation status

Independent dense 2D/3D tests cover both sources, exact operators/diagonals,
clamped and signed Jacobi steps, contact gravity applied only once, pressure
histories across changing dt and density units, and nonfinite rejection.
Lifecycle tests cover pre-viscosity divergence projection, a single shared
geometry assembly, one position advance, fixed samples, unchanged inputs,
checkpoint restoration and sound-speed independence. Wide-support pressure-mode
tests protect the relaxation choice. Region and analytic-lattice initialization
are shared with native IISPH, including its one-time mass correction.

Recorded CPU controls at support 4 include TGV at nx=24/32, a halved timestep,
sampled hydrostatic walls, analytic self/contact walls, and a short dam break.
The TGV controls reach t=.06 with bulk density within .0013% of rest density;
their final kinetic energies differ by less than .05% when dt is halved.
With matched .3 relaxation at nx=24, DFSPH takes about 25 ms/step versus 20
for standalone IISPH, with final kinetic energy differing by less than .001%.
These are small CPU timing controls, not a GPU benchmark.

Wall controls reach t=.1 without solver-cap hits, but are not quiet equilibrium
solutions: the sampled nx=32 column peaks near .86 m/s, versus 1.01 m/s for
matched IISPH. The analytic contact tank's peak falls from .192 m/s at nx=32
to .138 m/s at nx=100; its final speed at nx=100 is .0585 m/s. Surface kernel
deficiency remains present and is not repaired by filling density. Full
hydrostatic accuracy, wall-load balance, long-time pairing, impact pressures,
and the seven-second sloshing benchmark remain unqualified. Periodic pressure
gauge/roundoff sensitivity also remains visible when dt changes in FP32.

The shared contact-point helper converts provider distances as well as normals
to particle precision, preventing a double-precision provider from constructing
incompatible single-precision search queries. The standalone IISPH equations
and defaults remain unchanged. Its wide-support .5 relaxation failure was
reproduced on the untouched baseline; the initializer ownership test now uses
an explicitly stable .3 fixture rather than depending on a random early exit.

Raw controls, videos, commands, failure evidence and solver-policy A/Bs are
retained in Git-ignored `.tmp/dfsph-native-2026-10-25/`. Removing the density
seed gate did not materially improve the sampled column and was not adopted.
CUDA is unavailable on the validation host, so GPU behaviour and performance
remain unverified.


## Opt-in comparison compositions

`dfsph.ordering` selects `df_np_cd`(default), `np_df_cd`, or `np_df_shift`.
The predictor-first alternatives reuse the same native C/F/diagonal and
independent projection histories. The shift composition rebuilds geometry
following drift, solves the density constraint with zero predictor velocity,
shifts positions and resamples velocity with its difference gradient. Its
physical pressure is the divergence pressure; the density multiplier is
auxiliary. Known wall-gravity acceleration enters the physical predictor
once and is absent from the positional solve. Periodic coordinates wrap
before rebuilding shifted geometry. These alternatives are exposed for
comparison; the standard DF-first production default is unchanged.

See [the prepared matrix](../fluid_matrix/README.md) for shared initial states,
explicit periodic source policy and accuracy/cost qualification.


## Experimental re-entry remedies — 26 October 2026

A single equal-mass particle released4dx above a shared quasi-relaxed465-particle
analytic hydrostatic tank reproduces delayed re-merging. Production DFSPH has
rho/rho0=.276 and nearest bulk distance2.10dx after2s, still descending slowly.
During1.5–2s its upward DF/CD accelerations average7.56/2.16m/s². This is not
proof of permanent hovering; the full dam-break establishes longer-lived sparse
layers separately. GPUFP32, C4/4dx, physical fluid viscosity.01m²/s, classic
ordering, DF.1%/CD.01%, cap100 and weak-factor cutoff retained. Preparation
uses gamma20/s damping for5s; all subsequent holds/drops are undamped. The
strict1mm/s RMS rest target is not met (final3.02mm/s), and the paired surface
is inherited. Two small corner wall excursions exist before the drop.

Three private local hypotheses were tested, without modifying this module's
production solver/configuration:

- Pin the DF pressure unknown to0 when rho/rho0<.9, including every update.
  This differs from masking only seeds/statistics; the original symmetric
  force operator still transmits neighbouring pressure.
- Add density headroom to the compression bound:
  `C(v_out) <= max(1-rho/rho0, 0)/tau`, or source
  `max(1-rho/rho0,0)/tau - C(v_in)`. Test tau=dt and tau=.02s separately.
  Only clamped DF is changed; signed periodic projection and CD remain intact.

All three reach sustained.85rho0 by.294/.287/.286s and rho≈1 at2s, with
nearest spacing≈.71dx on the existing paired surface. The hard threshold
produces a visible recovery plateau around.9rho0. Headroom permits smoother
recovery; the.02s case at half dt merges at.2835s and reaches.707dx versus
.709dx. Seven private numerical gates verify unchanged complete-bulk projection
(including signed Fourier-mode recovery/momentum) and unchanged CD, plus true
disabling/allowance of deficient constraints. Exact physical input tensors match
across all five drops. No solve caps, positive density/nonnegative pressure,
constant mass and videos/source checks pass. Maximum bulk density overshoot
in1ms drops is.0294%; maximum sampled corner penetration reaches.262dx.

This is a useful re-entry result, not a default qualification. In matched
no-drop holds, finite headroom raises late RMS6.15→6.54mm/s, pressure L2
error10.70→12.55%, and temporal mean-pressure noise105→195Pa. Mean pressure
changes only1.22Pa. With the drop, pressure noise changes283→4.28Pa on
halving dt, despite similar merging; pressure chatter remains strongly timestep
sensitive. Original pressure-impulse history is preserved/rescaled by the
existing dt contract, with a short initial half-step transient. Wall containment,
resolution, sampled walls and impact/spray qualification remain open.

**Verdict:** density headroom is the preferred hypothesis for a further re-entry
trial; keep it experimental. No production default or wider matrix changed.
Primary2s drops cost42.3–43.6s wall/16.7–17.3ms engine per step with direct
observers/video; half-step control76.3s/15.2ms. Preparation107.7s/17.7ms;
no-drop2s controls47.2/45.5s wall,17.7/17.4ms engine. No measured remedy cost
regression in these controls. Full fields, metrics, numerical/source/video
audits, figures and reproducible private drivers: [re-entry report](../../.tmp/dfsph-reentry-2026-10-26/README.md).


## Dam-break re-entry remedy comparison — 26 October 2026

All three private remedies complete the exact saved7200-fluid sampled-wall
Marrone case to2.45s on GPUFP32. No physical input, mass, viscosity, timestep
policy, CDconstraint, production solver or default changes; production source
hashes match the cutoff baseline. Original alpha.01/c0=47.3091 and C4/4dx,
classic ordering, omega.25, DF.1%/CD.01%, cap100 and cutoff1e-5 retained.
One constraint choice per run: active09 pins the DFunknown below.9rho0;
headroom allows C(v_out)<=(1-rho/rho0)_+/tau with tau=dt or20ms.

| DF rule | Persistent near-self riders | Final rho<.3 count | Integrated K/M | Endpoint K/M | Pressure error [%] | Wall [s] / ms step |
|---|---:|---:|---:|---:|---:|---:|
| Cutoff baseline | 7 | 82 | 2.20177 | 0.64534 | 42.24 | 134.36/23.87 |
| DF active at rho>=.9rho0 | 0 | 2 | 2.18816 | 0.59105 | 40.70 | 456.92/81.72 |
| Headroom/dt | 0 | 20 | 2.16413 | 0.61107 | 40.78 | 404.70/74.30 |
| Headroom/20ms | 0 | 7 | 2.17738 | 0.63147 | 39.72 | 243.33/42.28 |

Near-self cohort: normalized sampling recovery<.02 for80%recorded late samples
(t>=1.9s), near rho>=.85bulk/interior for50%. Baseline saved-frame cadence
differs from10msremedy snapshots; endpoint counts provide an additional check.
Broader persistent rho<.5 cohort is43/0/0/2: finite20ms retains2moderately
under-sampled particles, so the worst near-self layer is removed but complete
merging is not proved for every particle. Global sparse counts include spray.

Peak K/M is1.439/1.436/1.437 versus1.443baseline. Integratedenergy remains
98.29–99.38%ofbaseline; there is no return to the former grossly overdamped
case. Finite20ms has the lowest measured pressure error39.72% and closest
endpointenergy, while active09 has the fewest globally rho<.3 particles. The
pressure improvement is modest and remains far from a physical qualification.

Reported pairing rises5.08→7.89/8.39/8.82%, while voidfraction drops
2.03→.319/.417/.319%. Peakrho1.0134/1.0217/1.0094 versus1.0061baseline.
Maximumwall excursions.832/.847/.823dx versus.811; maximumcounts more than
.5dx beyond tank AABB are7/5/8 versus4baseline. Endpoint counts by that
>.5dx criterion are0, but active09/20ms retain smaller excursions. Do not
interpret this diagnostic's count0 as perfect containment. No new wall model.

All cases retain0DFcaps/3CDcaps, the same startup.004/.005/.006s. Mass remains
fixed; sampled fields finite, densitypositive, pressureunknowns nonnegative.
All3full evidence/source/video/input/timestep audits and per-stage dispatch/
endpoint parity checks pass;7repeated private numerical gates pass. No new
dam-break resolution/timestep sweep; single-particle sensitivity is separate.
Engine costs46.87/42.46/27.92ms versus18.38baseline; direct snapshot copies
1.67/1.18/.46s total. Unrelated GPU work shares the device with changing load,
so neither the large measured slowdown nor differences among remedies are an
isolated algorithm-cost ranking. MeanCDiterations5.72/6.19/5.55 versus5.65.

**Verdict:** all three address the near-self-density layer in this case.
Headroom remains attractive without a hard density gate;20ms balances pressure
and endpointenergy here, but leaves a small moderately deficient cohort and
more pairing. Keep experimental; no default or broad-matrix promotion.
[Full report, figures, recorded videos and private drivers](../../.tmp/dfsph-remedy-dambreak-2026-10-26/README.md).


### Four-way sloshing re-entry comparison — 26 October 2026

All four variants complete the same saved 4,200-fluid particle-wall tank to
7 s on GPU FP32. Calibrated masses, hydrostatic pressure seed, C4/4dx, classic
DF → nonpressure → CD → drift ordering, omega=0.25, DF=0.1%/CD=0.01%,
cap=100 and weak-factor cutoff=1e-5 are shared. The repaired consistency-v1
case uses physical nu=1e-6 m²/s, reflected free-slip viscosity references,
and a 0.2 ms ceiling with CFL=0.2. No new IC preparation or additional damping.
These are local experimental DF constraints; CD and production defaults are unchanged.

| DF rule | Persistent near-self / rho<.3 / rho<.5 | Final rho<.3 count | Integrated K/M, relative baseline | Pressure trace error [%] | Pairing final [%] | Wall [s] / ms step |
|---|---|---:|---:|---:|---:|---|
| Cutoff baseline | 8/39/62 | 199 | 100.00% | 96.03 | 6.75 | 832.93 / 23.80 |
| DF off below 0.9 rho0 | 0/0/0 | 56 | 98.73% | 71.50 | 7.38 | 650.56 / 18.56 |
| Headroom / dt | 0/0/0 | 35 | 95.46% | 83.07 | 10.81 | 623.41 / 17.77 |
| Headroom / 20 ms | 0/0/0 | 67 | 98.27% | 66.93 | 8.83 | 604.46 / 17.27 |

The persistent cohort is observational: after 6.2 s, low density for at least
80% of saved samples and near dense interior fluid for at least 50%. Near-self
means normalized recovery (rho-rho_self)/(rho0-rho_self)<0.02; rho_self≈0.179rho0.
Global endpoint sparse counts also include spray. All three remedies eliminate
the measured persistent layer, including the broader rho<0.5 cohort. Common
geometry/color snapshots show the baseline's separate low-density surface row
and its removal by each remedy. This does not prove that every airborne
particle has merged.

Driven wave motion remains substantial: integrated K/M is 98.73%, 95.46% and
98.27% of baseline. COM x ranges are 0.3025/0.2864/0.2708/0.2803 m in table
order. Higher energy alone
is not an accuracy criterion for this externally driven problem.

Pressure error is the standard full-record relative L2/NRMSE against measured
Sensor1, with matched 10 ms smoothing and no phase alignment; it includes
pulse timing and shape. The 20 ms rule scores best overall, but its third
smoothed peak is 3.045 kPa versus 1.171 kPa measured, worse than baseline
2.405 kPa or hard-gate 2.557 kPa. Its first two peaks are 1.966/2.080 kPa
versus measured 1.800/2.322 kPa, with timings closer to the reference. Thus
the lower full-trace score does not establish accurate impact amplitudes.

Pairing rises from 6.75% to 7.38/10.81/8.83%; void fraction falls from 5.37%
to 2.17/2.94/1.86%. Peak rho/rho0 is 1.0022/1.0088/1.0185/1.0066. Maximum
wall excursions are 0.776/0.848/0.891/0.827dx; simultaneous counts more than
0.5dx outside the tank AABB reach 4/7/8/11. Endpoint counts by that half-dx
criterion are zero, but strict crossing counts are 0/0/6/1. Wall contact
therefore remains unresolved. Peak speeds are 3.757/5.337/5.627/4.645 m/s.

All four retain fixed mass, positive finite sampled density and nonnegative
pressure unknowns, with zero DF/CD iteration caps and no timestep violations.
Four full evidence audits pass, as do seven private operator/constraint gates.
Initial physical fields match bitwise, pressure impulses match the shared
hydrostatic seed and actual timestep, stage call counts match steps, final
UID-sorted snapshots match final tensors, and every video fully decodes. All
1,021 current production source hashes match the archived baseline. Engine
costs are 16.92/14.48/14.11/13.92 ms per step; snapshot copies total
0.55/0.26/0.23/0.19 s (exact receipts retained). Shared GPU contention means
wall/step differences are not an isolated algorithm-cost ranking.

**Verdict:** keep the simple density gate and finite-horizon headroom as
experimental re-entry remedies. Both recover sampling with little change in
bulk kinetic motion. The gate has less pairing; 20 ms gives the lowest full
pressure-trace error but a larger third impact overshoot. One-step headroom
has the largest pairing, density peak and motion reduction here. Pressure,
wall contact and resolution/timestep sensitivity remain open; no production
default or normal-matrix promotion.

[Full report, common-scale figures, recorded videos and retained evidence](../../.tmp/dfsph-remedy-sloshing-2026-10-26/README.md).


### Candidate selection and proposed qualification — 26 October 2026

User selects **DF active only at rho>=0.9rho0** as the candidate to carry
forward. Sloshing first-impact amplitude, second-impact width/ringing, bulk
kinetic motion and sampling quality take precedence over the single aggregate
pressure score. The headroom experiments remain retained comparison evidence;
this selection does not promote a production default or restart the matrix.

Proposed evaluation sequence (not launched by this planning discussion):

1. Freeze the exact experimental gate semantics: pin the clamped DF pressure
   unknown, including warm start, below threshold; retain symmetric neighbour
   pressure forces, CD, weak-factor cutoff and signed periodic policy. Before
   replacing the private adapter, expose an opt-in configuration and establish
   numerical parity with saved trials plus checks for activation, warm starts,
   force symmetry and convergence statistics.
2. Add the missing active09 no-drop hydrostatic control and a half-timestep
   active09 drop control. Measure hydrostatic pressure slope/error/noise, RMS
   motion, merger density/spacing and the observed pause near 0.9rho0. Track
   activation transitions and pressure/velocity kicks around the threshold.
3. Run active09 sloshing at half the timestep from the identical saved IC;
   then one finer spatial resolution using the same physical state and
   preparation policy. Compare all three impact amplitudes, arrival times,
   widths/impulses and post-impact ringing separately; retain raw and matched
   smoothed pressures, wave displacement, K/M, pairing, wall excursions,
   pressure-stage work, solve failures and measured runtime. No time alignment
   or threshold fitting to the experimental trace.
4. Verify boundary transfer with analytic hydrostatic/drop and sloshing cases,
   using established density-compatible wall placement. Verify that the gate
   consumes the same boundary-inclusive density ratio as the DF operator.
5. Test bulk dynamics on prepared TGV and nominal-mass relaxed closed-box
   decay, with matched IISPH controls. Check velocity-field error where a
   reference exists, energy/enstrophy decay, divergence, density and symmetry.
   Record the inactive fraction: well-sampled bulk should not routinely rely
   on the surface gate. The signed periodic policy remains unchanged.

Proceed to focused matrix comparison only after these controls show stable
re-entry and hydrostatic support without materially worsening pressure, bulk
dynamics or wall behaviour. Threshold sensitivity (e.g.0.85/0.95) is a later
robustness check if activation diagnostics warrant it, not an immediate tuning
sweep. Costs remain part of every run; this is a proposed sequence, not new
qualification evidence.


### Native opt-in density gate and initial qualification — 26 October 2026

The selected trial is now available as
`schemeConfig.dfsph.minDivergenceDensityRatio = 0.9`. Default `0.0` retains
the original constraint. Values must be finite in [0,1] and round-trip through
the normal configuration serializer. Only clamped DF pressure unknowns below
the threshold are disabled, including their warm history and convergence
statistic; neighbour forces, density projection and signed projection are
preserved. Equality remains active. This is a local experimental extension,
not a claim about the published/reference DFSPH method.

119 projection/configuration regressions pass (12 optional tests skipped),
including force symmetry, warm-history clearing, inclusive activation and
unchanged CD/signed/complete-bulk response. Twelve frozen-operator comparisons
match exact pressure, acceleration, iteration count and residual: clamped DF
against the actual archived trial; signed and CD against the untouched original.
The trial adapter's unused signed low-density residual mask is deliberately
not transferred. Whole-trajectory native baseline hold and active09 1ms drop
endpoint fields match archived trajectories bitwise.

The matched 2s no-drop hold changes late RMS6.145→6.152mm/s, mean-pressure
noise104.687→104.688Pa and pressure L2error10.701→10.720%. The gate does not
materially disturb this quasi-relaxed control. The fixture retains its
pre-existing pairing, nonzero residual motion and fitted analytic-wall
excursions; it is not exact hydrostatic equilibrium. Its recorded physical
nu=.01 with fluid-only two-sided viscosity differs from the sloshing case's
nu=1e-6; settings are held fixed within each comparison.

Both 1ms and .5ms drops recover sustained .85rho0 near.294/.288s with a single
gate activation, no repeated threshold chatter, zero DF/CD caps and fixed mass.
The 1ms drop reaches sustained .99rho0 at1.289s; the half-step remains at
.986rho0 at2s. Near-threshold settling is timestep-sensitive even though
nearest spacing is already .709/.714dx and the detached self-density stand-off
is removed. Late pressure noise268.43→3.35Pa and L2error19.92→10.86% demand
further controls; slower recovery does not prove a pressure improvement.
Initial physical pressures are identical: the 1ms preparation impulse histories
are rescaled with dt/dt² for DF/CD, removing the earlier half-step startup-seed
confound. Native0.1% DF tolerance is retained, so its density-rate bound also
scales as1/dt; this is a timestep-policy check, not an isolated integrator study.

The matched half-step no-drop controls and .1ms sloshing replay are now
complete; see the measured results below. Pressure convergence, refinement
and boundary/bulk transfer remain unqualified. No production default or broad matrix promotion. Full retained
evidence: `.tmp/dfsph-active09-qualification-2026-10-26/`.


### Initial active09 qualification completed — 26 October 2026

Six matched 2s tank controls and one full7s half-step sloshing replay complete
on GPUFP32. The selected gate is opt-in and remains off by default. All six
control source/input/seed/endpoint/video audits and the normal sloshing audit
pass. Seven trajectories have fixed mass, positive sampled density, nonnegative
pressure, zero DF/CD caps and no timestep violations. Native baseline hold and
active09 1ms drop reproduce archived endpoint fields bitwise; 119 regressions
and12 frozen-operator parity checks pass (12 optional tests skipped).

No-drop hydrostatics are essentially invariant to the gate at each timestep:
1ms RMS6.145/6.152mm/s and pressure noise104.687/104.688Pa for original/gated;
.5ms RMS6.116/6.114mm/s and noise1.011/.970Pa. The large pressure-noise change
with dt occurs with the gate off as well; it is not threshold chatter. Drop
activation happens once per trajectory. Sustained .85rho0 recovery is.294/.288s
and final nearest spacing.709/.714dx at1/.5ms. Complete recovery toward1rho0
is slower at half-step (.986rho0 at2s, versus1.000), so early merger alone does
not demonstrate convergence. Physical pressure seed is fixed across dt by
rescaling DF/CD impulses with dt/dt², not preserving the old impulse magnitude.

| Sloshing ceiling | Peak1 / Peak2 / Peak3 [kPa] | Full trace error [%] | Integrated K/M | Pairing final [%] | Wall [s] / ms step |
|---|---|---:|---:|---:|---|
| .2ms, selected archived trial | 2.060 / 2.160 / 2.557 | 71.50 | .727675 | 7.38 | 650.56 / 18.56 |
| .1ms, equivalent native setting | 1.535 / 2.428 / 2.687 | 74.20 | .713944 | 8.18 | 1328.30 / 18.98 |
| Measured, same10ms smoothing | 1.800 / 2.322 / 1.171 | — | — | — | — |

Bulk motion remains close: integratedK/M−1.89%, COMx range.2864→.2795m.
No persistent near-self/.3/.5 sparse cohort at either dt. Endpoint rho<.3
counts56→49 include spray. Peakdensity1.00880→1.00066; maxwallexcursion
.848→.783dx with7 simultaneous>.5dx crossings in each, noendpointcrossings.
Both remain unqualified for exact containment/isotropy. Sloshing post-second
impact residual RMS after a10ms trend changes sensor115.49→140.88Pa, mean
DF3.89→5.56Pa, meanCD35.09→27.22Pa: unlike the quiet tank, this does not
uniformly improve. Residuals include physical fast transients and probe effects.
Second-pulse width66→57ms versus42ms measured, but window impulse457→488Pa·s
versus310Pa·s; separate amplitude, width, phase and impulse conclusions.

| Control | Wall [s] / ms step | Engine ms step |
|---|---|---:|
| Original hold1ms | 74.63 / 37.32 | 26.93 |
| Gate hold1ms | 69.71 / 34.86 | 25.00 |
| Gate drop1ms | 70.28 / 35.14 | 25.58 |
| Gate drop.5ms | 171.49 / 42.87 | 31.42 |
| Original hold.5ms | 68.11 / 17.03 | 13.44 |
| Gate hold.5ms | 69.43 / 17.36 | 13.63 |

Shared-device load and differing diagnostic/video work prevent an isolated
performance ranking. Sloshing engine14.48→15.31ms/step,70k vs35,053steps;
direct field copies.228s. Full runtime and all retained videos are in report.

**Verdict:** the gate remains the selected experimental candidate and passes
the initial rest/re-entry robustness checks. Impact pressure has a material
timestep dependence; the remaining evaluation programme is not complete.
Establish temporal/solve-accuracy convergence before proceeding to the costly
spatial, analytic-sloshing and bulk transfer sweeps or promoting a default.
Native percentage tolerances imply dt-dependent rate/pressure residual bounds,
so this two-level test does not isolate integration error. A further timestep
level and a separately declared solve-accuracy control are the next numerical
gate; no threshold fitting/hysteresis change. Finer spatial IC must preserve
actual fluid mass/volume, avoiding silent water-depth changes from rounding.
The sloshing reference also retains its existing gravity-only rotating-frame
approximation; no forcing model is changed here.

[Full report, per-impact metrics, source/input audits and recordings](../../.tmp/dfsph-active09-qualification-2026-10-26/README.md).


### Pressure convergence controls — 26 October 2026

Completed the declared first-impact study through 2.9 s on the identical
4,200-fluid-particle GPU FP32 sloshing fixture, with particle walls, the 0.9ρ₀
gate, physical ν = 1e-6 and matched hydrostatic pressure initialization.
Three new runs: native 0.2 ms with 112 frozen stage/variant records, native
0.05 ms, and 0.2 ms with both tolerances divided by 100. The archived 0.1 ms
trajectory supplies the middle timestep level. Mass, IC, forcing, threshold
and production solver code are unchanged.

With matched 10 ms filtering, native first peaks are non-monotonic:
2.060 → 1.535 → 1.888 kPa, versus 1.800 kPa measured. Times of 2.369, 2.371
and 2.373 s remain early versus 2.390 s. Through 2.85 s, integrated K/M is
0.126170, 0.127894 and 0.128613; centre-of-mass excursion is 0.237553,
0.237111 and 0.237088 m. Bulk motion changes much less than peak pressure.
Native median density residual scales roughly with dt² while DF/CD mean
iterations remain 1/~2. No temporal order is inferred.

The tighter control raises the peak to 2.396 kPa. FWHM is 35 ms versus
40 ms native and 49 ms measured. Pressure error over 2.2–2.85 s grows from
45.99% to 50.37%; positive pressure integral changes from 407.72 to
408.77 Pa·s versus 363.97 measured. Integrated K increases 1.59%; peak K
changes only 0.057%. Both stage tolerances changed together, so their
individual contributions remain unresolved. CD averages 99.316 iterations,
with 11,035/14,521 steps capped (75.99%). Reported CD residual/bound has
median 1.121, 99th percentile 1.584 and maximum 1.837. DF averages 1.148
iterations with no caps. This is **not a converged reference at the requested
tolerance**; retain its finite-budget result without retuning.

On frozen native peak geometry, DF takes one iteration and CD two, satisfying
their native global compression bounds. At 512 iterations, maximum velocity
kick changes reach 0.319/0.295 m/s. Pressure differences between 128 and 512
iterations remain 30%/35% relative to 512, so 512 itself is an unqualified
pressure reference. Tightly projecting an accumulated density error in one
small step can create very large pressure; this is not evidence that such
pressure is physical. The advancing tighter run maintains normal mean
pressure. Observer purity and bitwise replay parity are verified.

Noise does not improve uniformly: native→tight post-impact raw sensor
residual is 91.55→91.77 Pa, and the common 1 ms-minus-10 ms band changes
64.10→48.99 Pa. Mean DF residual grows 2.15→10.79 Pa while mean CD falls
28.22→19.75 Pa. At native 0.05 ms, raw sensor residual is 122.28 Pa and the
common-band residual 65.91 Pa. These statistics contain physical transients
and probe effects. Maximum pairing remains 8–10%, with wall excursions of
0.63–0.83 dx. This short prefix does not requalify late rider cohorts.

All three terminal, source, initialization, physical-timestep, mass and video
audits pass. All 1,029 source hashes per run and endpoint observer tensors
match. The instrumented native replay matches the earlier trajectory bitwise
before 2.85 s. Costs with video:

| New run | Wall seconds | Wall ms/step | Engine ms/step |
|---|---:|---:|---:|
| Native 0.2 ms replay | 278.40 | 19.20 | 15.59 |
| Native 0.05 ms | 1007.70 | 17.37 | 14.20 |
| Tolerances /100 at 0.2 ms | 1775.52 | 122.27 | 118.25 |

Replay includes 10.87 s of frozen projections. Tight accuracy effort costs
about 6.4× per step. The archived 0.1 ms cost of 1328.30 s / 18.98 ms per step
covers its full 7 s, not this prefix. Shared-device load limits timing isolation.

**Verdict:** retain the 0.9 gate as the experimental candidate. Pressure
convergence and solve cost remain unresolved. Establish stage-specific,
adequately converged pressure/velocity corrections before expensive spatial,
analytic-wall or matrix expansion and default promotion. Do not select
tolerances to fit a measured peak. Existing gravity-only rotating-frame and
fluid-probe/reference-transducer limitations remain unchanged. This closes
the declared convergence experiment, not the entire qualification programme.

[Full report, figures, recordings and evidence](../../.tmp/dfsph-pressure-convergence-2026-10-26/README.md).


### Analytic-wall density-gate transfer — 26 October 2026

The prepared matrix adapter now consumes
`df_sampling_gate: {minDivergenceDensityRatio: 0.9}` for either wall
representation. The shared native solver already handles analytic pressure
geometry; no separate gate or force mask is introduced. A new analytic
projection test verifies that wall-completed density keeps the dense bottom
layer active when its fluid-only sum is below the threshold. Six matrix
configuration cases cover absent/zero/0.9 settings on both representations.
All 47 focused checks pass.

Completed matched baseline (gate 0) and candidate (0.9) sloshing runs through
7 s on GPU FP32, native DF 0.1% / CD 0.01% bounds, cap 100, 0.2 ms step ceiling,
C4/4dx, physical ν = 1e-6 and the exact saved fitted-wall analytic fixture.
This user-directed transfer proceeds with the acknowledged chaotic sensitivity;
no tolerance or threshold fitting, new relaxation or default promotion.

| Analytic result | Baseline | Gate 0.9 |
|---|---:|---:|
| Persistent near-self / rho<0.3 / rho<0.5 cohorts | 6 / 44 / 64 | 0 / 0 / 0 |
| Integrated K/M | 0.725202 | 0.722891 |
| Filtered peaks [kPa] | 1.789 / 1.509 / 2.178 | 1.801 / 2.614 / 2.410 |
| Peak times [s] | 2.361 / 4.057 / 5.623 | 2.380 / 4.030 / 5.660 |
| Pressure trace error [%] | 74.45 | 71.65 |
| Final paired fraction [%] | 5.77 | 8.09 |
| Saved-frame max excursion beyond fitted faces [dx] | 0.857 | 0.866 |
| Saved-frame max count beyond 0.5dx | 4 | 9 |
| DF / CD caps | 0 / 0 | 0 / 0 |
| Wall seconds / ms per step | 797.07 / 22.77 | 803.65 / 22.96 |
| Engine ms per step | 19.40 | 19.54 |

Persistent cohorts use the same late 6.2–7 s near-bulk residence criterion as
the sampled-wall tests. Ordinary sparse spray remains; the gate removes the
persistent surface layer. Baseline cohort samples have positive own DF pressure
93.68% of the time, while own CD pressure is zero. Every saved gated sample
below 0.9ρ₀ has zero own DF pressure. Integrated kinetic energy changes only
−0.32%, preserving driven motion. No mass drift, timestep violations, solver
caps or speed alarms; both source, initialization, terminal-field and video
audits pass.

Pressure is compared with the same 10 ms filter, without alignment. Measured
peaks are 1.800 / 2.322 / 1.171 kPa at 2.390 / 4.066 / 5.697 s. First/third
phase improves in this realization, second phase worsens, and the third peak
remains too large. All-fluid mean DF post-second-impact residual falls
348.78→4.09 Pa, but sensor residual rises 79.55→133.93 Pa and mean CD residual
29.80→37.29 Pa. Large raw sensor transients remain (12.08→19.27 kPa maxima).
These signal residuals include physical transients and probe effects. Pairing
and transient wall crossings worsen, despite no endpoint crossing in the gate
run. Saved frames can miss excursions between frames; nominal every-step
half-dx crossing counts reach 4→12.

Analytic baseline/gate initial tensors and first-step scalar diagnostics match
exactly. Both retain the existing corner density defect (initial maximum
1.01088ρ₀) and first-step speed 0.3727 m/s. Across wall representations, fluid
positions, velocities, supports, UIDs and physical pressure match exactly;
fluid mass differs by only 0.000036%, while wall-completed density differs.
The direct analytic A/B isolates the gate; the cross-boundary curves include
these discretization/startup differences. Existing gravity-only rotating-frame
and fluid-probe/reference-transducer limitations are unchanged.

**Verdict:** the selected gate transfers successfully for re-merging on analytic
walls at essentially the same measured cost. Pressure precision, packing and
containment remain follow-up issues. No statistical ranking or pressure
convergence claim is made from one realization per variant.

[Report, figures, videos and source/field audits](../../.tmp/dfsph-analytic-sloshing-2026-10-26/README.md).


### Performance follow-up — 26 October 2026

[Measured native DFSPH performance investigation](PERFORMANCE_2026-10-26.md):
shared-input CPU/GPU tank costs, bare/prepared sloshing windows, call audits,
CUDA trace, temporary bitwise geometry/transform ablations and the recommended
shared-operator optimization sequence. Checkpoint `529f431`; no production
performance changes or numerical-default changes in this investigation.

Particle-wall [startup scaling](STARTUP_SCALING_2026-10-26.md) now measures the
fixed overhead and CPU crossover explicitly, with a pinned boundary dependency
and unchanged solver paths.


## Follow-up: periodic conditioning and coupled wall pressures

The [27 October TGV investigation](TGV_2026-10-27.md) reproduces the IISPH
stop, separates divergence tolerance from near-lattice sampling, and measures
pressure work and viscous loss. Improved capped controls remain unqualified.

The opt-in [native pressure-boundary experiment](PRESSURE_BOUNDARIES.md)
extends Band's volume-centric fluid/solid pressure unknowns to both DFSPH
constraints and all three native compositions. It is a distinct sampled-wall
model with unresolved wall-pressure conditioning and flow dissipation; it
does not replace the default native IISPH self/contact operators.
