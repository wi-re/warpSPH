# Driven turbulence: native SPH source support — 2026-10-27

The case now supports **CRKSPH, CompSPH, Monaghan, GSPH and InutsukaGSPH**,
alongside MFM/MFV. This is CPU implementation qualification. GPU source checks,
uncontended performance, long chaotic statistics and resolution convergence
for the new routes remain open. PESPH has a distinct thermodynamic-variable
interface and is not included in this extension.

## Source and integrator policy

The forcing process, seed, modes, bath/adiabatic choices, mean-force removal and
physical-clock interpolation are shared across methods. Each fixed-mass SPH
RHS receives body acceleration and specific heat at every native stage. Native
hydro, viscosity and reconstruction formulas are unchanged. The default SPH
tableau is explicit midpoint (`simulation.integrationScheme=rungeKutta2`);
RK4 is also controlled. MFM/MFV retain their existing SSPRK source route.
No derivative reuse, state clipping, method-specific driving retuning or
failure-driven CFL retry is added. Native compatible pair-list replay remains
available and is tested to commit work/heat only once.

SPH uses the compressible examples' declared Owen support policy with grad-h
corrections disabled, B7 and n_h=4. Its native CFL is simulation.cflFactor=0.15
in this case; meshless solver.cfl remains 0.1. These are different native
criteria, not matched per-method timestep errors. Initial SPH masses are
rho0 times periodic cell volumes, density follows the native kernel/support
initialization, and every method starts at the same prescribed u0 and sound
speed. MFM/MFV continue to use meshless geometry for initial density. Native
SPH dissipation settings, including each Godunov reconstruction policy, remain
visible in resolved solver metadata.

Initially supported CRK policy: native energy partition/predictor and
summation-density evolution. Experimental CRK energy/evolution controls are
refused. All routes require eager execution and one finite-gamma ideal gas;
this is not an exact-isothermal solver.

### Compatible source work

[Frontiere et al. section 3.3](https://arxiv.org/html/1605.00725#S3.SS3)
uses half-kick velocities to balance discrete hydro pair work. That derivation
explicitly excludes external sources. The source extension below is a local
construction, not a claim that the paper specifies an OU thermostat.

For accepted Butcher weights b_s, let a_h and a_f denote hydro and body
accelerations, and q the specific thermal source. Use

```
A_h = sum_s b_s a_h,s      A_f = sum_s b_s a_f,s
v_half = v_n + dt/2 (A_h + A_f)
Delta K = dt sum_i m_i v_half,i . (A_h,i + A_f,i)
```

Native compatible hydro cancels its own pair contribution using this same
half-kick. Add `dt sum_s b_s q_s` to the compatible accepted internal energy.
The external budget is consequently

```
Delta W_force = dt sum_i m_i v_half,i . A_f,i
Delta Q_bath  = dt sum_s b_s sum_i m_i q_i,s
Delta E      = Delta W_force + Delta Q_bath       (to rounding)
```

Interim u still includes native hydro dudt and the thermal source. The native
pair-energy weights retain hydro dudt, rather than being overwritten with bath
heating. Source histories are accepted only after any native pair-list replay
has succeeded. Pair antisymmetry, compatible fractions and integrator weights
are retained.

The pinned local Spheral reference (116c71f2ea559d569456ba694194e5a6de3c75f7)
was checked at `Hydro/NonSymmetricSpecificThermalEnergyPolicy.cc` (pair work and
half-kick), `CRKSPH/CRKSPHBase.cc` (velocity/source registration) and
`ExternalForce/ConstantAcceleration.cc` (body-acceleration derivative). It
supports the separation of hydro pair work and body forcing as a reference
check. There is no claim of a matched Spheral OU/bath experiment or unchanged
Spheral external-work policy.

Monaghan/GSPH/Inutsuka evolve thermal energy with their native RK update.
Their budget uses stage quadrature of instantaneous force power and bath heat.
Physical `sum m(u+v²/2)` then has the native RK kinetic-energy truncation error;
it is not hidden by replacing the force budget with the measured energy change.
Accepted EOS fields use accepted u and the native retained stage density.
No additional density solve or changed density/support evolution is introduced.

## CPU controls

- Analytic uniform acceleration with and without a thermal perturbation,
  exponential relaxation, RK2/RK4, all five added methods.
- Cold strong-driving controls in relaxation and adiabatic modes; fresh stage
  indices/times and raw u/P positivity, not only floored sound speeds.
- An independent algebraic compatible-work control with simultaneous hydro,
  nonuniform body acceleration and bath heat.
- Forced compatible-energy replay: fields, work/heat and OU generator state
  agree without committing the sources twice.
- Native checkpoint splits cross forcing knots and preserve fields/budgets
  bitwise on the tested CPU/source/precision controls. SPH checkpoints retain
  padded neighbor indices, CSR offsets/counts and the original reference
  positions/supports so pair order and list validity history survive restart.
  Missing neighbor history is refused. Derived Q is an output view for SPH,
  never a second integrated state. Complete frame-level forcing remains saved.

**45 focused FP64 tests passed** (53.32 s). The expanded float32 suite including
previous meshless/source controls passed 133 tests before the final replay
controls. The final broader CPU invocation passed **395 tests**, with 12
GPU-only skips and 10 explicitly CUDA-only cases deselected (124.61 s). It
covered new source controls, meshless lifecycle, configuration/runner, PESPH
integration, Monaghan energy balance and CRK operators/energy controls.
GPU checks are deferred while the independent GPU job runs.

A separate same-input comparison with pre-extension commit 3e57222 reproduced
all saved fields bitwise for MFM/MFV driven controls and source-free Sod controls
of CRKSPH, CompSPH, Monaghan, GSPH and InutsukaGSPH. The final API adjustment only
appended the optional system field to preserve positional construction; it does
not change the recorded pilot/refinement arithmetic.

## Developing-flow pilots

Ten CPU runs use nx=24 (576 particles), t=0.5, seed 1234, relaxation time 0.2,
maxDt=0.002, native midpoint, B7/n_h=4 and acceleration inputs 0.3/6. Software
VisPy recordings are on; every RHS and accepted state is observed. All finish,
all source inventories stay unchanged, and all 5,022 RHS checks are finite and
positive across 2,511 accepted steps. These are developing weak/mildly
supersonic flows, not the paper's 3D Mach targets or a stationary dataset.

| Method | Driving | Steps | Final mass-weighted Mach | Min raw RHS u | Max absolute budget residual / E0 |
| --- | --- | ---: | ---: | ---: | ---: |
| CRKSPH | weak | 250 | 0.09616 | 0.89075 | 3.544e-08 |
| CRKSPH | strong | 256 | 1.07285 | 0.38568 | 5.204e-08 |
| CompSPH | weak | 250 | 0.09383 | 0.89371 | 3.851e-08 |
| CompSPH | strong | 250 | 1.09299 | 0.55109 | 3.364e-08 |
| Monaghan | weak | 250 | 0.10024 | 0.89138 | 1.840e-08 |
| Monaghan | strong | 252 | 1.12702 | 0.52127 | 9.372e-05 |
| GSPH | weak | 250 | 0.09852 | 0.89274 | 1.937e-08 |
| GSPH | strong | 253 | 1.15733 | 0.51762 | 1.028e-04 |
| InutsukaGSPH | weak | 250 | 0.09768 | 0.89310 | 1.621e-08 |
| InutsukaGSPH | strong | 250 | 1.00087 | 0.61443 | 5.832e-05 |

The larger non-compatible strong-flow residuals require timestep accounting;
finishing with positive states alone is not their accuracy gate. They are
consistent with the independent refinement below. No common tight compatible
energy tolerance is asserted for these distinct native time updates.

CPU observed wall costs below include setup after system construction,
per-RHS observers, accepted diagnostics, snapshots and software video. Other
CPU validation ran concurrently. These are not GPU rankings or clean solver
throughput measurements, and there is no matching earlier CPU pilot baseline.

| Method | Driving | Observed wall s | Wall ms / accepted step |
| --- | --- | ---: | ---: |
| CRKSPH | weak | 38.26 | 153.06 |
| CRKSPH | strong | 43.18 | 168.69 |
| CompSPH | weak | 34.10 | 136.41 |
| CompSPH | strong | 37.58 | 150.32 |
| Monaghan | weak | 35.41 | 141.64 |
| Monaghan | strong | 50.97 | 202.26 |
| GSPH | weak | 33.33 | 133.32 |
| GSPH | strong | 44.32 | 175.18 |
| InutsukaGSPH | weak | 22.31 | 89.23 |
| InutsukaGSPH | strong | 32.94 | 131.74 |

### FP64 temporal refinement

Fifty tiny CPU A/B controls (video omitted; step/time progress retained) cover
analytic ramp acceleration plus relaxation at dt caps 0.004/0.002/0.001/0.0005,
and weak/strong OU flow to t=0.2 at caps 0.004/0.002/0.001. The analytic case is
nx=12; OU controls are nx=16. The prescribed analytic force is spatially
uniform a(t)=(0.5,-0.3)+t(0.4,0.2), with initial u=1.26 and bath u0=0.9.
All 7,250 steps / 14,500 RHS evaluations remain positive (minimum raw u=0.7681).

Every method's analytic thermal error falls from 8.963e-6 to 2.224e-6,
5.539e-7 and 1.382e-7: second-order convergence. Compatible energy budgets stay
near FP64 roundoff (at most 1.75e-15). Ordinary RK analytic energy-budget error
falls by four per halving, from 1.016e-7 to 1.587e-9.

For the nonuniform strong OU flow, max budget residuals are:

| Method | maxDt 0.004 | maxDt 0.002 | maxDt 0.001 |
| --- | ---: | ---: | ---: |
| Monaghan | 3.852e-05 | 1.015e-05 | 2.545e-06 |
| GSPH | 4.060e-05 | 1.066e-05 | 2.668e-06 |
| InutsukaGSPH | 4.492e-05 | 1.177e-05 | 2.929e-06 |

Velocity-field errors against the finest control also decrease, but not at a
uniform second-order rate. At maxDt=0.002 the strong-flow relative velocity
L2 differences from maxDt=0.001 are 3.91e-5 (CRKSPH), 3.87e-5 (CompSPH),
1.57e-3 (Monaghan), 4.35e-5 (GSPH) and 3.36e-4 (Inutsuka). The finest control
is not analytic truth; dissipation/switch/spatial effects still need their own
longer and finer qualification. This study validates source integration and
its timestep dependence, not converged turbulent statistics or method rankings.

## Reproduce and remaining gates

Use `examples/configuration/drivenTurbulenceCRKSPH.yaml` for the new case route.
The maintained qualification harness now accepts all seven methods; e.g.:

```bash
CUDA_VISIBLE_DEVICES='' LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
PYTHONPATH=src /home/lu26029/miniconda3/envs/warp/bin/python \
  scripts/qualify_drivenTurbulence.py --scheme CRKSPH --device cpu --nx 24 \
  --time .5 --max-dt .002 --plot-interval 25 --acceleration 6 \
  --root .tmp/new-driven-crk-pilot
```

Each root must be fresh. Source/input/driver archives, initial/final/sampled
arrays, history, stage minima, forcing state and video are preserved under
`.tmp/driven-sph-implementation-20261027/`. `pilots-v2` is the complete cohort;
`refinement` contains the full FP64 controls. The original failed harness
attempt is retained: its observer hid the `stageIndex` signature required by
the native compatible driver. It was fixed without changing hydro stages.
Early restart tests exposed missing native neighbor history; that was fixed
before the complete cohort. Logs, summaries, source hashes and old/new parity
arrays accompany the final validation.

Next gates: GPU source/restart controls when free; controlled performance
windows with solver/observer/I/O costs separated; longer matched-seed runs;
multiple resolutions; dissipation and density/velocity statistics; sufficient
correlation-time coverage. Existing MFM/MFV long-run qualification and its
stationarity caveats remain unchanged. These new CPU rows do not inherit it.
