# Small thermal controls — 2026-10-27

The audited boundary and heat-transport limitations can be measured with small
CPU problems. A reusable [probe](../scripts/probe_thermalBoundaries.py) now
separates three effects without running a gravity/interface instability or
large convection simulation. No production numerical model or default changed.

## Measured controls

| Toy | Size | Quantity isolated |
|---|---|---|
| Uniform prescribed heating | 64 and 128 actual 1D gas particles | Initial, stage and accepted u errors; caloric heat required versus actual energy change |
| Smooth-density prescribed heating | 64 and 128 particles, PESPH E/A | Whether converting the target to the authoritative variable actually preserves u after closure |
| Thermal sine wave | 32 and 64 fixed 1D particles | Decay rate, effective diffusivity, mode distortion, total heat and entropy production |
| Isobaric temperature variation | 64 fixed particles | Whether the artificial operator transports heat when pressure is constant |
| Pressure reservoir under expansion | 8 particles, prescribed volumes | Fixed pressure versus fixed internal energy/temperature |

The boundary toys reuse the periodic mirrored Sod sampler with equal left/right
gas states: **nx=32/64 means 64/128 actual particles**, unlike the separately
constructed conduction lattice. All measured controls use CPU and one Torch
compute thread. The full final cohort is 92 diagnostics per precision, 184
across FP32/FP64, with 2,640 accepted/integrated steps and zero execution errors.
These are completed diagnostics, not 184 qualification passes: some intentionally
expose existing incorrect boundary behavior. Nine diagnostic/native-stage tests
pass in each precision.

Tiny controls intentionally omit video, retain stage/accepted histories and
field arrays, and stream progress. They use a single ideal gas with gamma=1.4
and caloric units c_v=1, so u=T. They do not qualify material-dependent heat
capacities or multiple EOS thermal boundaries.

## Prescribed heating: a lifecycle failure, independent of RT

Hold positions and velocity fixed and prescribe `u_target=1+t` everywhere to
t=0.02. The whole-domain thermostat is a manufactured constraint, not a physical
wall or a stationary Euler solution. The uniform setup removes spatial pressure
work and RT's initialization, gravity and buffer geometry. Repeat at dt=0.002
and 0.001 with native explicit midpoint RK2 and RK4 for ordinary/PESPH schemes;
meshless uses its native SSPRK stages and finalization.

| Scheme | Result |
|---|---|
| CRKSPH, CompSPH, Monaghan, GSPH, Inutsuka | Every sampled RHS holds the target; accepted u remains 1 instead of 1.02. Relative error 1.960784%, unchanged by dt, resolution or RK2/RK4. |
| MFM, MFV | Accepted target error <=1.12e-11 FP64 and <=1.36e-7 FP32 on the fixed geometry. |
| PESPH PressureEnergy | Accepted target error <=2.18e-16 FP64; <=1.37e-7 FP32 including smooth-density controls. |
| PESPH PressureEntropy | Uniform-lattice target error 1.91e-7 FP64; with rho=1+0.2 cos(pi x), maximum error 2.61046e-4 at nx32 and 6.53132e-5 at nx64. |

The ordinary-SPH uniform gas has total mass 2. Its prescribed temperature rise
requires heat 0.04; the accepted state gains none. The probe reports this
**missing thermostat heat**, not a violation of closed-system energy conservation.
It also reports the additional reset heat needed to correct any initial target
mismatch, so a startup correction is not mistaken for ramp power. Meshless and
PESPH E include their small startup differences in this accounting; MFM/MFV
FP64 residual including startup is <=2.28e-11. No boundary heat ledger is added
to the production schemes by this diagnostic.

PESPH entropy's smooth-density error decreases with resolution but persists at
a fixed resolution when dt decreases: the closure conversion, rather than
Runge–Kutta accuracy, is being measured. Global heat agreement alone would miss
these local target errors.

Implementation detail of the measurement: the existing meshless Sod factory
requires an adaptive startup timestep when called without a dt argument. The
probe permits that factory initialization, then restores the exact requested
fixed dt **before any measured step**. Native stage flags are forwarded intact;
meshless stage mass/velocity/u are measured from Q and its fresh context rather
than absent or retained public diagnostic fields.

## Heat transport: conservation is not calibrated diffusivity

Freeze particle positions, density and mechanical evolution. Start from
`u=1+A sin(pi x)` on a periodic interval of length 2. Fix compact support at
4dx, the operator's signal speed at 1, alpha=1, Cu_l=1, Cu_q=0 and the overall
artificial-conductivity multiplier at 0.5. Evaluate the existing shared
conductivity kernel directly, as used by Monaghan/dissipative PESPH; this is
not a claim that CRK/MFM invoke it or that a full gas evolves with fixed sound
speed. No artificial operator is relabeled as a physical heat operator.

Project the rate r onto the initial perturbation f=u-1, with mass weights:

    lambda_fit = -sum(m f r) / sum(m f^2)
    chi_fit = lambda_fit / pi^2
    nonFourierFraction = ||r + lambda_fit f||_m / ||r||_m

For physical linear constant-coefficient heat diffusion, one expects the sine
mode to remain a sine with prescribed decay `exp(-chi pi^2 t)`. Here chi is
**measured from the numerical operator**, not supplied as a physical property.
The measurement checks whether that fit depends on resolution or amplitude.

FP64 fitted chi, A=0.1:

| Existing artificial formulation | N=32 | N=64 |
|---|---:|---:|
| Price2012_98 label | 0.0144151 | 0.00729487 |
| Price2008 | 0.00167463 | 0.000601871 |
| Hopkins2015 | 0.00101454 | 0.000259092 |

At N=64, increasing A from 0.01 to 0.1 leaves the Price2012_98 rate unchanged,
increases Price2008's fitted chi from 0.000190328 to 0.000601871, and increases
Hopkins2015's from 0.0000258585 to 0.000259092. The latter two rates also contain
components outside the initial Fourier mode: respectively about 34% and 52% of
the rate norm at A=0.1. These measurements distinguish nonlinear shock/contact
smoothing from a constant material diffusivity. The Price2012_98 label is the
existing code enum; the toy does not change its historical formula attribution.

A matched equal-pressure probe sets rho=1/u, m=rho dx, and explicit pressure
exactly 0.4. Price2008 and Hopkins2015 then give **zero rate**, despite varying
u/T. This follows their pressure-contrast dependence. Exact prescribed pressure
avoids amplifying floating-point pressure noise through a square root. The
Price2012_98 operator still transports heat, with fitted chi≈0.00725808.

Four small explicit-midpoint conduction evolutions use N=64, A=0.1, t=0.02,
dt=0.0005/0.00025 and the first two formulations. FP64 energy changes remain
below 1.95e-16. The Price2012_98 endpoint matches the exponential using its
own fitted rate to L2 errors 2.20e-14 and 5.63e-15. Price2008 has a roughly
3.02e-6 mismatch at both timesteps, consistent with its non-Fourier rate.
This is an integration/operator consistency check, **not** validation of a
chosen physical diffusivity.

FP32 reproduces the effective-rate trends, but these very small per-step
thermal increments accumulate rounding error: the Price2012_98 L2 errors are
9.76e-7 and 1.81e-6, and maximum absolute energy change across the four evolutions
is 1.11e-6. Halving dt therefore does not improve this FP32 control. The toy
separates this precision limit from the method's resolution-dependent heat rate.

## Reservoir volume: pressure is not temperature

Call the existing `DrivenParticles` prescription with fixed masses, zero
velocity and constant pressure, while supplying volume ratios 1, 1.1 and 1.25.
The current implementation sets `E=P V/(gamma-1)`. It holds pressure to roundoff
while u changes from 1 to 1.1 and 1.25. Thus a 25% volume increase gives a 25%
temperature increase under that pressure prescription. This is correct for its
specified pressure-reservoir contract, but is not a fixed-temperature boundary.
Volumes are supplied analytically: no hydro or expansion-flow accuracy is claimed.

The fixed-geometry MFM/MFV thermostat above holds u because P can be prescribed
using its initial, unchanging geometry density. A moving/flexible-volume thermal
boundary needs the current geometry or a direct u/E prescription.

## Next small physical gates

After the boundary-state repair and physical transport implementation, keep the
next qualification small:

1. A 1D hot/cold slab (64/128 particles): analytic conductive profile, boundary
   temperature error, wall heat flux and net-energy balance. Add a sinusoidal
   transient with **prescribed** chi and check spatial/temporal convergence.
2. An insulated 1D layer: zero net external heat, conservative internal exchange,
   entropy production, and no spurious heating at uniform temperature.
3. A gravity-supported conductive layer: equilibrium residuals and startup
   transients before buoyancy instability; verify work and heat together.
4. A modest 2D convection-onset box, followed by resolution refinement before
   grading Nusselt transport or moving to chaotic convection.

The first three gates can stay on CPU. Large GPU runs are useful after those
contracts hold; they are not needed to diagnose the current implementation gaps.
Physical wall flux, an insulated-wall operator and convection onset are not
implemented or qualified by this probe. WCSPH/incompressible thermal evolution
remains a separate model extension.

## Reproduction and evidence

From this worktree with the warp environment (CPU only):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=src WARP_CACHE_PATH=/tmp/warpsph-warp-cache \
/home/lu26029/miniconda3/envs/warp/bin/python -u scripts/probe_thermalBoundaries.py \
  --root .tmp/thermal-toys-new-f64 --precision float64 \
  --nx 32 64 --dt .002 .001 --time .02
```

Repeat with a fresh root and `--precision float32`. `--sections boundary`,
`conduction` or `reservoir` selects a smaller subset. Exit status reports
execution errors; inspect measured errors rather than interpreting zero exit
status as a passing thermal-boundary qualification.

Final evidence is in `.tmp/thermal-toys-20261027-final-{f64,f32}/`: exact copied
driver, source archives/hashes, arguments, per-control results, accepted/stage
histories and NPZ fields. Source inventories remain unchanged across each
cohort. Particle counts are independently verified from saved fields. Earlier
harness-only failures remain under the smoke/f64/v2/v3 directories and logs;
they are excluded from the final results. They concerned unused schema inputs,
report-key duplication, meshless factory/stage arguments and copied-field
observation, not numerical-method failures.

The 92 controls total 14.36 s FP64 and 14.13 s FP32 of measured control wall time,
plus process imports/source capture. Native boundary runs cost 0.055–3.06 s each,
about 4.9–306 ms/step including setup and observations. All 19 heat-operator
controls total about 0.15 s per precision; the explicit evolutions cost roughly
0.48–0.56 ms/step. The algebraic reservoir probe costs <0.5 ms. These are small
CPU diagnostic costs, including warm/cache-loading differences, not scheme
throughput rankings. The independent GPU job was not used or interrupted.

[Test controls](../tests/test_thermalBoundaryToys.py): analytic caloric accounting,
known Fourier eigenmodes, zero transport versus harmonic contamination, the
native pressure-reservoir law, and two-step MFM/MFV/PESPH-E stage observations.
Nine pass in FP32 (1.16 s) and nine in FP64 (1.16 s). See the preceding
[boundary audit](thermal_boundary_audit_2026-10-27.md) for the RT-specific findings.
