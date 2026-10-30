# Driven-flow forcing and periodic-domain audit

2026-10-28. Tracked under [driven compressible turbulence](roadmap/driven_compressible_turbulence.md)
and the existing [plan index](../PLANS.md). No hydrodynamic trajectories or physical
source changes were made for this investigation.

**The audit finds no substantial periodic seam defect in the tested field evaluation.**
The forcing design is a plausible contributor to the common large-scale behavior:
it directly drives the box modes and prescribes stochastic acceleration, so its
mechanical input power depends on the flow. The saved CRKSPH192² and256² runs
receive mean powers differing by a factor2.175 despite identical OU histories.
This does not establish causality: numerical dissipation and evolving velocity
also change the correlation with the force. None of these trajectories is yet
stationarity-qualified.

## Evidence and scope

The offline [audit driver](../scripts/audit_drivenForcing.py) reads all14 overnight
64²/128² runs and the two new CRKSPH192²/256² runs. All16 have identical forcing
parameters and803 bitwise-identical raw OU knots, covering t=0..40. The live
Fourier evaluator, primitive source adapter, conserved source, driven case and
domain constructor are byte-identical to those in the frozen run snapshot.
Their SHA256 values and the input hashes are retained with the evidence.

The evaluator is checked against an independent NumPy FP64 construction of the
integer modes, projection, interpolation, normalization, real Fourier sum and
analytic spatial Jacobian. CPU FP32/FP64 checks cover87 times,129 points per
signed box edge,513 random positions per time, integer translations of up to
two boxes,800 knot boundaries and five quadrature grids16²..256². Nine saved
particle states per run (144 total, t≈0,5,...,40) also test the deformed particle
coordinates and actual masses/velocities. Snapshot times are read from the files.
Every accepted-step history supplies cumulative input work and bath exchange.

The quantitative roundoff measurements here are **CPU** evaluations of the same
field source at retained coordinates. The GPU was occupied by another process,
so no concurrent CUDA check or rollout was launched. These checks do not certify
CUDA-specific arithmetic, hydrodynamic neighbor correctness or the eventual
statistical impact of small perturbations in a chaotic trajectory.

Local evidence is in
`.tmp/driven-forcing-audit-20261028/wrapped-comparison/`:
`audit.json`, `forcing-history.npz`, `forcing-power.png/.pdf`,
`tests.log` and `source-verification.json`. The initial audit is retained in
the parent directory. Its driver hash predates the added wrap comparison;
the final subdirectory's driver hash matches the saved script.

## Periodicity, modes and clock

Both boundaries and forcing declare L=1 with the physical box[-0.5,0.5]².
The six independent real Fourier modes are
`(0,1), (0,2), (1,-1), (1,0), (1,1), (2,0)`.
Each has wavevector2πn/L; sine and cosine coefficients already represent both
signs of a mode. There is no missing negative-wavevector partner or duplicated
Fourier normalization. Integer modes are periodic regardless of the coordinate
origin or an integer box translation.

| Maximum absolute discrepancy in the tested field | CPU FP32 | CPU FP64 |
|---|---:|---:|
| Opposite-edge acceleration | 2.83e-7 | 5.00e-16 |
| Opposite-edge spatial Jacobian | 2.03e-6 | 3.11e-15 |
| Independent acceleration at random points | 2.56e-7 | 2.22e-16 |
| Independent spatial Jacobian | 2.26e-6 | 2.22e-15 |
| Integer-translated acceleration | 1.62e-6 | 2.16e-15 |

At t=23.173, uniform-grid spatial RMS is0.20208777 at every tested resolution,
matching Parseval's coefficient sum. Power outside the declared modes is at
most1.68e-14 of the total on these FP32 grids. There is no measured resolution
aliasing of this low-mode forcing. The solenoidal coefficients have the expected
perpendicular projection; divergence in the derivative checks is zero in these
FP32 evaluations and at most1.12e-15 in FP64.

The fixed clock has spacing0.05 and correlation time0.5. Queries interpolate
retained chronological OU knots; changing RHS query order does not change the
realization. The maximum coefficient difference across adjacent FP64 times on
either side of a knot is3.77e-14. Linear interpolation is continuous, with
piecewise slopes that change at knots. Rounding the queried time to FP32 changes
the evaluated acceleration by at most2.63e-6 in this test set. There is no
observed force jump at the knot boundaries.

## Unwrapped SPH coordinates

Saved SPH positions extend across many periods (up to |x|≈23.70 in the sampled
states), whereas MFM/MFV store positions inside the box. The Fourier formula
remains periodic analytically. FP32 products with large phases do accumulate
more rounding error: the largest CPU evaluator error against the independent
FP64 field at these actual represented coordinates is6.05e-6. The independent
FP64 wrapped and unwrapped answers agree within1.08e-14.

A diagnostic evaluation after wrapping coordinates into the box reduces the
largest field discrepancy to3.25e-6 for this particular wrap implementation;
wrapping itself also rounds FP32 coordinates. The maximum instantaneous input
power discrepancy attributable to the original field evaluation is3.02e-8
across the sampled states, falling to9.78e-9 with this diagnostic wrap. These
are far below the observed mean power differences of order1e-2.

This identifies a finite-precision issue worth retaining as a control for much
longer integrations, rather than evidence for a large seam artifact. Wrapping
the field input cannot recover precision already lost while evolving unwrapped
particle coordinates. No production position or phase policy was changed.

Actual mass-mean acceleration subtraction is spatially constant and preserves
periodicity and spatial derivatives. Its direct contribution to power is
`-M mean(v)·mean(a)`; the largest magnitude among the sampled states is2.06e-8.
Thus the mean-removal correction itself is also small in the measured energy
input, though its value depends on the evolving particle distribution.

## Acceleration amplitude and mechanical input power

The prescribed0.3 is an ensemble RMS acceleration parameter at OU knots. It is
not a per-query amplitude normalization, constant power constraint or Mach
controller. For the retained realization over10..40:

- Continuum spatial acceleration RMS, sampled every0.005 in time, ranges0.1225..0.4715.
- The square root of time-mean acceleration variance is0.29774.
- Direct box modes(|n|=1) carry36.27% of time-integrated acceleration variance.
  This is a force variance fraction, **not** a fraction of injected energy.
- Linear knot interpolation attenuates ensemble variance by
  `(2+exp(-0.05/0.5))/3 = 0.96828`, or about1.6% in RMS. This is the declared
  finite-clock approximation and does not explain a large seam effect.

Instantaneous work with mass-mean removal is

`P(t) = sum_i m_i [v_i - mean_mass(v)] · a(x_i,t)`.

Identical stochastic coefficients therefore do not imply identical power:
positions, masses and velocity-force correlations all enter. The native SPH
compatible-work adapter uses its accepted half-kick work; the table below uses
that accumulated diagnostic, not a sparse snapshot approximation to it.

| CRKSPH resolution | Mean input power10..40 | Mean signed bath power10..40 | Mean K10..40 |
|---|---:|---:|---:|
| 64² | 0.02647 | -0.02361 | 0.23259 |
| 128² | 0.03171 | -0.02964 | 0.22300 |
| 192² | 0.01602 | -0.01689 | 0.19768 |
| 256² | 0.03484 | -0.03390 | 0.25489 |

Mass is1 to the recorded run precision. Cumulative-work endpoints at10,20,30,40
are interpolated from the dense accepted-step histories. Input work, total
energy change and bath exchange retain their signs; the bath is not mistaken
for a direct kinetic drag. Across all16 runs mean power spans0.00928..0.03484.
The complete three-block and snapshot comparisons are in `audit.json`.

There is no secular increase of force variance common to all three measurement
blocks: means of acceleration squared are0.08430,0.09333,0.08832. Thus a simple
increasing-amplitude forcing bug is not supported either. Different input work
is an energy-balance observation, not independent proof that the forcing causes
the non-monotonic K trend. It cannot separate force-flow feedback from changing
numerical dissipation or insufficient statistical sampling.

![Common forcing and CRKSPH input power](/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH/.tmp/driven-forcing-audit-20261028/wrapped-comparison/forcing-power.png)

## Physical interpretation and the next discriminating control

Direct forcing of |n|=1, a small number of stochastic modes, two-dimensional
periodic geometry and the absence of prescribed large-scale momentum drag
make coherent, slowly varying box-scale motion a plausible outcome. There is
no specified physical viscosity here either, so resolution changes the effective
numerical dissipation. None of this proves an inverse cascade: determining the
direction of energy transfer requires a transfer/flux measurement, especially
because the box modes are directly forced.

[Hopkins §4.4.4](https://arxiv.org/html/1409.7395#S4.SS4.SSS4) uses3D periodic
isothermal driven turbulence; its apparent 2D illustrations are slices of3D
boxes. Our2D finite-gamma relaxation problem shares the low-mode OU idea, but
its steady-state expectations cannot be borrowed directly from that benchmark.
[Chan, Mitra & Brandenburg](https://arxiv.org/abs/1109.6937) study finite-box2D
condensates and their sensitivity to viscosity and spectral resolution. That
is motivation to check box-scale energy balance, not a validation of this
compressible run or a measured cascade.

The local [Spheral stirring example](/home/lu26029/dev/spheral/tests/functional/Hydro/Turbulence/Stir-3d.py:207)
constructs projected3D FFT perturbations and normalizes the perturbation cube.
Its shown callback only constructs that cube. It does not establish equivalent
OU time evolution, constant mechanical power or this compressible energy budget.

Before another resolution campaign, the useful controlled comparison is one
scheme/resolution with a forcing band excluding |n|=1, preserving acceleration
scale, thermal model and integrator. Compare box-mode energy and block power,
then test another seed. That is a declared forcing-model experiment; it is not
a silent correction to these runs. Large-scale drag, fixed physical viscosity
and constant-power forcing are separate choices to assess individually. A
CUDA field-only check when the GPU is idle would close the remaining backend
qualification gap without another long trajectory.

## Regression checks

The new [field tests](../tests/test_forcingPeriodicity.py) cover a closed-form
three-mode divergence-free field, signed-seam values/Jacobians at unit and
nonunit L in both precisions, late-clock knot continuity, and coordinates
spanning24 box crossings. Together with the existing OU/projection/order/
checkpoint/bound checks, **15 selected CPU tests pass**. Native hydrodynamic
sources and all retained trajectories remain unchanged.
