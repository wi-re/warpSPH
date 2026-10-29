# Overnight driven compressible cohort: analysis, 2026-10-28

Plan index: [driven turbulence and multiphysics](../PLANS.md).

**All fourteen runs complete and numerically viable in the measured scope;
stationary training-data and resolution qualification remain open.** This adds
long GPU evidence for MFV and the five SPH routes. It does not establish an
accuracy ranking or select a new default. Analysis used saved results on CPU;
no simulations were started and the original campaign evidence was preserved.

## Evidence located and independently checked

The serial queue finished at 05:30:57 London time on 2026-10-28. Raw campaign:
[/home/lu26029/dev/warpSPH/.tmp/overnight-acsph-driven-2026-10-27](/home/lu26029/dev/warpSPH/.tmp/overnight-acsph-driven-2026-10-27).
The [launch/completion record](/home/lu26029/dev/warpSPH/docs/acsph/OVERNIGHT_2026-10-27.md)
identifies its frozen sources and actual execution order. Source digest, not the
absence/presence of a clean checkout, is the authority for these archived runs.

The analysis reconciles fourteen individual reports against all 554,267 accepted
steps and 1,108,534 RHS checks. Every run reached exactly t=40, its reported stage
count is twice its accepted steps, its accepted times advance monotonically,
and its reported energy residual agrees with independent reconstruction from
initial conserved energy and accepted force-work/bath-heat histories. The saved
sample mean and dense-history mean differ by at most 1.7e-6 for Mach and 1.2e-6
for K; sparse sampling does not explain the large differences discussed below.
All per-RHS validity/positivity guards pass; this is not an audit of every
internal reconstructed Riemann state or every possible limiter/floor activation.

All 2,106 manifest-pinned files remain unchanged. All fourteen runs record the
same source inventory digest:
`e834516dc49825e6774840b64153ea2d78a6c43f47a541065e0b5b3e30beea8a`.
The launch record pins warpSPH to 8832040 plus dependency snapshots. The per-run
Git metadata instead resolves to the enclosing dev checkout (5d4f41d, dirty),
because these source copies live under that repository's .tmp directory. Do not
mistake that discovery artifact for the actual source revision: use the frozen
source copies, file hashes, manifest and original launch snapshot together.

The final forcing checkpoints retain 803 finite OU knots per run, including
the required bracketing knots through t=40. Parameters and all common raw knots
match bit for bit across the fourteen runs. This confirms a matched stochastic
input, with full past knot data available; it does not replace qualification of
production per-frame conditional export.

Both repeated MFM runs match their earlier cohort's final positions, Q, support,
mass, density, velocity, pressure, u and source budget **bit for bit** at both
resolutions. They reproduce prior evidence rather than add independent samples.

## Physical and numerical scope

All runs start from rest in the unit periodic 2D box, seed1234, gamma=5/3,
rho0=1, initial sound speed1/u0=0.9, OU acceleration amplitude0.3,
correlation time0.5, knot interval0.05, solenoidal modes1–2, mass-mean force
removal and finite-time thermal relaxation0.2. This is a finite-gamma cooled
gas, not an exact-isothermal system. No drag, Mach feedback or physical
viscosity/conductivity is added. One seed is shared across schemes/resolutions.

MFM/MFV use native two-stage SSPRK, CFL0.1 and neighborSkin1. SPH uses native
midpoint, CFL0.15 and its declared support/dissipation/energy policy. These are
native-policy comparisons, not matched temporal or spatial error budgets.
**Resolved GSPH uses shockSwitchC=0; Inutsuka uses shockSwitchC=3.** The latter
has velocityPairPolicy=Limited. This cohort must not be relabelled as the
matrix's GSPH+c=3 or an unverified Inutsuka-tangent variant. No switch A/B was
run here; no inference about the switch's causal effect follows from these rows.

Means below use physical-time quadrature over t=10..40. K differences are signed
symmetric fractions: 2(K_128-K_64)/(K_128+K_64). Run wall includes integration,
observers, sample compression/progress and software-rendered recording; the
independent warmup/setup is excluded. All physical runs had video enabled.

| Scheme in this cohort | Mean Mach 64² | Mean Mach 128² | K difference 64²→128² | 128² run wall |
|---|---:|---:|---:|---:|
| MFM | 0.552 | 0.673 | +39.0% | 10.52 min |
| MFV | 0.550 | 0.733 | +58.5% | 10.58 min |
| CRKSPH | 0.679 | 0.673 | -4.2% | 8.05 min |
| CompSPH | 0.579 | 0.660 | +25.3% | 6.51 min |
| Monaghan | 0.578 | 0.619 | +15.9% | 8.82 min |
| GSPH (c=0) | 0.571 | 0.545 | -7.5% | 7.24 min |
| Inutsuka (c=3) | 0.552 | 0.577 | +8.1% | 7.79 min |

CRKSPH and GSPH have smaller paired K differences in this realization, but none
passes the stationarity screen; e.g. CRKSPH's mass-weighted log-density spread
still changes by 17.8%. One matched seed cannot distinguish numerical bias from
finite-time chaotic divergence and stochastic variability. MFM's earlier weak64
seed range (mean Mach0.552–0.764) already contains the new fine-run means. No
method is selected as more accurate or less dissipative from this table.

## Stationarity, observations and training-data implications

None of the fourteen runs passes the existing screen: three ten-unit blocks
with <=10% relative spread in Mach, K and mass/volume log-density spread,
uncensored correlation estimates, sufficient block length relative to correlation
time, and >=10 box crossings. K block spreads span 19.7–75.5%; MFV128's K block
means rise 0.1703→0.2859→0.3809. The screen cannot by itself distinguish secular
evolution from inadequate sampling, but the runs do not meet its acceptance rule.
Only about 4–10 effective velocity-field samples are estimated in the assessment
window, despite 401 stored snapshots per trajectory.

The exact box-wavelength modes k²=1 contain 83.7–92.2% of the deposited velocity
spectral power in the 32² observation. The first rounded radial shell, which also
includes diagonal k²=2 modes, contains 91.6–96.3%; it is a different diagnostic.
These are filtered velocity-power fractions, not density-weighted kinetic-energy
fractions. The large low-mode contribution is consistent with box-scale flow;
it does not establish spectral transfer direction or an inertial-range cascade.

A common 32² observation still has varying filter radii for coarse particle
flows (H can exceed0.0625), whereas fine runs use H=0.0625 throughout. The 16²
observations provide a stricter matched filter here: **all runs use constant
H=0.125**, with conservative per-particle normalization. Comparing shells1–4
there still gives symmetric spectral L1 differences of 39.7% for MFM and 59.1%
for MFV, so the large low-mode resolution differences cannot be explained only
by unequal observation radii. This very broad filter says little about small
scales. Across methods, mass log-density PDF total variation between resolutions
ranges 1.8–13.1%. Neither matching a mean nor a smooth spectrum establishes
resolution convergence in this nonstationary single-seed cohort.

Weak driving also does not guarantee everywhere-subsonic flow. MFV128 reaches
instantaneous mass RMS Mach1.016; final local Mach exceeds1 in MFM/MFV/CRKSPH/
CompSPH/Monaghan. At t=40, about 12.6–17.8% of their fine-run mass is locally
supersonic. GSPH/Inutsuka have no supersonic particles in that final snapshot;
this does not classify their entire histories. Any training label must say
whether subsonic means time-averaged RMS, instantaneous RMS or a local bound.

![Mach and kinetic-energy evolution](../.tmp/driven-cohort-analysis-20261028-final/evolution.png)

## Conservation, positivity and source controls

| Scheme | Max relative energy residual, 64² / 128² | ms/step, 64² / 128² | Steps, 64² / 128² |
|---|---:|---:|---:|
| MFM | 1.91e-07 / 5.67e-07 | 9.07 / 9.20 | 34,633 / 68,606 |
| MFV | 1.62e-07 / 7.53e-08 | 9.16 / 9.24 | 34,622 / 68,694 |
| CRKSPH | 7.36e-08 / 1.09e-07 | 9.58 / 9.75 | 23,583 / 49,550 |
| CompSPH | 8.40e-08 / 1.59e-07 | 8.22 / 8.48 | 21,971 / 46,060 |
| Monaghan | 2.23e-05 / 1.10e-05 | 11.43 / 11.58 | 22,993 / 45,660 |
| GSPH | 1.23e-05 / 5.12e-06 | 9.17 / 9.42 | 22,828 / 46,083 |
| InutsukaGSPH | 9.82e-06 / 6.50e-06 | 9.87 / 10.10 | 22,694 / 46,290 |

The worst accepted energy-budget residual is2.23e-5 of initial energy (0.00223%).
Compatible CRKSPH/CompSPH have much smaller residuals; ordinary midpoint source
integration is allowed truncation error, not falsely advertised as exact work
conservation. Small residuals alone do not measure flow accuracy or stationarity.
MFV's maximum relative total-mass drift is7.61e-7; other schemes keep particle
masses fixed. Fine MFV endpoint particle masses range0.970–1.045 of initial mass,
confirming that its mass-exchange route is active. All raw checked u/P remain
positive, with no source-induced failure in this cohort.

The GPU tests originally report 50 passes and five failures. The five failures
are export assertions comparing a CPU HDF5 tensor against a CUDA expected tensor.
The postflight copy changes exactly one expression to `.cpu()` on the expected
tensor, with rtol=atol=0 unchanged. Its source hashes, exact diff, receipt and
JUnit XML agree, and all five pass. Thus all 55 controls have passing evidence;
the original failure status/logs remain valid historical records. No production
solver or physical trajectory was changed by that correction.

## Performance result and neighbor caching

At 128², CompSPH finishes in6.51min versus MFM10.52min (1.62x physical-time
throughput); CRKSPH8.05min is1.31x MFM's throughput. MFM/MFV take about 68.6k
steps versus 45.7–49.6k for SPH: ms/step alone misses this difference. Timesteps
are not error-matched, and the trajectories/dissipation differ, so these are
costs for the declared controls rather than an accuracy-adjusted ranking.
The cohort totals 88.4min of measured loops.

Twelve short MFM/MFV windows use the same weak64 MFM snapshot near t=25,
20 warmup and 80 measured steps, three alternating repetitions, minimal
diagnostics and no video. There are no other compute PIDs at each GPU job's
launch according to the supervisor. This is a serial, launch-screened local
comparison, not a hardware saturation measurement or assurance against all
unobserved system activity.

| Method | Skin1 median window mean | Skin1.25 median window mean | Wall reduction |
|---|---:|---:|---:|
| MFM | 7.989 ms/step | 6.332 ms/step | 20.7% |
| MFV | 7.898 ms/step | 6.402 ms/step | 18.9% |

All windows complete. Short skin1/1.25 endpoint conserved fields differ by about
1.0–1.1e-6 in relative L2: the optimization is not bitwise identical. These are
local workloads initialized from MFM, not mature cached MFV turbulence or long
statistical equivalence. Keep neighborSkin1.25 opt-in; no default promotion.
The prior38ms/step figures were under GPU contention and cannot be interpreted
as a code regression/improvement against these7.9ms/step windows. Small128²
Torch peak allocations (~89–184MiB) are also not a total-device-memory or GH200
capacity measurement.

![Comparison and matched-filter spectra](../.tmp/driven-cohort-analysis-20261028-final/comparison.png)

## Separate ACSPH result in the same queue

The corrected Lobovsky replay also completed10,000steps to the native reference's
actual endpoint1.91842s, not the originally requested2s. Its existing comparison
reports2,748 capped solves versus142 native; endpoint KE52.91 versus45.26J/m
and maximum pressure10.63 versus3.97kPa. That is completion with an unresolved
convergence/pressure gap, not ACSPH qualification. Its dedicated report and
source recordings remain in the campaign; this review does not retune that solver.

## Verdict and next work

**Keep:** seven implemented hydrodynamic routes now have long GPU weak-flow
viability evidence; source/restart controls pass after the documented test-device
correction; MFM/MFV short-window neighbor reuse has a measured 19–21% benefit.
**Unresolved:** stationarity, ensemble/resolution precision, production forcing
output qualification, strong-driving rows for the newly added schemes and
long cached-method equivalence. This campaign does not qualify MHD or physical
thermal transport.

Next qualification should address the observation window and independent seeds
for a deliberately small scheme shortlist, then temporal/resolution controls.
If the aim changes to stationary small-scale statistics, evaluate domain size,
forcing scale and any physically specified large-scale sink as separately named
models, rather than silently adding drag to make this screen pass. Transient
chaotic training data is a possible distinct objective, requiring appropriate
conditional inputs, quality labels and trajectory-level splits. No new campaign
is started by this analysis.

## Reproduction and saved analysis

[Offline analyzer](../scripts/summarize_drivenCohort.py) reads all fourteen
per-run summaries instead of the overwritten top-level index (which only lists
the final Inutsuka128 invocation). It produces merged reports, independently
reconstructed accepted budgets, input/result/history hashes, resolution/PDF/
spectral comparisons, timing field differences and static PNG/PDF figures.
Original evidence and the original top-level index are left untouched.

[cohort.json](../.tmp/driven-cohort-analysis-20261028-final/cohort.json),
[individual reports](../.tmp/driven-cohort-analysis-20261028-final/individual-reports.json)
and [analysis provenance](../.tmp/driven-cohort-analysis-20261028-final/analysis-provenance.json)
retain the machine-readable results. Reproduce into a fresh output directory:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/warpsph-mpl \
/home/lu26029/miniconda3/envs/warp/bin/python -u scripts/summarize_drivenCohort.py \
  --campaign /home/lu26029/dev/warpSPH/.tmp/overnight-acsph-driven-2026-10-27 \
  --baseline .tmp/driven-qualification-20261027 \
  --output .tmp/driven-cohort-analysis-20261028-reproduction
```
