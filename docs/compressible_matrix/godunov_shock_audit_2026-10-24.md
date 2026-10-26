# Godunov cold-shock failure investigation (2026-10-24)

The matrix failures are reproducible numerical failures, with two different
mechanisms. They should not be explained away by FP32, an endpoint tolerance,
or the generic integrator tests passing. No solver defaults or saved matrix
entries were changed during this investigation.

## What actually failed

| Scheme / case | Saved matrix outcome | Worst accepted specific internal energy |
| --- | --- | ---: |
| GSPH, open Noh | Reaches t=0.6; 26 diagnostic rows nonpositive | -3.0642e-8 |
| GSPH, Sedov 1D | Reaches endpoint; 3,175 rows nonpositive | -1.3352e-6 |
| GSPH, Sedov 2D | Reaches endpoint; 1,185 rows nonpositive | -7.6916e-6 |
| GSPH, Sedov 3D | Reaches endpoint; 952 rows nonpositive | -1.8112e-6 |
| Inutsuka, open Noh | Stops at t=0.01349397, step 33 | -6.9204 |
| Inutsuka, Sedov 1D / 2D / 3D | All reach endpoints with positive accepted histories | Initial ambient minimum 1e-8 |

The initial ambient energy is 1e-8. Even a small absolute negative number is
therefore a real loss of cold-gas positivity. Inutsuka's Noh failure is much more
severe: final maximum speed 8,286 for unit initial inflow, physical energy about
1.41 million times its initial value. The velocity alarm stops an already
catastrophic trajectory; it is not the cause.

## Simplified GSPH: reconstruction loses thermal positivity

At the saved resolution, initial conditions and shared integrator:

- The Noh onset is reproduced at exactly t=0.007744302432552383, with
  u_min=-3.1299235321496973e-9 at particles 122 and 133, near x=+-0.03522.
- Sedov 1D first goes negative near t=5.1065e-5. At t=0.0003 its minimum is
  -1.32868e-6 in FP32 and -1.32861e-6 in FP64.
- Quarter CFL gives -1.35844e-6 at the same Sedov time. This is not a time-error
  or rounding error that vanishes with those controls.
- First-order Riemann states remove the negative-energy onset in Noh and Sedov
  1D. The existing compressive-pair first-order shock switch, `shockSwitchC=3`,
  also removes it, while retaining second-order reconstruction outside its
  triggered pairs.
- With that switch, full open Noh reaches t=0.6 in 1,760 steps, without negative
  accepted or RHS-stage energies. Post-shock density error is -0.01915%; total
  energy change is -0.0002991%.
- Full Sedov 1D also reaches t=0.5556113729836043 in 20,996 steps, without
  negative accepted or RHS-stage energies. Its minimum accepted u remains the
  1e-8 ambient value and total energy change is -0.02321%.
- The later full-row campaign reaches every supported endpoint (13 cases),
  including native/open Noh and full Sedov 2D/3D at nx=256/64, with finite,
  positive accepted and RHS-stage states. Sedov 2D/3D physical energy changes
  are -0.03464% / -0.03590%. Fresh matched C=0 controls still lose positivity
  in both Noh columns and all three Sedov dimensions.
- The switch has a measured accuracy cost: volume-weighted density L1 increases
  5.831% in Sod and 1.669% in 2D Sod. The wave, Kidder, Gresho, common-gamma
  pressure-equilibrium and both KH results are unchanged in these controls.
  This does not establish that the switch is inactive in every other smooth
  flow or that first-order triggering has no diffusion cost.

This isolates the high-order reconstruction as the positivity problem under
these cold-shock inputs. Exact pair/global conservation tests do not guarantee
local internal-energy positivity. The current default GSPH shock switch is off;
Inutsuka already has it enabled, so it does not address that scheme's separate
Noh instability.

## Inutsuka Noh: close-pair cubic volume amplification

The exact saved blow-up is reproduced in FP64, using the same stage calls and
inputs. The critical pair is particles 127 and 129 near the central shock.
At a midpoint stage, t=0.012712900474595825:

| Quantity | Value |
| --- | ---: |
| Pair separation r | 1.74747824e-5 |
| Gaussian width h | 0.00353648621 |
| r/h | 0.00494128 |
| Endpoint specific volumes | 0.41557924, 0.41568547 |
| Endpoint axial derivatives | -4.48307, -4.66652 |
| Cubic volume moment V_ij^2 | 44,521.42 |
| Linear volume moment | 0.172866 |
| Cubic / linear moment | 257,549 |

Both derivatives have the same sign, so the implementation's existing
opposite-sign fallback does not activate. The cubic is extrapolated over a
Gaussian far wider than the distance between its interpolation endpoints. Its
value at half a Gaussian width is about 54.9, despite endpoint values near 0.416.
The resulting acceleration reaches about 77,122. The following accepted state
has negative energy and speed about 68.4; later the speed jumps to 8,286.

Independent numerical quadrature reproduces the analytic cubic moment to about
9e-15 relative error. This is not an erroneous coefficient in the closed-form
moment calculation. It is an unstable close-pair extrapolation and unresolved
force variation in this implementation / timestep combination.

The interpolation and moment formulas were checked against
[Inutsuka (2002), equations 60-65](https://arxiv.org/abs/astro-ph/0206401).
Agreement with those formulas does not establish robustness of the surrounding
variable-width, density, reconstruction and time-discretization choices.

Controls that do **not** repair the Noh onset:

- Periodic outer boundaries reproduce the same central instability.
- First-order Riemann states still blow up near t=0.0141.
- Clamping the interface position gives the same baseline trajectory.
- RK4 also blows up, near t=0.0144. Increasing formal integrator order does not
  resolve this close-pair stiffness at the tested timestep policy.
- Capping by the particle-spacing acoustic CFL still blows up. A simple nearest
  approaching-pair crossing-time cap instead stalls near t=0.00446; it is not a
  usable remedy.

## Full open-Noh controls, nx=256, FP64, t=0.6

All rows below retain positive accepted **and stage** energies. These are labeled
diagnostic trials, not replacements for the original matrix rows or measured
performance comparisons. Several runs overlapped and the probe records detailed
stage data, so their wall times must not be ranked.

| Inutsuka control | Steps | Post-shock density error | Total energy change |
| --- | ---: | ---: | ---: |
| Original cubic, quarter CFL (0.0375) | 6,973 | +0.01162% | +0.01879% |
| Gaussian width eta=0.75, original CFL | 1,736 | +0.05372% | +0.004475% |
| Linear volume, original CFL | 1,754 | -0.009827% | +0.001385% |
| Hold initial dt=0.000121 fixed | 4,959 | +0.01282% | +0.004482% |

There is a concrete reason earlier validation appeared stronger: the historical
Godunov campaign held its setup timestep fixed (the retained Noh stability log
shows dt=0.00011), whereas the matrix re-evaluates and grows the timestep. The
current adaptive trajectory reaches dt=0.000883 before the first catastrophic
midpoint. Holding the current matrix's initial timestep fixed succeeds even at
the current resolution, precision and open boundary protocol. This is a change
in the tested operating regime, not proof that a previous source-code regression
caused the failure.

Linear-volume and narrower-Gaussian successes are not a blanket recommendation
to change defaults: the historical stability campaign had other linear-volume
failures, and Gaussian width changes spatial accuracy and neighbour coupling.
The quarter-CFL control is a demonstrated immediate workaround; proper close-pair
robustness still needs a reconstruction/timestep treatment and convergence tests.

## Variable-width consistency of the cubic data

The large moment also has an identifiable upstream cause in the current
[Gaussian density and pair operator](../../src/warpSPH/modules/godunov/wp_inutsuka.py).
Its symmetrized density is

\[
 \rho(x,h_q)=\tfrac12\sum_k m_k\left[
 W(x-x_k,\sqrt2 h_q)+W(x-x_k,\sqrt2 h_k)\right].
\]

The endpoint densities use different query widths, \(h_q=h_i\) and \(h_q=h_j\).
The supplied gradients differentiate position while holding that query width
fixed. Those values and slopes therefore do not describe a single fixed-width
field. Hermite interpolation amplifies the mismatch as the pair approaches.
This is distinct from an error in the polynomial's analytic Gaussian moment.

At the critical pair above, the specific-volume secant has magnitude 6.0794,
while the mean supplied slope has magnitude 4.5748. Recomputing both endpoint
values and slopes with the same query width reduces the volume moment from
44,521 to 0.1868. Direct quadrature of that inverse-density field gives 0.1850;
the remaining difference is the cubic approximation error.

An alternative consistent construction lets query width vary linearly along
the pair. Its axial density derivative includes the chain rule

\[
 \frac{d\rho}{ds}=\left.\frac{\partial\rho}{\partial s}\right|_{h_q}
       +\frac{\partial\rho}{\partial h_q}\frac{h_i-h_j}{r_{ij}},
 \qquad
 \frac{\partial\rho}{\partial h_q}
 =\tfrac12\sum_k m_k W(x-x_k,\sqrt2 h_q)
 \frac{|x-x_k|^2/h_q^2-d}{h_q}.
\]

Applying this derivative to the specific-volume slopes reduces the critical
moment to 0.1881 and preserves the symmetric pair coefficient. The correction
vanishes for equal query widths. It changes the interpolation policy; it is
not the standard Monaghan force's grad-h factor.

An isolated implementation passes full open Noh at nx=128, 256 and 512 with the
original CFL, positive accepted and RHS-stage thermal energies, and decreasing
absolute post-shock density error. Matched FP64 acoustic-wave controls remain
essentially unchanged. Close-pair RHS conservation and a targeted automatic-
gradient check also pass. These checks support the density/derivative consistency
diagnosis; they do not qualify a new default across the wider solver suite.
The wider fixed-preset campaign rejects this unregularized correction as a
general default. All twelve candidate cases reach finite, positive endpoints,
including all RHS stages, but the common-gamma pressure-equilibrium contact
suffers a numerical instability, not a bounded accuracy regression: final
maximum pressure error 8.87 versus 0.0370 for the original scheme, and physical energy growth 25.9%
versus 0.025%. A sharp impulse near t=4.707 generates the defect; peak pressure
error is about 966. The separately integrated energy buffer changes by only
about -0.0023%, so it must not substitute for the physical energy computed from
mass, internal energy and velocity.

An instrumented replay matches every recorded physical diagnostic through the
failure (only the final cap at the shorter replay endpoint differs). The
trigger is a near-coincident pair, particles 6725 and 6925, with separation
9.21e-6 and Gaussian width about 0.011824. At t=4.706483 the maximum RHS
acceleration is 125,846; the following RK midpoint reaches speed 57.58 while
remaining finite and thermally positive. The native sound-speed timestep,
about 9.44e-4, does not resolve this force impulse. It shrinks only after the
accepted pressure/energy jump. The configured velocity stop ceiling is 100
on the case's unit velocity scale; the accepted peak speed 19.2 does not
trigger it. Neither this ceiling nor positivity tests bound contact pressure
error or physical energy production.

The failing pair also exposes FP32 amplification in periodic distance
calculation. Its forward x-separation is 1.3113022e-6 but its reverse is
-1.2814999e-6. Subtracting directly within the primary domain gives equal and
opposite values. The cubic volume moment amplifies the wrapping roundoff:
native forward/reverse moments are about 257,204 and 28,287, compared with a
linear moment about 0.213. A standalone reproduction of all contributions to
each affected particle matches the captured acceleration/heating to FP32
roundoff. A distance-only arithmetic control restores equal-and-opposite
pair forces, but leaves the large cubic moment; it is not a qualified remedy.
Evaluating the saved FP32 fields with FP64 arithmetic leaves a moment about
23,456. This is not a full-precision RHS test: recomputing all density and
derivative sums in FP64 at the same particle positions reduces the moment to
0.21397, close to direct inverse-density quadrature (0.21353). FP32 summation
errors are therefore another amplified input to the extrapolation. The onset
and exact pair evidence are indexed in
`.tmp/inutsuka-hydrostatic-failure-20261026/`; the recomputed-field controls are
in `.tmp/inutsuka-remedies-20261026/`.

The pressure-equilibrium candidate must therefore be labelled **numerical
failure** even though its execution record says it completed. Completion,
finite/positive states and accuracy qualification are separate properties;
the failed contact is excluded from ordinary relative-error comparisons.

Sod, Kidder and Gresho remain close to the controls. Sedov density L1 improves
in 1D and 3D but increases by 3.6% in 2D; the FP32 linear-wave error increases
by 3.1%. KH mode amplitudes remain unranked diagnostics. These outcomes use
identical physical initial arrays and one archived source set, with full matrix
resolutions/endpoints and no failure-specific tuning. Evidence and the explicit
rejected qualification are indexed in `.tmp/inutsuka-validation-20261026/`.

### Close-pair extrapolation and the Gaussian cutoff

The failure has a more general source than query-width consistency. With
pair separation s, the cubic coefficient is A = -2 delta/s^3, where
`delta = Vi - Vj - (di + dj)s/2`. The leading squared Gaussian moment is
`(15/16)(h/s)^6 delta^2`. Extending a cubic fitted over a very short interval
across the whole Gaussian amplifies small endpoint/derivative mismatches.

A full FP64 contact control also fails, stopping at t=1.311993 with speed
about 640. Its captured trigger is a single neighbor crossing the hard Gaussian
cutoff: particle 4680 is at 3.001342 Gaussian widths from endpoint 4574 but
2.999960 from endpoint 4474. The resulting density contribution, about 7.02e-6,
is present at only one endpoint and absent from the local derivatives. At pair
separation 1.66346e-5, this creates a specific-volume compatibility residual
1.20e-6 and a moment 174,668. Recomputing with an uncut Gaussian or one shared
neighbor mask reduces the residual to about 9e-12 and the moment to
0.1784–0.1785. These controls isolate the cutoff discontinuity; they do not
silently replace the production density kernel. Precision alone is insufficient.

An isolated alternative fits the cubic only **between** the endpoints and
continues with their tangents outside. Its Gaussian moments use interior
quadrature for s/h<=2 and analytic polynomial tails for larger separations.
Independent adaptive quadrature checks cover s/h from 1e-6 to 20 and agree
to about 5e-15 in the selected fields. The first fixed-node prototypes fail
this check at large normalized separations and remain rejected evidence.

This alternative initially exposes an exact-coincidence division by zero.
A captured stage has particles 2574/2673 at identical FP32 positions and
density 2.58459, with positive thermal energy but nonfinite force updates.
For the finite-moment piecewise profile the Gaussian pair force and work tend
to zero at coincidence; the experimental operator handles that case before
any division, without a finite separation floor or particle displacement.

The simpler candidate uses native partial density gradients, tangent
continuation, antisymmetric nearest-image geometry for wrapped positions, and
that exact-zero pair limit. It does not require the query-width derivative.
At unchanged B7, n_h=4 and native CFL=0.15, five matched full cases reach their
exact endpoints with finite, positive accepted and RHS-stage states:

| Case | Candidate result relative to original |
|---|---|
| Pressure equilibrium, nx=100, t=7, FP32 | Pressure error 0.04557 versus 0.03703; physical energy drift -0.000294% versus +0.02504%; no impulse |
| Open Noh, nx=256, t=.6, FP64 | Completes where original stops; plateau-density error 0.005884%; physical energy drift +0.001892% |
| Sod, nx=800 | Volume-weighted density L1 decreases 0.643% |
| Gresho, nx=256, t=3 | Tangential-velocity L1 increases 0.0556% |
| FP32 analytic-mass linear wave, nx=128, t=1 | Normalized velocity L1 decreases 0.861% |

Physical initial positions, masses, velocities and thermal energies match
exactly in each pair. Noh plateau-density absolute relative errors decrease
from 0.165815% at nx=128 to 0.005884% at 256 and 0.000837% at 512. All three
reach t=.6 with positive accepted/stage energies and physical energy drift
below 0.00222%. These plateau diagnostics are not a proof of global convergence.
FP64 automatic-gradient checks pass; close/coincident-pair tests with three
Riemann solvers give relative conservation residuals below 1.4e-8 in FP32 and
1.4e-14 in FP64 (coordinate torque only in the open domain).

The candidate removes the observed instability but retains a roughly 23%
pressure-error increase relative to the original stable contact run. The later
full-row campaign below extends this limited qualification and exposes a
periodic-geometry bug. First-order policies, extreme width contrasts,
opposite-sign-slope linear fallback and coincident-point automatic derivatives
still need qualification before promotion. The hard cutoff remains part of
the density definition. These initial controls and exact runtime archives are
indexed in `.tmp/inutsuka-remedies-20261026/`.

The original paper derives its interpolation first at constant h and discusses
variable smoothing separately ([sections 3.1 and 3.3](https://arxiv.org/html/astro-ph/0206401)).

The prototype and quantitative evidence live in
`.tmp/inutsuka-noh-20261026/`, with physical choices and limitations indexed
there. Production operators, defaults and saved matrix selections remain
unchanged. If the width-derivative policy is pursued separately, its derivative
must use a dedicated scalar field rather than the experimental extra gradient
row. The tangent-continuation candidate avoids that additional derivative;
its remaining qualification requirements are described above.

## Full-row controls and a periodic-geometry correction (2026-10-26)

The row campaign preserves B7, n_h=4, native adaptive CFL=0.15, original
resolutions/endpoints and physical initial arrays. Noh and Kidder use FP64;
other cases use FP32. All accepted states and both RK2 RHS stages are checked.
Fresh C=0 GSPH controls isolate the effect of C=3. Other original Inutsuka
controls use retained runs with matching runtime-library hashes, with a fresh
native-Noh failure control. Unequal-gamma square and advected-ellipse columns
remain explicitly unsupported for both Godunov schemes; common-gamma pressure
equilibrium is the two-fluid density-1/4, P=1 contact without gravity/free surface.

The first tangent-continuation row reaches all 13 executable endpoints with
positive states, but **fails the KH physical-quality comparison**. Sharp and
smooth KH minimum pressures fall to about 0.05 and peak speeds exceed 4, versus
original minimum pressures above 2 and speeds below 0.81. Small global energy
drift does not excuse these local defects. The failed trial is retained.

The cause is an experimental geometry helper that corrects at most one periodic
image while native integrators retain unwrapped coordinates. A captured real
neighbor has raw x difference -1.999475 in a unit box. Subtracting one image
leaves -0.999475, falsely excluding a pair whose nearest-image separation is
0.001463 and whose interaction reach is about 0.017. The corrected helper uses
`delta -= sign(delta) * L * floor(abs(delta)/L + 0.5)` on periodic axes. This
handles arbitrary image counts, preserves small differences and is odd under
pair reversal. FP32/FP64 tests check 256 pairs plus reversals over up to 24
image differences, close pairs and periodic coincidence; revised FP64 gradient
checks also pass.

A separate corrected row reruns every column rather than inferring a pass from
KH alone. Both corrected KH cases restore bounded pressure/density fields:
minimum accepted pressure is about 2.08 / 2.21 and peak speed about 0.85.
Final mode amplitudes are 0.123332 / 0.187153 versus original 0.120068 /
0.185636. These amplitudes remain unranked diagnostics, not accuracy claims.
The corrected full row reaches all 13 supported endpoints with finite,
positive accepted and RHS-stage states, including native/open Noh, Kidder,
Sedov 1D/2D/3D, Gresho and pressure equilibrium. Its closed-system energy
excursion stays below 0.028%; Kidder is driven and is not ranked by closed
energy conservation. No floors or reduced-CFL retries are used.

| Corrected Inutsuka case | Primary error change versus admissible original |
|---|---:|
| Linear wave | -0.861% |
| Sod / 2D Sod strip | -0.643% / -0.0652% |
| Kidder | +0.244% |
| Sedov 1D / 2D / 3D | -2.071% / -0.204% / +0.208% |
| Gresho | +0.0556% |
| Pressure equilibrium | +23.083% |

Negative changes mean lower error in that case's native primary metric. Failed
original Noh trajectories receive no endpoint accuracy delta; corrected native
and open Noh both have plateau-density error 0.005884%. Pressure-equilibrium
error is 0.0455728 versus 0.0370259, with energy drift -0.000294% versus
+0.02504%: removing the catastrophic impulse still leaves a contact-accuracy
tradeoff. The hydrostatic result is not an unconditional accuracy pass.

All 39 matched comparisons (C=3, first Inutsuka trial and corrected trial)
have identical physical initial arrays and matching resolved solver parameters
apart from the named C=3 switch. Each frozen phase has stable runtime hashes;
production library source files match across phases. Corrected-operator
close/coincident-pair checks for three Riemann solvers, including many-image
periodic coordinates with fixed valid adjacency/reconstruction, have maximum
relative conservation residuals below 1.4e-8 in FP32 and 1.4e-14 in FP64.
The corrected fields,
failed trial, full histories, `verification.json`, `RESULTS.md` and `rows.pdf`
are retained separately in `.tmp/godunov-rows-20261026/`. Production operators
and defaults remain unchanged.

## Priorities

1. **GSPH:** assess the measured Sod diffusion cost before changing the default.
   The full-row C=3 control now covers Sedov 2D/3D and the smooth/contact/KH
   cases. Do not clip negative u or relax the matrix admissibility criterion.
2. **Inutsuka:** address the linear fallback, extreme-width and gradient limits
   before promotion. Keep tangent continuation with native gradients, correct
   arbitrary-image periodic geometry and the exact-zero pair limit together.
   The chain-rule-only candidate fails pressure equilibrium despite passing
   Noh. Preserve pair symmetry/conservation and assess contact accuracy alongside
   physical energy. Merely selecting RK4 or relabelling the alarm does not fix it.
3. Keep the robust-control rows and original defaults separately in future
   comparisons. Measure performance only in isolated production runs after the
   numerical policies are selected.

## Evidence and reproduction

Evidence is retained under `.tmp/godunov-shock-audit-20261024/`:

- `probe.py`: saved-input replay through the shared integrator, bounded by time
  and step count; stage/accepted-state capture and explicitly named controls.
- `controls-summary.json`, `followups-summary.json`, `full-noh-metrics.json`,
  `control-definitions.json` and `close-pair-analysis.json`.
- Per-trial `result.json`, initial/final arrays, first invalid arrays and Noh
  onset stage snapshots. The original matrix attempt is identified in each record.
- `source-snapshot/`: 1,217 source files, Git commits, dirty patches and hashes.
- `failure-controls.png`: onset traces demonstrating the different mechanisms.

The scientific controls change only their named options. Native matrix defaults
and the saved matrix PDF remain unchanged.
