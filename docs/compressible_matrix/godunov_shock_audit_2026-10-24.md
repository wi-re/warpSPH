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
- The switch passes the first-failure windows in all three Sedov dimensions:
  t=0.0003 at nx=800 / 256 / 64. These short 2D/3D controls are not full-case passes.

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
absolute post-shock density error. Matched acoustic-wave controls remain
essentially unchanged. Close-pair RHS conservation and a targeted automatic-
gradient check also pass. These checks support the density/derivative consistency
diagnosis; they do not qualify a new default across the wider solver suite.
The original paper derives its interpolation first at constant h and discusses
variable smoothing separately ([sections 3.1 and 3.3](https://arxiv.org/html/astro-ph/0206401)).

The prototype and quantitative evidence live in
`.tmp/inutsuka-noh-20261026/`, with physical choices and limitations indexed
there. Production operators, defaults and saved matrix selections remain
unchanged. A production implementation should pass the query-width derivative
as a dedicated scalar field, preserve the constant-width limit and pair
symmetry, handle coincident pairs and cutoff transitions, and qualify the
wider shock/contact/vortex suite before promoting this policy.

## Priorities

1. **GSPH:** extend the successful switch control to full Sedov 2D/3D
   and rerun Sod/contact, Gresho and KH controls before changing the
   default. Do not clip negative u or relax the matrix admissibility criterion.
2. **Inutsuka:** resolve the variable-width density/derivative inconsistency
   before adding a generic close-pair limiter. Preserve pair symmetry/conservation
   and qualify smooth-flow accuracy and the wider shock suite. Reconcile the
   adaptive timestep with any remaining force stiffness.
   Merely selecting RK4 or relabelling the alarm does not fix it.
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
