# Step 468: hard-bounce limit cycle — 2026-10-28

The corrected Lobovsky replay's first post-startup cap is caused by a
**two-cycle across the hard position-bounce threshold**. The same input fails
in FP32 and FP64; graph/eager agree bitwise. Existing smooth bounce removes
this particular cycle and meets the position stopping criterion in 13
iterations. This is a fixed-step diagnostic result, not a qualified full-run
remedy or a production default change.

[Mechanism figure](../../.tmp/acsph-step468-2026-10-28/mechanism.png) ·
[Summary](../../.tmp/acsph-step468-2026-10-28/summary.json) ·
[Evidence root](../../.tmp/acsph-step468-2026-10-28/).

## Exact reproduction and input integrity

Replayed the original supplied physical ICs through step **468**,
**t=0.16351700694626495 s**, using the corrected overnight runtime and its
original native timestep sequence. The physical prefix accepts only the
original hard-bounce solution. All **469 physical diagnostic rows** (including
startup) and all **468 iteration counts/metrics** exactly reproduce the
completed overnight trajectory. Derived flat-wall normals remain correct.

Captured state includes the original CSR/Verlet neighbor ordering, surface and
contact topology, bounce pairs/normals, body forces, BDF coefficients and all
position/velocity history, previous displacement and retained low parts.
The fixed-solve bundle omits only construction-time region/rigid-body metadata
lists, whose initializer closures are not used by this fixed solve. Loading
that bundle into a fresh FP32 process reproduces **all five output tensors
bitwise**: positions, velocities, pressures, displacement and position low part.
It is a fixed-solve replay bundle, not a complete evolving-driver checkpoint.

The FP64 control promotes the exact captured FP32 tensor values and dtype
metadata in a fresh FP64 runtime, preserving integer topology/order and all
inputs. It improves arithmetic precision, not the accuracy of the starting
state. It is not a separately evolved FP64 trajectory or the native reference's
own step-468 state.

## Same-input controls

Target epsilon is −8; maximum pseudo-iterations is 200.

| Arithmetic / control | Iterations | Final epsilon |
|---|---:|---:|
| FP32 original graph, hard bounce | 200 | −7.74318838 |
| FP32 eager, hard bounce | 200 | −7.74318838 |
| FP32 bounce off | 13 | −8.13863277 |
| FP32 smooth ramp 0.5 | 13 | −8.13863754 |
| Promoted FP64 graph, hard bounce | 200 | −7.74320968 |
| Promoted FP64 eager, hard bounce | 200 | −7.74320968 |
| Promoted FP64 bounce off | 13 | −8.13919683 |
| Promoted FP64 smooth ramp 0.5 | 13 | −8.13920154 |

Graph/eager outputs are bitwise identical within **both** precisions. The
FP64 hard result differs from FP32 by at most 3.05e−8 m in coordinates,
6.23e−6 m/s in velocity and 0.0866 Pa in pressure. Those small arithmetic
changes do not remove the cycle. Native's own evolved trajectory takes 15
iterations at the corresponding real step; it does not share this exact entry
state, so this test alone does not explain why it avoids the cycle there.

## Mechanism and neighborhood

At the last hard-bounce iteration, UIDs **286801 and 278800** carry
**99.95065%** of the squared position-row residual. They lie along the flat
bottom, far from a corner, with bounce normals exactly **(0, 1)**. This is
not a recurrence of the earlier diagonal-SDF-normal fixture error.

The leading particle's distance from its nearest wall row alternates between
about **0.249718 dx and 0.250869 dx** in the final iterations, straddling the
hard threshold **0.25 dx**. Its physical normal velocity remains approximately
−0.00975 m/s while its normal position-row residual alternates between about
**+0.01238 and −0.01238 m/s**. Each hard switch changes the velocity seen by
the position equation, moves the next iterate across the threshold, and
switches back. Increasing arithmetic precision preserves this behavior.

| UID | Active fluid neighbors, including self | Active wall neighbors | Fluid kernel-volume sum | Fluid + wall kernel-volume sum |
|---|---:|---:|---:|---:|
| 286801 | 18 | 30 | 0.2446 | 1.1296 |
| 278800 | 16 | 30 | 0.2420 | 1.1272 |

These counts filter the retained candidate list at current support and exclude
ghost rows. Kernel sums use the actual Wendland C2 support and particle volumes.
The fluid stencil is truncated by the wall, but these are populated wall/fluid
neighborhoods, not self-only particles. No neighborhood cutoff was changed.

![Threshold cycle](../../.tmp/acsph-step468-2026-10-28/mechanism.png)

## Equation residuals and field changes

The stopping norm tests the position row. It is not a claim that every
momentum/pressure residual is zero. Separate final-equation checks in promoted
FP64 give:

| Control | Position RMS [m/s] | Momentum RMS [m/s²] | Pressure RHS RMS [Pa/s] |
|---|---:|---:|---:|
| Hard, 200 iterations | 2.480e−5 | 0.003063 | 3502.76 |
| Off, 13 iterations | 9.963e−6 | 0.038305 | 3913.01 |
| Smooth 0.5, 13 iterations | 9.963e−6 | 0.038306 | 3913.00 |

These dimensionful final-iterate residuals are not the native code's
last-RK-stage pressure stopping observable. The 13-iteration controls meet the
unchanged local stopping criterion while retaining larger momentum RHS than
the 200-iteration hard control. Long-run accuracy and wall behavior still
need measurement.

In FP32 the smooth control differs from the hard result by at most 6.89e−6 m
(0.0092 dx) in position and 0.00283 m/s in velocity. Fluid KE is nearly unchanged
(43.20065 versus 43.20076 J/m), but fluid pressure RMS differs by **66.10 Pa**,
with maximum difference **209.95 Pa** and mean pressure 1206.86 versus
1251.55 Pa. The extra hard-bounce iterations are not merely wasted identical
updates: pressure continues evolving while a few contact rows hold up the
position stopping norm. This does not establish the accumulated effect in
later physical steps.

A 95% trimmed metric would read −9.43994 here, but it discards the offending
rows. It was not adopted. Applying the native five-sample stall trigger to
this observed local epsilon sequence never activates it: the final oscillation
range 0.04498 exceeds its 0.00776 threshold. This is an offline criterion check,
not a native same-input solve or a claim that native's stopping path is identical.

## Costs, recordings and scope

- Physical 468-step prefix: **270.54 s**, **578.07 ms per physical step**,
  including same-state diagnostics, capture, video and startup caps. This is
  not a clean throughput benchmark.
- Instrumented FP32 eager controls: hard **5.48 s**, off **0.325 s**, smooth
  **0.351 s** for the target solve. All use identical input; observations add
  overhead.
- Fresh-process FP32 graph reload: **8.70 s** for the solve (13.15 s process).
- Fresh FP64 controls: graph **56.54 s**, eager **29.00 s**, off **1.84 s**,
  smooth **1.88 s**; 94.20 s process. The first control includes compilation,
  graph setup/validation and precision-specific initialization. Do not infer
  graph/eager or FP32/FP64 steady throughput ratios from these costs.

The physical prefix has a decoded **48-frame VisPy recording**; its actual last
frame at t≈0.1635 s and the mechanism figure were inspected. Tight same-state
controls explicitly omit video and stream iteration progress. GPU work ran
serially with a bounded prefix wall limit. No full replay was started.

## Verdict and next validation

**Established at this exact step:** hard-bounce threshold switching causes the
position-residual cycle; it persists in FP64, with graph/eager parity and
correct flat-wall geometry. Existing ramp 0.5 is a viable candidate for the
next bounded trajectory comparison. Bounce off remains a causal control.

**Still open:** why the native evolved state avoids this cycle at step 468,
how many later local caps share this cause, and whether smoothing preserves
wall penetration, pressure transients and energy through impacts. The existing
Marrone case already uses a smooth ramp; the Lobovsky reference-alignment
harness explicitly selects hard bounce. Changing that selection would be a
named experimental variant, not a claim of unchanged reference-code behavior.
No production numerical, shared boundary or tolerance defaults were changed.
