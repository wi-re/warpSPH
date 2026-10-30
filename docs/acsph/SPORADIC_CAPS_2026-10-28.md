# Additional sporadic ACSPH caps — 2026-10-28

Three additional isolated caps depend on the same **hard position-bounce
closure** as step 468. The end-of-iteration behavior varies: a near-threshold
stall, a five-iteration cycle and a three-iteration cycle. In every sampled
case, bounce off or the existing smooth ramp 0.5 meets the unchanged position
criterion in 13 iterations instead of hitting 200.

[Residual/contact plots](../../.tmp/acsph-sporadic-caps-2026-10-28/sampled-caps.png) ·
[Summary](../../.tmp/acsph-sporadic-caps-2026-10-28/summary.json) ·
[All-run cap catalogue](../../.tmp/acsph-sporadic-caps-2026-10-28/cap-catalogue.json).

## Same-input sample

| Real step | Time [s] | Native iterations | Local hard | Local off | Local ramp 0.5 | Leading UID | Leading row's squared residual share |
|---|---:|---:|---:|---:|---:|---:|---:|
| 475 | 0.165239 | 15 | 200 | 13 | 13 | 278800 | 99.944% |
| 510 | 0.173726 | 15 | 200 | 13 | 13 | 278800 | 99.969% |
| 606 | 0.196109 | 15 | 200 | 13 | 13 | 276800 | 99.984% |

Each original hard-bounce solve and its eager replay agree **bitwise** in
position, velocity and pressure. Alternatives use identical solver-entry
state/history and are diagnostic only: the original hard result feeds the
next physical step. All **607 physical diagnostic rows** through step 606
exactly match the overnight trajectory, including iteration counts and epsilon.
Thus observations at earlier sampled steps do not change later inputs.
The new samples are FP32; the prior exact-input FP64/graph controls apply to
step 468 and are not silently extrapolated as new FP64 measurements here.

For all four sampled events (468, 475, 510 and 606), the physical-step iteration
sequence immediately around the event is **13 → 200 → 13**. This can arise as
particles encounter a discontinuous contact condition briefly while the bulk
continues moving normally. It is not evidence that extra iterations represent
more difficult bulk dynamics.

## Related behavior, with different cycles

- **Step 468**, already established: two-cycle across 0.25 dx; two rows carry
  99.95065% of the squared residual; persists in promoted FP64.
- **Step 475:** the leading row ends each pseudo-iteration at approximately
  0.248954 dx, just inside the threshold, with a nonzero residual plateau.
  Off/smooth counterfactuals isolate dependence on bounce. Completed-iteration
  samples do not prove which RK substages switch the contact condition; that
  detail remains open. Do not label this an observed two-cycle.
- **Step 510:** completed-iteration distances repeat with period five and span
  0.248613–0.250792 dx, crossing 0.25 dx.
- **Step 606:** completed-iteration distances repeat with period three and span
  0.248448–0.250523 dx, also crossing 0.25 dx.

Periods above use the last 30 observed iterations and a 1e−6 dx comparison
tolerance. They classify the saved trace, not a mathematical proof of a
permanent periodic orbit.

![Additional cap controls](../../.tmp/acsph-sporadic-caps-2026-10-28/sampled-caps.png)

## Relation to the older ACSPH findings

The Marrone ceiling-rider study already isolated D5 threshold switching:
hard bounce took 200 iterations, off 12–16 and smooth bounce 13–16. The
[reference comparison report](DYNGPUSPH_COMPARISON.md), section
“Launch overhead and the ceiling riders”, and OPEN_PROBLEMS §33 retain that
study and its wall-penetration tradeoffs.

Current dam-break and sloshing case presets already select ramp **0.5**.
The custom supplied-Lobovsky comparison explicitly selects ramp **0** for
native hard-bounce alignment. These new observations therefore recover a known
class of contact nonconvergence on different states and with stricter tolerance;
there is no new global default change in this probe.

## How far this conclusion extends

The completed local run contains **2,748 capped steps in 607 contiguous groups**:
368 single-step groups, 209 groups of 2–5 steps and 30 longer groups. Between
0.1 and 0.4 s there are 14 caps, all isolated. Four of those now have same-input
bounce evidence, including the earlier step 468.

Do not yet assign every later cap to this cause. The long block at
1.379789–1.544510 s alone spans 1,144 capped steps. Impact/free-surface changes,
neighborhoods and other residual modes must be sampled there separately.
The evidence supports a bounce-related family among isolated early spikes;
it does not classify the full late trajectory or qualify a global remedy.

## Cost and validation

One unchanged physical prefix reached step 606/t=0.196108807 s in **346.57 s**,
**571.90 ms per physical step**, including startup, capture, observation,
three sets of same-state controls and video. This includes diagnostics and
is not a clean throughput comparison. The preceding 468-step investigation
was 578.07 ms per physical step with different instrumentation.

At each target, instrumented eager hard bounce costs **5.44–5.45 s**,
off **0.320–0.323 s**, and smooth **0.353–0.356 s** for that fixed solve.
Recordings remain on for the physical prefix; tiny same-input controls omit
video explicitly and stream iteration progress. All 62 encoded VisPy frames
decode, and the final physical frame and contact plots were inspected.
Numerical runtime sources remain pinned. Production/shared boundary code,
tolerances and physical defaults are unchanged. No long-impact replay started.
