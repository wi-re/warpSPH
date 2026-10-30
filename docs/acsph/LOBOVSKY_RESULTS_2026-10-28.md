# Corrected ACSPH versus dynGPUSPH — 2026-10-28

The corrected fixture reproduces the collapse and first-impact bulk dynamics
closely. It still differs in convergence, later pressure transients and final
energy retention. This comparison uses the supplied physical particles/mass,
all 10,000 recorded native timesteps and 193 matched physical frames through
**1.918419918 s**. Native is FP64; local is FP32 with declared input rounding.
No new simulation or model change was made for this analysis.

[Detailed generated report](../../.tmp/acsph-comparison-2026-10-28/dynamics.md) ·
[Machine-readable metrics](../../.tmp/acsph-comparison-2026-10-28/dynamics.json) ·
[Primary comparison](../../.tmp/acsph-comparison-2026-10-28/report.md).

## Bulk dynamics

Time-weighted mean KE differs by **−0.15%** over 0–0.4 s, **+0.20%** over
0.4–1.0 s, and **+0.11%** over 1.0–1.35 s. Wave motion remains visibly similar
through the early returning jet. Over 1.6–1.918 s local mean KE is **6.96%**
higher. At the endpoint:

| Observable | dynGPUSPH FP64 | warpSPH FP32 |
|---|---:|---:|
| Kinetic energy [J/m] | 45.257 | 52.907 |
| Mechanical energy K + gravity PE [J/m] | 154.103 | 161.190 |
| Maximum particle speed [m/s] | 3.405 | 4.773 |
| 99th-percentile particle speed [m/s] | 1.753 | 1.921 |
| Mean pressure [Pa] | 504.56 | 452.40 |
| Maximum pressure [Pa] | 3,973.8 | 10,630.4 |

Local endpoint KE is **16.9% higher**. These results do not support a globally
excessively dissipative local scheme. They do not establish energy accuracy:
K+PE has physical/numerical loss mechanisms in both implementations, and later
chaotic flow structures separate. Endpoint material displacement RMS is 0.120 m;
velocity RMS on common 1 cm cells is 0.547 m/s. The latter excludes cells with
fewer than four particles in either run. Sparse spray differs in both directions;
maximum particle values alone do not characterize bulk motion.

![Bulk dynamics](../../.tmp/acsph-comparison-2026-10-28/bulk.png)

## Pressure comparison requires cadence and stage labels

The primary probe comparison uses the **same fluid-only normalized Wendland C2
estimator on post-shift coordinates at identical clocks**. Broad first-impact
pressure histories are close. Narrow pulses need finer observations: the local
P1 maximum over 0.4–0.7 s is **36.76 kPa** in the full accepted-step trace but
only **6.94 kPa** in the ~10 ms common frames. Native's raw P1 first-impact
peak is about 35.07 kPa, but that raw log samples an earlier RK state and has
its own timestamp convention. It is contextual evidence, not an identical
post-shift observable or a formally matched peak-error measurement.

Later pressure structure differs strongly. Over 1.35–1.6 s:

| Probe | Native common-frame peak [kPa] | Local common-frame peak [kPa] | Local every-step peak [kPa] |
|---|---:|---:|---:|
| P1, y=3 mm | 23.59 | 5.72 | 6.96 |
| P2, y=15 mm | 22.69 | 5.20 | 6.42 |
| P3, y=30 mm | 21.70 | 4.81 | 5.96 |
| P4, y=80 mm | 15.91 | 5.23 | 6.97 |

The native raw RK log also has much larger positive and negative spikes during
this later interaction; it is plotted separately. The native is not a smooth
pressure reference. These sampled extrema and impulses are not temporally
converged peak estimates, and no experimental pressure-validation claim is made.
A matched high-cadence post-shift native recording would be required to compare
narrow transients rigorously. Do not compare the maximum pressure of an arbitrary
fluid particle with a wall sensor value.

![Matched pressure curves](../../.tmp/acsph-comparison-2026-10-28/pressure.png)

## Convergence and runtime

| Metric | dynGPUSPH FP64 | warpSPH FP32 |
|---|---:|---:|
| Capped steps / 10,000 | 142 (1.42%) | 2,748 (27.48%) |
| Median iterations | 15 | 15 |
| End-to-end wall [s] | 15,923.0 | 19,002.7 |
| End-to-end wall [ms/step] | 1,592.3 | 1,900.3 |

The first post-startup cap is **step 468, t=0.163517007 s**, with local epsilon
−7.74319 against target −8, while native converges in 15 iterations. This is
before the opposite-wall impact near 0.416 s. It cannot be dismissed solely as
late chaotic separation. At 1.2–1.5 s local caps on **1,265/1,609 steps**, versus
56/1,609 native. All 10,000 local steps retain CUDA graphs. Capped steps consume
**80.46% of measured local solve wall time**; loss of graph capture does not
explain this workload. Measured solve time is 17,736.8 s, with remaining wall
cost in other work/output/setup.

The native run shared the GPU for part of its execution; local is a separate
campaign. These end-to-end timings are not a controlled precision/hardware
throughput comparison. FP64/FP32 differences remain a confounder for attributing
the convergence gap.

![Convergence](../../.tmp/acsph-comparison-2026-10-28/convergence.png)

## Verdict and next bounded control

The SDF correction repaired the comparison fixture and restored the early bulk
trajectory. Solver parity remains unresolved. The most useful next discriminator
is the solver-entry state/history at step 468: per-particle residual contribution,
hard-bounce activation, precision and native stopping semantics. The evidence
above does not yet identify which causes that cap. Physical snapshots omit full
BDF/low-part history and cannot be treated as exact restart checkpoints. No new
long replay, tolerance relaxation, bounce change or numerical default promotion
was performed here.

![Matched spatial frames](../../.tmp/acsph-comparison-2026-10-28/spatial.png)

Reproduce from the repository root with the warp conda Python:

```bash
python scripts/acsph_reference/analyze_trajectory.py \
  --reference .tmp/acsph-lobovsky-long-2026-10-27/dynGPUSPH-lobovsky.h5 \
  --local .tmp/overnight-acsph-driven-2026-10-27/acsph/warpSPH-lobovsky.h5 \
  --out .tmp/acsph-comparison-2026-10-28
python scripts/acsph_reference/analyze_lobovsky_dynamics.py \
  --summary .tmp/acsph-comparison-2026-10-28/summary.json \
  --out .tmp/acsph-comparison-2026-10-28
```

Checks: all physical clocks align to 1e−12 s; source UID/kind order matches;
all 10,000 local convergence receipts map to accepted steps. Generated bulk,
pressure, native-RK context, convergence and spatial figures were inspected.
