# Lobovsky convergence-gap investigation — 2026-10-27

The long local replay and its continuation supervisor were stopped at the user's
request. Last completed real step: **4221, t=0.842944542 s**; last retained full
snapshot: t=0.840078946 s. The native reference remains intact through 1.918419918 s.
The local recording is **invalid for solver ranking** because of the fixture
geometry defect isolated below. No automatic long replay remains active.

## Finding: incorrect SDF return contract in the comparison harness

The custom `compare_lobovsky.py` tank SDF returned an N-element value tensor.
`ParticleRegion` ghost construction expects **`(values, normals)`**. Its merged
surface code reads `sdf(points)[0]`, so a bare tensor supplied the first point's
scalar value and broadcast it to the other points. Since that first boundary
point is a corner, every flat bottom-wall ghost acquired a diagonal offset and
the ACSPH position bounce used a **(0.7071, 0.7071)** normal instead of **(0, 1)**.
The SDF file values and all physical input arrays had been checked, but this
derived geometry had not. This was a comparison-harness mistake.

The first anomalous cap is **step56, t=0.023652369 s**, before the leading front reaches the opposite tank wall:
native takes14 iterations, local takes200 and ends at epsilon_v=-6.08669.
One fluid particle (UID319600 near x=.60753, y=-.0003404 m) carries **99.9984%**
of the squared position residual. Its residual components are approximately
(0.13254, 0.13262) m/s, consistent with the spurious diagonal contact direction.
The exact hard-bounce step matches graph versus eager **bitwise** in position,
velocity and pressure, ruling out graph execution as the cause of this step.

Same-input diagnostic controls on that exact solver-entry state/history:

| Control | Iterations | Final epsilon_v |
|---|---:|---:|
| Original geometry, hard bounce | 200 | -6.086690 |
| Original geometry, bounce off | 13 | -8.048549 |
| Original geometry, existing smooth bounce (ramp=.5) | 13 | -8.048611 |

These controls isolate the bounce path; they are not fixes promoted to defaults.
The actual correction is to return the proper SDF tuple. Shared boundary code,
ACSPH numerical operators, original hard bounce, tolerance and iteration budget
are unchanged. The corrected geometry passes the flat-wall audit on all four
walls, with zero measured component error in the captured state. The old state
is rejected with a bottom-normal component error of .707109.

## Bounded corrected validation

With the correct SDF tuple, step56 takes **13 iterations, epsilon_v=-8.076293**
using the original hard bounce. Graph/eager results remain bitwise identical;
off/smooth controls produce the same solution because that spurious bounce is
no longer activated. Physical x/v/p/rho/volume/marker IC arrays are bitwise
equal to the old FP32 fixture: no particles, mass or walls were moved.

The corrected prefix was extended only to **236 steps, t=0.099947030 s**:

| Configuration | Startup caps (t<.005 s) | Caps after startup | Median iterations after startup |
|---|---:|---:|---:|
| Native FP64 reference | 17 | 0 /212 | 14 |
| Original invalid FP32 fixture | 16 | 29 /212 | 13 |
| Corrected FP32 fixture, original hard bounce | 16 | **0 /212** | **13** |

Costs include video and diagnostics. The original-geometry 56-step diagnostic
cost116.922 s /2087.9 wall ms per physical step, including its extra same-state
solves. The corrected236-step prefix cost174.640 s /740.0 wall ms per step,
including startup caps, diagnostics and same-state controls. These are different
prefix lengths/instrumentation; do not use their total averages as a speed ratio.
The corrected final ordinary step was357.8 ms. Videos decode to7 and25 frames;
the corrected endpoint frame and residual/iteration plot were inspected.

Native diagnostics sampled every tenth real step from .02 to .843 s report
`Truncated:0` at their final pseudo-iteration (417 sampled steps). Thus the
reference's 95%-residual fallback does not explain the sampled early gap.
All4221 completed original local steps captured graphs without fallback.
There were3452 local cap hits versus17 native cap hits over the same dt sequence.

## Guard, limits and reproduction

`fixture_geometry.py` now implements the tuple and a derived-flat-wall audit;
`compare_lobovsky.py` calls that audit before a run and fingerprints the geometry
source. Three CPU checks cover flat-wall normals through the actual merged-SDF
consumer, sign/finite surface normals, and rejection of diagonal derived normals.
The audit was also applied to the two saved solver-entry states above.

The cause of the first failure and early cap spikes is established. **Later
impact/void-closure behaviour has not been revalidated.** The old stopped local
trajectory must not be used to infer a fundamental ACSPH or FP32 limitation.
It remains possible that additional contact, precision or stopping differences
emerge with correct geometry; no looser tolerance or trimmed stopping default
was adopted. OPEN_PROBLEMS §36 remains open for that qualification.

Evidence: `.tmp/acsph-convergence-gap-2026-10-27/` (KEEP), including
`history-summary.json`, `summary.json`, `convergence.png`, `step56/solver-entry.pt`,
per-iteration particle residuals, alternate result tensors and videos. The
corrected prefix's `used-harness/` preserves the exact harness used; the later
preflight audit is separately checked against its captured state.

Reproduce a bounded diagnostic into a fresh output directory:

```bash
python scripts/acsph_reference/probe_convergence.py \
  --campaign .tmp/acsph-lobovsky-long-2026-10-27 \
  --out .tmp/NEW-CORRECTED-PREFIX \
  --harness scripts/acsph_reference/compare_lobovsky.py \
  --step 56 --prefix-steps 236
```

Omit `--harness` to use the retained original faulty harness for the56-step
failure control. Numerical runtime comes from the frozen campaign in both cases.
`analyze_convergence_gap.py` reproduces the summaries and scientific figure.
The stopped run's lossless viewer export is
`.tmp/acsph-lobovsky-long-2026-10-27/warpSPH-lobovsky-partial-stopped.h5`;
its header explicitly marks the fixture invalid. Raw snapshots/logs are unchanged.
