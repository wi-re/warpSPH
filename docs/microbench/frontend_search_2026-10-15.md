# Frontend simulations after the core search/index rework

Progress: [SCHEME_CLEANUP_PLAN.md](../../SCHEME_CLEANUP_PLAN.md), indexed in [PLANS.md](../../PLANS.md). Measured 2026-10-15 on RTX PRO 6000 Blackwell; float32.

90 serial trials / 18,000 simulation steps cover CompSPH and CRKSPH Sod tubes, 2D/3D, and the sun2017DeltaSPH dam break. Each trial has 40 warmup + 160 measured steps; two trials per setting. Plotting/video/store are off for this small A/B sweep; progress, case diagnostics, adaptive timesteps and rebuilds remain on.

The new index/collection/query defaults preserve final particle tensors and scalar trajectories **bitwise** in the matched comparisons. Default pipelined graph runs also match the sequential runs bitwise. The explicit eager-grid override has small float32 differences (maximum final position error below 4e-7); it is treated separately.

## Matched current-code controls

Full loop ms/step, pipeline off. The control uses current sources with 64-bit IDs/offsets and fused/input search; it is not an old checkout. The intermediate control uses 32-bit storage with fused/input search.

| Case | N | 64-bit fused/input | 32-bit fused/input | New defaults | Peak Torch MB, wide → new |
|---|---:|---:|---:|---:|---:|
| CompSPH 2D eager | 2,500 | 6.295 | 6.168 | 6.273 | 61.2 → 54.3 |
| CompSPH 2D graph | 2,500 | 2.380 | 2.368 | 2.388 | 78.3 → 75.9 |
| CRKSPH 2D eager | 2,500 | 7.634 | 7.807 | 7.594 | 61.3 → 54.4 |
| CRKSPH 2D graph | 2,500 | 2.928 | 2.952 | 2.927 | 78.3 → 75.9 |
| Delta-SPH eager | 6,361 | 14.586 | 14.479 | 14.817 | 94.8 → 77.4 |
| Delta-SPH graph | 6,361 | 3.966 | 3.946 | 4.014 | 194.8 → 178.2 |
| CompSPH 3D eager | 20,032 | 22.325 | 21.936 | 22.247 | 1,445.3 → 1,149.0 |
| CRKSPH 2D eager | 40,000 | 10.829 | 10.732 | 10.784 | 477.2 → 368.4 |
| CompSPH 3D eager | 79,984 | 73.331 | 71.916 | 72.631 | 5,252.3 → 4,148.9 |

The main gain is **20–23% lower peak Torch allocation** in the larger eager cases. There is no large full-step speedup in these naturally ordered cases. Small timing differences need caution: two trials only, with delta-SPH eager automatic spanning 14.32–15.31 ms.

Independent 32-bit attribution trials: 80k CompSPH takes 71.891 ms fused/input, 71.908 ms split/input and 72.627 ms split/spatial. Spatial scheduling costs about 1% here; `compute.neighborQueryOrder='input'` is available. The search microbenchmarks' gains on shuffled storage do not imply equivalent gains on an ordered simulation that mostly reuses its lists. During the 160 active steps there are 10 Python CSR builds for delta-SPH, 32 for 40k CRKSPH and 53 for 80k CompSPH.

Forcing eager grid construction is slower on the small compressible graph cases: CompSPH 2.397 → 2.533 ms, CRKSPH 2.949 → 3.084 ms. It changes rounding after about step 140: positions differ by at most 3.97e-7 / 2.68e-7, with identical simulated times. Every tensor's maximum absolute error divided by max(1, its infinity norm) remains below 1.83e-5 / 7.12e-6. Delta-SPH has no scheme pair-grid factory, so this setting is a no-op there.

With the **default output pipeline enabled**, new-default full-loop times are 2.188 ms CompSPH, 2.722 ms CRKSPH and 3.496 ms delta-SPH, versus 2.177 / 2.721 / 3.505 ms for matched wide controls. CompSPH/CRKSPH have 2 captures, 199 replays, zero fallbacks; delta-SPH has 5 captures, 199 replays, 7 rebuild-related fallbacks for every policy. No graph is disabled.

## Comparison with existing records

[The earlier CompSPH/CRKSPH study](../compsph_step/README.md) and [scheme cleanup table](../../SCHEME_CLEANUP_PLAN.md) report CRKSPH eager around 7.5–7.6 ms at 2.5k and 10.7 ms at 40k, graph 2.8 ms at 2.5k. The new runs are in the same range: 7.594 / 10.784 ms eager, 2.722 ms with the default pipeline. CompSPH's default pipelined 2.188 ms is also close to the earlier 2.2 ms example.

The old delta-SPH audit's 18.2 / 4.7 ms eager/graph figures include audit overhead. Other old measurements use different windows or mention concurrent GPU work. The current 3D slab at nx=40 has 20,032 particles, whereas older 3D tables include smaller geometries. These historical numbers are context; use the new same-source controls to attribute performance changes.

This is performance-policy parity over the measured windows, not an independent physical-accuracy validation. Existing case physics defaults, including the CRK Gather-mode warning, were preserved. Full pressure-solver/MFM trajectory batteries remain separate work.

## Evidence and reproduction

The [full core report and figures](../../../warpSPHCore/docs/development/frontend-simulation-validation.md) contain methods, the detailed override comparison and commands. The [consolidated source-pinned record](../../../warpSPHCore/docs/regression/frontend_simulations_2026_10_15.json) indexes all 90 trial JSONs. The harness is [bench_frontend_search.py](../../../warpSPHCore/scripts/benchmarks/bench_frontend_search.py); the initial SDF setup error was corrected in the harness without modifying simulation sources.

![Full-loop time and peak memory](../../../warpSPHCore/docs/development/assets/frontend-simulations-2026-10-15.png)

## Developed-flow follow-up

[Sloshing motion measurements](sloshing_motion_2026-10-15.md) add matched startup and impact windows: 0 → 122 / 118 CSR build calls per 1,000 steps, with loss of original-row locality. Complete rebuilds improve 7–8%, while whole-loop gains remain around 2% at 10.7k particles. This qualifies the short-window conclusion above; larger moving flows remain unmeasured.
