# Moving delta-SPH: developed sloshing benchmark

2026-10-15. Follow-up to [the frontend index/search study](frontend_search_2026-10-15.md), tracked in [PLANS.md](../../PLANS.md).

Particle motion changes the workload materially. Matched windows from stored TC10 sloshing frames give **zero CSR build calls per 1,000 measured startup steps**, versus **122 / 118** around the first / third impacts. The original particle order is preserved; at the third fixture, consecutive fluid-array rows have a 95th-percentile spatial separation of 42 particle spacings, versus one initially.

35 serial trials / 42,000 steps cover 24 uninstrumented timing runs, nine event-instrumented diagnostics and two frontend video runs. All final tensors, physical scalar trajectories and rebuild histories match **bitwise** within each phase, including diagnostic and video runs. No requested graph is disabled.

## Full-loop performance

10,660 particles (4,200 fluid), deltaSPH, symplectic Euler, Wendland4, dt 1e-4, nx=200, default whole-step graph/output pipeline. Two trials per policy, 200 warmup + 1,000 active steps. The wide control is current code with 64-bit IDs/offsets and fused/input search, not an old checkout.

| Phase | Measured time, s | Wide fused/input | 32-bit fused/input | 32-bit split/input | New defaults | CSR calls / 1,000 steps |
|---|---|---:|---:|---:|---:|---:|
| Startup | 0.0202–0.1201 | 2.737 ms | 2.675 ms | 2.675 ms | 2.675 ms | 0 |
| First impact | 2.3402–2.4400 | 5.295 ms | 5.198 ms | 5.169 ms | 5.195 ms | 122 |
| Third impact period | 5.5802–5.6803 | 5.282 ms | 5.181 ms | 5.132 ms | 5.162 ms | 118 |

The new defaults are about **2% faster end to end** against the wide control; changes beyond 32-bit fused/input save 0–0.4% here. At the impacts, peak Torch allocation falls about 214 → 182 MB (15%). Larger moving flows remain unmeasured at this point.

Separate diagnostic passes measure complete grid + CSR rebuilds at 2.502 → 2.327 ms (first impact) and 2.445 → 2.244 ms (third): **7–8% faster**. Stored Verlet output throughput rises 0.487 → 0.523 and 0.490 → 0.534 billion pairs/s. This counts stored skin neighbours and is not whole-scheme physical interactions/s. Diagnostic event overhead and host-fed GPU idle gaps are included; these passes are excluded from headline timings.

Graph activity also changes: startup has 1 capture / 0 fallbacks, first impact 93 / 92, third 96 / 95 over each full 1,200-step trial. Every policy has the same counters. CSR calls include capture validation/additional stage work, not just unique generations. Complete rebuild intervals amortize to about 0.27–0.28 ms per active step under the defaults; the rest of the phase's cost increase also includes physical work, capture and fallback handling. **Profile capture/fallback cost next**, alongside grid construction and per-stage kernels.

## Fixtures and evidence

The existing Oct 12 trajectory provides positions, velocities, densities and static particle/ghost metadata at t≈0, 2.32 and 5.56 s. Static fields and fixed wall/ghost positions match the current case exactly. Current code rebuilds derived fields and adjacency; roll/gravity are set at fixture time before the first step. This is a matched workload fixture, not a claim of exact historical continuation or independent physical-accuracy validation.

[Full core report, native videos and reproduction commands](../../../warpSPHCore/docs/development/sloshing-motion-validation.md). [Source-pinned results and portable fixture directory](../../../warpSPHCore/docs/regression/sloshing_motion_2026_10_15/summary.json).

![Motion benchmark timing and stored-pair throughput](../../../warpSPHCore/docs/development/assets/sloshing-motion-2026-10-15.png)
