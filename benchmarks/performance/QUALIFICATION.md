# Initial qualification — 3 November 2026

These are small functional/timing-contract checks, **not hardware-performance
conclusions**. Host: Intel Core Ultra 9 285K, RTX PRO 6000 Blackwell Workstation
Edition, Torch 2.13.0+cu130, Warp 1.17.0. Frontend/core/integrator source versions
are 0.7.1; installed editable-distribution version strings on this workstation
are older, so the helper records both source and distribution versions.
Core scripts are published in commit `4ecfa52`.

| Check | Fixture | Result |
|---|---|---|
| Harness protocol and report tests | stdlib only, runtime-free planning, flag identity, failure exit, timeout/log retention, no allocation on capacity skip | 9 pass |
| CPU initial component suite | 12³ FP32, input order, 17 operator/matrix stages, 3 search phases, 3 memory controls, CRK accepted steps and RHS | 25 pass |
| CUDA initial component suite | 12³ FP32, input order, 17 operator/matrix stages, 3 search phases, 3 memory controls | 23 pass |
| CPU Morton components | 12³ FP32, every particle/value/matrix row permuted before topology construction | all 17 stages pass |
| CUDA production compressed-gas steps | 12³, CSR/chunked input paths for CRKSPH, CompSPH, MFM, MFV; Morton/chunked CRKSPH and CompSPH | all 10 supported step configurations pass |
| CPU native solver families | 16², input/Morton, deltaSPH, Sun2017, ACSPH, IISPH, DFSPH, Predictor, Shift | 14 pass |
| CUDA native solver families | same 16² configurations | 14 pass |
| CPU FP64/thread isolation | 12³, 2 Torch threads, input/Morton density and compatible-gas chunked steps, input meshless chunked steps | 8 pass; 2 MFM/MFV Morton entries unsupported |
| CUDA graph selection | KH16² CRKSPH, input/Morton × eager/auto/graph, fresh trajectories/repetitions | 6 pass; effective eager/graph selection verified |
| CUDA extra compute flags | Morton12³ gradient, FP32/FP64 × grid/rebuild, 8 lanes, int32 CSR offsets | 4 pass |
| Separate CUDA profiler pass | memory, search in both orders, gradient in both orders, chunked CRK steps in both orders, input RHS | 14 pass; Morton per-call RHS instrumentation unsupported |

Profiler Chrome JSON was parsed and checked for GPU kernel/memcpy activity,
CPU/core sections and synchronization API summaries. Morton step traces retained
GPU work without replacing the audited production RHS. Base timings ran with core
profiler hooks disabled. CSV/Markdown/JSON analysis was exercised on component,
profile and graph sweeps. Repetition windows preserve the same warmup age, and
final states are checked outside timed windows.

Expected unsupported combinations are documented in the [guide](README.md).
The initial MFM/MFV Morton probe exposed the missing audited adapter; the final
worker explicitly identifies that limitation before constructing the system.
No library solver behavior was changed to make a benchmark pass.

No 3090, GH200, large/developed-flow or distributed performance conclusion is
claimed here. No Isambard job was submitted or resumed. Run matched configurations
on each machine before comparing throughput; the small smoke timings are not
suitable for hardware ranking.
