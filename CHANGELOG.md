# Changelog

## 0.7.1 — 2026-11-03

Coordinated GitHub checkpoint with warpSPHCore and warpSPHIntegrators 0.7.1.

- Add opt-in fixed-count Morton particle layouts, persistent particle identity,
  UID-ordered exports and declared boundary/ghost/body/source adapters.
- Preserve accepted continuation, numerical histories and supported FD-JFNK
  inputs across layout changes; retain native solver and CUDA graph paths.
- Integrate eager whole-row chunked neighbours and replicated-state multi-GPU
  execution, native checkpoints, rank/partition changes and allocation recovery.
- Include the committed native incompressible solver, ACSPH, driven compressible
  flow and memory/lifetime improvements accumulated since the prior checkpoint.
- Require warpSPHCore>=0.7.1 and warpSPHIntegrators>=0.7.1.

Input ordering remains the default. Chunked execution retains documented eager
periodic support limits; native C++ backend development, population changes and
physical inlet/outlet models remain separate. See the particle-layout integration
report and individual numerical qualification records for supported scope.
