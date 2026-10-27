# MFM / MFV staged-module and performance pass, 2026-10-21

Tracked by [SCHEME_CLEANUP_PLAN.md](../../SCHEME_CLEANUP_PLAN.md).
Priority: correctness, modularity and structure, general performance, then
focused optimization. This pass closes the staged-module cleanup and measures
bounded overhead reductions. It does not change the physical model or claim
that the open MFM/MFV numerical questions are solved.

## Ownership and retained behavior

Time staging lives in `warpSPHIntegrators.schemes.ConservedLagrangian` and
`drivers/mechanics/conserved_lagrangian.py`. Meshless systems use the existing
initialize/preprocess/postprocess/finalize lifecycle, following CRK's compatible
update pattern. Reconstruction and limiters live in `modules/reconstruction`;
moving-face Riemann policy lives in `modules/riemann`; ideal-gas primitive
conversion lives in `modules/eos`. The prepared structured launcher moved from
`mfm.fastLaunch` to `autograd.prepared` in core, retaining the old import shim.
See [the architecture record](ARCHITECTURE_2026-10-21.md).

MFM auto SSPRK and MFV auto Hancock, KDK, default closure none, CFL convention,
face guards, Gaburov/Roe/PVRS/Rusanov retries and the mass-flux cap are preserved.
The historical moving-position SSPRK predictor is explicitly retained; generic
Heun on `(x,Q)` would be a different method. The optional MFV primitive kick
remains experimental and off by default. Public primitives/supports retain
their existing start-stage lag. Actual driver metadata is separate from the
round-trippable SimulationConfig enum.

General improvements retained:

- State-owned geometry handover with explicit clone lineage and tensor-version
  checks replaces two GPU equality reads in the normal runner. Legacy direct
  callers keep their compatibility API; owned paths cannot consume its global slot.
- Fused primitive/EOS and support/diagnostic refresh removes intermediate tensor
  operations. Shared EOS retains Torch reverse/JVP fallbacks and caller-stream
  ownership. Forward-only meshless kernels skip unused backward generation.
- Derived diagnostic fields use copied lifecycle semantics, avoiding stage
  clones of values that will be recomputed; initial public snapshots are preserved.
- Grid and neighbour-list flux entry points share one pair function, reconstruction
  and Riemann implementation. Mode/order/prediction remain runtime choices.
  Broad specialization was rejected because its small gain did not justify its
  compilation cost. No cache-policy or blanket register-cap change is adopted.

Grid qualification also found a correctness bug: incoming momentum accumulation
was reset between cells while mass/energy were retained. Frontend and core
compatibility flux paths now carry all three. The normal single-range list
path is unchanged by this fix. Prepared traversal metadata also remains valid
when a no-grad call falls back because an input carries requires-grad metadata.

## Qualification

- Frontend FP32: **238 passed**, covering meshless modules/staging, ordinary
  Riemann solvers, runner and configuration/schema contracts.
- Focused frontend FP64: **72 passed**, including 1D/2D/3D rate-oracle and grid/list
  comparisons, device timesteps, closure/face guards, EOS derivatives and
  nondefault-stream/external-capture ownership.
- Core meshless/stream contracts: **125 passed, 12 skipped**, including
  independent conservation/affine-field oracles, degenerate geometry, retry
  behavior, forward-only guards and shared prepared/ordinary stream handling.
- Integrators full CPU gate: **8,792 passed, 639 skipped**. Independent oscillator
  recurrences, stage clocks, lifecycle, ownership and reverse gradcheck cover
  the new package driver. Integrator source was unchanged during final qualification.
- **60 old/new saved-state pairs**: Sod, Sod2D and KH at eight steps, default and
  explicit Hancock/SSPRK/KDK, both precisions, plus first-order reconstruction,
  projected closure and MFV primitive updates. All fields present and finite.
  Largest FP32 differences normalized by each old field's maximum: Q **2.11e-6**,
  velocity **1.29e-5**, position **6.03e-8**. FP64: Q **1.08e-10**, velocity
  **4.31e-10**, position **5.65e-13**; the largest FP64 case is Sod2D SSPRK,
  with absolute velocity difference **1.51e-10**. These are measured bounds,
  not bitwise claims. The earlier package-driver move alone was bitwise across
  all 24 FP32 temporal/case states against the fused implementation.

The old frontend is a saved source snapshot; both sides use the current shared
core and integrators. Thus the throughput comparison measures the staged consumer change on common
dependency versions and includes no claimed speedup from the separately
reported shared core register pass. Tiny A/B dumps and timing loops omit video; physical
controls retain video, progress and daemon publishing.

## Throughput

RTX PRO 6000 Blackwell, 65,536 KH particles (`nx=256`), ten warm-up and
100 timed steps, three alternating trials per source/precision. Entries are
medians of trial means; tight loops disable rendering. Both sides use the
same current core and integrators.

| Scheme / precision | Wall ms/step before → after | Integrator ms before → after |
|---|---:|---:|
| MFM / FP32 | 7.256 → 7.249 | 4.468 → 4.431 |
| MFV / FP32 | 4.346 → 4.066 | 1.505 → 1.286 |
| MFM / FP64 | 20.783 → 20.650 | 15.788 → 15.710 |
| MFV / FP64 | 12.857 → 12.650 | 7.862 → 7.659 |

MFV FP32 improves **6.4%** in total step time and **14.5%** in integrator
time. MFM is effectively neutral (0.1% FP32, 0.6% FP64). MFV FP64 improves
only 1.6% by the median, with overlapping trial ranges; no robust large FP64
speedup is claimed. FP32 wall ranges: MFM before 7.252–7.431, after
7.220–7.277; MFV before 4.305–4.370, after 4.038–4.189 ms.

Three-step FP32 CUDA traces count **594→522 launches, 23→20 stream
synchronizations per MFM step**, and **326→278 launches, 17→12 stream
synchronizations per MFV step**. These are whole profiled steps including
common runner/benchmark work; profiler boundary device synchronizations are
excluded from that comparison. Remaining geometry/runner overhead is real.

A single additional pair from the saved periodic step-3000 developed-flow
snapshot (same 65,536 particles, ten warm-up plus 50 timed steps) gives MFM
**11.104→10.958 ms** and MFV **6.343→6.216 ms**. Both complete positively.
These are indicative developed-flow costs, not repeated-trial speedup claims.
Only positions, supports and Q are imported for this workload; it is not a
restart-codec qualification.

The earlier bounded NCU screen measured the old generic FP64 flux at 4.25 ms,
184 registers/thread and 16.67% theoretical occupancy. Broad specialization
reached 4.13 ms and 178 registers at the same occupancy, with costly
compilation. It was rejected; those counters do not describe the final kernel.
The retained traversal-only pilot and the separately qualified shared core
register improvements are documented independently. No new cache policy is
introduced while solver correctness questions remain open.

## Recorded physical controls

Old/new FP32 KH runs at `nx=128` (16,384 particles) reach exactly **t=0.3**,
with progress, VisPy video, retained trajectories and the existing daemon.
MFM completes 819 steps on each source; MFV 817. All six daemon records
(four initial controls plus two title-corrected updated controls) are verified
completed. The title-only reruns are bitwise identical in every saved field.
Final video frames were inspected; they reach t=0.3 and show the actual drivers.

| Updated run | Relative mass drift | Relative energy drift | Minimum raw u |
|---|---:|---:|---:|
| MFM | 0.000e+00 | -3.161e-08 | 1.868284 |
| MFV | -9.689e-08 | -5.918e-09 | 1.868205 |

MFM per-particle mass change is exactly zero. MFV mass exchange is expected;
its minimum mass is 6.102e-5. Old/new final Q differences normalized by the
old field maximum are **8.63e-6 (MFM)** and **8.80e-6 (MFV)**; position
differences are below 7.16e-7. These short positive/conservation controls and
reference/recurrence checks establish implementation preservation; they do
not establish long-time convergence, face consistency or every physical-policy
choice. Projected closure and primitive MFV remain optional.

## Evidence and reproduction

[Curated values and run IDs](evidence/performance_2026-10-21.json). Raw states,
traces, logs, command records, NCU reports and source snapshots are archived at:
`/home/lu26029/dev/warpSPHDevelopment/core/experiments/meshless-cleanup-2026-10-21`. The archive contains a SHA-256 manifest. No raw profiler outputs
are added to Git. The earlier interrupted FP64/codegen attempt and corrected
stream-test setup failures are retained as superseded logs, not passing gates.

For each precision/source, run the following sequentially; alternate source
order across three trials. Set PYTHONPATH to the archived old frontend src
for the reference; use the editable frontend for the updated source.

```bash
/home/lu26029/miniconda3/envs/warp/bin/python scripts/bench_compressible.py \
  --scheme MFV --case kelvinHelmholtz --nx 256 --precision float32 \
  --warmup 10 --steps 100 --out /tmp/mfv-timing

/home/lu26029/miniconda3/envs/warp/bin/python scripts/bench_compressible.py \
  --scheme MFV --case kelvinHelmholtz --nx 128 --precision float32 \
  --warmup 10 --steps 100 --end .3 --video --plot-interval 50 --daemon \
  --out /tmp/mfv-recorded-control
```

No timing-driven physics, temporal, closure or precision defaults changed.
AC cleanup, long-time numerical investigations and focused cache/load/register
experiments remain separate work.
