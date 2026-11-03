# CPU and single-GPU performance

Run from the **warpSPH source checkout**, using the same Python environment as
the editable `warpSPHCore`, `warpSPHIntegrators` and `warpSPHPlotting` checkouts.
Update both frontend and core to the benchmark commits. No new Python dependency
is required. The simulator fixtures use the Warp runtime. After the native core
merge, `--operator-backend native` selects native core operators inside that
runtime; this is a hybrid run, and both selections are recorded. A native request
cannot silently run Warp under a native label. Native-only operator/chunk tests
are available in the core's `scripts/benchmarks/bench_chunking.py`; see its
`CHUNKING.md`. The older flat operator/search fixtures still qualify Warp only.
The initial CPU/CUDA/FP64, layout, solver-family and graph checks are recorded in
[QUALIFICATION.md](QUALIFICATION.md).

The default sweep compares **input and Morton storage**, including operators,
search, and real accepted simulator steps. Input order is the regular lattice's
lexicographic order, rather than randomly shuffled particles. Defaults are FP32,
3D, eager execution, one CPU thread and lane, and an 8 GiB conservative capacity
screen. No simulation exports, videos or physical diagnostics are timed.

## A matched 3090 / workstation / GH200 comparison

First inspect the requests without importing Torch/Warp or touching the GPU:

```sh
python benchmarks/performance/run.py --device cuda:0 --threads 4 \
  --nx 16 32 48 --particle-order input morton --plan
```

Run exactly the same options on each GPU; only change the output directory:

```sh
python benchmarks/performance/run.py --device cuda:0 --threads 4 \
  --nx 16 32 48 --particle-order input morton \
  --warmup 5 --repeats 7 --calls 10 --steps 10 \
  --output results/performance/3090
```

Use a fresh output directory for every run. The default suites cover copy/triad/
random gather, hash-grid build, CSR query against a retained grid, complete search
rebuild, density/gradient/divergence/Laplacian/CRK factors/pseudoinverse operators,
and CRKSPH/CompSPH/MFM/MFV steps on the existing driven-turbulence fixture. The
fixture starts from rest (seed 1234); this measures startup-age steps, **not a
developed turbulent checkpoint**. Warmup age must match across machines.

```sh
python benchmarks/performance/analyze.py \
  results/performance/3090 results/performance/blackwell results/performance/gh200 \
  --output results/performance/comparison
```

This writes a Markdown report, flat measurements CSV, cross-machine ratios JSON,
and compute-flag comparisons JSON. It pairs identical numerical requests and
timing windows, including precision, threads, layout, topology and index widths.
It also shows input/Morton and other flag comparisons **within each machine**.
Failures and capacity/unsupported cases remain visible and never become timings.
Raw JSON retains samples, hardware, environment, source revisions/digests, complete
solver specifications and effective compute flags. GPU clock/temperature samples
are metadata, not proof that clocks stayed constant throughout a run.

## CPU and individual components

```sh
# Thread sweeps are fresh worker processes (OMP/MKL set before runtime import).
python benchmarks/performance/run.py --device cpu --threads 1 4 8 \
  --nx 16 32 --output results/performance/cpu

# All seven SPH operators, CRK/renormalization stages and local matrix stages.
python benchmarks/performance/run.py --device cuda:0 --suite operators search \
  --operators all --traversal adjacency grid rebuild --lanes 1 8 32 \
  --nx 32 --output results/performance/operators

# Separate instrumented RHS pass; leave full-step timings free of RHS events.
python benchmarks/performance/run.py --device cuda:0 --suite rhs \
  --particle-order input --neighborhood csr chunked --nx 32 \
  --output results/performance/rhs

# The native WCSPH/ACSPH/projection case fixtures are currently 2D only.
python benchmarks/performance/run.py --device cuda:0 --dim 2 --nx 32 64 \
  --suite steps --schemes deltaSPH sun2017DeltaSPH artificialCompressible \
    iisph dfsph dfsphPredictor dfsphShift \
  --output results/performance/periodic2d

# Compare iteration-check cadence and neighbour-list skins in separate runs.
python benchmarks/performance/run.py --device cuda:0 --dim 2 --nx 64 \
  --suite steps --schemes dfsph --iteration-checks every adaptive \
  --verlet-scale 1.3 --output results/performance/df-cadence
```

`--neighbors` controls only the fixed microbenchmark lattice; actual directed
edges and mean degree are reported. Solver steps use each case's physical support
policy, recorded in the full specification. Do not equate requested degree with
the actual solver degree. CPU threads control Torch/OMP/MKL; Warp CPU kernels may
not use those threads. Record affinity and compare identical thread counts first.
Use `taskset` externally if you need fixed cores/NUMA placement.

Memory controls use preallocated Torch tensors. `--memory-mib 256` is the size of
**each floating array**; gather additionally holds an int64 random permutation.
Copy counts one read/one write; triad two reads/one write; gather one index read,
one source read and one destination write. Reported GB/s is **logical traffic**,
not measured DRAM bandwidth; gather can be limited by latency and cache behavior.
Increase the working set to test cache sensitivity. Actual HBM/GDDR/CPU DRAM
traffic requires separate hardware-counter profiling.

## Compute flags and unsupported combinations

- `--particle-order input morton` permutes **all** microbenchmark particle fields
  before topology construction. Microbenchmark sorting is setup, outside timing.
  Simulator steps use the production adapter, so maintenance/adoption is timed.
- `--neighborhood csr chunked` sweeps simulator topology representation. Chunked
  cases use `--edge-budget 8000000 --max-rows 65536` by default on one device,
  without distributed communication. Keep these budgets identical across GPUs.
  The driven-turbulence source/Morton adapter currently requires chunked lists;
  its CSR/Morton entry is explicitly unsupported. A single chunked GPU provides
  the useful baseline for the distributed path used on Isambard.
  MFM/MFV currently have no audited single-device Morton adapter, including the
  chunked route; both Morton entries are recorded as unsupported.
  Use the core chunking diagnostic to isolate identical fixed neighbour lists
  split into 1/2/4/8 pieces. The original flat operator microbenchmarks do not
  exercise chunk dispatch. Sweep `--edge-budget` and `--max-rows` in separate
  solver runs: their values can strongly affect GPU utilisation. Native operator
  selection does not switch the MFM/MFV controller, which follows the runtime.
- `--execution eager auto graph` is an optional sweep. The driven-turbulence case
  currently requires eager execution. Supply another migrated case with
  `--configuration path/to/RunConfiguration.json` for graph tests, choosing a
  compatible `--schemes` and dimension. The script records requested and effective
  execution, graph presence, solver flags and topology type. Explicit graph
  requests without a graph route are unsupported rather than mislabeled timings.
  `auto` can legitimately choose eager. Pipelined output is disabled because the
  measurement uses accepted-step boundaries.
- `--lanes`, `--iteration-checks`, `--krylov-check-every`,
  `--neighbor-omit-i` / `--no-neighbor-omit-i`, `--collection` and `--query-order`
  expose relevant compute choices. Particle storage and spatial **query scheduling**
  are distinct. Effective flags are retained; some algorithms do not consume every
  flag. Graph captures cannot be counted by Python RHS calls, and Morton adapters
  require audited RHS identities: per-call RHS instrumentation is therefore
  input/eager only. Full-step timing and core-section traces support Morton.
  The benchmark pins `--lanes 1` by default for a stable scalar baseline.
  Use `--lanes default` to measure the production selection, inheriting the
  environment and per-operator defaults. Qualified CUDA kernels normally use
  32 lanes; CRK volume and compatible work retain scalar defaults on flat/grid
  inputs and use 32 lanes on bounded chunks. Large flat lists can be slower
  when these two kernels are forced to 32 lanes. Use `--lanes default 1 32`
  to compare all three selections; the requested default is recorded as JSON
  `null`, distinct from either forced override. CPU kernels stay scalar.
- Custom versioned configurations keep case physics and solver settings. The
  benchmark overrides scheme, size, dimension, device/precision, kernel, compute
  options, accepted-step count and output controls. Fresh starts only; arbitrary
  restart/checkpoint adapters are not yet part of this suite. Default 2D TGV
  fixtures declare zero initial relaxation/jitter and no shifting.

The stopped Isambard 128³ job's retained configuration and run manifest both say
`particleOrder: morton`, `execution: eager`, `cudaGraph: false`. Its historical
four-GPU runtime is not itself a controlled scaling measurement.

## Timing and profiler interpretation

Every case is a fresh subprocess, isolating precision, CPU settings, caches and
allocator lifetime. Warmup and validation are untimed; raw repetitions are saved.
Each microbenchmark sample times `--calls` public operations including allocation
and normal output release, then synchronizes. CUDA events are initialized before
timing. Wall time means completed work; submission time stops before the final
sync; stream intervals include launch gaps and are **not pure kernel execution**.

For solver steps each repetition builds the same deterministic initial state,
advances `--warmup` accepted steps, then measures `--steps` accepted steps. The
window includes neighbour maintenance, timestep hooks and ordinary runner checks/
observation between accepted boundaries; the last observation is excluded and
the initial warmup boundary's observation included. Output and physical diagnostics
are disabled. Per-step runner times are also retained but have a narrower scope.
RHS timings are a separate instrumented trajectory with raw per-call host/stream
samples; they add event/dispatch overhead and must not replace full-step timings.

Add `--profile` to make an additional pass **after** base timings. Core profiler
hooks remain disabled during the base pass. Chrome traces and sync API summaries
are saved next to each result. Step traces start after warmup and stop at the last
measured accepted boundary. They include the benchmark end sync. Open traces in
Perfetto or a Chrome trace viewer to inspect CPU/GPU overlap and core sections.
API durations can overlap; do not subtract summed kernels from wall time and
call the remainder synchronization overhead. Use Nsight Compute on an individual
recorded worker request for hardware counters, separately from these timings:

```sh
ncu --set full --target-processes all -o operator-counters \
  python ../warpSPHCore/scripts/benchmarks/bench_single_device.py \
  --request results/performance/operators/requests/0000-operator-density.json \
  --output results/performance/operator-counters.json
```

`--budget-gib` is a deliberately conservative **estimated** capacity screen, not
an allocator limit. Oversized cases are written as `skipped_capacity`; OOM or
numerical failures remain failures. A 128³ full CRKSPH run previously exceeded
24 GiB locally, so use smaller sizes for the 3090 and compare only common cases.
Workers have a configurable compilation-inclusive timeout (default 900 seconds);
errors and logs survive, and a failed case makes the sweep exit nonzero. Later
multi-GPU strong/weak scaling and native-only simulator qualification remain
separate phases. Core native fixed-topology timings must not be presented as
native end-to-end simulator timings.

Harness-only checks (no runtime/GPU imports):

```sh
python -m unittest discover -s benchmarks/performance -p test_harness.py -v
```
