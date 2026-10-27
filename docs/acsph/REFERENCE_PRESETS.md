# dynGPUSPH ACSPH reference presets

Plan: [ACSPH_PLAN.md](../../ACSPH_PLAN.md). Bring-up started 2026-10-27.
The reference is a separate executable; warpSPH's ACSPH equations/defaults are
unchanged by this work.

## Source and supplied cases

Local checkout: `/home/lu26029/dev/dynGPUSPH`, commit `2dbc27a` (the
`FSI_devel` paper-era code identified in [the earlier source comparison](DYNGPUSPH_COMPARISON.md)).
The four user-supplied presets live under `tosend/`. Neither that checkout nor
its inputs are edited. The runner copies the six input files it actually reads;
shipped `fluid_final.dat`, `probe.plt` and other results are not used as inputs.
Every copied input, source file and binary has a SHA-256 receipt.

| Preset | Initial fluid | Input spacing | Original endpoint | Dual-time controls |
|---|---:|---:|---:|---|
| `Impact200` | 161,202 | 0.005 | 1 s | dt/dtau 2, log residual −8, cap 200 |
| `LDCAVITY_400_2L_RE1000` | 160,801 | 0.002493766 | 60 s | dt/dtau 2, log residual −6, cap 100 |
| `SquareRotation400_2L` | 160,801 | 0.0025 | 6 s | dt/dtau 10, log residual −6, cap 200 |
| `dambreak_Lobovsky_400_2L` | procedural rectangle | 0.00075 | 2 s | dt/dtau 2, log residual −8, cap 200 |

These controls belong to ACSPH's **dual-time** solve, not a DFSPH divergence
projection. All four select `pmethod=0` (2L pressure diffusion) and h/dx = 2.
Impact is two blocks moving toward each other, without walls/gravity; its
initial gap closes at about 0.1 s. The square carries its supplied negative
pressure field and rotational velocity. Lobovsky starts with hydrostatic
pressure. The cavity starts at rest and ramps a prescribed lid to U=1 over
0.1 s, through `fsi.dat` structure type 7 / marker 10 (`dynbound.cu:159,326`);
the initially zero wall velocity in `boundaries.dat` is deliberate.

**Keep the cavity's two spacings distinct.** Its coordinates span [0,1] with
401 samples per axis (coordinate increment 1/400), but `fluid.dat` specifies
volume spacing 0.002493766 ≈ 1/401. Mass comes from the latter. Reconstructing
the IC solely from its geometric lattice would change the total mass.
The square instead uses coordinate and volume spacing 1/400 and therefore
has the endpoint-inclusive lattice's excess volume. Do not calibrate or relax
either reference IC silently when matching it in warpSPH.

## Build recipe and flags

Tracked tooling: [`scripts/acsph_reference`](../../scripts/acsph_reference).
The script makes an isolated source copy, applies `cuda13.patch`, substitutes
an output adapter, and records the complete compiler/dependency/source recipe.
Local toolkit: CUDA 13.0.48, GCC 13, Blackwell `sm_120`.

```bash
conda activate warp
python scripts/acsph_reference/build.py \
  --out .tmp/dyngpusph-reference --download
```

Defaults can be overridden with `--source`, `--cuda`, `--eigen`, `--cxx` and
`--arch`. The current recipe expects the `2dbc27a` source and an Eigen tree;
its complete Eigen content hash is recorded. CUDA sample headers are pinned
to NVIDIA `cuda-samples/v12.9`, nanoflann to `v1.4.3`, and all downloaded
headers are checked against tracked hashes. Dependencies are not installed
system-wide. An existing source/build directory is never silently replaced.

Base flags: `-std=c++17 -O3 -use_fast_math -arch=sm_120 -DDOF2=1 -DLINEARPG`,
plus the three Eigen flags in the upstream README. **No `SINGLEP`: the supplied
reference runs in FP64.** `DOF2` must have value 1 because upstream mixes
`#ifdef DOF2` with `#if DOF2`. The active pseudo-integrator is hard-coded SSPRK3
with CFL_tau=0.75 in `gpusph3d.cu:905–914`; the old makefile's `RK4` macro has
no active consumers in this checkout and does not select that tableau.

The cavity executable additionally compiles `HashFunctions.cu` with
`NOSLIP` and `LDCAVITY`; these select physical no-slip viscosity and the
cavity's explicit corner-normal rules. Other translation units have no active
consumers of these two macros. Base/free-surface cases use the source's
free-slip handling. `PGTIC`, `PGTIC_2`, `NOSHIFT`, `SHIFTNB` and surface-tension
macros are left off. The presets contain no build manifest proving which TIC
flag produced their shipped results; the present baseline is the supplied
source's unselected-TIC build, not an assertion of exact paper build flags.

Compatibility changes are explicit in the patch:

- Replace removed Thrust base classes/identity and annotation names without
  changing functor arithmetic; supply the absent positive-stride range helper.
- Include missing standard headers and adapt obsolete GPU diagnostic fields.
- Keep the CUDA context alive until local Thrust containers are destroyed.
  The original end-of-main `cudaDeviceReset` causes a modern-Thrust invalid
  free after otherwise completed runs. The startup reset is retained.
- Add per-real-step progress, maximum-speed checking and a bounded-run step
  stop. Add a terminal snapshot even when stopping between output times.

The patch preserves the actual solver, including its `(3/2)` integer-division
quirk, truncated convergence fallback, pressure/velocity JST terms, acoustic
damper, wall position bounce and external shifting. They must be accounted for
in a reference comparison; matching only the published equations is insufficient.
A positive viscosity also remains when input `nu=0` if `alpha>0`:
`nu_eff=max(nu, alpha*h*c0/8)` in 2D (`kernel.cuh:283–288`).

## Runs, output and interpretation

```bash
python scripts/acsph_reference/run.py Impact200 \
  --binary .tmp/dyngpusph-reference/build/gpuSph2d \
  --out .tmp/dyngpusph-reference/runs/impact \
  --steps 200 --t-limit 0.12
```

Use `gpuSph2d-cavity` for the cavity. Runs are sequential on one GPU.
`--t-limit` is an explicit input override; particle resolution, mass, initial
fields, tolerances and output cadence stay as supplied. Wall timeout, timeout
without a completed real step, nonfinite-state and maximum-speed stops retain
failure receipts. `result.json` records exit status, every real step, iterations,
dt, step-event time and wall time. The source's event interval includes host
work/launch gaps and output; the historical field name `solver_ms` is **not**
a kernel-only timing. Wall ms/step includes startup, output and teardown.

The TecIO replacement writes full-precision particle snapshots instead of
requiring a proprietary output library. Positions, velocity, pressure, constant
physical density, marker, lambda and auxiliary field retain their source values.
The output volume is **m/rho**, reconstructed from immutable particle masses:
the original `V_d` output buffer is initialized to zero and is not the particle
mass/volume state. No reconstructed quantity is fed back into the solver.
Changing particle membership is refused by this adapter. Kernel sampling
density is a separate diagnostic; constant `rho` does not measure sampling quality.

`run.py` always invokes `render.py` after execution (including retained failed
runs) and checks its exit status. `render.py` uses VisPy/EGL and writes `output.mp4`, initial/final PNGs and
`snapshot-metrics.json`. Colors share fixed ranges across a run. A sample shorter
than the original output cadence has only initial/final frames; these are
startup checks, not evidence of a well-resolved transient or long-time stability.

Local receipts: `.tmp/dyngpusph-reference-2026-10-27/` (preserve with `KEEP`).
The original 20-step impact smoke advanced to t=0.00224026 with unchanged KE
2.015025 and then failed during CUDA teardown; that failed run is retained.
The corrected four-case samples are recorded separately below once complete.

Before promoting this to a quantitative reference, extend the relevant case
through its discriminating event, verify convergence/cap behavior, and compare
warpSPH using these exact particles, mass, initial pressure/velocity, kernel,
viscosity and wall motion. Do not infer a solver ranking from the startup samples.

## Bring-up results — 2026-10-27

All four samples kept their original resolution and numerical controls, exited
0, retained all particles/mass, and rendered successfully. Initial/final fields
were visually inspected. These endpoints deliberately stop far short of the
full supplied campaigns. Only the impact sample crosses its first collision.

| Sample | Actual final t | Real steps | Iterations / step | Steps at cap | Wall seconds | Wall ms/step |
|---|---:|---:|---:|---:|---:|---:|
| Impact | 0.120420 | 69 | 15–200 | 19 | 114.103 | 1653.7 |
| Cavity startup | 0.011809 | 29 | 1–9 | 0 | 6.307 | 217.5 |
| Rotating square startup | 0.020664 | 40 | 48–52 | 0 | 47.842 | 1196.1 |
| Lobovsky startup | 0.005052 | 25 | 20–200 | 17 | 175.116 | 7004.6 |

Timings are observed costs, **not isolated performance benchmarks**. A separate
`.tmp/driven-turbulence-20261027/probe.py` GPU workload appeared during Lobovsky;
a CPU-only clean rebuild also overlapped part of that sample. Neither workload
was stopped. Source event timing includes host work and launch gaps.

Impact preserves KE=2.015025 before contact, then reaches KE=1.015377 at the
endpoint, with visible lateral jets. Snapshot pressure reaches +59.2639 during
the collision and −56.0135 at the endpoint (rho0=1 in this preset). Its
post-contact steps hit the iteration cap, so this is **runnable reference
behavior, not a converged pressure validation**. Keep this failure to converge
visible when comparing a warpSPH port.

The rotating square starts at KE=84.16927344 and ends at 84.16903097
(−0.0002881%); maximum speed stays near 0.7071. It preserves the supplied
negative-pressure field in this short window. The cavity develops the expected
thin moving-lid layer during its ramp (fluid vmax=0.0540 at the endpoint);
this is not a developed-cavity/Re=1000 profile validation. Lobovsky starts at
rest with p_max=2935.6425 and ends at KE=0.0605161 / vmax=0.174653. Its early
17 steps hit the cap; the final step needs 21 iterations. This is not a
long-time dam-break validation.

Physical initial volumes from immutable m/rho are 4.03005 (impact),
1.000000332 (cavity), 1.00500625 (square) and 0.18 (Lobovsky). Their constancy
is bookkeeping for this constant-density solver, not proof of geometric
volume conservation or good particle sampling.

A second clean build from the tracked recipe succeeds for both executables.
Its patched-source hashes match the sample build. A repeated 20-step impact
control exits 0 with automatic video generation and matches **every stored
field bitwise** against the retained pre-teardown-fix smoke (including x/v/p,
rho, marker and reconstructed volume). Binary file hashes differ across build
roots, so each run retains the hash of the executable actually used. The
runner also rejects the base executable for the cavity before launching CUDA.
The retained failed 20-step smoke took 10.840 s (542.0 wall ms/step); the
clean-rebuild 20-step control took 9.373 s (468.6 wall ms/step). These include
startup/output and are not a performance A/B.
No warpSPH numerical implementation/default was changed. The existing comments
which described the 1/N convergence normalization as stricter at larger N were
corrected: fixed log tolerance permits RMS residual proportional to sqrt(N).

### Reference diagnostic traps

Two upstream diagnostics need interpretation before curve comparisons:

- `probe.plt`'s `VOL` is an inverse kernel-count volume, not sum(m/rho).
  `getVolHashD` (`HashFunctions_cuda.cuh:2675`) initializes its count with W(0)
  and then includes the self row again in the neighbor loop. For example,
  impact starts with native `VOL=3.557622`, while sum(m/rho)=4.03005. That
  difference does not mean the solver removed physical mass. The source
  diagnostic is retained untouched; use explicit, separately named sampling
  and physical-volume measures in comparisons.
- The final standalone `getOutputParams` call labels the terminal state with
  `act_time + deltatc`, one step later than its actual time. Its `NITER` also
  includes the counter's initial 1. Use snapshot/per-step timestamps and the
  runner's corrected iteration count; preserve `probe.plt` as raw evidence.

Local movies are under the receipt root's `runs/{impact-contact,cavity-startup,
square-startup,lobovsky-startup}/output.mp4`; raw logs, inputs, snapshots,
metrics, build receipts and the failed smoke are retained beside them.
Next: initialize warpSPH from these exact reference particles and fields, verify
initial mass/kernel/viscosity/wall-motion agreement, then compare short
trajectories and convergence before extending the campaigns.
