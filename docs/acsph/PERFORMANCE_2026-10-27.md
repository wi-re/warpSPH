# ACSPH Lobovsky performance spot check — 2026-10-27

**Subsequent fixture finding:** the extended comparison exposed a bad SDF return
contract in the local harness, corrupting derived bounce normals. The old long
local trajectory is invalid for solver ranking. The arithmetic/kernel startup
controls below remain historical bounded measurements; they do not qualify
later wall contact. See [convergence investigation](CONVERGENCE_GAP_2026-10-27.md).

The 54.3 s/step result is dominated by FP64 neighbour kernels and an unfused
operator chain, rather than audit output or launch latency. The initial spot
check found less than 1% benefit from a valid CUDA graph and a 7.2x FP32
speedup with a position-BDF residual floor. The implemented follow-up below
removes that floor and fuses stage residuals: 34.8 s/step in FP64 and 6.66
s/step in FP32, versus the original 54.3 s/step. Shared boundary code is
unchanged.

## Fixed workload and measurements

Same aligned static Lobovsky fixture as the
[operator comparison](LOBOVSKY_COMPARISON_2026-10-27.md): 320000 fluid + 32176
physical wall particles + 32176 internal ghosts, CUDA on RTX PRO 6000 Blackwell,
SSPRK3, **200 pseudo-iterations**, one real step dt=1.2e-5 s. Mapped coefficients,
source history, wall flux, divergence damper, RK treatment and physical-wall
velocity sampling are unchanged. Stage/shift dumps and frozen replays are off.
Each run saves physical initial/final snapshots, progress, watchdogs and video.
The FP32 control explicitly rounds the original IC arrays; it is not an exact
FP64-IC or converged-solution comparison.

| Control | Total wall s (wall ms/step) | Integrator ms/step | Dual-time solve ms | Steady median ms/pseudo-iteration |
|---|---:|---:|---:|---:|
| FP64 eager + bounded profiling | 58.358 (58357.6) | 53767.8 | 53404.4 | 265.19 |
| FP64 graph + known stored-ghost specialization | 57.183 (57183.1) | 53451.1 | 53081.5 | 263.17 |
| FP32 graph, rounded IC performance control | 11.783 (11782.6) | 8162.0 | 7366.5 | 36.45 |

Steady medians exclude the first 15 iterations, including compilation/capture,
three profiler iterations and one call-audit iteration. Full solves include
those one-time costs. The eager run enables named profiler scopes; CPU/profile
instrumentation makes this a spot check, not a precise sub-percent benchmark.
The original three-step audited wall cost was 54.3 s/step. Removing stage I/O
does not remove the roughly 53 s of actual solve work.

The graph trial specializes only the existing early-return branch of
`computeViscosityBoundaryVelocities`: this fixture has immutable sampled wall
and stored ghost rows, and `boundaryVelocitiesPrepared=True`, so it returns
`state.velocities` unchanged. The normal implementation first scans kinds and
reads two device booleans per RK stage. The specialization avoids **six host
syncs per pseudo-iteration**, permits capture and leaves all numerical fields
unchanged. Both precisions capture successfully, make 199 replays and pass the
built-in bitwise eager validation. All stored FP64 fields are additionally
bitwise equal between complete eager/graph steps. The shared boundary source
is untouched; this is a process-local benchmark patch, not a general topology
cache implementation.

## Where the time goes

Three warmed eager iterations contain 798.9 ms of actual CUDA kernel execution:

| Kernel group | CUDA ms / 3 iterations | Share |
|---|---:|---:|
| Divergence | 220.16 | 27.6% |
| Gradients | 143.72 | 18.0% |
| Physical viscosity | 136.96 | 17.1% |
| Pressure force | 99.62 | 12.5% |
| Surface detection | 54.41 | 6.8% |
| Covariance | 47.45 | 5.9% |
| Pressure diffusion | 45.20 | 5.7% |
| Interpolation | 22.73 | 2.8% |
| Surface dilation | 15.10 | 1.9% |
| Other/Torch | 13.50 | 1.7% |

Over 98% is SPH neighbourhood work. The profiler records 1925 CUDA kernel
launches across the three iterations (~642/iteration, including tiny Torch
kernels). The call audit counts **60 SPH operator launches per iteration**:
30 interpolation, 8 gradient, 8 divergence and the remaining pressure,
viscosity, diffusion/surface/covariance operations. That is about **12000 SPH
operator passes per physical step**. This count excludes the few auxiliary
Warp matrix kernels and the generic Torch kernels.

CPU self-time is predominantly waiting: cudaStreamSynchronize is 86.2% of
profiled self CPU time; actual cudaLaunchKernel calls are only 0.56%. The
profiler also adds a stop-time device sync, so that CPU percentage is not a
clean production sync-budget measurement. It does explain why interpreting
CPU wall time as Python computation would be wrong. The graph experiment
independently confirms that removing host gaps has little payoff in this FP64
case.

The native reference fuses pressure, momentum viscosity, continuity and extra
damping into its residual neighbour pass. Its `ResidualHash3` chain has three
wall/divergence passes, one delta/renormalization pass, one surface detection
pass and three fused stage residual passes: roughly **8 neighbourhood passes
per iteration**, compared with the local 60. This is a source-level loop count,
not a native per-kernel GPU profile. Repeated pair geometry/kernel evaluation
and separate field passes are the primary optimization target. The FP64
native reference remains about 8 s/step for the same capped startup workload;
timing inclusion differs and the operators/extra calculations are not identical.

## Precision control and stopping caveat

The FP32 graph uses the same fixed 200 iterations and runs 7.2x faster per
steady iteration. Relative to the FP64 graph's final physical fields:

- pressure RMS 0.02466 Pa / maximum difference 0.1800 Pa;
- velocity RMS magnitude 6.79e-8 m/s / maximum component difference 5.57e-7;
- position RMS magnitude 1.10e-8 m;
- KE 6.50171955e-7 vs 6.50171684e-7 J per unit breadth.

Nevertheless the reported position-row convergence metric is **-2.377** in
FP32 versus **-6.384** in FP64 (target -8, neither control converged).
`bdfSource` forms `alpha*x + beta*x_prev + gamma*x_prev2` using absolute
coordinates O(0.6 m) and coefficients O(1e5 /s). Even setting all three arrays
to the same initial positions, the mathematically zero BDF source has RMS
magnitude **0.002268 m/s**, max component 0.005859, in FP32; FP64 gives RMS
1.68e-11 m/s. This directly demonstrates cancellation. Also, first-step
physical displacements approach/below the FP32 absolute-coordinate ULP.
Stable difference/displacement-based BDF arithmetic and position-increment
storage (or a deliberate mixed-precision path) need qualification before
switching converged runs to FP32. The close first-step fields do not validate
later FP32 evolution or its stopping test.

## Priority

1. **Fuse the stage neighbour residuals**, sharing pair geometry and gradW;
   preserve source choices and compare frozen-state outputs before trajectories.
   Reuse static wall geometry exactly, without freezing evolving fluid geometry.
2. **Qualify displacement-based BDF/position arithmetic**, then reassess FP32 or
   mixed precision using field errors and a meaningful convergence residual.
3. Keep the graph-enabling topology/prepared-velocity contract explicit, but
   expect a small FP64 gain at this particle count. It may matter much more for
   small cases or faster arithmetic.
4. Keep tracking the 200-iteration cap. Reducing the cap to make a runtime number
   look better would change the solve and does not fix this performance issue.

Evidence root: `.tmp/acsph-lobovsky-2026-10-27/`. `performance-summary.json`,
`perf-eager-1/performance/{trace.json,events.json,profiler-cpu.txt,
profiler-cuda.txt,call-audit.txt,timings.json}`, and the two graph runs'
`performance/timings.json`. Retained entry points: `compare_lobovsky.py
--performance profile|graph`, `performance.py`; the FP32 control adds
`--precision float32`. All use the aligned operator flags from the comparison.
The asynchronous one-step runner recordings contained only an initial frame;
complete two-frame initial/final videos were additionally rendered from saved
snapshots with explicit warpSPH/precision labels into each run's root
`output.mp4`. The original recordings are retained.

## Implemented: retained increments, fused residual and production graph capture

User authorized continuation after the spot check. ACSPH now defaults to
`incrementalPositions=True` and `fusedResidual=True`; both are serialized in
its parameter block. Explicit `False` values retain the historical arithmetic
and separate operators. The comparison fixture deliberately sets these from
`--position-form absolute|increment` and `--fused-residual` so changing a
production default cannot silently rewrite earlier controls. Precision itself
is not changed globally. Reference-only wall flux/velocity, damping and RK
choices remain explicit comparison controls.

### Arithmetic and lifecycle

Eq. (42) has alpha+beta+gamma=0, so the position BDF derivative can be written
`alpha * delta_x_current - gamma * delta_x_previous`. Velocity BDF likewise
uses `alpha*(v-v_previous) - gamma*(v_previous-v_previous2)`. The dual-time
position unknown is now the *increment*, while neighbour/bounce operators
still see world coordinates. Both real-step displacement and the low part of
world position are retained on the integration system. Compensated addition
keeps motion below an absolute-coordinate ULP instead of discarding it at the
end of each physical step. This is a change in numerical representation of
the same BDF equations, not a different time discretization.

`initializeNewState` forwards the retained history; finalization copies it,
rolls normal BDF history, and handles shifts consistently. With shift correction
on, a shift changes the coordinate/low part but does not become a physical BDF
displacement. With it off, the shift enters the retained displacement, matching
the existing explicit choice. Update boundary conditions that change dxdt are
reconciled into the retained increment and coordinate; prescribing zero motion
can cancel a sub-ULP displacement. Non-fluid rows remain geometry/closures.

Tests check exact zero-motion derivatives, unequal-step constant-velocity BDF,
sub-ULP motion over repeated real steps, cloned-state continuation, update BCs,
and both shift-correction settings. FP32 now reports -6.875804 at the three-step
endpoint versus FP64 -6.875804, replacing the previous artificial -2.377 floor.
The -8 target is still not reached within 200 iterations: fixing the arithmetic
does not assert convergence where it has not happened.

### Fused pair loop and graph contract

`modules/artificialCompressible/wp_residual.py` combines the continuity
divergence, pressure acceleration and Monaghan-Gingold viscosity pair sums.
The explicit reference damper can also supply its separately computed fluid
divergence so its gradient shares this loop. That prior field is completed
before the fused kernel: no cross-thread dependency is introduced. The sums
remain separate reductions and use the existing kernel normalization,
minimum-image distance, support symmetrization and core operator ABI.

The new kernel covers six pressure-force formulations, 2D/3D, hash traversal
or an adjacency list, variable support/density, ignored ghost references and
ordinary boundary rows. Unilateral or analytic wall residuals retain their
existing operators. Velocity/pressure adjoints agree with the existing path.
The graph-safe prepared-viscosity decision is now made **once per real step**
from that solve's frozen topology, before capture. This replaces the six
per-iteration host checks for stored-ghost/prepared velocity cases and avoids
caching mutable topology across steps. Shared boundary files are untouched.

The first fused FP64 launch exposed an incorrectly untyped viscosity scalar
(expected float64, received float32); it advanced no physical time. The failed
run/test logs are retained. Explicit `scalar_t` conversion fixed it; both
precision suites then passed. This was an argument-type defect, not a failed
physical trajectory.

### Measured results

Same exact FP64 ICs/timestep replay and same fixed **200 iterations per step**;
FP32 explicitly rounds the ICs. Final trials are three physical steps to
4.368e-5 s. All three steps in each final run capture successfully and replay
199 times, without the historical global boundary-function benchmark patch.

| Aligned comparison | Total wall s | Wall ms/step | Integrator step ms | Steady graph iteration ms |
|---|---:|---:|---|---:|
| Prior audited FP64 | 163.007 | 54335.8 | 54368 / 52837 / 52116 | prior one-step graph 263.17 |
| Final optimized FP64 | 104.310 | 34770.1 | 34395 / 33272 / 32931 | 165.50 |
| Final optimized FP32 | 19.992 | 6663.9 | 5663 / 5347 / 5437 | 26.55 |

Final dual-time GPU spans are 33991/33128/32786 ms in FP64 and
5408/5319/5411 ms in FP32. End-to-end three-step cost improves 1.56x at the
same FP64 precision and 8.15x with the qualified bounded FP32 control. Warmed
FP32 solve cost is roughly 9.8x lower than the prior FP64 graph solve. These
are bounded startup measurements, not a long-run precision qualification.

The isolated first-step BDF change has pressure RMS difference 5.10e-11 Pa /
velocity RMS 2.28e-16 m/s versus the old arithmetic. Adding fusion changes
pressure by another 9.30e-14 Pa RMS / velocity 3.12e-19 m/s. At three steps,
optimized FP64 differs from the prior aligned FP64 endpoint by pressure RMS
1.49e-10 Pa / max 1.23e-9, velocity RMS 1.23e-15 m/s and position RMS 3.50e-16 m.
Thus the same-precision speedup does not come from a different residual,
iteration budget or IC.

FP32 vs optimized FP64 at three steps: pressure RMS **0.03080 Pa**, maximum
component difference 0.2038 Pa; velocity RMS magnitude **1.96e-7 m/s**;
position RMS magnitude **1.15e-8 m**; KE 8.11342739e-6 vs 8.11344535e-6 J per
unit breadth. FP32 initial mass rounds to 180.00000637 kg vs 180 kg, then
remains constant; that explicit 3.54e-8 relative quantization is not mass
calibration. Reference parity remains the previously documented ~1.08 Pa
pressure RMS; none of the source-only wall/RK choices were promoted here.

The post-fusion profiler records 51 SPH operator launches per iteration rather
than 60. Three profiled iterations take ~505.6 ms of CUDA execution versus
798.9 ms before; stage residual scopes drop ~550 -> 261 ms. The remaining
stage-0 surface/covariance/pressure-gradient work is now a larger fraction.
It is the next fusion target; topology checks/launch suppression alone still
cannot recover the native reference's roughly eight fused neighbour passes.

Validation: **89 distinct FP32 checks** (86-case ACSPH/optimization suite plus
the added 3D and two shift-lifecycle cases), **23 focused FP64 checks**
(15 fused/operator/adjoint cases, six increment/lifecycle cases, two wall-flux
invariants). Production defaults are exercised by the full FP32 suite.
All final videos decode; endpoint fields, mass, clock and graph receipts are
retained. No broad comparison matrix or full impact campaign was started.

Additional evidence: `increment-f64-1`, `fused-increment-f64-1` (retained type
failure), `fused-increment-f64-1-v2`, `fused-increment-f32-3`,
`optimized-final-f64-3`, `optimized-final-f32-3`, `optimized-summary.json` and
`optimized-*-f32/f64.log` below the same evidence root. The isolated
increment FP64 run costs 56.193 s / 56192.6 ms per step; the fused profiled
FP64 run costs 40.873 s / 40873.2 ms per step (includes first compilation);
the intermediate three-step FP32 run costs 22.572 s / 7524.0 ms per step.
Their warmup/profile inclusion is different from the final graph trials.


## Paper cost expectation (user lookup, 2026-10-27)

De Courcy et al., CMAME 420 (2024) 116700, DOI 10.1016/j.cma.2023.116700,
compares against delta-ALE-SPH with RK4/frozen diffusion (Sec. 3.4, Eq. 64).
It claims comparable solution cost overall, not a universal speedup. All paper
runs use a single P100 in FP64 (Sec. 4). Effective cost Eq. 65 counts accumulated
pseudo-iterations times RK stages across the actual physical steps; it is
separate from wall time and from physical timestep enlargement.

For the matched 2D Lobovsky case (Sec. 4.5, H/dx=400, RK3, CFL_t=.4,
dt/dtau=2, epsilon_v=-8), Fig. 34(a), p.36, ends at approximately 85 min AC-2L
versus 75 min delta-SPH: roughly .9x speedup / 1.1–1.2x cost. These are visual
estimates from the graph, not tabulated values. AC-2L is cheaper over much of the
trajectory, but impact-related solve work erodes that advantage. The corresponding
3D curves end near 450 versus 625 min, about 1.4x speedup; its H/dx=75 and
larger artificial-viscosity dissipation prevent treating that as a same-resolution
2D/3D performance comparison.

Table 1 (oscillating droplet, RK3) explicitly lists wall-cost ratios AC/delta:
1.12 at CFL_t=.4, dt/dtau=2, versus .64 at CFL_t=.6, dt/dtau=2. The latter
is 1.56x faster but has larger kinetic-energy/shape errors than delta-SPH
(1.24/1.21 relative error versus .53/.51 at CFL_t=.4). The jet-impact case
(Sec.4.4/Fig.26) finishes within 1.5x delta cost. The stated value proposition
is incompressible, quieter pressures at similar total cost. Our local frequent
200-iteration caps remain a concern; a bigger real timestep alone cannot establish
an algorithmic speedup. Native campaign wall time was partially contended, so
it is not a clean hardware comparison with the paper's P100 timings.
