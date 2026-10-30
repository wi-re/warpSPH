# Corrected Lobovsky and driven turbulence overnight queue — 2026-10-27

User authorized restarting the corrected ACSPH comparison and chaining the
pending driven subsonic checks from the MFM/MFV multiphysics worktree afterwards.
This supersedes the earlier prohibition on automatically restarting the stopped
local replay. Failed fixture evidence remains untouched.

Queue root: [retained campaign](../../.tmp/overnight-acsph-driven-2026-10-27/).
The source-pinned [manifest](../../.tmp/overnight-acsph-driven-2026-10-27/manifest.json)
is the complete ordered command/input record. The atomic
[status](../../.tmp/overnight-acsph-driven-2026-10-27/status.json) records current
phase, child PID, outcomes, elapsed times and available ms/step. Separate logs,
source hashes, original repository heads/patches and test-port receipts accompany it.

## Ordered work

**User correction:** execute the originally described groups **1 → 3 → 2**:
ACSPH first, all fourteen turbulence comparisons next, then GPU source/restart
checks and timing controls. The current ACSPH process is unchanged. Its
supervisor was replaced with an observer that adopts the same live child;
`status-before-reorder.json` and `manifest-original-1-2-3.json` preserve the
previous queue. No physical runtime/source/input was changed by this reorder.
The later GPU checks now qualify the already recorded cohort retrospectively;
a failed check must be reflected in interpretation, not silently treated as a
passing method. Per-RHS/accepted-state guards remain active during every run.

1. Corrected GPU FP32 ACSPH, exact supplied Lobovsky physical ICs rounded to
   FP32, source timestep replay, retained position increments/fused residuals,
   production CUDA graphs, original hard bounce and declared reference controls.
   The tuple SDF and flat-wall geometry preflight are included. Reuse the
   completed native FP64 trajectory: 10,000 steps to **1.9184199183159294 s**.
   Its earlier step guard prevents claiming a full 2 s reference. Export the
   new trajectory for Remote and run the pressure/KE/spatial comparison.
2. Matched weak-driving cohort: **MFM, MFV, CRKSPH, CompSPH, Monaghan, GSPH,
   InutsukaGSPH**, each at **64² and 128²**, seed **1234**, from rest to **t=40**.
   Coarse runs precede fine runs; a fine run requires its own successful coarse
   run. Repeating MFM establishes a current-source comparator. Each successful
   trajectory receives offline conservative-kernel observations on16²/32² grids
   and the existing t=10–20–30–40 stationarity blocks before the next simulation.
3. Deferred GPU source checks: MFM/MFV cold raw-stage positivity, analytic
   force/heating, knot anticipation and native restart; CRKSPH, CompSPH,
   Monaghan, GSPH and InutsukaGSPH analytic RK2/RK4 source, cold stage,
   checkpoint/export and applicable compatible replay controls. Frozen copies
   of the existing tests change exactly one shared device literal per module
   from CPU to CUDA; assertions/tolerances remain unchanged.
   Then run MFM/MFV timing controls: seed-1234 weak64² saved state near t=25,
   native SSPRK, skin1 versus opt-in1.25, three alternating repetitions,
   20 warmup +80 measured steps. These are local workload comparisons, not
   mature MFV flow. New long trajectories retain the native default skin=1.

## Fixed physical and numerical policies

Periodic 2D unit square, gamma=5/3, rho0=1, initial sound speed=1, u0=0.9,
B7/n_h=4; OU acceleration input=0.3, correlation time=0.5, knot interval=0.05,
modes 1≤|k|≤2, solenoidal acceleration, actual mass-mean acceleration removal;
relaxation bath time=0.2. No Mach feedback, drag or initial mass calibration.
This reproduces the existing weak/subsonic family; amplitude is not a Mach
constraint. Earlier MFM weak runs reached mean RMS Mach roughly 0.55–0.76.

MFM/MFV use native SSPRK and CFL 0.1; the SPH schemes use native midpoint and
CFL 0.15 with their documented support and dissipation policies. These native
criteria are not matched temporal errors. No numerical defaults are promoted.
Source tests and longer runs retain the distinction between compatible-energy
budgets and ordinary RK truncation error. The existing CPU refinement controls
remain relevant. One common seed plus two resolutions cannot establish ensemble
convergence or stationary training quality.

The second runtime is pinned from
`/home/lu26029/.codex/worktrees/meshless-boundary-balance/warpSPH`, commit
`8832040`, plus the exact current Core/Plotting/Integrators Python sources.
Its physical forcing/thermal source implementation is separate from main's
ACSPH runtime. Production MHD remains unimplemented; these are hydrodynamic runs.
See that worktree's `docs/driven_turbulence_scheme_support_2026-10-27.md`,
`docs/driven_turbulence_qualification_2026-10-27.md` and
`docs/driven_turbulence_performance_2026-10-27.md` for the pending gates.

## Execution and failure policy

The detached supervisor survives closing this chat. GPU jobs are serial and
wait for other compute users; unrelated jobs are never stopped. Each job has a
wall limit and output watchdog; physical runs additionally retain runner
progress, velocity/finite/stall guards and live software VisPy recordings.
Every driven RHS is checked for raw positivity/finiteness; every accepted
state contributes diagnostics, budgets and saved observations. Tiny source tests
and timing A/Bs omit video explicitly. Full trajectories have video enabled.

Failures preserve evidence, block dependent runs, and allow unrelated methods
to continue. There is no automatic tolerance/CFL/forcing retry. A successful
process exit is additionally checked against its physical endpoint and result
status. An ACSPH failure does not suppress the independent turbulence study;
its export is attempted but full comparison requires a successful replay.
This initial launch record is not a result or completion guarantee.

Initial launch inspection: all four flat-wall geometry audits pass exactly.
The former failure at step56 converges in13 iterations, epsilon=-8.076293,
with a captured graph (previous bad fixture capped200). The first inspected
post-startup window has zero cap hits and median solve cost about326 ms/step.
Startup compilation and the expected early capped solves are included in the
retained launch wall record; this is not a full-run throughput measurement.
All55 queued GPU source controls pass collection in the frozen environment;
GPU execution remains queued. See `launch-health.json` and the collection log
under the campaign root.

## Completion review — 2026-10-28

The requested1→3→2 queue finished at05:30:57 London time. ACSPH reached
10,000 steps/t=1.918419918 s, exported193 physical frames and completed the
comparison:19,002.7 s wall (5 h17 min),1,900.3 ms/step. The geometry repair
does not resolve all later convergence:2,748 steps capped200 versus142 in the
native reference. The first post-startup cap is step468. Endpoint KE is
52.91 versus45.26 J/m; maximum pressure10.63 versus3.97 kPa. These differences
and the convergence gap remain open; finishing is not numerical qualification.

All14 turbulence trajectories reached40, all14 analyses completed, and every
one of1,108,534 observed RHS states was valid/positive across554,267 steps.
Recorded integration/observation costs span8.22–11.58 ms/step, totaling5,304.3 s
(~88.4 min), excluding each independent warmup/setup. Native method timestep
criteria differ, so ms/step alone does not rank physical-time efficiency.
All12 MFM/MFV timing windows completed. Stationary/ensemble quality is not
established by endpoint completion.

The GPU source cohort initially had50 passing controls and five export-test
failures: a CPU HDF5 tensor was compared with a CUDA expected tensor. A separate
postflight test copy transfers only the expected field to CPU, retaining exact
zero tolerances and the original frozen tests/results. All five export controls
then pass (6.01 s wall). Thus all55 controls have passing evidence; the original
supervisor status remains `complete with failures` to preserve that history.
`completion-review.json` reconciles it with the postflight receipt/XML.
The GPU is idle following this review; no new long runs were started.
