# Extended Lobovsky comparison — 2026-10-27

**Stopped / invalid local wall geometry:** the local harness returned a bare SDF
vector instead of `(values, normals)`, corrupting ghost-derived bounce normals.
The native trajectory remains valid reference evidence. The stopped local
trajectory must not be used to rank ACSPH against it. See OPEN_PROBLEMS §36
and `.tmp/acsph-convergence-gap-2026-10-27/`. The harness contract is corrected;
only bounded prefixes are being qualified, with no automatic long rerun.

User requested a trajectory far into the dynamics against dynGPUSPH, with a
reference recording loadable by warpSPHRemote. The campaign targets the supplied
**2 s** endpoint at the original resolution: 320000 fluid + 32176 physical wall
particles, unchanged mass, initial hydrostatic pressure, viscosity, tolerances
and 200-iteration cap. This extends the bounded
[startup comparison](LOBOVSKY_COMPARISON_2026-10-27.md) and
[optimization qualification](PERFORMANCE_2026-10-27.md).

## Campaign and status

Evidence root: `.tmp/acsph-lobovsky-long-2026-10-27/` (KEEP).
At launch the GPU was occupied by a separate qualification run. The persistent
supervisor initially waited for compute users to exit. The user then explicitly
authorized sharing the GPU: the same frozen campaign was resumed immediately
with `--resume-waiting --allow-concurrent-gpu`, without stopping another task.
Native/local phases remain serial; costs are contended measurements.
Current state is in `status.json`; `supervisor.pid` identifies the supervisor.
The adjacent `.tmp/acsph-lobovsky-long-2026-10-27.log` logs phase transitions.
Queued work is not a completed 2 s validation: check the recorded endpoint and
exit statuses before interpreting results.

Phases run serially:

1. `native-f64`: qualified `dynGPUSPH@2dbc27a` binary, original preset, explicit
   terminal time 2 s, up to 10000 real steps. Retain original 0.01 s output
   cadence, dense `probe.plt`, per-step progress/timings and automatic VisPy video.
2. Export `dynGPUSPH-lobovsky.h5`, even if the native phase fails, preserving
   the available physical snapshots. A failed native run stops the full campaign.
3. `warpsph-f32-aligned`: replay every native real timestep with optimized
   FP32, retained position increments and fused stage residual. The exact supplied
   FP64 arrays are explicitly rounded, with an IC audit. Mapped support constants,
   source BDF history, wall diffusion/velocity, acoustic damper and source RK
   placement remain **explicit diagnostic controls**, not promoted model defaults.
4. Export `warpSPH-lobovsky.h5`; build `analysis/{summary.json,report.md,comparison.png}`.

The Python runtime and harness are copied before queueing, including Core,
Integrators and Plotting. `source-sha256.json`, working-tree status/patch and the
native build/input manifests preserve provenance across concurrent development.
Shared boundary source is untouched. The existing native binary/input hashes
are checked by the reference runner before execution.

Each GPU phase has a 12 h wall bound and a 300 s progress watchdog. Native and
local velocity stops use 100 m/s, far beyond the expected flow scale. Failures,
logs and available snapshots are retained. Local snapshots follow exact native
output-step indices, including the endpoint; live VisPy video samples every ten
real steps and draws every step during a velocity alarm. Per-step graph receipts
retain scalar counters/timings, releasing graph objects rather than accumulating
the startup profiler's entire iteration history.

Launch a fresh campaign (the output root must not exist):

```bash
python -u scripts/acsph_reference/campaign_lobovsky.py \
  --root .tmp/NEW-LOBOVSKY-CAMPAIGN \
  --binary .tmp/dyngpusph-reference-2026-10-27/qualified/build/gpuSph2d \
  --end-time 2 --phase-wall-limit 43200
```

The actual retained campaign was launched with `setsid nohup` so it survives a desktop/session restart;
its two simulations still run sequentially in the supervised process chain.

## Viewer recording and comparison semantics

`export_viewer.py` writes the viewer's documented chunked HDF5 layout:
`frames/{positions,velocities,pressures,densities}`, `time`, fixed
`particles/{UIDs,kinds,masses,supports}`, explicit tank/domain metadata, fluid
diagnostics and pressure probes. Standard gzip/shuffle chunks span **one frame**
and at most 8192 particles. Coordinates and numerical fields remain FP64 in the
container; FP32 local values are losslessly promoted, without claiming additional
simulation precision. Native physical density is constant rho, not an inferred
kernel sampling density. Do not use the source's known biased `VOL` diagnostic
as physical mass or volume.

Physical row identity is retained by both unsorted output writers; no internal
ghosts are exported. UID is that stable output row. Membership and mass changes
are rejected. Identical terminal/scheduled frames with duplicate physical time
are removed; distinct physical states at the same clock are rejected. Sidecars
record each raw snapshot hash and the finished HDF5 checksum. A `.partial` file
is renamed only after successful export; open the finished `.h5` in the viewer.

The analysis compares KE, peak speed, 99th-percentile front/height, physical
pressure probes, material-row errors, and velocity on common 1 cm cells with
at least four particles from each run. Chaotic material-row separation alone
does not establish a macroscopic flow error. Common pressure probes use
fluid-only normalized Wendland C2 at (1.61, 0.003/0.015/0.030/0.080) m on each
stored **post-shift** physical frame. Dry probes are zero with an explicit wet
mask. Raw native `probe.plt` samples an earlier RK field and is retained separately;
its final one-dt-ahead diagnostic clock must not be confused with the physical
snapshot clock. Comparison plots use common physical snapshots at ~0.01 s, so
their peaks can miss features narrower than that output cadence.

The generated report includes wall seconds, wall ms/step and cap counts.
FP32 versus FP64 and unconverged/capped steps remain explicit limitations;
close startup agreement does not pre-validate the extended trajectory.

## Export qualification

Three CPU checks pass: lossless field/identity/time export with duplicate-final
handling; rejection of distinct same-clock states; and exact reference-output
clock/real-step mapping. Exporting existing native and optimized local startup
snapshots reproduces the prior endpoint pressure RMS **1.083535 Pa**.
The actual `warpSPHRemote` Python reader loads all 352176 physical rows, pressure,
velocity, masses/UIDs, physical time and KE from the exported native startup
recording. This verifies the file adapter, not a completed 2 s trajectory.

Retained startup-only format checks use `format-*` / `viewer-format-check*` names
under the campaign root and must not be mistaken for its final recordings.

## Live safety-guard correction

The 10000-real-step bound was too low for the smaller late-splash timesteps:
at t≈1.85 s the native run had already exceeded 9400 steps. The live solver
is left untouched. `finish_lobovsky.py --root <campaign>` observes the existing
supervisor and, only if it ends successfully at that step guard, continues the
local replay/export/analysis at the actual saved reference endpoint. It does
not recover numerical failures or claim a complete 2 s run. The original short-
endpoint status and corrected duration remain in receipts and the final report.
Only downstream reporting was updated; frozen numerical code is unchanged.
