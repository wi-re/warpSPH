# Michel transport and BDF trial — 2026-10-30

User-approved sequence, separate from the dynGPUSPH compatibility ladder:

| Row | Static geometry/direct fields | First-two half-clock warmup | Shift | Full p/v remap | BDF shift correction |
| --- | --- | --- | --- | --- | --- |
|00-michel-baseline|on|on|normal Michel|off|off|
|01-michel-transport|on|on|normal Michel|on|off|
|02-michel-bdf|on|on|normal Michel|on|on|

The generic BDF correction was already on in the previous retained Michel
trajectory. It is explicitly off in rows00/01 to isolate inclusion in row02.
No new dynGPUSPH shift, cohesion/collision, diagonal-only remap, spatial map,
last-RK stopping metric, pressure stopping gate or plateau/trim is introduced.

The baseline retains the existing comparison fixture's consistent wall
closure, source-aligned wall diffusion, acoustic damper, RK coefficients,
startup history, volume ghosts and smooth bounce0.5. Warmup is an explicitly
retained code-inspired choice, not a published requirement. These are scoped
controls around that baseline, not a claim to reproduce every paper choice.

## Physical support and endpoints

Every trajectory starts from the original supplied Lobovsky ICs:320000 fluid,
32176 physical wall particles, dx0.75mm, support3mm, rho1000, fluid mass180kg/m,
nu0.0001608375, FP32, original native accepted timestep sequence. EpsilonV=-8,
max200, accepted-state velocity metric and the same physical/stall watchdogs.
The physical endpoint is7641 steps/t1.530014125s, including the first impact
and complete1.44–1.53s late ringing window. All new runs have VisPy video,
progress, per-step solve timing and complete late checkpoints.

## Transport definition

De Courcy §3.2 Eq58: phi'=phi+grad(phi) dot dx. Reuse the shared SPH
`warpOperation` difference-gradient operator for pressure and the full
component-by-coordinate velocity tensor. Evaluate both gradients at the
accepted unshifted state, using physical m/rho volume weights and a validated
neighborhood; apply only to fluid outputs, then move coordinates. Density,
mass, boundary fields and Michel displacement calculation stay unchanged.
There are no added pressure clamps or source remap safeguards.

The normal delta-SPH path already supports divergence-form density/velocity
transport. Its `Summation` product divergence and `Difference` shift divergence
are not discretely identical to the direct Taylor gradient on an incomplete
neighborhood. This trial uses Eq58 directly rather than assuming a discrete
product rule, and leaves delta-SPH behavior unchanged. Gradient renormalization
is not added in this control. A full-tensor remap is distinct from the supplied
dynGPUSPH diagonal-only velocity formula.

The third row uses the existing generic external-shift BDF correction: retain
physical incremental displacement separately from the artificial shift and
relabel the stored position history. It does not copy dynGPUSPH's duplicated
previous-shift bookkeeping. Existing sub-ULP/BDF tests cover inclusion and
exclusion of that artificial displacement.

## Qualification and execution

Twenty-one focused CPU tests pass: analytic pair pressure/cross-component
velocity transport, constant preservation, fluid-only application, unchanged
input state, linear-EOS equivalence, row deltas, BDF/compensated history and
the previous ablation contracts. CPU kernel caches are redirected to/tmp.

A frozen runtime includes frontend/Core/Integrators/Plotting dependencies and
hashes every file. Before long runs, serial3-step GPU checks must establish:

- With warmup off and BDF on, the new plumbing reproduces retained stage02's
  snapshot bitwise; the newly added helper is inactive in this oracle.
- All three physical IC snapshots are bitwise identical to the retained IC.
- All graphed preflights actually capture and replay, with no disabled graph.
- Transport and BDF trajectories match their eager3-step snapshots bitwise.

The supervisor waits for the existing campaign to finish and for an idle GPU,
then performs preflights and the three new runs serially. Unknown errors stop
the queue; numerical/budget failures remain explicit results with accepted
prefixes retained. No safety threshold is waived. A STOP file cancels only
this supervisor's owned descendants. A persistent supervisor is needed so the
queue survives the chat turn.

Root: `.tmp/acsph-michel-transport-2026-10-30/`; live `status.json`, immutable
`manifest.json`, frozen `source/`, `controller.py`, preflight `qualification/`
and per-row `analysis-through-*/{summary.json,report.md,adjacent-pressure.png}`.

Compare raw pressure swings and1ms/5ms detrended RMS at P1–P4, first-impact
and late windows, KE, mean iterations/caps, completed support, failures and
solve/runtime cost. Pressure smoothness or code-reference agreement alone
is insufficient for promotion. Retain all raw outcomes before choosing a
candidate. No long-run result is claimed at dispatch.

## Integration hold — 2026-10-30

The preceding compatibility campaign is complete (8 successful intermediate
rows and one retained numerical guard stop). This Michel campaign's first
oracle failed during configuration-receipt serialization, before physical
state initialization or timestepping: `PressureSmoothingScheme` enum passed
to raw JSON serialization. The working driver now uses the supported ACSPH
configuration serialization schema; a CPU regression checks the actual enum
and BDF flag. This is a receipt-only repair, not a numerical change.

At the user's request the queue remains held for a clean dev checkpoint,
MFM/core integration and subsequent branching. `STOP` and `merge-hold.json`
record the hold in the failed root. Do not mutate/restart that frozen evidence
in place. Prepare a fresh root from the repaired, committed driver after the
integration, qualify it and retain explicit runtime dependency hashes before
starting the three long trajectories. The failed physical-step count is zero.

Checkpoint validation:142 ACSPH-focused CPU tests pass (including the receipt
serialization regression),51 reference scripts compile, and diff whitespace
checks pass. No new GPU run started during the integration hold.

## Post-integration resume — 2026-10-30

User lifted the hold after frontend8b58008 (integration mergeb9df6a6) and
Core/main04e126e; Integrators1fea2a3 and Plotting05caeb3 are clean. The failed
pre-merge root and STOP remain unchanged. New campaign root:
`.tmp/acsph-michel-transport-integrated-2026-10-30/`.

Preparation now freezes the current full frontend and all three resolved
runtime dependency trees rather than cloning the old pre-integration runtime.
The manifest pins969 files, repository revisions/status, original native
inputs and repaired controller. Each child asserts and records that all four
package imports resolve inside this copied runtime. Development in separate
chunk/multi-GPU worktrees cannot change the running equations or kernels.

The22 focused CPU checks pass in7.21s against the integrated frontend/Core.
A persistent supervisor starts the serial GPU qualification then the original
three long controls, with the same physical settings/endpoints/watchdogs.
First oracle: warmup off, BDF on, compare the retained accepted-fields3-step
snapshot bitwise. Remap/BDF eager and graph controls follow; actual graph
capture/replay receipts are required. No completed long result is claimed
by this resume entry.

### Qualification outcome and physical dispatch

All six serial3-step GPU gates pass. The integrated oracle matches its retained
snapshot bitwise, all physical ICs match, each graphed control actually captures
and replays, and transport/BDF eager snapshots are bitwise equal. Import receipts
resolve all four packages inside the frozen source tree. Aggregate preflight
wall155.02s includes startup/video/output and is not a step benchmark.

The original-IC Michel baseline has started. The unchanged supervisor then runs
the full Taylor transport and BDF-corrected candidates serially, with per-row
analysis, progress, videos, checkpoints and explicit failure receipts. Long-run
outcomes remain pending. Current state is in the integrated root's status.json.
