# warpSPH — Artificial-Compressibility SPH (ACSPH) Implementation Plan

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

**Priority checkpoint — 2026-10-27 (user):** merge the current `dev` path
into `main`, then make ACSPH the next target. Before choosing the next
implementation change, reconcile the current numerical baseline, retained
matrix failures and the remaining work in [ACSPH_REFACTOR_PLAN.md](ACSPH_REFACTOR_PLAN.md).
The older TODO list below is historical and needs that reconciliation; this
priority decision does not start new runs or change numerical defaults.

## Lobovsky comparison — 2026-10-27 (active)


**Daemon-requested viewer video — scoped (2026-10-30, user):** user proposes
an external recording intent relayed through the daemon to its connected viewer,
including requested scene settings. Read-only inspection of warpSPHRemote confirms
existing synchronized producer capture, per-step cadence, viewer ACK gating,
pressure presets and browser WebCodecs encoder. Missing pieces are an externally
requestable recording job, viewer consumption/status of that job, and returning
the encoded artifact to daemon storage rather than only a browser download.
Proposed job binds run ID plus preset/range/camera, capture cadence/window and
encoding settings; viewer acknowledges capability/readiness before synchronized
capture activates. Track queued/preparing/recording/saving/completed/failed or
cancelled; expose recorded start/end times and missed requested start explicitly.
Completed video and scene/time provenance should be retrievable by job ID. Reuse
the existing attached viewer and encoder, preserving lease ownership and explicit
capture-hold recovery. No viewer means visibly queued, not active capture. This
is a design scope, not an implemented endpoint or started video; no companion
code, daemon process or current simulation changed.

**Shifting consistency and event video (2026-10-30, user):** disorder may
couple pressure waves back into particle motion if artificial displacement is
not accompanied by field transport and consistent time history. This is the
purpose of the existing three controls: Michel only; full pressure/velocity
Taylor remap; remap plus external-shift BDF correction. Code audit confirms
Taylor transport evaluates both difference gradients before displacement and
uses the full velocity tensor. Generic BDF correction excludes artificial
shift from the precise physical position increment and relabels position
history. It is a position/time-history correction, not a separate BDF derivative
of pressure: the pressure pseudo-time residual uses velocity divergence plus
pressure diffusion. Pressure responds through the coupled velocity/geometry
solve. Do not equate this plumbing check with validated cavity stability.

Use the daemon viewer's normal **Pressure · symmetric log** preset and
synchronized high-quality video at every accepted step for event localization;
keep mapping/range/camera consistent between controls and retain time/colorbar.
User points out that compressed video permits much denser observations than raw
snapshots. Keep sparse raw states and dense scalar probes for quantitative checks.
Video rendering/ACK can affect wall time, so label capture cost separately. The
current daemon supports synchronized step video; read-only inspection found
video inactive. Encoding/export is performed by the connected browser, which is
not exposed among this chat's controllable tabs. Do not activate daemon video
without an encoder/ACK consumer or take over the user's existing viewer. No
recording was started by this analysis and the running solver was not changed.

**Cavity-window data check (2026-10-30):** current warpSPH Michel baseline,
1.340249–1.399848s:255 accepted steps, zero200-iteration caps, epsilonV
−8.334105…−8.000004, P4 minimum−5.091kPa and global particle minimum−21.936kPa.
Thus the accepted scalar velocity gate passes during these oscillations; this
is not a pressure-residual certificate. Seven spatial states about10ms apart
show7.05% of all fluid and37.83% of lower-right ROI particles below−1kPa at
1.360282s, but the intervening dense sensor troughs are missed. Fixed interior
strip has no particles below−1kPa at these sparse sample times. This does not
establish whole-bulk wave extent or disorder gain, and the screenshot frames
are not precisely aligned. ROIs: lower-right x>1.45,y<0.25; interior strip
0.8<x<1.4,0.02<y<0.25m (coordinate selections, not support-classified bulk).
Retained CPU analysis, JSON and fixed-scale figure under live observations/
`analyze_cavity.py`, `cavity-pressure-summary.json`, `cavity-pressure-window.png`.
Next compare pressure excursions along fixed particle cohorts against support
anisotropy/gradient closure, surface flags and pressure diffusion around each
event; distinguish geometric focusing from numerical amplification. Denser
video can locate those windows, but cannot alone measure numerical gain.

**Cavity closure / disorder amplification (2026-10-30, user):** at about
1.37s, a lower-right empty cavity closes into a distorted particle distribution,
followed by a broad negative-pressure disturbance. User further observes that
pre-existing disorder may amplify an arriving pressure wave. Track two separate
hypotheses: closure as a source, and disorder as an amplifier. Inspect saved
accepted states and dense probes without changing the ongoing controls; compare
pressure extent, sampling/anisotropy and accepted convergence. Spatial snapshots
are only about10ms apart, so cannot establish acoustic propagation or causal gain.
This single-fluid simulation has an empty cavity, with no modeled gas cushion.
Four screenshots retained under the live root observations/cavity-closure-around-
1p37s-{a,b,c,d}.png. Initial offline check recorded above; do not promote a remedy from appearance.

**Lower-wall shear/negative-pressure sequence (2026-10-30, user):** around
1.3s, two shearing flows with a free surface form a larger negative-pressure
patch near the lower right wall. The supplied image pair shows localized
particle disorder and a coherent patch, rather than only thin fluid fingers.
User identifies this as a more unstable event potentially nearer the sensors.
Probe locations are confirmed in export_viewer.py: right wall x=1.61m,
y=0.003,0.015,0.030,0.080m. Screenshots have no coordinate/color legend or exact
per-image times, so precise probe overlap, pressure magnitude and rate of change
need numerical data/fixed color scale. Negative gauge pressure by itself is
not proof of instability; distinguish shear/curvature-driven depression from
numerical amplification, particle ordering/tensile sensitivity and pressure
recovery during local sheet/film closure. Correlate patch position/extent/minimum,
support anisotropy, pseudo-time convergence and P1–P4 on accepted clocks around
1.3s, then check its relationship to the later1.44–1.53s ringing. Compare all
three current controls and retained dynGPUSPH without changing models/settings.
Sequence retained under .tmp/acsph-michel-live-2026-10-30/observations/
`lower-wall-shear-around-1p3s-sequence-{a,b}.png`. No new simulation launched.

**Back-wall return-stream observation (2026-10-30, user):** at approximately
1.2s, the downward stream/film from the wall contact intersects the collapsing
main bulk, producing fine fluid fingers with locally oscillating pressure.
Candidate explanation for part of the sensor ringing: repeated sheet/film
re-impact combined with poorly supported, anisotropic neighborhoods. The image
alone does not establish temporal oscillation, pressure amplitude, causality,
or whether the effect is physical or numerical; no quantitative color scale
or run-variant identity is encoded in the screenshot. Retain the event as a
hypothesis, not a new model choice. In the existing trajectory analysis, correlate
local pressure/relative motion/support quality near the intersection with P1–P4
signals, and distinguish event-forced pulses from ringing that persists after
contact changes. Compare the same event across Michel/remap/BDF controls and
the retained dynGPUSPH reference; do not tune to a single chaotic snapshot.
Image retained under
`.tmp/acsph-michel-live-2026-10-30/observations/back-wall-return-stream-around-1p2s.png`.
No run/setting changed for this observation.

**Native daemon running and verified (2026-10-30):**26 focused CPU checks
and all7 GPU preflights pass, including daemon-on/off bitwise snapshot parity.
Native baseline restarted from original ICs, registered as
`ACSPH Lobovsky 00-michel-baseline`; daemon confirms the existing viewer attached
and frames delivered (no takeover). The next transport/BDF rows register when
they start. Frozen runtime991 files includes the viewer adapter. Atomic status
receipts suspend external watchdog budgets while paused; daemon stop cancels
the campaign. Long fixture runs default to native publication, short fixtures
opt-in; policy in CLAUDE.md. Public run ID/endpoint only in receipts. Previous
prefix preserved; no new model/default/solve threshold changes. Live root:
`.tmp/acsph-michel-live-2026-10-30/`.

**Native daemon restart requested (2026-10-30, user):** register long runs
by default and restart the short elapsed Michel baseline with the native
completed-step publisher. Prior accepted prefix preserved through~0.147s;
owned supervisor cancelled, no solver failure attribution. Add publication
without runner recursion, freeze the adapter, retain public run IDs, and
respect viewer pauses in external watchdogs. Requalify daemon-on versus
daemon-off bitwise before the full original-IC three-row sequence. No numeric
settings changed. Default recorded in CLAUDE.md.

**Post-merge gates passed; Michel baseline running (2026-10-30):**22 CPU
checks and all6 serial3-step GPU preflights pass. Integrated oracle snapshot
matches retained accepted-fields endpoint bitwise; physical ICs identical;
actual graph capture/replay required; full-remap and BDF graph/eager snapshots
bitwise. GPU preflights total155.02s including output, not a throughput
benchmark. Full original-IC00-michel-baseline is now running;01 full p/v
transport and02 BDF correction follow serially to7641/1.530014s. Fresh runtime
pins969 frontend/Core/Integrators/Plotting files, checks all imports stay in
snapshot, and records integrated revisions. Old failed/held root untouched.
Live`.tmp/acsph-michel-transport-integrated-2026-10-30/status.json`.
No long-run result or default promotion yet.

**Resume after integration (2026-10-30, user):** frontend dev8b58008 includes
MFM/turbulence/memory mergeb9df6a6; shared Core/main04e126e integrated, paired
chunk/multi-GPU workspace prepared separately. Resume the authorized three
Michel/remap/BDF trajectories. Prepare a fresh root using the current full
frontend/Core/Integrators/Plotting sources and record their revisions/status,
not the pre-merge frozen dependencies. Keep the failed/held root unchanged.
Re-run focused CPU checks, then exact-IC/plumbing/graph/eager GPU gates before
long serial original-IC runs. Same controls and1.530014s endpoint; no guard or
tolerance relaxation. Concurrent infrastructure work cannot alter this snapshot.

**Checkpoint validation (2026-10-30):** all142 ACSPH-focused CPU tests pass,
including the new enum-receipt regression. All51 reference scripts compile;
`git diff --check` passes. No GPU simulation launched for cleanup.

**Clean integration checkpoint (2026-10-30, user):** hold new Michel trials
until the MFM worktree/core integration and subsequent branching can proceed.
The source ablation campaign completed:8 new rows reached1.530014s, row08
retained its motion-coverage guard stop. Row09 pressure gate has late P1 RMS
4.991kPa/P4 RMS4.390kPa/caps110; full plateau/trim endpoint14.092/12.566/48,
so the final adjacent comparison is now available, pending detailed followup.
Michel preflight failed before any physical timestep because its new config
receipt used raw dataclass enums. Repaired the working driver to use the
existing JSON config schema and added a regression test. Preserve that frozen
failed root unchanged, with STOP/merge-hold receipt; no automatic restart.
Commit ACSPH implementation, tests, maintained analysis/figures and handoff
notes on dev. Raw trajectories/frozen runtimes remain ignored under.tmp.
Re-freeze the repaired driver in a fresh root and requalify after integration;
long runs must stay independent of concurrent frontend/Core edits.

**Michel controls implemented and queued (2026-10-30):** three frozen
original-IC rows to7641/1.530014s: static/direct/warmup Michel with BDF off;
+full pressure/velocity Taylor remap with BDF off; +existing generic BDF
correction. Shared difference-gradient operators preserve constants and full
velocity cross terms; no delta-SPH behavior/default change.21 focused CPU
checks pass, including existing compensated/BDF tests. Serial GPU preflights
(plumbing oracle, exact ICs, actual captured/replayed graphs, eager parity)
precede long rows, all after the current compatibility queue. Same physical
setup/native clocks/FP32/nu/closure/solve thresholds; video/watchdogs/checkpoints
and per-row pressure/KE/convergence/cost analysis. Results pending; explicit
candidate, not default promotion. [Trial](docs/acsph/MICHEL_TRANSPORT_TRIAL_2026-10-30.md),
root`.tmp/acsph-michel-transport-2026-10-30/`.

**Michel transport trial authorized (2026-10-30, user):** baseline = generic
consistent-closure solver + exact static geometry + direct accepted fields +
first-two half-clock warmup + normal Michel displacement. Compare adding full
velocity and pressure field transport, then enabling the external-shift BDF
correction as a third original-IC candidate. Explicitly disable existing
bdfShiftCorrection in the first two rows; same ICs/native clocks/FP32/nu/walls/
solve controls throughout. Retain current source campaign unchanged and queue
new serial trajectories after GPU idle. Verify field transport cross terms,
constant preservation, EOS equivalence and initialization/graph invariants
before long rows. Normal delta-SPH behavior and all defaults stay unchanged.

**Paper/source distinction verified (2026-10-30):** §3.2 explicitly specifies
Michel shifting (Eqs55–57), full Taylor field remapping (Eq58) and discusses
external-shift BDF correction (Eq59). Source collision/cohesion/changed shift
coefficient, diagonal-only velocity remap, warmup, pressure stopping gate and
plateau/trim are not the published prescription. Spatial block is mixed;
last-RK metric sampling is not explicitly specified. Baseline is closer on
shifting but lacks general field remapping and already contains fixture
code-specific alignments: no existing ladder row is a complete paper control.
Stage-level mapping added to the ladder design. Preserve a distinct paper
target; source agreement is compatibility evidence, not adoption evidence.

**Intermediate ablation results (2026-10-30):** seven of nine new rows completed
7641 steps/1.530014s. Static geometry and direct accepted-field handoff leave
P1 late RMS near1.22kPa. First screening flag is source shift/history: RMS
1.219→2.325kPa, swing14.883→43.411kPa, caps6→20 (cap criterion).
Pressure remap→3.159kPa, velocity remap→5.540, warmup→2.772,
coordinated spatial operators→5.300. Row08 last-stage metric stopped after
7224/1.472653s: attempt7225 iteration9 motion2.256215mm exceeds existing
2.250000mm certificate; saved guard exception precedes process signal11.
No full-window score for this failed row. Row09 pressure-gate run still
advancing; plateau/trim attribution remains pending. Retained full endpoint
RMS14.092kPa. Completed new rows58.6–81.7min; solve362ms baseline→546ms
spatial block→660ms retained full endpoint over7641 steps. Coupled shift
needs subdivision/confirmation, not single-term blame or default promotion.
[Intermediate report](docs/acsph/ABLATION_RESULTS_2026-10-30.md).

**Ablation ladder qualified and launched (2026-10-29):**11 cumulative stages,
9 new original-IC trajectories to7641/1.530014s; retained baseline/current
anchors reused.11 focused CPU tests pass.14 serial3-step GPU controls pass:
baseline matches historical diagnostics and independent old frozen-runtime
snapshot bitwise; current endpoint matches retained snapshot/diagnostics
bitwise; all physical ICs equal, graphs active; new warmup/operator graph and
eager endpoints bitwise. Historical schema mismatch was only two missing
nullable diagnostics; repaired with exact recorded-value comparison, failed
receipt retained and all successful controls reused. Frozen numerical source
unchanged. Intermediate rows now serial under persistent supervisor; video,
100m/s/advancing-clock/wall watchdogs, complete late checkpoints, failure input/
trial capture, per-row adjacent pressure/convergence/cost analysis. Unknown
implementation/output errors halt; recognized numerical/budget stops remain
explicit row outcomes. Smoke analysis handles pending rows without false
adjacent claims. No shared-boundary/default change or geometric waiver.
[Design](docs/acsph/ABLATION_LADDER_2026-10-29.md), live
`.tmp/acsph-ablation-2026-10-29/status.json`. Intermediate results are recorded above.



**Cumulative ablation requested (2026-10-29, user):** build a progressive ladder
from the retained consistent-closure/Michel solver to current reference mode,
separating geometry/arithmetic handoff, source shift/history, p remap, v remap,
pseudo-time warmup, coordinated source operator block, metric location,
pressure gate and plateau/trim. Pin all physical ICs/native clocks/FP32/nu/
volume ghosts/ramp0.5. Bound first pass at7641steps/1.530014s, which covers
completed late ringing window; baseline/current endpoint data are reusable.
Preflight graph, endpoint equivalence and individual-switch behavior before
long rows; all original-IC trajectories serial, video/progress/watchdogs,
complete-history checkpoints, per-row automatic same-clock analysis and
explicit numerical failures. No shared boundary/default edit or guard waiver.
A coordinated operator block is an honest first-pass group, not an isolated
single-formula attribution; subdivide if it produces the first notable jump.



**Retained before/after comparison complete (2026-10-29):** all5 local physical
IC arrays bitwise identical;8395 accepted clocks exact against each other and
dense dynGPUSPH. Previous consistent-closure/Michel versus current reference
solver/shift-remap:1.44–1.53s P1 swing22.996→232.297kPa (10.1x),1ms detrended
RMS1.222→14.092kPa (11.5x),caps11→48. Over common8395 steps caps28→84,
iterations15.264→21.098 (+38.2%), recorded solve379.97→712.94ms (+87.6%).
All4 older trajectories completed10000/1.91842s; current guard stop remains.
Additional pressure gate explains12/84 cap endpoints alone;72 also fail
checked velocity. Current late shared-clock meanKE is+0.108% native versus
old+3.794%, and common maximum speed26.07 versus35.06m/s, so do not call this
uniform global velocity instability. Combined reference port clearly raises
ringing/work despite better bulk reference agreement; guard-only fix is separate.
[Report](docs/acsph/HISTORICAL_COMPARISON_2026-10-29.md). Preserve previous
consistent run as baseline, current opt-in reference path without default
promotion. Offline7.11s, plots inspected; no new physical run or model edit.



**Historical trajectory comparison requested (2026-10-29, user):** compare
retained original hard, mass-spacing ramp0.5, volume-spacing mixed-closure,
and volume-spacing consistent-closure full Lobovsky trajectories directly
against the current static reference solver and dense dynGPUSPH. Use exact
common accepted-step clocks through8395/1.637500808s, separate pressure
amplitude, convergence/cost, bulk dynamics and guard-only stops, retain all
configuration/transport differences. Offline analysis first; no new physical
run or changed safety threshold is authorized by this comparison task.



**Second stop captured (2026-10-29):** checkpoint8300 replay reproduces all96
accepted diagnostic rows (initial restart status fields excluded only at the
initial row), then guarded attempt8396 exactly.225.42s diagnostic wall;
trial finite at iteration12, maxspeed16.066m/s, endpointmotion2.250437mm,
pressure range−17.590 to55.803kPa. Pair/bounce budgets still pass; separate
sufficient cone bound fails by0.437micrometers. Saved exact solve input,
trial fields and complete checkpoint8395 in `reproduce-cone-8396/`; further
controls need only one physical-step solve. Native same accepted8395 speed
15.732m/s versus local15.922, so speed itself is not uniquely local evidence.
No physical continuation is active. Next: test actual stage-zero cone/stencil
coverage and source-consistent handling, without waiving a safety bound.



**Partial late comparison recovered (2026-10-29):** exact clocks through
1.637500808s,8395 accepted steps. The fully covered1.44–1.53s window shows
the large pulse/ringing in both dynGPUSPH and warpSPH. P1 raw peak-to-peak
213.127kPa native versus232.297kPa local (+9%);1ms detrended RMS13.264 versus
14.092kPa (+6%), caps45 versus48. Native peak21 contributors. This window
does not support much larger local-only late ringing. Full target/windows
remain incomplete. Plot/metrics in motion report; bounded checkpoint8300
replay is capturing attempted8396 with exact accepted-state parity gates.



**Continuation stopped again (2026-10-29; status follow-up):** 1907 accepted
suffix steps,8395 total, end1.637500807764885s;356 steps remain to8751/1.70015s.
After2861.15s physical wall, attempted8396 hits the separate surface-cone
certificate at pseudo-iteration12:2.25043716mm versus2.25000002mm budget.
This is a sufficient geometry-bound failure, not proof of a missing cone pair
or nonfinite fields. Do not waive it or claim complete late coverage. Latest
complete checkpoint8300 at1.6259203166616585s; original failure/suffix preserved.
No physical run remains active. Recovering partial accepted prefix+suffix
comparison through1.6375s; the1.44–1.53s window is complete,1.25–1.65 and
1.59–1.68 are incomplete. Native dynGPUSPH remains fully complete. Next is
capturing attempted8396 from the checkpoint and checking actual cone/stencil
coverage before changing any bound or restarting.



**Crash diagnosed and guard repaired (2026-10-29):** unchanged replay exactly
reproduces all6489 diagnostic rows and the step6489 rejection. Trial converges
in53 iterations, finite; maxmotion1.530606mm versus old1.500mm guard. Actual
CSRcutoff8.485281mm includes sqrt(2)Verlet margin; all RK stages pass pair,
surface-cone and bounce coverage. Searchsupport2/3/4 controls give maxpressure
difference0.01404Pa. Replace fixed cutoff with all-stage geometric budgets
(limit2.250mm here), preserving equations/candidate list and rejecting unsafe
geometry before persistent surface commit. Old-history restart reproduces
failure bitwise; repaired step accepts the captured trial bitwise.18 focused
CPU checks pass, graph remains active; warm fixed53-iteration solve1.834s old
versus1.825s repaired (no measurable overhead in3 warm pairs). User-authorized
late trajectory resumed from complete step6488 checkpoint, source-pinned,
video/watchdogs/rolling checkpoints, automatic merged late comparison/export.
Early suffix has ringing/caps: crash repair does not resolve late amplitude.
[Report](docs/acsph/MOTION_GUARD_2026-10-29.md), live
`.tmp/acsph-motion-investigation-2026-10-29/continuation-status.json`.
Original failed campaign preserved; dynGPUSPH8751-step reference complete.


**Crash investigation started (2026-10-29, user):** dense dynGPUSPH completed
8751steps/1.70015s; clocks exact, probe/snapshot difference7.28e-12Pa. Separate
postprocessing variable shadowing stopped automatic analysis after both viewer
exports; recover analysis without a physical rerun. Reproduce warpSPH's6489
candidate-motion guard with captured entry, trial fields and complete restart
history; diagnose candidate completeness before altering its safety contract.
Preserve original failed trajectory and qualify any candidate-list repair on
identical inputs. GPU is idle at investigation start.

**Progress / guarded stop (2026-10-29):** repaired warpSPH exited1 after6488
accepted steps, t=1.351960141s,3757.12s wall. Attempted step6489 raises
`Reference solve exceeded the declared half-support candidate motion bound`.
Last accepted maxspeed4.347m/s,29 pseudo-iterations, full-residual convergence,
graphs active; the failing solve's fields/motion were not returned, so this
does not establish whether its trial state is otherwise well behaved. Do not
disable the guard or label the cause solved. Its fixed2R candidate pool and
0.5R motion contract need a targeted reproduction/instrumentation review.
The independent dynGPUSPH reference continues (about4590steps/0.936s at16:24
UTC); queued exports/analysis must retain incomplete warpSPH coverage. No
patched physical rerun has been queued. Late comparison remains incomplete.

**Late comparison authorized (2026-10-29, user):** launch original-IC repaired
warpSPH FP32 and dense accepted-state dynGPUSPH FP64 through1.70s, covering
1.25–1.65s and the two earlier proposed late windows. Freeze the already
qualified solver and observer builds; serialize GPU work, video/progress/
velocity/stall/wall watchdogs, automatic amplitude/decay/dynamics comparison.
No FP64 warpSPH branch or numerical/default changes are added. Preserve the
named bounce/trim/precision differences and any failed trajectory.

**Queue launched (2026-10-29):** persistent serial supervisor PID2683053,
`.tmp/acsph-late-ringing-2026-10-29/status.json`. Repaired FP32 warpSPH is
active; observed FP64 dynGPUSPH and automatic validation/exports/analysis are
queued.8751 steps to1.700150835s each;934 previously qualified source hashes
verified before copying, native observer/source/binary and all six preset
inputs pinned. First14 diagnostics exactly reproduce the qualified short
run; graphs active. Known-sinusoid amplitude and incomplete-coverage checks
pass; analysis smoke correctly rejects0.460s as late coverage. Video,300s
advancing-clock/100m/s/8h-phase watchdogs, periodic GPU occupancy and owned-
descendant cancellation. Approx4–6h, potentially longer with late caps.
No late result yet; no extra precision branch/model change. Root README
documents artifacts and `STOP` cancellation.

**Late-ringing scope clarification (2026-10-29, user):** the concern is the
larger late long-run oscillations (not the first-impact pulse). Dense-reference
agreement through0.460s does not resolve1.25–1.65s. Updated warpSPH has not run
there, dense accepted-state dynGPUSPH data are unavailable there, and cancelled
FP64-window controls provide no result. Keep the late discrepancy open and
limit conclusions to the measured interval. No new physical run launched.

**Dense dynGPUSPH reference complete (2026-10-29):**2113 steps to0.460049714s,
1821.95s /862.26msstep; all clocks exact,48 saved fields bitwise identical to
the original reference, probe/snapshot error1.82e-12Pa. Native FP64 P1 peak
36.53kPa/rebound−3.92kPa versus repaired warpSPH FP32 35.92/−4.55kPa,
at identical clocks. The native trace also spikes and rings; the old10ms
curve missed it. Dense-reference P1 RMSE improves2619.98→522.39Pa (80.06%);
P2–P4 also improve. Peak is first wet sample with just2fluid contributors;
weight/count retained. Original hard/lagged bounce vs smooth/fresh ramp0.5,
precision and trim differences remain named. No causal isolation or default
promotion. Finite fields, mass/membership/static walls and video qualify;
no queued run. [Dense report](docs/acsph/DYNGPUSPH_DENSE_PRESSURE_2026-10-29.md).

**Dense dynGPUSPH reference requested (2026-10-29, user; completed above):** rerun the
supplied Lobovsky IC through the same 2113-step first-pulse window, observing
accepted post-shift pressure every step with the same fluid-only Wendland C2
probes. Preserve the original reference equations and retain build/input/clock
receipts; qualify observation against unchanged snapshots. Resolve the actual
reference peak before drawing conclusions from the sparsely sampled curve.
The hard dynGPUSPH bounce versus smooth warpSPH bounce remains a named control,
not an implied equality of all numerical choices.

**Reference repair / first-pulse replay complete (2026-10-29):** corrected
static-wall path reaches2113 steps /0.460049714s,1194.08s /565.11msstep.
All graphs active;17 startup caps,2086 full-residual exits and10 trimmed exits.
934 frozen hashes verify;47 finite frames, fixed wall coordinates/zero wall
velocity, no wall crossing, constant membership/mass,213 decoded video frames.
Final FP64 fixed-operator/map gates and complete graph tensors pass. At the
matched endpoint KE error improves+0.277%→+0.0136%, mean-pressure error
−3.66%→−0.141%, pressure-field RMS52.9→32.3Pa. P1 spike persists/worsens
32.37→35.92kPa, rebound−1.32→−4.55kPa; shared-frame P1 RMSE187.5→504.7Pa
while P2–P4 improve. Static-wall reconstruction accounts for <0.5% of the
repaired P1 peak, so it is not the primary cause. Smooth bounce/source-angle
quirks remain explicit. Matched solve cost+44%, iterations+14.8%; performance
follow-up remains. Keep named reference mode; no generic-default promotion,
full-duration claim, or queued run. Source/evidence and remaining precision/
dense-reference-probe discriminator: docs/acsph/DYNGPUSPH_COMPARISON.md.

**Static-wall invariant follow-up (2026-10-29):** the first repaired replay
was stopped after2399 accepted steps (~0.497s): routine rigid-body finalization
reconstructs stationary walls and shifts18436rows by up to2.98e-8m in FP32.
Its first pulse still spikes (P1 36.09kPa, rebound-4.59kPa); preserve this
as a stopped control, not a completed qualification. The reference-only
finalizer now keeps wall geometry fixed and rejects moving bodies. New unit
regression passes; three-step graph/eager fields and static wall coordinates
are bitwise exact. A fresh IC replay is bounded at2113 native steps,
t=0.460049714s, sufficient for the first pulse and immediate ringing. Source
root `.tmp/acsph-parity-repair-static-2026-10-29`; shared boundaries unchanged.

**Reference repair qualified; bounded replay running (2026-10-29):** added
explicit `acParams.referenceModel=dyngpusph-static-2d`, with physical/static wall
fields, pair-dependent viscosity, source moment/lambda/gradient, frozen momentum
JST, RK-stage surface union/closure lag, first-two half pseudo-clock, pressure
and position stop metrics, and retained full diagnostics for trimmed/plateau
exits. Frozen source cell membership is filtered against current queries, with
a guarded candidate-motion bound. External remap consumes accepted last-RK wall
p and physical v; reference full-state handoff avoids derivative cancellation.
Smooth bounce0.5 remains the declared comparison control, and source mixed-index
trim bookkeeping is deliberately not copied (stable physical fluid IDs instead).
Captured FP64 force/diffusion operators pass roundoff gates; complete first
pseudo-step x/v/p/wall-p matches at 1.2e-16m / 1.7e-16m/s / 6.1e-11Pa /
2.3e-12Pa. All graph auxiliary outputs match eager bitwise. Three full FP32
physical steps match eager bitwise, graph active. Focused tests36 pass in each
precision plus78 production/fixture checks. FP64 adaptive-dt and isolated-zero
serialization defects repaired/verified. Sources frozen under
`.tmp/acsph-parity-repair-2026-10-29/source`; serial video-enabled FP32 replay
from IC is running only3579 native steps to0.7001s, with watchdogs and analysis.
Original cadence-check and harness/compile failures retained. No shared boundary
edit, precision campaign or full-duration replay.

**Parity repair authorized (2026-10-29):** user says go ahead with the audit
priorities. Implement an explicit static-2D dynGPUSPH reference solve path,
qualify closure/support/JST/RK/stopping on captured fields, and run only the
FP32 first-impact/ring-down prefix after fixed-state and graph gates pass.
Keep ramp0.5 as the declared bounce control; retain hard-cohesion/angle quirks,
with no generic-default promotion or shared-boundary edits. Fix independently
reproduced FP64 timestep/config serialization defects with focused checks.
AC-4 and moving-body model changes remain separate from this AC-2L repair.

**Detailed audit complete (2026-10-29):** fixed controls demonstrate a finite
15 micrometre cohesion jump at the wide cutoff even in FP64, and invalid
antiparallel `acos` inputs explain a separate normal-rounding sensitivity.
The integrated reference-shift caller still supplies zero wall pressure and
mirrored wall velocity, rather than closed pressure and physical velocity;
fixed impact fields change remapped p by up to 403 Pa and v by 0.0081 m/s.
Also confirmed: pair-dependent viscosity wall rule missing, active momentum
JST omitted, first-two pseudo-step warmup and pressure stopping gate absent,
and deficient-support lambda differs (isolated local 1 vs reference 0).
Surface/closure timing, neighbor construction and intentional bounce choices
remain distinct. Production AC-4 scaling, FP64 adaptive-dt failure and lost
isolatedZeroPressure serialization are separately documented. Full source
catalog, controls and priorities: [audit](docs/acsph/DYNGPUSPH_COMPARISON.md#2026-10-29--end-to-end-parity-and-precision-sensitivity-audit).
The previous bounded replay is a partial-alignment experiment: its isolated
operator qualification does not establish integrated parity. Correct these
functional gaps before attributing ringing to precision. No numerical solver,
default or shared-boundary source changed, and no physical run is queued.

**Detailed parity/sensitivity audit authorized (2026-10-29):** user requests
isolate the demonstrated FP32 cutoff/cohesion sensitivity, then re-read the
actual compiled supplied dynGPUSPH and current warpSPH ACSPH paths end-to-end.
Audit both production defaults and the explicit Lobovsky comparison hooks;
classify missing/extra terms, arithmetic/stage/history/guard differences and
intentional controls. Use fixed-input/synthetic non-evolving experiments for
branch attribution, retain failed evidence; no new full physical trajectory or
numerical-default promotion. Record source locations and executable evidence
in docs/acsph/DYNGPUSPH_COMPARISON.md. Shared boundary source remains untouched.

**Shift/remap gap closure authorized (user, 2026-10-29):** close the known
dynGPUSPH/warpSPH external shifting and field-transport implementation gap,
then FP32 from original IC only through first impact/ring-down (~0.70s),
analyze before extending. Source audit includes Michel coefficient/cap/surface
normal/cohesion, wall-collision displacement, p/v remap, pressure safeguards
and shift-history treatment; no claim of complete parity from remap alone.
Check paper Eq58/59 and Michel against the compiled supplied reference; retain
explicit reference-code quirks (componentwise velocity remap, duplicated prior
shift history) in a named comparison mode, not silently in published/default
solver behavior. Require analytic pair oracles and same-state dynGPUSPH shift
replay before evolved run. Shared boundary code stays untouched; video,
progress, serial GPU and watchdogs. No full-duration/FP64 campaign queued.

**Shift port qualification / bounded replay (2026-10-29):** named
`dyngpusph-reference` external-shift mode ports raw pressure and diagonal
velocity remapping, collision displacement, source cohesion/caps, pressure
guards and shift-BDF history. Source-specific 0/1/2 tags, erroneous clamp call
and threefold extended-stencil z wrapping are explicit compatibility behavior,
not paper/default promotions. The fixed-input FP64 shift replay agrees to
1.1e-16m / 4.8e-19m/s / 2.6e-12Pa. The shift's last-stage-zero lambda/color
normal/short-cone inputs are also ported: lambda error1.3e-15, normal max2.9e-7;
31/320000 collinear lattice surface tags differ under the source's strict
abs(cos)<1 roundoff discontinuity. FP32 regressions18 pass, FP64 operator tests9
pass. FP32 one-step with graph/video succeeds,19.77s including startup.
Original-IC FP32 replay is now bounded at3579 native steps (~0.7001s), no full
run, under `.tmp/acsph-shift-remap-2026-10-29/manifest.json` with913 source hashes,
serial supervisor, watchdog, video and per-step remap/solve/probe diagnostics.
First prefix stopped after identifying the first-step graph validation overwriting
the Python surface hook; its evidence is retained in `stopped-stale-first-stage-hook`.
Corrected hook preserves captured surface tensors. Complete one-step shifted
state (x/v/p/rho/volume/kinds) matches eager bitwise; graph9.112s, eager9.029s
including startup/video. Source re-sealed and bounded run restarted from IC.
The old consistent-closure/Michel run is the declared baseline; shared boundary
implementation and all prior solver controls remain unchanged. Analysis pending.

**Bounded shift/remap replay complete (2026-10-29):**3579steps,
t=0.700116863s,1477.06s/412.70msstep, all graphs, only16 startup caps.
P1 peak40.37→32.37kPa and undershoot−4.61→−1.32kPa, but ringing persists:
P1 tail highpassRMS9.73→20.13Pa. KE119.489→119.453J/m, dynGPUSPH119.283J/m.
Shared-frame pressure envelope improvesP1–P3, slightly worsensP4; sparse
reference frames cannot establish its true narrow impact peak. Mass constant,
no physical-wall leakage,71frames/video verified. Additional fixed-input FP32
strict parity gate fails at6micrometres: traced to cutoff membership toggling
the source angular-cohesion veto; FP64 same-input port agrees to roundoff.
Retain failed evidence and reference mode, no generic-default promotion. A
short precision check was proposed here; the subsequent detailed audit above
found further model gaps, so a same-model claim is premature. Full results
in docs/acsph/PRESSURE_SPACING_COMPARISON_2026-10-28.md. No full run queued.

**Precision campaign stopped by user (2026-10-29):** user does not want further
comparisons while dynGPUSPH and current warpSPH have a known model gap. Stopped
the identified serial supervisor and its FP32 checkpoint prefix; verified both
inactive. All remaining precision windows/analysis jobs cancelled, with no
automatic resume. Qualification and partial prefix evidence retained under
`.tmp/acsph-ringing-precision-2026-10-29/user-stop-receipt.json`. No late FP64
window result exists. The relevant gap is explicit: dynGPUSPH remaps pressure
and velocity after shifting, while the current warpSPH Lobovsky configuration
does not; native collision shifting is also absent from that warpSPH setup.
These differences must be addressed/qualified before another cross-code
comparison campaign. No numerical change or replacement physical run started.

**Precision campaign restart gate passed (2026-10-29):** source-pinned wrapper
captures/restores particle state, BDF histories, compensation, adjacency/domain
and all non-callable rigid-body data. Three-step prefix matches retained full
FP32 diagnostics bitwise; its2→3 restart matches every saved particle/history/
rigid-body tensor bitwise and the prefix's physical snapshot. FP64 promotion
preserves captured inputs exactly and completes the same one-step startup
control with CUDA graph active. Videos decode. Qualification costs: prefix
19.25s/6394.64msstep (3steps, capture overhead), FP32 restart9.07s/9006.07msstep,
FP64 restart35.13s/35068.37msstep (1step each, initialization/compile included;
not throughput benchmarks). Queue setup/helper packaging and snapshot-gate
errors were repaired with failed receipts retained; successful physical
qualification runs were reused. Long FP32 prefix to8055 now generates late
checkpoints at6917/t1.440025 and8055/t~1.590s. Paired FP32/FP64 branches then
run6917→7641 (to1.530014s) and8055→8650 (to1.680013s), followed by native/precision
pressure plots and window metrics. Each FP32 branch must reproduce retained
baseline diagnostics and physical snapshots. Serial, video/progress/watchdogs,
same frozen consistent-closure runtime; no solver/shared-boundary change.
Evidence `.tmp/acsph-ringing-precision-2026-10-29/launch-health.json`.

**Late ringing precision controls authorized (user, 2026-10-29):** compare
FP32/FP64 evolutions over the1.47s and1.625s ringing events with unchanged
consistent closure, volume ghosts, ramp0.5, timestep sequence and solver
tolerances. Existing trajectory snapshots lack BDF/compensated-position
history, so generate complete checkpoints by exact source-pinned FP32 prefix
replay and require restart parity before interpreting precision differences.
Restore particle state, histories/position compensation, adjacency/order and
static rigid-body geometry; FP64 promotes the same captured state and cannot
undo prior FP32 error. Candidate mechanisms remain hypotheses: finite-precision
pressure/moment arithmetic, stopping on the position equation while pressure/
momentum remain unconverged, wall-bounce/contact stiffness and shifting without
the native field-remap/collision terms. Record each separately; no model changes
or new defaults during the precision test. Serial GPU with videos/progress and
wall/silence/stall watchdogs.

**Consistent-closure results reviewed (2026-10-29):** full10,000-step replay
reached1.918419918s with no stop/divergence; all10,000 solves graphed,1001 video
frames decoded. Allserial analysis jobs passed. Cost5539.67s (1h32m20s),
553.97msstep versus636.49 prior mixed/volume; caps133→37 (16startup).
The growing0.65–1.0s mode disappears: P1 native shared-frame RMS error over
0.95–1.05s6.068→0.0172kPa, versus0.1458 original orange. Broad first-impact
RMS also improves. Dense first P1 peak remains40.37kPa. Later impact ringing
remains: P4 minimum over1.25–1.65s worsens−7.98→−19.19kPa; P1 late RMS error
4.589→4.776kPa. MeanKE0.4–1.25s+0.32% native, late1.25–endpoint+6.54%;
endpoint58.05 versus45.26J/m (+28.27%, prior volume+15.26%). Mixed closure
strongly implicated in early regression; keep consistent fixture plumbing,
but late dynamics/pressure remain unqualified. No new runs/default changes.
Four-trajectory plot/metrics in
`.tmp/acsph-lobovsky-consistent-closure-2026-10-28/closure-review`;
[review](docs/acsph/PRESSURE_SPACING_COMPARISON_2026-10-28.md).

**Consistent-closure full replay running (2026-10-28):**9 CPU regressions pass,
including accepted-field closure refresh, repeated against the frozen runtime.
The20-step physical preflight passed:90.36s/4518.20msstep,16 known startup caps,
all20 solves graphed,20 video frames decoded and first frame inspected. A
validation-only duplicate-video selection error was corrected, its failed
receipt retained, and the passed physical preflight reused. Full10,000-step
replay now running with source-consistent entry/RK/pre-shift closure, unchanged
volume ghosts/ramp0.5 and all other controls. Source-pinned serial supervisor
survives chat exit; full validation, H5 export, native analysis and pressure
comparisons against previous blue and orange trajectories are queued. Evidence
`.tmp/acsph-lobovsky-consistent-closure-2026-10-28/launch-health.json`.
No shared boundary edits or numerical-default promotion; pressure outcome pending.

**Consistent-closure replay authorized (user, 2026-10-28):** opt-in comparison
fixture must use the source wall-velocity closure at step entry, all RK stages
and freshly from the accepted fluid field before shifting. Retain volume ghost
placement, ramp0.5, original IC/mass and all10,000 native timesteps; no collision
shift or shared boundary edits. Qualify handoff then launch source-pinned serial
replay with video/progress/watchdogs and automatic native/previous-run pressure
comparisons. Previous trajectories remain evidence.

**Mixed-closure sensitivity confirmed (2026-10-28):** source wall-velocity
fixture hook affects only RK residual stages; step entry and therefore
post-solve Michel shifting still receive ghost-sampled wall velocities. Two
same-field/layout audits at0.8s (one per trajectory) change shifting up to
0.232/0.247dx; using source closure reduces this to0.0000795/0.000159dx.
Source-closed spatial operators barely change. Both fixed-field controls
passed,~1s each,0physicalsteps/no video. P1 shared-frame native RMS error
over0.95–1.05s worsens0.146→6.068kPa. Ringing cause remains unproven; separate
normals-only from node placement and make wall closures consistent before
another trajectory comparison. No shared-boundary/model-default change.
[Details](docs/acsph/PRESSURE_SPACING_COMPARISON_2026-10-28.md).

**Pressure regression investigation (user, 2026-10-28):** the old orange
trajectory follows native more closely and lacks the new ringing. Separate the
full ghost-layout change from the already-qualified bounce-normal correction:
the fixture correction changes node depths as well as normals. Audit consumers
and frozen geometry before proposing another trajectory run. Increasing pressure
oscillation is already visible at0.65–0.95s with13–14 iterations and no caps;
the41 capped steps at0.95–1.05s therefore do not explain its onset. Retain both
trajectories; no shared-boundary edits or numerical-default promotion.


**Pressure before/after plots (2026-10-28):** four wall probes plotted at shared
193-frame cadence with native and separately at all10,001 local accepted-step
observations, plus first/returning-impact zooms. First P1 peak51.22→40.31kPa;
broad first-impact curves barely change. Corrected trajectory has new~1s pulse,
later P4 minimum−3.15→−7.98kPa despite positive peak40.45→17.87kPa in1.25–1.65s.
No blanket pressure-noise improvement claim. PNG/SVGs/metrics retained, allthree
figures inspected. [Pressure comparison](docs/acsph/PRESSURE_SPACING_COMPARISON_2026-10-28.md).


**Volume-spacing full replay complete (2026-10-28):** all10,000 steps reached
t1.918419918 s with no stop/divergence, all10,000 solves graphed. Full run
6364.86s (1h46m05s),636.49ms/step; caps214→133
(native142), median13. EndpointKE52.163J/m versus native45.257, excess15.26%
(previous ramp20.51%). H5 export and193-frame native analysis passed; GPU idle.
Pressure/containment and phase-wise physical qualification still require review;
no collision shift or shared boundary code/default change. Evidence
`.tmp/acsph-lobovsky-volume-ghost-2026-10-28` and its completion-receipt.json.



**Corrected-initialization replay launched (2026-10-28):** frozen full10,000-step
Lobovsky job is running on previously idle GPU. Startup audit confirms volume
spacing0.0007500000065m versus legacy mass spacing0.023717083m; hybrid placement
preserved. Allnear-corner flat normals havezero component error (margin6mm),
allFP32 physical IC checks pass,320,000 fluid particles/180kg unchanged. SixCPU
regressions pass (gridsnap/hybrid density rescaling, flat geometry, hook cleanup).
Only two comparison fixture files changed in the frozen914-file runtime; shared
boundary sources, collision shift absence, ramp0.5 and allsolver controls retained.
Supervisor and video/progress/watchdogs/export/native analysis survive chat exit.
[Launch evidence](.tmp/acsph-lobovsky-volume-ghost-2026-10-28/launch-health.json).



**Corrected-initialization full replay authorized (user, 2026-10-28):** rerun
Lobovsky from original ICs through all10,000 native timesteps with ramp0.5 and
volume-derived ghost spacing. Keep native collision shift out of this run to
isolate geometry. Scope correction to opt-in comparison-fixture initialization;
shared boundary implementation remains untouched. Same physical mass/ICs and
all other mapped/source controls, frozen runtime, serial GPU, video/progress,
wall/silence/stall watchdogs; export and compare with native automatically.
Density-unit rescaling/flat-near-corner geometry and hook-restoration tests
are the launch gate. Evidence `.tmp/acsph-lobovsky-volume-ghost-2026-10-28`.



**Remaining-cap cause confirmed (2026-10-28):** exact1727-step smooth prefix
matches all accepted diagnostics/counts; original graph/eager and fresh fixed-solve
reload match allfive outputs bitwise. Ghost initializer takes sqrt(mass), not
sqrt(volume): rho0=1000 gives dx0.0237171 instead of0.00075m (31.62x). This
spreads false corner normals across centimetres; at1727 an almost tangential
(3.33476,−0.000250)m/s velocity becomes−1.31385m/s along the bad normal.
Correcting only captured D5 normals using proper dx makes both1707/1727 converge
200→13. Broadening ramp1 fails1707; this is a geometry unit error, not a reason
to loosen tolerance. Shared boundary code untouched; fix volume-based spacing
with density-rescaling regression before corrected-initialization impact trials.
Collision-only200-step control finishes0.0882s: at common0.08024s wall penetration
1.098→0.454dx versus native0.134; KE−0.00293%, pressureRMS63.45→61.87Pa.
Partial containment improvement, not full shift/late-flow qualification.
Prefix842.83s/488.03msstep; collision164.16s/820.81msstep (startup/instrumentation
included). All serial controls passed, videos174/21frames decoded; GPU idle.
[Full investigation](docs/acsph/RAMP_QUALIFICATION_2026-10-28.md).
No production/shared-boundary/default changes or further full replay.



**Ramp offline qualification (2026-10-28):** three193-frame trajectories show
ramp meanKE within0.5% native through1.35 s; late mean+6.79% versus hard+6.96%.
Endpoint+20.5% does not isolate a new ramp energy defect. Broad first P1 impulse
changes−0.58% hard→ramp but narrow peak36.8→51.2kPa; pressure unqualified.
New wall lead: both local paths penetrate~2.1dx by0.4 s versus native0.134dx;
full max0.296/2.714/3.813dx native/hard/ramp. Compiled native post-solve collision
shift is active and absent locally; its BRPF force is explicitly zeroed. Same-input
collision-term audit and independent pair oracle pass. Exact1727-step smooth
prefix capturing1707/1727 is running; collision-only200-step early diagnostic
queued after it, serial GPU. [Report](docs/acsph/RAMP_QUALIFICATION_2026-10-28.md).
No production/shared-boundary/default changes or new full replay.


**Ramp trajectory investigation started (user, 2026-10-28):** compare all three
retained trajectories at common clocks, separate pre-impact/bulk/late behavior,
pressure cadence, signed wall penetration and remaining cap groups. First smooth
post-startup cap is1707/t0.41194, near first impact; classify from exact prefix
solver entry if needed. Retain source-pinned numerical baseline; no default change.


**Ramp0.5 full replay complete (2026-10-28):** all10,000 steps reached
native endpoint t=1.918419918 s; no divergence/stop, all10,000 solves graphed.
Wall6681.64 s (1h51m22s),668.16 ms/step versus prior hard-bounce ~5h17m;
caps214 versus2748 (198 versus2732 after startup). Median13 iterations;
final100 no caps. Viewer H5 export and193-frame native comparison passed.
GPU idle at completion. Numerical/pressure/penetration qualification remains
open: endpoint KE54.538 J versus native45.257 J (+20.5%); fewer caps do not
establish trajectory agreement. Evidence `.tmp/acsph-lobovsky-ramp05-2026-10-28`.



**Ramp0.5 handoff verified (2026-10-28, 09:13 UTC):** the CRKSPH192/256 runs
and analyses all passed; upstream queue finished09:03:20 UTC. Ramp replay
started09:03:38 with GPU idle, now ~1,250/10,000 steps at t≈0.328 s after9 min.
Recent100 solves use13 iterations, ~332 ms/solve, no caps. Only the ACSPH child
uses GPU. Full-impact behavior and total-cost qualification remain pending.



**Full ramp0.5 replay authorized and queued (2026-10-28):** repeat all10,000
recorded native timesteps (t=1.918419918 s), FP32 increment/fused corrected
reference controls, with only `wallBounceRamp=0.5` changed. Wait for the entire
active CRKSPH192²/256² turbulence queue including pending GPU jobs, then require
GPU idle; source-pinned serial supervisor survives this chat. Runtime914 baseline
files verified; only the fixture CLI changes, no production/shared boundary
edits. Video/progress/physical stall and wall/silence watchdogs enabled; export
viewer H5 and compare native automatically after successful endpoint validation.
Retain hard-bounce comparison and all failures. Cost/cap/pressure/penetration
qualification remains pending; no default promotion. Queue:
`.tmp/acsph-lobovsky-ramp05-2026-10-28/{manifest,status}.json`.


**Additional isolated caps confirmed (2026-10-28):** sampled475/510/606 in one
unchanged606-step prefix; all607 physical diagnostic rows match overnight.
Hard/eager are bitwise and cap200; off/existing smooth0.5 reach tolerance13 at
all three. One leading wall row carries99.94–99.98% of residual. Steps510/606
show period5/3 threshold cycles;475 is a near-threshold residual plateau whose
substage details remain open. Same known D5 family as Marrone; Lobovsky probe
explicitly uses hard0 while normal dam-break/sloshing presets use0.5. Broad
late blocks remain unclassified and remedy trajectory qualification open.
[Evidence](docs/acsph/SPORADIC_CAPS_2026-10-28.md). No defaults changed.


**Sporadic-cap sampling (2026-10-28):** user asks whether other sporadic high
iterations share the step468 cause. One pinned, unchanged physical prefix
through606 samples475/510/606 with same-input hard/off/smooth controls; original
hard result remains accepted. No long impact replay or default change. Early
0.1–0.4 s has14 isolated cap events; later sustained blocks remain separate.


**Step468 cause established (2026-10-28):** exact overnight prefix and fresh
FP32 fixed-solve reload reproduce bitwise. Two flat-bottom particles carry
99.95065% of position residual; the leading distance alternates across0.25 dx.
Hard bounce caps200 in FP32 and promoted FP64; graph/eager are bitwise in both.
Off/existing ramp0.5 reach unchanged position tolerance in13. Neighborhoods
are populated; flat normals correct. Extra cap iterations change fluid pressure
by66.1 Pa RMS vs the smooth control while KE is nearly unchanged. Next: bounded
trajectory qualification of ramp0.5 for pressure/penetration and later-cap
attribution; native's different evolved entry remains a separate question.
No production/shared-boundary/tolerance changes. [Evidence](docs/acsph/STEP468_CONVERGENCE_2026-10-28.md).


**Step468 investigation authorized (user, 2026-10-28):** reproduce the exact
corrected overnight FP32 prefix from original ICs with pinned numerical sources,
capture solver-entry topology/BDF/low-part history, compare graph/eager and
hard/off/smooth bounce, then promote the same input into a fresh FP64 process.
Track per-particle position and physical residuals and bounce threshold distance.
No production default or shared boundary change is authorized by the probe.


**Full comparison (user, 2026-10-28):** original mass/ICs and all10,000 dt
values align;193 matched physical frames analyzed. Mean KE difference is+0.20%
over0.4–1 s and+0.11% over1–1.35 s; local endpoint KE is16.9% higher. Later
pressure pulses differ strongly, with explicit cadence/RK-stage caveats.
First later cap is step468/t=0.163517 s (native15 versus local200), before
opposite-wall impact. Capped steps consume80.46% of local measured solve time;
allsteps remain graphed. Next bounded discriminator is exact solver-entry
history/residual/bounce/precision at468; no new physical run or model change
in this comparison. [Report/figures](docs/acsph/LOBOVSKY_RESULTS_2026-10-28.md).


**Overnight completion reviewed2026-10-28:** corrected replay/export/analysis
finished10,000 steps/t=1.918419918 s in5 h17 min (1,900 ms/step). Later numerical
convergence remains unresolved:2,748 cap hits versus142 native, first later cap
step468; endpoint pressure/KE still differ. All14 chained turbulence runs and
analyses reached40 with zero invalid observed RHS; all12 timing windows finish.
GPU checks have50 original passes plus5 exact export passes after fixing only
the CPU/CUDA assertion mismatch in a separate test copy. Original failures and
all frozen physical sources remain retained. See the overnight completion
record and `completion-review.json`; GPU idle, no new long runs launched.


**Overnight rerun now authorized and launched (user):** corrected GPU FP32
Lobovsky replay from the original ICs, with flat-wall geometry preflight and
unchanged hard bounce/reference controls; reuse the completed native 10,000-step
trajectory to its actual1.918419918 s endpoint. Then chain deferred driven
subsonic GPU checks, MFM/MFV timing controls and seven-scheme64²/128² comparisons
from the meshless multiphysics worktree. Both Python runtimes are pinned; jobs
are serial, watchdogs/videos on, independent failures retained. This new user
instruction supersedes the earlier stop/no-automatic-restart instruction below.
[Manifest, scope and live status](docs/acsph/OVERNIGHT_2026-10-27.md). User subsequently specified
**1 →3 →2**: ACSPH, then all14 turbulence comparisons, then GPU source/restart
checks and timing controls. The live ACSPH child was adopted without restarting
it. GPU checks now assess the recorded cohort afterwards rather than gate its
launch; live physical guards are unchanged.


**Bounded correction result:** corrected236-step prefix to0.099947 s has0/212
post-startup caps (old fixture29/212, native0/212), median13 vs native14.
174.640 s wall /740.0 ms per step including startup/diagnostics; three CPU
geometry regressions pass. Full long rerun remains stopped.
[Investigation](docs/acsph/CONVERGENCE_GAP_2026-10-27.md).

**Concrete fixture bug found:** the harness SDF returned a bare tensor instead
of `(values, normals)`. Ghost construction silently broadcast the first corner
value, producing 45-degree normals on flat walls. One particle dominated the
first cap at step56; graph/eager agree bitwise and bounce controls converge13.
The corrected tuple restores the original hard-bounce solve13 iterations at
that step, with identical physical ICs. Added flat-wall geometry guard and
three CPU regression checks. Full stopped local replay is invalid for ranking;
only a bounded0.1 s prefix is being tested. Shared boundaries/numerical defaults
are untouched; later-time behaviour remains to qualify.

**User stop / investigation priority:** stop the long FP32 replay at t≈0.84 s
and investigate its persistent cap200/epsilon≈-5 versus the native reference as
an actual scheme/implementation gap. Preserve all evidence; no automatic long
rerun. Track OPEN_PROBLEMS §36. First isolate onset and particle residuals;
separate hard-bounce chatter (§33), precision, graph semantics and reference
truncated stopping behavior. Do not attribute the gap to FP32 or bounce without
a controlled replay.

**Paper cost lookup:** Sec.4.5/Fig.34 expects roughly comparable 2D Lobovsky
cost (visual endpoint ~85 min AC versus75 min delta on FP64 P100); 3D ~1.4x
faster. Physical dt enlargement is offset by pseudo-iterations/RK stages
(Eq.65). Table1 shows strong dt/dtau/CFL sensitivity. Frequent local cap200
hits require investigation rather than interpreting AC as inherently faster.
Details in the performance report.

**Live-run guard correction:** native reached t≈1.85 s after >9400 steps;
late splash motion shrinks dt, so the originally chosen 10000-step safety guard
may end it before 2 s. Do not alter/restart the live solver or falsify clocks.
A continuation observer lets the existing native process finish, then resumes
the frozen local replay/export pipeline at its actual saved endpoint only if
that guard stops an otherwise successful reference run. Short duration is
explicit in status/report; actual native simulation failures remain failures.

**Long trajectory authorized (user, 2026-10-27):** extend the exact supplied
Lobovsky fixture to its original 2 s endpoint, then replay the native timestep
sequence with optimized FP32 warpSPH and all explicitly aligned reference
controls. Keep resolution/mass/tolerances/cap fixed, report convergence and
precision limits, and compare KE, pressure probes and spatial fields through
impact. Serialize both physical trajectories as lossless FP64, chunked HDF5
readable by warpSPHRemote; FP32 local values are promoted without claiming
additional precision. Initially queued behind the existing GPU workload; native/local phases run serially.
User subsequently authorized GPU sharing with the ongoing runs: start immediately, keep the native/local phases serial, and label timings as contended. Do not edit shared boundary code or stop other tasks. Campaign receipts:
`.tmp/acsph-lobovsky-long-2026-10-27/`; [campaign/format notes](docs/acsph/LOBOVSKY_TRAJECTORY_2026-10-27.md). The old startup-only gate is superseded
by this explicit authorization; reference wall/RK controls remain diagnostic.

**Performance follow-up (user):** profile the 54.3 s/step aligned fixture,
separating audit I/O, pseudo-iteration GPU work, launches and host syncs.
Check graph capture with an exact fixture-local specialization of the known
viscosity guard; shared boundary code remains untouched.

**Implementation authorized:** add cancellation-safe BDF/retained position
increments, then fuse the principal stage neighbour operators. Keep legacy
comparison controls, validate frozen residuals and FP64 trajectories, and
measure the fixed Lobovsky workload before promoting precision changes.

**Implemented and qualified:** default retained position increments/low parts,
BDF differences and fused stage pair sums; per-step prepared-viscosity decision
permits production graph capture without shared boundary edits. Three-step
FP64 pressure RMS vs prior aligned result 1.49e-10 Pa; wall cost 163.0 ->104.3 s.
FP32 costs20.0 s (6.66 s/step; warm solve5.4 s), pressure RMS vs FP64 .031 Pa,
convergence metric restored to -6.88 rather than -2.38. All still cap200.
89 FP32 /23 focused FP64 checks; update BCs, shifts, sub-ULP accumulated motion,
cloned-state continuation, 3D/hash/list pressure formulations and adjoints
covered. Precision unchanged globally; reference-only wall/RK choices remain
probe controls. Next fusion target: stage-0 covariance/gradient/surface work.
Detailed receipts and limits in the performance report.

**Follow-up requested:** exact ICs do not explain the remaining discrepancy.
Audit the first spatial residual and RK stages directly in both implementations,
then isolate wall closure, pressure diffusion and pseudo-time integration before
extending the trajectory. Record observational build hashes and parity checks.

User: begin the warpSPH comparison with the supplied Lobovsky preset. Import
its exact initial physical particle arrays (320,000 fluid + 32,176 wall),
constant density/mass and hydrostatic pressure; do not resample, renormalize
mass or relax the IC. Use CUDA FP64 for parity with the reference, Wendland C2
support 4 dx, common viscosity, SSPRK3 / CFL_tau / iteration controls, and
replay the reference timestep sequence for the first controlled comparison.
Retain current warpSPH numerical operators and list remaining reference-code
differences explicitly. First inspect a bounded startup comparison before
extending toward the wall impact. Video/progress/watchdogs and per-step costs
are required. Record the fixture audit and field/energy/pressure errors in
`docs/acsph/LOBOVSKY_COMPARISON_2026-10-27.md`.

**Startup comparison complete:** all physical initial arrays bitwise equal,
CUDA FP64 / shared reference dt. A confirmed extra 1/rho in the ACSPH pressure
acceleration was removed (Eq. 25 + actual shared-kernel units; both ordinary
and unilateral paths). Full FP32 ACSPH suite 66 pass; FP64 focused checks 3
pass, including a density-unit-scaling regression that failed before fixing.
At three steps, pressure RMS vs reference falls 508 -> 110 Pa; p_max 9.26 ->
3.22 kPa (reference 2.94 kPa). Probe-only h/coefficient and BDF2-history
mapping gives KE 1.23x reference / pressure RMS 108 Pa, but all solves cap at
200 and residual wall/free-face differences remain. No h/startup defaults
promoted. CUDA graph fallback traced to a host bool in shared boundary
viscosity; that code is untouched. Do not extend to the full impact until
initial operators and the convergence-controlling rows are compared.


**Performance spot check complete:** FP64 eager/graph solves 53.4/53.1 s,
bitwise endpoint parity; stage dumps are not the bottleneck. Over 98% of CUDA
time is neighbour kernels (~60 passes/iteration versus ~8 fused native passes).
FP32 fixed-work graph solves in 7.37 s, steady iterations 7.2x faster, with
pressure RMS difference 0.025 Pa at the first step. Its position stopping metric
is limited by absolute-coordinate BDF cancellation (zero-motion RMS 0.00227
m/s). No production/shared-boundary edits. Next: fuse stage residual kernels,
qualify displacement-based BDF/position storage, then revisit precision and
convergence. [Profile and costs](docs/acsph/PERFORMANCE_2026-10-27.md).

**Follow-up operator audit complete:** on identical native RK fields, local
pressure acceleration agrees to 2.45e-12 m/s², but mDBC changes divergence by
up to 0.3505 s⁻¹. Wall-point Shepard/mirror restores roundoff agreement.
The AC-2L solid-neighbour gradient also produces false 7.84 MPa/s hydrostatic
diffusion; using the fluid-side gradient removes it on flat walls. Successive
wall flux / acoustic damper / source RK placement / wall velocity probes
reduce final three-step pressure RMS 108.4 -> 1.08 Pa, KE difference +0.189%,
peak speed +0.242%. Two FP32 and two FP64 wall-flux invariant checks pass.
Native shifting is separately measured (first shift max 0.008 dx); all solves
still cap at 200. Probe-only options retained, shared boundaries and production
defaults unchanged. Next qualify a clean ACSPH wall closure and common stopping
criterion; then extend toward impact. Full report and costs in the linked
Lobovsky comparison document.

## Reference bring-up — 2026-10-27

User: use the four supplied `~/dev/dynGPUSPH/tosend` presets as a runnable
reference, allowing build/compiler compatibility work. Presets: `Impact200`,
`LDCAVITY_400_2L_RE1000`, `SquareRotation400_2L`,
`dambreak_Lobovsky_400_2L`. Source checkout: `FSI_devel` paper-era commit
`2dbc27a`. Build in an isolated copy; preserve original source and input hashes,
record compile flags per case, and separate compatibility/output changes from
numerical changes. First qualify short original-resolution runs with snapshots,
video, progress and stop conditions before longer reference comparisons.
The old makefile points at CUDA 7.5 / private include paths; local CUDA is 13.0
on Blackwell. Track recipes and results in `docs/acsph/REFERENCE_PRESETS.md`.

**Bring-up complete:** CUDA 13 / Blackwell base and no-slip cavity binaries;
all four original-resolution bounded samples exit 0 with video. Impact crosses
first collision (19/69 steps capped), Lobovsky startup caps on 17/25 steps;
cavity and square startup have no cap hits. Clean-rebuild 20-step impact
control matches every stored field bitwise; the original teardown failure is
retained. Native `VOL` double-counts the self kernel and the final probe's time
is one dt ahead, so neither is used silently as physical-volume/time truth.
Next: import these exact ICs into warpSPH, match mass/kernel/viscosity/wall motion,
then compare fields and convergence. No warpSPH equations/defaults changed.
The normalization's old “stricter at higher N” interpretation is corrected
below; the formula is unchanged.

Target paper — `literature/decourcy2024_incompressible-delta-sph-artificial-compressibility.pdf`, bib key `decourcy2024`:

> **Incompressible δ-SPH via artificial compressibility**
> J.J. De Courcy, T.C.S. Rendall, L. Constantin, B. Titurus, J.E. Cooper
> *Computer Methods in Applied Mechanics and Engineering* **420** (2024) 116700
> `doi:10.1016/j.cma.2023.116700` — CC BY, open access. 40 pp, 85 refs.
> Precursor: De Courcy et al., SPHERIC 2023 (ref [32]).

Synced into `literature/` on 2026-09-05 together with seven of its references
(see Part 6); `python scripts/check_literature.py` passes with all eight
abstracts verified verbatim against their PDFs.

---

# TO DO — extracted 2026-09-21 by a full audit pass

This list is the output of a line-by-line audit of this document against
(a) `literature/decourcy2024_*.pdf` itself, (b) the code it describes, and
(c) the probe scripts it cites. **What the audit verified is in §0.4 below
— don't redo it.** Everything here is either open work or a doc chore the
audit could not settle without a judgement call.

## 0.1 Real work, highest value first

1. **Root-cause the `t* ≈ 6` ACSPH dam-break divergence.** The headline open
   item (status board 2026-09-20 item 2): stable to `t ≈ 1.34`, then `maxV`
   jumps 6.1 → 183 at the wave breaking and goes NaN, while density stays
   exactly 1.00000 — a momentum/pressure failure, not a density one.

   **Start by re-running it at the corrected free-slip wall** (§0.3). Every
   stored run under `scripts/out_deltaSPHMarrone/` — the ACSPH divergence *and*
   both δ-SPH references — carries `wallBC: None` in its metadata, i.e. all of
   them were taken at the old `'constant'` default, which is an effective
   no-slip bed. That is not a detail: `DELTASPH_VALIDATION_PLAN.md` §5.7
   measured that same BC driving `pb@front` *negative* (−0.014/−0.018) as the
   front passes — an **attracting** wall that pulls particles into the boundary
   band — and this divergence is a wall-localised momentum blowup at the moment
   of wave impact. The two could be the same mechanism. This is also much
   cheaper than the alternative and re-baselines the A/B either way.

   If it survives that, the plan's own next discriminating test — **the same
   dam break with `wallPeriodic=True`** — is still unrun; the status board's
   item 7 periodic TGV/randomFlow A/B already localises the fault to the wall /
   free-surface interaction.
2. **Register that divergence in `OPEN_PROBLEMS.md`.** It meets every bar the
   file states (traced to a real mechanism, investigated across sessions, not
   a quick fix), and the repo convention is to look there before
   re-investigating. Currently `OPEN_PROBLEMS.md` has one passing ACSPH
   mention and no entry.
3. **Run the Lobovsky dam break** (§4.5 / step 9). The two prerequisites the
   plan listed as blockers are both built and committed —
   `scripts/probe_acsphDambreakLobovsky.py` and `dambreak.py`'s
   `pressureProbeHeights` — and `scripts/out_acsphDambreakLobovsky/` does not
   exist for either scheme. Match the paper's §4.5 settings (free-slip,
   `H/Δx = 400`, RK3, `CFL_t = 0.4`, `Δt/Δτ = 2.0`, displacement shifting)
   before quoting any discrepancy.
4. **Re-run `impact` at `aspectRatio = 0.25`.** §4.4's headline (`0.2492` vs
   the analytic `0.2495` at nx=64) was measured at `aspectRatio = 0.5`, which
   is `H/L = 1` — Marrone's own figure configuration, not De Courcy's, whose
   §4.4 case is `H = 2L` ⇒ `r = π/4` ⇒ `aspectRatio = 0.25` in this
   parameterisation. Either run is defensible; the plan just has to say which
   reference it is claiming.

   **Decided (user, 2026-09-28): the Marrone configuration
   (`aspectRatio = 0.5`) is the reference** -- it is the widely used one, and
   if the solver cannot handle it, something is wrong. De Courcy's `H = 2L`
   (`aspectRatio = 0.25`) is a final-validation extra only, not a gate.
5. **Commit the working tree.** It now holds both the 2026-09-20 entry's
   items 1 and 7 (which the plan already describes as landed —
   `rigidBody/update.py`, `wp_densityDelta.py`, the two periodic cases, two
   test files) and this audit's own changes (the `wallBC` flip in
   `cases/freeSurface/dambreak.py` + `caseUtils/weaklyCompressible.py`, the three probe
   docstrings, `uChar` on the two periodic cases, this plan and
   `DELTASPH_VALIDATION_PLAN.md` §5.7). Worth splitting into two commits.

   **Pre-existing red test, unrelated to any of it:**
   `test_runner.py::test_everyCaseDeclaresItsParamsAsScalarsOrLists`. One half
   was a real bug and is fixed (`dambreak.surfacePressureProbes` was a tuple,
   which `buildArgumentParser` silently gives no CLI flag). The other half —
   `dambreak.alpha is NoneType` — is the *test* being too strict: `None` is a
   deliberate "leave the scheme's own default alone" sentinel in five cases
   (`dambreak`, `openFlow`, `droplet`, `sloshingTank`, `tgv-wc`). Either the
   test should allow `None` or those sentinels should go; not decided here.
6. **Still open at scale** (unchanged, no new finding): the Table 1/2 error
   cliff at paper resolution (§4.3 / step 6); Fig. 15/16 vs. `letouze2013`'s
   BEM crosses, blocked on digitization (step 8); AC-4's standalone
   divergence vs. the paper's milder "struggles to maintain a converged
   kinetic energy" (step 8 / the §4.1.1 authors' question).

## 0.2 Doc chores left open (each needs a call, not just a fix)

- **`.claude/skills/gradcheck/SKILL.md`'s script list is stale** — it names 15
  scripts and omits `gradcheck_jstSwitch.py` (this plan's step 8),
  `gradcheck_michelUChar.py` and `gradcheck_nearestSurfaceNormal.py`. They
  *do* run in CI (`tests/test_gradcheck_scripts.py` globs `gradcheck_*.py`),
  so this is a skill-listing bug, not a coverage hole — but `/gradcheck`
  users read that list.
- **Promote `marrone2011` to the core literature set** and add the §6.2
  residue to `literature/EXPANSION_CANDIDATES.md` (step 1's leftovers,
  still open). Genuinely missing refs are now only [70] `marrone2010` and
  [40] `jameson1981` (which would settle §5.1).

## 0.3 Fixed in this pass (recorded so it isn't re-audited)

### Model / config changes

- **`dambreak`'s `wallBC` default flipped `'constant'` → `'freeSlip'`**
  (`cases/freeSurface/dambreak.py`, `caseUtils/weaklyCompressible.py::buildRegions`, which
  `dambreak` is the only caller of). Every reference this family is graded
  against specifies free slip — De Courcy §4.5 explicitly, Marrone 2011 §3
  (3.1 and 3.4.1), Lobovský 2014 / Buchner 2002 — and `'constant'` is not it:
  it adds no *physical* viscosity term but leaves the wall at `v = 0` while the
  **artificial** viscosity runs `AllToAll`, so the fluid is dragged against a
  stationary bed. `DELTASPH_VALIDATION_PLAN.md` §5.7 measured that cost (front
  3.4 → 4.13 m/s, impact `t*` 2.8 → 2.6 against a reference 2.3, and a
  *negative* `pb@front` that attracts fluid into the boundary band) and then
  kept `'constant'` as the default purely so recorded runs would not move.
  **Blast radius:** every `dambreak`-family run taken without an explicit
  `--wallBC` now changes — the Marrone 3.1/3.4 probes, the Lobovsky probe, and
  the ACSPH cases. Every run recorded since §5.7 already passed `freeSlip`
  explicitly, so the *current* baselines are unaffected; older stored numbers
  taken at the old default are reproducible with `--wallBC constant`, and the
  probes' `--wallBC` help now says so. §3.4.2's viscous half still wants
  `noSlip` and must ask for it.
  - **Every stored Marrone baseline predates the flip.** All the `.npz` runs
    under `scripts/out_deltaSPHMarrone/` — the ACSPH `t* ≈ 6` divergence and
    both δ-SPH references — record `wallBC: None`, i.e. the old `'constant'`
    default. They are still reproducible (`--wallBC constant`) but they are no
    longer what a bare re-run produces, and §0.1 item 1 now starts from that.
  - Consistency argument, independent of the references: `hydrostaticColumn`,
    `columnCollapse` and `sloshingTank` already defaulted to `'freeSlip'`.
    `dambreak` was the only walled case that did not.
  - The `scripts/crossengine/*` harness passes `--wallBC constant` explicitly
    (it matches diffSPH's own configuration), so it is pinned and unaffected.
  - Three stale docstrings fixed with it: `probe_deltaSPHMarrone.py` claimed
    "the free-slip spec is met without a slip-mode knob" (it is not — that was
    §5.7's exact finding), and `probe_acsphDambreakLobovsky.py` documented and
    reported "no-slip tank walls (the case's own default)".
- **`randomFlow` and `tgvWeaklyCompressible` now set `acParams.uChar`**
  (`uMag` for TGV; the seeded field's own `|v|_max` for the noise case, which
  has no closed-form velocity scale). They were the only ACSPH-wired cases
  without one, so Eq. (48)'s `U_ε` fell back to the instantaneous `|v|_max` —
  a moving target that makes a fixed `ε_v` mean different things at different
  times, which is the exact failure §3.1.3 describes.
- **§5.4 closed from the paper's own text** rather than worked around: `𝕍` is
  one set, defined identically before Eq. (36) and before Eq. (57), so the
  planned "expose the dilation iteration count separately" would have invented
  a degree of freedom the paper does not have. Dropped from the authors'
  questions.
- **§5.5 corrected**: `uChar` is `Optional` with a documented `|v|_max`
  fallback, not the "required per-case config value" the plan asked for — and
  the section now lists what all seven wired cases actually use.

### Doc errors found against the code and corrected in place

- Step 5b's *"the domain is made non-periodic on this branch"* and §4.4's
  *"Restriction, deliberate"* both contradicted decision 3 and the code:
  `wp_wallMoment.py` made the moment minimum-image correct and
  `hydrostaticColumn.py:441` is back on `periodic=True`. Both rewritten.
- Step 6 claimed Eq. (46)'s first term is implemented as `CFL_t √(h/‖a‖_max)`.
  It is not — the code implements §5.6's
  `CFL_t h / max(REFERENCE_VELOCITY, ‖v‖_max)`; `CFL_t √(h/‖a‖_max)` is the
  *separate* body-force constraint behind `dt_accelerationConstraint`.
- Part 3 mapped Eq. (25)'s `(p_i+p_j)` to `PressureForceScheme.Antuono`; it is
  `nonConservative` (as decision 2 and the config default already say).
- §1.1 labelled the *continuous* momentum/velocity rows Eqs. (25)/(26); those
  are the discrete forms — the continuous system is Eq. (50)/(51).
- §1.6 called `Δt/Δτ = 5` "the best cost/accuracy point in Table 2" while
  Part 2 said 2; Table 2 holds `Δt/Δτ = 5` fixed and cannot speak to the
  ratio at all. Reconciled against the paper's own sentence (`Δt/Δτ = 2`,
  `CFL_t = 0.4–0.6`).
- Part 7's acceptance target omitted AC-4, which the paper says *does* resolve
  the hydrostatic profile — sharpening what our AC-4 divergence contradicts.
- Part 9.2 said "nothing beyond §4.1.1 has been run"; step 7 said PST Stage A
  was "still the next action"; step 9 said the Lobovsky comparison needed
  prerequisites that exist. All three replaced with the real state.
- §6.2 listed [77] Monaghan & Rafiee and [6] Molteni & Colagrossi as "still
  not obtained"; both are in `literature/` with bib entries and abstracts.
- Paths: `DFSPH_IMPROVEMENT_PLAN.md` → `docs/historic_plans/` (4×);
  `EXPANSION_CANDIDATES.md` → `literature/`;
  `tests/test_artificialCompressibleScaffold.py` no longer exists (folded into
  `tests/test_artificialCompressible.py`).
- Drifted anchors: `runner.py:269` → `:288`; `caseSpec.py:286` → `:304`.
  Integrator names `rungeKutta2`/`sspRK3`/`rungeKutta4` →
  `getButcherTableau('midpoint'/'SSPRK3'/'RK4')`.
  `validateIntegrationScheme` accepts `explicitEuler` too, not only
  `forwardEuler`.
- Counts/figures: `test_wallPressure.py` has 8 tests, not 6; the wall
  under-read is 13 % (= the quoted `1.3e−1`), not 12.5 %; Table 1's floor is
  0.43, not ~0.46. `𝒞_e` → the paper's own symbol `e`.

Added: **Part 2's per-case settings table** (what §4.1–§4.5 actually ran, which
was nowhere in the plan and is what a faithful reproduction needs), and
**§5.7**, a new paper ambiguity — §4.4 prints `Δτ/Δt = 2` where Table 1, §4.5
and §3.1.3 all imply `Δt/Δτ = 2`.

## 0.4 What the audit verified — no action needed

- **Every equation in Part 1 against the PDF**: Eqs. (23)–(26), (30)–(37),
  (38)–(48), (49)–(51), (55)–(62), (63)–(65). All transcriptions match,
  including the three the plan flags as paper errors — Eq. (37) really does
  print `min(0, κ₄ − ε₂)` (§5.1), Eq. (30) really does carry the stray `h`
  (§5.3), and Eq. (46)'s first term really is `CFL_t h`, a length (§5.6).
- **Every constant in Part 2**: `δ = 0.1`, `k₂ = 0.1hβ` (ceiling `0.2hβ`),
  `κ₂ = 0.5`, `κ₄ = 1/32`, `CFL_τ = 0.5/1.0/1.5`, `ε_s = 1e−5`,
  `α_ν = 0.01` with `ν = α_ν h c₀/K`, `λ < 0.4`, the `1 + 0.2(W/W(Δx))⁴`
  tensile term, `CFL_wc = 0.75`.
- **Every acceptance target in Part 7**: square-patch KE loss 32.1/26.4/25 %
  and cost 2.81/2.60/2.38×; jet-impact cost "no more than 50 % greater";
  void closure at `t√(g/H) ≈ 8.4`; Table 1's 2.34× jump from `CFL_t` 0.4→0.6
  and Table 2's ~10× at 1.0. Marrone 2015 Eq. (A.32) `r = (π/2)(L/H)` also
  confirmed against its own PDF.
- **The code the plan describes**: every `acParams` field of §4.6; the four
  `PressureSmoothingScheme` members; Eq. (46) in
  `modules/timestep/artificialCompressible.py`; Eqs. (47)/(48) in the
  driver's `_convergenceMetric`; the Butcher dispatch; `ShiftingScheme`/
  `ShiftingProjectionScheme.michel2022`; the per-case `uChar` defaults on all
  five Part 7 cases; `ctx.scratch['lastStageUpdate']`; `config.maxDt = 1e-2`.
- **All 16 probe scripts the plan cites exist and parse.**
- **Tests green** (2026-09-21, `envs/warp`): `test_artificialCompressible.py`
  + `test_wallPressure.py` + `test_deltaSPHDiffusion.py` = 46 passed;
  `test_physics.py -k acsph` (`test_acsphColumnHoldsTheWall`) = 1 passed.

---

# Status board — the history behind the list above

**Done (2026-09-05):** steps 1–6 of Part 8, and step 7 in full — implemented
*and* measured. The scheme runs end to end: `--scheme artificialCompressible`
builds, the dual-time driver solves, and `hydrostaticColumn` runs to
`t = 0.94` without diverging. Step 7 (Michel et al. particle shifting) has its
own document, **`PST_ALE_PLAN.md`**, whose Stage A landed 2026-09-05: the
law, its free-surface treatment, and the wiring into
`ArtificialCompressibleSystem.finalize` all exist and are gradchecked/tested
(off by default — `buildDefaultACSPHShiftProperties`).

**Re-measured this column with it on** (`scripts/probe_michelHydrostaticColumn.py`,
nx=32, 200 steps, one seed — see `PST_ALE_PLAN.md` §7.1 for the full table).
A real units bug turned up while doing this and is now fixed:
`computeMichelShift` was applying Eq. (22)'s shifting *velocity* straight to
`positions` with no `dt` factor (Eq. (58) is `dx/dt = u + delta_u`) —
confirmed ~8x the particle spacing per call before the fix, ~sane after it.
Numbers below are post-fix. Split result: `pairedFraction` — "exactly and
only what particle shifting exists to prevent" — **does move**, 0.065 →
**exactly 0.000** with the shift active (better than the pre-fix run, not
worse — the fix made the finding stronger, not weaker). `‖v‖` at the corners
**does not**: the shift alone (peak 2.33) is well above `noPenetrationShift`
alone (0.75); running both together helps (1.63) but doesn't match the
safeguard alone. **This overturns decision 4 below** — the shift is not a
replacement for `noPenetrationShift`, it fixes a different failure mode.
**The `δu_max` convergence-rate reproduction (Fig. 1) is now done too**
(`scripts/probe_michelConvergenceRate.py`, `PST_ALE_PLAN.md` §7.1) — the
test the `rotatingSquarePatch` footprint-drift proxy above turned out not to
substitute for. Clean pass: `michel2022` measures first-order convergence
(log-log slope 0.949 vs. the theoretical 1.0) against `deltaSPH`'s flat
-0.063, on a jittered periodic lattice swept nx 16→128 at fixed `R/Δx`.
Table 2's interior-consistency claim is now directly confirmed against this
implementation, not just inherited from the paper.

**`sloshingTank` is measured, root-caused, and fixed** (`scripts/probe_sloshingTankSurfaceShift.py`,
`PST_ALE_PLAN.md` §7.1): `michel2022` runs the full `tLimit=4.0` (20001
steps) with density held to `[0.994, 1.011]` — the only one of four shift
configurations tested that doesn't diverge; `noShift`/`surfaceNormal` diverge
at `t=0.34`/`t=0.68`, roughly an order of magnitude earlier than
`docs/historic_plans/WCSPH_SHIFTING_PLAN.md` documents (`t≈2.6-3.5s`).
**Confirmed root cause** (not just suspected): re-ran the identical probe
against the commit *before* the psi-sign fix in an isolated `git worktree`,
and got the documented pre-fix numbers back almost exactly
(`noShift` diverges at `t=2.76`, `surfaceNormal` survives to `t=4.0`). See
decision 1's update below for why that fix is correct and should not be
reverted. **Fix applied**: `cases/freeSurface/sloshingTank.py`'s `deltaSPH` branch now
defaults to `ShiftingScheme.michel2022`/`ShiftingProjectionScheme.michel2022`
instead of the shared default, confirmed end-to-end through the case's own
real entry point (`run(sloshingTankCase, params={'shifting': True}, nx=60,
tLimit=4.0)` → 20001/20001 steps, not diverged). Full repo test suite green.

**The full `deltaSPH` validation-case sweep this regression made urgent is
also done** (Part 9.1): all 11 other `deltaSPH`-family weakly-compressible
cases in the registry, each run to its own `tLimit`, pass clean, and
genuinely exercise the same shift-dependent code (`surfaceNormal`,
`active=True` by default in every one). `sloshingTank` was the only case
affected. The psi-sign fix is confirmed **not** a broad regression — this
line item is closed.

**Michel §5.4's own free-surface `δu_max` convergence rate is also done now**
(`scripts/probe_michelFreeSurfaceConvergenceRate.py`, `PST_ALE_PLAN.md` §7.1):
a rigid-rotation field turned out to be degenerate for this measurement
(`U_char` is exactly zero for solid-body rotation at every pair — an
identity, not a resolution effect), so it reuses the interior probe's own
Taylor-Green field on a non-periodic (genuinely free-surface-bounded) box.
The domain-wide `δu_max` just reproduces the periodic bulk result (Eq. 48
can only shrink a shift, never grow the global max); the maximum restricted
to the dilated free-surface set converges at slope **1.811** — faster than
first order, clearing Table 3's claim with margin. That closes out both
halves (interior + free surface) of the Table 2/3 audit for this
implementation.

**The multi-seed/multi-resolution `hydrostaticColumn` sweep is also done now**
(`scripts/probe_michelHydrostaticColumnSweep.py`, `PST_ALE_PLAN.md` §7.1),
and it **refines rather than simply confirms** the single-seed finding above.
The single-seed run used a perfectly regular lattice — the case's own jitter
path is dead code, re-enabled from outside for this sweep — across
`nx ∈ {24,32,48} × 3 seeds × 4 modes`. Two things hold up: `pairedFraction`
is lower under the shift than under `neither` at every resolution and every
seed (the core claim), and **`both` (shift + safeguard) is the uniformly
best and most seed-stable configuration** at every resolution. But **decision
4's overturn above needs qualifying, not retracting**: `michelShift` alone is
not *categorically* worse than `noPenetrationShift` alone — at nx=32 it's
actually better on mean `‖v‖` peak (0.470 vs. 0.612) — it is just markedly
less *seed-stable* (std up to 0.896 vs. `noPenetrationShift`'s 0.014-0.338,
tightening with resolution). So the single regular-lattice seed happened to
land on a case where the shift underperforms; the safeguard's real advantage
is consistency across configurations, not a categorically lower velocity.
See decision 4 below and `PST_ALE_PLAN.md` §7.1 for the full table.

**Stage A (Michel PST) is fully closed. Per user direction, `PST_ALE_PLAN.md`
Stages B/B′/C/D are on hold** until Part 8 steps 6, 8, 9 and §4.2 below —
the paper's own remaining validation cases — are done, since none of them
need anything from those later PST stages.

**2026-09-05, second pass — all four remaining validation cases wired for
ACSPH** (`oscillatingDroplet`, `rotatingSquarePatch`, `impact`, `dambreak`;
`hydrostaticColumn` was already done). Each follows the same pattern
`hydrostaticColumn` established: an `_configureArtificialCompressible`
branch calling `configureArtificialCompressible` (or, for `dambreak`, whose
`configureScheme` never went through that shared helper, the handful of
things it would have done, applied by hand), a per-case `acParams.uChar`
default (§5.5's gap — `A*R` for the droplet's edge speed, `omega*size` for
the patch, `impactVelocity` for the bodies, `sqrt(g*fillRatio*L)` for the
column, matching `dambreak`), and a `timestep=` hook dispatching to Eq. (46)
for ACSPH while leaving every other scheme's behaviour bit-for-bit
unchanged. All four pass smoke tests with sane physics, not just
"doesn't crash": `oscillatingDroplet` runs 200 steps to `t=1.85` with KE
oscillating 0.003-0.76 (a real oscillation, not a decay); `rotatingSquarePatch`
tracks its own WCSPH run's KE/`‖v‖` closely at 20 steps; `impact` shows the
bodies actually colliding (`gap` closing to 0.043, KE dropping from 0.203 to
0.142 on impact, `comDrift` ~3.5e-8 — momentum conserved); `dambreak` runs
150 steps under gravity with zero wall penetration. Full test suite green
throughout (no case's non-ACSPH path changed behaviour).

**Step 6, the Table 1/2 reproduction — real machinery built, mechanism
demonstrated, full paper-scale numbers still open.** Needed De Courcy's own
`IRMSE(KE)/IRMSE(a)` (the semi-major-axis error), which requires the
droplet's *exact* time-dependent shape — the two already-encoded constants
(`DROPLET_STRETCH`/`DROPLET_PERIOD`) only sample two isolated instants. That
solution is Monaghan & Rafiee (2013) [77], cited but not itself in
`literature/` (§6.2 already flagged this as the one genuine gap); **the user
supplied the PDF** (filed under its own title, "A simple SPH algorithm for
multi-fluid flow with high density ratios", IJNMF 71(5) 537-561, DOI
10.1002/fld.3671 — Wiley's "2012" Early View date on the filename, "2013" the
print-issue date the citation uses). Re-derived the governing ODE from its
Appendix A rather than trusting the OCR'd equation text directly (which is
easy to misread as `dsigma/dt = -Omega**2 + 2*(b**2-a**2)/(a**2+b**2)` — the
"2" is `Omega`'s superscript, not a coefficient; that misreading fails the
check below):

```
dsigma/dt = (sigma**2 + Omega**2) * (b**2 - a**2) / (a**2 + b**2)
da/dt = sigma * a
db/dt = -sigma * b
```

with `Omega**2 = B**2` (`computePotentialFieldGravity`'s acceleration is
`-magnitude**2 * r`) and `sigma(0)=A, a(0)=b(0)=R`. **Verified against the
already-encoded constants**: at the case's own `A=B=R=1` default, this ODE's
first peak of `a(t)` lands at 1.931852 at `t=1.207` — against
`DROPLET_STRETCH=1.931843` — with the next peak at `t=6.034`, a period of
4.827 against the encoded `DROPLET_PERIOD` exactly. Implemented as
`oscillatingDroplet.analyticSolution` (also derives `KE(t)` from Appendix A
Eq. A.28-A.29 specialised to one fluid), plus `_measuredSemiAxes` (mass-weighted
second moments of the fluid particle cloud — exact for this problem, since
the straining IC has no rotational component, so the ellipse never leaves
the x/y axes) wired into `diagnostics` as `semiAxisA`/`semiAxisB`.

Eq. (65)'s effective cost `e` needed the per-step pseudo-iteration count,
which nothing outside the scheme's own `update` object could see before now
— `diagnostics`/`postStep` receive `runningState`, never `stepResult`. Fixed
with one general, non-case-specific line in `runner.py`: after each step,
`ctx.scratch['lastStageUpdate']` is stashed from `stepResult.stages[-1].update`
(the ad hoc `pseudoIterations`/`epsilonV`/`bdfOrder` attributes ACSPH's step
already sets on it), so any case's `diagnostics` can read scheme-specific
extra step data with `getattr(..., None)`; `1` (one function evaluation) for
every scheme that sets nothing extra. `oscillatingDroplet.diagnostics` reads
it as `pseudoIterations`. Since diagnostics are recorded once per real step
of size `dt`, Eq. (65)'s integral telescopes exactly to
`sum_steps(pseudoIterations) * s_RK` — no quadrature needed.

`scripts/probe_acsphOscillatingDropletTable1.py` runs the sweep: a δ-SPH
reference plus an ACSPH cell per `(CFL_t, Δt/Δτ)`, both scored against the
same real analytic ground truth, ACSPH's own IRMSE/`w`/`e` reported
normalised by the reference's (Table 1's own convention). **Found and fixed
a real bug in the process**: `config.maxDt` defaults to `1e-2` (`runner/caseSpec.py`,
a generic global default) and is unconditionally one of Eq. (46)'s `min()`
candidates (`modules/timestep/artificialCompressible.py`) — so with the
default, every `CFL_t` in `[0.1, 0.6]` produced *bit-identical* trajectories
on the first sweep, `dt` silently pinned to `maxDt` throughout. Fixed by
raising `--maxDt` (default `1.0`) in the sweep script; not a scheme bug, but
a sharp footgun for any case whose natural adaptive `dt` exceeds the generic
default, worth knowing before trusting *any* ACSPH sweep result on a new
case.

**Reduced-scale sweep result** (nx=24, one period, RK3 — not the paper's
`L/Δx=200`, multi-period scale; a single real step already costs ~1s at
nx=32 on one CPU core here, so that reproduction is a separate, larger,
budgeted run):

| `CFL_t` | `Δt/Δτ` | steps | `IRMSE(KE)` | `IRMSE(a)` | `w` | `e` |
|---|---|---|---|---|---|---|
| 0.1 | 2 | 391 | 0.188 | 0.201 | 1.78 | 6.06 |
| 0.1 | 5 | 391 | 0.191 | 0.202 | 1.82 | 6.08 |
| 0.2 | 2 | 208 | 0.183 | 0.213 | 0.64 | 2.16 |
| 0.2 | 5 | 208 | 0.281 | 0.247 | 0.97 | 3.23 |
| 0.4 | 2 | 118 | 0.184 | 0.220 | 0.30 | 1.02 |
| 0.4 | 5 | 118 | 0.175 | 0.194 | 0.41 | 1.36 |
| 0.6 | 2 |  90 | 0.227 | 0.260 | 0.22 | 0.76 |
| 0.6 | 5 |  90 | 0.219 | 0.223 | 0.29 | 0.95 |

(all four ratio columns normalised by the δ-SPH reference's own
`IRMSE(KE)=0.345`, `IRMSE(a)=0.224`, `w=155s`, `e=3.86e4` at the same
resolution.) **Cost drops monotonically and sharply with `CFL_t`, matching
the paper's own claim exactly** (`w`/`e` fall ~8x from 0.1 to 0.6). **The
error side does not yet reproduce Table 1's sharp 0.4→0.6 jump** — `CFL_t=0.4`
is if anything the *lowest*-error row here, not a cliff edge — most likely
because one period at nx=24 is too short/coarse to resolve BDF2's accuracy
cliff cleanly against this much noisier baseline (every ratio here sits well
under the paper's own Table 1 floor of 0.43, meaning ACSPH is doing
comparatively better against δ-SPH at this coarser scale than at the
paper's, which dilutes the accuracy-cliff signal specifically, not the cost
one). **Open**: a longer, finer, multi-period reproduction before trusting
the error trend; the mechanism (real analytic ground truth, real cost
accounting, both schemes measured identically) is now correctly built and
does not need revisiting.

**Step 8, AC-4 — implemented, formula verified against the actual PDF, but
found genuinely unstable.** `PressureSmoothingScheme.biharmonic`
(Eq. 35) is `-h**2 * (AC-2 applied to AC-2's own raw output)` — no new
kernel, `computeScalarFieldDiffusion(..., densityOnly, field=...)` called
twice (`modules/artificialCompressible/pressureSmoothing.py`). Re-checked
character-by-character against `decourcy2024`'s own PDF text (not just this
plan's transcription) and matches exactly. Run on `hydrostaticColumn` *and*
`rotatingSquarePatch` (nx=24, no walls, so not a boundary-treatment
artefact) it diverges within ~15 real steps (pressure/velocity reaching
`O(1e5-1e6)`) — measured directly on the initial hydrostatic state alone
(before any dynamics): AC-2's own raw output is already large near a free
surface (the known-bad truncation Eq. 35 is built from, unrenormalised), and
nesting it without correction amplifies that ~20x in one more pass. The
paper's own text (§4.1.1) says AC-4 "struggles to maintain a converged
kinetic energy" but does *not* report an outright blow-up, and separately
reports AC-4 resolving the hydrostatic free-surface profile fine at
`t=50s` — so this is either a resolution/parameter sensitivity the coarse
nx=24 test here falls into more severely than the paper's own setup, or a
subtler issue than the formula itself (already verified correct). **Not
validated for production use** — implemented and callable for
experimentation, but flagged in its own docstring as unstable pending a
resolution/parameter study; do not default to it. AC-JST (Eqs. 36-37) is
not attempted yet — it needs a genuinely new kernel (the `chi` switch is a
nonlinear function of both `p_i` and `p_j`, not a per-particle field
`computeScalarFieldDiffusion` can express) plus the `𝕍`/`min`-vs-`max`
questions §5.1 already flagged, a larger and more self-contained piece of
work than AC-4 turned out to be.

**2026-09-05, third pass — AC-JST implemented, closing out step 8 and every
remaining *mechanism* item in the plan.** See Part 8 step 8's own entry for
the full account: one new gradchecked kernel (`wp_jstSwitch.py`, Eq. (37)'s
`chi_i`), the `𝕍`-gated AC-2L/AC-4 blend, §5.1's `min`/`max` resolved as
already decided. **AC-JST also cross-validates AC-4** — both stay bounded
where AC-4 alone diverges, confirming AC-4's formula is correct and its
standalone instability is real rather than a bug. Full test suite green.

**Every "build it" item in this plan is now done**, and every sweep/validation
item has now had a real attempt at reduced scale (2026-09-05, fourth pass) —
see each step's own entry above for the numbers:

| item | result |
|---|---|
| Step 6, Table 1/2 sweep | **Done at reduced scale** (nx=24, one period). Cost trend matches the paper cleanly; the accuracy-cliff trend does not resolve cleanly at this scale (§4.3's own entry above and the new authors' question). A larger (nx=32, 1.5 periods) attempt was started but abandoned after ~40 minutes of CPU time still on the reference run alone — not worth the wall-clock cost for a plan closeout; the existing reduced-scale numbers stand. |
| Step 8, Fig. 2 (hydrostaticColumn operators) | **Done at reduced scale** (nx=48, 300 steps). Qualitatively matches: AC-2 degrades over time (can't hold the gradient), AC-2L/AC-JST hold it, AC-JST clearly best (`pressureSlopeRatio` 0.995 vs. 0.783 vs. 0.721). AC-4 diverges immediately, as already known. |
| Step 8/§4.2, Fig. 15/16 (rotatingSquarePatch vs. BEM) | **Deferred.** Needs digitized reference data from a figure, not a formula — not attempted rather than risk silently encoding wrong numbers. |
| §4.4, `impact` vs. Marrone 2015 | **Done, genuine formula-level validation.** Found and implemented their exact closed-form energy-loss ratio; at nx=64 the simulation's KE trajectory passes through the analytic target almost exactly (0.1% off) at an intermediate time, though "the instant to compare" has no clean definition in a continuously-evolving sim. Real, encouraging, not a clean pass/fail. |
| §4.5, `dambreak` vs. Lobovsky 2014 | **Deferred.** Needs the experiment's own physical geometry and new per-height pressure probes, not present in this repo's dimensionless case as configured. |

**What's left in this plan is now entirely a matter of scale, not substance**:
finer/longer versions of the four sweeps above, and the two deferred items'
own prerequisite work (BEM digitization, dambreak geometry/probe matching).
Nothing here blocks anything else. `PST_ALE_PLAN.md` Stages B′/B/C/D remain
queued after all of this, per the user's own sequencing.

**2026-09-20 — boundary regression fixes, the long-horizon Marrone run, and
the §4.4 closeout.** Six things landed:

1. **Two regressions that were blocking *every* walled ACSPH run, fixed.**
   `updateBodyParticlesWCSPH` read/wrote `boundaryAccelerations`
   unconditionally, which `ArtificialCompressibleState` does not declare →
   `AttributeError` at init and in every `finalize`. It is now threaded
   through only when the state type has it. Separately,
   `computeDensityDiffusionDeltaSPH` passed bare Python floats for `rho0`/`c0`
   into a generic kernel (warp infers them as float32, mismatching the
   scalar_t build precision); wrapped in `scalar_t(...)` — gradcheck green.
   The stale `test_theUnimplementedSmoothingOperatorsRaise` (AC-4/AC-JST now
   *are* implemented) is replaced by `test_ac4AndAcJstRunWithFinitePressures`
   + `test_ac4AndAcJstAreDifferentOperators`.

2. **The Marrone §3.1 long-horizon run — the headline — *diverges* at
   t\* ≈ 6.** ACSPH (nx=70, Michel PST on, `wallBC=constant`, `ε_v=−5.0`,
   video) ran to `t = 1.477` (t\* = 5.97) then blew up (non-finite
   velocities, step 7845, 20483 s ≈ 5.7 h wall). It is stable to `t ≈ 1.34`
   (KE ≈ 0.7–0.9, `maxV` ≈ 6.1, steady `Δt` ≈ 3.5e-4), then at t\* ≈ 5.5
   `maxV` jumps discontinuously 6.1 → 183 — right at the wave-impact/breaking
   — oscillates, and goes NaN. **Density stays exactly 1.00000 throughout**
   (the constant-density invariant holds), so this is a *momentum/pressure*
   instability at the nonlinear wave breaking, not a density failure. The
   delta-SPH reference (same tank, sensors, `wallBC`) runs cleanly to
   t\* = 20.22 (density [0.977, 1.022], max penetration 0.01 dx). A/B
   (`scripts/out_deltaSPHMarrone/`, `compare` in `.tmp`): ACSPH tracks the
   reference through the early phase (P1 arrival t\* 2.57 vs 2.81;
   KE@1 s 0.94 vs 0.65) but generates an unphysical P3 pressure spike (first
   peak 5.77 vs 0.000) and then diverges. This is the concrete, measured
   confirmation of the "seemed unstable" validation concern: **ACSPH as
   implemented is not yet robust through a full dam-break impact/breaking**,
   while the incompressible delta-SPH family is.

3. **`ε_v` convergence plateau, measured.** At the nx=70 dam break the
   dual-time fixed point asymptotes at `ε_v ≈ −5.8` and does not reach the
   −6 default within the 200-iteration cap, so a −6 target runs the full
   budget every step. `ε_v = −5.0` converges in ~31 iterations. (A cost
   finding, not a correctness one; the long run used −5.0.)

4. **Corner table re-measured post the `786e300` freeSlip fix** (the step 5b
   table above). The fix changed the picture sharply: `neither` went 2.9 →
   **6.77** (the old BC let fluid leak out; the correct one reflects it back
   at high speed), `noPenetrationShift` stays the best `‖v‖` bound (0.77),
   `michelShift` is unchanged (2.35), and `both` (2.54) is now *worse than
   the safeguard alone* — so the pre-fix "both is uniformly best" reading
   does not carry over to the regular lattice with the correct wall BC.

5. **§4.4 boundary audit closed.** `v` is correct per BC type, `δv` is
   satisfied trivially, and `ṽ` is a documented deviation whose magnitude is
   now measured: the in-loop wall **position** excursion is **0.046 `Δx`**
   (negligible) — **verdict: not worth fixing.** See §4.4.

6. **New walled ACSPH test** `test_acsphColumnHoldsTheWall` (green): the
   density invariant, the recovered hydrostatic gradient (median of the
   reported `pressureSlopeRatio`, robust to the degenerate-band divergence
   that closes the bulk fit once the surface drifts down), and boundedness.

7. **Periodic A/B: the t\*≈6 dam-break blowup is boundary-related, not a
   scheme-level instability.** Two wall-free, divergence-free-in-initial-state
   cases were given ACSPH support and run under both schemes at the same
   resolution and physical `ν` (`.tmp/probe_randomFlowAB.py`,
   `.tmp/probe_tgvAB.py`):

   * **TGV** (`tgvWeaklyCompressibleCase`, k=2, L=2π, uMag=1, ν=0.05, Re≈63):
     the scheme-vs-*physics* comparison, since the analytic answer is
     `KE(t) = KE(0) exp(−4νk²t)` (target 0.2 1/s). **Both schemes reproduce
     it to ~3%**: nx=48 smoke — delta-SPH 0.199 (×0.995), ACSPH 0.2005
     (×1.003); full nx=128, t=2.0 (KE down to ⅔) — delta-SPH 0.1976
     (×0.988), ACSPH 0.2057 (×1.028). ACSPH's excess is the expected
     BDF2/frozen-smoothing dissipation at its much larger dt (577 steps vs
     2001); its density is exactly 1.0 throughout.
   * **Random flow** (nx=64, t=1.0): both stable; ACSPH retains ~45% more KE
     (0.381 vs 0.262), dominated by the dt artifact (delta-SPH at a fixed
     2.5e-4, 4001 steps, vs ACSPH adaptive ~3e-3, 343 steps), and its maxV
     doesn't decay (1.0→1.08 vs 1.0→0.75).

   Conclusion: where there are no walls and no free surface, ACSPH is stable
   and dissipates at the correct physical rate. The dam-break divergence
   (item 2) localises to the **wall boundary / free-surface-breaking
   interaction** — the next discriminating test is the same dam break with
   `wallPeriodic=True`.

## Decisions taken without you — overturn any of these if you disagree

1. **The δ-SPH `ψ` sign was wrong and is now fixed** (Part 3, "The sign
   error"). This changes the *default WCSPH scheme's* behaviour repo-wide, so
   it is the one with the widest blast radius. The evidence is not a judgement
   call — the Antuono correction must annihilate a linear field pair-by-pair
   and did the opposite — and a single-variable A/B on `sloshingTank` improves
   both the density floor and the pressure peak. **Worth reporting upstream and
   checking against diffSPH, which this kernel was ported from.**
   → **Confirmed and closed out** (2026-09-05, `PST_ALE_PLAN.md` §7.1, found
   incidentally while measuring Stage A on `sloshingTank`, then root-caused
   deliberately): `--scheme deltaSPH`'s baseline (`shiftProperties.active =
   False`) now diverges at `t = 0.34`, and the `surfaceNormal` shift fix
   diverges at `t = 0.68` — both roughly **10x earlier** than
   `docs/historic_plans/WCSPH_SHIFTING_PLAN.md` documents (`t≈2.6-3.5s`) for
   the same case. **Confirmed this commit is the cause**: re-running the
   identical probe against `790a7c7`'s parent commit, in an isolated `git
   worktree` so the working tree never moved, reproduces the documented
   pre-fix numbers almost exactly (`noShift` diverges at `t=2.76`,
   `surfaceNormal` survives to `t=4.0`, density `[0.969, 1.042]`).
   **This is not a reason to revert the fix.** The sign correction is
   independently re-derived from Marrone et al. 2011 Eq. (6), backed by its
   own O(N²) torch reference and `tests/test_deltaSPHDiffusion.py`'s
   linear/quadratic-field cancellation checks, and its own commit message is
   explicit about the mechanism: the old sign made the diffusion operator
   accidentally twice as strong (second-order, not the intended
   fourth-order) — "diffusive either way, hence never a blow-up, hence never
   caught". `sloshingTank`'s stability was quietly resting on that extra,
   undocumented damping. (The "improves... density floor and pressure peak"
   A/B claim above and this finding aren't actually in tension once that's
   understood: more diffusion generically looks "better behaved" on a
   short-window A/B even though it is quantitatively wrong and, on this
   specific violent-impact case, was load-bearing for stability over the
   full run.) **Fix**: `cases/freeSurface/sloshingTank.py` no longer relies on the
   shared shift default for this case — see `PST_ALE_PLAN.md` §7.1 for the
   applied fix and its end-to-end confirmation.
   → **Full validation-case sweep also done** (2026-09-05, Part 9.1): every
   other `deltaSPH`-family weakly-compressible case in the registry (11 of
   them), each run to its own `tLimit`, passes clean — no divergence, and
   all 11 genuinely exercise the same shift-dependent code path
   (`shiftProperties.active=True` by default, untouched, in every one of
   them). `sloshingTank` was the only case affected, and it is fixed. The
   sign fix is confirmed **not** a broad regression.
2. **ACSPH's pressure force defaults to `nonConservative`** (the literal
   `(p_i + p_j)` of Eq. 25), not `Antuono`. The Antuono switch is a
   tensile-instability guard the paper does not use; it is one config field
   away.
3. ~~**`hydrostaticColumn` runs non-periodic under ACSPH.**~~ **Withdrawn** —
   the underlying limitation is gone. Eq. (61)'s position moment is now a real
   operator (`modules/incompressible/wp_wallMoment.py`) that takes `x_ij` from
   `computeDistanceVec` like everything else, so it is minimum-image correct and
   periodic domains need no workaround. The case is back on the shared
   `periodic=True` default and runs 120 steps at `band=5` where the old
   two-gather decomposition died at step 45.
4. **`noPenetrationShift` is off by default** and is *not* in the paper — it is
   a wall *safeguard*, not particle shifting. The actual shift is a
   `finalize`-step displacement (Eq. 58, outside the pseudo-time loop), which
   is what step 7 built; `ArtificialCompressibleSystem.finalize` is where it
   goes, alongside where `WeaklyCompressibleSystem.finalize` runs
   `solveShifting`. The flag stays because turning it on shows exactly what a
   walled case does with neither — measured both ways in step 5b.
   → **Updated 2026-09-05, now that step 7 is measured, not just built**: the
   original expectation here (recorded in `ArtificialCompressibleSPHConfig`'s
   own field comment) was that this flag "should not be needed once the shift
   exists". Measured on `hydrostaticColumn`, one regular-lattice seed at
   nx=32: it looked still needed — the shift fixes `pairedFraction` (interior
   particle clustering) completely (0.065 → 0.000) but the shift alone (peak
   2.33) sat well above the safeguard alone (0.75).
   → **Refined 2026-09-05 by the multi-seed/multi-resolution sweep**
   (`scripts/probe_michelHydrostaticColumnSweep.py`, `PST_ALE_PLAN.md` §7.1,
   `nx ∈ {24,32,48} × 3 seeds × 4 modes`, jitter re-enabled from the case's
   own dead code path): the single seed above was not representative on the
   corner-velocity question. `michelShift` alone is not *categorically*
   worse than `noPenetrationShift` alone — at nx=32 its mean `‖v‖` peak
   (0.470) actually beats the safeguard's (0.612) — it is simply markedly
   less seed-stable (std up to 0.896 against the safeguard's 0.014-0.338,
   which tightens with resolution). What *does* hold at every resolution and
   seed: `pairedFraction` is lower under the shift than under `neither`, and
   **`both` together is the uniformly best and most seed-stable
   configuration** (lowest mean `‖v‖` peak at every `nx`, and — at 32/48 —
   the lowest variance too). So the safeguard's real value is consistency
   across configurations, not a categorically lower velocity than the shift
   — and the practical recommendation is "run both", not "the safeguard is
   still required because the shift alone underperforms it".
5. **`approachOnly=False` was added to `computeVelocityDiffusion`** rather than
   writing a new kernel. The default is unchanged, so no existing scheme moves.

   **This is half of `docs/historic_plans/DFSPH_IMPROVEMENT_PLAN.md`'s open item 1** (the
   shear-carrying Morris term), which asks for "full `v_ij` vector, *no
   approach-only clamp*". `approachOnly=False` removes the clamp — the term is
   now two-sided, which is what makes it Monaghan & Gingold (1983)'s velocity
   Laplacian and what Eq. (25) requires. It does **not** remove the normal
   projection: the contribution is still `mu_ij * gradW` with
   `mu_ij = (v_ij . x_ij)/|x_ij|^2`, a vector along `x_ij`, so there is still no
   tangential stress. A real Morris et al. 1997 laminar term needs the full
   `v_ij` vector and is still a separate, unwritten operator. Both papers use
   the normal-projected form, so ACSPH does not need it — but if that item is
   picked up, the clamp half is already done and gradchecked.

## Questions for the authors, all in one place

| Where | Question |
|---|---|
| §5.1 | Eq. (37) prints `ε₄ = min(0, κ₄ − ε₂)`, which makes the JST operator **vanish** in smooth flow. Standard JST is `max`. Which does the CUDA code do? |
| §5.2 | Eq. (40)'s low-storage form cannot represent Fig. 1's SSPRK3 or RK4 at all. Which is the code — Jameson coefficients, or the full tableaus? |
| §5.5 | `U_char` per case; the `𝕍` branch of Eq. (36) being unscaled. (The `β` interpolation question is closed — `michel2022` §5.3 says linear.) |
| §5.6 | Eq. (46)'s `CFL_t h` is a length. Is `h` there carrying an implicit reference velocity (which is what the term *does*)? And is the absence of a body-force constraint deliberate? |
| §5.7 | §4.4 prints `Δτ/Δt = 2` where Table 1, §4.5 and §3.1.3 all imply `Δt/Δτ = 2`. Transposition? |
| Part 3 | The `ψ` sign error above — does their δ-SPH reference implementation have it? |
| §4.1.1 | **AC-4 run standalone (not blended into AC-JST) diverges catastrophically for us** — pressure/velocity reaching `O(1e5-1e6)` within ~15 real steps on both `hydrostaticColumn` and a wall-free case (`rotatingSquarePatch`), nx=24. The paper reports only that AC-4 alone "struggles to maintain a converged kinetic energy" — bounded but non-convergent, not a blow-up — and separately that it resolves the hydrostatic free-surface profile fine at `t=50s`. Is the discrepancy resolution (our nx=24 vs. their finer grids), or something in `k2`/timestep tuning specific to AC-4 that isn't shared with AC-2L? We cross-checked the *formula* against the PDF character-by-character and again indirectly via AC-JST (which uses the identical AC-4 operator, scaled by `epsilon_4 <= kappa_4 = 1/32`, and stays bounded on the same cases) — confident the transcription is right, curious whether the severity we see is expected at coarser resolution or points at a parameter we're missing. **Re-checked 2026-10-13 with the current scheme (graph, D3, D5): unchanged** (\|v\| 1e17 by t = 0.005 at nx 24, also with D5 off); open hypothesis: `_biharmonic` uses `supports^2` (kernel support radius) where the paper's `h` is the smoothing length (`supports / xi`, as `acParameters` uses for beta and k2), i.e. `xi^2` too strong -- `scripts/probe_ac4Scale.py` tests it. |
| §4.3 | **Table 1/2's `CFL_t=0.4 -> 0.6` error cliff did not reproduce cleanly for us at reduced scale** (nx=24, one oscillation period only, vs. the paper's `L/Δx=200`-class, presumably multi-period runs): our `CFL_t=0.4` row was the *lowest*-error row measured, not a cliff edge, though the **cost** trend (`w`/`e` falling sharply with `CFL_t`) reproduced cleanly at every scale we tried. Is the accuracy cliff itself only clean at finer resolution / longer integration, or is there a specific measurement window (post-transient, particular oscillation number) Table 1's numbers are drawn from that we should match? |

---

## Why this is worth doing

The paper's own closing claim is the reason: *"in terms of software, it is clear
any weakly compressible δ-SPH code may be transformed to ACSPH by removing the
equation of state, specifying the artificial compressibility parameter k₁ and
adding a pseudo-time loop within the existing time loop."* This repo **is** a
δ-SPH code with all of that already built. The three things ACSPH needs that
δ-SPH does not — a pressure-evolution equation instead of an EOS, a BDF2
real-time source, and an inner pseudo-time RK loop — are the only genuinely new
code. Everything else (§4 below) is a rename or a config flag away.

Strategically it gives us a **third** incompressible baseline that is
structurally unlike both existing ones: DFSPH iterates a *pressure Poisson-like*
Jacobi solve on a velocity constraint; ACSPH iterates a *differential* equation
in pseudo-time. The paper's §2 argues these are the same thing solved two ways,
which makes it a genuinely informative comparison rather than a third opinion.

---

# Part 1 — Complete equation inventory

Numbering follows the paper. Symbols: `p` pressure, `v` momentum (Lagrangian)
velocity, `ρ` density (invariant, `= ρ₀`), `V_j = m_j/ρ_j` particle volume,
`h` smoothing length, `κh` kernel support radius, `x_ij = x_i − x_j`,
`(·)_ij = (·)_i − (·)_j`, `τ` pseudo-time, `t` real time, `n` real-time index,
`m` pseudo-time index, `s` RK stage index.

## 1.1 Governing system (continuous) — Eqs. (49)–(50)

The incompressibility constraint `∇·v = 0` is converted from elliptic to
hyperbolic by adding a pseudo-time derivative, and the momentum equation gains a
matching one:

```
Dp/Dτ  = −k₁ ρ ∇·v  +  k₂ 𝒟^p                       (continuity, Eq. 50/51)
Dv/Dτ  +  Dv/Dt  = −∇p/ρ + ν∇²v + f                 (momentum,   Eq. 50)
Dx/Dτ  +  Dx/Dt  = v                                (velocity,   Eq. 51)
```

`𝒟^p` is the pressure-smoothing operator (§1.4). It is *not* an ad-hoc artificial
viscosity: §2 derives it as the divergence of the momentum residual folded into
the pressure equation (Eq. 6), which is the same construction that produces the
ISPH pressure Poisson equation (Eqs. 10–15). That derivation is the paper's main
theoretical contribution and is worth reading in full, but nothing in it needs
implementing.

**Dropped by the paper itself:** the `k₃ D(∇·v)/Dτ` term of Eqs. (9)/(22) is set
to zero ("*during experimentation this third term was found to have little
influence*"). Implement the field, default it to 0, do not spend time on it.

## 1.2 Discrete continuity — Eq. (23)

```
Dp_i/Dτ = −k₁ ρ_i Σ_j (v_j − v_i)·∇W_ij V_j  +  k₂ 𝒟^p_i
```

Standard difference-form velocity divergence. Anti-symmetric, hence
volume-conserving (Eq. 52–53 discussion).

## 1.3 Discrete momentum and velocity — Eqs. (25)–(26)

```
Dv_i/Dτ + Dv_i/Dt = −(1/ρ_i) Σ_j (p_i + p_j) ∇W_ij V_j
                    + ν K Σ_j (v_ij·x_ij)/‖x_ij‖² ∇W_ij V_j
                    + f_i

ṽ_i := Dx_i/Dτ = v_i − Dx_i/Dt                                    (Eq. 26)
```

- Pressure gradient is the symmetric `(p_i + p_j)` form. The paper notes it is
  equivalent to `Σ m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij` under constant mass and
  invariant density — which holds here by construction.
- Viscosity is the Monaghan–Gingold (1983) velocity Laplacian, `K = 8` in 2D,
  `K = 10` in 3D.
- `ṽ` is the pseudo-time particle velocity. **On convergence `ṽ → 0`**, and it is
  simultaneously the position-row residual and the convergence metric (§1.6).

## 1.4 Pressure smoothing operators `𝒟^p` — Eqs. (32)–(37)

Four variants, named AC-2 / AC-2L / AC-4 / AC-JST:

**AC-2** — plain Laplacian of pressure (Molteni & Colagrossi form), Eq. (32):
```
𝒟^Δ_i = Σ_j 2(p_i − p_j) (x_ij·∇W_ij)/‖x_ij‖² V_j
```
Known-bad at free surfaces: kernel truncation diffuses the surface and it cannot
hold a hydrostatic gradient (§4.1.1 confirms this, Figs. 2–4). Implement it, but
it is a negative control, not a candidate default.

**AC-2L** — renormalised bi-Laplacian (Antuono correction), Eqs. (33)–(34).
**This is the paper's working default** and the operator used in every
head-to-head against δ-SPH:
```
𝒟^ΔL_i = Σ_j 2{ (p_i − p_j) − ½(⟨∇p⟩^L_i + ⟨∇p⟩^L_j)·x_ij } x_ij/‖x_ij‖² · ∇W_ij V_j

⟨∇p⟩^L_i = −Σ_j (p_i − p_j) L_i ∇_i W_ij V_j
L_i      = −[ Σ_j (x_i − x_j) ⊗ ∇_i W_ij V_j ]⁻¹
```

**AC-4** — nested (bi-harmonic) Laplacian, Eq. (35):
```
𝒟^Δ²_i = −h² Σ_j 2(𝒟^Δ_i − 𝒟^Δ_j) (x_ij·∇W_ij)/‖x_ij‖² V_j
```
Two neighbour loops, no correction. Inherits AC-2's truncation error but weaker
(§4.1.1: slight surface-row separation, volume perturbations, KE fails to fully
settle).

**AC-JST** — Jameson–Schmidt–Turkel blend, Eqs. (36)–(37):
```
𝒟^JST_i = 𝒟^ΔL_i                        if i ∈ 𝕍  (free-surface region)
        = ε₂ 𝒟^ΔL_i + ε₄ 𝒟^Δ²_i         otherwise

ε₂ = κ₂ min(1, χ),   ε₄ = min(0, κ₄ − ε₂),   κ₂ = 0.5, κ₄ = 1/32
χ_i = Σ_j |(p_i−p_j)/(p_i+p_j)| W_ij V_j  /  Σ_j W_ij V_j
```
`𝕍` = every particle within one kernel support radius of a free-surface particle.
**Not pairwise-symmetric, therefore not locally conservative** — the paper says
so explicitly and points at Lee et al. [51] for a conservative version.
See §5.1 for the `min`/`max` problem in ε₄.

**Frozen diffusion**: computed on the first RK stage and held across stages
(Antuono [8] / Jameson [40] technique). But **re-evaluated at every dual-time
iteration** — "*the diffusive terms are evaluated at each dual-time iteration
and cannot be fixed without loss of stability*".

## 1.5 Parameters — Eq. (24)

```
β = CFL_τ · h / Δτ        k₁ = β²        k₂ = 0.1 h β
```

`β` is the pseudo-time wave speed. In finite volumes `β` is prescribed and `Δτ`
varies locally; the paper **inverts this** — `Δτ` is a fixed fraction of `Δt`
(spatially constant, to keep particle displacements smooth) and `β` is the
derived variable.

`k₂ = 0.1 h β` is deliberately the δ-SPH `δ h c₀` prefactor with `β` playing the
role of `c₀` and `δ = 0.1`. The measured stability ceiling is `k₂ = 0.2 h β`,
consistent with Antuono's linear stability analysis [7,8]; `0.1` is kept for
consistency with δ-SPH practice.

`CFL_τ` is set by the pseudo-time integrator: **0.5 / 1.0 / 1.5 for RK2 / RK3 /
RK4**.

## 1.6 Dual-time integration — Eqs. (38)–(48)

State vector and residual (Eq. 38–39):
```
u = {p, x, v}ᵀ                 I_c = diag{0, 1, 1}
Du/Dτ + I_c Du/Dt = r          r* := Du/Dτ = r − I_c Du/Dt
```
The `0` in `I_c` is the whole point: the continuity equation has no real-time
derivative, so driving `r* → 0` enforces `∇·v = 0` *at* time level `n+1`.

**Pseudo-time RK sweep** (Eq. 40):
```
u^{n+1,m+1,0} = u^{n+1,m}
u^{n+1,m+1,s} = u^{n+1,m+1,0} + α_s Δτ r*^{,n+1,m+1,s−1},   s = 1..s_RK
u^{n+1,m+1}   = u^{n+1,m+1,s_RK}
```

**BDF2 real-time source** (Eq. 41), with variable-`Δt` coefficients (Eq. 42):
```
r*^{,n+1,m+1,s−1} = (1/α_PI) [ r^{n+1,m+1,s−1} − I_c( α_t u^{n+1,m+1,0} + β_t u^n + γ_t u^{n−1} ) ]

α_t = (2Δtⁿ + Δtⁿ⁻¹) / ((Δtⁿ + Δtⁿ⁻¹) Δtⁿ)
β_t = −(Δtⁿ + Δtⁿ⁻¹) / (Δtⁿ Δtⁿ⁻¹)
γ_t = Δtⁿ / ((Δtⁿ + Δtⁿ⁻¹) Δtⁿ⁻¹)
```
(Fixed-`Δt` limit: `α_t = 1.5/Δt`, `β_t = −2/Δt`, `γ_t = 0.5/Δt`.) Note the BDF
source is evaluated at the **frozen stage-0 value** `u^{n+1,m+1,0}`, not the
current stage.

**Point-implicit source treatment** (Eqs. 43–45): `α_PI = 1 + α_s Δτ α_t`,
applied to all three equations for temporal consistency. The paper then says
**`α_PI = 1` works fine here** ("*no noticed adverse behaviour*") because
`Δτ < Δt` and `Δτ` is spatially constant. Implement it (it is one scalar), keep
it on by default, expose the switch.

**Real timestep** (Eq. 46), with growth limiting to protect BDF2 accuracy:
```
Δtⁿ = max( min( CFL_t h , CFL_t h/‖v‖_max , 0.125 h²/ν , 1.2 Δtⁿ⁻¹ ) , 0.8 Δtⁿ⁻¹ )
```
`CFL_t ≈ 0.2`. §4.3 measures a sharp accuracy cliff above `CFL_t = 0.4`
(Table 1/2: error jumps ~2.4× from 0.4 → 0.6, ~10× at 1.0). Treat 0.4 as the
hard ceiling.

**Convergence metric** (Eqs. 47–48):
```
ε_v = log10( (1 / (N U_ε)) · sqrt( Σ_i^N ‖ṽ_i‖² ) )
U_ε = max( min(‖v‖_max, U_char), ε_s ),   ε_s = 1e−5
```
Iterate until `ε_v` drops below target. Recommended targets from the paper:
**−6 for general use, −8 for violent impact** (dam break, jet impact).

> ⚠ Note the `1/N` (not `1/√N`) — this is *not* an RMS. Per-particle residual
> `v̄` gives `ε_v = log10(v̄ / (√N U_ε))`, so a fixed `ε_v` target is a
> **looser per-particle tolerance at higher resolution**: the allowed RMS
> residual is `10^ε_v U_ε √N`. At fixed RMS residual the metric drops by
> `½log₁₀N`; across the paper's `L/Δx = 200 → 800` sweep that is ~0.6 decades. Reproduce
> it verbatim (it is what their numbers mean) but record it, and consider
> exposing a normalised variant as a non-default option.

**Butcher tableaus** (Fig. 1, image-only — transcribed here):

| RK2 (explicit midpoint) | RK3 (SSPRK3 / Shu–Osher) | RK4 (classical) |
|---|---|---|
| `c = [0, 1/2]` | `c = [0, 1, 1/2]` | `c = [0, 1/2, 1/2, 1]` |
| `A = [[0,0],[1/2,0]]` | `A = [[0,0,0],[1,0,0],[1/4,1/4,0]]` | `A = [[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]]` |
| `b = [0, 1]` | `b = [1/6, 1/6, 2/3]` | `b = [1/6, 1/3, 1/3, 1/6]` |

See §5.2 — these tableaus and Eq. (40) are mutually inconsistent for RK3/RK4.

**§4.3 finding: higher-order pseudo-time RK buys nothing.** Accuracy is set by
the BDF2, and cost rises near-linearly with stage count. Table 2 (all rows at
`Δt/Δτ = 5`) puts RK2 at or below RK3/RK4 on error at every `CFL_t` while
costing ~⅔ of RK3 and ~½ of RK4 — so **default to RK2**.

On the *ratio* Table 2 cannot speak (it holds `Δt/Δτ = 5` fixed); that is
Table 1's axis, and the paper's own sentence is *"for the same recorded error
as δ-SPH the solution cost can be matched or reduced, in the region of
`Δt/Δτ = 2` and `CFL_t = 0.4 − 0.6`"*. Table 1 also shows the trade directly:
raising `Δt/Δτ` from 2 to 20 buys ~10 % error and costs ~9× — so **2 is the
cost/accuracy point, 5–20 only if accuracy is the binding constraint.**
Lowest-error cell in either table is Table 2's RK2/`CFL_t = 0.2` row
(0.47/0.45), but its RK2/`CFL_t = 0.4` neighbour (0.50/0.48) is the same error
at half the cost.

## 1.7 Optional `ṽ` material-derivative correction — Eqs. (27)–(31)

Recasting `D̃(·)/Dτ = ∂(·)/∂τ + ṽ·∇(·)` adds advective terms (Ramachandran et
al. [29]):
```
D̃p_i/Dτ  += −p_i Σ_j (ṽ_j − ṽ_i)·∇W_ij V_j + Σ_j (p_j ṽ_j + p_i ṽ_i)·∇W_ij V_j
D̃v_i/Dτ  += Σ_j (ṽ_j ⊗ v_j + ṽ_i ⊗ v_i) ∇W_ij V_j − v_i Σ_j (ṽ_j − ṽ_i) ∇W_ij V_j
```

**The paper's conclusion is to leave these off** (§4.2): *"generally ignore the
ṽ material derivative corrections."* With them on, the minimum stable `ε_v`
degrades from −8 to −6 and residual `ṽ` leaves non-physical terms that blow up
at thin free-surface tips. The second momentum term violates both linear and
angular momentum conservation. Implement behind a flag, default off, do not
tune.

## 1.8 Particle shifting — Eqs. (55)–(58), Michel et al. [66]

```
δv*_i = 0.5 · { −U^shift_i β_i (κh) ∇C_i               if ‖β_i(κh)∇C_i‖ < 0.5 κh/Δx
              { −0.5 U^shift_i (κh/Δx) ∇C_i/‖∇C_i‖     otherwise

U^shift_i = max_j( |(v_i − v_j)·(x_i−x_j)/‖x_i−x_j‖| )
∇C_i      = Σ_j [ 1 + 0.2 (W_ij / W(Δx_i))⁴ ] ∇_i W_ij V_j
β_i       = (κh/Δx)³ in the interior, decreased to 1 for surface particles
```
Free-surface correction (Eq. 57):
```
δv_i = 0                                          if λ_i < 0.4
     = λ_i² ( δv*_i − σ_i (δv*_i·n_i) n_i )       if i ∈ 𝕍
     = δv*_i                                      otherwise

σ_i = min[ 1, max( 0, (κh − d^fs_i) / (0.5 κh) ) ]
```
`λ` = min eigenvalue of `L_i` (Eq. 34); `d^fs` = distance to nearest surface
particle.

Applied **outside** the pseudo-time loop as a displacement (Eq. 58):
```
x'_i = x_i + δv_i Δt ,   φ'_i = φ_i + ∇φ_i·(δv_i Δt)
```
§4.2 tested internal (per-pseudo-iteration, Eq. 60) vs external shifting and
**chose external**: internal shifting stalls `ṽ_δ → 0` convergence during strong
impacts and worsens volume conservation. Eq. (59) offers a BDF correction
`ṽ = v − Dx'/Dt + D(δx)/Dt` so the real-time derivative stays Lagrangian;
§4.2 found it makes little difference. Implement, default on, cheap.

The paper picked Michel et al. specifically because it has **no `c₀`/Mach
dependence** — which matters since ACSPH has no `c₀`. Our existing δ⁺ shift is
Mach-scaled (`modules/shifting/delta.py`), so this is a real gap (§4.2 below).

## 1.9 Boundary conditions — Eqs. (61)–(62), Adami et al. [68]

Fixed ghost particles. Boundary pressure by extrapolation:
```
p_b = Σ_f [ p_f + ρ_f (g − a_b)·x_bf ] W_bf V_f  /  Σ_f W_bf V_f
```
Other fields by Shepard interpolation with a mirroring condition:
```
f_b = Σ_f f_f W_bf V_f / Σ_f W_bf V_f
no-penetration:  f_b^⊥ = 2 f_p·n − f_b·n ,   f_b^∥ = f_b·τ
no-slip:         f_b^⊥ = f_b·n ,             f_b^∥ = 2 f_p·τ − f_b·τ
```
Applied per field: `δv` and `ṽ` get no-penetration (so `δv·n = 0`); `v` gets
no-penetration in the velocity divergence; the velocity Laplacian gets no-slip
or free-slip depending on the case.

Free surface: handled implicitly by the discretisation (Colagrossi et al. [69]);
detection by Marrone et al. [70] with Sun et al. [11] modifications, producing
the surface set `𝔽` and the dilated set `𝕍`.

## 1.10 Reference δ-SPH baseline — Eqs. (63)–(64)

The paper's comparison scheme is δ-ALE-SPH (Sun et al. [14] / Antuono et al.
[15]), i.e. **quasi-Lagrangian δ⁺-SPH with the shifting terms folded into the
equations of motion** — which is exactly what
`docs/historic_plans/WCSPH_SHIFTING_PLAN.md` delivered here
(`ShiftProperties.correctdrhodt` / `correctdvdt`). RK4, frozen diffusion,
`CFL_wc = 0.75`. Our existing `--scheme deltaSPH` with those flags on is the
right control.

---

# Part 2 — Every constant the paper fixes

| Symbol | Value | Where |
|---|---|---|
| `δ` (δ-SPH density diffusion) | 0.1 | Eq. (16) |
| `k₂` prefactor | `0.1 h β` (max stable `0.2 h β`) | Eq. (24) |
| `κ₂`, `κ₄` (JST) | 0.5, 1/32 | Eq. (37) |
| `CFL_τ` | 0.5 / 1.0 / 1.5 for RK2 / RK3 / RK4 | §3.1.3 |
| `CFL_t` | ~0.2; hard ceiling 0.4 | Eq. (46), Table 1–2 |
| `Δt/Δτ` | 2 (best cost/accuracy), 5–20 only if accuracy binds | Table 1, §1.6 |
| `ε_v` target | −6 general, −8 impact-heavy | §4.2, §4.4, §4.5 |
| `ε_s` | 1e−5 | Eq. (48) |
| `α_ν` (artificial viscosity) | 0.01, `ν = α_ν h c₀ / K` | §4 |
| `K` | 8 (2D), 10 (3D) | Eq. (25) |
| `λ` surface cutoff | 0.4 | Eq. (57) |
| Shift tensile term | `1 + 0.2 (W_ij/W(Δx))⁴` | Eq. (56) |
| Kernel | Wendland C2, `h/Δx = 2` | §4 |
| `α_PI` | `1 + α_s Δτ α_t`, or 1 | Eq. (41) |
| `k₃` | 0 (dropped) | §3.1 |

## The settings the paper actually ran each case with

The Part 2 table above is the paper's *general* recommendation; §4 overrides it
per case, and a faithful reproduction has to match the per-case row, not the
general one. Read off §4.1–§4.5 directly:

| Case | Operator | RK | `CFL_τ` | `CFL_t` | `Δt/Δτ` | `ε_v` | Resolution | Walls |
|---|---|---|---|---|---|---|---|---|
| §4.1.1 hydrostatic column | AC-2/2L/4/JST (the comparison) | RK2 | **0.3** | — | 5 | **−7** | — (read at `t = 50 s`) | rigid base |
| §4.2 rotating square patch | AC-2L (Fig. 15); all four (Fig. 16) | — | — | — | — | −8 → −6 sweep (Fig. 13) | `L/Δx = 200/400/800` | none |
| §4.3 oscillating droplet | AC-2L | RK3 (Table 1), RK2/3/4 (Table 2) | — | 0.1–1.0 sweep | 2–20 sweep | — | — | none |
| §4.4 jet impact | AC-2L | **RK4** | — | 0.4 | **2** (printed `Δτ/Δt = 2`, see §5.7) | **−8** | `L/Δx = 200` | none; `H = 2L`, `L = 1 m`, `U = 1 m/s`, `ρ₀ = 1000`, `c₀ = 100` (δ-SPH only) |
| §4.5 dam break | AC-2L | **RK3** | — | 0.4 | 2.0 | — | `H/Δx = 400` (2D), `75` (3D) | **free-slip**; geometry from Buchner [81] + Lobovský [82], displacement shifting |

One deviation from what this repo currently runs: `§4.4`'s case is `H = 2L`,
which in `impactCase`'s parameterisation is `aspectRatio = 0.25` (§4.4's own
probe headline used `0.5`, i.e. `H/L = 1` — Marrone's own figure case, not
De Courcy's).

`§4.5`'s **free-slip** walls used to be a second one — `dambreak`'s `wallBC`
defaulted to `'constant'`, which adds no *physical* viscosity term but still
drags the fluid against a `v = 0` wall through the `AllToAll` **artificial**
viscosity, i.e. an effective no-slip bed. Every reference this case family is
graded against specifies free slip (De Courcy §4.5, Marrone 2011 §3,
Lobovský/Buchner), so **the default was flipped to `'freeSlip'` on 2026-09-21**;
`DELTASPH_VALIDATION_PLAN.md` §5.7 has the A/B that motivated it and §0.3 of
this plan's to-do list has the blast radius.

> ⚠ **`ν` still needs a nominal `c₀`.** The paper says ACSPH "*does not require
> the definition of c₀*" and then defines `ν = α_ν h c₀ / K` with `α_ν = 0.01`,
> using "*the same value of ν*" as the δ-SPH run. So a reference `c₀` is still
> needed as an input to fix the physical viscosity — it just never enters the
> scheme. Our config must make this explicit rather than silently reusing
> `fluid.fixedSoundSpeed` as if it were an acoustic parameter.

---

# Part 3 — What the repo already has

This is the good news: most of the paper's spatial discretisation is a config
away.

| Paper | Repo | Status |
|---|---|---|
| Eq. (25) pressure gradient `(p_i+p_j)` | [surfaceAware.py](src/warpSPH/modules/pressure/surfaceAware.py), `PressureForceScheme.nonConservative` | ✅ direct (the literal `(p_i+p_j)`; `Antuono` is the tensile switch, decision 2) |
| Eq. (25) Monaghan–Gingold viscosity | [velocityDissipation.py](src/warpSPH/modules/deltaSPH/velocityDissipation.py), `ViscosityTerms.MonaghanGingold1983` | ✅ direct |
| Eq. (23) velocity divergence | [modules/momentum/](src/warpSPH/modules/momentum/), `WarpOperation.Divergence` | ✅ direct |
| Eq. (34) `L_i`, `⟨∇·⟩^L` | [gradRhoL.py](src/warpSPH/modules/density/gradRhoL.py), `computeRenormalizationMatrices` | ✅ **generalised 2026-09-05** (`computeGradRhoL(field=)`) |
| Eq. (33) bi-Laplacian ψ operator | [wp_densityDelta.py](src/warpSPH/modules/deltaSPH/wp_densityDelta.py) `DensityDiffusionScheme.deltaSPH` | ✅ **generalised 2026-09-05**; sign error found + fixed — see note |
| Eq. (61) value term | [wallPressure.py](src/warpSPH/modules/incompressible/wallPressure.py) `'shepard'` mode | ✅ exact |
| Eq. (61) body-force term | `wallPressureExtrapolation(bodyForce=)` | ✅ **built 2026-09-05**, §4.4 |
| Eq. (62) Shepard + mirroring | [modules/mdbc/velocity.py](src/warpSPH/modules/mdbc/velocity.py) | ✅ **audit closed 2026-09-20**, §4.4 (`ṽ` is a documented, measured deviation) |
| §3.3 free-surface detection (Marrone/Sun) | [maronneDetection.py](src/warpSPH/modules/surfaceDetection/maronneDetection.py) + [dilation.py](src/warpSPH/modules/surfaceDetection/dilation.py) | ✅ gives `𝔽`, `𝕍`, `n`, `λ` |
| Eq. (57) `λ < 0.4` surface gating | `ShiftingProjectionScheme.surfaceNormal`, `surfaceLambdaThreshold` | ✅ direct |
| Eq. (46) adaptive `Δt` | [timestep/artificialCompressible.py](src/warpSPH/modules/timestep/artificialCompressible.py) | ✅ **recast 2026-09-05**, §4.5 / §5.6 |
| §3.4 δ-ALE-SPH baseline | `--scheme deltaSPH` + `correctdrhodt`/`correctdvdt` | ✅ direct |
| RK2/SSPRK3/RK4 tableaus | `warpSPHIntegrators.getButcherTableau` (`'midpoint'`, `'SSPRK3'`, `'RK4'`) | ✅ reuse tableaus |
| Test cases (§4.1–4.5) | `hydrostaticColumn`, `rotatingSquarePatch`, `oscillatingDroplet`, `impact`, `dambreak` | ✅ all five exist |

### The one that matters most: Eq. (33) ≡ our existing δ-SPH ψ

> ⚠ **The A/B this section asked for found a sign error in the existing δ-SPH
> kernel.** Fixed 2026-09-05, see "The sign error" below. Everything in this
> subsection describes the operator *after* that fix.

`wp_densityDelta.py`'s `deltaSPH` branch computes
`ψ_ij = −(∇ρ^L_i + ∇ρ^L_j) − 2(ρ_j−ρ_i) x_ij/‖x_ij‖²`, dotted with `∇W_ij` —
the **unprojected** Marrone 2011 form (its Eq. 6, read off the PDF:
`ψ_ij = 2(ρ_j−ρ_i) r_ji/‖r_ij‖² − [⟨∇ρ⟩^L_i + ⟨∇ρ⟩^L_j]` with
`r_ji = r_j − r_i = −x_ij`). The paper's Eq. (33) writes the **projected**
form, gradient term contracted onto `x̂_ij` first.

These are algebraically identical whenever `∇W_ij ∥ x_ij`, which holds for any
isotropic kernel:
```
((∇p_i+∇p_j)·x̂)(x̂·∇W) = W'(r) (∇p_i+∇p_j)·x_ij / r = (∇p_i+∇p_j)·∇W
```

**Measured** (`scripts/probe_deltaSPHPsiProjection.py`, float64, 20×20 jittered
lattice, rms over interior rows, against a from-scratch `O(N²)` torch reference
that shares no code with the kernel):

| claim | linear `f` | quadratic | cubic |
|---|---|---|---|
| warp kernel − torch reference (unprojected) | 6e−14 | 2e−13 | 3e−13 |
| unprojected − projected, renormalisation **off** | 2e−15 | 2e−15 | 2e−15 |
| unprojected − projected, renormalisation **on** (relative) | **1.00** | 0.54 | 0.32 |

So **reusing the existing kernel reproduces AC-2L exactly**, with the caveat
confirmed quantitatively: they diverge completely once
`useGradientRenormalization` puts an `L_i` in front of `∇W` (then `L∇W ∦ x_ij`).
Assert that path off for ACSPH. Note the probe also shows *which* form survives
that: the projected one still annihilates a linear field with `L` on (its brace
`{(p_i−p_j) − ½(⟨∇p⟩_i+⟨∇p⟩_j)·x_ij}` is a scalar that vanishes regardless of
what multiplies `∇W`), while the unprojected one does not. If renormalised
gradients on this operator ever become desirable, implement Eq. (33) literally.

### The sign error (found by that A/B, fixed 2026-09-05)

`ψ` had the gradient term's sign flipped — `+grad − rho` where Marrone Eq. (6)
is `−grad − rho` — in all four gradient-carrying branches (`deltaSPH`,
`deltaOnly`, `denormalized`, `denormalizedOnly`). `densityOnly`
(Molteni–Colagrossi, no gradient term) was and is correct.

The defining property of the Antuono correction is that the two terms **cancel
pair-by-pair on a field linear in space**: for `f = a·x`,
`(∇f_i+∇f_j)·∇W = 2a·∇W` and `2(f_j−f_i)(x_ij·∇W)/r² = −2a·∇W` identically.
That is what promotes the Molteni–Colagrossi Laplacian to a *bi*-Laplacian and
lets the diffusive term reach a free surface without eating the smooth field
underneath. With the sign flipped the terms *added*: `deltaSPH` was numerically
`2 × densityOnly` on any smooth field — a second-order diffusion twice as
strong as the uncorrected one, in place of the fourth-order one. Measured
before/after on the linear field, rms over interior rows:

| | linear | quadratic |
|---|---|---|
| `deltaSPH`, before | 1.6e+00 | — |
| `deltaSPH`, after | **1.0e−13** | 1.8e−02 |
| `densityOnly` (unchanged, the control) | 6.2e−01 | 4.1e+00 |

It never blew up — the sign was still diffusive — which is exactly why only a
property test catches it. Pinned as `tests/test_deltaSPHDiffusion.py` (4 tests,
including the specific shape of the bug: `deltaOnly` came out *equal* to
`densityOnly` rather than its negative). Full suite green after the fix.

**This is not cosmetic for either scheme.** For ACSPH it is the whole of
§4.1.1: AC-2L is separated from AC-2 precisely by whether the operator can hold
a hydrostatic — i.e. linear — pressure gradient, and before the fix ours could
not. For WCSPH it means the repo's default δ-SPH density diffusion has been
over-strong and second-order, actively diffusing exactly the density gradients
it exists to preserve. **Worth reporting to the authors' circle / checking
against diffSPH, which this kernel was ported from.**

**Measured end-to-end on `sloshingTank --scheme wcsph`**, nx=100, `t ≤ 3.6 s`,
today's code both sides, the only difference being the sign (the pre-fix
operator is reproduced *exactly* by negating the gradient field handed to the
kernel, so this is a clean single-variable A/B — see
`scripts/probe_deltaSPHPsiSignAB.py`):

| | pre-fix ψ | post-fix ψ |
|---|---|---|
| diverged | no | no |
| min density over the run | 0.396 | **0.532** |
| max density over the run | 1.849 | 1.837 |
| peak \|p\| at Sensor 1, `t > 2 s` | 39.2 kPa | **32.5 kPa** |

(Measured band for the first impact peak: 2.2–13.1 kPa.) So the fix improves
both the density floor and the pressure overshoot, and degrades nothing. Note
neither is *good* — the case has separate open problems recorded in
`examples/sloshingTank/PLAN.md`, and the peak is still ~2.5× the band top. For
reference the run recorded on 2026-09-03 (`examples/sloshingTank/output/`)
**diverged** at `t = 3.41` with `ρ ∈ [0, ∞]`; that is older code, not this
variable, and is why the A/B above was run rather than compared against it.

---

# Part 4 — What has to be built

Ordered by how much is genuinely new.

## 4.1 The dual-time driver — *entirely new, the core of the work*

`schemes/incompressible/artificialCompressible.py`. Owns the whole real-time step:

```
for m in 0..maxPseudoIters:
    u0 = u                                   # freeze stage-0 for the BDF source
    compute frozen 𝒟^p at stage 0            # §1.4 frozen diffusion
    for s in 1..s_RK:
        r  = spatial residual at u^{s−1}     # Eqs. 23, 25, 26
        Dx_Dt = α_t x0 + β_t x^n + γ_t x^{n−1}
        Dv_Dt = α_t v0 + β_t v^n + γ_t v^{n−1}
        r* = (1/α_PI) [ r − I_c·(Dx_Dt, Dv_Dt) ]     # I_c zeroes the p row
        u^s = u0 + α_s Δτ r*
    u = accumulate(b_s)                      # see §5.2 on Eq. (40) vs Fig. 1
    ṽ = v − Dx_Dt ;  if ε_v(ṽ) < target: break
apply shifting displacement (Eq. 58) + BDF correction (Eq. 59)
roll history: (x^{n−1},v^{n−1}) ← (x^n,v^n) ← (x^{n+1},v^{n+1})
```

**Framework integration — recommended approach.** The runner drives steps via
`ctx.integrator.function(state, f=ctx.stepFunction, dt=...)`
([runner.py:288](src/warpSPH/runner/runner.py#L288)), which does not fit a
scheme that owns its own time advance. Rather than adding a `dualTime`
integrator to `warpSPHIntegrators` (which would couple a general library to one
scheme), use the **exact-delta trick**: have `acsph_step` run the full dual-time
solve and return
```
dxdt = (x^{n+1} − x^n)/Δt ,  dvdt = (v^{n+1} − v^n)/Δt ,  dpdt = (p^{n+1} − p^n)/Δt
```
Forward Euler on an exact delta is the identity, so the runner reproduces the
converged state byte-for-byte with zero framework changes. Pin
`config.integrationScheme = forwardEuler` and **validate it at build time** —
a silent RK2 here would run the whole dual-time solve twice per step and blend,
which is wrong and not obviously wrong. (Precedent for this class of trap:
`dambreak.py`'s note that `divergenceFree` needs `semiImplicitEuler` and
"nothing in the code enforces this yet". Do enforce it.)

**BDF history storage.** `x^n, x^{n−1}, v^n, v^{n−1}` are needed. Put them on the
*system*, not the state — `WeaklyCompressibleSystem` already carries non-state
fields (`adjacency`, `t`, `domain`), which is the established precedent and
avoids `initializeNewState` cloning them per stage. Startup: step 0 has no
`u^{n−1}`, so fall back to BDF1 (`α_t = 1/Δt, β_t = −1/Δt, γ_t = 0`) for the
first step.

## 4.2 Michel et al. (2022) shifting law — new

Our shift is δ⁺-SPH (Sun 2017 scaling, Mach-dependent):
`modules/shifting/delta.py`'s docstring already notes *"An equivalent
shifting-velocity form (Michel 2022 scaling) is present in a comment but not
used."* ACSPH has no `c₀` and no Mach number, so this must be built:
`U^shift` (Eq. 56), the `β = (κh/Δx)³` interior scaling with surface decay to 1,
the two-branch magnitude clamp (Eq. 55), and the `σ` ramp (Eq. 57).

The surface machinery it needs (`𝕍`, `n`, `λ`, `d^fs`) is all already produced by
`detectFreeSurface`. Add as `ShiftingScheme.michel2022`; it is independently
useful to the existing WCSPH scheme, so it should live in `modules/shifting/`,
not inside the ACSPH scheme.

## 4.3 Generalised scalar-field diffusion operators — modify existing

~~`wp_densityDelta.py` reads `ρ_i`/`ρ_j` from `getParticle(referenceState, j)`,
i.e. bound to the state's density field. Add an optional
`queryField`/`referenceField` tensor pair...~~ **Done 2026-09-05.** The pair is
threaded through the existing `ExtraSpec` mechanism, guarded so both or neither
must be supplied, and the volume weight `m_j/ρ_j` is deliberately left on the
density (it is a quadrature weight, not the diffused quantity).
`computeScalarFieldDiffusion` is the raw-operator entry point;
`computeGradRhoL(field=)` supplies Eq. (34). AC-2 and AC-2L now exist with no
new kernel. Gradcheck covers the new branch. See Part 3 for the sign error this
work uncovered.

Then add, new:
- **AC-4** (Eq. 35): a second pass over the AC-2 output. Trivial once AC-2 is a
  reusable scalar-field operator.
- **AC-JST** (Eqs. 36–37): the `χ` switch is one extra interpolation loop; the
  blend is elementwise. Needs `𝕍` from the dilated surface mask.

> ⚠ `modules/deltaSPH` is covered by the `gradcheck` skill. Touching this kernel
> means running `/gradcheck deltaSPH` before and after.

## 4.4 Boundary conditions — one real gap, plus an audit

`modules/incompressible/wallPressure.py` is closer than it first looks:

- **Eq. (61)'s value term is already exact.** Its `'shepard'` mode is literally
  `p_b = Σ_f V_f p_f W_bf / Σ_f V_f W_bf`.
- ~~**Eq. (61)'s body-force term is the gap.**~~ **Done 2026-09-05.**
  `wallPressureExtrapolation(..., bodyForce=g)` now adds
  ```
  p_w = [ Σ_f p_f W_wf + (g − a_w)·Σ_f ρ_f r_wf W_wf ] / Σ_f W_wf ,   r_wf = r_w − r_f
  ```
  (`adami2012` Eq. 27) on the `'shepard'` **and** `'mirror'` closures.
  `'shepard' + bodyForce` is De Courcy's Eq. (61) exactly — Adami weights by
  `W_wf` alone where Eq. (61) weights by `W_bf V_f`, and the two agree here
  because ACSPH is density-invariant so `V_f = V₀` cancels. `'mls'` raises
  rather than taking it: its Liu–Liu linear fit already carries the local
  pressure gradient, so adding the correction would double-count.

  *Implementation.* The vector moment `Σ_f V_f ρ_f (r_w − r_f) W_wf` is its own
  kernel, [wp_wallMoment.py](src/warpSPH/modules/incompressible/wp_wallMoment.py),
  because it is not any single `WarpOperation`. It takes `x_ij` from
  `computeDistanceVec` like every other operator in the package, so it is
  minimum-image correct by construction.

  ~~*Restriction, deliberate.*~~ **Lifted — superseded by the moment kernel.**
  The first version assembled the moment from two `Interpolate` gathers,
  `r_w·Σ_f V_f ρ_f W_wf − Σ_f V_f ρ_f r_f W_wf` (legitimate because `(g − a_w)`
  is a per-*wall* quantity and comes out of the sum). That split discards the
  minimum-image convention, so a wrapping pair contributed `±L_d` of error per
  periodic direction `d`, which cost a guard, a case-level `periodic = False`
  override, and a `hydrostaticColumn` run that died at step 45. `wp_wallMoment.py`
  removed all three; `tests/test_wallPressure.py`'s last three tests pin the
  equivalence, the minimum-image difference, and the lifted restriction.

  *Verified* by `tests/test_wallPressure.py`: for a pressure field linear in
  space every neighbour's contribution `p_f + ρ_f g·(r_w − r_f)` is already
  the analytic wall value, so the weighted average is exact regardless of how
  truncated the wall neighbourhood is. Measured on a 24×24 column over three
  wall rows: **corrected 2.0e−7 relative error (float32 machine precision),
  plain Shepard 1.3e−1** — i.e. the uncorrected wall under-reads by up to
  `3 Δx · ρ₀ g`, 13 % of the whole column's pressure drop. That is precisely
  the error that stops a hydrostatic column from holding.

  Note the wall-acceleration `a_b` still has no per-particle source anywhere in
  the codebase (the same gap `modules/mdbc/velocity.py` documents for the
  velocity mirror's dead `2 u_wall` term) — static walls make `a_b = 0`, which
  covers every case in Part 7, but a moving-wall ACSPH case would need it. The
  `(N, dim)` `bodyForce` form is already accepted, so wiring a source is all
  that would be left.

  **Wired into ACSPH on 2026-09-05** (`schemes/incompressible/artificialCompressible.py`'s
  `wallPressures`): recomputed at every RK stage from the current fluid
  pressure, `clampNonNeg=False` because ACSPH's pressure is a solved field that
  legitimately goes negative. Without it the wall reads `p = 0` (the non-fluid
  rows are masked every step) and the column simply falls out of the box.
  Verified exact against the analytic profile on the real case — see step 5b.

  Independently a DFSPH improvement: no DFSPH caller passes `bodyForce` yet, so
  it stays additive there, but it is exactly the term
  `docs/historic_plans/DFSPH_IMPROVEMENT_PLAN.md` Part 23 needs for a gravity-driven wall.
- **A structural note**: these live under `modules/incompressible/`, i.e. they
  are DFSPH-facing. ACSPH needs them too, so either relocate to a shared module
  or import across. Prefer relocating — a third consumer makes the current home
  misleading.

Eq. (62)'s Shepard + no-penetration/no-slip mirroring largely exists in
`modules/mdbc/`. The audit item: **ACSPH extrapolates three velocity-like fields
where WCSPH extrapolates one** — `v` (no-penetration in the divergence, no-slip
or free-slip in the Laplacian), `ṽ` (no-penetration), `δv` (no-penetration, so
`δv·n = 0`). Confirm each gets the right condition rather than inheriting `v`'s.

**Audit closed 2026-09-20**, field by field:

- **`v` (the wall velocity): correct, per BC type.** `computeBoundaryVelocities`
  (`modules/mdbc/velocity.py`) runs once per real step (driver step 02) and
  writes the boundary rows through the ghost mapping: `freeSlip` is
  `u_g = u_body + w_t − w_n` — the **reflected** normal, i.e. the published
  no-penetration form of Eq. (62); `noSlip` is `u_body − w_t` ((0,−1), the
  deliberate deviation — `docs/historic_plans/DFSPH_IMPROVEMENT_PLAN.md` Part 9 measured the
  reflecting form as worse on the bounded DFSPH case); `constant` leaves the
  row unchanged; `extended` is the Liu–Liu MLS extrapolation. The reflect
  fix is from `786e300` (2026-09-11) — **every walled number this plan
  recorded before that date predates it** (re-measured 2026-09-20, below).
  The `dambreak` case runs `constant`, which for its static wall is
  `u_g = 0` — an effective no-slip bed, the same condition the delta-SPH
  Marrone reference runs used, so the A/B isolates the scheme.
- **`δv` (the particle shift): satisfied trivially.** `solveShifting`
  (driver `finalize`, Eq. (58)) zeroes the shift for `kinds != 0`
  (`modules/shifting/wrapper.py`), so the wall is never shifted: `δv·n = 0`
  holds identically rather than being imposed.
- **`ṽ` (the pseudo-time velocity): not extrapolated — documented deviation,
  measured.** The driver integrates x/v/p on **all** rows through the
  dual-time loop: wall rows carry the full SPH residual (pressure force,
  viscosity, and `bodyForce` — gravity — is not kind-masked), so their
  velocity and position wander away from the BC values *inside* the loop.
  The wander is discarded at hand-off: step 08 zeroes the non-fluid
  `dxdt/dvdt/dpdt`, so at every real-step boundary the wall state is exactly
  the BC value, and the time-level deviation is zero for a static wall.
  What remains is that the fluid solves against the wandered wall during the
  solve. Magnitude, measured 2026-09-20 on `hydrostaticColumn` (nx=32, 1 s,
  static freeSlip container, `.tmp/probe_acsphWallDrift.py`, spies on
  `wallPressures`/`convergenceMetric` capturing every in-loop wall slice):
  the **position** excursion is at most **0.046 `Δx`** over the whole run
  (1.45e-3 m against `Δx` = 3.1e-2 m, 1.31e6 in-loop samples) — the wall
  stays pinned to within half a particle spacing. The in-loop wall
  **velocity** does spike (up to 58 m/s, median 10 m/s, measured against the
  step-start BC value — the freeSlip BC is a position-based construction that
  itself changes as the near-wall fluid moves, so the absolute figure
  overstates the "true" wander), but the BDF2 step-ratio makes the net
  position change over the loop negligible, and step 08 resets it exactly at
  hand-off. **Verdict: not worth fixing.** Re-imposing the BC on the wall
  rows inside the loop (zeroing the normal increment per iteration) is a
  small change, but the geometric quantity the fluid actually solves against
  — the wall *position* — never wanders more than half a spacing, so the
  documented deviation is benign. If a future walled case shows a pressure or
  force artefact traceable to the in-loop wall *velocity*, clamping that is
  the targeted change; there is no evidence for it on this case.

## 4.5 Timestep — recast

Eq. (46) is close to but not the same as `modules/timestep/weaklyCompressible.py`:
the acoustic constraint is replaced by an advective one (`CFL_t h/‖v‖_max`), the
viscous constraint is `0.125 h²/ν`, and there is a symmetric growth/shrink clamp
(`[0.8, 1.2]×`) that exists specifically to protect BDF2 accuracy. `Δτ = Δt / R`
with `R` a config constant.

## 4.6 New scheme family: config, state, system, wiring

Per the user's expectation, ACSPH gets its own family rather than being bolted
onto WCSPH:

- **`enumTypes.py`**: new `ArtificialCompressibleSPHScheme` enum (single member
  `artificialCompressible`), plus `PressureSmoothingScheme` (`laplacian`,
  `renormalizedBiLaplacian`, `biharmonic`, `jst`) mapping to AC-2/2L/4/JST.
- **`systems/artificialCompressible.py`**: `ArtificialCompressibleState`. The
  key structural difference from `WeaklyCompressibleState` — **`pressures`
  becomes an integrated field** (`integrated('dpdt')`) and `densities` becomes
  `constant` at `ρ₀`. Carries `surfaceIndicators`/`surfaceNormals`/
  `surfaceLambdas` and the ghost bookkeeping unchanged.
  `ArtificialCompressibleSystemUpdate` = `{dxdt, dvdt, dpdt}`.
  `ArtificialCompressibleSystem` carries the BDF history (§4.1).
- **`configurations/artificialCompressible.py`**:
  `ArtificialCompressibleSPHConfig`, modelled on `WeaklyCompressibleSPHConfig`
  (fluid, viscosity, BCs, shifting, regions, rigid bodies, surface detection,
  gravity) with the EOS/`densityDiffusion` block swapped for a new
  `acParams`: `{ pressureSmoothing, CFL_tau, CFL_t, dtOverDtau, rkStages,
  epsilonV, epsilonS, uChar, maxPseudoIterations, minPseudoIterations,
  k2Factor (=0.1), kappa2 (=0.5), kappa4 (=1/32), k3 (=0.0),
  usePointImplicit (=True), useTildeVAdvection (=False),
  shiftInsidePseudoLoop (=False), bdfShiftCorrection (=True),
  referenceSoundSpeedForViscosity }`. Plus the round-trip
  `artificialCompressibleConfigToDict` / `dictTo...` pair.
- **Registration touchpoints** (mirroring the `WeaklyCompressibleSPHScheme`
  surface): `schemes/builder.py` (`SchemeBundle`), `io/parsers.py`,
  `io/export.py`, `io/importIO.py`, `runner/caseSpec.py` (enum sweep at
  [caseSpec.py:304](src/warpSPH/runner/caseSpec.py#L304), `schemeNames`), `warpSPH/__init__.py`,
  `modules/timestep/wrapper.py` (dispatch on system type).

---

# Part 5 — Ambiguities and errors in the paper

Flagging these matters for two reasons: they must be resolved before coding, and
(given we work with the authors) they are worth reporting back.

## 5.1 Eq. (37): `ε₄ = min(0, κ₄ − ε₂)` — almost certainly should be `max`

Verified against the rendered page (p. 9), so this is not a text-extraction
artefact. As printed with `κ₄ = 1/32` and `ε₂ = 0.5 min(1, χ) ≥ 0`:
`κ₄ − ε₂ ≤ 1/32`, so `min(0, ·) ≤ 0` always. In smooth flow (`χ → 0`) it gives
`ε₂ = 0, ε₄ = 0` and the JST operator **vanishes entirely** — the exact opposite
of the stated design ("*fourth-order dissipation in smooth regions*"). Standard
JST (Jameson–Schmidt–Turkel 1981) is `ε₄ = max(0, κ₄ − ε₂)`.

**Decision: implement `max`.** Expose the paper-literal `min` behind a flag so
the discrepancy is reproducible, but do not default to it. Ask the authors.

## 5.2 Eq. (40) and Fig. 1 are mutually inconsistent for RK3/RK4

Eq. (40)'s update `u^s = u^0 + α_s Δτ r*(u^{s−1})` is the Jameson low-storage
form. It can only represent tableaus whose `A` is non-zero on the sub-diagonal
only *and* whose `b` equals the final stage row. The RK2 midpoint tableau in
Fig. 1 satisfies this (`α = {1/2, 1}`). **SSPRK3 and classical RK4 do not** —
SSPRK3 has `a₃₁ = a₃₂ = 1/4`, and RK4's `b = [1/6,1/3,1/3,1/6]` is not any stage
row. So either Fig. 1 is decorative and the code uses Jameson coefficients
(`{1/4,1/3,1/2,1}` for 4 stages), or Eq. (40) is a simplification and the code
uses the full tableaus.

**Decision: implement the general explicit Butcher form** (reuse
`warpSPHIntegrators.butcher` tableaus), which reproduces Fig. 1 exactly and
degenerates to Eq. (40) for RK2. Since §4.3 concludes RK2 is the best operating
point anyway, the ambiguity is largely academic — but it must be a deliberate
choice, not an accident. Ask the authors which the CUDA code does.

## 5.3 Eq. (30) has a stray `h`

Eq. (30) reads `+ k₂ h 𝒟^p_i` (verified on the rendered page 8) while Eqs. (23),
(51) and (54) all read `+ k₂ 𝒟^p_i`. Dimensional analysis settles it: `k₂ = 0.1hβ`
already carries the length scale, `𝒟^p ~ [p]/L²`, so `k₂𝒟^p ~ [p]β/L ~ [p]/T` ✓
and the extra `h` is wrong. Typo in Eq. (30) only; use the `k₂ 𝒟^p` form.

## 5.4 `𝕍` is used for two different things — **closed, it is one set**

`𝕍` appears in both Eq. (36) (JST switching) and Eq. (57) (shifting), and the
original worry was that the same symbol might carry two different dilation
radii. Re-read against the PDF, it does not: the paper defines it the same way
both times, in the same words.

- after Eq. (36): *"where `𝕍` is the free-surface region, any particle within a
  kernel support radius to a free surface particle"*
- before Eq. (57): *"Any particle within the free surface vicinity `𝕍` (within
  the kernel support radius of a surface particle)"*

Same `𝔽`, same one-support dilation, one set. **Resolved as written — no
separate knob.** The implementation matches: both consumers read the single
`currentState.surfaceIndicators` that `schemes/incompressible/artificialCompressible.py`
computes once per real step, dilated by the shared
`SurfaceDetectionConfig.expansionIterations`. A second, JST-only dilation count
would be inventing a degree of freedom the paper does not have; if a future
measurement wants one, that is a deliberate extension, not a transcription
choice.

## 5.6 Eq. (46)'s first term is dimensionally a length

Verified against the rendered page 10, so not an extraction artefact:

```
Δtⁿ = max( min( CFL_t h , CFL_t h/‖v‖_max , 0.125 h²/ν , 1.2 Δtⁿ⁻¹ ) , 0.8 Δtⁿ⁻¹ )
```

`CFL_t h` is a **length**, not a time. What it *does*, though, is unambiguous
from the structure — paired with the next entry it is exactly

```
min( CFL_t h , CFL_t h/‖v‖_max )  ==  CFL_t h / max(1, ‖v‖_max)
```

i.e. the advective constraint with its denominator floored at one. In a code
whose velocities are O(1) that floor is a reference velocity of 1 left implicit,
which is precisely the missing dimension.

**Decision: implement it that way**, with the floor named
(`REFERENCE_VELOCITY`) rather than hidden as a bare `1`. Ask the authors whether
their `h` there carries such an implicit reference velocity.

**This is not pedantry — it is load-bearing.** Without that floor *nothing*
bounds `Δt` in a quiescent case: `‖v‖_max → 0` makes the advective term
infinite, and an inviscid run makes the viscous term infinite too, so `Δt`
climbs 1.2× every step to `config.maxDt`. Measured on `hydrostaticColumn`:
`Δt` went 5e−4 → 6.4e−3 in fifteen steps and the near-wall velocity error grew
with it. With the floor, `Δt` settles at 4.9e−3 and stays there.

**Separately**, Eq. (46) has no body-force constraint, which every other δ-SPH
timestep in this repo and in the literature carries and which a gravity-driven
case needs. `CFL_t √(h/‖a‖_max)` is implemented behind the existing
`dt_accelerationConstraint` flag; turn it off for the paper's literal set. Also
worth asking about.

## 5.5 Under-specified

- **`U_char`** in Eq. (48) is never given a definition per case. It is the
  case's own characteristic velocity, and every ACSPH-wired case now supplies
  one: `√(g·fillRatio·L)` (column, dam break), `A·R` (droplet), `ω·size`
  (patch), `impactVelocity` (jets), `uMag` (TGV), and — for `randomFlow`, which
  is band-limited noise with no closed-form scale — the seeded field's own
  `|v|_max`. **Not** a required field: `acParams.uChar` stays `Optional` and
  `_convergenceMetric` falls back to the instantaneous `|v|_max`, so a new case
  runs without one. But the fallback is a moving target — it drifts with the
  flow, so a fixed `ε_v` means different things at different times — which is
  exactly the failure mode §3.1.3 describes for a stationary dam break. Set one
  per case.
- **The `𝕍` branch of Eq. (36)** returns `𝒟^ΔL` *unscaled*, i.e. `ε₂ = 1`
  implicitly at the surface, while the interior uses `ε₂ 𝒟^ΔL + ε₄ 𝒟^Δ²`. That
  is a discontinuity in the operator at the `𝕍` boundary. Presumably intended
  (the text says the bi-Laplacian is "activated at the free surface"), but worth
  confirming.
- ~~**`β` in Eq. (57)'s surface decay**~~ — **answered from `michel2022`
  itself** (its §5.3), which De Courcy is restating: "`β_i = (R/Δx)³` for the
  inner particles with a **linear** decreasing in the free-surface region to
  reach `β_i = 1` for the free-surface particles". Not `σ`, not `λ²`. See
  `PST_ALE_PLAN.md` §2.3.
- **Symbol collision**: `β` is both the AC wave speed (Eq. 24) and the shifting
  scaling `(κh/Δx)³` (Eq. 56). Unrelated quantities. Use distinct names in code.

## 5.7 §4.4 prints `Δτ/Δt = 2` where everything else is `Δt/Δτ`

§4.4's jet-impact setup reads *"setting `Δτ/Δt = 2` and `CFL_t = 0.4`"*, the
reciprocal of the ratio every other section uses. It is almost certainly a
transposition: Table 1 sweeps `Δt/Δτ ∈ {2, 5, 10, 20}`, §4.5 writes
`Δt/Δτ = 2.0`, and §3.1.3 is explicit that `Δτ` is *"set at a fixed fraction of
`Δt`"* with `Δτ < Δt` — which `Δτ/Δt = 2` contradicts outright.

**Decision: read it as `Δt/Δτ = 2`.** Low-risk (it is the ratio both other
cases use) but worth one line to the authors alongside the §5.1/§5.2/§5.6
questions.

---

# Part 6 — Cited literature

## 6.0 Status

**Every blocking reference is now in `literature/`.** On 2026-09-05 the target
paper and seven of its references were synced in (bib keys `decourcy2024`,
`antuono2010`, `antuono2012`, `letouze2013`, `michel2022`, `ramachandran2021`,
`lobovsky2014`, `marrone2015`), each identified from its own front matter,
verified field-by-field against its DOI record, and abstracted verbatim —
`scripts/check_literature.py` passes.

Four more the ACSPH discretisation leans on were **already here**: `marrone2011`
(δ-SPH, Eqs. 16–17), `sun2017` (δ⁺-SPH), `sun2019` (consistent shifting / the
δ-ALE baseline), `adami2012` (wall BC, Eq. 61). Also present: `cummins1999` [2].

Nothing in Parts 1–5 or Part 7 is now blocked on a document we do not have.

## 6.1 Obtained for this plan

| Ref | Key | Unblocks |
|---|---|---|
| — | `decourcy2024` | The scheme itself. |
| [8] | `antuono2012` | The `k₂ ≤ 0.2hβ` stability bound; the bi-Laplacian interpretation of Eq. (33); frozen diffusion; why Eq. (32) fails at free surfaces. |
| [7] | `antuono2010` | Origin of the corrected Laplacian; co-cited for the stability bound. |
| [76] | `letouze2013` | Square-patch **initial pressure field** (a Poisson solve — §4.2 cannot be initialised without it), the analytic stretching solution, and BEM/LDFM reference data. |
| [82] | `lobovsky2014` | Dam-break probe geometry and the 2.5%/97.5% experimental bounds (Figs. 28/30). Supplementary Materials carry the raw signals. |
| [80] | `marrone2015` | The analytic incompressible KE drop the jet-impact case (§4.4) is scored against. |
| [66] | `michel2022` | The shifting law of §4.2 — its derivation of `β = (κh/Δx)³` and its PST-conditions checklist, which is also worth auditing our existing shift against. |
| [29] | `ramachandran2021` | Cross-check `1/α_PI = 2Δt/(2Δt+3Δτ)` (the paper's own §3.1.3 form, for `α_s = 1`) and the `ṽ` material derivative. Closest prior art, **with an open-source reference implementation**. |

## 6.2 Still not obtained — non-blocking, the paper reproduces the equations in full

| Ref | Paper | Note |
|---|---|---|
| [40] | Jameson, Schmidt, Turkel (1981), AIAA-81-1259 | Would settle §5.1 (`min` vs `max` in ε₄) definitively. |
| [70] | Marrone, Colagrossi, Le Touzé, Graziani (2010), *Fast free-surface detection and level-set function definition*, JCP 229(10) 3652–3663 | Already implemented here (`maronneDetection.py`); the citation is missing from the library, not the code. |
| [77] | Monaghan & Rafiee (2013), IJNMF 71(5) 537–561 | ~~Not obtained~~ — **in `literature/` as `monaghan2013`** (`monaghan2013_multi-fluid-high-density-ratios.pdf`, bib + abstract). Appendix A is the droplet ODE `oscillatingDroplet.analyticSolution` integrates. |
| [6] | Molteni & Colagrossi (2009), CPC 180(6) 861–872 | ~~Not obtained~~ — **in `literature/` as `molteni2009`**. AC-2 (Eq. 32) is given in full in De Courcy anyway. |
| [15] | Antuono, Sun, Marrone, Colagrossi (2021), *δ-ALE-SPH*, C&F 216 104806 | ~~Not obtained~~ — **synced 2026-09-05** as `antuono2021`, for `PST_ALE_PLAN.md`. |
| [50],[54],[55] | Monaghan & Gingold 1983; Bonet & Lok 1999; Randles & Libersky 1996 | Standard operators, already implemented. |
| [19] | Sun, Pilloton, Antuono, Colagrossi (2023), *Acoustic damper term in WCSPH*, JCP 483 112056 | The competing "fix WCSPH instead" approach — interesting for the comparison narrative. |
| [32] | De Courcy et al., SPHERIC 2023 | The precursor; may carry implementation detail cut from the journal version. |
| [21],[23],[25],[39],[28],[27],[31],[26] | Chorin 1997; Turkel 1987; McHugh & Ramshaw 1995; Dupuy 2020; Clausen 2013; Ramachandran & Puri 2019; Chola & Shintake 2021; Rouzbahani & Hejranfar 2017 | §2 theory context only. No implementation content. |
| [60] | Vila 1999 | ~~§2 theory context only~~ — **synced 2026-09-05** as `vila1999`, for `PST_ALE_PLAN.md` stage D, where it is the scheme rather than the context. |

## 6.3 Out of scope

[72] Fourey et al. (2017) and the FSI chain [33–35], [73–75], [85] support §4.1.2
(elastic-base hydrostatic column) and Appendix A (modal structural solver + RBF
coupling). **This repo has no structural solver**, so §4.1.2 is not reproducible
and Appendix A needs nothing. Skip.

---

# Part 7 — Validation plan

All five of the paper's cases already exist here, which is unusually lucky.

| § | Case | Repo case | Measures | Reference |
|---|---|---|---|---|
| 4.1.1 | Hydrostatic column, rigid base | `hydrostaticColumn` | Hydrostatic profile, free-surface integrity, KE decay | Analytic gradient |
| 4.1.2 | Hydrostatic column, elastic base | — | FSI energy dissipation | **Skip** — no structural solver |
| 4.2 | Rotating square patch | `rotatingSquarePatch` | KE decay, centre pressure, momentum conservation | BEM/LDFM [76] |
| 4.3 | Oscillating droplet | `oscillatingDroplet` | IRMSE(KE), IRMSE(semi-major axis), cost | Analytic [77] ✅ already encoded |
| 4.4 | Normal impact of 2D jets | `impact` | Instantaneous KE drop, pressure smoothness | Analytic [80] |
| 4.5 | Dam break (2D + 3D) | `dambreak` | 4 wall pressure probes, KE | Experiment [82] |

**Ordering.** §4.1.1 first — it is the operator discriminator (it is what
separates AC-2 from AC-2L/AC-JST) and it is cheap. Then §4.3, which is the only
case with a clean analytic score and is the paper's own parameter-sweep vehicle
(reproduce Tables 1 and 2 — they are the single best acceptance test for the
dual-time machinery). Then §4.2 for conservation, then §4.4/§4.5.

**Acceptance targets from the paper:**
- AC-2L, **AC-4** and AC-JST all hold a hydrostatic gradient with no
  free-surface diffusion; AC-2 visibly fails. AC-4 is separated from the other
  two not on the *profile* (Fig. 2) but on a slight first-surface-row
  separation (Fig. 3c) and a KE that "struggles to maintain a converged" value
  (Fig. 4). (§4.1.1.) **Our AC-4 does not clear even the profile bar** — it
  diverges outright, which is a stronger failure than the paper reports; see
  step 8 and the authors' question on §4.1.1.
- Paper's settings for this case: `CFL_τ = 0.3`, `Δt/Δτ = 5`, `ε_v = −7`, RK2,
  read at `t = 50 s`.
- Table 1/2 reproduce qualitatively: error flat for `CFL_t ≤ 0.4`, ~2.4× jump at
  0.6; cost linear in `Δt/Δτ` and in RK stage count; RK order buys no accuracy.
- Square patch: KE loss 25–32% *less* than δ-SPH across `L/Δx = 200/400/800`;
  **zero visible pressure oscillation at every resolution**; cost 2.4–2.8× δ-SPH.
- Jet impact: correct KE drop within a few time steps, no oscillatory ringing;
  total cost ≤ 1.5× δ-SPH.
- Dam break: noise-free `P1` through the void-closure event at `t√(g/H) ≈ 8.4`
  where δ-SPH's acoustic noise swamps the signal.

**Cost metric.** The paper's effective cost `e = ∫₀ᵗ (m_iter · s_RK / Δt) dt` (Eq. 65) is
implementation-independent and should be recorded alongside wall time — it is
how their numbers are quoted and the only fair way to compare against our δ-SPH.

---

# Part 8 — Sequencing

1. ~~**Literature sync.**~~ **Done 2026-09-05** — the paper and seven references
   synced, checker green (§6.0). Remaining optional follow-ups: add the §6.2 set
   to `literature/EXPANSION_CANDIDATES.md`, and consider promoting `marrone2011` from the
   extended set to the core (it is cited throughout Parts 1 and 3 but carries no
   abstract, the same case that promoted `dehnen2012` on 2026-09-04).
2. ~~**Finish the Adami `bodyForce` term** (§4.4).~~ **Done 2026-09-05** —
   `wallPressureExtrapolation(..., bodyForce=...)` on the `'shepard'` and
   `'mirror'` closures, exact on a linear pressure field to float32 machine
   precision (`tests/test_wallPressure.py`, 8 tests — the last three cover
   `wp_wallMoment.py`). See §4.4 for the moment kernel that replaced the
   original two-gather decomposition and lifted its periodic-axis restriction.
3. ~~**Generalise the diffusion kernel to an arbitrary scalar field** (§4.3), and
   A/B the projected vs unprojected ψ form.~~ **Done 2026-09-05.**
   - `computeDensityDiffusionDeltaSPH` takes an optional
     `queryField`/`referenceField` pair; the volume weight `m_j/ρ_j` is
     untouched (it is quadrature, not the diffused quantity). New public entry
     `computeScalarFieldDiffusion` (no `schemeConfig`, no prefactor);
     `computeDensityDiffusion` is now its δ-SPH specialisation.
     `computeGradRhoL` takes `field=` — with the pressure it is Eq. (34)
     verbatim.
   - `/gradcheck deltaSPH` green before and after; the script now runs every
     `DensityDiffusionScheme` twice, once through the field pair, so the new
     branch's adjoint is covered.
   - The A/B (`scripts/probe_deltaSPHPsiProjection.py`) confirmed the Part 3
     equivalence to 2e−15 with renormalisation off, quantified the divergence
     with it on (100 %/54 %/32 %), **and found the ψ sign error** — see §3's
     "The sign error". Full suite green after the fix.
4. ~~**Scaffold the new family** (§4.6).~~ **Done 2026-09-05.** `--scheme
   artificialCompressible` resolves, builds, and runs a step end to end through
   the real integrator; the step is a deliberate no-op
   (`schemes/incompressible/artificialCompressible.py::PHYSICS_IMPLEMENTED = False`, warns
   once on entry) with a marked socket where step 5's driver goes.
   - `ArtificialCompressibleSPHScheme` + `PressureSmoothingScheme` +
     `isArtificialCompressibleScheme` in `enumTypes.py`.
   - `systems/artificialCompressible.py`: state with `pressures` **integrated**
     and `densities` **constant**; system carrying the BDF history plus
     `rollHistory` and `bdfCoefficients` (Eq. 42, with the BDF1 first-step
     fallback reported through an `order` return).
   - `configurations/artificialCompressible.py`: `ArtificialCompressibilityParams`
     (every Part 2 constant, `uChar` and `referenceSoundSpeedForViscosity`
     `Optional` on purpose) + the round-trip pair.
   - Registered in `schemes/builder.py`, `io/parsers.py`, `io/export.py`,
     `io/importIO.py`, `runner/runner.py::_resolveScheme`,
     `runner/caseSpec.py::schemeNames`, `warpSPH/__init__.py`.
   - **The integrator trap is enforced**, not documented:
     `validateIntegrationScheme` raises on anything outside
     `_EXACT_DELTA_INTEGRATORS` (`forwardEuler` / `explicitEuler`) at step entry, because the exact-delta hand-off is silently wrong under a
     multi-stage integrator (it would run the whole dual-time solve per stage
     and blend). `modules/timestep/wrapper.py` likewise raises rather than
     letting an ACSPH system fall through to the acoustic timestep.
   - The scaffold tests (originally `tests/test_artificialCompressibleScaffold.py`,
     14 tests; since folded into `tests/test_artificialCompressible.py`). Note
     `test_variableStepBdf2DifferentiatesAQuadraticExactly`: the fixed-step
     limit alone would not catch a swapped `Δtⁿ`/`Δtⁿ⁻¹` in Eq. (42).
5. ~~**The dual-time driver** (§4.1) with AC-2L and RK2 only.~~ **Done
   2026-09-05.** `schemes/incompressible/artificialCompressible.py` now runs the full
   Eqs. (38)–(48) loop: frozen stage-0 BDF source, frozen-per-iteration `D^p`,
   a general explicit Butcher RK sweep, `I_c = diag{0,1,1}`, point-implicit
   `α_PI`, and the `ε_v` convergence test on `ṽ`. `ṽ` advection, internal
   shifting and `k₃` raise rather than silently no-opping; AC-4/AC-JST raise
   pointing at step 8.
   - **New supporting work.** `modules/artificialCompressible/pressureSmoothing.py`
     (AC-2 / AC-2L dispatch onto `computeScalarFieldDiffusion`);
     `approachOnly=False` on `computeVelocityDiffusion`, which turns its
     `inviscid=False` branch into the Monaghan–Gingold velocity Laplacian
     proper — the clamp that made it one-sided is an artificial-viscosity
     device, and Eq. (25) has no such clamp. Both `nu * rho0` (for the kernel's
     `mean(ρ)` division) and the `1/ρ_i` on the pressure gradient are applied
     explicitly, so nothing here assumes `ρ₀ = 1` the way δ-SPH does.
   - **Measured** (`tests/test_artificialCompressible.py`, 24×24 periodic box,
     Taylor–Green plus a compressive perturbation, so only the solve is
     graded): `rms(∇·v)` **1.84 → 2.6e−3**, monotone in the iteration budget
     and flat between 100 and 400 iterations (converged). §4.3's finding
     reproduces: RK3/RK4 buy no accuracy over RK2 at equal iteration count and
     cost linearly more.
   - **Still open on the driver itself:** convergence to `ε_v = −6` took ~400
     iterations from that (deliberately extreme) initial transient. Whether
     that is the initial condition, `Δt/Δτ`, or something real is exactly what
     step 6's Table 1/2 reproduction answers — do not tune it before then.
   - **Validated on `hydrostaticColumn`** (§4.1.1, the paper's own first case),
     see step 5b below for the wiring that took and what it measures.
5b. **`hydrostaticColumn` under ACSPH** — done 2026-09-05.

   *Wiring.* `initializers/weaklyCompressible.py::initializeState` now builds
   any of the three state classes from one construction (it had a
   character-identical `if`-branch per class and no `else`, so an unknown class
   fell through to a `NameError`); `cases/common/weaklyCompressible.py` gains
   `configureDomain` + `configureArtificialCompressible`, which also **forces
   `forwardEuler`** (loudly) since the exact-delta contract requires it;
   `hydrostaticColumn` branches on `isArtificialCompressibleScheme` and takes
   the *raw* hydrostatic seed rather than the mean-shifted one — ACSPH has no
   pressure gauge, so the shift is a half-column-drop error at the free
   surface, not a gauge choice.

   ~~*The domain is made non-periodic on this branch.*~~ **Withdrawn — see
   decision 3.** The reason it was is worth keeping: the original two-gather
   wall-pressure moment was not minimum-image safe, so once a fluid particle
   drifted within a support radius of the bottom face it became a wrapped
   neighbour of the *top* wall and the moment picked up a whole domain height
   along gravity, and `wallPressureExtrapolation` refused rather than returning
   a wrong wall pressure. `modules/incompressible/wp_wallMoment.py` replaced
   that decomposition with a real kernel taking `x_ij` from
   `computeDistanceVec`, so the moment is minimum-image correct by construction.
   The case is back on the shared `periodic=True` default
   ([hydrostaticColumn.py:441](src/warpSPH/cases/boundaries/hydrostaticColumn.py#L441)) and
   needs no guard, no restriction and no case-level workaround.

   *Measured — the scheme is right.* With `p` seeded analytically and `v = 0`
   (`scratchpad` probe, nx=32):
   - Adami wall pressure vs analytic at 199 wall rows: **max error 0.0000**
     against a column pressure drop of 4.905. The Part-2 work does exactly its
     job.
   - Bulk momentum residual: **‖r_v‖ = 0.0079 against g = 9.81**, i.e. the
     discrete pressure gradient balances gravity to 0.08 %. The hydrostatic
     state *is* a discrete equilibrium for this scheme.
   - `r_p = 0` in the bulk; density exactly invariant, as it must be.

   *What it still lacks: the shift (step 7).* Run forward, the column holds its
   pressure profile (`p ∈ [0.31, 4.87]` against an analytic drop of 4.5) but
   develops a near-wall velocity error concentrated in the **bottom corners**,
   and the free surface drifts down slowly.

   The Eq. (46) advective floor (§5.6) was needed to keep `Δt` bounded at all —
   without it `Δt` ran 1.2× per step to `maxDt` and the corner error grew with
   it. Beyond that, the corner behaviour is measured **both ways**, because the
   difference is exactly the size of the hole step 7 fills:

   | | `‖v‖_max` peak | worst corner particle | verdict |
   |---|---|---|---|
   | as the paper has it (no safeguard) | 2.9 | `x = -0.62` | inside the wall band; fluid is leaving the box |
   | `noPenetrationShift = True` | 0.29 | `x = -0.497` | bounded, wall plane at `-0.5` holds |

   `noPenetrationShift` is **off by default** and is *not* the particle shift —
   it is the repo's mDBC wall safeguard, applied as an acceleration the way
   `deltaSPH_step` applies it. The paper has neither, because it always has the
   shift. The 200-step figures below were taken with it on.

   **Re-measured 2026-09-20, after the `786e300` freeSlip fix** (same probe,
   nx=32, 200 steps, one seed — the full four-mode table, with the
   pairing/void metrics the two-mode table above dropped):

   | mode | `‖v‖_max` | `pairedFraction` | `nnDistP01` | `voidFraction` | `dispMax` | KE | `t` |
   |---|---|---|---|---|---|---|---|
   | neither | **6.77** | 0.000 | 0.759 | 0.027 | 0.659 | 1.44e−1 | 0.408 |
   | `noPenetrationShift` | **0.77** | 0.145 | 0.046 | 0.000 | 0.250 | 4.7e−3 | 0.913 |
   | `michelShift` | 2.35 | 0.000 | 0.862 | 0.012 | 0.482 | 1.14e−2 | 0.616 |
   | both | 2.54 | 0.000 | 0.826 | 0.020 | 0.250 | 1.34e−2 | 0.796 |

   The freeSlip fix changed the picture sharply. The old ghost velocity
   `u_g = u_body − w_t` carried **no normal reflection**; the published form
   `u_g = u_body + w_t − w_n` does. So `neither` went from 2.9 to **6.77**:
   the old BC let fluid leak out the corners at moderate speed, the correct one
   reflects it back at high speed — the failure mode moved from "leaving the
   box" to "violent corner oscillation" (and the adaptive `Δt` shrinks to
   match, which is why 200 steps now reach only `t = 0.408`).
   `noPenetrationShift` stays the best `‖v‖` bound (0.77, bounded,
   `voidFraction 0`) but its pairing worsened (0.065 → 0.145). `michelShift`
   is essentially unchanged (2.33 → 2.35) — the shift is insensitive to the
   wall-velocity sign. And `both` (2.54) is now **worse than the safeguard
   alone** on `‖v‖` (and marginally worse than `michelShift` alone), so the
   pre-fix "both is the uniformly best configuration" reading (from the
   jittered-lattice sweep) does **not** carry over to this regular lattice once
   the wall BC is correct.

   *200 steps, nx=32, to `t = 0.94` (the case's full `tLimit`):* no divergence,
   density **exactly 1.000** throughout (invariant by construction, as it must
   be), `voidFraction 0`, `neighbourCountCV` flat at 0.24, `‖v‖_max` 0.58
   (peak 0.73), KE 6.1e−3, `dispMax` 0.27. The one clearly bad number is
   **`pairedFraction` 0.065** with `nnDistP01` down to 0.32 — particle pairing,
   which is exactly and only what particle shifting exists to prevent.

   So: **step 7 (Michel et al. shifting) is the next action** — was, as of
   this writing; it has since landed and this case has been re-measured
   (`scripts/probe_michelHydrostaticColumn.py`, `PST_ALE_PLAN.md` §7.1, after
   fixing a units bug found in the process — `computeMichelShift` was
   applying Eq. (22)'s shifting *velocity* with no `dt` factor). `pairedFraction`
   moved, cleanly, to exactly 0.000 with the shift on; `‖v‖_max` did not carry
   the corners on the shift alone the way the paper's phrasing implied it
   should — `noPenetrationShift = False` with the shift active still peaks
   `‖v‖` around 2.3, against 0.75 for the safeguard alone. See decision item 4
   above: this overturns the "shift replaces the safeguard" expectation
   rather than confirming it.

   *One loose end:* the case's own `pressureSlope`/`pressureSlopeRatio` figures
   of merit stop being reported once the run develops (`hydrostaticDiagnostics`
   returns early when its bulk band has fewer than 8 rows). They report fine on
   the first steps — `pressureSlopeRatio 0.79` after three. The direct probe
   above is the stronger measurement anyway (the *residual* against `g`, not a
   fit), but the band gate is worth understanding before quoting this case's
   published axes for ACSPH.

6. **Timestep + convergence control** (§4.5, §1.6). Reproduce Tables 1 and 2 on
   `oscillatingDroplet`. This is the real acceptance gate for the machinery.
   - **Eq. (46) landed 2026-09-05**: `modules/timestep/artificialCompressible.py`,
     wired into `modules/timestep/wrapper.py`'s dispatch. Advective in place of
     acoustic, `0.125 h²/ν` viscous, the symmetric `[0.8, 1.2]×` step-ratio
     clamp, and a `CFL_t > 0.4` warning (Tables 1–2's measured cliff). Eq. (46)
     as printed is dimensionally impossible — see the new §5.6 — so its first
     two terms are implemented together as
     `CFL_t h / max(REFERENCE_VELOCITY, ‖v‖_max)`, with `REFERENCE_VELOCITY = 1`
     a named module constant rather than a hidden floor. (Not to be confused
     with the *separate* body-force constraint `CFL_t √(h/‖a‖_max)`, which
     Eq. (46) does not have at all and which sits behind
     `dt_accelerationConstraint` — see §5.6's second half.)
   - **The Table 1/2 sweep mechanism landed 2026-09-05**
     (`scripts/probe_acsphOscillatingDropletTable1.py`, `oscillatingDroplet`
     wired for ACSPH, Monaghan & Rafiee (2013)'s analytic solution re-derived
     and verified — see the status board). Cost trend (`w`/`e` falling
     sharply with `CFL_t`) reproduces cleanly at a reduced scale (nx=24, one
     period). **Still open**: the error-side accuracy-cliff trend at
     paper scale (finer resolution, more periods) — not yet reproduced
     cleanly at this reduced scale.
7. **Michel shifting** (§4.2). Validate on `rotatingSquarePatch`.
   → **Split out into `PST_ALE_PLAN.md`** (2026-09-05). Reading `michel2022`
   properly, this is not one law: it is a set of requirements, an audit of every
   PST in the literature against them, the new law, and three SPH schemes to
   validate it on — two of which (Vila's ALE and Parshikov & Medin) this repo
   does not have, and both of which need a Riemann solver it also does not have.
   That plan's **stage A is this step** — **closed** (status board: law,
   free-surface treatment, `finalize` wiring, Fig. 1 convergence rate, §5.4
   free-surface rate, the `hydrostaticColumn` sweep and the `sloshingTank`
   fix all landed 2026-09-05). Stages B′/B/C/D are the new scope it opens and
   remain queued behind this plan's own remaining items, per the user's
   sequencing.
8. **Remaining operators**: AC-2, AC-4, AC-JST. Reproduce Fig. 2 / Fig. 16.
   → **Done, all four operators implemented, 2026-09-05.** AC-2/AC-2L from
     step 4/5; AC-4 and AC-JST land together in this pass, and AC-JST turned
     out to double as AC-4's own validation.
   - **AC-4** (`PressureSmoothingScheme.biharmonic`, no new kernel — two
     calls to `computeScalarFieldDiffusion`) formula verified character-by-
     character against the PDF, but **run alone** it diverges within ~15
     real steps on both `hydrostaticColumn` and `rotatingSquarePatch`
     (nx=24) — pressure/velocity reaching `O(1e5-1e6)`.
   - **AC-JST** (`PressureSmoothingScheme.jst`) needed one genuinely new
     warp kernel — `modules/artificialCompressible/wp_jstSwitch.py`, Eq. (37)'s
     `chi_i` switch (a pairwise nonlinear function of *both* `p_i` and `p_j`,
     which nothing in `computeScalarFieldDiffusion`'s per-particle-field
     family can express) — gradchecked
     (`scripts/gradcheck_jstSwitch.py`, w.r.t. both pressures and positions,
     plus a constant-field sanity check giving exactly `chi=0`). §5.1's
     `min`/`max` ambiguity resolved as already decided: `epsilon_4 = max(0,
     kappa_4 - epsilon_2)` by default, `acParams.jstUsePrintedMin` for the
     paper-literal `min` (which the plan had already predicted zeroes the
     operator in smooth flow). The dilated free-surface set `𝕍`
     (`currentState.surfaceIndicators`, already computed once per real step
     by `schemes/incompressible/artificialCompressible.py`) gates AC-2L-alone at the
     surface vs. the interior blend elsewhere, no new surface-detection work
     needed.
   - **AC-JST cross-validates AC-4, and settles the question its own
     instability raised**: run on `hydrostaticColumn`, `rotatingSquarePatch`
     *and* `oscillatingDroplet` (nx=24, 60 real steps each), AC-JST stays
     bounded on all three (`‖v‖` peaks 0.26-5.0) despite using AC-4's exact
     same operator internally. `epsilon_4` caps AC-4's contribution at
     `kappa_4=1/32` (vs. standalone AC-4's implicit `epsilon_4=1`) and zeroes
     it entirely once the smoothness switch `chi_i > kappa_4/kappa_2 =
     1/16` — throttling the ~20x per-pass amplification measured on AC-4
     alone by at least that factor of 32 wherever it matters. **This
     confirms the AC-4 formula itself is correct** (a genuine sign/coefficient
     bug would still show up scaled down in the JST blend, not vanish) —
     the standalone blow-up is a real, resolution-sensitive property of
     running it unblended, matching (if more severely than) the paper's own
     "AC-4 struggles to maintain a converged kinetic energy" report against
     AC-JST's clean behaviour. AC-4 stays flagged not-production-ready
     standalone; AC-JST is the one to reach for if AC-4's fourth-order
     smooth-region behaviour is wanted.
   - Full test suite green throughout.
   - **Fig. 2's operator comparison — reduced-scale reproduction done,
     qualitatively matches.** `scripts/probe_acsphPressureSmoothingComparison.py`
     on `hydrostaticColumn`, nx=48. At 80 steps (`t~0.17`) all three
     surviving operators (AC-4 diverges immediately here too, as already
     known) look similar — too early for the paper's own discriminator to
     show, which is measured at `t=50s`. **At 300 steps (`t~0.31`) the
     expected separation appears**: `pressureSlopeRatio` (1.0 = exact
     hydrostatic gradient) is **AC-2 0.721** (down from 0.824 at 80 steps —
     actively *degrading*, exactly the "cannot hold a hydrostatic gradient"
     failure the paper describes), **AC-2L 0.783**, **AC-JST 0.995** (up
     from 0.809 — converging toward the exact gradient). AC-JST clearly
     beats both AC-2 and AC-2L here, plausibly because it gets AC-2L's own
     treatment at the free surface *plus* AC-4's higher-order correction in
     the smooth interior. `pressureResidual` (noise, independent of slope)
     tells a different, consistent story: AC-2 0.051 (worst), AC-2L 0.035
     (best), AC-JST 0.043 (between) — slope accuracy and noise are separate
     axes, as the case's own diagnostic docstring says. **Not the paper's
     `t=50s`/Fig. 2 numbers themselves** (far more real time than tractable
     here), but the qualitative claim — AC-2 fails, AC-2L/AC-JST hold the
     gradient — reproduces cleanly at this reduced scale, and the trend
     (AC-2 worsening, AC-JST improving, over the same time window) is
     itself informative, not just the endpoint values.
   - **`rotatingSquarePatch` vs. BEM (Fig. 15) — deferred, not attempted.**
     `letouze2013`'s BEM reference for the *square* patch (as opposed to
     the *circular*-patch case in the same paper, which has a closed-form
     analytic solution) is read off a scatter plot in their own figure —
     genuine digitized data, not a formula. Reproducing it validly needs
     those numbers, which this pass did not extract (no digitization tool
     available, and eyeballing points off a rendered PDF figure risks
     silently encoding wrong numbers as "the reference" — exactly the kind
     of error this repo's culture is built to avoid). A qualitative
     sanity check (does the patch grow arms, does energy decay monotonically,
     no blow-up) is a much lower bar than what Fig. 15 actually validates,
     so this is left open rather than substituted for one.
9. **Impact and dam break** (§4.4, §4.5), including the `𝒞_e` cost metric.
   - **Both cases wired for ACSPH 2026-09-05** (`impact`, `dambreak` — see
     the status board): smoke-tested with real physics (body collision,
     column collapse under gravity, zero wall penetration), full test suite
     green.
   - **`impact` vs. Marrone et al. 2015 — attempted, real formula, resolution-
     limited result.** Found their exact closed-form (`literature/marrone2015`
     Appendix A Eq. A.34): the instantaneous KE ratio `Ek(0+)/Ek(0-)` for two
     identical rectangular jets impacting head-on, as a function of aspect
     ratio `r = π·L/(2H)`. Maps cleanly onto `impactCase`'s own parameters —
     `size=H`, and (since each jet's near edge sits at the impact plane once
     `touching=True` closes the gap) `r = π·aspectRatio` directly, no unit
     conversion needed. Self-check against the paper's own two limits
     (`r→0` gives ratio `→0`, `r→∞` gives `→1`) passes exactly
     (`scripts/probe_acsphImpactEnergyLoss.py`).
     Measured (`aspectRatio=0.5`, i.e. `r=π/2`, analytic ratio 0.499, target
     `KE=0.2495` from `KE(0-)=0.5`): the naive "value at first contact"
     reading is a poor proxy — 0.84 at nx=24 — because KE does not drop in
     one instantaneous step; it declines over dozens of steps with the
     contact `gap` oscillating (repeated bounce/re-contact), not a single
     inelastic collision. **But at nx=64, tracking the full KE trajectory
     rather than just the first-contact instant, `KE` passes almost exactly
     through the analytic target**: `0.2492` at step 94 (vs. the target
     `0.2495`, 0.1% off) — then keeps declining afterward (`0.229` by step
     149), consistent with the paper's own remark that the true incompressible
     solution keeps evolving past `t=0+` as "thin horizontal jets develop"
     and carry away further energy, which the idealised snapshot theory
     does not capture and a real simulation necessarily will. **Reads as a
     genuine, encouraging sign** — resolution matters a lot here (nx=24's
     crude reading was 68% too high; nx=64 crosses the target almost
     exactly) — but "the instant to compare against" has no clean operational
     definition in a continuously-evolving simulation, so this is not a
     clean pass/fail validation, just real evidence the scheme is in the
     right regime once resolved enough.
     The paper's own reference (LS-FVM) shows the same qualitative
     complication — "thin horizontal jets develop, that requires a proper
     spatial resolution to be captured" — at their `L/Δx=400`; ours tops out
     far below that. **Open**: this needs the paper's own resolution scale
     (or at least a convergence sweep clearly trending to the analytic
     value) before treating any single number here as validated — logged as
     a real, resolution-limited attempt, not a pass.
   - **`dambreak` vs. Lobovsky et al. 2014 — harness built, not yet run.**
     Their pressure bounds (Table 2, `H=300mm`: median/2.5%/97.5% percentile
     peak pressure at 4 wall sensor heights) are genuine experimental
     statistics from 100 repeated trials, quoted at their own physical tank
     scale, so reproducing them validly needed (a) the tank/column geometry at
     their `mm` scale rather than this repo's historical Koshizuka & Oka
     proportions, (b) per-height wall pressure probes, and (c) their own
     non-dimensionalisation (`t* = t√(g/H)`, `P* = P/(ρ₀gH)`).
     **All three now exist**: `scripts/probe_acsphDambreakLobovsky.py`
     (committed — tank 1610×600 mm, gate at 600 mm, column 600×300 mm,
     sensors at z = 3/15/30/80 mm, Table 2 percentiles encoded in mbar against
     `ρgH`) and `dambreak.py`'s `pressureProbeHeights` diagnostic (MLS
     interpolation of the fluid pressure at fixed wall sensor points, emitted
     every step). **What is missing is the run** — `scripts/out_acsphDambreakLobovsky/`
     does not exist, for either scheme. Note the paper's own §4.5 uses
     **free-slip** walls, `H/Δx = 400` (2D), RK3, `CFL_t = 0.4`, `Δt/Δτ = 2.0`
     and displacement shifting; match those before quoting a discrepancy.
10. **Optional/experimental**, only if the above is clean: `ṽ` material
    derivative (§1.7), internal shifting (Eq. 60), `k₃` term, RK3/RK4.

---

# Part 9 — The validation sweep

## 9.1 `deltaSPH`, because the `ψ` sign changed under it — **done, 2026-09-05**

Part 3's sign fix alters the *default* density-diffusion operator: `deltaSPH`
was behaving as twice the uncorrected Molteni–Colagrossi Laplacian on any smooth
field and is now the fourth-order Antuono operator it was meant to be. That is a
behaviour change in every WCSPH run in the repo. What existed before this sweep
was a single-variable A/B on one case (`sloshingTank`: density floor 0.396 →
0.532, peak sensor pressure 39.2 → 32.5 kPa, neither run diverging) plus a
green test suite with no `deltaSPH`-scheme physics case in it.

**`scripts/run_sweep.py --cases dambreak drivenSquare droplet impact kolmogorov
ldc movingObstacle openFlow randomFlow sloshingTank squarePatch tgv-wc --full
--timeout 1800`** — every `deltaSPH`-family weakly-compressible case in the
registry, each run to its own `tLimit`, not just a smoke pass (a stability
regression like the one below only shows up deep into a long run). **11/12
passed clean** — no `non-finite`/`diverged`/NaN marker in any of their logs.
Confirmed all 11 genuinely exercise the shift-dependent code path this fix
touches: none of their `configureScheme`s override `shiftProperties`, so each
inherits the shared dataclass default (`ShiftingScheme.deltaSPH`,
`ShiftingProjectionScheme.surfaceNormal`, `active=True`) untouched — this is
real positive evidence, not 11 cases that happened not to test the thing that
broke.

**The one failure, `sloshingTank`, is not a new finding — it's the same
already-diagnosed and already-fixed issue** (`PST_ALE_PLAN.md` §7.1,
decision 1 above), and in a bare sweep invocation it isn't even the same
mechanism: `sloshingTank`'s own `params=dict(shifting=False, ...)`
pre-populates the `shifting` key, so `ctx.param('shifting', True)` in its
`configureScheme` resolves to `False` regardless of the `True` fallback
written there — a bare `warpSPHRun sloshingTank` runs with **no shift at
all**, which is the case's own long-documented, deliberately-demonstrated
"why shifting exists" baseline failure, unrelated to which projection scheme
is configured. (The `surfaceNormal`-specific regression this sweep does *not*
exercise — since shifting never turns on in a bare run — is the one already
found and fixed by hand: see `PST_ALE_PLAN.md` §7.1.)

**Conclusion: the sign fix is not a broad regression.** One case
(`sloshingTank`, and specifically only when shifting is deliberately turned
on for it) needed its free-surface treatment updated, already done. Every
other `deltaSPH` case in the registry, running the same `surfaceNormal`
default this fix touches, is unaffected over a full run. `scripts/probe_deltaSPHPsiSignAB.py`
remains available for a single-variable A/B on any specific case someone
wants to re-check, but the broad sweep this section called for is complete.

## 9.2 `artificialCompressible`, because it is new

All five cases of Part 7 are wired and have been run (status board, Part 8
steps 5b/6/8/9). What remains on each is scale, not wiring:

| § | Case | State |
|---|---|---|
| 4.1.1 | `hydrostaticColumn` | Run; Fig. 2 operator comparison reproduced qualitatively at nx=48/300 steps. Paper's own settings for this case: `CFL_τ = 0.3`, `Δt/Δτ = 5`, `ε_v = −7`, RK2, `t = 50 s`. |
| 4.2 | `rotatingSquarePatch` | Run (smoke + AC-4/AC-JST). Fig. 15/16 vs. the BEM crosses of `letouze2013` still deferred on digitization. |
| 4.3 | `oscillatingDroplet` | Table 1/2 sweep done at reduced scale (nx=24, one period, RK3); cost trend reproduces, error cliff does not. |
| 4.4 | `impact` | Analytic Marrone Eq. (A.34) implemented; nx=64 crosses the target to 0.1 %. Run at `aspectRatio = 0.5` (`H/L = 1`, Marrone's own figure case), **not** De Courcy's `H = 2L` (`aspectRatio = 0.25` here). |
| 4.5 | `dambreak` | Lobovsky harness built, not run (step 9). Separately, the Marrone §3.1 long-horizon ACSPH run **diverges at `t* ≈ 6`** — the headline open item (status board 2026-09-20 item 2). |

## 9.3 Note for later

`ψ` is one member of a broader family of diffusion/dissipation operators worth
implementing properly (the `DensityDiffusionScheme` enum is the current, narrow
slice of it). Not scheduled here; recorded so the sign fix is understood as a
correction to one operator rather than a redesign of the family.

---

## Relationship to the other plans

This displaces `docs/historic_plans/DFSPH_IMPROVEMENT_PLAN.md` as the active priority, per the
current decision. `COUPLED_INCOMPRESSIBLE_NEWTON_PLAN.md` remains queued behind
both. Steps 2 and 3 above are shared infrastructure that benefit the DFSPH work
regardless of ordering, so they are not sunk cost if priorities shift back.
