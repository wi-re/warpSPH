# Lobovsky pressure: before/after ghost spacing — 2026-10-28

Orange is the prior ramp0.5 trajectory with mass-derived ghost spacing; blue
is the new ramp0.5 trajectory with volume-derived ghost initialization. The
collision shift remains absent in both. All original particle masses, physical
ICs and10,000 timestep clocks are retained.

![Shared clocks including native](../../.tmp/acsph-lobovsky-volume-ghost-2026-10-28/pressure-comparison/pressure-shared-frames.png)

The native overlay uses only the same193 post-shift frames (~10ms cadence),
with the same fluid-only normalized Wendland C2 probe estimator. Native raw
probe.plt is an earlier RK state and is not mixed into these plots.

![Every accepted local step](../../.tmp/acsph-lobovsky-volume-ghost-2026-10-28/pressure-comparison/pressure-every-step.png)

Local high-cadence plots retain all10,001 observations, including initial state;
no smoothing, averaging or clipping. Dry probes are recorded as zero. P1–P4
lie at x1.61m, heights3/15/30/80mm. Shared time arrays, completion attributes
and finite pressure arrays were checked. Allthree generated PNG figures were
visually inspected; matching SVGs are retained for export.

The broad first-impact curves remain similar. Every-step P1 maximum over
0.4–0.7s falls51.221→40.312kPa. The correction is not uniformly quieter:
there is a pronounced new pulse around1.0s and later negative ringing remains.
Over1.25–1.65s P4's positive maximum falls40.445→17.868kPa, while its minimum
becomes more negative (−3.149→−7.982kPa). These are extrema over the named
windows, not matched-event or experimentally validated peak errors. Identical
~10ms sampling can miss these narrow pulses, so the separate dense plots matter.

[Impact details](../../.tmp/acsph-lobovsky-volume-ghost-2026-10-28/pressure-comparison/pressure-impact-details.png) ·
[Metrics and provenance](../../.tmp/acsph-lobovsky-volume-ghost-2026-10-28/pressure-comparison/pressure-metrics.json).

Reproduce with `scripts/acsph_reference/plot_pressure_spacing_comparison.py`
using `--before`, `--after`, `--native` H5 recordings and `--out`. This is an
offline plot-only pass; no GPU simulation, solver change or numerical promotion.

## Follow-up: pressure regression and mixed wall closures

The user's observation is supported quantitatively. On the same native/local
recording clocks, P1 RMS error against native is0.225→0.588kPa over0.65–0.95s,
and0.146→6.068kPa over0.95–1.05s (orange→blue). The full shared-frame mean
absolute errors increase at allfour probes. This is not uniformly worse in
every window: broad first-impact RMS errors improve slightly, and returning-wave
RMS errors also improve slightly. Dense local traces establish that the new
oscillation grows before convergence caps: blue uses13–14 iterations with no
caps over0.65–0.95s, then41 caps over0.95–1.05s. Native high-frequency behavior
remains limited by its shared-frame recording cadence.

The initialization correction changed ghost sampling positions as well as
bounce normals. Flat bottom nodes moved from11.8585mm inside the fluid to
0.375–2.625mm. A code audit found a mixed closure in the fixture:

- `audit_local.install(wall_velocity='source')` replaces only
  `dualTime.computeBoundaryVelocities` for RK residual stages.
- `artificialCompressible_step` still calls its separately imported mDBC
  `computeBoundaryVelocities` at physical-step entry.
- Non-fluid solve updates arezero, so those initial mDBC wall velocities remain
  in the accepted state passed to `finalize`'s Michel shifting. Michel's
  characteristic velocity includes wall particles. Rigid-body kinematics reset
  static wall velocities after shifting, so recorded wall velocities do not
  expose this intermediate closure.

`scripts/acsph_reference/audit_ghost_layout.py` compares two ghost layouts on
exactly the same retained physical fields, rebuilding adjacency for each.
The complete step1707 bundle supplies material/configuration metadata only;
this is **not a restart or a physical time step**. It loads each trajectory's
step4038 snapshot at0.800139004s and uses its actualdt0.000228648414s for
coefficient/shift evaluation. Pressure/velocity and surface detection are
evaluated with the source closure; shifting is evaluated with each closure.
No bounce or BDF equation is advanced. Static wall body velocity iszero.

| Fixed fluid field | Ghost-closure wall-velocity difference, RMS / max [m/s] | Max shift difference with ghost closure [dx] | Max shift difference with source closure [dx] |
|---|---:|---:|---:|
| Blue trajectory at0.8s |0.0843 /0.668|0.2317|0.0000795|
| Orange trajectory at0.8s |0.0907 /1.553|0.2473|0.000159|

About8,000 of320,000 particles change shifting by more than0.001dx under the
ghost-closure layout swap. None exceeds that threshold with the source
closure. Source-closed momentum RHS differences have component RMS~1.7e-5m/s²;
pressure RHS differences~0.3Pa/s against~3e4Pa/s field RMS on the blue state.
Surface eigenvalue differences have RMS~3.4e-8. These small differences are
consistent with changed neighbor/reduction ordering, whereas shifting has a
substantial extra geometry dependence through its wall velocities.

**Verdict:** the old orange trajectory is the better pressure baseline between
impacts. The full spacing correction is not physically qualified by reduced
iteration counts. Mixed closures are a demonstrated sensitivity, not yet proof
of the ringing's cause: bounce normals also changed, and trajectory history is
different. The next controlled trajectory should separate normals-only from
node placement and use a declared consistent wall-velocity closure before
another full replay or any default promotion. No shared boundary code changed.

Evidence: `.tmp/acsph-ghost-layout-audit-2026-10-28/pressure-errors.json`,
`t080-dt-matched/summary.json`, `orange-t080-dt-matched/summary.json`.
The two final fixed-field audits took1.005s and1.003s respectively (operator
work after loading;0physicalsteps, so ms/step is inapplicable). Video was
explicitly omitted for these fixed-field evaluations. The earlier100µs
shift-only exploratory audit is retained in`t080`, superseded by actual-dt
controls. Script syntax compilation passed; both final audit processes exited0.

## Consistent-closure replay

User authorized making the closure consistent and rerunning on2026-10-28.
The comparison CLI now distinguishes historical RK-only `--wall-velocity source`
from `source-consistent`. The latter installs the same static source Shepard/
reflection formula at step entry and RK stages, and refreshes it from the
accepted fluid velocities immediately before Michel shifting. The pre-shift
closure rebuilds/reuses the Verlet list at accepted positions; shifting then
uses that same adjacency. This is fixture plumbing, with no shared boundary
implementation or production numerical-default change. It removes the
demonstrated mixing of two existing wall-velocity rules; no new pressure or
damping term was introduced.

The focused CPU regression suite passes9 checks. The new handoff regression
uses a uniform flow against a flat wall, checks normal reflection/tangential
preservation at entry and RK stages, then changes the fluid field and poisons
stale wall velocities before shifting. The shift hook must receive the fresh
closure while preserving fluid velocities exactly. Both geometry and existing
pressure-diffusion invariants are retained.

Run evidence: `.tmp/acsph-lobovsky-consistent-closure-2026-10-28`.
The previous frozen source was hash-verified before copying; only the fixture
hook, CLI/metadata and pressure-plot labeling scripts changed. Volume ghosts,
ramp0.5, supplied particle IC/mass, mapped coefficients, FP32 increment/fused
solver, tolerance/iteration budget and all10,000 native timesteps are retained.
Native collision shifting remains absent. A20-step video-enabled graph
preflight must pass endpoint/IC/graph/video checks before the full replay.
The detached serial supervisor preserves progress and wall/silence/stall
watchdogs, then validates/export the full trajectory and generates native
analysis and pressure comparisons against both retained local baselines.

Launch qualification passed:9 regressions repeated against frozen source,
20-step preflight90.36s/4518.20msstep (16 known startup caps), all20 solves
graphed. Its20 primary-video frames decode and the first frame was inspected.
The validation script initially counted an additional `images/output.mp4`
encode as a second recording; primary-video selection was corrected, the failed
gate receipt retained and the successful physical preflight reused. The full
10,000-step replay is running; physical pressure improvement remains pending.

## Completed replay review — 2026-10-29

All8 serial phases passed. Full replay reached10,000steps/t1.918419918s with
no divergence or stop; all10,000 solves used CUDA graphs and1001 primary-video
frames decoded. Physical IC/mass checks, H5 export,193-frame native comparison
and both pressure-plot jobs passed. Runtime5539.67s (1h32m20s),553.97ms/step,
12.96% below the prior mixed-closure/volume run6364.86s/636.49msstep. Capped
steps133→37, including16startup caps; median iterations13. Wall comparisons
are observed campaign costs, not an isolated benchmark against native's older
partly shared-GPU run.

| Quantity | Old mass spacing / mixed | Volume spacing / mixed | Volume spacing / consistent |
|---|---:|---:|---:|
| P1 native RMS error0.65–0.95s [kPa] |0.2248|0.5880|0.0179|
| P1 native RMS error0.95–1.05s [kPa] |0.1458|6.0680|0.0172|
| P1 native RMS error1.25–1.65s [kPa] |5.0614|4.5890|4.7759|
| P1 dense peak0.4–0.7s [kPa] |51.221|40.312|40.366|
| P4 dense minimum1.25–1.65s [kPa] |−3.149|−7.982|−19.185|
| Capped solves /10000 |214|133|37|
| EndpointKE [J/m] |54.538|52.163|58.053|

Native endpointKE is45.257J/m. The consistent run's time-averaged KE differs
by+0.32% over0.4–1.25s, then+6.54% over1.25–1.91842s (allaverages use the
same193-frame clock). EndpointKE excess rises to28.27%, versus15.26% in the
prior volume run and20.51% in original orange. Endpoint pressure material-row
RMS difference is1.029kPa and velocity RMS difference on common1cm cells is
0.530m/s. Endpoint errors alone do not establish a dissipation mechanism in
these separated trajectories.

**Verdict:** consistent closure removes the growing oscillation and pulse
around1s, with much better native agreement before the late impacts, and
substantially fewer caps. This full replay supports the fixed-field diagnosis
of mixed wall closures as the early regression's source. It does not qualify
late-impact pressure: strong positive/negative ringing persists and some
negative extrema worsen. Keep the consistent comparison-fixture plumbing;
the later pressure/dynamics gap remains open. This is not a solver-default
promotion, and no additional physical run was started during this review.

![Closure review](../../.tmp/acsph-lobovsky-consistent-closure-2026-10-28/closure-review/closure-review.png)

Four-trajectory PNG/SVGs and quantitative window data are retained in
`.tmp/acsph-lobovsky-consistent-closure-2026-10-28/closure-review`.
Reproduce with its parent`summarize_closure.py` (offline only,0.47s). The
combined figure and the two automatic pressure figures were visually inspected.
Local curves use every accepted step, native only shared~10ms post-shift
frames; native high-frequency behavior therefore remains incompletely resolved.
No smoothing, pressure clipping or raw earlier-RK native-probe overlay was used.

## Late-ringing precision controls — 2026-10-29

The closure change addressed the early growing mode; it did not establish the
cause of late ringing. User requested FP64 reruns of selected ringing sections.
Retained physical snapshots do not carry enough BDF/low-part history for an
exact restart. `probe_precision_sections.py` therefore replays the unchanged
frozen FP32 baseline to capture complete-history checkpoints at step6917
(1.440025206s) and8055 (~1.590s). Each checkpoint seeds separate FP32 and FP64
branches through7641 (1.530014125s) and8650 (1.680013257s), respectively. The
same state, static wall geometry, timesteps, tolerance, iteration budget,
consistent closure and ramp0.5 are retained. FP64 promotion cannot undo earlier
FP32 error; this isolates arithmetic during the requested windows rather than
claiming a globally FP64 trajectory.

Restart qualification passed before the long prefix: a3-step FP32 prefix
reproduces the retained full trajectory's diagnostics, and its step2→3 restart
matches every saved particle/history/compensation/rigid-body tensor and the
uninterrupted physical snapshot bitwise. The FP64 branch verifies exact
promotion of captured fields and completes its one-step CUDA-graph control.
Allqualification videos decode. Costs include initialization/compilation and
checkpoint overhead: prefix19.25s/6394.64msstep, restartFP329.07s/9006.07msstep,
restartFP6435.13s/35068.37msstep. They are not steady-state throughput figures.

Campaign `.tmp/acsph-ringing-precision-2026-10-29` contains a detached serial
supervisor, frozen source hashes, videos/progress, checkpoint/precision gates
and queued pressure plots. Initial queue environment/helper packaging errors
and a short-control snapshot-selection gate error were repaired; failed
receipts remain, and successful physical controls were reused. Prefix and all
FP32 branches require retained-baseline parity. Precision outcomes are pending.

Remaining hypotheses are distinct: finite-precision pressure/moment arithmetic;
position-only stopping before pressure/momentum settle; wall-bounce/contact
stiffness or deficient supports; and local shifting without native collision/
pressure-velocity field remapping. The precision branches hold all of these
model choices fixed. No production/shared-boundary/default edits were made.

**User cancellation (2026-10-29):** precision campaign stopped during the FP32
checkpoint prefix. Its supervisor/child were verified inactive and all pending
windows/analysis jobs cancelled; completed qualification and partial evidence
remain. There are no late FP64-window results. User requests no comparison
campaign while the model gap remains: dynGPUSPH applies pressure/velocity
remapping after shifting, whereas the current warpSPH Lobovsky setup does not
(and also omits dynGPUSPH's collision shift). No automatic resume or replacement
physical run is queued. Receipt: precision campaign `user-stop-receipt.json`.


## 2026-10-29 — external shift/transport gap closure and first-ring replay

User authorized closing the identified implementation gap before another
comparison, then FP32 only until the first impact ring decays. The declared
baseline is the completed consistent-wall-closure FP32 run above. Its source
was hash-verified and copied; the bounded run overrides only the new named
shift operator and explicit reference-fixture hooks/CLI. It retains all ICs,
masses, native timesteps, volume-derived ghosts, ramp0.5, mapped coefficients,
RK/history/damper/diffusion settings and iteration controls. Shared boundary
code remains untouched. This is a combined external-shift parity experiment,
not an attribution of changes to remapping alone.

The new `--shifting dyngpusph-reference` mode ports the supplied compiled
`getContribShift3`/`shiftandInterpolateD`: coefficient0.2, caps/lambda gates,
color-normal/cone surface inputs, cohesion and collision shift, pressure
safeguards, raw-gradient p remap, diagonal-only v remap, and shift-BDF history.
Paper Eq58 calls for full Taylor remapping; the compiled velocity expression
is componentwise. Other explicitly retained reference quirks include source
surface tags1(raw)/2(vicinity), `clamp(0,1,1-d/dx)` evaluating to1, duplicated
prior displacement in BDF, and the supplied two-cell z grid visiting each pair
three times in the extended shift stencil (the ordinary SPH stencil visits
it once). These are comparison compatibility choices, not new solver defaults.

Qualification evidence is in `.tmp/acsph-shift-remap-2026-10-29`:

- 18 FP32 regression checks pass;9 FP64 shift/surface tests pass.
- On identical captured dynGPUSPH FP64 fields, the shift/transport operation
  differs by at most1.11e-16m in position,4.75e-19m/s in velocity,
  2.56e-12Pa in pressure,1.59e-17m in recorded shift.
- Stage-zero surface input replay: lambda max1.23e-15; normal max2.86e-7
  (near-zero color gradients amplify reduction roundoff).31/320000 raw/vicinity
  tags differ because the source excludes exactly collinear cosine±1 on the
  nearly uniform lattice. Preserve that discontinuity rather than tune a tag
  tolerance. This is numerical sensitivity, not a different cone formula.
- Completed FP32 one-step preflight: graph active, finite fields, no stop;
  19.766s /19765.99msstep including initialization/compilation, video2frames.
  Maxshift.00812dx, maxpremap1.874Pa, maxvremap3.259e-7m/s;
  shift-remap KE change−1.30e-11J/m. Not a throughput measurement.

The source-sealed replay is3579native steps (~0.70s), with video, streamed
progress, stall/wall watchdog, per-step pressure/KE and remap diagnostics;
export follows serially. The original stopped precision campaign stays
cancelled. No full-duration run or FP64 physical campaign is queued.

Results pending. dynGPUSPH saved post-shift frames (~10ms) can compare the
impact envelope, not resolve high-frequency between-frame ringing. Its raw
`probe.plt` samples a different RK stage and is not overlaid as an equivalent
pressure trace.


A first short prefix was stopped during qualification: graph validation makes
an eager evaluation after capture, overwriting Python-cached surface geometry
on the first physical step. The hook now preserves the captured tensors while
the graph remains active; those tensors update on every replay. A complete
one-step FP32 control matches x/v/p/rho/volume/kinds **bitwise** with graph
versus eager (9.112s vs9.029s including startup/video). Failed prefix and the
CLI-control packaging failure are retained, not used for impact results. The
913-source manifest was re-sealed and the3579-step physical run restarted
from the original IC. Controls: `graph-eager-shift-parity.json` and associated
preflight logs/videos in the campaign root.


### Completed bounded FP32 result (2026-10-29)

The original-IC replay completed3579steps to **t=0.700116862867s**. It stopped
at the declared bound, not due to divergence, a velocity alarm or an iteration
failure. Wall1477.056s (24.618min),412.701msstep including video/diagnostics.
All steps retained CUDA graphs. Matched-prefix solve-only means:333.856ms
(previous Michel) versus348.005ms (reference shift/remap), +4.24%;49563 versus
49671 pseudo-iterations. Both have16 capped startup steps and no later caps.
The partial-prefix startup-hook failure is excluded from these results.

| Quantity | warpSPH FP32, previous Michel | warpSPH FP32, dynGPUSPH shift/remap |
|---|---:|---:|
| P1 first peak |40.3659kPa at0.414996s|32.3691kPa at0.415355s|
| P1 first undershoot |−4.60654kPa|−1.32075kPa|
| P1 fast-tail RMS |9.73062Pa|20.1349Pa|
| P2 fast-tail RMS |9.18679Pa|19.2264Pa|
| P3 fast-tail RMS |10.6508Pa|18.1213Pa|
| P4 fast-tail RMS |2.38688Pa|2.70826Pa|
| KE at0.700117s |119.48919J/m|119.45322J/m|
| Mean particle pressure at endpoint |559.534Pa|555.585Pa|

Fast-tail metric: resample accepted post-shift pressure at0.2ms, subtract a
Gaussian trend with sigma5ms over0.46–0.65s, discard10ms at either edge.
It measures fast variation, not error relative to experimental sensor noise.
The new P1 peak is19.81% lower and undershoot is smaller, but the narrow pulse
and ringing remain. Small tail fluctuations increase, rather than disappearing.
The new KE is only0.0301% below the previous warpSPH run; no substantial extra
bulk damping accompanies that peak reduction.

At identical saved post-shift clocks over0.38–0.70s, pressure RMSE against
dynGPUSPH changes P1:123.13→102.84Pa, P2:92.19→57.46Pa,
P3:39.88→29.21Pa, P4:7.87→8.31Pa. The pressure envelope improves forP1–P3,
notP4. dynGPUSPH's saved frames are~10ms apart; its sampled first P1 peak
7.959kPa does **not** establish its true peak between frames, or absence of
high-frequency ringing. The earlier-RK raw probe.plt is not mixed into these
comparisons.

dynGPUSPH endpoint KE119.28320J/m: the new warpSPH result is+0.1425%.
Mean particle pressure remains7.69% lower than dynGPUSPH601.879Pa (the previous
warpSPH value was already7.04% lower). Material-row position/velocity/pressure
RMS differences at the endpoint are3.554mm /0.02580m/s /91.82Pa; velocity
RMS on common occupied1cm Eulerian cells is0.00654m/s. Thus this bounded check
does not establish complete evolved solver equality or cure late anomalies.

The remap itself cumulatively changes KE by−0.08962J/m and momentum by
[+0.00615,−0.06306]kg/s per unit breadth. These are isolated remap contributions,
not total conservation errors (gravity/wall forces also act). Maximum actual
shift0.2858dx includes wall collision added after the regular0.1dx cap;
maximum single-step field remap is2152.94Pa /0.29703m/s. These are retained
compiled-source choices, not newly justified physical defaults.

Validation:913 pinned source hashes unchanged, same rounded initial state,
71 finite saved frames, fixed membership320000fluid particles, no mass change
(stored FP32 mass promoted in H5 sums to180.00000637kg/m), two duplicate video
outputs decode180frames each. No fluid lies outside physical wall midplanes
[-0.000375,-0.000375] to[1.609625,1.399625]. The nominal viewer box starts at0,
but the supplied IC starts fluid rows at0 and inner walls at−dx; negative
coordinates within half a spacing are not wall leakage. dynGPUSPH and the
previous warpSPH run show the same convention.

### FP32 branch sensitivity found after the bounded run

An additional fixed-input FP32 operator audit intentionally compared rounded
inputs against the captured FP64 dynGPUSPH output. Its strict equality-scale
gate **fails**: max shift difference6.006micrometres, position6.012micrometres,
pressure0.03401Pa, velocity2.925e−9m/s (shift componentRMS7.51nanometres).
The failure is retained; no tolerance or model formula was tuned to hide it.
The FP64 same-input port still agrees to roundoff, and the FP32 regression
oracles/complete graph-versus-eager state checks pass.

The largest discrepancy is traced to a strict wide-cutoff pair at
(0.5977500001,0.2932499994). FP64 includes17 raw-surface neighbors and obtains
max normal angle1.54912rad, vetoing cohesion at0.3pi. FP32 rounding includes16
and obtains0.785398rad, permitting a6micrometre cohesion shift. The pair can
flip the angular veto even with nearly zero kernel weight at the cutoff.
This is a demonstrated arithmetic-sensitive discontinuity in the compiled
reference model, distinct from a missing implementation term. Surface cone
collinearity already has the separate31-row FP64 reduction-sensitive tag
qualification above. No smoothing or other numerical remedy was added here.

**Verdict, corrected after the detailed 2026-10-29 audit:** retain the named
reference mode as an isolated external-shift operator control. The kernel's
identical-input qualification remains valid, but its integrated caller supplies
incorrect wall fields: zero wall pressure and mirrored wall velocity in place
of closed pressure and physical wall velocity. Further active gaps include
momentum JST, viscosity wall handling, startup pseudo-clock, convergence gates
and support-quality lambda. This bounded trajectory is a partial-alignment
experiment; a same-model precision attribution is premature. See the
[detailed audit](DYNGPUSPH_COMPARISON.md#2026-10-29--end-to-end-parity-and-precision-sensitivity-audit)
for fixed-field measurements and source evidence. No full-duration simulation
was run or scheduled, and no generic defaults were promoted.

Artifacts in `.tmp/acsph-shift-remap-2026-10-29`:
`analysis/pressure-first-ring.png`, `analysis/pressure-impact-detail.png`,
`analysis/dynamics-first-ring.png`, `analysis/summary.json`,
`trajectory-analysis/summary.json`, `validation-receipt.json`,
`operator-parity.json` (FP64), `operator-parity-f32.json` (strict failed audit),
`fp32-branch-audit.json`, `graph-eager-shift-parity.json`, videos and
`warpSPH-lobovsky-dyngpusph-shift.h5` for the remote viewer.
