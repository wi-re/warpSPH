# Lobovsky D5 ramp0.5 qualification — 2026-10-28

The smooth ramp repairs most hard-bounce iteration overhead without materially
changing early bulk dynamics. It does not establish physical parity with
native. A separate, measurable wall-containment gap starts before impact in
both local variants. Native contains an active post-solve collision displacement
that the local fixture lacks. A short collision-only control partially improves
containment. More decisively, both sampled remaining ramp caps are removed by
correcting a mass-versus-volume unit error in ghost-derived bounce normals.

[Metrics](../../.tmp/acsph-lobovsky-ramp05-2026-10-28/qualification/metrics.json) ·
[Recorded-frame metrics](../../.tmp/acsph-lobovsky-ramp05-2026-10-28/qualification/frame-metrics.json) ·
[Native collision-term evaluation](../../.tmp/acsph-lobovsky-ramp05-2026-10-28/qualification/collision-shift.json).

## Controlled full trajectory and cost

All three trajectories use 320,000 fluid particles, approximately180 kg/m,
the supplied mass/ICs and all10,000 recorded native timesteps to1.918419918 s.
Native is FP64; both local trajectories use explicitly rounded FP32 input.
The new full replay differs from the corrected hard-bounce baseline only in
`wallBounceRamp=0.5`. All914 baseline runtime files were hash-verified; the
only changed runtime file adds/selects this fixture CLI option. No production
numerical/shared boundary/default changes were made.

| Observable | Native FP64 | Local hard | Local ramp0.5 |
|---|---:|---:|---:|
| End-to-end wall [s] |15923.0|19002.7|6681.6|
| End-to-end ms/step |1592.3|1900.3|668.2|
| Mean / median pseudo-iterations |23.36 /15|68.23 /15|20.11 /13|
| Steps at200 cap |142|2748|214|
| Measured solve wall [s] |not recorded|17736.8|5395.3|
| Solve wall spent on capped steps |not recorded|80.46%|20.44%|

Both local paths graph all10,000 steps and finish without a stop/divergence.
Ramp is2.84× faster end-to-end than hard. Native shared the GPU for part of
its run, so its wall cost is not an isolated precision/hardware speed comparison.

The first post-startup ramp cap is1707/t0.411939720 s, near first impact;
hard first caps468/t0.163517007 s. Ramp has no caps between0.005 and0.4 s.
The longest ramp post-startup block has31 steps (6295–6325,
t1.29952–1.30738), versus1144 in hard. Remaining ramp caps:29 at0.4–0.8 s,
2 at0.8–1.2,143 at1.2–1.5,24 after1.5. They cannot all be labelled the old
hard-threshold two-cycle without inspecting their residuals.

## Energy: a late discrepancy shared by both local paths

Time-weighted mean KE relative to native, from the same193 physical frames:

| Physical window [s] |Local hard|Local ramp0.5|
|---|---:|---:|
|0–0.4|−0.148%|−0.137%|
|0.4–1.0|+0.199%|+0.237%|
|1.0–1.25|+0.245%|+0.469%|
|1.25–1.35|−0.327%|+0.475%|
|1.35–1.6|+0.952%|+1.162%|
|1.6–1.91842|+6.961%|+6.792%|

Ramp differs from hard by less than0.81% in every listed phase mean. The late
energy discrepancy predates the remedy; it is not evidence of a new global
ramp-induced energy increase. Endpoint KE is45.257/52.907/54.538 J/m
(native/hard/ramp), or+16.9%/+20.5% local. Ramp endpoint K+gravity PE is
162.435 J/m versus154.103 native (+5.41%). K+PE is not a conserved quantity
for these viscous, shifted numerical trajectories, and reference agreement
alone does not establish physical accuracy.

At the endpoint the fastest1% of particles carry7.55% of ramp KE versus7.40%
native. Excluding each run's within1dx/outside-wall band still leaves19.58%
more ramp KE. The discrepancy is not explained by one fast outlier or only
the particles currently touching the walls. Wave/jet shapes separate later;
these measurements do not yet attribute the gap to a single operator.

![Energy and wall geometry](../../.tmp/acsph-lobovsky-ramp05-2026-10-28/qualification/bulk-wall.png)

## Pressure: broad agreement and unresolved narrow transients

Post-shift fluid-only normalized Wendland C2 probes are evaluated at the same
physical clocks in all three common-frame recordings. Native raw RK probes
sample a different state/stage; they cannot be treated as identical peak data.

First-impact P1 common-frame peaks are7.959/6.943/6.856 kPa
(native/hard/ramp). Local every-step peaks are36.764/51.221 kPa
(hard/ramp), at0.416243/0.416128 s. Ramp's sharp peak increases39.3% while
its0.4–0.7 s every-step P1 impulse changes only−0.58%
(1087.72→1081.40 Pa·s). Removing iteration chatter therefore does not, by
itself, qualify pressure transients. Common-frame impulse is1169.88/1100.03/
1096.48 Pa·s; the10 ms sampling misses narrow pulses.

Later sampled pressure differences remain large in both local paths. Over
1.25–1.6 s P1 common-frame maxima are23.59/5.72/4.84 kPa; local every-step
maxima are6.96/7.05 kPa. Ramp also has a40.45 kPa every-step P4 pulse near
1.33333 s which is absent from its sparse common frames. A matched high-cadence
post-shift native recording is needed before narrow-peak accuracy claims.

![Caps and common-frame pressure](../../.tmp/acsph-lobovsky-ramp05-2026-10-28/qualification/caps-pressure.png)

## Wall containment exposes a separate reference-code gap

Signed distance is measured from the theoretical tank planes x=0,1.61,
y=0,1.4; fluid ICs include centres on x=0/y=0 and the first wall-particle
layer sits1dx outside. Thus a small negative centre distance is distinct
from escaping through the full wall layers. These are recorded-frame bounds,
not maxima over unobserved timesteps/RK stages.

| Recorded geometry |Native|Hard|Ramp0.5|
|---|---:|---:|---:|
|Deepest penetration over all193 frames [dx]|0.296|2.714|3.813|
|Deepest penetration at t≈0.4 s [dx]|0.134|2.075|2.082|
|Particles deeper than1dx at t≈0.4|0|82|86|
|Particles deeper than1dx at endpoint|0|3|1|

The deepest ramp particle is UID316413; at endpoint it has4 fluid neighbors
including itself and29 wall neighbors within support. Fluid kernel-volume
sum is0.1393 (almost entirely the self contribution), while fluid+wall sum is
0.8858. Physical stored density is constant1000 and must not be interpreted
as sampling density. Boundary coverage can hide a deficient fluid stencil.

The compiled native source, verified against its retained build receipt,
contains a post-solve collision displacement in `getContribShift3` and
`shiftandInterpolateD`:

```
Δx_collision = -Σ_b m_b (r_ib - 0.9 dx) (x_i-x_b)/r_ib
               / (m_i + Σ_b m_b),   for r_ib < 0.9 dx.
```

It is added after the bounded Michel/surface-cohesion displacement, and is
also applied for particles with no fluid neighbors. The local comparison uses
its Michel shift and has no equivalent collision term; its optional mDBC
no-penetration acceleration is explicitly off. Native's separate BRPF
acceleration is explicitly zeroed in this compiled path, so it is not an
active missing repulsive force.

Evaluating only native's collision term on retained ramp frames yields14
nonzero rows by t≈0.020 s,200 at0.100 s and831 at0.400 s. Maximum hypothetical
single-call displacements are0.064/0.289/0.433 dx respectively. This is a
same-input operator observation, not an evolved repair or full native shift.
The term can oppose shallow penetration before a particle reaches the deeper
wall lattice; once embedded, corrections need not all point toward the domain.
An independent equal-mass one-pair oracle and unequal-mass cKDTree/pair-list
implementations agree numerically. The source's other shift differences
(surface cohesion, coefficient, p/v transport and shift-history treatment)
remain separate choices.

![Common physical frames](../../.tmp/acsph-lobovsky-ramp05-2026-10-28/qualification/spatial.png)

## Remaining caps isolate a ghost-spacing unit bug

The exact1727-step smooth physical prefix reproduces all1728 accepted diagnostic
rows and every iteration count/epsilon from the completed full ramp run.
Original graph/eager results match allfive saved output tensors bitwise at
both targets. Fresh-process fixed-solve reload also matches allfive bitwise.
Original results alone advance the physical prefix; alternatives never enter it.

| Same captured FP32 input |Step1707|Step1727|
|---|---:|---:|
|Original ramp0.5 graph/eager|200|200|
|Bounce off|13|13|
|Hard switch|200|200|
|Wider ramp1|200|13|
|Ramp0.5 with correct-spacing bounce normals only|**13**|**13**|

A single particle carries99.9943%/99.9978% of squared position residual:
UID318001/318805. Step1707 shows a period3 cycle in selected wall normal;
its nearest-wall normal distance is≈0.001dx, far inside the fully active ramp.
Step1727 has a period2 cycle within the ramp. Widening the ramp does not address
both cases; disabling bounce is a causal control, not an adopted wall model.

The root spacing calculation is in `rigidBody/ghostParticles.py`, inside
`addBoundaryGhostParticles`: `dx = particleState.masses.mean() ** (1/dim)`.
The initializer has already multiplied particle volumes by rho0, so this takes
a root of physical **mass**, not volume. At rho0=1000 in this2D fixture:

|Quantity|Value|
|---|---:|
|Physical particle mass|0.0005625 kg/m|
|Mass-derived ghost spacing|0.023717085 m|
|Configured / volume-derived spacing|0.000750000 m|
|Spacing error factor|31.6228|

That spacing is used as the SDF-normal mollification step and for ghost-node
placement/validity thresholds. It falsely spreads the corner blend over centimetres
instead of about one particle spacing. At x≈1.588 m (22mm from the right wall),
bottom normals vary in x from−0.00343 to−0.01261 as the nearest wall row changes;
the correct-spacing bottom normal is(0,1). At x≈1.602 m, the affected normal is
(−0.39392,0.91915), even though the point is about8mm from the actual corner.
The physical velocity at step1727 is(3.33476,−0.000250) m/s, but that false normal
turns almost purely tangential motion into **−1.31385 m/s wall-normal velocity**.
The bounce then injects a large artificial vertical position-row response.

The causal fixed-solve control recomputes only the captured D5 wall normals
using the same grid-snap geometry routine with configured dx. It retains all
original positions, ghost topology/closures, BDF/low-part history, adjacency,
pressure operator, ramp and stopping tolerance. Both caps fall200→13. Fluid
pressure differs by27.22/41.10 Pa RMS from the capped results. This is not a
complete corrected-initialization trajectory, and the dimensionful pressure/
momentum residuals remain nonzero after meeting the position stopping norm.
It does not prove that every one of the214 caps has the same cause.

The shared boundary implementation has not been edited. The next implementation
work is a volume-based spacing correction with a density-rescaling geometry
regression, then bounded corrected-initialization impact checks. Retain the
old frozen results as diagnostic baselines rather than declaring them full
reference-aligned boundary behavior.

[Complete input/trace evidence](../../.tmp/acsph-ramp-cap-2026-10-28/analysis-summary.json) ·
[Spacing/normals audit](../../.tmp/acsph-ramp-cap-2026-10-28/spacing-audit.json) ·
[Fresh controls](../../.tmp/acsph-spacing-control-2026-10-28/).

![Remaining-cap mechanism](../../.tmp/acsph-ramp-cap-2026-10-28/remaining-cap-mechanism.png)

## Short collision-only control: partial containment improvement

A separate200-step trajectory reaches0.088223747 s, adding only native collision
displacement to the unchanged local Michel shift; no cohesion, p/v remap, ghost
geometry or ramp change. At the last common recorded frame, t0.080244558 s:

|Observable|Native|Original ramp0.5|+collision displacement|
|---|---:|---:|---:|
|Deepest penetration [dx]|0.134|1.098|0.454|
|Particles deeper than1dx|0|1|0|
|KE [J/m]|13.61396|13.61708|13.61668|
|Fluid pressure material-row RMS vs native [Pa]|0|63.45|61.87|

Collision alone reduces early deep penetration while changing KE by only
−0.00293% relative to the local baseline at that frame. It does not recover
native containment/pressure, and is not uniformly better at every earlier
frame. This short diagnostic stops before impact and retains the incorrect
ghost geometry; it cannot qualify late energy, pressure peaks or the combined
corrected-geometry/collision behavior.

[Collision-only comparison](../../.tmp/acsph-collision-control-2026-10-28/comparison.json).
No second full-duration trajectory or production model/default change was made.

## Diagnostic costs and scope

- Exact1727-step physical prefix:842.83 s,488.03 ms/physical step, including
  full payload captures, eight same-state controls, video and startup caps;
  not a clean throughput comparison.
- Original instrumented eager target solves:6.27/5.67 s. Bounce-off target
  solves:0.354/0.333 s.
- Fresh original graph reload:5.33/5.29 s; corrected-normal eager control:
  0.325/0.324 s. Includes observation/setup; do not infer graph/eager throughput.
- Collision-only200-step trajectory:164.16 s,820.81 ms/step, heavily weighted
  by the original capped startup. No stop/divergence; CUDA graphs retained.

All GPU controls were serial. Tiny same-input solves explicitly omit video;
their physical prefixes record VisPy movies. All jobs reached their declared
endpoints and terminal passed states; no new GPU run remains queued.

## Validation and reproduction

The offline analyzer checks UID/kind ordering, identical physical clocks,
positive masses, finite recorded physical x/v/p and10,000 convergence receipts.
All1001 full-run VisPy video frames decode; the last recorded frame at≈1.917 s
and generated scientific figures were inspected. The bounded physical-prefix
and collision-control movies also decode completely (174 and21 frames); their
last recorded frames were inspected. Physical endpoint1.91842 s
is present in the snapshots/H5. Offline analysis takes about40–60 s, with no
GPU simulation. Maintained scripts:
`analyze_ramp_qualification.py`, `audit_collision_shift.py`,
`probe_collision_shift.py` under `scripts/acsph_reference`.

## Follow-up: corrected-initialization full replay launched

User authorized a fresh full dam-break replay after this investigation. The
new campaign `.tmp/acsph-lobovsky-volume-ghost-2026-10-28` uses an explicit
`--ghost-spacing volume` fixture context during initialization, restoring all
temporary hooks before simulation starts. It corrects both the normal mollifier
and ghost-node depth while retaining the existing hybrid placement algorithm.
Shared boundary source code is untouched. Ramp0.5 and allother baseline controls
remain fixed; the collision shift is not included, keeping geometry isolated.

SixCPU geometry tests pass, including rho1/rho1000 invariance in both hybrid
and gridsnap paths. Full startup verifies0.0007500000065m spacing, exactly flat
near-corner wall normals,320,000 fluid particles/180kg and all declared FP32
physical IC checks. All10,000 native timesteps are queued in the active serial
GPU job, with video/watchdogs and automatic H5 export/native comparison. Results
are pending; initial geometry repair is not yet a pressure/trajectory verdict.

### Corrected-initialization replay completion

The new run completed all10,000 timesteps through1.918419918 s on2026-10-28,
with no early stop/divergence and CUDA graphs on every step. Wall time6364.86s
(636.49ms/step), versus6681.64s for the preceding ramp0.5 run. Capped steps
fall214→133; native has142 under its different stopping semantics. EndpointKE
is52.163J/m versus45.257 native (+15.26%, versus+20.51% before the spacing
correction). This is an endpoint observation, not a physical-accuracy verdict.
Viewer export and the193-frame native comparison passed; detailed containment,
pressure-transient and phase-energy review remains pending. GPU is idle.

[Comparison](../../.tmp/acsph-lobovsky-volume-ghost-2026-10-28/analysis-native/report.md) ·
[Completion receipt](../../.tmp/acsph-lobovsky-volume-ghost-2026-10-28/completion-receipt.json).
