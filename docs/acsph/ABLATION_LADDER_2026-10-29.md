# Cumulative Lobovsky ablation — 2026-10-29

The user requested progressive runs to locate the increase in pressure
oscillation/convergence cost identified by the
[retained before/after comparison](HISTORICAL_COMPARISON_2026-10-29.md).
This ladder starts from the quieter consistent-wall-closure/Michel run and
adds one declared functional change at each stage. It ends at the existing
named dynGPUSPH compatibility path. No production solver default or shared
boundary implementation is edited by these controls.

## Fixed setup and scope

Each new physical trajectory starts from the original supplied Lobovsky ICs:
320,000 fluid particles,32,176 physical static-wall particles, dx0.75mm,
support3mm, rho1000, original mass, constant nu0.0001608375, FP32, volume
ghosts, source-consistent wall velocity, smooth fresh bounce ramp0.5, source
startup BDF history and exact original native timesteps. No mass calibration,
relaxed tolerances, altered iteration cap, viscosity retuning or sparse
pressure measurements are introduced. epsilonV=-8 and the200-iteration cap
stay fixed; only the explicitly listed stopping logic changes.

The first pass ends at step7641, t=1.530014125s, covering the complete
1.44–1.53s large-ringing window and the earlier intervals. This avoids
spending another0.17s on the separate unresolved geometric stop near1.6375s.
It does not disable or enlarge any reference coverage bound. Current source
guards apply wherever the source operator block is active; older operator
rows retain their own neighborhood assumptions and all common physical
velocity/nonfinite/stall watchdogs. Guard stops remain a distinct outcome.

The retained older baseline and current endpoint already cover this target;
they are reused after short endpoint qualification. Nine intermediate rows
require new original-IC trajectories. There are no late restarts from visual
snapshots. Complete-history checkpoints before1.25s/1.44s and rolling late
checkpoints support subsequent fixed-input/restart investigation.

## Declared cumulative order

Every row includes all preceding enabled changes. `ablation_matrix.py`
enforces exactly one changed option between adjacent rows and rejects
non-cumulative configurations.

| Row | Added change | Interpretation |
| --- | --- | --- |
|00|Previous consistent-closure solver + Michel shift|Retained quieter full run; baseline endpoint qualified against its own frozen runtime.|
|01|Preserve static geometry|Keep exact stationary wall/ghost coordinates and offsets while retaining the old body velocity reset.|
|02|Direct accepted p/v handoff|Remove the derivative roundtrip; retain the existing surface/closure behavior.|
|03|Source shift geometry and its BDF history|Use compiled-source displacement/surface inputs/collision/cohesion/history; suppress all p/v field-remap assignments.|
|04|Pressure remap|Enable the source pressure-gradient transport and its associated pressure safeguards.|
|05|Velocity remap|Enable the compiled componentwise diagonal velocity transport.|
|06|First-two-iteration warmup|Half pseudo-clock for the first two iterations, with beta/k1/k2 and destination RK factors scaled consistently.|
|07|Coordinated source spatial operators|Replace wall/moment/surface/residual/frozen-cell operators and their accepted auxiliary-field handoff; keep accepted-state velocity stopping.|
|08|Last-stage residual metric|Use source last-RK-stage velocity residual and its double-precision norm instead of the accepted-state legacy metric.|
|09|Pressure stopping gate|Also require epsilonP < epsilonV+1; plateau/trim still disabled.|
|10|Plateau/trim policy|Enable source-inspired plateau/trim with stable fluid-ID selection; unchanged current reference numerical endpoint.|

The baseline already has the destination-row RK factors through its historical
fixture hook. They are not reintroduced as a purported new row. Row06 changes
their effective clock ratio for the two warmup iterations consistently with
the smaller pseudo timestep.

Rows03,06 and07 are explicitly coupled groups. Shift geometry and its BDF
history must be coherent; the smaller clock needs its associated coefficients;
the source closure/moment/surface kernels share auxiliary-field conventions.
A jump at one of these rows identifies that block, not a single formula.
The first block producing a notable change should be subdivided or reversed
in a confirming control. A cumulative ladder can contain interactions and
order dependence; it is not a complete factorial sensitivity study.

## Paper agreement audit — 2026-10-30

This ladder approaches the supplied dynGPUSPH implementation, not progressively
the published method. De Courcy §3.2 explicitly uses Michel shifting, Eqs55–57;
Eq58 specifies external displacement with Taylor remapping and Eq59 discusses
the corresponding position-history correction. §4.2 prefers external shifting.
The source collision/cohesion additions and changed coefficient are not the
published Michel prescription.

| Stage | Agreement with the published method |
| --- | --- |
|00 baseline|Closer on the Michel displacement law, but not a complete paper control: generic ACSPH finalization lacks the general p/v Taylor remap, and this comparison fixture already aligns several code-specific terms/RK choices.|
|01 static geometry|Implementation consistency for stationary walls, not a new published model.|
|02 accepted fields|Implementation consistency with accepting converged fields, not a new published model.|
|03 source shift/history|Displacement law differs from Eqs55–57. Correcting shift history is supported in principle by Eq59; the copied source history bookkeeping is not itself established as the correct general discretization of Eq59.|
|04 pressure remap|Taylor pressure transport is supported by Eq58. The source-specific extreme-pressure replacement and low-support pressure reset are additional behavior.|
|05 velocity remap|Eq58 requires the full velocity-gradient tensor. The supplied componentwise diagonal-only remap omits cross terms.|
|06 warmup|First-two-iteration half pseudo-clock is not specified in the paper.|
|07 source spatial block|Mixed: standard AC operators and Adami boundaries are published, but source tapers, extra momentum dissipation, auxiliary-field scheduling and cell selection are not collectively specified by those equations. Do not label the whole block paper-faithful.|
|08 last-stage metric|The paper defines the pseudo-velocity norm in Eqs47–48, but does not explicitly prescribe the source last-RK-stage sampling policy or FP64 reduction as separate requirements.|
|09 pressure gate|Not the published stopping rule: §3.1.3 explicitly uses one convergence metric, epsilonV.|
|10 plateau/trim|Not specified in the paper; dropping the largest5% of residuals changes the stopping test.|

Keep Michel shifting, full Taylor remapping and a separately verified Eq59
history treatment together when constructing a paper-faithful control. Do not
discard paper-required remapping merely because the source-specific version
in this ladder produces more ringing. Conversely, matching source behavior
does not qualify these code-specific additions as improvements to ACSPH.

Source: local `literature/decourcy2024_incompressible-delta-sph-artificial-compressibility.pdf`,
§3.1.3, §3.2, §3.3 and §4.2; supplied-source differences are traced in
`DYNGPUSPH_COMPARISON.md` (A/B audit and D6–D8/D15). No run or numerical
default changed for this audit.

## Qualification before expensive rows

Eleven CPU checks verify cumulative deltas, invalid combinations, physical-time
invariance of the half-clock control, field-remap suppression without changing
geometry/history, and independent manufactured pressure-gate/plateau behavior.

Serial GPU preflights run three original-IC physical steps per row with video:

* Baseline diagnostics must reproduce the retained historical run exactly;
  its snapshot must match an independent three-step replay of the old frozen
  runtime bitwise.
* The fully enabled endpoint must match the retained qualified current
  three-step snapshot and diagnostics bitwise.
* Every row must retain the identical physical IC arrays, finite fields,
  unchanged static geometry where enabled, all three expected steps and
  successful CUDA graphs.
* Separate three-step eager controls for the new warmup and source-operator
  combinations must match the graphed endpoints bitwise.

Qualification failures stop the campaign before long runs. Their logs and
outputs remain. No gate is replaced by an enlarged numerical tolerance.
These preflights validate plumbing and endpoints; they do not establish late
pressure quality, which requires the actual original-IC trajectories.

## Measurements and interpretation

All pressure/convergence comparisons require exact accepted physical clocks.
Each completed stage is compared with its immediate predecessor, baseline00,
current10 and dense accepted-state dynGPUSPH. The primary screen is the
completed1.44–1.53s window, with earlier0.4–0.46s,0.65–0.95s,0.95–1.05s and
the broader1.25–1.53s interval retained as secondary windows.

Record raw pressure minimum/maximum/peak-to-peak at P1–P4; detrended RMS with
1ms/5ms Gaussian trends after0.1ms uniform interpolation, excluding3sigma at
window edges; pressure-trace differences versus native; KE; pseudo-iteration
counts/caps; recorded solve costs; guard/velocity/nonfinite/stall failures;
and actual completion support. Raw plots retain accepted pressure without
filtering/clipping. Native FP64/hard-lagged-bounce/source-trim differences
remain explicit; native is a compatibility reference, not a pressure-accuracy
oracle.

Flag the first adjacent row where P1 or P4 detrended RMS more than doubles
and rises by more than1kPa, or where caps more than double and increase by
at least10. A physical row stop is also flagged. These are declared
descriptive screening criteria, not confidence tests or numerical model
parameters. Small absolute early noise cannot trigger a jump solely from a
large ratio. Incomplete windows are marked unavailable, rather than compared
as though complete. A numerical row failure remains a matrix result; other
independent original-IC rows can proceed after qualification has passed.

Stopping policies intentionally differ after08/09/10, so counts alone do not
prove a worse numerical map. Costs are shared-system recorded observations,
not isolated hardware benchmarks. Bulk KE and pressure amplitude are reported
separately; a quieter trace or closer dynGPUSPH match alone is not promotion
evidence.

## Runtime and retained artifacts

Campaign root: `.tmp/acsph-ablation-2026-10-29/`.

`manifest.json` pins the prior qualified runtime and exact native input/clock
files. A copied runtime receives only the fixture ablation scripts. The old
baseline runtime is hash-verified independently. Root`status.json` mirrors
the current phase; `qualify-status.json` retains preflight gates and
`run-status.json` tracks serial physical rows. A `STOP` file cancels only this
supervisor's owned descendants.

All physical runs enable VisPy video, streamed progress, a100m/s velocity
limit, advancing-clock/stall and wall watchdogs. Each row has its own source
option receipt, progress, snapshots, per-step timings, video and late complete
checkpoints. A solve exception captures its input and available rejected trial
without modifying the safety check. Numerical failures preserve accepted
prefixes and get an explicit failure receipt and rendered prefix where possible.

After every new row, an offline report compares the currently available
ladder and identifies any first notable adjacent change. Long trajectories
are serialized and wait for an idle GPU before each row. Existing runs and
their failed receipts remain unchanged. Expect several hours for the nine
new rows; the exact cost depends on caps introduced by each hybrid.

```bash
python scripts/acsph_reference/campaign_lobovsky_ablation.py \
  --root .tmp/acsph-ablation-2026-10-29 --mode prepare
python scripts/acsph_reference/campaign_lobovsky_ablation.py \
  --root .tmp/acsph-ablation-2026-10-29 --mode qualify
python scripts/acsph_reference/campaign_lobovsky_ablation.py \
  --root .tmp/acsph-ablation-2026-10-29 --mode run
```

Use the warp conda Python. `prepare` requires a fresh root, `qualify` must pass
before`run`. The implementation is in `ablation_matrix.py`,
`ablation_controls.py`, `run_lobovsky_ablation.py`,
`campaign_lobovsky_ablation.py` and `analyze_lobovsky_ablation.py` under
`scripts/acsph_reference/`. No completion or late result is claimed by this
setup document until the corresponding live campaign receipts exist.


## Launch qualification receipt

All14 serial three-step physical preflights pass, including bitwise historical
baseline/current endpoint and graph/eager controls. Eleven CPU checks pass.
The first historical-schema gate found two extra null diagnostic fields
(`epsilonP`, `exitReason`); no numeric value differed. The original failed
receipt remains, only those missing null fields are normalized, and successful
physical preflights were reused under unchanged source hashes. Numerical
snapshots at both ends match bitwise. The analysis smoke test retains nine
pending stages and makes no false adjacent-jump claim across them.

The nine intermediate long rows have been launched serially. Root`status.json`
now tracks physical rows. Unknown implementation/output failures halt the
queue; recognized numerical guard/budget failures remain explicit row results.
No late ablation result is available at launch.
