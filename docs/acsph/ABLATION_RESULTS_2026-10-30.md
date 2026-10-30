# Lobovsky cumulative ablation: intermediate results — 2026-10-30

Seven of nine new original-IC trajectories completed all 7,641 steps to 1.530014125 s. Row 08 stopped on the existing motion-coverage guard; row 09 is still advancing. The older baseline and fully enabled endpoint are reused qualified trajectories. This is an intermediate result, not campaign completion.

All compared accepted physical clocks match exactly. All new trajectories use the same supplied physical ICs, FP32, viscosity, volume ghosts, consistent wall closure and bounce ramp 0.5. Every row adds its named change to all preceding rows. Ringing RMS removes a 1 ms Gaussian trend; pressure swing is raw peak-to-peak. Caps and pressure metrics below refer to the complete 1.44–1.53 s window. Recorded solve costs average each available trajectory prefix and are shared-machine observations.

| Stage added cumulatively | P1 RMS [kPa] | P4 RMS [kPa] | P1 swing [kPa] | Capped steps | Mean solve [ms] | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 00-baseline | 1.22 | 1.57 | 23.00 | 11 | 362.47 | reused |
| 01-static-geometry | 1.20 | 1.69 | 17.88 | 5 | 363.31 | completed |
| 02-accepted-fields | 1.22 | 1.55 | 14.88 | 6 | 364.36 | completed |
| 03-source-shift | 2.33 | 2.64 | 43.41 | 20 | 373.10 | completed |
| 04-pressure-remap | 3.16 | 3.33 | 41.40 | 44 | 389.15 | completed |
| 05-velocity-remap | 5.54 | 4.51 | 88.09 | 14 | 381.95 | completed |
| 06-warmup | 2.77 | 2.44 | 51.11 | 18 | 410.94 | completed |
| 07-source-operators | 5.30 | 4.83 | 76.99 | 30 | 545.87 | completed |
| 08-last-stage-metric | — | — | — | — | 540.18 | failed |
| 09-pressure-gate | — | — | — | — | — | pending |
| 10-plateau-trim | 14.09 | 12.57 | 232.30 | 48 | 660.04 | reused |

## Findings

* Preserving exact static wall/ghost geometry and direct accepted-field handoff leave late ringing essentially unchanged: P1 RMS 1.222 → 1.200 → 1.219 kPa.
* The first declared screening flag is row 03, the dynGPUSPH shift geometry plus coherent BDF history. Relative to row 02, P1 RMS rises 1.219 → 2.325 kPa (1.91×), raw swing 14.883 → 43.411 kPa (2.92×), and caps 6 → 20. It triggers the cap criterion; the RMS increase itself is below the declared 2× threshold. This coupled block needs subdivision before attributing one formula.
* Pressure remapping raises P1 RMS to 3.159 kPa and caps to 44. Velocity remapping raises RMS to 5.540 kPa while caps fall to 14: fewer capped steps alone do not imply quieter pressure.
* The first-two-iteration half-clock warmup reduces P1 RMS to 2.772 kPa; the coordinated dynGPUSPH spatial-operator block raises it to 5.300 kPa. The response is non-monotone and may include interactions and chaotic trajectory differences.
* Row 08 changes only the convergence metric location/norm policy from row 07. It accepted 7,224 steps through 1.472652582 s, then rejected attempt 7,225 at iteration 9: certified motion 2.256215 mm versus the 2.250000 mm bound. Input/trial and rendered accepted prefix are retained. The process subsequently exited with signal 11; the saved numerical guard exception precedes that exit. This establishes a coverage-guard stop, not a demonstrated physical blow-up or missing-neighbor event. Its incomplete late window has no comparable full-window RMS.
* The fully enabled retained endpoint has P1 RMS 14.092 kPa and raw swing 232.297 kPa, versus baseline 1.222 and 22.996 kPa. Row 09 is needed before isolating the final pressure-gate versus plateau/trim effects; neither is exonerated or blamed by the missing comparison.
* Through the primary window, mean KE is 72.42 J/m for the baseline, 72.62 after warmup, 69.35 after the spatial block and 69.29 for the fully enabled endpoint. The spatial block changes bulk dynamics as well as ringing. Closer dynGPUSPH agreement is not evidence of physical accuracy.
* Completed new rows cost 58.6–81.7 minutes each, including output. Mean solve cost rises from about 362 ms at the baseline to 546 ms with the spatial block and 660 ms for the retained full endpoint over this shorter 7,641-step comparison. These differ from earlier 8,395-step cost averages because their support differs.

## Pending run and evidence

Latest observed supervisor update: 2026-10-30T07:33:18.707981+00:00. physical-09-pressure-gate reached t=0.5192189626845426 s; no completion result yet. The existing serial queue continues unchanged.

Primary evidence: `.tmp/acsph-ablation-2026-10-29/analysis-through-08-last-stage-metric/{summary.json,report.md}`, per-row `campaign-row-status.json`, and row 08 `failure/receipt.json`. The automatic analysis labels row 09 pending because it had not started when that immutable report was generated; the live supervisor shows it running.

[Screening figure](figures/ablation-screening-interim-2026-10-30.png) · [Raw adjacent pressure traces](figures/ablation-adjacent-pressure-interim-2026-10-30.png) · [Design and coupled groups](ABLATION_LADDER_2026-10-29.md).

Next: complete row 09, then subdivide or reverse the shift/history block in a confirming control. Keep production defaults and coverage guards unchanged pending that evidence. No new physical run was launched for this results review.

## Campaign completion receipt — 2026-10-30

The supervisor finished at08:52:29 UTC. Eight of nine new original-IC rows
completed7641steps; row08 remains the only failed row. Row09 pressure gate
completed1.530014125s: late P1 RMS4.991251kPa, P4 RMS4.389699kPa,
110 capped steps, mean solve708.456ms over7641steps. Its final automatic
analysis is `.tmp/acsph-ablation-2026-10-29/analysis-through-09-pressure-gate/`.
The fully enabled retained plateau/trim endpoint therefore has2.823× its P1
ringing RMS while reducing caps110→48. These remain cumulative trajectory
comparisons; a confirming control is needed before attributing one mechanism.
The earlier table and figures above are retained as the explicitly intermediate
snapshot generated before row09 finished. No new physical run was launched for
this completion record. The subsequent Michel queue is held for integration.
