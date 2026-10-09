# Sloshing tank (SPHERIC TC10) 7 s, nx = 200: δ⁺ particle walls, δ⁺ analytic walls, DFSPH — 2026-10-09

Evidence for [ANALYTIC_BOUNDARIES_PLAN.md](../../../ANALYTIC_BOUNDARIES_PLAN.md) / [ANALYTIC_BOUNDARIES_PORT_SURVEY.md](../../../ANALYTIC_BOUNDARIES_PORT_SURVEY.md): the first full-length run of the warpSPH-hosted analytic δ⁺ (the gate for retiring the boundaries repo's `DeltaSPH2D`).
Commit base 4b01541 plus the uncommitted `--wallRepresentation analytic` for `sloshingTank`; run with `examples/sloshingTank/run_sloshingTank.py --scheme {wcsph, wcsph --wallRepresentation analytic, dfsph} --tLimit 7`.
Raw output (videos, series, logs) stays in `examples/sloshingTank/output/analytic_ab_2026-10-09/` (gitignored); the overlay is `scripts/plot_sloshingOverlay.py`.

| run | smoothed Sensor-1 peaks [Pa] at t [s] | KE max | density range | steps / wall time |
|---|---|---|---|---|
| measured (SPHERIC) | ~3700 @ 2.375, ~3900 @ 4.05, ~2700 @ 5.68 (raw record) | — | — | — |
| δ⁺, particle walls, Wendland4 (case default) | 4877 @ 2.343, 5157 @ 4.005, 5277 @ 5.557 | 0.0340 | 0.82 – 1.33 | 70 001 / 1201 s |
| δ⁺, analytic walls, Wendland2 | 4549 @ 2.351, 5041 @ 4.016, 4835 @ 5.623 | 0.0288 | 0.75 – 1.34 | 70 001 / 2590 s (a parallel GPU job ran: not comparable) |
| DFSPH (`divergenceFree`), particle walls | 1911 @ 2.787, 4631 @ 4.173, 3779 @ 5.795 | 0.0336 | 0.33 – 1.03 | 15 555 / 237 s |

(Peaks are the 10 ms Gaussian-smoothed values of `scripts/plot_sloshingOverlay.py`. Numbers quoted earlier in the session came from a slightly different smoothing (window edges) and differ by 2-4 %; these are the ones to use.)

* All three runs finish, no velocity alarm. The kinetic energy of the three agrees through 7 s (bottom panel): the bulk flow is the same.
* δ⁺, particle vs analytic: impact times within 8 ms (1st), 11 ms (2nd), 66 ms (3rd, the analytic run is the closer one to the measurement); smoothed peaks 2–9 % lower with analytic walls. Both overshoot the measured peaks by about 25–100 %, as the earlier particle-wall record did.
* The comparison is not a single-factor A/B: kernel (C4 vs C2, forced by the wall shifting term), wall treatment, and Sensor 1 (nearest wall particle vs Gaussian fluid probe within 2 cm) all differ. The analytic run shows short probe spikes between impacts (t ≈ 3.0, 4.7, 6.4 s); the frame at t = 3.03 has isolated near-wall particles (not checked further).
* DFSPH: first impact late and low, later impacts a train of spikes; the frame shows a sparse, stretched layer with density holes. The known free-surface de-densification of `divergenceFree` at nx = 200 (DFSPH_IMPROVEMENT_PLAN Part 23).
* Frames: `frame_particles_t3.03.png`, `frame_analytic_t3.03.png` (same instant, the wave overturn of the second cycle), `frame_dfsph_t4.87.png`.

> **Correction (2026-10-09, later): comparison with the measurement.** The peaks quoted here and in the messages of the day compared 10 ms-smoothed *simulated* peaks with the *raw* measured peaks (3.7 / 3.9 / 2.7 kPa). The measured record smoothed the same way peaks at 1.80 / 2.32 / 1.17 kPa
> (t = 2.390, 4.066, 5.697 s), so the delta+ analytic run is +153 % / +117 % / +313 % and 39-74 ms early, not "25-60 ms early with larger peaks" against the raw record. See `../omni_sloshing_fixed_2026-10-09/README.md` for the like-with-like table.
