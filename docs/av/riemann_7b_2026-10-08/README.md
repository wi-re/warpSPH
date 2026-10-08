# AV_PLAN Phase 7b: first-order Riemann dissipation, validation (2026-10-08)

`ViscosityTerms.RiemannDissipation` (`modules/dissipation/pi/terms.py::riemannDissipation`): `Pi_ij = (p*(w) - p*(0)) (1/rho_i^2 + 1/rho_j^2)`,
left state j, right state i, `w` the (reconstructed) pair velocity along `x_ij / r`, no alpha switch. Configs `riemann*` in
`scripts/av_report.py` (group `phase7b`). Full profile, video on; reference rows from `results/av_sweep_2026-10-07_17-30`
(the Phases 3-6 sweep). The raw reports are the `report_*.md` files here; final frames of the limited run are in `frames/`.

| config | pair velocity | solver |
|---|---|---|
| riemann | raw | Adaptive |
| riemannLimited | limited (Frontiere limiter) | Adaptive |
| riemannLimitedB2 | limited + Balsara p = 2 | Adaptive |
| riemannAcoustic / TSRS / HLLC | raw | as named (Sod 1D only) |

| case | metric | riemann | riemannLimited | riemannLimitedB2 | gsAV (raw, alpha=1, beta=2) | gsAVSLR (limited) | Rosswog limited coupled (default) |
|---|---|---|---|---|---|---|---|
| Sod 1D | L1(v_x) | 0.00369 | 0.00389 | 0.00369 | 0.00369 | 0.00385 | 0.00431 |
| Sod 1D | P contact spike | 0.069 | 0.063 | 0.068 | 0.076 | 0.068 | 0.063 |
| Sod 2D | L1(v_x) | 0.0155 | 0.0176 | 0.0155 | 0.0201 | 0.0170 | 0.0165 |
| Sod 2D | P contact spike | 0.071 | 0.079 | 0.071 | 0.069 | 0.078 | 0.078 |
| Noh (1D) | post-shock rho error | -0.00095 | -0.0027 | -0.00095 | -0.00076 | 0.00089 | 0.00082 |
| Gresho | L1(v_phi) | 0.210 | 0.0906 | 0.0910 | 0.210 | 0.0899 | 0.0603 |
| Gresho | peak speed (exact 1.0 at t=0, ~0.7 expected here) | 0.345 | 0.702 | 0.678 | 0.348 | 0.707 | 0.816 |
| Sedov 2D | peak rho ratio (exact 4) | 2.32 | 2.46 | 2.35 | 2.10 | 2.28 | 2.28 |
| Sedov 2D | shock radius error | -0.020 | -0.023 | -0.021 | -0.013 | -0.015 | -0.015 |
| Sedov 2D | energy drift | 5.2e-4 | 5.2e-4 | 5.2e-4 | 5.2e-4 | 5.2e-4 | 5.1e-4 |

Sod 1D solver comparison (raw velocity, `report_sod1d_all_solvers.md`): Acoustic / Adaptive / TSRS / HLLC differ by < 4 % on every metric.

## Reading

- The first-order Riemann term is on par with the existing AV rows everywhere it was run: it does not beat the Phase 4 / default rows on
  Gresho, it is the best row on Sod 2D L1(v_x) (raw 0.0155 vs gsAV 0.0201), and Sedov's shock-radius error is slightly worse (-0.020 / -0.023
  vs -0.015). As with every other term, the limited velocity reconstruction is what fixes Gresho (0.210 -> 0.091); the Riemann part itself
  changes little. Sedov's 5e-4 energy drift is the known RK2 time-integration error (OPEN_PROBLEMS §18), unchanged across rows.
- The solver choice barely matters on these cases (Sod 1D < 4 %); the quadratic (shock) term of TSRS / Adaptive is small at these Mach numbers.
- Frames (`frames/`): Noh clean (the usual wall-heating dip at the origin), Gresho vortex kept with the same lattice disorder as the other limited
  rows, Sedov shock without fliers. No velocity alarm in any run.

## Defects found on the way

- The first `riemann` run NaN'd at step 0 on Noh and Sedov (`report_riemann_raw_PRE_FIX_noh_sedov_diverged.md`): the term divided
  `dp / (|w| + 1e-12 c_bar)`, which is 0/0 for a cold-gas pair at relative rest (`c = 0`). Fixed by an absolute epsilon;
  `report_riemann_raw_noh_sedov_after_fix.md` is the re-run.
- `avEnergyLinear/Quadratic` (`dissipation.avPower`) books a Riemann term entirely as linear: it splits by the `C_l` / `C_q` coefficients.
- TRRS alone overflows (`inf`) for a pair of cold states (p ~ 1e-30): its `(...)^(1/z)` with `1/z = 7`; `Adaptive` never selects it there.
