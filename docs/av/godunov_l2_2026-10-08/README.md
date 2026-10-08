# GODUNOV_SPH_PLAN layer 2: simplified Godunov SPH (2026-10-08)

`schemes/gsph.py` + `modules/godunov/wp_gsph.py`: `a_i = -sum m_j p*_ij [grad W(h_i)/rho_i^2 + grad W(h_j)/rho_j^2]`,
`du_i/dt = -sum m_j p*_ij (u*_ij - u_i)(n . F)` with `(p*, u*)` from `modules/riemann` (Adaptive solver), no viscosity switch, no artificial
viscosity, no conductivity (Cha & Whitworth Case 3 / Puri Eq. 15 / Iwasaki Eq. 24). Full profile, video, no velocity alarm; final frames of the
second-order run in `frames/`. Configs (`av_report` group `gsph`): `gsphO1` first-order states; `gsphO2v` second-order velocity only; `gsphO2` second-order
velocity + reconstructed `(rho, P)` (layer 1). AV rows from the Phases 3-6 sweep and `docs/av/riemann_7b_2026-10-08/`.

| case metric | gsphO1 | gsphO2v | **gsphO2** | riemannLimited (AV) | gsAVSLR (AV) | Rosswog limited coupled (default AV) |
|---|---|---|---|---|---|---|
| Sod 1D  L1(v_x) | 0.01197 | 0.00406 | **0.00286** | 0.00389 | 0.00385 | 0.00431 |
| Sod 1D  P contact spike | 0.0056 | 0.0060 | **0.0104** | 0.0626 | 0.0681 | 0.0630 |
| Sod 1D  entropy (A) contact spike | 0.093 | 0.065 | 0.085 | 0.018 | 0.021 | 0.024 |
| Sod 1D  energy drift | 7.1e-5 | 6.5e-5 | 5.4e-5 | 2.0e-6 | 2.3e-6 | 2.1e-6 |
| Sod 2D  L1(v_x) | 0.0372 | 0.0254 | 0.0167 | 0.0176 | 0.0170 | 0.0165 |
| Sod 2D  P contact spike | 0.049 | 0.045 | 0.053 | 0.079 | 0.078 | 0.078 |
| Noh  post-shock rho error | -0.0003 | -0.0006 | -0.0007 | -0.0027 | +0.0009 | +0.0008 |
| Gresho  L1(v_phi) | 0.272 | 0.133 | 0.130 | 0.0906 | 0.0899 | 0.0603 |
| Gresho  peak speed (~0.7 expected) | 0.17 | 0.58 | 0.61 | 0.70 | 0.71 | 0.82 |
| Sedov  peak rho ratio (exact 4) | 2.15 | 2.32 | 2.41 | 2.46 | 2.28 | 2.28 |
| Sedov  shock radius error | -0.044 | -0.043 | -0.048 | -0.023 | -0.015 | -0.015 |
| Sedov  energy drift | 5.3e-4 | 5.3e-4 | 5.7e-4 | 5.2e-4 | 5.2e-4 | 5.1e-4 |

## Reading

- **It works without any artificial viscosity**, on shocks (Sod, Noh, Sedov) and a smooth vortex (Gresho), stable and finite in all five cases at all three orders.
- **The pressure blip at the contact is essentially gone** (Sod 1D P spike 0.010 against 0.063-0.068 for every AV row, a factor 6) and the 1D velocity error
  is the best of any row (0.0029 against 0.0037-0.0043). The price is the entropy: the contact's entropy spike is 4x larger (0.085 vs 0.018-0.024): the
  Riemann solution mixes internal energy across the contact (Murante: "thermal diffusion"; Puri: the dissipative heating).
- **Second order is essential**, as Murante found: first-order states are 4x worse on Sod 1D and 2x worse on Gresho / Sod 2D than second order. Reconstructing
  `(rho, P)` on top of the velocity matters here (Sod 1D 0.0041 -> 0.0029, Sod 2D 0.025 -> 0.017), unlike in the Phase 7b dissipation term, where it was a null result (docs/av/godunov_l1_2026-10-08).
- **Weaker than the best AV rows on smooth rotation and the blast radius**: Gresho L1 0.13 against 0.09 (limited AV) and 0.06 (the default); the Gresho frames show a more
  disordered lattice than the AV runs (no viscous regularisation of the particle distribution); the Sedov shock radius is 3x further off (-0.048).
- **Energy drift is 25x the AV rows' on Sod 1D** (5e-5 vs 2e-6; still inside the 1e-4 budget): the papers' time-centred `v_i + a_i dt/2` is omitted
  (Puri Eq. 36's higher-order term). Sedov's 5e-4 is the known RK2 error, the same for every row. Worth measuring the time-centred variant.
- **The simplified form keeps the standard SPH density-gradient inconsistency**: `tests/test_gsph.py` shows that with constant pressure its force equals the standard
  SPH force to 2e-3 (float32), i.e. it cannot fix Cha et al. (2010)'s spurious repulsion. Layer 3 (Inutsuka's convolution form) is the scheme for that.
- Noh shows the usual wall-heating dip at the origin (Puri 5.6: it persists for every Godunov solver).

## Build checks

`tests/test_gsph.py` (10, also float64): momentum and total-energy conservation to 2e-4 (float32) / 1e-10 (float64) for the acoustic, adaptive and HLLC solvers at first and
second order; rest state and the standard-SPH identity above; Sod 1D finite, opening, energy within 1e-3. `scripts/gradcheck_godunov.py`: 6/6 (three solvers x two
orders, the Jacobian and the `(rho, P)` gradients as inputs).
