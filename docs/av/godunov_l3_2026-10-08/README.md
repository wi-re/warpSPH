# GODUNOV_SPH_PLAN L3: Inutsuka (2002) Godunov SPH, 2026-10-08

`InutsukaGSPH` (`schemes/gsphInutsuka.py`, `modules/godunov/wp_inutsuka.py`): symmetrised Gaussian density, the convolution-form force and energy with the
pair volumes `V_ij^2(h)` and interface position `s*` (cubic Hermite `V(t)`), Riemann `p*, u*` on the reconstructed interface states, no artificial viscosity, no
conductivity, adaptive support, Gaussian width `h_G = eta (m/rho)^(1/d)` (Inutsuka Eq. 81, `eta = 1`). Full profile, video, no velocity alarm, float32, RK2.
`av_report --config godunovL3` (`report.md`, final frames in `frames/`). Configs: `gsphO2H` simplified GSPH with the harmonic state limiter (the layer-2 reference);
`inutsuka` Inutsuka with the harmonic limiter; `inutsukaM` van Leer monotonised; `inutsukaS` harmonic + first-order shock switch (C = 3); `inutsukaO1` first-order states.

**Config names in this table are those of the run** (before the defaults changed the same evening): `inutsuka` = harmonic limiter, no switch (now `inutsukaN`), `inutsukaS` = harmonic + switch (now
the default, `inutsuka`), `inutsukaM` / `inutsukaO1` = no switch (unchanged: the configs now pin `shockSwitchC=0`).

| case metric | gsphO2H | inutsuka | inutsukaM | inutsukaS | inutsukaO1 |
|---|---|---|---|---|---|
| Sod 1D  L1(v_x) | 0.00439 | 0.00297 | 0.00331 | 0.00314 | 0.0147 |
| Sod 1D  P contact spike | 0.0090 | 0.0383 | 0.0382 | 0.0423 | 0.0285 |
| Sod 1D  entropy contact spike | 0.081 | 0.040 | 0.034 | 0.074 | 0.091 |
| Sod 2D  L1(v_x) | 0.0155 | 0.0185 | 0.0195 | 0.0196 | 0.0431 |
| Noh  post-shock rho error | -0.0032 | +0.0018 | -0.0004 | +0.0003 | +0.0004 |
| Noh  energy drift | 2e-6 | 7.5e-3 | 1.3e-2 | 2.5e-5 | 1.9e-5 |
| Gresho  L1(v_phi) | 0.109 | 0.0909 | **0.0760** | 0.0909 | 0.283 |
| Gresho  peak speed (~0.7) | 0.66 | 0.69 | **0.73** | 0.69 | 0.15 |
| Sedov  peak rho ratio (exact 4) | 2.65 | 2.87 | (blown up) | 2.10 | 2.00 |
| Sedov  shock radius error | -0.043 | -0.074 | (blown up) | -0.061 | -0.066 |
| Sedov  energy drift | 8e-4 | 4.5e-2 | 121 | 6.6e-4 | 3.5e-4 |
| KH (nx 128)  A(t = 1.5), reference 0.148 | 0.1044 | 0.0983 | 0.1020 | 0.0983 | 0.0030 |

For the AV rows' numbers on the same cases see `docs/av/godunov_l2_2026-10-08/` and the bake-off.

## Stability probe (which Gaussian width / volume / switch survives the cold shocks)

`scripts/probe_inutsukaStability.py`, full Noh and Sedov (`stability_probe.txt`): Noh post-shock density error | Sedov peak density ratio.

| variant | Noh | Sedov |
|---|---|---|
| eta 0.75, cubic V | +0.0012 | 3.06 |
| **eta 1.0, cubic V (default)** | +0.0018 | 2.87 |
| eta 1.33, cubic V | diverged @ 2299 | diverged @ 108 |
| eta 1.0 + shock switch | +0.0003 | 2.10 |
| eta 1.0, linear V | diverged @ 1923 | 2.67 |
| eta 0.75, linear V | +0.0008 | 3.03 |
| earlier chain, `h = support/3` (about 1.3 spacings), all variants | diverged | diverged without the switch |

## Reading

- **Pairing is a width problem.** A Gaussian wider than the particle spacing leaves the pair force still rising at the nearest neighbours, and cold shocks collapse into pairs
  (eta 1.33 dies within 108 Sedov steps; the earlier `h = support/3` chain diverged on Noh for every variant). `eta <= 1` (Inutsuka's own Eq. 81) is stable; the cubic
  interpolant of the pair volume is needed at `eta = 1` (the linear one diverges on Noh). The price: the Fig. 1 density-jump force is not removed at that width
  (`docs/av/godunov_densityJump/`), the wide Gaussian that does halve it does not survive shocks.
- **Where it wins** (the smooth and shear tests, with no AV and no switch): Gresho L1 0.091 (monotonised 0.076) against 0.109 for the simplified GSPH with the same limiter;
  KH amplitude 0.098-0.102 (0.148 reference), comparable to the best AV rows and to the layer-2 scheme (0.104); Sod 1D velocity 0.0030 (AV rows 0.0037-0.0043; simplified GSPH with the ratio limiter 0.0029).
  The KH roll-up (`frames/inutsuka_kelvinHelmholtz_final.png`) is clean, the contact does not open a gap.
- **Where it loses:** the Sod contact pressure spike is 4x the simplified scheme's (0.038 vs 0.009; the AV rows 0.063-0.068, `docs/av/godunov_l2_2026-10-08/`), the entropy spike twice the AV rows' (0.040 vs 0.018-0.024), Sod 2D velocity slightly worse (0.0185 vs 0.0155),
  and **total energy is not conserved at strong shocks**: Noh 7.5e-3 and Sedov 4.5e-2 (the simplified scheme: 2e-6 / 8e-4). The pair operators conserve momentum and pair energy exactly (tests); the drift
  comes from the strongly non-uniform flow (RK2 error in `V_ij^2` and `s*` as the Gaussian widths vary) and is the same noisy pre-shock velocity visible in `frames/inutsuka_sedov_final.png`.
- **The first-order shock switch repairs all of it**: Sedov energy drift 4.5e-2 -> 6.6e-4, Noh 7.5e-3 -> 2.5e-5, a smooth Sedov profile (`frames/inutsukaS_sedov_final.png`) with a lower, smeared peak (2.10 vs 2.87), and
  *no change* on Gresho and KH (the switch is never triggered: rows identical to 4 digits). Sod: velocity unchanged, entropy spike doubles.
- **Monotonised limiter is not robust here:** best on Gresho (0.076) but the Sedov run is chaotic (`frames/inutsukaM_sedov_final.png`), energy drift 121. Use it only with the switch.
- **Second order is essential** (first order: Gresho 0.28, KH does not grow), as for layer 2.
- **Time-centred energy (D) stays off**: the time-centred velocity in the energy equation (`timeCentredEnergy`, `gsphO2T` / `inutsukaT`) made the
  energy drift about 30x worse on the simplified scheme (Sod 5e-5 -> 1.5e-3, Sedov 8e-4 -> 2e-2): the repo's integrators evaluate `f(state)` without an explicit `dt`, so the `dt/2` is the step size passed to
  the RHS in every stage (cause of the drift not isolated). Negative result, left in the code as an opt-in.

## Decision (user, 2026-10-08): adopted

`InutsukaGSPH` default: eta 1.0, cubic V, `VanLeerHarmonic`, first-order shock switch C = 3 **on** (it costs nothing on smooth flows and is what makes the scheme conserve energy and survive strong
shocks). `GSPHConfig` default state limiter: `VanLeerHarmonic` (was `PairRatio`; the `gsphO2` / `gsphO2T` av_report rows pin `PairRatio`).
