# AV baseline (M0c) — 2026-09-30

Produced by `scripts/av_report.py --profile full --repeat 2` on `dev` after AV_PLAN S5 (`2d06210`). Host scheme
**Monaghan**, kernel B7, RK2, `C_l = 1`, `Price2012_98`. Three column families:

1. `none / cullenDehnen2010 / readHayfield2012` at the Monaghan default **`C_q = 0`** — all ten cases (`--config baseline`).
2. The same three at **`C_q = 2`** — `sod noh sedov gresho` (`--config baselineQ`).
3. The AV_PLAN S3 switches `balsara1995 / colagrossi2004 / morrisMonaghan1997 / rosswog2000` at `C_q = 0` — `sod gresho`
   (`--config switchesS3`).

Videos: `results/av_M0c_*/runs/` (gitignored). **Reproducibility lock: all 50 (config, case) pairs are bit-identical across
two runs** (max relative difference 0); nothing diverged, no velocity alarm. `av_report.py --compare A B` is the
bit-for-bit instrument, `--merge` combines report directories.

## How this baseline was corrected twice

- **M0b** (harness): the first M0 applied the switch alpha range as case *parameters*, which only `sod`/`sodND` read, so
  Read-Hayfield ran in its designed 0.2-1.0 range on the Sod family only. Now applied to every case (`AVConfig.switchParams`).
- **M0c** (scheme bug, OPEN_PROBLEMS §16): the Monaghan viscous **heating lacked the 1/2** of
  `du_i/dt = 1/2 sum m_j Pi_ij v_ij . gradW_ij`, so it was exactly twice the kinetic energy the viscous force removes and
  every Monaghan-host run gained one dissipation's worth of energy (Sedov +47..68 %, Sod +0.6 %, ...). Fixed in
  `modules/dissipation/wp_dissipation.py`; the viscous force (so the meaning of `C_l`, `C_q`) is unchanged.
  **All Monaghan-host numbers before this are superseded** — including my earlier statements that Noh "still fails at
  `C_q = 2`" and that Sedov/Noh were "confounded": see below.

## Reading the numbers

1. **Energy now conserves**: `none` and Cullen-Dehnen show total-energy drift 0.0000 (round-off) on Sod 1D/2D/3D, Gresho,
   Yee, KH, Noh, and 5e-4 on Sedov (was 47-52 %). **Read-Hayfield does not**: 1.8e-3 (Sod), 6e-3 (Sod 2D), 1.5e-2 (Sod 3D),
   1.3e-2 (Sedov), and **+23 % on Noh at `C_q = 2`** — its entropy-dissipation term is not energy-conserving (new
   OPEN_PROBLEMS §17, uninvestigated). The budget column uses the new Monaghan bound 1e-4 (was 5e-3, sized to the bug).
2. **`C_q = 0` means no quadratic AV and, in cold gas, no AV at all** (`v_sig = C_l c`, `c = 0`): **Noh is degenerate at `C_q = 0`**
   (-50 % post-shock density: the two lattice halves stream through each other), independent of the switch.
   **At `C_q = 2` Noh passes the plan's +-3 % gate: `none` -0.08 %, Cullen-Dehnen +0.06 %** (was -23..-24 % with the heating
   bug) — the failure I had attributed to Phase 5A was §16. Read-Hayfield at `C_q = 2`: -16 % with the §17 energy gain.
   So **the baseline's `C_q = 0` default cannot run Noh (or any cold-gas shock) at all; `C_q > 0` is required there.**
3. **Sedov (3D, nx 40)**: peak rho/rho0 1.8-2.5 vs 4 (resolution-limited, never overshoots), shock-radius error 0.3-3 %.
4. **Sod** improved across the board with the fix: `L1(v_x)` none 0.0035, C&D 0.0052, R&H 0.0040; R4 density error < 1 %
   (was -3 %); contact P spike none 6 %, C&D 13 %, R&H 4.5 % (C&D is still the largest; `C_q = 2` brings it to 8 %).
5. **Gresho** is essentially unchanged (its thermal energy barely enters): `none` loses the vortex (L1 0.21, peak 0.34); among
   the switches the result is ordered by the **alpha floor**, because in smooth flow every one sits at it: Cullen-Dehnen
   (floor 0.02) 0.073; Morris-Monaghan / Rosswog2000 (0.1) 0.080; Read-Hayfield at its designed floor 0.2 0.101 (worse than
   C&D — my earlier "R&H marginally better" is withdrawn); Balsara 0.080, Colagrossi 0.073 (alpha from the limiter).
6. **Balsara == Colagrossi exactly on 1D Sod** (curl and trace-free shear are both zero in 1D). Morris-Monaghan's Sod peak alpha
   is 0.51 vs the paper's Eq. (19) no-decay ceiling ~0.67.
7. **Kelvin-Helmholtz** (nx 128, t = 1.5), mode amplitude vs McNally 14.79e-2: `none` 0.020, R&H 0.068, **C&D 0.098**.
8. **Not quality numbers:** `sod3d` runs at nx = 20 (15 steps; ~4 s/step at 580 neighbours — a perf question); `linearWave`
   runs at `A = 1e-4`, t = 1/4 crossing (the case default `A = 1e-6` is ~7x float32 noise); `rayleighTaylor` reports interface
   tips only; `wallMsPerStep` is wall-clock including setup (StepTimer not wired).

---

# Family 1 — `C_q = 0`, all ten cases

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.003493 | 0.06033 | 0.03273 | -0.0007077 | -0.003597 | -0.0001576 | 1 | 1 | 1 | 0.0107 | 0 | 0.02347 | 3.974e-06 | 16.91 | 30.03 |
| cullenDehnen2010 | 0.00524 | 0.1336 | 0.02183 | -0.0003972 | -0.002563 | -0.002588 | 0.08314 | 0.9462 | 0.224 | 0.01011 | 0 | 0.02191 | 1.358e-05 | 16.85 | 30.58 |
| readHayfield2012 | 0.004032 | 0.04523 | 0.0209 | -0.002983 | -0.006288 | -0.002056 | 0.2406 | 0.8755 | 1 | 0.01102 | 0 | 0.01693 | 0.001771 | 16.89 | 29.89 |

## sod: energy-drift budget (Group E)

OVER: readHayfield2012 (1.77e-03 > 1e-04)

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.01714 | 0.07428 | 0.02595 | -0.0394 | 0.001933 | 0.01302 | 1 | 1 | 1 | 0.002688 | 0 | 0.004688 | 7.642e-07 | 106.7 | 42.33 |
| cullenDehnen2010 | 0.0172 | 0.08507 | 0.01448 | -0.04128 | 0.0141 | 0.02215 | 0.2855 | 1.043 | 1 | 0.002376 | 0 | 0.003742 | 3.751e-06 | 106.7 | 37.78 |
| readHayfield2012 | 0.02319 | 0.05963 | -0.1137 | -0.03915 | 0.1038 | 0.01728 | 0.2666 | 0.8738 | 1 | 0.002804 | 0 | -0.002389 | 0.006336 | 106.7 | 43.33 |

## sod2d: energy-drift budget (Group E)

OVER: readHayfield2012 (6.34e-03 > 1e-04)

## sod3d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.06054 | 0.1153 | -0.1461 | -0.0908 | 0.2264 | -0.009444 | 1 | 1 | 1 | 0.01354 | 0 | -0.001246 | 4.472e-05 | 580.8 | 1277 |
| cullenDehnen2010 | 0.05873 | 0.1236 | -0.4098 | -0.08982 | 0.2136 | -0.02704 | 0.8288 | 1.638 | 1 | 0.01467 | 0 | -7.153e-07 | 4.591e-05 | 578.6 | 109.6 |
| readHayfield2012 | 0.07029 | 0.146 | -0.3654 | -0.07607 | 0.3503 | -0.1456 | 0.3903 | 0.8863 | 1 | 0.01102 | 0 | -0.1114 | 0.01511 | 579.4 | 116.7 |

## sod3d: energy-drift budget (Group E)

OVER: readHayfield2012 (1.51e-02 > 1e-04)

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.532 | 4 | 0 | -0.03165 | 0 | 1 | 1 | 1 | 0.4044 | 0 | -0.002139 | 0.0005215 | 550.2 | 100.2 |
| cullenDehnen2010 | 2.154 | 4 | 0 | -0.01112 | 0 | 1.799 | 1.996 | 1 | 0.4519 | 0 | 0.00514 | 0.0005298 | 550.2 | 116.3 |
| readHayfield2012 | 2.071 | 4 | 0 | -0.03366 | 0 | 0.826 | 1 | 1 | 0.15 | 0 | -0.03782 | 0.01258 | 547.7 | 158.4 |

## sedov: energy-drift budget (Group E)

OVER: none (5.22e-04 > 1e-04), cullenDehnen2010 (5.30e-04 > 1e-04), readHayfield2012 (1.26e-02 > 1e-04)

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | -0.4996 | 4 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 17.44 | 13.94 |
| cullenDehnen2010 | -0.4996 | 4 | 1.425 | 2 | 0.99 | 0 | 0 | 0 | 0 | 17.44 | 15.92 |
| readHayfield2012 | -0.4996 | 4 | 0.3004 | 0.9175 | 1 | 0 | 0 | 0 | 0 | 17.44 | 16.74 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.2098 | 0.3487 | 0.1737 | 1 | 1 | 1 | 0.06615 | 0 | 0.04373 | 1.107e-07 | 105.2 | 38.45 |
| cullenDehnen2010 | 0.07259 | 0.8091 | 0.008044 | 0.04148 | 0.1786 | 0.0084 | 0.02628 | 0 | 0.01659 | 2.147e-05 | 105.3 | 40.26 |
| readHayfield2012 | 0.1014 | 0.6646 | 0.01416 | 0.2 | 0.2 | 1 | 0.03582 | 0 | 0.02318 | 1.439e-06 | 104.9 | 40.77 |

## gresho: energy-drift budget (Group E)

all within budget

## yee (full)

| config | L1_speed | peakSpeedRatio | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.04584 | 0.8337 | 0.02218 | 1 | 1 | 1 | 0.6081 | 0 | 0.2606 | 3.798e-06 | 102.6 | 33.97 |
| cullenDehnen2010 | 0.01303 | 0.9559 | 0.005676 | 0.05777 | 0.1058 | 0.03723 | 0.1919 | 0 | 0.08474 | 1.345e-06 | 102.4 | 36.43 |
| readHayfield2012 | 0.01152 | 0.9594 | 0.00438 | 0.2 | 0.2 | 1 | 0.1666 | 0 | 0.07431 | 3.323e-06 | 102.4 | 36.93 |

## yee: energy-drift budget (Group E)

all within budget

## linearWave (full)

| config | waveVelocityErr | waveVelocityCoefficient | waveExactCoefficient | waveResidualRms | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.02366 | -0.9763 | -1 | 0.07331 | 1 | 1 | 1 | 2.686e-10 | 0 | 2.98e-07 | 0 | 15.75 | 26.89 |
| cullenDehnen2010 | 0.008538 | -0.9914 | -1 | 0.08837 | 0.03807 | 0.03808 | 0 | 1.15e-10 | 0 | 1.788e-07 | 6.624e-08 | 15.75 | 27.5 |
| readHayfield2012 | 0.003385 | -0.9966 | -1 | 0.06971 | 0.2 | 0.2 | 1 | 1.314e-10 | 0 | 1.788e-07 | 6.624e-08 | 15.75 | 28.41 |

## linearWave: energy-drift budget (Group E)

all within budget

## kelvinHelmholtz (full)

| config | khAmplitude0 | khAmplitudeAt1p5 | khAmplitudeMax | khReference1p5 | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.01054 | 0.0204 | 0.0206 | 0.1479 | 1 | 1 | 1 | 0.0318 | 0 | -0.0007558 | 5.489e-07 | 104.3 | 71.36 |
| cullenDehnen2010 | 0.01054 | 0.09842 | 0.09842 | 0.1479 | 0.06022 | 0.6499 | 0.1185 | 0.00628 | 0 | -0.00719 | 2.439e-07 | 104.1 | 73.65 |
| readHayfield2012 | 0.01054 | 0.0683 | 0.0683 | 0.1479 | 0.2 | 0.2 | 1 | 0.01115 | 0 | -0.01133 | 0.001558 | 104.2 | 74.47 |

## kelvinHelmholtz: energy-drift budget (Group E)

OVER: readHayfield2012 (1.56e-03 > 1e-04)

## rayleighTaylor (full)

| config | heavyMinY | lightMaxY | mixingWidth | maxVelocity | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.3691 | 0.6289 | 0.2598 | 0.1506 | 1 | 1 | 1 | 0.0009286 | 0 | -0.2694 | 0.0004591 | 102.9 | 48.73 |
| cullenDehnen2010 | 0.2926 | 0.6812 | 0.3886 | 0.3723 | 0.04045 | 0.6234 | 0.08428 | 0.0006806 | 0 | -0.009928 | 0.001521 | 102.9 | 51.68 |
| readHayfield2012 | 0.2858 | 0.6906 | 0.4048 | 0.3566 | 0.2 | 0.2112 | 1 | 0.001158 | 0 | -0.05657 | 0.001192 | 102.9 | 52.93 |

## rayleighTaylor: energy-drift budget (Group E)

OVER: none (4.59e-04 > 1e-04), cullenDehnen2010 (1.52e-03 > 1e-04), readHayfield2012 (1.19e-03 > 1e-04)

## Reproducibility lock (tol 1e-06)

| run | max rel diff | worst metric | state |
|---|---|---|---|
| none/sod | 0.00e+00 | - | LOCKED |
| none/sod2d | 0.00e+00 | - | LOCKED |
| none/sod3d | 0.00e+00 | - | LOCKED |
| none/sedov | 0.00e+00 | - | LOCKED |
| none/noh | 0.00e+00 | - | LOCKED |
| none/gresho | 0.00e+00 | - | LOCKED |
| none/yee | 0.00e+00 | - | LOCKED |
| none/linearWave | 0.00e+00 | - | LOCKED |
| none/kelvinHelmholtz | 0.00e+00 | - | LOCKED |
| none/rayleighTaylor | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/sod | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/sod2d | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/sod3d | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/sedov | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/noh | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/gresho | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/yee | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/linearWave | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/kelvinHelmholtz | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/rayleighTaylor | 0.00e+00 | - | LOCKED |
| readHayfield2012/sod | 0.00e+00 | - | LOCKED |
| readHayfield2012/sod2d | 0.00e+00 | - | LOCKED |
| readHayfield2012/sod3d | 0.00e+00 | - | LOCKED |
| readHayfield2012/sedov | 0.00e+00 | - | LOCKED |
| readHayfield2012/noh | 0.00e+00 | - | LOCKED |
| readHayfield2012/gresho | 0.00e+00 | - | LOCKED |
| readHayfield2012/yee | 0.00e+00 | - | LOCKED |
| readHayfield2012/linearWave | 0.00e+00 | - | LOCKED |
| readHayfield2012/kelvinHelmholtz | 0.00e+00 | - | LOCKED |
| readHayfield2012/rayleighTaylor | 0.00e+00 | - | LOCKED |


---

# Family 2 — `C_q = 2`, `sod noh sedov gresho`

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | 0.003681 | 0.07584 | 0.03666 | -0.001115 | -0.003813 | -0.0002094 | 1 | 1 | 1 | 0.007923 | 0.003338 | 0.02421 | 4.263e-06 | 16.91 | 29.33 |
| cullenDehnen2010Q | 0.004615 | 0.08215 | 0.03171 | -0.001766 | -0.0009482 | -0.001855 | 0.06663 | 0.8153 | 0.184 | 0.006457 | 0.003906 | 0.02211 | 1.358e-05 | 16.84 | 28.66 |
| readHayfield2012Q | 0.004051 | 0.05636 | 0.01753 | -0.002819 | -0.00453 | -0.001391 | 0.2337 | 0.8408 | 1 | 0.007225 | 0.004059 | 0.01779 | 0.001613 | 16.88 | 30.31 |

## sod: energy-drift budget (Group E)

OVER: readHayfield2012Q (1.61e-03 > 1e-04)

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | -0.0007594 | 4 | 1 | 1 | 1 | 0.1636 | 0.2911 | 0.2191 | 3.219e-06 | 17.93 | 13.98 |
| cullenDehnen2010Q | 0.0006001 | 4 | 0.4356 | 2 | 0.37 | 0.1764 | 0.269 | 0.2215 | 1.323e-05 | 17.99 | 15.85 |
| readHayfield2012Q | -0.1635 | 4 | 0.3318 | 0.9982 | 1 | 0.1482 | 0.2186 | 0.3289 | 0.2297 | 17.53 | 16.75 |

## noh: energy-drift budget (Group E)

OVER: readHayfield2012Q (2.30e-01 > 1e-04)

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | 2.087 | 4 | 0 | -0.01264 | 0 | 1 | 1 | 1 | 0.2278 | 0.365 | 0.09046 | 0.0005208 | 551.4 | 104.4 |
| cullenDehnen2010Q | 1.825 | 4 | 0 | -0.003258 | 0 | 1.807 | 1.993 | 1 | 0.2445 | 0.3437 | 0.07066 | 0.0005306 | 550.5 | 111.7 |
| readHayfield2012Q | 1.913 | 4 | 0 | -0.03153 | 0 | 0.8146 | 1 | 1 | 0.1101 | 0.1235 | -0.01476 | 0.02798 | 547.9 | 159.1 |

## sedov: energy-drift budget (Group E)

OVER: noneQ (5.21e-04 > 1e-04), cullenDehnen2010Q (5.31e-04 > 1e-04), readHayfield2012Q (2.80e-02 > 1e-04)

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | 0.2099 | 0.3419 | 0.1775 | 1 | 1 | 1 | 0.06478 | 0.0015 | 0.04383 | 1.107e-07 | 105.2 | 38.55 |
| cullenDehnen2010Q | 0.0718 | 0.8426 | 0.007943 | 0.0402 | 0.1486 | 0.0073 | 0.02475 | 0.00136 | 0.01653 | 2.025e-05 | 105.2 | 40.77 |
| readHayfield2012Q | 0.1021 | 0.6562 | 0.01433 | 0.2 | 0.2 | 1 | 0.03474 | 0.001309 | 0.02336 | 1.771e-06 | 104.9 | 41.48 |

## gresho: energy-drift budget (Group E)

all within budget

## Reproducibility lock (tol 1e-06)

| run | max rel diff | worst metric | state |
|---|---|---|---|
| noneQ/sod | 0.00e+00 | - | LOCKED |
| noneQ/noh | 0.00e+00 | - | LOCKED |
| noneQ/sedov | 0.00e+00 | - | LOCKED |
| noneQ/gresho | 0.00e+00 | - | LOCKED |
| cullenDehnen2010Q/sod | 0.00e+00 | - | LOCKED |
| cullenDehnen2010Q/noh | 0.00e+00 | - | LOCKED |
| cullenDehnen2010Q/sedov | 0.00e+00 | - | LOCKED |
| cullenDehnen2010Q/gresho | 0.00e+00 | - | LOCKED |
| readHayfield2012Q/sod | 0.00e+00 | - | LOCKED |
| readHayfield2012Q/noh | 0.00e+00 | - | LOCKED |
| readHayfield2012Q/sedov | 0.00e+00 | - | LOCKED |
| readHayfield2012Q/gresho | 0.00e+00 | - | LOCKED |


---

# Family 3 — AV_PLAN S3 switches, `C_q = 0`, `sod gresho`

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| balsara1995 | 0.003492 | 0.06266 | 0.03373 | -0.0007142 | -0.003614 | -0.0001633 | 0.7627 | 0.9998 | 0.932 | 0.0107 | 0 | 0.02349 | 3.829e-06 | 16.88 | 30.09 |
| colagrossi2004 | 0.003492 | 0.06266 | 0.03373 | -0.0007142 | -0.003614 | -0.0001633 | 0.7627 | 0.9998 | 0.932 | 0.0107 | 0 | 0.02349 | 3.829e-06 | 16.88 | 27.8 |
| morrisMonaghan1997 | 0.008319 | 0.07933 | 0.03545 | -0.0009167 | -0.00214 | -0.002979 | 0.1165 | 0.5114 | 1 | 0.00986 | 0 | 0.02139 | 1.517e-06 | 16.83 | 27.49 |
| rosswog2000 | 0.006237 | 0.07841 | 0.03572 | -0.0009267 | -0.002615 | -0.002403 | 0.1225 | 0.6799 | 1 | 0.01005 | 0 | 0.02206 | 3.107e-06 | 16.83 | 27.97 |

## sod: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| balsara1995 | 0.07972 | 0.7312 | 0.009033 | 0.1545 | 0.9709 | 0.4507 | 0.02921 | 0 | 0.01869 | 5.423e-06 | 104.9 | 38.99 |
| colagrossi2004 | 0.07304 | 0.7897 | 0.008423 | 0.1506 | 0.9391 | 0.4989 | 0.02702 | 0 | 0.01713 | 9.297e-06 | 104.9 | 38.37 |
| morrisMonaghan1997 | 0.08037 | 0.7397 | 0.007781 | 0.1009 | 0.1053 | 1 | 0.02948 | 0 | 0.01886 | 7.636e-06 | 105 | 38.89 |
| rosswog2000 | 0.08026 | 0.7477 | 0.008484 | 0.1017 | 0.1143 | 1 | 0.02945 | 0 | 0.01885 | 6.972e-06 | 105 | 39.71 |

## gresho: energy-drift budget (Group E)

all within budget

## Reproducibility lock (tol 1e-06)

| run | max rel diff | worst metric | state |
|---|---|---|---|
| balsara1995/sod | 0.00e+00 | - | LOCKED |
| balsara1995/gresho | 0.00e+00 | - | LOCKED |
| colagrossi2004/sod | 0.00e+00 | - | LOCKED |
| colagrossi2004/gresho | 0.00e+00 | - | LOCKED |
| morrisMonaghan1997/sod | 0.00e+00 | - | LOCKED |
| morrisMonaghan1997/gresho | 0.00e+00 | - | LOCKED |
| rosswog2000/sod | 0.00e+00 | - | LOCKED |
| rosswog2000/gresho | 0.00e+00 | - | LOCKED |

