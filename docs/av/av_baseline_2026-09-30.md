# AV baseline (M0) — 2026-09-30

Produced by `scripts/av_report.py --profile full --repeat 2` (commit `c15cbf5` + the case
extension). Host scheme **Monaghan**, kernel B7, RK2, `C_l = 1`, `Price2012_98`. Two column
families: **`none / cullenDehnen2010 / readHayfield2012`** at the Monaghan default **`C_q = 0`**
(`--config baseline`, all ten cases), and the same three at **`C_q = 2`** (`--config baselineQ`,
`sod noh sedov gresho`). Videos: `results/av_baseline_2026-09-30_all/runs/` and
`results/av_baselineQ_2026-09-30/runs/` (gitignored).

**Reproducibility lock: all 30 + 12 (config, case) pairs are bit-identical across two runs**
(max relative difference 0). Nothing diverged, no velocity alarm fired. (A first attempt failed
the lock for R&H/Gresho: its entropy-dissipation term used a CUDA atomic `scatter_sum`;
fixed with a fixed-order `segment_reduce`.)

## Reading the numbers

1. **`C_q = 0` means no quadratic AV at all** (`avEnergyQuadratic = 0` everywhere in the first family),
   and in **cold gas the linear term is also zero** (`v_sig = C_l c`, `c = 0`). **Noh's row is
   therefore degenerate at `C_q = 0`**: u stays 0, the two halves of the lattice stream through each
   other, the density plateau is exactly 2.0 (the overlap of two streams, -50 % vs the exact 4) and
   all three switches give identical output. With `C_q = 2` Noh dissipates (quadratic AV energy 0.23-0.26
   vs linear 0.18-0.19) but the post-shock density is still **-23 % / -24 % / -31 %** (none / C&D /
   R&H), far from the plan's +-3 % gate. A Phase 5A / resolution question, not a detector one.
2. **Sedov (3D, nx 40) peak rho/rho0 is 1.7-2.4 vs 4** — coarse, never overshoots. The Monaghan host
   **does not conserve energy here (47-68 %, R&H 20-25 %; OPEN_PROBLEMS §16)**, and `C_q = 2` makes it
   *worse* (68 %), so the missing quadratic term is not the cause. CompSPH holds 1.000 exactly on
   the same case. Noh at `C_q = 2` also drifts 42-65 %. Treat every strong-shock Monaghan number
   here as confounded until §16 is resolved.
3. **Gresho**: `none` (alpha = 1) loses the vortex (peak speed 0.34, L1 0.21, 17.6-18 % angular
   momentum); both switches keep it (peak 0.78-0.80, L1 0.07, <1 % loss). C&D ~ R&H, R&H marginally
   better. `C_q = 2` changes nothing material (quadratic AV energy 0.0015 vs linear 0.025-0.065).
4. **Sod**: `L1(v_x)` ~0.005 for all; C&D (C_q = 0) has the largest contact P spike (19 % vs 7-8 %),
   which `C_q = 2` removes (8.7 %). The 5e-3 Monaghan energy budget is exceeded by all configs
   (5.4e-3 - 6.7e-3); the budget comes from `tests/test_physics.py` (different setup): recorded, not a conclusion.
5. **Kelvin-Helmholtz** (nx 128, t = 1.5): mode amplitude vs McNally 14.79e-2 — see table;
   `none` ~2.0e-2, the scale of garciasenz2026 Table 3 KH1 (AV alone, 2.87e-2).
6. **Not quality numbers:** `sod3d` runs at nx = 20 (15 steps; R4 density error 21-34 %) because 3D
   costs ~4 s/step at 580 neighbours (a perf question); `linearWave` is run at `A = 1e-4`, t = 1/4
   crossing (the case default `A = 1e-6` is ~7x float32 noise); `rayleighTaylor` reports interface
   tips only (no reference); `wallMsPerStep` is wall-clock including setup (StepTimer not wired).

---

# Family 1 — `C_q = 0` (Monaghan default), all ten cases

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.005008 | 0.08011 | 0.1209 | 0.006933 | -0.03122 | 0.01055 | 1 | 1 | 1 | 0.01072 | 0 | 0.04653 | 0.006466 | 16.92 | 29.6 |
| cullenDehnen2010 | 0.005882 | 0.1918 | 0.1016 | 0.00669 | -0.03063 | 0.00782 | 0.08196 | 0.9506 | 0.192 | 0.01058 | 0 | 0.04484 | 0.006441 | 16.83 | 50.51 |
| readHayfield2012 | 0.004847 | 0.07148 | 0.1113 | 0.006118 | -0.03692 | 0.009014 | 0.2389 | 0.8257 | 1 | 0.01111 | 0 | 0.04262 | 0.005473 | 16.86 | 30.15 |

## sod: energy-drift budget (Group E)

OVER: none (6.47e-03 > 5e-03), cullenDehnen2010 (6.44e-03 > 5e-03), readHayfield2012 (5.47e-03 > 5e-03)

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.02136 | 0.1023 | 0.1003 | -0.02914 | -0.03157 | 0.03964 | 1 | 1 | 1 | 0.002709 | 0 | 0.01022 | 0.00632 | 106.7 | 39.88 |
| cullenDehnen2010 | 0.0176 | 0.1097 | 0.08694 | -0.03208 | -0.01645 | 0.04397 | 0.2839 | 1.055 | 1 | 0.002383 | 0 | 0.008443 | 0.005549 | 106.6 | 38.22 |
| readHayfield2012 | 0.0204 | 0.1064 | -0.01889 | -0.03794 | 0.05445 | 0.06085 | 0.2725 | 0.9187 | 1 | 0.00279 | 0 | 0.003623 | 0.0006754 | 106.7 | 43.71 |

## sod2d: energy-drift budget (Group E)

OVER: none (6.32e-03 > 5e-03), cullenDehnen2010 (5.55e-03 > 5e-03)

## sod3d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.05968 | 0.1246 | -0.1007 | -0.09172 | 0.22 | 0.03689 | 1 | 1 | 1 | 0.01354 | 0 | 0.02374 | 0.004866 | 580.8 | 117.8 |
| cullenDehnen2010 | 0.05785 | 0.1346 | -0.3724 | -0.09083 | 0.2065 | 0.02186 | 0.8284 | 1.636 | 1 | 0.01465 | 0 | 0.0269 | 0.005278 | 581.1 | 389.5 |
| readHayfield2012 | 0.06939 | 0.1558 | -0.3357 | -0.07724 | 0.3444 | -0.1056 | 0.3938 | 0.9135 | 1 | 0.01106 | 0 | -0.09334 | 0.01136 | 579.4 | 859.5 |

## sod3d: energy-drift budget (Group E)

OVER: cullenDehnen2010 (5.28e-03 > 5e-03), readHayfield2012 (1.14e-02 > 5e-03)

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.355 | 4 | 0 | 0.0269 | 0 | 1 | 1 | 1 | 0.4669 | 0 | 0.3043 | 0.4684 | 553.3 | 92.69 |
| cullenDehnen2010 | 2.018 | 4 | 0 | 0.04502 | 0 | 1.801 | 1.995 | 1 | 0.5183 | 0 | 0.3463 | 0.5203 | 553.2 | 109.1 |
| readHayfield2012 | 2.044 | 4 | 0 | -0.01569 | 0 | 0.7932 | 1 | 0.9348 | 0.1585 | 0 | 0.08989 | 0.1995 | 549.1 | 161.3 |

## sedov: energy-drift budget (Group E)

OVER: none (4.68e-01 > 5e-03), cullenDehnen2010 (5.20e-01 > 5e-03), readHayfield2012 (1.99e-01 > 5e-03)

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | -0.4996 | 4 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 17.44 | 14.02 |
| cullenDehnen2010 | -0.4996 | 4 | 1.425 | 2 | 0.99 | 0 | 0 | 0 | 0 | 17.44 | 15.87 |
| readHayfield2012 | -0.4996 | 4 | 0.1681 | 0.9175 | 0.44 | 0 | 0 | 0 | 0 | 17.44 | 16.84 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.209 | 0.3417 | 0.1756 | 1 | 1 | 1 | 0.06614 | 0 | 0.088 | 0.007676 | 105.2 | 38.52 |
| cullenDehnen2010 | 0.07312 | 0.787 | 0.007717 | 0.04137 | 0.1487 | 0.0105 | 0.02635 | 0 | 0.03427 | 0.003083 | 105.3 | 40.61 |
| readHayfield2012 | 0.07049 | 0.8014 | 0.005125 | 0.04186 | 0.3484 | 0.0597 | 0.02534 | 0 | 0.03284 | 0.002992 | 105.3 | 41.88 |

## gresho: energy-drift budget (Group E)

OVER: none (7.68e-03 > 5e-03)

## yee (full)

| config | L1_speed | peakSpeedRatio | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.04573 | 0.8335 | 0.02283 | 1 | 1 | 1 | 0.6076 | 0 | 0.5164 | 0.003147 | 102.6 | 34.98 |
| cullenDehnen2010 | 0.01299 | 0.956 | 0.005747 | 0.05772 | 0.1051 | 0.03723 | 0.1918 | 0 | 0.1652 | 0.0009891 | 102.5 | 36.27 |
| readHayfield2012 | 0.002998 | 0.9927 | 0.001322 | 0.02 | 0.02001 | 0 | 0.04782 | 0 | 0.04198 | 0.0002412 | 102.4 | 37.04 |

## yee: energy-drift budget (Group E)

all within budget

## linearWave (full)

| config | waveVelocityErr | waveVelocityCoefficient | waveExactCoefficient | waveResidualRms | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.02367 | -0.9763 | -1 | 0.07332 | 1 | 1 | 1 | 2.686e-10 | 0 | 2.98e-07 | 0 | 15.75 | 26.94 |
| cullenDehnen2010 | 0.007133 | -0.9928 | -1 | 0.08793 | 0.03807 | 0.03808 | 0 | 1.151e-10 | 0 | 1.788e-07 | 0 | 15.75 | 27.44 |
| readHayfield2012 | 0.008745 | -0.9912 | -1 | 0.1012 | 0.02 | 0.02 | 0 | 2.854e-11 | 0 | 1.788e-07 | 0 | 15.75 | 28.76 |

## linearWave: energy-drift budget (Group E)

all within budget

## kelvinHelmholtz (full)

| config | khAmplitude0 | khAmplitudeAt1p5 | khAmplitudeMax | khReference1p5 | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.01054 | 0.02007 | 0.02034 | 0.1479 | 1 | 1 | 1 | 0.03163 | 0 | 0.01626 | 0.008091 | 104.3 | 70.97 |
| cullenDehnen2010 | 0.01054 | 0.09657 | 0.09657 | 0.1479 | 0.06037 | 0.6846 | 0.1213 | 0.006299 | 0 | -0.003796 | 0.00161 | 104.1 | 71.32 |
| readHayfield2012 | 0.01054 | 0.1123 | 0.1123 | 0.1479 | 0.09034 | 0.5344 | 0.3514 | 0.004488 | 0 | -0.006793 | 7.806e-06 | 104.1 | 74.33 |

## kelvinHelmholtz: energy-drift budget (Group E)

OVER: none (8.09e-03 > 5e-03)

## rayleighTaylor (full)

| config | heavyMinY | lightMaxY | mixingWidth | maxVelocity | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.3692 | 0.6294 | 0.2601 | 0.1501 | 1 | 1 | 1 | 0.0009278 | 0 | -0.269 | 0.0007991 | 102.9 | 49.07 |
| cullenDehnen2010 | 0.2925 | 0.6819 | 0.3893 | 0.3875 | 0.04057 | 0.5278 | 0.08714 | 0.0006828 | 0 | -0.009677 | 0.001769 | 102.8 | 50.19 |
| readHayfield2012 | 0.2922 | 0.6837 | 0.3915 | 0.4608 | 0.04135 | 0.4568 | 0.0803 | 0.0004917 | 0 | -0.006851 | 0.001297 | 102.8 | 52.37 |

## rayleighTaylor: energy-drift budget (Group E)

all within budget

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
| noneQ | 0.005229 | 0.1014 | 0.1294 | 0.006513 | -0.03177 | 0.01082 | 1 | 1 | 1 | 0.007884 | 0.003268 | 0.0481 | 0.006714 | 16.92 | 29.19 |
| cullenDehnen2010Q | 0.005258 | 0.08684 | 0.1256 | 0.005427 | -0.02748 | 0.008378 | 0.06711 | 0.8049 | 0.18 | 0.006608 | 0.003952 | 0.04482 | 0.006377 | 16.83 | 28.8 |
| readHayfield2012Q | 0.004754 | 0.07756 | 0.1054 | 0.005524 | -0.03339 | 0.009186 | 0.2364 | 0.8081 | 1 | 0.007238 | 0.003945 | 0.04274 | 0.005451 | 16.87 | 29.35 |

## sod: energy-drift budget (Group E)

OVER: noneQ (6.71e-03 > 5e-03), cullenDehnen2010Q (6.38e-03 > 5e-03), readHayfield2012Q (5.45e-03 > 5e-03)

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | -0.2411 | 4 | 1 | 1 | 1 | 0.182 | 0.2606 | 0.4393 | 0.4415 | 17.57 | 14.05 |
| cullenDehnen2010Q | -0.2298 | 4 | 0.328 | 2 | 0.32 | 0.1921 | 0.2316 | 0.4342 | 0.4233 | 17.58 | 16.16 |
| readHayfield2012Q | -0.3072 | 4 | 0.1943 | 0.9548 | 0.475 | 0.1575 | 0.1975 | 0.5478 | 0.6534 | 17.28 | 16.89 |

## noh: energy-drift budget (Group E)

OVER: noneQ (4.41e-01 > 5e-03), cullenDehnen2010Q (4.23e-01 > 5e-03), readHayfield2012Q (6.53e-01 > 5e-03)

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | 1.965 | 4 | 0 | 0.0603 | 0 | 1 | 1 | 1 | 0.2815 | 0.396 | 0.5386 | 0.679 | 555.6 | 87.55 |
| cullenDehnen2010Q | 1.741 | 4 | 0 | 0.07092 | 0 | 1.798 | 1.992 | 1 | 0.3012 | 0.369 | 0.5193 | 0.6722 | 554.2 | 106.9 |
| readHayfield2012Q | 1.873 | 4 | 0 | -0.00181 | 0 | 0.7744 | 1 | 0.9159 | 0.1202 | 0.1303 | 0.1784 | 0.2547 | 549.7 | 172 |

## sedov: energy-drift budget (Group E)

OVER: noneQ (6.79e-01 > 5e-03), cullenDehnen2010Q (6.72e-01 > 5e-03), readHayfield2012Q (2.55e-01 > 5e-03)

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | 0.2099 | 0.3407 | 0.1798 | 1 | 1 | 1 | 0.06494 | 0.001498 | 0.08841 | 0.00771 | 105.2 | 38.59 |
| cullenDehnen2010Q | 0.07258 | 0.7768 | 0.006645 | 0.04104 | 0.1682 | 0.0112 | 0.02507 | 0.001381 | 0.03444 | 0.003094 | 105.3 | 40.29 |
| readHayfield2012Q | 0.06976 | 0.8033 | 0.005832 | 0.04133 | 0.3277 | 0.0616 | 0.02371 | 0.001429 | 0.03262 | 0.002968 | 105.2 | 41.78 |

## gresho: energy-drift budget (Group E)

OVER: noneQ (7.71e-03 > 5e-03)

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

