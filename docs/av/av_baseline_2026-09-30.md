# AV baseline (M0b) — 2026-09-30

Produced by `scripts/av_report.py --profile full --repeat 2` (M0 tag `milestone/av-baseline` = `0ed8c72`;
this is **M0b**, the same baseline with a harness correction, regenerated on `dev` after AV_PLAN S3).
Host scheme **Monaghan**, kernel B7, RK2, `C_l = 1`, `Price2012_98`. Three column families:

1. `none / cullenDehnen2010 / readHayfield2012` at the Monaghan default **`C_q = 0`** — all ten cases
   (`--config baseline`).
2. The same three at **`C_q = 2`** — `sod noh sedov gresho` (`--config baselineQ`).
3. The AV_PLAN S3 switches `balsara1995 / colagrossi2004 / morrisMonaghan1997 / rosswog2000` at `C_q = 0` — `sod gresho`
   (`--config switchesS3`).

Videos: `results/av_M0b_*/runs/` and the older `results/av_baseline*/runs/` (gitignored). **Reproducibility lock:
every (config, case) pair in every family is bit-identical across two runs** (max relative difference 0; nothing
diverged, no velocity alarm). `scripts/av_report.py --compare A B` is the bit-for-bit instrument; merged reports come
from `--merge`.

## What changed between M0 and M0b

The first M0 report applied the switch's alpha range (`alpha_min`, `alpha_max`) as case *parameters*, which only
`sod`/`sodND` read (`configureCompressible` ignores them). So **Read-Hayfield ran in its designed 0.2-1.0 range on the
Sod family only; every other case ran it at the default 0.02-2.0.** M0b applies the range to the scheme config for every
case (`AVConfig.switchParams`). Only Read-Hayfield (and the S3 source-and-decay switches) are affected: `none`,
Cullen-Dehnen and all Sod numbers are bit-identical to M0. Read-Hayfield changes on the smooth cases, where alpha now sits
at its 0.2 floor (alpha max = 0.2 on Gresho, Yee, linearWave): Gresho `L1(v_phi)` 0.071 -> 0.101, Yee `L1` 0.003 -> 0.012,
KH amplitude at t = 1.5 0.112 -> 0.068, linearWave wave error 0.0087 -> 0.0034, Sedov/Noh/RT small.
**So the earlier statement "Read-Hayfield marginally beats Cullen-Dehnen on Gresho" is withdrawn: at its designed range it
is worse (0.101 vs 0.073), because its alpha floor keeps dissipating the smooth vortex.** (The S2 bit-for-bit checks
were against the first M0 and stay valid: they tested code changes, and the harness fix is separate.)

## Reading the numbers

1. **`C_q = 0` means no quadratic AV at all** (`avEnergyQuadratic = 0` in family 1), and in **cold gas the linear term is
   also zero** (`v_sig = C_l c`, `c = 0`). **Noh is degenerate at `C_q = 0`**: u stays 0, the two halves of the lattice stream
   through each other (density plateau exactly 2.0, -50 % vs the exact 4) and the switches differ only in their alpha. With
   `C_q = 2` Noh dissipates but the post-shock density is still ~-23..-31 % (plan gate: +-3 %).
2. **Sedov (3D, nx 40): peak rho/rho0 1.7-2.4 vs 4**, never overshoots. The Monaghan host **does not conserve energy on
   strong shocks (OPEN_PROBLEMS §16)**: Sedov +18..68 %, Noh +42..65 % at `C_q = 2`; `C_q = 2` makes Sedov worse, so the missing
   quadratic term is not the cause; CompSPH conserves exactly. Treat strong-shock Monaghan numbers as confounded until §16.
3. **Gresho**: `none` (alpha = 1) loses the vortex (peak speed 0.34, L1 0.21, 18 % angular momentum). Among the switches the
   result is ordered by the **alpha floor**, because in a smooth flow every one of them sits at it: Cullen-Dehnen (floor 0.02)
   L1 0.073 / peak 0.79; Morris-Monaghan and Rosswog2000 (floor 0.1) 0.080-0.081 / 0.74; Read-Hayfield at its designed floor 0.2
   0.101 / 0.68. Balsara 0.082 and Colagrossi 0.073 (alpha from the limiter, mean 0.15-0.17, peak ~0.9-1.0). `C_q = 2` changes
   Gresho nothing material. (Withdrawn: "Read-Hayfield marginally better than Cullen-Dehnen": see above.)
4. **Sod**: `L1(v_x)` ~0.005 for none / C&D / R&H / Balsara / Colagrossi (Balsara == Colagrossi **exactly** in 1D: curl and
   trace-free shear are both zero), 0.007-0.008 for the source-and-decay switches; C&D (`C_q = 0`) has the largest contact P
   spike (19 % vs 7-9 %), which `C_q = 2` removes. The 5e-3 Monaghan energy budget is exceeded by all configs (5.4e-3 - 6.7e-3);
   the budget comes from `tests/test_physics.py` (a different setup): recorded, not a conclusion.
5. **Morris-Monaghan peak alpha on Sod is 0.51**, against the paper's Eq. (19) no-decay ceiling `alpha_inf + ln(v1/v2)`
   = ~0.67 for this shock; consistent.
6. **Not quality numbers:** `sod3d` runs at nx = 20 (15 steps; R4 density error 21-34 %) because 3D costs ~4 s/step at 580
   neighbours (a perf question); `linearWave` runs at `A = 1e-4`, t = 1/4 crossing (the case default `A = 1e-6` is ~7x float32
   noise); `rayleighTaylor` reports interface tips only; `wallMsPerStep` is wall-clock including setup (StepTimer not wired).

---

# Family 1 — `C_q = 0`, all ten cases

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.005008 | 0.08011 | 0.1209 | 0.006933 | -0.03122 | 0.01055 | 1 | 1 | 1 | 0.01072 | 0 | 0.04653 | 0.006466 | 16.92 | 29.6 |
| cullenDehnen2010 | 0.005882 | 0.1918 | 0.1016 | 0.00669 | -0.03063 | 0.00782 | 0.08196 | 0.9506 | 0.192 | 0.01058 | 0 | 0.04484 | 0.006441 | 16.83 | 50.51 |
| readHayfield2012 | 0.004847 | 0.07148 | 0.1113 | 0.006118 | -0.03692 | 0.009014 | 0.2389 | 0.8257 | 1 | 0.01111 | 0 | 0.04262 | 0.005473 | 16.86 | 32.98 |

## sod: energy-drift budget (Group E)

OVER: none (6.47e-03 > 5e-03), cullenDehnen2010 (6.44e-03 > 5e-03), readHayfield2012 (5.47e-03 > 5e-03)

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.02136 | 0.1023 | 0.1003 | -0.02914 | -0.03157 | 0.03964 | 1 | 1 | 1 | 0.002709 | 0 | 0.01022 | 0.00632 | 106.7 | 39.88 |
| cullenDehnen2010 | 0.0176 | 0.1097 | 0.08694 | -0.03208 | -0.01645 | 0.04397 | 0.2839 | 1.055 | 1 | 0.002383 | 0 | 0.008443 | 0.005549 | 106.6 | 38.22 |
| readHayfield2012 | 0.0204 | 0.1064 | -0.01889 | -0.03794 | 0.05445 | 0.06085 | 0.2725 | 0.9187 | 1 | 0.00279 | 0 | 0.003623 | 0.0006754 | 106.7 | 46.16 |

## sod2d: energy-drift budget (Group E)

OVER: none (6.32e-03 > 5e-03), cullenDehnen2010 (5.55e-03 > 5e-03)

## sod3d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.05968 | 0.1246 | -0.1007 | -0.09172 | 0.22 | 0.03689 | 1 | 1 | 1 | 0.01354 | 0 | 0.02374 | 0.004866 | 580.8 | 117.8 |
| cullenDehnen2010 | 0.05785 | 0.1346 | -0.3724 | -0.09083 | 0.2065 | 0.02186 | 0.8284 | 1.636 | 1 | 0.01465 | 0 | 0.0269 | 0.005278 | 581.1 | 389.5 |
| readHayfield2012 | 0.06939 | 0.1558 | -0.3357 | -0.07724 | 0.3444 | -0.1056 | 0.3938 | 0.9135 | 1 | 0.01106 | 0 | -0.09334 | 0.01136 | 579.4 | 125.7 |

## sod3d: energy-drift budget (Group E)

OVER: cullenDehnen2010 (5.28e-03 > 5e-03), readHayfield2012 (1.14e-02 > 5e-03)

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.355 | 4 | 0 | 0.0269 | 0 | 1 | 1 | 1 | 0.4669 | 0 | 0.3043 | 0.4684 | 553.3 | 92.69 |
| cullenDehnen2010 | 2.018 | 4 | 0 | 0.04502 | 0 | 1.801 | 1.995 | 1 | 0.5183 | 0 | 0.3463 | 0.5203 | 553.2 | 109.1 |
| readHayfield2012 | 2.016 | 4 | 0 | -0.01534 | 0 | 0.8153 | 1 | 1 | 0.1576 | 0 | 0.06862 | 0.1826 | 548.7 | 155.3 |

## sedov: energy-drift budget (Group E)

OVER: none (4.68e-01 > 5e-03), cullenDehnen2010 (5.20e-01 > 5e-03), readHayfield2012 (1.83e-01 > 5e-03)

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | -0.4996 | 4 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 17.44 | 14.02 |
| cullenDehnen2010 | -0.4996 | 4 | 1.425 | 2 | 0.99 | 0 | 0 | 0 | 0 | 17.44 | 15.87 |
| readHayfield2012 | -0.4996 | 4 | 0.3004 | 0.9175 | 1 | 0 | 0 | 0 | 0 | 17.44 | 16.85 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.209 | 0.3417 | 0.1756 | 1 | 1 | 1 | 0.06614 | 0 | 0.088 | 0.007676 | 105.2 | 38.52 |
| cullenDehnen2010 | 0.07312 | 0.787 | 0.007717 | 0.04137 | 0.1487 | 0.0105 | 0.02635 | 0 | 0.03427 | 0.003083 | 105.3 | 40.61 |
| readHayfield2012 | 0.1012 | 0.6787 | 0.01419 | 0.2 | 0.2 | 1 | 0.03578 | 0 | 0.04707 | 0.004148 | 104.9 | 40.7 |

## gresho: energy-drift budget (Group E)

OVER: none (7.68e-03 > 5e-03)

## yee (full)

| config | L1_speed | peakSpeedRatio | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.04573 | 0.8335 | 0.02283 | 1 | 1 | 1 | 0.6076 | 0 | 0.5164 | 0.003147 | 102.6 | 34.98 |
| cullenDehnen2010 | 0.01299 | 0.956 | 0.005747 | 0.05772 | 0.1051 | 0.03723 | 0.1918 | 0 | 0.1652 | 0.0009891 | 102.5 | 36.27 |
| readHayfield2012 | 0.01152 | 0.9596 | 0.004419 | 0.2 | 0.2 | 1 | 0.1665 | 0 | 0.1438 | 0.0008607 | 102.4 | 37.26 |

## yee: energy-drift budget (Group E)

all within budget

## linearWave (full)

| config | waveVelocityErr | waveVelocityCoefficient | waveExactCoefficient | waveResidualRms | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.02367 | -0.9763 | -1 | 0.07332 | 1 | 1 | 1 | 2.686e-10 | 0 | 2.98e-07 | 0 | 15.75 | 26.94 |
| cullenDehnen2010 | 0.007133 | -0.9928 | -1 | 0.08793 | 0.03807 | 0.03808 | 0 | 1.151e-10 | 0 | 1.788e-07 | 0 | 15.75 | 27.44 |
| readHayfield2012 | 0.003385 | -0.9966 | -1 | 0.06971 | 0.2 | 0.2 | 1 | 1.314e-10 | 0 | 1.788e-07 | 6.624e-08 | 15.75 | 28.69 |

## linearWave: energy-drift budget (Group E)

all within budget

## kelvinHelmholtz (full)

| config | khAmplitude0 | khAmplitudeAt1p5 | khAmplitudeMax | khReference1p5 | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.01054 | 0.02007 | 0.02034 | 0.1479 | 1 | 1 | 1 | 0.03163 | 0 | 0.01626 | 0.008091 | 104.3 | 70.97 |
| cullenDehnen2010 | 0.01054 | 0.09657 | 0.09657 | 0.1479 | 0.06037 | 0.6846 | 0.1213 | 0.006299 | 0 | -0.003796 | 0.00161 | 104.1 | 71.32 |
| readHayfield2012 | 0.01054 | 0.06797 | 0.06797 | 0.1479 | 0.2 | 0.2 | 1 | 0.01104 | 0 | -0.005459 | 0.001262 | 104.2 | 75.27 |

## kelvinHelmholtz: energy-drift budget (Group E)

OVER: none (8.09e-03 > 5e-03)

## rayleighTaylor (full)

| config | heavyMinY | lightMaxY | mixingWidth | maxVelocity | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.3692 | 0.6294 | 0.2601 | 0.1501 | 1 | 1 | 1 | 0.0009278 | 0 | -0.269 | 0.0007991 | 102.9 | 49.07 |
| cullenDehnen2010 | 0.2925 | 0.6819 | 0.3893 | 0.3875 | 0.04057 | 0.5278 | 0.08714 | 0.0006828 | 0 | -0.009677 | 0.001769 | 102.8 | 50.19 |
| readHayfield2012 | 0.2856 | 0.6905 | 0.4049 | 0.3539 | 0.2 | 0.2125 | 1 | 0.001162 | 0 | -0.05616 | 0.00162 | 102.9 | 53.25 |

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
| readHayfield2012Q | 0.004754 | 0.07756 | 0.1054 | 0.005524 | -0.03339 | 0.009186 | 0.2364 | 0.8081 | 1 | 0.007238 | 0.003945 | 0.04274 | 0.005451 | 16.87 | 32.59 |

## sod: energy-drift budget (Group E)

OVER: noneQ (6.71e-03 > 5e-03), cullenDehnen2010Q (6.38e-03 > 5e-03), readHayfield2012Q (5.45e-03 > 5e-03)

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | -0.2411 | 4 | 1 | 1 | 1 | 0.182 | 0.2606 | 0.4393 | 0.4415 | 17.57 | 14.05 |
| cullenDehnen2010Q | -0.2298 | 4 | 0.328 | 2 | 0.32 | 0.1921 | 0.2316 | 0.4342 | 0.4233 | 17.58 | 16.16 |
| readHayfield2012Q | -0.3071 | 4 | 0.2862 | 0.9547 | 1 | 0.1576 | 0.197 | 0.5478 | 0.6533 | 17.29 | 16.41 |

## noh: energy-drift budget (Group E)

OVER: noneQ (4.41e-01 > 5e-03), cullenDehnen2010Q (4.23e-01 > 5e-03), readHayfield2012Q (6.53e-01 > 5e-03)

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | 1.965 | 4 | 0 | 0.0603 | 0 | 1 | 1 | 1 | 0.2815 | 0.396 | 0.5386 | 0.679 | 555.6 | 87.55 |
| cullenDehnen2010Q | 1.741 | 4 | 0 | 0.07092 | 0 | 1.798 | 1.992 | 1 | 0.3012 | 0.369 | 0.5193 | 0.6722 | 554.2 | 106.9 |
| readHayfield2012Q | 1.854 | 4 | 0 | -0.001747 | 0 | 0.7957 | 1 | 1 | 0.1193 | 0.127 | 0.1501 | 0.233 | 549.2 | 154.8 |

## sedov: energy-drift budget (Group E)

OVER: noneQ (6.79e-01 > 5e-03), cullenDehnen2010Q (6.72e-01 > 5e-03), readHayfield2012Q (2.33e-01 > 5e-03)

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noneQ | 0.2099 | 0.3407 | 0.1798 | 1 | 1 | 1 | 0.06494 | 0.001498 | 0.08841 | 0.00771 | 105.2 | 38.59 |
| cullenDehnen2010Q | 0.07258 | 0.7768 | 0.006645 | 0.04104 | 0.1682 | 0.0112 | 0.02507 | 0.001381 | 0.03444 | 0.003094 | 105.3 | 40.29 |
| readHayfield2012Q | 0.1024 | 0.6655 | 0.01464 | 0.2 | 0.2 | 1 | 0.03476 | 0.001306 | 0.04747 | 0.004181 | 104.9 | 41.57 |

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


---

# Family 3 — AV_PLAN S3 switches, `C_q = 0`, `sod gresho`

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| balsara1995 | 0.00501 | 0.08123 | 0.1219 | 0.006939 | -0.03125 | 0.01051 | 0.752 | 0.9998 | 0.92 | 0.01072 | 0 | 0.04654 | 0.006466 | 16.89 | 30.26 |
| colagrossi2004 | 0.00501 | 0.08123 | 0.1219 | 0.006939 | -0.03125 | 0.01051 | 0.752 | 0.9998 | 0.92 | 0.01072 | 0 | 0.04654 | 0.006466 | 16.89 | 27.33 |
| morrisMonaghan1997 | 0.008332 | 0.08554 | 0.1251 | 0.006073 | -0.02977 | 0.006747 | 0.1168 | 0.5057 | 1 | 0.0103 | 0 | 0.04349 | 0.006265 | 16.81 | 27.13 |
| rosswog2000 | 0.006597 | 0.08553 | 0.1265 | 0.006227 | -0.03027 | 0.007566 | 0.123 | 0.6671 | 1 | 0.0104 | 0 | 0.04457 | 0.006326 | 16.82 | 27.24 |

## sod: energy-drift budget (Group E)

OVER: balsara1995 (6.47e-03 > 5e-03), colagrossi2004 (6.47e-03 > 5e-03), morrisMonaghan1997 (6.27e-03 > 5e-03), rosswog2000 (6.33e-03 > 5e-03)

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| balsara1995 | 0.08169 | 0.7351 | 0.01044 | 0.1651 | 0.9778 | 0.4606 | 0.0296 | 0 | 0.03871 | 0.003435 | 104.9 | 38.73 |
| colagrossi2004 | 0.07288 | 0.7741 | 0.008686 | 0.1539 | 0.8923 | 0.5039 | 0.02692 | 0 | 0.03502 | 0.003129 | 104.9 | 38.87 |
| morrisMonaghan1997 | 0.08048 | 0.7403 | 0.008233 | 0.1009 | 0.1054 | 1 | 0.02931 | 0 | 0.03836 | 0.00341 | 105 | 38.92 |
| rosswog2000 | 0.08136 | 0.7441 | 0.008045 | 0.1017 | 0.1132 | 1 | 0.0297 | 0 | 0.03887 | 0.003455 | 105 | 38.5 |

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

