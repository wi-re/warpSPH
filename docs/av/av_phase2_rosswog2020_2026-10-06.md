# AV_PLAN Phase 2 — Rosswog (2020) entropy trigger, results (2026-10-06)

Host: Monaghan (`C_l = 1`, `C_q = 0` default and the `C_q = 2` family), trigger at the paper's constants
(`alpha_0 = 0`, `alpha_max = 1`, `eps_0 = 1e-4`, `eps_1 = 5e-2`, decay `30 tau`, `h = support/2`).
Implementation notes: AV_PLAN Phase 2 "As built". `none` = NoneSwitch, i.e. **alpha fixed at 1**, not zero dissipation.

**Reference: M0e.** At the default `supportVolumeClamp='walls'` (OPEN_PROBLEMS §20, resolved the same day) pure-fluid runs
are bit-identical to the M0e baseline, so the none / C&D columns are `results/av_M0e_baseline{,Q}`. Rosswog2020 runs:
`results/av_noclamp_rosswog2020{,Q}` (video). Isolation: adding the trigger moves no existing number (smoke 30/30
bit-identical). Lock: `rosswog2020Q` x sod / noh / sedov / gresho run twice, bit-identical.
Sweeps `results/av_M2_sweeps_walls/` (`scripts/probe_rosswogSweeps.py`), map `results/av_M2_maps_walls/`
(`scripts/probe_rosswogSedovMap.py`).

## Gate table (AV_PLAN Phase 2 Validation)

| Gate | Result | Verdict |
|---|---|---|
| Sod 1D: spikes < 0.5, alpha max > 0.3, alpha mean < 0.3 | P 0.059, A 0.036, 1.0, 0.151 | pass |
| Sod 1D: L1(v_x) <= C&D | 0.0037 vs 0.0052 | pass |
| Sod 2D/3D: contact spikes <= none | P 0.0756 vs 0.0743 (2D), 0.1157 vs 0.1153 (3D, nx 20); A below none in 2D | marginal fail on P (+1.8 % / +0.3 %) |
| Sedov: peak rho/rho0 approaches 4 from below | 2.53 (C_q 0), 2.09 (C_q 2), no overshoot; C&D 2.15 / 1.83 | pass, sharper than C&D |
| Noh: post-shock rho within 3 % | C_q 2: -0.08 %; C_q 0 degenerate for every switch (-50 %, cold gas) | pass at C_q 2 |
| Gresho: alphaMean < 0.05 and L1(v_phi) <= C&D | 0.047; 0.0772 vs 0.0726 | alpha passes; L1 6 % worse than C&D |
| Kelvin-Helmholtz: A(1.5) >= C&D | 0.0123 vs 0.0984 (none 0.0204) | **fail** |
| Resolution (Gresho nx 50/100/200): alphaMean non-increasing | 0.074 / 0.047 / 0.029 | pass |
| Timestep (Sod, cfl 0.3 vs 0.15): alpha within 10 % L1 | 6.0 % | pass |

## Findings

1. **Kelvin-Helmholtz is suppressed.** The trigger fires along the density-contrast shear layer (alphaMean 0.096 vs C&D 0.060)
   and the billows shear flat instead of rolling up (frames in `results/av_M2_rosswog2020/runs/rosswog2020_kelvinHelmholtz`
   vs `results/av_M2_ref_cd/runs/cullenDehnen2010_kelvinHelmholtz`, clamp-on runs; the clamp-off run is the same picture with
   less growth); A(1.5) is below even the fixed alpha = 1 run. Most likely standard SPH's contact error: summation density
   changes as particles slide along the rho 1:2 interface while u evolves adiabatically, so s = P/rho^gamma is noisy there.
   The paper says schemes without MAGMA2's velocity reconstruction may need different parameters; it does not say which.
   Not yet confirmed by an epsdot map of KH.
2. **Low-resolution floor.** At Sod nx 100 (62 particles per half) the rarefaction's ~1 % entropy error gives epsdot ~ 3e-3 and
   alpha 0.5-0.9 there; at nx 400 / 800 smooth regions are at alpha < 0.03 (paper: ~0.01). Gresho shows the same trend
   (the resolution row above): alpha falls ~1.6x per resolution doubling.
3. **Cold gas saturates the trigger.** Sedov's background has u = 0 exactly, so s = 0 and tau = h/c is infinite: any entropy
   the precursor makes is an infinite relative rate and alpha = 1 ahead of the shock. Inside the blast epsdot stays 0.02-0.1
   (>= eps_1), so alpha ~ 1 everywhere (the paper: ~1 at the shock, ~0.4 inside, at 200^3 with MAGMA2). Rosswog's own Sedov
   uses u_bg = 1e-10 u_c (float64). C&D is saturated at its alpha_max = 2 almost everywhere too. See the map (C_q = 2):

![Sedov alpha map](av_phase2_sedov_alpha_map.png)

4. Smooth and wave cases are where it beats C&D: linear wave alpha = 0 exactly (C&D 0.038; velocity error 0.00053 vs 0.0085),
   Yee peak-speed ratio 0.986 vs 0.956, Sedov peak 2.53 vs 2.15.

**Superseded first pass (clamp on everywhere, `results/av_M2_rosswog2020`, `results/av_M2_ref_*`):** same verdicts, except
Gresho alphaMean 0.0506 (marginal fail) and KH 0.020; see OPEN_PROBLEMS §20 in RESOLVED_PROBLEMS.md for the clamp itself.

## Tables (final-time metrics, default `supportVolumeClamp='walls'`)

### C_q=0

**sod**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| L1_vx | 0.003493 | 0.00524 | 0.003655 |
| contactSpikeP | 0.06033 | 0.1336 | 0.05883 |
| contactSpikeA | 0.03273 | 0.02183 | 0.03561 |
| R3_rho_err | -0.0007077 | -0.0003972 | 0.0002739 |
| R4_rho_err | -0.003597 | -0.002563 | -0.003656 |
| R4_P_err | -0.0001576 | -0.002588 | -0.0006576 |
| alphaMean | 1 | 0.08314 | 0.1508 |
| alphaMax | 1 | 0.9462 | 1 |
| avEnergyTotal | 0.0107 | 0.01011 | 0.01062 |

**sod2d**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| L1_vx | 0.01714 | 0.0172 | 0.01662 |
| contactSpikeP | 0.07428 | 0.08507 | 0.07563 |
| contactSpikeA | 0.02595 | 0.01448 | 0.02511 |
| alphaMean | 1 | 0.2855 | 0.2457 |

**sod3d**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| L1_vx | 0.06054 | 0.05873 | 0.05989 |
| contactSpikeP | 0.1153 | 0.1236 | 0.1157 |
| contactSpikeA | -0.1461 | -0.4098 | -0.1474 |
| alphaMean | 1 | 0.8288 | 0.5966 |

**sedov**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| peakRhoRatio | 2.532 | 2.154 | 2.53 |
| peakOvershoots | False | False | False |
| shockRadiusErr | -0.03165 | -0.01112 | -0.03123 |
| energyDrift | 0.0005215 | 0.0005298 | 0.0005111 |
| alphaMean | 1 | 1.799 | 0.9856 |

**noh**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| postShockRhoErr | -0.4996 | -0.4996 | -0.4996 |
| alphaMean | 1 | 1.425 | 0 |
| energyDrift | 0 | 0 | 0 |

**gresho**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| L1_vphi | 0.2098 | 0.07259 | 0.0772 |
| peakSpeed | 0.3487 | 0.8091 | 0.751 |
| angularMomentumLoss | 0.1737 | 0.008044 | 0.006895 |
| alphaMean | 1 | 0.04148 | 0.04712 |
| alphaMax | 1 | 0.1786 | 0.2802 |

**yee**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| L1_speed | 0.04584 | 0.01303 | 0.01182 |
| peakSpeedRatio | 0.8337 | 0.9559 | 0.9861 |
| alphaMean | 1 | 0.05777 | 0.05078 |

**linearWave**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| waveVelocityErr | 0.02366 | 0.008538 | 0.0005317 |
| alphaMean | 1 | 0.03807 | 0 |

**kelvinHelmholtz**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| khAmplitudeAt1p5 | 0.0204 | 0.09842 | 0.01234 |
| khAmplitudeMax | 0.0206 | 0.09842 | 0.02089 |
| alphaMean | 1 | 0.06022 | 0.09624 |

**rayleighTaylor**

| metric | none | cullenDehnen2010 | rosswog2020 |
|---|---|---|---|
| mixingWidth | 0.2598 | 0.3886 | 0.4222 |
| maxVelocity | 0.1506 | 0.3723 | 0.3368 |
| alphaMean | 1 | 0.04045 | 0.1072 |


### C_q=2

**sod**

| metric | noneQ | cullenDehnen2010Q | rosswog2020Q |
|---|---|---|---|
| L1_vx | 0.003681 | 0.004615 | 0.003808 |
| contactSpikeP | 0.07584 | 0.08215 | 0.06788 |
| contactSpikeA | 0.03666 | 0.03171 | 0.03832 |
| R3_rho_err | -0.001115 | -0.001766 | 9.45e-05 |
| R4_rho_err | -0.003813 | -0.0009482 | -0.003689 |
| R4_P_err | -0.0002094 | -0.001855 | -0.0005603 |
| alphaMean | 1 | 0.06663 | 0.1539 |
| alphaMax | 1 | 0.8153 | 0.9997 |
| avEnergyTotal | 0.01126 | 0.01036 | 0.01117 |

**sedov**

| metric | noneQ | cullenDehnen2010Q | rosswog2020Q |
|---|---|---|---|
| peakRhoRatio | 2.087 | 1.825 | 2.086 |
| peakOvershoots | False | False | False |
| shockRadiusErr | -0.01264 | -0.003258 | -0.01253 |
| energyDrift | 0.0005208 | 0.0005306 | 0.0005074 |
| alphaMean | 1 | 1.807 | 0.9922 |

**noh**

| metric | noneQ | cullenDehnen2010Q | rosswog2020Q |
|---|---|---|---|
| postShockRhoErr | -0.0007594 | 0.0006001 | -0.0008222 |
| alphaMean | 1 | 0.4356 | 0.4539 |
| energyDrift | 3.219e-06 | 1.323e-05 | 3.159e-06 |

**gresho**

| metric | noneQ | cullenDehnen2010Q | rosswog2020Q |
|---|---|---|---|
| L1_vphi | 0.2099 | 0.0718 | 0.07829 |
| peakSpeed | 0.3419 | 0.8426 | 0.7521 |
| angularMomentumLoss | 0.1775 | 0.007943 | 0.006751 |
| alphaMean | 1 | 0.0402 | 0.04558 |
| alphaMax | 1 | 0.1486 | 0.2662 |
