# AV baseline (M0) — 2026-09-30

Produced by `scripts/av_report.py --config baseline --profile full --repeat 2` on
warpSPH `0b063d5` + the uncommitted M0 work (diagnostics, `avPower.py`, `av_report.py`,
Read-Hayfield determinism fix). Host scheme Monaghan, kernel B7, RK2, `C_l=1, C_q=0`,
`Price2012_98`. Sod: `nx=400, nSteps=400`; Gresho: `nx=100, tLimit=3`. Videos
(one per config x case) are in `results/av_baseline_2026-09-30/runs/` (gitignored).

**Cases so far: `sod`, `gresho` only** — the M0 list in AV_PLAN §Phase 0 also wants
sod2d/3d, sedov, noh, yee, linearWave, KH, RT. Not yet included: shock width, peak
`rho/rho0` (Sedov), StepTimer-based ms/step (`wallMsPerStep` below includes setup).

## Reading the numbers

- **`avEnergyQuadratic = 0` everywhere**: the Monaghan default has `C_q = 0`, so this
  baseline has no quadratic viscosity at all (AV_PLAN §2.4 / Phase 5A finding, confirmed).
- **NoneSwitch (alpha = 1) on Gresho** loses the vortex (peak speed 0.34, L1 0.21,
  17.6 % angular momentum lost); both switches keep it (peak ~0.79-0.80, L1 ~0.07,
  <1 % angular-momentum loss). C&D and R&H are close on Gresho; R&H marginally better.
- **Sod**: all three within ~0.005 `L1(v_x)`; C&D has the largest contact P spike (19 %
  vs 8 % NoneSwitch / 7 % R&H) — a point for the Phase 2 comparison.
- **Energy-drift budget (Group E)**: the Monaghan budget `5e-3` is exceeded on Sod by all
  three configs (5.5e-3 – 6.5e-3) and on Gresho by NoneSwitch (7.7e-3). The budget comes from
  `tests/test_physics.py`, which runs a different (smaller) setup; record, do not conclude.
- **Reproducibility lock**: all six (config, case) pairs are bit-identical across two runs.
  The first attempt failed for R&H/Gresho (`alphaMax` differed by 19 %): its
  entropy-dissipation term used `scatter_sum` (CUDA atomic add, run-to-run summation
  order), and the alpha switch amplified it. Fixed with a fixed-order `segment_reduce`
  (`ReadHayfield2012.py`); the pre-fix numbers were not kept as the baseline.

# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.005008 | 0.08011 | 0.1209 | 0.006933 | -0.03122 | 0.01055 | 1 | 1 | 1 | 0.01072 | 0 | 0.04653 | 0.006466 | 16.92 | 29.39 |
| cullenDehnen2010 | 0.005882 | 0.1918 | 0.1016 | 0.00669 | -0.03063 | 0.00782 | 0.08196 | 0.9506 | 0.192 | 0.01058 | 0 | 0.04484 | 0.006441 | 16.83 | 29.94 |
| readHayfield2012 | 0.004847 | 0.07148 | 0.1113 | 0.006118 | -0.03692 | 0.009014 | 0.2389 | 0.8257 | 1 | 0.01111 | 0 | 0.04262 | 0.005473 | 16.86 | 29.99 |

## sod: energy-drift budget (Group E)

OVER: none (6.47e-03 > 5e-03), cullenDehnen2010 (6.44e-03 > 5e-03), readHayfield2012 (5.47e-03 > 5e-03)

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.209 | 0.3417 | 0.1756 | 1 | 1 | 1 | 0.06614 | 0 | 0.088 | 0.007676 | 105.2 | 38.03 |
| cullenDehnen2010 | 0.07312 | 0.787 | 0.007717 | 0.04137 | 0.1487 | 0.0105 | 0.02635 | 0 | 0.03427 | 0.003083 | 105.3 | 40.54 |
| readHayfield2012 | 0.07049 | 0.8014 | 0.005125 | 0.04186 | 0.3484 | 0.0597 | 0.02534 | 0 | 0.03284 | 0.002992 | 105.3 | 41.26 |

## gresho: energy-drift budget (Group E)

OVER: none (7.68e-03 > 5e-03)

## Reproducibility lock (tol 1e-06)

| run | max rel diff | worst metric | state |
|---|---|---|---|
| none/sod | 0.00e+00 | - | LOCKED |
| none/gresho | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/sod | 0.00e+00 | - | LOCKED |
| cullenDehnen2010/gresho | 0.00e+00 | - | LOCKED |
| readHayfield2012/sod | 0.00e+00 | - | LOCKED |
| readHayfield2012/gresho | 0.00e+00 | - | LOCKED |

## meta

```
timestamp: 2026-09-30T14:29:51+01:00
warp: 1.17.0
torch: 2.13.0+cu130
device: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
cudaAvailable: True
precision: float32
python: 3.13.14
warpSPH@git: 0b063d5
warpSPHIntegrators@git: cb86424
warpSPHCore@git: 894a6fe
profile: full
config: baseline
repeat: 2
video: True
```
