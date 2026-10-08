# AV report (full)

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannLimitedB2 | 0.01553 | 0.07076 | 0.02732 | -0.04007 | -0.003002 | 0.006319 | 1 | 1 | 1 | 0.002746 | 0 | 0 | 0.004723 | 3.821e-06 | 103.9 | 66.35 |

## sod2d: energy-drift budget (Group E)

all within budget

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannLimitedB2 | -0.0009468 | 4 | 1 | 1 | 1 | 0.4652 | 0 | 0 | 0.2176 | 1.788e-07 | 17.78 | 22.81 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannLimitedB2 | 0.09099 | 0.6776 | -0.0009943 | 1 | 1 | 1 | 0.03241 | 0 | 0 | 0.02074 | 1.107e-07 | 101.1 | 45.99 |

## gresho: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannLimitedB2 | 2.347 | 4 | 0 | -0.02125 | -5.96e-08 | 1 | 1 | 1 | 0.4899 | 0 | 0 | 0.03528 | 0.0005215 | 546.3 | 138.9 |

## sedov: energy-drift budget (Group E)

OVER: riemannLimitedB2 (5.21e-04 > 1e-04)

## meta

```
timestamp: 2026-10-08T15:50:15+01:00
warp: 1.17.0
torch: 2.13.0+cu130
device: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
cudaAvailable: True
precision: float32
python: 3.13.14
warpSPH@git: 75c6254
warpSPHIntegrators@git: cb86424
warpSPHCore@git: 894a6fe
profile: full
config: riemannLimitedB2
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
