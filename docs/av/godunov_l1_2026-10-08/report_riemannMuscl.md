# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMuscl | 0.003663 | 0.06845 | 0.02995 | -0.0009695 | -0.003177 | -1.732e-05 | 1 | 1 | 1 | 0.01081 | 0 | 0 | 0.02351 | 3.901e-06 | 16.87 | 32.89 |

## sod: energy-drift budget (Group E)

all within budget

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMuscl | 0.01727 | 0.07093 | 0.02921 | -0.03938 | -0.004704 | 0.00492 | 1 | 1 | 1 | 0.002769 | 0 | 0 | 0.004792 | 3.473e-06 | 103.9 | 48.25 |

## sod2d: energy-drift budget (Group E)

all within budget

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMuscl | -0.001016 | 4 | 1 | 1 | 1 | 0.4687 | 0 | 0 | 0.2174 | 6.557e-07 | 17.79 | 17.75 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMuscl | 0.21 | 0.3414 | 0.1725 | 1 | 1 | 1 | 0.06615 | 0 | 0 | 0.04376 | 1.107e-07 | 101.2 | 50.46 |

## gresho: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMuscl | 2.329 | 4 | 0 | -0.02055 | -5.96e-08 | 1 | 1 | 1 | 0.491 | 0 | 0 | 0.03604 | 0.0005217 | 546.3 | 159.8 |

## sedov: energy-drift budget (Group E)

OVER: riemannMuscl (5.22e-04 > 1e-04)

## meta

```
timestamp: 2026-10-08T17:00:56+01:00
warp: 1.17.0
torch: 2.13.0+cu130
device: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
cudaAvailable: True
precision: float32
python: 3.13.14
warpSPH@git: e094a75
warpSPHIntegrators@git: cb86424
warpSPHCore@git: 894a6fe
profile: full
config: riemannMuscl
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
