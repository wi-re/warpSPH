# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2v | 0.004055 | 0.00599 | 0.06541 | -0.00168 | -0.00141 | 0.0005945 | 1 | 1 | 1 | - | - | - | 0.02613 | 6.524e-05 | 16.9 | 29.8 |

## sod: energy-drift budget (Group E)

all within budget

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2v | 0.02541 | 0.04546 | 0.04389 | -0.04527 | -0.02353 | -0.003853 | 1 | 1 | 1 | - | - | - | 0.007543 | 0.0003433 | 103.8 | 37.79 |

## sod2d: energy-drift budget (Group E)

all within budget

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2v | -0.0006404 | 4 | 1 | 1 | 1 | - | - | - | 0.2176 | 0 | 17.9 | 11.86 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2v | 0.1333 | 0.5784 | 0.01082 | 1 | 1 | 1 | - | - | - | 0.02947 | 5.534e-07 | 101.3 | 34.1 |

## gresho: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2v | 2.316 | 4 | 0 | -0.0434 | -5.96e-08 | 1 | 1 | 1 | - | - | - | 0.1691 | 0.0005291 | 544.4 | 82.67 |

## sedov: energy-drift budget (Group E)

all within budget

## meta

```
timestamp: 2026-10-08T17:57:36+01:00
warp: 1.17.0
torch: 2.13.0+cu130
device: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
cudaAvailable: True
precision: float32
python: 3.13.14
warpSPH@git: 8a9e66f
warpSPHIntegrators@git: cb86424
warpSPHCore@git: 894a6fe
profile: full
config: gsphO2v
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
