# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2 | 0.002856 | 0.01043 | 0.08483 | 0.001944 | -5.043e-07 | -0.0002238 | 1 | 1 | 1 | - | - | - | 0.02203 | 5.39e-05 | 16.91 | 27.66 |

## sod: energy-drift budget (Group E)

all within budget

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2 | 0.01665 | 0.05288 | 0.05356 | -0.02321 | -0.02149 | -0.01278 | 1 | 1 | 1 | - | - | - | 0.004333 | 0.000284 | 103.9 | 35.74 |

## sod2d: energy-drift budget (Group E)

all within budget

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2 | -0.0007251 | 4 | 1 | 1 | 1 | - | - | - | 0.2177 | 5.96e-07 | 17.84 | 13.01 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2 | 0.13 | 0.6136 | 0.005969 | 1 | 1 | 1 | - | - | - | 0.02766 | 7.747e-07 | 101.2 | 36.13 |

## gresho: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO2 | 2.407 | 4 | 0 | -0.04786 | -5.96e-08 | 1 | 1 | 1 | - | - | - | 0.2218 | 0.0005681 | 544.1 | 132.8 |

## sedov: energy-drift budget (Group E)

all within budget

## meta

```
timestamp: 2026-10-08T17:47:47+01:00
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
config: gsphO2
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
