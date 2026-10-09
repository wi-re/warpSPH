# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO1 | 0.01197 | 0.005557 | 0.09299 | -0.006694 | -0.004325 | 0.0002018 | 1 | 1 | 1 | - | - | - | 0.03268 | 7.145e-05 | 16.9 | 26.52 |

## sod: energy-drift budget (Group E)

all within budget

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO1 | 0.03722 | 0.04913 | 0.07227 | -0.04628 | -0.05 | -0.02489 | 1 | 1 | 1 | - | - | - | 0.01024 | 0.0003404 | 104 | 33.79 |

## sod2d: energy-drift budget (Group E)

all within budget

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO1 | -0.0003331 | 4 | 1 | 1 | 1 | - | - | - | 0.2203 | 1.311e-06 | 17.77 | 11.96 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO1 | 0.2715 | 0.1737 | 0.4902 | 1 | 1 | 1 | - | - | - | 0.05193 | 9.961e-07 | 101.3 | 36.73 |

## gresho: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsphO1 | 2.154 | 4 | 0 | -0.04439 | -5.96e-08 | 1 | 1 | 1 | - | - | - | 0.2052 | 0.0005326 | 544.1 | 68.35 |

## sedov: energy-drift budget (Group E)

all within budget

## meta

```
timestamp: 2026-10-08T17:53:12+01:00
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
config: gsphO1
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
