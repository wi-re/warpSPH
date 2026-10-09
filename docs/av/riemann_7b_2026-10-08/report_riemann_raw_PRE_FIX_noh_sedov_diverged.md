# AV report (full)

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemann | 0.01551 | 0.07078 | 0.02733 | -0.04006 | -0.003002 | 0.006333 | 1 | 1 | 1 | 0.002746 | 0 | 0 | 0.004724 | 3.821e-06 | 103.9 | 353.8 |

## sod2d: energy-drift budget (Group E)

all within budget

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemann **DIVERGED** | nan | 4 | 1 | 1 | 1 | - | - | - | nan | nan | 16.09 | 4.129e+04 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemann | 0.2098 | 0.345 | 0.1726 | 1 | 1 | 1 | 0.06609 | 0 | 0 | 0.04373 | 1.107e-07 | 101.2 | 39.15 |

## gresho: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemann **DIVERGED** | 1 | 4 | 0 | nan | -5.96e-08 | 1 | 1 | 1 | - | - | 0 | nan | nan | 515 | 6.885e+04 |

## sedov: energy-drift budget (Group E)

all within budget

## meta

```
timestamp: 2026-10-08T15:38:14+01:00
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
config: riemann
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
