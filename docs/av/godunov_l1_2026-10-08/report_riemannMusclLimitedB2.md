# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimitedB2 | 0.003666 | 0.0683 | 0.02993 | -0.0009496 | -0.003161 | -3.732e-06 | 1 | 1 | 1 | 0.01081 | 0 | 0 | 0.02351 | 3.901e-06 | 16.87 | 33.68 |

## sod: energy-drift budget (Group E)

all within budget

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimitedB2 | 0.01726 | 0.07094 | 0.0292 | -0.0394 | -0.004704 | 0.004906 | 1 | 1 | 1 | 0.002768 | 0 | 0 | 0.004791 | 3.543e-06 | 103.9 | 54.88 |

## sod2d: energy-drift budget (Group E)

all within budget

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimitedB2 | -0.001017 | 4 | 1 | 1 | 1 | 0.4687 | 0 | 0 | 0.2174 | 1.132e-06 | 17.79 | 21.68 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimitedB2 | 0.09006 | 0.6876 | -0.001705 | 1 | 1 | 1 | 0.03191 | 0 | 0 | 0.02042 | 1.107e-07 | 101.1 | 41.08 |

## gresho: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimitedB2 | 2.354 | 4 | 0 | -0.02163 | -5.96e-08 | 1 | 1 | 1 | 0.4819 | 0 | 0 | 0.03164 | 0.0005217 | 546.3 | 143.8 |

## sedov: energy-drift budget (Group E)

OVER: riemannMusclLimitedB2 (5.22e-04 > 1e-04)

## meta

```
timestamp: 2026-10-08T17:08:00+01:00
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
config: riemannMusclLimitedB2
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
