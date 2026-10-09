# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimited | 0.003852 | 0.06285 | 0.01897 | -0.0005649 | -0.003985 | -0.002269 | 1 | 1 | 1 | 0.009947 | 0 | 0 | 0.02229 | 2.167e-06 | 16.86 | 138.7 |

## sod: energy-drift budget (Group E)

all within budget

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimited | 0.0175 | 0.08017 | 0.006292 | -0.04329 | 0.01208 | 0.01269 | 1 | 1 | 1 | 0.002224 | 0 | 0 | 0.003716 | 6.461e-06 | 103.7 | 114.5 |

## sod2d: energy-drift budget (Group E)

all within budget

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimited | -0.002967 | 4 | 1 | 1 | 1 | 0.5023 | 0 | 0 | 0.2163 | 5.96e-08 | 17.9 | 17.77 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimited | 0.09055 | 0.701 | -0.001561 | 1 | 1 | 1 | 0.03196 | 0 | 0 | 0.02044 | 1.107e-07 | 101.1 | 40.42 |

## gresho: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannMusclLimited | 2.463 | 4 | 0 | -0.02623 | -5.96e-08 | 1 | 1 | 1 | 0.4215 | 0 | 0 | 0.003658 | 0.0005217 | 545.8 | 248.7 |

## sedov: energy-drift budget (Group E)

OVER: riemannMusclLimited (5.22e-04 > 1e-04)

## meta

```
timestamp: 2026-10-08T16:52:19+01:00
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
config: riemannMusclLimited
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
