# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crkNone | 0.006691 | 0.03011 | 0.007282 | -0.007535 | -0.00428 | -0.004956 | 1 | 1 | 1 | - | - | 0.02324 | 3.829e-06 | 16.75 | 45.11 |

## sod: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crkNone | 0.03655 | 1.016 | -0.0001381 | 1 | 1 | 1 | - | - | -0.006409 | 1.107e-07 | 101.2 | 78.95 |

## gresho: energy-drift budget (Group E)

all within budget

## kelvinHelmholtz (full)

| config | khAmplitude0 | khAmplitudeAt1p5 | khAmplitudeMax | khReference1p5 | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crkNone | 0.01054 | 0.09008 | 0.09008 | 0.1479 | 1 | 1 | 1 | - | - | -0.0144 | 6.099e-08 | 100.6 | 140.6 |

## kelvinHelmholtz: energy-drift budget (Group E)

all within budget

## meta

```
timestamp: 2026-10-07T11:02:15+01:00
warp: 1.17.0
torch: 2.13.0+cu130
device: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
cudaAvailable: True
precision: float32
python: 3.13.14
warpSPH@git: 2b24036
warpSPHIntegrators@git: cb86424
warpSPHCore@git: 894a6fe
profile: full
config: crkNone
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
```
