# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crkCullenDehnen2010 | 0.008401 | 0.03151 | -0.002811 | -0.007295 | -0.005051 | -0.008119 | 0.07308 | 0.8137 | 0.184 | - | - | 0.02227 | 3.829e-06 | 16.79 | 48.33 |

## sod: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crkCullenDehnen2010 | 0.02757 | 1.067 | -0.0009937 | 0.02684 | 0.1403 | 0.0009 | - | - | -0.003189 | 2.213e-07 | 101.2 | 81.5 |

## gresho: energy-drift budget (Group E)

all within budget

## kelvinHelmholtz (full)

| config | khAmplitude0 | khAmplitudeAt1p5 | khAmplitudeMax | khReference1p5 | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crkCullenDehnen2010 | 0.01054 | 0.1232 | 0.1232 | 0.1479 | 0.0466 | 0.7318 | 0.08185 | - | - | -0.01779 | 6.099e-08 | 100.8 | 139.1 |

## kelvinHelmholtz: energy-drift budget (Group E)

all within budget

## meta

```
timestamp: 2026-10-07T11:11:48+01:00
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
config: crkCullenDehnen2010
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
```
