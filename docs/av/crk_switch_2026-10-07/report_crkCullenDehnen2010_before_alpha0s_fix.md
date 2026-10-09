# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crkCullenDehnen2010 | 0.006712 | 0.03005 | 0.006814 | -0.007558 | -0.004314 | -0.005017 | 0.9794 | 0.9934 | 1 | - | - | 0.02322 | 3.829e-06 | 16.75 | 91.47 |

## sod: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crkCullenDehnen2010 | 0.03315 | 1.018 | -0.0004195 | 0.9704 | 0.9744 | 1 | - | - | -0.006477 | 0 | 101.1 | 186.4 |

## gresho: energy-drift budget (Group E)

all within budget

## kelvinHelmholtz (full)

| config | khAmplitude0 | khAmplitudeAt1p5 | khAmplitudeMax | khReference1p5 | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| crkCullenDehnen2010 | 0.01054 | 0.09036 | 0.09036 | 0.1479 | 0.983 | 0.9931 | 1 | - | - | -0.01441 | 6.099e-08 | 100.6 | 290.8 |

## kelvinHelmholtz: energy-drift budget (Group E)

all within budget

## meta

```
timestamp: 2026-10-07T10:27:10+01:00
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
