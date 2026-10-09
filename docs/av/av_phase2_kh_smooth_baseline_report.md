# AV report (full)

## kelvinHelmholtz (full)

| config | khAmplitude0 | khAmplitudeAt1p5 | khAmplitudeMax | khReference1p5 | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.01054 | 0.01835 | 0.01835 | 0.1479 | 1 | 1 | 1 | 0.03164 | 0 | 0.01173 | 0 | 100.3 | 137.3 |
| cullenDehnen2010 | 0.01054 | 0.1064 | 0.1064 | 0.1479 | 0.06069 | 0.5475 | 0.1295 | 0.00566 | 0 | 0.001402 | 8.538e-07 | 100.3 | 139 |
| readHayfield2012 | 0.01054 | 0.07126 | 0.07126 | 0.1479 | 0.2 | 0.2 | 1 | 0.01058 | 0 | 0.003052 | 1.83e-07 | 100.2 | 140.6 |

## kelvinHelmholtz: energy-drift budget (Group E)

all within budget

## meta

```
timestamp: 2026-10-07T08:29:16+01:00
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
config: baseline
repeat: 1
video: True
supportVolumeClamp: None
caseParam: ['smoothDensity=1']
schemeParam: []
```
