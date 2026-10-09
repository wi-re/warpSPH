# AV report (full)

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemann | -0.0009458 | 4 | 1 | 1 | 1 | 0.4652 | 0 | 0 | 0.2176 | 1.192e-06 | 17.78 | 15.56 |

## noh: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemann | 2.322 | 4 | 0 | -0.02026 | -5.96e-08 | 1 | 1 | 1 | 0.4984 | 0 | 0 | 0.03946 | 0.0005222 | 546.4 | 192.5 |

## sedov: energy-drift budget (Group E)

OVER: riemann (5.22e-04 > 1e-04)

## meta

```
timestamp: 2026-10-08T16:24:58+01:00
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
