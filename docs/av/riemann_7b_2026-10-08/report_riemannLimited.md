# AV report (full)

## sod2d (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannLimited | 0.01759 | 0.07937 | 0.005089 | -0.04372 | 0.01371 | 0.0144 | 1 | 1 | 1 | 0.002211 | 0 | 0 | 0.003672 | 6.322e-06 | 103.7 | 417.3 |

## sod2d: energy-drift budget (Group E)

all within budget

## noh (full)

| config | postShockRhoErr | postShockRhoExact | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannLimited | -0.002666 | 4 | 1 | 1 | 1 | 0.4999 | 0 | 0 | 0.2163 | 3.576e-07 | 17.72 | 26.28 |

## noh: energy-drift budget (Group E)

all within budget

## gresho (full)

| config | L1_vphi | peakSpeed | angularMomentumLoss | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannLimited | 0.09056 | 0.7021 | -0.001458 | 1 | 1 | 1 | 0.03197 | 0 | 0 | 0.02044 | 1.107e-07 | 101.1 | 44.61 |

## gresho: energy-drift budget (Group E)

all within budget

## sedov (full)

| config | peakRhoRatio | peakRhoExact | peakOvershoots | shockRadiusErr | E0Recovery | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemannLimited | 2.463 | 4 | 0 | -0.02348 | -5.96e-08 | 1 | 1 | 1 | 0.4288 | 0 | 0 | 0.006198 | 0.0005221 | 545.9 | 149.5 |

## sedov: energy-drift budget (Group E)

OVER: riemannLimited (5.22e-04 > 1e-04)

## meta

```
timestamp: 2026-10-08T15:42:41+01:00
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
config: riemannLimited
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
