# AV report (full)

## sod (full)

| config | L1_vx | contactSpikeP | contactSpikeA | R3_rho_err | R4_rho_err | R4_P_err | alphaMean | alphaMax | alphaActiveFraction | avEnergyLinear | avEnergyQuadratic | chenNixonRatioMedian | entropyGain | energyDrift | neighboursMean | wallMsPerStep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| riemann | 0.003687 | 0.06854 | 0.02867 | -0.001074 | -0.002978 | -8.225e-08 | 1 | 1 | 1 | 0.01076 | 0 | 0 | 0.0234 | 3.757e-06 | 16.87 | 104.9 |
| riemannAcoustic | 0.003603 | 0.0653 | 0.02984 | -0.0008573 | -0.003207 | -3.415e-05 | 1 | 1 | 1 | 0.01071 | 0 | 0 | 0.0234 | 3.829e-06 | 16.87 | 30.48 |
| riemannTSRS | 0.00369 | 0.07061 | 0.02952 | -0.001082 | -0.003091 | -8.092e-06 | 1 | 1 | 1 | 0.01084 | 0 | 0 | 0.02352 | 3.829e-06 | 16.87 | 28.76 |
| riemannHLLC | 0.003707 | 0.07166 | 0.02961 | -0.001141 | -0.003106 | -1.205e-05 | 1 | 1 | 1 | 0.01088 | 0 | 0 | 0.02356 | 3.757e-06 | 16.87 | 28.65 |
| riemannLimited | 0.00389 | 0.06262 | 0.01803 | -0.000588 | -0.004052 | -0.002408 | 1 | 1 | 1 | 0.009948 | 0 | 0 | 0.02226 | 2.023e-06 | 16.85 | 30.11 |
| riemannLimitedB2 | 0.003694 | 0.06837 | 0.02864 | -0.001063 | -0.002967 | 7.015e-06 | 1 | 1 | 1 | 0.01075 | 0 | 0 | 0.02339 | 3.685e-06 | 16.87 | 29.72 |
| riemannAcousticLimited | 0.003809 | 0.061 | 0.01955 | -0.0004463 | -0.004399 | -0.002571 | 1 | 1 | 1 | 0.009919 | 0 | 0 | 0.02232 | 2.167e-06 | 16.85 | 29.43 |

## sod: energy-drift budget (Group E)

all within budget

## meta

```
timestamp: 2026-10-08T15:35:03+01:00
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
config: phase7b
repeat: 1
video: True
supportVolumeClamp: None
caseParam: []
schemeParam: []
runParam: []
```
