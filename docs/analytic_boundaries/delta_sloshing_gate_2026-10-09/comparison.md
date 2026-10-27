| run | impact 1 [Pa] @ s | impact 2 | impact 3 | lag vs measurement [ms] | rms err of the smoothed record in the windows [Pa] |
|---|---|---|---|---|---|
| measured, same smoothing | 1800 @ 2.390 | 2322 @ 4.066 | 1171 @ 5.697 | | |
| DeltaSPH2D, analytic walls (C4, Sun shift) | 12361 @ 2.336 | 4198 @ 3.997 | 10393 @ 5.579 | -54 / -69 / -118 | 1818 / 1174 / 2316 |
| warpSPH delta+, analytic walls (C2, Michel) | 4549 @ 2.351 | 5041 @ 4.016 | 4835 @ 5.623 | -39 / -50 / -74 | 810 / 1080 / 1078 |
| warpSPH delta+, boundary particles (C4, Michel) | 4877 @ 2.343 | 5157 @ 4.005 | 5277 @ 5.557 | -47 / -61 / -140 | 901 / 1087 / 1155 |

kinetic energy against DeltaSPH2D on a 1 ms grid (mean KE of DeltaSPH2D 9.472e-03):
| run | correlation | rms difference / mean | KE peak ratio | cross-correlation lag [ms] | density range | steps |
|---|---|---|---|---|---|---|
| warpSPH delta+, analytic walls (C2, Michel) | 0.990 | 0.113 | 1.05 | +0 | [0.75, 1.34] | 70001 |
| warpSPH delta+, boundary particles (C4, Michel) | 0.974 | 0.204 | 1.24 | -20 | [0.82, 1.33] | 70001 |
DeltaSPH2D density range [0.83, 1.20], 70001 steps, 5439 s
