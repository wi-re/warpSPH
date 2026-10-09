# AV_PLAN Phase 7 -- bake-off verdict (run `av_bakeoff_2026-10-09`, code abdaf11)

Full tables: [`report.md`](report.md) (also `results/av_foundation/report.md`). Raw runs and videos stay in
`results/av_bakeoff_2026-10-09/` (54 GB, gitignored). Detector maps: `sod2d_detectors.png`, `sedov2d_detectors.png`.
Written 2026-10-09 from the report tables only; no new runs.

## Outcome

13 rows x 12 cases, CompSPH cross-check (4 switches), KH nx 256 (2 rows), timing, maps: all 18 jobs rc 0
(4.5 h), **no Monaghan row diverged**. One CompSPH cross-check run blew up (below). Monaghan rows reproduce the
2026-10-08 sweep rows to the printed digits (rosswogLimited / Coupled, Sod to Sedov).

## Gates missed (Part 0 thresholds)

| gate | rows | reading |
|---|---|---|
| Sedov energy drift 1e-4 | all 13 (4.9e-4 .. 5.9e-4) | scheme-independent; OPEN_PROBLEMS §18: RK2 time-integration error, converges at 2nd order in dt. Not an AV finding |
| Rayleigh-Taylor energy drift 1e-4 | all 13 (4.3e-4 .. 2.1e-3) | same order for every row; the fixed-alpha row 1 is lowest, the Rosswog family highest. Not investigated |
| Noh 2D post-shock rho +-3 % | 11 of 13; **pass: rows 7, 11 (Wadsley, -1.3 % / -3.0 %)** | best of the rest rows 8/10/12 (limited pair velocity, -5.6 / -5.7 %); raw rows -8 .. -13 %; Sphenix -15.8 %. Wadsley is also the row whose cylindrical-Noh pre-shock alpha stays ~0 (4e-8; the others 0.25 .. 2); a link between the two is suggested, not tested |
| Gresho energy drift 1e-4 | rows 6, 11, 13 (1.6e-4, 1.3e-4, 3.1e-4) | the Sphenix pair and wadsleyLimited; the rest are 5e-6 .. 9e-5 |

## Per-row notes

| row | config | note |
|---|---|---|
| 1 | gsAV (alpha = 1) | No detector cost, near-best Sod 1D (0.00369), but worst everywhere smooth: Gresho 0.21, Yee 0.051, linear wave 0.025, KH A(1.5) 0.019, RT 0.258, shear box mode 0.844 |
| 2 | boCD | Sod 0.0043, Sedov -0.4 %, Gresho 0.078; KH 0.076 (half the reference), shear box 0.98 |
| 3 | boCH | Like C&D; best Sod 3D (0.0544); Sedov radius -1.5 %, linear wave 0.017 |
| 4 | boRH | Worst Gresho of the detectors (0.109), Sedov -2.4 %; alpha is active everywhere on Sod and Sod 2D (activeFraction 1) |
| 5 | cnRosswogFixed2 | Best linear wave (6e-4), but KH collapses (A(1.5) 0.012; the sharp-IC transient, see Phase 2) and RT energy drift 1.8e-3 |
| 6 | sphenixFixed3 | Sphenix at beta 3: Sod 2D 0.0159, Yee 6.6e-4, Gresho 0.072, KH 0.116, shear box 1.000; Sod 1D 0.0046 and Gresho energy drift 1.6e-4 are the costs. See R1 below |
| 7 | wadsley2017 | Sod 1D 0.0092 (2x the rest), Sod 2D contact spike 0.22 and R4 error +3.7 %; best cylindrical Noh pre-shock alpha and Noh 2D, Gresho 0.067 (best raw-velocity row), linear wave 6.5e-4; cost 13.2 ms/step (+33 % over C&D) |
| 8 | gsAVSLR | alpha = 1 with the limited pair velocity: Sod 2D 0.017, Noh 2D -5.7 %, Gresho 0.090 (vs 0.21 raw) |
| 9 | gsAVSLRB2 | alpha = 1, limited + Balsara p = 2: best Sod 1D (0.003689); Sod / Sedov / Noh equal row 1, Gresho 0.089 and KH 0.097 equal row 8 (the Balsara factor switches the dissipation off in shear only) |
| 10 | rosswogLimited | Best Gresho (0.0587) and RT mixing width (0.445); Sod 2D 0.0165; Noh 2D -5.6 % |
| 11 | wadsleyLimited | Best Yee (6.0e-4); Sod 1D 0.0129 (worst), Sedov -3.2 % (worst), Noh 2D -3.0 %; Gresho energy drift 1.3e-4 |
| 12 | rosswogLimitedCoupled (default) | Within 3 % of row 10 on most cases (beta = 2 alpha changes the quadratic part only: AV quadratic energy 4-7x lower on Gresho / Yee, orders lower in the shear box); rarely best or worst. Costs vs Sphenix: Sedov radius -1.5 % (vs -0.3 %), KH 0.116 (vs 0.121), linear wave 0.0056 |
| 13 | sphenix (paper coupled beta) | Best Sedov (-0.3 %), Noh 1D (9e-5), KH nx 128 (0.121), Sod 2D (0.0159), shear box; worst Noh 2D (-15.8 %), Gresho 0.077 with energy drift 3.1e-4, Sod 1D 0.0050 |

(Row 9's Gresho / Yee match row 8; its Sod/Sedov/Noh numbers match row 1 -- the Balsara p = 2 limiter acts where the
gradient is shearing, so it leaves the compressive cases at their alpha = 1 values.)

## Phase 5B, R1: Sphenix at ell_V 5 (row 6, fixed beta 3) against C&D (row 2)

| metric | Sphenix (row 6) | C&D (row 2) | within 5 %? |
|---|---|---|---|
| Sod 1D L1(v_x) | 0.004556 | 0.004259 | no, +7.0 % |
| Sod 2D L1(v_x) | 0.01593 | 0.01861 | yes (better) |
| Sod 3D L1(v_x) | 0.05529 | 0.05544 | yes |
| Sedov shock radius | -0.41 % | -0.38 % | yes (3e-4 absolute) |
| Noh post-shock rho | -0.0010 | -0.0005 | both < 0.1 % |
| Noh 2D post-shock rho | -12.0 % | -11.7 % | yes; both miss the +-3 % gate |
| Gresho L1(v_phi) | 0.0720 | 0.0777 | **yes, below C&D's** (was 0.077 vs 0.073 in the sweep) |
| Gresho alphaMean | 0.02883 | 0.02851 | 1 % above C&D's, effectively equal |
| KH A(1.5), nx 128 | 0.1163 | 0.0759 | yes, above C&D's |
| ms/step (Gresho nx 128) | 8.79 | 9.90 | 11.2 % lower (target 15 %: diagnosed R4, the Balsara curl loop) |

The earlier ell_V 0.05 Sod failure (2.6x C&D) is gone at ell_V 5. Remaining differences: Sod 1D +7 % and a Gresho
energy drift of 1.6e-4 (gate 1e-4; C&D 9.6e-6).

## CompSPH cross-check

Switch behaviour carries over (Rosswog and C&D cut the Gresho error from 0.161 to 0.07; Wadsley the best KH, 0.103, C&D
0.098, Rosswog 0.034 on the sharp IC). **`compRosswog` Sedov blew up**: |v|max 1e7 at step 122 (t = 0.056), non-finite at
step 123, alphaMean 0.45. The other three CompSPH Sedov runs complete (`compNone`, `compCD`, `compWadsley`). Filed as OPEN_PROBLEMS §23.

## KH nx 256 (smooth IC, t = 3)

rosswogLimitedCoupled A(1.5) 0.1509 (peak 0.222), sphenix 0.1505 (0.220); McNally reference 0.1479. Both meet the
reference; CRKSPH 0.1446.

## Cost (Gresho nx 128, ms/step)

none 8.00, rosswog2020 8.09, sphenix 8.79, rosswogLimitedCoupled 8.86, gsAVSLRB2 8.90, C&D 9.90, Wadsley 13.15. The
default costs 11 % over no AV at all.

## Reading for M8

The default (row 12) stays: the plan chose it on the 2026-10-08 sweep and the bake-off reproduces those numbers. The
bake-off's own findings: both ingredients help smooth flow: on Gresho the limited pair velocity alone at alpha = 1 takes 0.21 to 0.090 (rows 1, 8), the
detector alone to 0.078 (row 2), together 0.059-0.060 (rows 10, 12), Sphenix at ell_V 5 is the competitive cheap alternative (rows 6 / 13), and Wadsley is the only row to pass
Noh 2D but costs the Sod contact and +33 % in time over C&D.
