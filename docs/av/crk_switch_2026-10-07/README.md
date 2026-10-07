# CRKSPH viscosity switch -- the evidence (2026-10-07)

Two defects hid each other (AV_PLAN Phase 1 clean-up, items 2-3), both fixed the same day:

1. CRK's one-sided `Q_i`, `Q_j` (Frontiere et al. 2017 Eq. 69) used raw `C_l`, `C_q`: the switched alpha and `BetaMode` never reached the viscosity.
2. `schemes/crkSPH.py` never wrote the switch's `alpha0s` back (the `updateViscositySwitch` call had been deleted in S2), so under C&D alpha stayed
   at ~0.97-0.98 (initial 1, decayed one step per step).

`report_crkCullenDehnen2010_before_alpha0s_fix.md` is the state after fix 1 only: C&D alphaMean 0.97-0.98, metrics ~ `crkNone`.
`report_*_full.md` are the final numbers (Monaghan-1992 `mu` through the shared `pi` package, `Frontiere2017` term; `alpha0s` restored).

Full profile, `none` -> `C&D` (McNally KH reference A(1.5) = 0.148):

| case | metric | crkNone (alpha = 1) | crkCullenDehnen2010 |
|---|---|---|---|
| Sod | L1(v_x) | 0.00669 | 0.00840 |
| Sod | alphaMean / alphaMax | 1 / 1 | 0.073 / 0.81 |
| Gresho | L1(v_phi) | 0.0366 | 0.0276 |
| Gresho | peak speed | 1.016 | 1.067 |
| Gresho | alphaMean / alphaMax | 1 / 1 | 0.027 / 0.14 |
| KH | A(1.5) | 0.0901 | **0.1232** |
| KH | alphaMean / alphaMax | 1 / 1 | 0.047 / 0.73 |

Frames: `kh_t1p5_*.png`, `gresho_t3_*.png` (last frame of the full-profile video runs; no fliers, no alarms). C&D rolls the KH billows up further
(peak speed 0.72 vs 0.60) and keeps the Gresho vortex ring; the Gresho peak speed above 1 is the intrinsic CRK spin-up (CRKSPH_LIMITER_PLAN) that the
lower alpha now damps less.

Every earlier `crkCullenDehnen2010` number (M0f, S4) was a fixed-alpha = 1 run.
