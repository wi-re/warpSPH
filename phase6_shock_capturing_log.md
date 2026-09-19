# Phase 6 Log — compressible shock-capturing frontend (C&D switch on Monaghan + R&H 2012 SPHS)

Newest entries at the BOTTOM. This is the resumability log — re-read it at the top of
any resumed session, then check `git log` for what is already committed.

## Task summary (from the phase brief)
1. Wire Cullen & Dehnen (2010) viscosity switch into `schemes/monaghan.py`
   (mirror the compSPH.py call pattern); validate equation-by-equation against
   cullen2010; RESOLVE the sign question in `computeSecondOrderV`; remove dead code.
2. Add `modules/shockCapturing/ReadHayfield2012.py` (SPHS artificial conductivity,
   per the `rnh2012_sphs` transcription in the sibling reference yaml).
3. Wire both in as first-class options: `ViscositySwitch` enum, `wrapper.py`
   dispatch, `ViscositySwitchConfig` params (+to-dict/from-dict).
4. Control test: `cases/greshoVortex.py` with/without C&D switch.
5. Sod validation 1D -> 2D -> 3D: NoneSwitch / CullenDehnen2010 / ReadHayfield2012,
   D&A Sod IC, exact-solution overlay at t=0.2.
Validation harness: `tests/test_shockCapturing.py` (equation-level unit checks +
1D Sod contact-overshoot metric). Overlay PNGs saved for user visual check.
Do NOT push. Commit per deliverable.

## Key code facts established (2026-09-19)
- `schemes/monaghan.py::compressibleSPH_Monaghan` does NOT call the switch yet;
  uses `computeViscosity` from `modules/dissipation` (accepts per-particle alphas
  via `queryAlphas`/`referenceAlphas`).
- `schemes/compSPH.py` wiring pattern (MIRROR THIS):
  1. `currentState.alphas, switchState = computeViscositySwitchTerms(dt,
     currentState, config, schemeConfig, SupportScheme.SuperSymmetric, adjacency)`
  2. pass `queryAlphas = currentState.alphas` into accel/dudt
  3. `currentState.alpha0s, switchState = updateViscositySwitch(switchState, dt,
     dvdt, currentState, config, schemeConfig, SupportScheme.SuperSymmetric, adjacency)`
- `schemes/crkSPH.py` calls `computeViscositySwitchTerms` at line ~226 (alive), but
  its `updateViscositySwitch` call is COMMENTED OUT (lines ~339) — CRKSPH is the
  default scheme; investigate whether vestigial (task says treat Monaghan as the
  authoritative validation target).
- `modules/shockCapturing/CullenDehnen2010.py`: `computeSecondOrderV` has the
  sign-question comment; active path returns `divdotdvdt - tr(V^2)`; DEAD
  alternative `return tr(A + V^2)` after the first return (unreachable code).
  `computeCullenTerms` computes `ddiv_dt = (div - particleState.divergence)/dt`
  (NOT `computeSecondOrderV`) — i.e. the C&D path in computeCullenTerms does not
  use computeSecondOrderV at all; only `computeHopkinsUpdate` calls it.
- `modules/shockCapturing/wrapper.py`: dispatch on `schemeConfig.viscositySwitchParams.scheme`;
  NoneSwitch passes through `particleState.alphas` / `alpha0s`.
- `enumTypes.ViscositySwitch` = {Balsara1995=0, Colagrossi2004=1, CullenDehnen2010=2,
  CullenHopkins=3, MorrisMonaghan1997=4, Rosswog2000=5, NoneSwitch=6}.
- `ViscositySwitchConfig` has `limitXi` declared TWICE (dataclass: 2nd, default True,
  is live). New R&H params (ns, balsara_const, alpha bounds) need adding here.
- `CompressibleState` (systems/compressibleMonaghan.py) carries `divergence`,
  `alpha0s`, `alphas` (all `constant`, default None). `CompressibleSystem.finalize`
  does UNCONDITIONAL `copy_` of divergence/alpha0s/alphas from lastState —
  NEEDS CHECK: NoneSwitch Monaghan run may crash in finalize if those stay None.
- Builder: `CompressibleSPHScheme.Monaghan` -> `compressibleSPH_Monaghan`;
  `cases/sod.py` defaults `scheme='CompSPH'` (so the 1D Sod case as-is does NOT
  exercise the Monaghan step).
- `sodSolution.solve(left_state, right_state, geometry, t, gamma, npts)` accepts
  arbitrary Riemann states; D&A IC = L(p=1,rho=1,u=0) R(p=0.1,rho=0.125,u=0),
  gamma=5/3, t=0.2.
- Reference yaml: `rnh2012_sphs` transcription + `sod` IC (problem B) +
  `sod_setup_phase6`: "domain [-0.5,0.5], 3D, 32x32x400 + 16x16x200 lattice,
  N_1D=600, t=0.2, initial alpha=1 on -0.05<x<0.05".
- Gresho control: `cases/greshoVortex.py`, scheme CRKSPH, 2D, nx=200, tLimit=3.0.

## Log

### 2026-09-19 (session 1)
- Created this log. Read: monaghan.py, compSPH.py, crkSPH.py (switch call sites),
  CullenDehnen2010.py, CullenHopkins.py, wrapper.py, switchState.py, common.py,
  enumTypes.py, viscositySwitchParameters.py, sodSolution.py, cases/sod.py,
  cases/greshoVortex.py, dissipation modules (pi.py, wp_diffusion.py,
  wp_dissipation.py), compressibleMonaghan.py, builder.py, da2012_reference.yaml.
- NEXT: (a) read C&D 2010 paper PDF (sign question, Appendix B eq. B11) and R&H
  2012 PDF; (b) check integrator finalize + state init for None fields;
  (c) check buildSod1D; (d) wire switch into monaghan.py; (e) unit-level checks.
