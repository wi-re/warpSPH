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

### 2026-09-19 (session 2)
- **SIGN QUESTION RESOLVED (C&D App. B, eq. B11).** Let `V_ij = dv_j/dx_i` and
  `D/Dt = d_t + v_k d_k`. Then
  `D V_ij/Dt = d_i(d_t v_j) + v_k d_k d_i v_j = d_i(d_t v_j + v_k d_k v_j) - (d_i v_k)(d_k v_j)
  = d_i a_j - V_ik V_kj = A_ij - (V^2)_ij`, so `Vdot = A - V^2` (eq. B11).
  Taking the trace: `D(div v)/Dt = div(dv/dt) - tr(V^2)`. Cross-checked with the
  identity `div(Dv/Dt) = D(div v)/Dt + tr(V^2)`. Therefore the ACTIVE path
  `divdotdvdt - tr(V^2)` (MINUS) is correct; the dead `tr(A + V^2)` (PLUS) branch
  was WRONG. Derivation now documented in the `CullenDehnen2010.py` module docstring.
- **C&D 2010 equation check (code vs paper):** `computeR` -> R_i = (1/rho_i) sum_j
  sign(div v_j) m_j W_ij = eq.17 (INTERPOLATE of dens*sign(div)). `computeXi` ->
  xi_i = |2(1-R_i)^4 div v_i|^2 / (|2(1-R_i)^4 div v_i|^2 + tr(S_i S_i^t)) = eq.18
  (beta_xi=2.0; trace=0 when S is None, i.e. the plain C&D path uses the simplified
  limiter with tr(S^2)~0, so xi~1 in non-shear flow). `computeCullenUpdate` ->
  alpha_loc = alpha_max h^2 A/(v_sig^2 + h^2 A) = eq.14 with A_i = xi_i max(-divdot,0)
  = eq.13; v_sig = max(cbar_ij - min(0, v_ij.xhat_ij)) = eq.15; relaxation
  alpha_dot = (alpha_loc - alpha)/tau, tau = h/(2 l v_sig), l=0.05 = eq.16
  (instant-up then exponential decay). All match.
- **Wired the switch into `schemes/monaghan.py`** (mirror of compSPH): after the
  gradHState block, `currentState.alphas, switchState = computeViscositySwitchTerms(
  dt, currentState, config, schemeConfig, SupportScheme.SuperSymmetric, adjacency)`;
  `queryAlphas = currentState.alphas` added to computeViscosity / computeConductivity /
  computeThermalDissipation; after dudt_thermal, `currentState.alpha0s, switchState =
  updateViscositySwitch(switchState, dt, dvdt + dvdt_diss, ...)` and
  `currentState.divergence = -drhodt/densities`. `dEdt` unchanged (Monaghan is
  deliberately non-energy-conserving; dudt_thermal not in dEdt -- pre-existing).
- **NoneSwitch = verified no-op baseline** (alphas stay 1.0); 10 compressible-solver
  tests in tests/test_physics.py pass.
- **1D Sod debug (`.tmp/probe_cullen_sod.py`, Monaghan, nx=400, nSteps=400, D&A IC,
  t=0.22262):** C&D alpha field is physically sane -- off ~0.02 in smooth regions,
  up to ~0.95 at the shock, mean ~0.08, no divergence.
- **KEY GEOMETRY FIX for the exact-solution overlay.** `buildSod1D` (and `buildSodND`)
  lay out a MIRROR-SYMMETRIC two-interface tube: dense (left) state fills the middle
  block |x| <= L/4, light (right) state the two outer quarters, interfaces at x=+/-L/4.
  The mirror symmetry makes x=0 and x=+/-L/2 act as reflecting walls, so the analytic
  Riemann solution describes ONLY the window x in [0, L/2] = [0,1] (L=2) with the
  interface at x=+0.5 -- exactly `sodUtil.plotSod_`'s call
  `solve(left, right, geometry=(0., 1., 0.5), t, gamma)`. The SPH profile must be
  binned on x in [0,1]. (First probe wrongly used geometry (-0.5,0.5,0) over the whole
  domain -> symmetric-looking profile, wrong overlay.)
- **1D Sod region-by-region (SPH mean vs exact, window [0,1], t=0.22262):** exact
  head=+0.213, foot=+0.462, contact=+0.687, shock=+0.911. Both methods reproduce all
  constant regions within a few % (R3 rho +0.7%, R4 rho -3.1%, P +1%). C&D preserves
  the smooth/undisturbed R5 BETTER than NoneSwitch (rho +0.2% vs +1.5%, P +0.3% vs
  +2.8%). Contact spike (window ~0.07 around contact): NoneSwitch P +8.0% / A +12.1%;
  C&D 2010 P +19.2% / A +10.2%. C&D's higher P-spike is the expected signature of
  cutting smooth-region viscosity (sharpens the density jump); it is NOT the method
  built for contact suppression -- that is R&H 2012 (entropy-dissipation term), which
  is the deliverable-5 acceptance (R&H must suppress the spike vs NoneSwitch).
  Overlay PNG -> `.tmp/sod_1d_cd_overlay.png`.
- **buildSod1D fixed-dt caveat:** it calls `computeTimestep(..., dt=None)` which
  REQUIRES adaptiveDt=True; a fixed dt cannot be forced through the sod case. So the
  probe runs nSteps steps and evaluates the exact solution at the run's actual final t.
- NEXT: (a) 2D C&D sanity run (alpha sane, no divergence); (b) write
  tests/test_shockCapturing.py (equation-level unit checks for R/Xi/target-alpha/decay
  on a manufactured state + 1D Sod contact-spike assertion); (c) commit deliverable 1;
  (d) deliverable 2: ReadHayfield2012.py (SPHS); (e) deliverable 3: enum + wrapper +
  config params; (f) gresho control; (g) Sod 1D/2D/3D validation + PNGs.
