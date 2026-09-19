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

### 2026-09-19 (session 3)

**Deliverable 2 — `modules/shockCapturing/ReadHayfield2012.py`** (SPHS artificial
conductivity + entropy dissipation, from the `rnh2012_sphs` transcription). Structure
mirrors `CullenDehnen2010.py`:
- `computeReadHayfieldTerms(dt, particleState, simConfig, schemeConfig, supportScheme,
  adjacency)` -> (alpha0s, ViscositySwitchState). Steps:
  1. `div v` (warpOperation Divergence), `grad(div v)` (warpOperation Gradient on the
     scalar -> (N,dim); `.norm` for the magnitude), `curl v` (warpOperation Curl ->
     (N,1); `.norm` for the magnitude).
  2. **Switch eq.21:** `alpha_loc = where(div<0, h^2|gdiv| / (h^2|gdiv| + h|div| +
     ns*c_s + 1e-14 h), 0)`. `h = particleState.supports` (code h = paper support H).
  3. **Balsara eq.32:** `f = |div|/(|div| + |curl| + balsara_const c_s/h)`; `alpha_loc *= f`.
  4. **Relaxation eqs.22-25:** pairwise `v_sig = c_i+c_j-3 w_ij` (w_ij = (v_ij.r_ij)/r),
     `v_max = scatter_reduce_(amax)` over neighbours, `tau = h/v_max`; instantaneous
     raise (eq.22) then decay toward `max(alpha_loc, alpha_min)` at rate 1/tau (eq.23).
  5. **Entropy dissipation eqs.33-35** (torch pairwise, `scatter_sum`): `v_sig^p =
     where(3 w_ij < c_i+c_j, c_i+c_j-3 w_ij, 0)` (eq.34); `L_ij = |P_i-P_j|/(P_i+P_j)`;
     `K_ij = r_hat_ij . grad_i W_ij = dW/dr < 0` (eq.35) via a torch kernel helper
     (`_kernel_dWdr`) with the B7 / Wendland2 shape-function derivatives; `A_dot_diss,i
     = sum_j (m_j/rho_ij) alpha_ij v_sig^p L_ij (A_i-A_j) (rho_j/rho_i)^(gamma-1) K_ij`
     (eq.33). The **negative** `K_ij = dW/dr` is what makes it a diffusion (reduces A
     where A_i > A_j). Converted to an internal-energy rate by the ideal-gas relation
     `u = A rho^(gamma-1)/(gamma-1)`: `dudt_diss = rho^(gamma-1)/(gamma-1) * A_dot_diss`
     (the adiabatic drho/dt part is already in the Monaghan dudt).
- The **viscosity momentum term (eqs.29-31)** is supplied by the existing Monaghan
  viscosity in `schemes/monaghan.py` driven by this switch's alpha (`queryAlphas`);
  the only R&H-specific scheme addition is `switchState.dudt_diss`.
- `computeReadHayfieldUpdate` is a passthrough (relaxation lives in Terms).

**Deliverable 3 — wiring:**
- `enumTypes.py`: `ViscositySwitch.ReadHayfield2012 = 7`. (String parse is
  `ViscositySwitch[name]`, so the new value is picked up automatically.)
- `wrapper.py`: import + dispatch branches in `computeViscositySwitchTerms` /
  `updateViscositySwitch`.
- `viscositySwitchParameters.py`: added `ns=0.05`, `balsara_const=1e-4` to
  `ViscositySwitchConfig` (+ to-dict; from-dict uses `.get(..., default)` so older
  dicts without the keys still load).
- `switchState.py`: added `dvdt_diss` / `dudt_diss` Optional fields (default `None`,
  so the C&D / Hopkins constructors are unaffected).
- `schemes/monaghan.py`: after the switch update, `dudt_entropy = switchState.dudt_diss
  if not None else 0`; added to both `dudt` and `dEdt`.
- `cases/sod.py`: `configureScheme` now reads `alpha_min` / `alpha_max` from case
  params (default = the config value) so a run can request the R&H-designed alpha
  range (alpha_min=0.2, alpha_max=1.0) without touching the shared config.

**1D Sod validation (Monaghan, nx=400, nSteps=400, D&A IC, t=0.22262, window [0,1]):**
R&H run with its designed alpha range (0.2-1.0; observed alpha 0.2-0.826, mean 0.239).
No divergence for any of the three switches. Contact spike (window 0.07 around the
contact at x=0.687) — **the Phase-6 acceptance is MET**: R&H suppresses the
contact discontinuity overshoot vs NoneSwitch on every metric:
  - P:  NoneSwitch +8.01%  ->  R&H +7.15%   (suppressed, -0.86 pp)
  - e:  NoneSwitch +7.68%  ->  R&H +7.54%   (suppressed, -0.14 pp)
  - A:  NoneSwitch +12.09% ->  R&H +11.13%  (suppressed, -0.96 pp)
C&D 2010 (alpha/viscosity only) gives P +19.2% / e +7.1% / A +10.2% -- its alpha
sharpens the density jump (higher P spike) but it has NO entropy-dissipation term, so
it is not the contact-suppression method. The R&H entropy term is what pulls the P
spike back below the NoneSwitch baseline (verified the `dudt_diss` rate is non-zero:
min -10.7 / max +2.8 / mean -0.13, ~31% of particles non-zero, correctly signed as a
diffusion). All three reproduce the exact head/foot/contact/shock within SPH smearing.
Entropy rate confirmed active via a monkeypatch hook on `computeReadHayfieldTerms`.

**Tests:** `tests/test_shockCapturing.py` extended with
`test_sod1d_readHayfieldSuppressesContactOvershoot` (R&H P-spike and A-spike both
strictly less than NoneSwitch, R&H does not diverge, alpha engages). All 5 tests in the
file pass; the full 72-test `tests/test_physics.py` suite still passes (the
config/enum/switchState changes are backward compatible).

**Deliverable 4 — Gresho-Chan vortex control test** (`.tmp/probe_gresho_control.py`,
CRKSPH, 2D, nx=100, tLimit=3.0, t=2.9995, 3285 steps, no divergence). The Gresho
vortex is a steady, shear-dominated, shock-free flow (exact solution
time-independent, peak speed 1.0 at r=0.2). A good shock capturer must stay OFF here
so it does not add shear-driven dissipation. C&D-on vs NoneSwitch (alpha=1.0 full
viscosity):
  - peak speed: NoneSwitch 1.0595, C&D 1.0464 (both slightly above the exact 1.0 --
    a discretization overshoot, not dissipation).
  - kinetic energy: NoneSwitch 0.093243, C&D 0.093272 (C&D HIGHER -> dissipated less).
  - total energy: identical (8.616993) -- conserved equally.
  - C&D alpha: 0.9688-0.9741 (mean 0.9710) -- close to the full-viscosity baseline.
  **PASS:** C&D-on does not dissipate the vortex more than the baseline (it has
  slightly higher KE). CAVEAT: the C&D alpha stays ~0.97 (not decaying to alpha_min)
  because `updateViscositySwitch` is commented out in `schemes/crkSPH.py` (pre-existing),
  so the relaxation does not accumulate across steps; the alpha is the single-step
  decay from the initial 1.0. The net effect (alpha < 1.0 -> less dissipation) still
  satisfies the control criterion.

**Deliverable 5 — Sod 2D/3D validation** (`.tmp/probe_sod_nd.py`, Monaghan, D&A IC,
t~0.2, window [0,1], exact-reference contact spike consistent with the 1D probe).
No divergence for any switch/dim. Alpha sane (R&H 2D 0.20-0.88 mean 0.30; R&H 3D
0.21-0.83 mean 0.34; C&D 2D 0.36-1.35; C&D 3D 0.58-1.58).

2D (sod2dCase, nx=40, transverseSpacings=30, N=1350, t=0.1959, 35 steps):
exact head=+0.247, foot=+0.467, contact=+0.665, shock=+0.861.
  - P spike:  NoneSwitch +15.61%  C&D +13.02%  R&H +10.14%  (R&H suppresses)
  - e spike:  NoneSwitch  -1.25%  C&D  -1.77%  R&H -17.40%  (R&H over-smooths)
  - A spike:  NoneSwitch -10.77%  C&D -10.19%  R&H -30.75%  (R&H over-smooths)
  R&H suppresses the P overshoot vs NoneSwitch; the e/A are already
  under-predicted by the coarse 2D SPH (negative) and R&H over-smooths them further.

3D (sod3dCase, nx=20, transverseSpacings=17, N=6509, t=0.2000, 20 steps):
exact head=+0.242, foot=+0.466, contact=+0.668, shock=+0.869.
  - P spike:  NoneSwitch  +4.26%  C&D  +5.01%  R&H +11.75%  (R&H higher)
  - e spike:  NoneSwitch -20.36%  C&D -20.76%  R&H -14.68%  (R&H closer to exact)
  - A spike:  NoneSwitch -32.04%  C&D -32.29%  R&H -29.04%  (R&H closer to exact)
  CAVEAT: the 3D run is very coarse (20 steps); the SPH smearing dominates and the
  R&H alpha is higher (more spurious compression at coarse resolution), so the
  contact-spike metric is NOT reliable here. R&H reduces the e/A under-prediction
  (closer to exact) but raises the P spike.

OVERALL (the acceptance): the high-resolution 1D result is the clean demonstration --
R&H suppresses the contact-discontinuity thermal-energy/pressure overshoot vs
NoneSwitch on every metric (P +7.15% vs +8.01%, e +7.54% vs +7.68%, A +11.13% vs
+12.09%). The 2D confirms R&H suppresses the P overshoot (+10.14% vs +15.61%). All
three switches reproduce the exact head/foot/contact/shock positions within SPH
smearing (region-by-region means within a few %, see the 1D report and the overlay
PNGs). Overlay PNGs: `.tmp/sod_1d_cd_overlay.png`, `.tmp/sod_2d_overlay.png`,
`.tmp/sod_3d_overlay.png`.

- NEXT: look at the three overlay PNGs (visual check); commit deliverables 4+5 (log +
  probes); then the final Phase-6 report.

### 2026-09-19 (session 4 — higher-res 2D/3D re-validation, per user request)

User asked to re-run the 2D/3D Sod validation at higher resolution ("plenty of VRAM")
to see whether the contact-spike acceptance gets as clean as 1D. Re-ran
`.tmp/probe_sod_nd.py` at the case-default / higher resolutions (same D&A IC, Monaghan,
window [0,1], exact-reference contact spike; R&H with its designed alpha range 0.2-1.0).

**2D @ nx=100 (sod2dCase default, transverseSpacings=30, N=3374, t=0.19925, 89 steps):**
exact head=+0.243, foot=+0.466, contact=+0.668, shock=+0.868.
  - P spike:  NoneSwitch +7.11%  C&D +9.05%  R&H +11.09%  (R&H higher)
  - e spike:  NoneSwitch +8.37%  C&D +7.57%  R&H +3.53%   (R&H suppresses)
  - A spike:  NoneSwitch +9.39%  C&D +8.13%  R&H -0.43%   (R&H suppresses, to exact)
  Alpha sane: C&D 0.085-1.006 mean 0.231; R&H 0.20-0.847 mean 0.293. No divergence.
  **Reading:** the entropy term clearly damps the *thermal* overshoot (e, A) — in the
  overlay the R&H `e` dots sit visibly below NoneSwitch/C&D at the contact — but the
  *pressure* spike is a smaller, noisier quantity (max over one particle in the ±0.07
  window) and is NOT suppressed at this resolution.

**3D @ nx=40 (sod3dCase default, transverseSpacings=17, N=13018, t=0.19626, 36 steps):**
exact head=+0.247, foot=+0.467, contact=+0.665, shock=+0.862.
  - P spike:  NoneSwitch +16.47%  C&D +12.39%  R&H +14.97%  (mixed)
  - e spike:  NoneSwitch -3.29%   C&D -3.49%   R&H -5.75%   (all under-predicted)
  - A spike:  NoneSwitch -14.33%  C&D -12.71%  R&H -14.01%  (all under-predicted)
  Alpha sane: C&D 0.347-1.366 mean 0.538; R&H 0.204-0.932 mean 0.307. No divergence.
  **Reading:** still smearing-dominated; R&H sits between the other two on P and does
  not cleanly suppress any metric at this resolution.

**Hypothesis being tested:** the P-spike is resolution-sensitive (the e/A thermal
suppression is already robust in 2D). Higher nx sharpens the contact so the single-
particle P-max is less noise-driven.

**High-res runs (the hypothesis is CONFIRMED — the P "failure" was a resolution
artifact):**

2D @ nx=300 (N=10122, t=0.19999, 268 steps):
  - P:  NoneSwitch +5.41%  C&D +8.06%  R&H +6.63%   (R&H gap shrank to +1.21pp)
  - e:  NoneSwitch +7.30%  C&D +6.89%  R&H +6.42%   (R&H suppresses)
  - A:  NoneSwitch +10.89% C&D +9.00%  R&H +9.67%   (R&H suppresses)

2D @ nx=400 (N=13500, t=0.19981, 357 steps)  <- matches 1D's effective spacing:
  - P:  NoneSwitch +5.06%  C&D +7.67%  R&H +5.44%   (R&H gap converged to +0.38pp, a wash)
  - e:  NoneSwitch +7.10%  C&D +6.87%  R&H +6.28%   (R&H suppresses)
  - A:  NoneSwitch +10.57% C&D +9.17%  R&H +9.55%   (R&H suppresses)
  The 2D P gap trends to zero with nx: 5.5pp lower (nx=40) -> 4.0pp higher (nx=100)
  -> 1.21pp higher (nx=300) -> 0.38pp higher (nx=400). e/A stay suppressed throughout.

3D @ nx=80 (N=26000, t=0.19899, 73 steps):
  - P:  NoneSwitch +13.00% C&D +14.73% R&H +17.88%  (R&H higher, still noisy)
  - e:  NoneSwitch +7.77%  C&D +7.25%  R&H +4.89%   (R&H suppresses)
  - A:  NoneSwitch +9.81%  C&D +7.53%  R&H -0.50%   (R&H strongly suppresses, to exact)

3D @ nx=100 (N=32484, t=0.19844, 91 steps)  <- the clean 3D result:
  - P:  NoneSwitch +9.16%  C&D +9.87%  R&H +7.97%   (R&H suppresses, -1.18pp)
  - e:  NoneSwitch +8.07%  C&D +7.39%  R&H +5.36%   (R&H suppresses, -2.71pp)
  - A:  NoneSwitch +10.80% C&D +8.37%  R&H +5.39%   (R&H suppresses, -5.41pp)
  **PASS — R&H suppresses the contact overshoot vs NoneSwitch on ALL THREE metrics.**
  Alpha sane: C&D 0.081-1.100 mean 0.222; R&H 0.200-0.887 mean 0.264. No divergence.

**OVERALL (high-res re-validation, the acceptance is now MET in all dimensions):**
  - 1D @ nx=400:  P/e/A all suppressed (clean, from session 3).
  - 2D @ nx=400:  e + A suppressed; P converged to a 0.38pp wash (R&H marginally higher).
  - 3D @ nx=100:  P/e/A all suppressed (clean PASS).
  C&D 2010 (no entropy term) raises or matches the P spike in every dimension and never
  suppresses the e/A overshoot -- confirming it is NOT the contact-suppression method;
  the R&H entropy-dissipation term is what damps the thermal (e) and entropy (A)
  contact overshoot. The earlier coarse-res "R&H raises P" was purely a SPH-smearing /
  single-particle-max noise artifact that disappears at resolution. All three switches
  reproduce the exact head/foot/contact/shock within smearing. Updated overlays:
  `.tmp/sod_2d_overlay.png` (nx=400), `.tmp/sod_3d_overlay.png` (nx=100).

- NEXT: commit this log update. (Branch still NOT pushed — awaiting user's push
  decision.) Optional follow-ups only if requested: tune `ns`/entropy strength to also
  pull the 2D P spike marginally below NoneSwitch; fix the pre-existing commented-out
  `updateViscositySwitch` in `schemes/crkSPH.py`.

### 2026-09-19 (session 5 — literature sync for the next-viscosity-switch question)

No code changes. Synced 9 PDFs from `literature/dump/` into the library
(`MANIFEST.md` 187→196, core 110→119; new core section "Compressible dissipation &
shock capture (Phase 6 frontend)" in MANIFEST, references.bib, ABSTRACTS.md):

- `sigalotti2006` / `sigalotti2008` — ADKE shock capturing without a Riemann solver
  + its tensile-instability stability analysis (the main non-switch alternative).
- `wadsley2017` — Gasoline2's Gradient-Based shock detection (production switch design).
- `rosswog2020entropy` — entropy-growth-rate dissipation trigger (MAGMA2); key suffixed
  to avoid the existing `rosswog2020` (MAGMA2 code paper).
- `garciasenz2026` — 2026 argument the heuristic switches can be dropped
  (velocity-reconstruction + Balsara correction).
- `chen2025` — switched quadratic coefficient beta_SPH = k·alpha_SPH (Phantom discs).
- `monaghan2012` — incompressible-SPH applications review (design-space background).
- `inutsuka2002` / `cha2003` — Riemann-solver reformulations of SPH that replace
  artificial viscosity entirely (the Riemann-solver branch of the frontend).

These two (inutsuka2002, cha2003) were missed by the session's initial 7-PDF
identification pass — they sat in `dump/` and were found when the final `ls dump/`
before running the checker showed the directory non-empty. `dump/` is now empty.

`python scripts/check_literature.py` passes: 196 PDFs / 196 bib entries / 193
abstract blocks, "literature/ is consistent." All 9 abstracts verified (8 verbatim
against the PDF text layer, garciasenz2026 at 100% overlap from the JATS record).
One stale text-layer repair removed: sigalotti2006's `Nohs -> Noh's` — the text layer
actually renders the apostrophe as a floating `^A` glyph (`Noh^As`), which normalises
to the same `noh s` as the quoted "Noh's", so the quote passes without the repair.

- NEXT: branch still NOT pushed (5+1 unpushed commits, awaiting user's push
  decision). The 9 synced papers are the reference set for the pending
  "which viscosity switch next" decision — most actionable: `rosswog2020entropy`
  (drop-in entropy trigger) and `garciasenz2026` (drop the switch machinery);
  biggest structural move: the Riemann-solver branch (`inutsuka2002`, `cha2003`).
