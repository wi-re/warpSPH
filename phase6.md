
## Phase 6 — frontend build-out in `warpSPH` (C&D 2010 + R&H 2012)

Prerequisite for Phase 7, valuable on its own. The frontend's
`src/warpSPH/modules/shockCapturing/CullenDehnen2010.py` already exists
(384 lines, wired into the compressible schemes) but carries an open
"the signs here should have been wrong, double check!" note and
commented-out alternate formulations from prior experimentation. The
compressible scheme itself is Monaghan-style with rudimentary switch
support — now is the time to build this properly.

**Approach (user-directed, 2026-09-19):** run this from within `warpSPH`
(no `warpSPHCore` code changes; the reference YAML there is read-only). The
self-contained task brief is `phase6_shock_capturing_prompt.md` (copy into a
fresh session in `warpSPH`). Key refinements:
- The C&D 2010 switch is a BASIC-COMPRESSIBLE (Monaghan) construct, NOT a
  CRKSPH one. In code, `schemes/monaghan.py` does NOT currently call the switch
  (only `compSPH.py` / `crkSPH.py` do, via `computeViscositySwitchTerms` →
  `queryAlphas` → `updateViscositySwitch`). FIRST wire it into `monaghan.py`
  (mirror the compSPH.py pattern) and validate it THERE — CRKSPH is the common
  default but is not the switch's physical target.
- Debug the C&D switch in **1D and 2D first** (cheap iteration) before any 3D
  work.
- The Sod exact-solution reference solver (`sodSolution.solve`) accepts
  ARBITRARY left/right Riemann states, so the D&A IC (the R&H 2012 problem) is
  just a parameter choice — no special-casing.
- Keep a resumable work log (`phase6_shock_capturing_log.md` in `warpSPH`) at
  every step — sessions can crash mid-task on a context limit.

- [x] Validate `CullenDehnen2010.py` equation-by-equation against Cullen
      & Dehnen 2010 (local: `warpSPH/literature/cullen2010_inviscid-sph.pdf`):
      resolve the sign question (the R/Ξ limiter of their eqs. 17–18),
      the shear target-α and the l = 0.05 decay integration; remove or
      demote the dead alternate formulations.
- [x] Add a Read & Hayfield (2012) artificial-conductivity module
      (paper ingested in Phase 0 if obtainable; D&A's §4.3 description as
      interim spec) — its job in the paper: suppress the thermal-energy
      overshoot at the contact discontinuity (accept the over-smoothing
      of e and ρ, the paper says so explicitly).
- [x] Wire both into the compressible scheme as first-class, selectable
      options (switch on/off, per the paper's usage: C&D switch always on,
      conductivity on for the shock test); keep the Monaghan-style path
      as baseline.
- [x] Control test (cheap, 2D): the existing `greshoVortex` case
      (CRKSPH, steady exact solution) with and without the switch —
      the switch must remove the shear-driven artificial dissipation
      (D&A intro / Springel 2010 argument; the paper's Fig. 10 cubic
      spline matches the no-viscosity case precisely because of the
      switch).

Acceptance: C&D module matches the paper's equations (sign question
resolved and documented); R&H module in place; control test passes; a
short smoke Sod run shows the contact overshoot suppressed.

**Status: COMPLETE (2026-09-19).** All four items done; acceptance met. C&D
sign question resolved (B11 `D(div v)/Dt = div(dv/dt) − tr(V²)`, minus sign;
dead `tr(A+V²)` branch removed). R&H SPHS module added
(`modules/shockCapturing/ReadHayfield2012.py`) with the entropy-dissipation
term. Gresho control passes (C&D adds no shear-driven dissipation). Sod
validation 1D/2D/3D: at high res (1D nx=400, 2D nx=400, 3D nx=100) R&H
suppresses the contact-discontinuity thermal-energy/entropy overshoot vs
NoneSwitch on every metric; all three switches reproduce the exact
head/foot/contact/shock within SPH smearing. Commits: `dcd0423` (C&D +
tests), `652ed28` (R&H + wiring), `3cf40f5`/`128d82b` (log). Full record in
`phase6_shock_capturing_log.md`. Branch not yet pushed.
