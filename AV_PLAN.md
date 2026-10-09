# warpSPH — Artificial-Dissipation Roadmap (AV_PLAN)

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

Modernising the compressible dissipation stack: from the one hard-wired
Monaghan-plus-switch path that exists today to a **numerical-method laboratory**
where shock detector, coefficient policy, velocity-pair policy and pair operator
are independently replaceable — and then handing that substrate to
[`PESPH_PLAN.md`](PESPH_PLAN.md) and, eventually, to Godunov-SPH.

Target papers, all local in `literature/` (bib keys as in `ABSTRACTS.md`):

| Key | Paper | Phase |
|---|---|---|
| `rosswog2020entropy` | Rosswog (2020), *A simple, entropy-based dissipation trigger for SPH*, ApJ 898, 60 | **2** (trigger), **3** (reconstruction) |
| `garciasenz2026` | García-Senz & Cabezón (2026), *Are heuristic switches necessary…?*, A&A 708, A205 | **4** |
| `chen2025` | Chen & Nixon (2025), *Minimizing the numerical viscosity… in discs*, MNRAS 540, 2465 | **5A** |
| `borrow2022` | Borrow et al. (2022), *Sphenix*, MNRAS 511, 2367 | **5B** |
| `wadsley2017` | Wadsley, Keller & Quinn (2017), *Gasoline2*, MNRAS 471, 2357 | **6** |
| `cullen2010` | Cullen & Dehnen (2010), *Inviscid SPH*, MNRAS 408, 669 | baseline (done) |
| `read2012` | Read & Hayfield (2012), *SPHS*, MNRAS 422, 3037 | baseline (done) |
| `frontiere2017` | Frontiere, Raskin & Owen (2017), *CRKSPH*, JCP 332, 160 | **3** (the SLR this repo already has) |
| `hopkins2013` | Hopkins (2013), *A general class of Lagrangian SPH*, MNRAS 428, 2840 | **8-9** → PESPH_PLAN |

Full reading checklist in [Part 5](#part-5--literature).

---

# Status board — read this first

**Status 2026-10-09: the bake-off ran (2026-10-09, 18 jobs rc 0, no row diverged; verdict and per-row notes in [`docs/av/bakeoff_2026-10-09/README.md`](docs/av/bakeoff_2026-10-09/README.md)). Phases 0-4, 5A, 5B (R1 read off) and 6 are done (a ✗ marker is a run whose result was a recorded finding, not undone work). Left for M8: the user's tag; then 7b's addendum rows and stage 2 (user decides).** Earlier status, 2026-10-08: Phases 0-4, 5A and 6 are done (a ✗ marker is a run whose result was a recorded finding, not undone work); 5B waits only on reading its ell_V 5 numbers off the bake-off; Phase 2 accepted (user, 2026-10-08); only the bake-off is left -- see the short list right below the table.** The plan's *input* is the work
logged in [`phase6.md`](docs/historic_plans/phase6.md) / [`phase6_shock_capturing_log.md`](docs/historic_plans/phase6_shock_capturing_log.md)
(Cullen & Dehnen 2010 and Read & Hayfield 2012 on [`schemes/monaghan.py`](src/warpSPH/schemes/monaghan.py),
validated on Sod and Gresho). On top of it, per the start-up order below:

- **S1 / M0** baseline instrument (`scripts/av_report.py`, 10 cases, `--compare`, bit-lock check) and the
  reference **M0c** (`results/av_M0c_*`, `docs/av/av_baseline_2026-09-30.md`, 50 pairs bit-locked);
- **S2** audit fixes + fixed-beta (`BetaMode`) + the switch registry; **S3** all 8 `ViscositySwitch` members
  implemented (Balsara, Colagrossi, Morris-Monaghan, Rosswog2000; the last now checked against its paper, 2026-10-01);
- **S5** the Monaghan viscous-heating factor 1/2 (OPEN_PROBLEMS §16);
- **S4** CRK `dudt.py` j-side sign: applied 2026-10-01 (see S4 below);
- **OPEN_PROBLEMS §17** (Read-Hayfield energy) resolved 2026-10-01: a transcription error in Eq. (33); baseline regenerated as **M0d**, then **M0e** after the R&H pair loops moved to warp kernels (see below).

Phase 2 (Rosswog 2020 entropy trigger) built and validated 2026-10-06 — passes shocks, sweeps and smooth flow; its Kelvin-Helmholtz failure (`docs/av/av_phase2_rosswog2020_2026-10-06.md`) was traced on 2026-10-07 to the sharp-IC contact transient plus resolution (nx 256: A(1.5) 0.131 vs C&D 0.137, [`docs/av/kh256_2026-10-07/`](docs/av/kh256_2026-10-07/README.md)). Phase 1 (2026-10-07) added the pair-velocity argument, the `pi` package, a Pi audit against the papers and the CRK switch fix (items 1-4 under Phase 1 below). Phases 3-6 were built 2026-10-07 and validated by the overnight sweep (2026-10-08, [`docs/av/sweep_2026-10-08/verdict.md`](docs/av/sweep_2026-10-08/verdict.md)); the markers were reconciled with it on 2026-10-08.

| Phase | Milestone | State |
|---|---|---|
| 0 | M0 Baseline locked | ✅ M0c (2026-09-30), tag `milestone/av-baseline` at the first M0; reference now M0g (Monaghan/CompSPH hosts) and `av_crk3_*` (CRK), see the handoff |
| 1 | M1 Old physics, new architecture | ✅ 2026-10-07: registry, `BetaMode`, tag fixes, stubs filled, `rawPairVelocity` + `computePi_pair`; smoke bit-identical to M0g, tests + gradcheck green. Tag `milestone/dissipation-abstraction` (4e662a1, local) |
| 2 | M2 Entropy trigger validated | ✅ accepted as is (user, 2026-10-08) with its two caveats; built 2026-10-06; KH explained and near-closed at nx 256 (0.131 vs C&D 0.137: gate `>= C&D` missed by 4.6 %, smooth IC); marginal Sod 2D/3D P spike (+1.8 % / +0.3 %) — `docs/av/av_phase2_rosswog2020_2026-10-06.md` |
| 3 | M3 Reconstruction engine works | ✅ 2026-10-08: unit tests (float64 annihilation, 1e-12 conservation), gradcheck green; sweep: Monaghan + Linear/Limited clean on 5 cases, CRK energy drift <= 3e-5 everywhere, Group A within 2e-4 of `av_crk3_*` |
| 4 | M4 Smooth-flow dissipation characterised | ✅ characterised 2026-10-08: Sod passes; Gresho ordering, KH 1.4x, AV-energy 10x headline not reproduced (recorded); Sedov overshoot untestable at nx 40; φ maps rendered |
| 5A | M5 Quadratic dissipation understood | ✅ 2026-10-08: Rosswog passes all, C&D Sod contact spike +15 % coupled; `C_q` decided (coupled, `C_q = 2`); mechanism reproduced, disc result not attempted (noted) |
| 5B | M6 Cheap modern switch characterised | ✅ 2026-10-09 (R1 read off the bake-off, `docs/av/bakeoff_2026-10-09/`): Sphenix ell_V 5 row 6 vs C&D: Sod 1D +7 %, Sod 2D/3D, Sedov, Noh within 5 % or better, Gresho 0.072 vs 0.078 (passes), KH 0.116 vs 0.076, 11 % cheaper. Earlier: swept 2026-10-08 at the paper's ell_V 0.05; default now ell_V 5 (decided); Gresho > C&D; speed-up 10-11 %, shortfall = the Balsara curl loop (16 % without it); left: R1 (from the bake-off) |
| 6 | M7 Detector-complete | ✅ swept 2026-10-08: uniform-compression claim reproduced, Gresho passes, five-detector maps rendered; Sod / Sedov shocks off by the kept prefactor 0.5 (decided trade-off); shear / sound tests pass; open finding: fires in uniform-pressure shear (Phase 6 note) |
| 7 | 🏁 **M8 SPH-AV-FOUNDATION** — hard gate | ✅ bake-off run 2026-10-09 (13 rows x 12 cases, CompSPH cross-check, KH nx 256, timing, maps; no divergence); verdict `docs/av/bakeoff_2026-10-09/README.md`, default (row 12) stands, gate misses listed there (Sedov / RT energy drift all rows, Noh 2D all but Wadsley; CompSPH Rosswog Sedov blow-up = OPEN_PROBLEMS §23). **Left: the M8 tag (user).** PESPH continues in its own plan |
| 7b | Riemann dissipation as an alternative AV (user, 2026-10-08) | ◐ 7b.1 solvers, 7b.2 `LimiterType`, 7b.3 first-stage Riemann Π term built, tested and validated, now on `dev` (uncommitted; bake-off smoke bit-identical, full suite green); left: addendum bake-off rows after the main run; stage 2 (MUSCL scalar states) re-scoped, user decides; see Phase 7b |
| 8+ | → [`PESPH_PLAN.md`](PESPH_PLAN.md) | blocked on M8 |

### Still to do before M8 (2026-10-08)

R2-R8 (the small items outside the bake-off) were done on 2026-10-08; results are in each phase's markers. Left:

| # | What | Closes |
|---|---|---|
| R1 | ~~Sphenix at the new default ell_V 5 on every case~~ done 2026-10-09 from bake-off row 6 (table in `docs/av/bakeoff_2026-10-09/README.md`) | 5B markers below |
| -- | ~~the bake-off~~ ran 2026-10-09 (4.5 h); **user:** the tag | M8 |

Done 2026-10-08: R2 Wadsley shear / sound-wave tests (pass; the shear box alpha is a uniform-pressure effect, Phase 6
note), R3 neighbour loops (Sphenix 1, C&D 3), R4 Sphenix cost (the Balsara curl loop is the whole shortfall: 16 %
without it), R5 φ maps, R6 shearing-Noh metric (correct; the offset is AV shear heating), R7 the chen2025 note, R8
CRK deltas (a stale pre-switch-fix reference in the verdict script, fixed). Decided 2026-10-08 (user): Phase 2's two
partials accepted as is (M2 ✅); the Wadsley uniform-pressure shear finding stays a recorded property, no remedy.

### Working order from Phase 3 on: build first, sweep once (user, 2026-10-07)

Phases 3-6 are implemented back to back. Each phase is gated only by its **cheap checks**, run right away:
unit tests, its gradcheck script, the bit-identical smoke `--compare` wherever the change is meant to be a
refactor, and a short no-divergence run of the new configs. The **expensive validation** of every phase (the
Validation tables: KH nx 256, full-profile Sod/Sedov/Gresho ladders, Table 1 rows, the 5A beta matrix,
Sphenix timing, Wadsley detector maps) goes into **one overnight sweep** (`scripts/av_sweep_overnight.py`, built
alongside), whose matrix is a superset of Phase 7's. Markers that need sweep results stay unticked until the
sweep has been read; each phase's build markers are ticked as they land. Decisions the sweep has to inform
(Phase 5A's `C_q = 0` default, Phase 7's default) wait for it.

### Overnight sweep results (2026-10-08) — read before deciding anything

`results/av_sweep_2026-10-07_17-30/` (all runs with video; evidence copied to [`docs/av/sweep_2026-10-08/`](docs/av/sweep_2026-10-08/verdict.md)):
**41 pass / 21 fail / 0 missing**; no run diverged except the cfl x 4 robustness configs (by design). Plus the KH nx 256
table (INFO). The FAILs, sorted by what they mean:

- **Not resolvable at this resolution (not a disagreement).** Phase 4 Sedov overshoot (AVSLR / AVSWSLR must exceed
  rho/rho0 = 4): every variant peaks at ~2.1-2.3 at 3D nx 40. Needs a much finer Sedov to test the paper's claim.
- **Real findings, Phase 4 (García-Senz).** KH ladder AV < AVSW < SLR reproduced (0.019 / 0.082 / 0.092-0.131), but
  only AVSWSLR reaches 1.4x AVSW; Gresho: SLR halves plain AV's error (0.21 -> 0.09) but does not beat the switch
  alone (0.079), AVSWSLR is best (0.058); **headline AV-energy reduction 2.1x, not >= 10x**; RT tie (0.379 vs 0.388).
  KH nx 256 (smooth IC, t = 3): AVSLR(B2) 0.138-0.139 at t = 1.5, = C&D, below CRK's 0.145; nothing on the
  Monaghan host keeps CRK's post-saturation amplitude (~0.12 vs 0.19).
- **Real findings, Phase 5B/6 (Sphenix / Wadsley shocks).** Sod L1(v_x) 2-3x C&D's (Sphenix 0.0137, Wadsley 0.0092,
  C&D 0.0052) with low shock alpha (0.65 / 0.42 vs 0.95); Sedov shock radius error larger (-4.1 % / -2.3 % vs
  -1.1 %); Wadsley KH / RT at nx 128 below C&D (0.081 vs 0.100; RT 0.317 vs 0.383). **Suspect first:** the
  step-boundary update (one-step alpha lag) these two share, and Wadsley's derived prefactor 0.5 -- check with the
  alpha maps and the Sod videos before tuning anything. Sphenix Gresho L1 0.090 vs C&D 0.073 (alpha lower, 0.016).
- **Phase 5B cost:** Sphenix 11 % cheaper per step than C&D (target 15 %); Wadsley the most expensive (+34 %).
  cfl x 4: Sphenix *and* C&D both diverge on Sod and Gresho, both survive Sedov and Noh -- no robustness advantage
  from the implicit decay at cfl 1.2.
- **Phase 5A mostly confirms Chen & Nixon.** Coupled beta cuts quadratic AV energy ~10x (Gresho), shear box
  quadratic fraction 41-78 % -> 5 %, shocks unchanged for Rosswog; C&D Sod contact spike +15 % coupled. The
  shearing-Noh front metric reads 0.27-0.35 vs exact 0.20 -- **the metric is not trusted yet** (check frames).
- **The positives.** Wadsley's central claim reproduced: cylindrical Noh pre-shock alpha 4e-8 vs C&D 0.72, post-shock
  density -1.3 % vs -6.5 % (its `alphaActiveFraction` 0.39 vs 0.78 misses the "< 1/2" bar narrowly). **Rosswog +
  limited reconstruction is the best KH combination**: nx 256 A(1.5) 0.151 (> McNally 0.148, > CRK 0.145), peak 0.224.
  Phase 3: everything clean; CRK 3D Sedov nx 40 with video now completes (OPEN_PROBLEMS §18 fix confirmed).

**Decided 2026-10-08 (user):** the Monaghan default is now Rosswog (2020) at alpha in [0, 1] + `C_q = 2` coupled +
the limited reconstruction (`av_report --config default`, bit-identical to `rosswogLimitedCoupled`; the M0-era
configs pin the old defaults and stay bit-identical); README states it as the reference configuration. Phases 3-6
committed one per phase. Next: the Sphenix / Wadsley shock under-dissipation. Under the new default operator the
unswitched Sod contact spike is 6 % and R&H's 10 % -- R&H's contact-suppression claim holds only on the old
`C_q = 0` raw operator (`tests/test_shockCapturing.py` pins that operator).

*Original list:* (1) Phase 5A `C_q = 0` default (the data says coupled beta is cheap in shocks and
removes smooth-flow quadratic dissipation; C_q = 0 leaves Noh 50 % off); (2) whether to chase the Sphenix / Wadsley
shock under-dissipation before Phase 7; (3) Phase 7 default -- the candidates on this evidence are Rosswog + limited
reconstruction (KH) and C&D-Q (shocks); (4) commits (all of Phases 3-6 is uncommitted on `dev`).

### Investigation: Sphenix / Wadsley shock under-dissipation (opened 2026-10-08, user)

Symptom (sweep): Sod L1(v_x) Sphenix 0.0137, Wadsley 0.0092 vs C&D 0.0052; the final frames show a noisy post-shock
plateau (v_x scatter 0.75-0.97, thermal energy / pressure noise) where C&D is flat; shock alpha 0.65 / 0.42 vs 0.95.
Suspects, cheapest first: (1) decay speed -- Sphenix's tau = 0.05 H / c vs C&D's ~10 H / v_sig, so alpha collapses
within a step or two behind the front; (2) Wadsley's derived prefactor 0.5; (3) the one-step alpha lag of the
step-boundary update. Probe: `scripts/probe_shockUnderdissipation.py` (Sod L1, post-shock plateau noise, alpha at
the shock; video). Results below as they come.

**Sod, 2026-10-08** (`results/probe_under/`; plateau noise = std of v_x over the post-shock plateau, relative):

| variant | L1(v_x) | plateau noise | alpha at shock |
|---|---|---|---|
| C&D (reference) | 0.0052 | 1.4 % | 0.95 |
| Sphenix as published (ell_V = 0.05) | 0.0137 | 7.7 % | 0.61 |
| Sphenix ell_V = 0.25 / 1 / 5 | 0.0109 / 0.0083 / **0.0050** | 5.8 / 3.5 / **1.1 %** | ~0.6 |
| Wadsley as derived (prefactor 0.5) | 0.0092 | 4.5 % | 0.42 |
| Wadsley prefactor 2 (the paper's literal number) | 0.0051 | 1.1 % | 1.08 |
| Wadsley 4x slower decay (wadsley_tau 0.05) | 0.0070 | 2.6 % | 0.42 |
| Wadsley prefactor 2 + slower decay | **0.0041** | **0.3 %** | 1.08 |

**Not porting bugs.** The rendered PDF confirms Sphenix's `tau_V = gamma_K ell_V h / c`, ell_V = 0.05, as implemented
(and that Eq. 21 is printed with the sign typo we corrected). Sphenix as published decays alpha within ~0.05 support
radii of sound travel, so the post-shock plateau keeps no damping; with a decay time comparable to C&D's (ell_V ~ 5)
it beats C&D on Sod at the *same* shock alpha. Wadsley's derived 0.5 reproduces Gasoline2's actual source strength
in this repo's h units, which is half of C&D's; the paper's literal 2 matches C&D. The one-step lag is not needed
to explain any of it. Both remedies are constant changes away from the papers, so they need checking on smooth
flow and other shocks before anyone adopts them (next: Sedov, Gresho, KH, cylindrical Noh).

**Across cases, 2026-10-08** (full profile, video; "published" = the sweep's numbers):

| variant | Sod L1 | Sedov radius err | Gresho L1(v_phi) | KH A(1.5), nx 128 | Noh 2D pre-shock alpha / post-shock rho err |
|---|---|---|---|---|---|
| C&D (reference) | 0.0052 | -1.1 % | 0.073 | 0.100 | 0.72 / -6.5 % |
| Sphenix published (ell_V 0.05) | 0.0137 | -4.1 % | 0.090 | 0.105 | -- |
| Sphenix ell_V 1 | 0.0083 | -1.4 % | 0.080 | 0.118 | -- |
| **Sphenix ell_V 5** | **0.0050** | **-0.3 %** | **0.077** | **0.121** | -- |
| Wadsley derived (prefactor 0.5) | 0.0092 | -2.3 % | **0.067** | 0.081 | 4e-8 / **-1.3 %** |
| Wadsley prefactor 2 | 0.0051 | -1.9 % | 0.075 | 0.095 | 1e-8 / -4.4 % |
| Wadsley prefactor 2 + slower decay | 0.0041 | -1.9 % | 0.082 | 0.099 | 1e-8 / -4.9 % |
| *default: Rosswog + limited + coupled* | *0.0043* | *-1.5 %* | *0.060* | *--* | -- |

**Readings.** Sphenix: a longer decay (ell_V 5, i.e. tau = 5 H / c, comparable to C&D's) is better on *every* case
here -- no trade-off found; the only cost is a lower, broader Sedov peak (1.85 vs 2.51 at an unresolved nx 40).
Wadsley: the paper's literal prefactor 2 fixes Sod and helps KH / Sedov but costs Gresho (0.067 -> 0.075) and the
cylindrical Noh post-shock density (-1.3 % -> -4.4 %); its uniform-compression blindness survives (pre-shock alpha
1e-8); a slower decay on top only costs more Gresho. A genuine trade-off, not a bug. The repo default (Rosswog +
limited reconstruction) does not ring at the Sod shock (alpha 1.0 there, L1 0.0043). **Decided 2026-10-08 (user):**
Sphenix `sphenix_ell` defaults to 5 (documented deviation from the paper's 0.05; `sphenix_ell = 0.05` restores the
paper's scheme); Wadsley stays at the derived prefactor 0.5. Investigation closed.

### Handoff: starting Phase 3 (2026-10-07)

**Where the code stands.** `dev` at the commit after `4e662a1` (tag `milestone/dissipation-abstraction`), working tree clean, full `pytest tests` and the dissipation / compSPH / CRK gradchecks green.
- The pair velocity is an argument: `computePi_term(term, ..., u_ij, ...)` / `computePi_pair` (formulation from the params); `computePi_actual(v_i, v_j)` is the thin `rawPairVelocity` wrapper. The Monaghan, CompSPH and conductivity kernels still call `computePi_actual`.
- CRKSPH reconstructs inline: `modules/crk/accel.py` and `dudt.py` compute `phi_ij` (`crkLimiter`, `computeVanLeer` from `modules/crk/limiter.py`), form `v_dot_i = v_i - phi/2 J_i x_ij`, `v_dot_j = v_j + phi/2 J_j x_ij` (the S4 sign question is settled: Eqs. 11-12) and hand `v_dot_i - v_dot_j` to `computeFrontiereQ`. That inline block exists twice (accel and dudt): Phase 3's extraction removes the duplication.
- `modules/reconstruction/` holds only `rawPairVelocity`. There is **no** `velocityPairPolicy` config field yet: add it with its first consumer (`DiffusionParameters` field, enum, dict round trip with `.get`, runner banner line, like `betaMode`).

**Suggested order.** (1) move `limiterVL` / `computeVanLeer` / `crkLimiter` to `modules/reconstruction/limiters.py` verbatim (AD guards and their comments, §2.6), re-exported from `modules/crk`; (2) `LinearReconstruction` / `LimitedReconstruction` as wp.funcs returning the pair velocity difference; (3) route CRK `accel` / `dudt` through them; (4) add the `velocityPairPolicy` field; (5) give the Monaghan viscosity kernels (`wp_diffusion.py`) the policy -- they have no per-particle velocity gradient yet, so this needs the `computeShearTensor` pass (Phase 3 Build, below). Tests as listed in the Phase 3 section (linear-field annihilation, limiter behaviour, pairwise conservation 1e-12, gradcheck script).

**References to compare against.**
| host | reference | how |
|---|---|---|
| Monaghan, CompSPH default | smoke `results/av_smoke_ref_M0g`, full `results/av_M0g_*` | `scripts/av_report.py --config baseline --profile smoke` then `--compare`; with the `Raw` policy it must stay IDENTICAL (tol 0) |
| CRKSPH | `results/av_crk3_{crkNone,crkCullenDehnen2010}` (Sod, Gresho, KH, full); smoke `results/av_crk2_*` | the old `av_M0f_crk` C&D column is superseded (it was fixed alpha = 1) |
| KH, the best Phase 3/4 target | `docs/av/kh256_2026-10-07/` (nx 256, t = 3, ~10 min a run) | Monaghan + C&D / Rosswog peak 0.21 and decay to 0.105; default CRK peaks 0.23-0.24 and holds 0.17-0.19: reconstruction should close that gap |

**Gotchas learned.** (i) Float32 contraction: with a live switch (alpha != 1), moving arithmetic across `wp.func` boundaries changes FMA fusion at 1 ulp, so only alpha = 1 paths and the Monaghan-host smoke are bit-identical across refactors. (ii) CRK Gresho at t = 3 amplifies a 1e-13 perturbation to ~1e-3 even in float64: do not read a CRK Gresho L1 change below ~5 % as signal; Sod / KH agree to ~1e-3. (iii) `git worktree add <scratch> HEAD` plus `PYTHONPATH=<wt>/src` is the clean before/after tool. (iv) Chain GPU runs one at a time, video on; `--out` paths for `av_report` need the `results/` prefix.

**Small open items (none blocks Phase 3).** `Monaghan1997b` / `Dukowicz` have no source on disk; `Monaghan1992`'s `1e-14 h` regulariser leaves a latent `r -> 0` singularity (CompSPH's default; CRK's `Frontiere2017` has the paper's `eps^2 = 1e-2`); the legacy `scaleBeta` flag (double alpha) is for Phase 5A; the Phase 2 KH gate at the sharp IC / nx 256 was not run for the Monaghan switches; Sedov's cold-gas background saturates the Rosswog trigger; CompSPH with a live switch is not covered by a stored baseline.

## Start-up work order (added 2026-09-30)

The phases below are the destination; this is the order the first weeks are
worked in, decided after re-auditing Part 2 against the code and against
`diffSPH/src/diffSPH/modules/switches/`. Tick boxes here and in the phase
markers; keep notes dated.

**Why this order.** Phase 1's bar is *bit-for-bit vs the M0 baseline*, so the
baseline must exist before any refactor (the state-tag fix in particular may move
numbers). The cheap audit fixes and the fixed-β policy come next because every
paper port needs them. Stubs come last.

**What diffSPH offers (checked 2026-09-30).** `modules/switches/` has ~40-line
`Balsara1995`, `Colagrossi2004`, `MorrisMonaghan1997`, `Rosswog2000` (plus C&D and
Cullen-Hopkins, which warpSPH already has). Balsara/Colagrossi return a
*limiter in [0,1]*, not an α — a different contract from a switch. diffSPH's
`Rosswog2000` is a Morris-Monaghan divergence-source switch with an `(2-α)` factor,
**not** Rosswog 2020's entropy trigger. diffSPH has **no** Sphenix, Wadsley,
entropy-trigger or García-Senz code: Phases 2/4/5B/6 come from the PDFs.

### S1 — Baseline first (= M0)
- [x] `L1` metric + unit test (2026-09-30)
- [x] Regression bar green before any change: `run_tests.sh`, gradcheck, 35-case smoke sweep (2026-09-30)
- [x] `compressibleDiagnostics`: entropy, angular momentum, `alphaMean/Max/ActiveFraction` (2026-09-30; `sod.py` had its own copy, now delegates)
- [x] Π linear/quadratic AV power split: `modules/dissipation/avPower.py` `computeAVPowerSplit` (diagnostic path only; CRKSPH -> NaN). `switchState.dvdt_diss` left alone (R&H's own field) — decide populate/remove in S2
- [x] neighbour-count mean/min/max: `avReportDiagnostics` in `cases/compressible.py` (opt-in with the power split; used by `av_report.py`)
- [x] promote the three `.tmp/` probes to `scripts/` (`probe_sod1D.py`, `probe_sodND.py`, `probe_greshoControl.py`; copied as-is, output paths moved to `results/probes/`; they are the old overlay probes with no video — `av_report.py` is the M0 instrument)
- [x] `scripts/av_report.py` (`--config baseline`, `--profile smoke|full`, `--repeat N` = lock check). **MVP: cases `sod`, `gresho` only.** Still to add: sedov, noh, yee, linearWave, KH, RT, sod2d/3d; StepTimer-based ms/step (current `wallMsPerStep` includes compile); shock width; `--compare`; `--maps`
- [x] reproducibility lock verified twice for `sod` + `gresho` x 3 configs (2026-09-30). First pass FAILED for R&H/Gresho: entropy-dissipation term used `scatter_sum` (CUDA atomic add, run-to-run order) amplified by the alpha switch; fixed with `torch.segment_reduce` in `ReadHayfield2012.py`, re-run bit-identical
- [x] report written: `docs/av/av_baseline_2026-09-30.md` (in the working tree, **not committed**)
- [x] report extended to sod2d/3d, sedov (3D), noh, yee, linearWave, KH, RT; 10 cases x 3 configs, all 30 pairs bit-locked across two runs (2026-09-30) — see notes for what each case measures and its caveats
- [x] `C_q = 2` reference family (`--config baselineQ`: sod, noh, sedov, gresho), 12 pairs bit-locked. Added because at `C_q = 0` Noh is degenerate (cold gas, c = 0 -> zero viscosity). Full report: `docs/av/av_baseline_2026-09-30.md`
- [x] **Step 0a (done 2026-09-30, local tag at 0ed8c72, not pushed):** tag `milestone/av-baseline` as a locked, documented baseline, known findings included (user agreed 2026-09-30; §16 is deliberately *not* a precondition)
- [x] **Step 0b (done 2026-09-30):** `av_report.py --compare A B [--tol T]` — per-scalar relative difference between two report runs; the instrument for every bit-for-bit check below
- [~] later, non-blocking: StepTimer ms/step with warmup (done as the sweep's synchronised `timing` job), shock width (sod/noh) (not done), `--maps` detector maps (done as the sweep's `maps` job, not as an `av_report` flag)

### Agreed order after S1 (user, 2026-09-30) — each step checked with `--compare` against the M0 report

Steps 1–5 are S2/S3 below in this order; S4 is independent (any time after step 0);
**S5 (OPEN_PROBLEMS §16) comes after the last step, not before** — it blocks nothing in S2–S4 and only limits how far
strong-shock (Sedov/Noh) detector comparisons can be trusted, so it must be done before Phase 2's Sedov/Noh validation rows.

### S2 — Cheap audit fixes + fixed β (start of Phase 1)
- [x] **1.** (done 2026-09-30, smoke baseline bit-identical, 30 pairs) deleted dead `limitXi`; renamed pi.py's local `xi` -> `kernelXi` with a comment (it is `sphKernel_xi` = packing ratio x kernel scale, not a support ratio). The serialised `correctXi` config field keeps its name (stored configs); its comment now says what it is
- [x] **2.** (done 2026-09-30; smoke baseline bit-identical, full test suite green) `entropies` -> `('entropy',)`, `pressures` -> `('pressure',)` in `systems/compressibleMonaghan.py` and `systems/compSPH.py`. Consumers checked first: nothing in warpSPHIntegrators / warpSPH / warpSPHCore / warpSPHPlotting reads the `damping` or `soundSpeed` tags (only `find_tagged_field` lookups exist, none for those names), so they were unused labels. **Closes PESPH_PLAN §2.1.** Same copy-paste `pressures ... ('damping',)` remains in `systems/incompressible.py` and `systems/weaklyCompressible.py` (out of this plan's scope, also unconsumed)
- [x] **3.** (done 2026-09-30; smoke baseline bit-identical at defaults, gradcheck_dissipation green, `tests/test_betaMode.py`) fixed-β: `BetaMode {Coupled (default, beta = alpha_bar C_q), Fixed (beta = C_q)}` on `DiffusionParameters.betaMode` (`pi.py`), dict round-trip (configs stored without the key load as Coupled), runner banner shows `(beta fixed)`. `scaleBeta` kept as the documented LEGACY flag (Coupled mode only: beta = alpha_bar^2 C_l C_q, matches no paper); ignored in Fixed. The plan's 3-variant `CoefficientPolicy` maps onto it: `Fixed(alpha,beta)` = `NoneSwitch` + either mode, `SwitchedAlphaFixedBeta` = switch + Fixed, `SwitchedAlphaCoupledBeta` = switch + Coupled. **Not covered:** CRKSPH's own viscosity (`modules/crk/accel.py`, `dudt.py`) reads `C_q` directly and ignores `betaMode`
- [x] **4.** (done 2026-09-30) `modules/shockCapturing/wrapper.py`: `SWITCHES` registry (`scheme -> (termsFn, updateFn)`, `registerViscositySwitch`), `PLANNED` names the AV_PLAN step for each stub; unimplemented members raise `NotImplementedError` (was a generic `ValueError`); `tests/test_viscositySwitchRegistry.py` covers every enum member
- [x] **S2 exit check passed (2026-09-30):** smoke baseline, full-profile `C_q = 0` baseline (10 cases x 3 configs, 30 pairs) and full `C_q = 2` family (12 pairs) all IDENTICAL (tol 0, i.e. bit-for-bit, stronger than the planned 1e-6 for C&D/R&H) vs the M0 reports; full test suite + all gradchecks green
- [x] decisions (user, 2026-09-30 "sounds good" to the stated defaults): deleted `crkSPH.py`'s commented-out `updateViscositySwitch` block, leaving a note; removed the unused `switchState.dvdt_diss` field (`avPower.py` supersedes it; only R&H's two constructor calls passed `None`). **Consequence worth knowing:** CRKSPH never advances `alpha0s`, so C&D / R&H under CRKSPH recompute alpha from the initial `alpha0s` each step — no decay memory. Restoring the call would change CRKSPH results; not done.

### S3 — Fill the stubs (step 5, after the registry) — DONE 2026-09-30
- [x] standalone Balsara factor `balsaraFactor` (`modules/shockCapturing/Balsara1995.py`), used by Read-Hayfield (bit-identical vs M0 smoke baseline) and as the `Balsara1995` switch (alpha = B, instantaneous); `Balsara1995` stub retired
- [x] `MorrisMonaghan1997` switch — checked against the PDF (Eqs. 5, 6, 13: `tau = h/(C_1 c)`, `C_1 = 0.2`, `S = max(-div, 0)`, `alpha_inf = 0.1`). **diffSPH's `tau = h c / l` is dimensionally wrong; not copied.** Decay taken implicitly (`switchRelaxation.relaxAlpha`, stable for any dt/tau, unit-tested at dt/tau 0.1..100); `h -> smoothing length` with CullenDehnen2010's `1/sphKernel_xi`. New tunable `morris_C1` (default 0.2, dict round-trip with `.get`)
- [x] `Rosswog2000` divergence-source switch, transcribed from diffSPH and **checked against the paper 2026-10-01** (`rosswog2000`, Appendix A, Eqs. A.5-A.6: source `max(-div,0)(alpha_max - alpha)`, `tau = h/(eps c)`, eps 0.2 = `morris_C1`; matches; the paper's constants are alpha_max 1.5, alpha_min 0.05; beta = 2 alpha and the Balsara factor in mu are pair-operator matters, not in the switch). Steady-state test against the paper's ODE added. The 2020 entropy trigger is still to come as a new `Rosswog2020` member (Phase 2)
- [x] `Colagrossi2004` limiter (trace-free shear norm); regularised with `eps c / h` (`balsara_const`), not diffSPH's dimensionally wrong `1e-14 h`
- [x] `PLANNED` is now empty; `SWITCHES` has all 8 enum members; `tests/test_dissipationSwitches.py` (relaxation stability/steady state/exponential limit, Balsara limits, per-switch wiring on Sod and Gresho); full suite + gradchecks green. No warp kernels were added (all torch-level), so no new gradcheck.
- [x] reference numbers: `--config switchesS3` on sod + gresho, bit-locked (`results/av_S3_switches/`). Findings: Balsara == Colagrossi **exactly** on 1D Sod (curl and trace-free shear are both 0 in 1D) and both ~ NoneSwitch there; MM97 Sod peak alpha 0.51 vs the paper's Eq. (19) no-decay ceiling `alpha_inf + ln(v1/v2)` ~ 0.67 for this Sod shock; on Gresho all four keep the vortex (peak 0.74-0.84, L1 0.07-0.08, <1.1 % L lost)

### S4 — Side track: CRK `x_ij` sign discrepancy (§2.2)
- [x] **Analysis by reading (2026-09-30), experiment pending.** `computeDistanceVec(x, y) = x - y`, so `x_ij = x_i - x_j` in both files. Linear extrapolation of each particle's velocity to the pair midpoint gives `v^_i = v_i - (1/2) phi J_i x_ij` and `v^_j = v_j + (1/2) phi J_j x_ij` — exactly garciasenz2026 Eqs. (11)-(12). `accel.py:152-156` does this (`+x_ij` on the j side, `v_dot_j = v_j + v_corr_j`). **`dudt.py:135-138` uses `matmul(gradV_j, -x_ij)` with the same `v_dot_j = v_j + v_corr_j`, i.e. `v_j - (1/2) phi J_j x_ij`: the j-side reconstruction in the energy equation has the opposite sign to the momentum equation's.** Also: `dudt.py` passes `cs_i, cs_i` to the first `computePi_actual` and `cs_j, cs_j` to the second, where `accel.py` passes `cs_i, cs_j` to both. If these are bugs they break the heating-vs-kinetic-loss consistency of the CRK viscosity whenever `phi > 0`.
- [x] **Experiment done (2026-09-30)**, CRKSPH host, `NoneSwitch` and C&D, sod / gresho / sedov / noh / yee, three variants of `dudt.py`: (a) as is, (b) the j-side sign fixed (`matmul(gradV_j, x_ij)`), (c) = (b) + symmetric `cs_i, cs_j` arguments. Reports: `results/av_S4_{a,b,c}/` (no video: scalar A/B; nothing adopted). **Result: essentially no effect.** Sod, Sedov, Noh and Yee agree to ~1e-4 relative across all three variants; CRKSPH total-energy drift is already <= 4e-6 in every case and variant (Sedov/Noh: 0, i.e. CRKSPH conserves exactly where the Monaghan host gains 20-68 %, OPEN_PROBLEMS §16). Only **Gresho** moves, and (b) moves it the right way: angular-momentum loss 6.6e-4 -> 1.6e-4 (NoneSwitch) and 1.1e-3 -> -1.3e-4 (C&D); `L1(v_phi)` 0.0413 -> 0.0404 / 0.0435 -> 0.0390; peak speed 1.067 -> 1.055 / 1.065 -> 1.045. (c) on top of (b) changes nothing distinguishable (angular momentum -3e-5 / 2e-4, L1 0.0405 / 0.0404, peak 1.049 / 1.070): **no evidence for or against the `cs` asymmetry**, which may be deliberate alongside the `useJ` flag.
- [x] **Adopted 2026-10-01 (user: "make the s4 change").** `dudt.py` now uses `matmul(gradV_j, x_ij)` (a comment gives the Eq. (12) reasoning); `cs` arguments left as they were. **Correction to the experiment above:** its Gresho numbers were single runs, and **CRKSPH Gresho is not run-to-run reproducible** (unlike the Monaghan/CompSPH hosts, which are bit-locked). A 5-vs-5 repeat of the same code (`results/av_S4_ab.log`): old sign L1(v_phi) 0.0401 +- 0.0017, peak 1.053 +- 0.011, angular-momentum loss 3.2e-4 +- 3.3e-4; fixed sign 0.0414 +- 0.0021, 1.059 +- 0.009, 1.9e-4 +- 3.3e-4. **The 4x angular-momentum gain was noise; the fix has no measurable effect on Gresho.** It is kept because it matches Eqs. (11)-(12) and `accel.py`, not because it measurably helps. Video-backed run: `results/av_S4_adopted/runs/`. **RESOLVED 2026-10-01:** the CRK nondeterminism was `compSPH_deltaU_multistep`'s `scatter_add_`; replaced by `segment_sum`, CRK runs are now bit-identical (CRKSPH_LIMITER_PLAN note 2026-10-01). Was: find the CRK nondeterminism (probably an atomic scatter; the R&H lock failure in S1 had the same cause) -- it means no CRK number in this plan resolves differences below ~1e-3 in angular momentum or ~5 % in L1(v_phi) without repeats.

### S5 — OPEN_PROBLEMS §16: Monaghan energy gain on strong shocks — DONE 2026-09-30
- [x] root cause found with `scripts/probe_monaghanEnergy.py` (net energy injected per RHS piece): the viscous heating kernel lacked the 1/2, heat = exactly 2x the kinetic loss; fixed, `tests/test_monaghanEnergy.py` (fails by 6-7 % without it), Monaghan budget 5e-3 -> 1e-4, baseline regenerated (M0c). Full write-up: `docs/historic_plans/RESOLVED_PROBLEMS.md`. Follow-up: OPEN_PROBLEMS §17 (R&H entropy dissipation)

### M0e — device port (2026-10-01): the reference is now `results/av_M0e_baseline`, `_baselineQ`, smoke `results/av_smoke_ref_M0e/`
The Read-Hayfield pair loops moved from host torch code (raw `x_i - x_j`, no minimum image; a hand-copied kernel derivative with a silent Wendland2 fallback) to warp kernels
(`wp_readHayfield.py`, gradchecked in `scripts/gradcheck_shockCapturing.py`, periodic-shift invariance in `tests/test_readHayfieldDevice.py`). Only the R&H columns moved, slightly
(KH mode amplitude 0.0665 -> 0.0722 is the largest); `none`/C&D and the S3 family are unchanged. Lock: 14 R&H pairs re-run, bit-identical. Details in `docs/av/av_baseline_2026-09-30.md`.

### M0d — scheme correction (2026-10-01): the Read-Hayfield columns were `results/av_M0d_*` (superseded by M0e)
OPEN_PROBLEMS §17 resolved: R&H Eq. (33) has the density ratio inside the bracket (`A_i - A_j (rho_j/rho_i)^(g-1)`), the code had it multiplying
the difference; `K_ij` is now pair-symmetric. Only the R&H columns changed (`none`/C&D bit-identical to M0c). R&H energy drift is now 1e-7..6e-4
(was up to 1.5e-2), **Noh at `C_q = 2`: +23 % -> 1.7e-6, post-shock density -16 % -> -0.11 %**; the "R&H is a blocker for strong-shock rows"
remark under M0c is withdrawn. New reference: `results/av_M0d_baseline`, `_baselineQ` (merges of M0c with the new R&H runs), smoke ref `results/av_smoke_ref_M0d/`;
the S3 family (`results/av_M0c_s3`) is unaffected. Lock: R&H re-run, 14 pairs bit-identical.

### M0c — scheme correction (2026-09-30): the reference is now `results/av_M0c_*`
OPEN_PROBLEMS §16 (Monaghan viscous heating lacked the 1/2) changed every Monaghan-host compressible number, so the baseline was regenerated:
`results/av_M0c_baseline`, `_baselineQ`, `_s3`, smoke ref `results/av_smoke_ref_M0c/`, doc `docs/av/av_baseline_2026-09-30.md`. 50 pairs, all bit-locked.
Consequences for the plan: `none`/C&D now conserve energy to round-off; **Noh at `C_q = 2` passes the +-3 % gate** (`none` -0.08 %, C&D +0.06 %), so the
"Noh needs Phase 5A" reading is withdrawn (the `C_q = 0` default still cannot run cold-gas shocks: `c = 0` -> no viscosity); **Read-Hayfield does not conserve
energy (OPEN_PROBLEMS §17)** — a blocker for using it as the R&H reference in strong-shock rows until understood. The Monaghan energy budget is now 1e-4.

### M0b — harness correction (2026-09-30)
`AVConfig.switchParams` now applies `alpha_min/alpha_max` to EVERY case; in the first M0 they were case params that only
`sod`/`sodND` read, so Read-Hayfield ran in its designed 0.2-1.0 range on the Sod family only. `none`, C&D and all Sod numbers are
unchanged (bit-identical); R&H changes on the smooth cases (Gresho L1 0.071 -> 0.101: its alpha floor keeps dissipating; KH amplitude
0.112 -> 0.068). New reference: `results/av_M0b_*`, smoke ref `results/av_smoke_ref_M0b/`, doc `docs/av/av_baseline_2026-09-30.md`
(regenerated; includes the four S3 switches). The `milestone/av-baseline` tag stays at the first M0 commit. `av_report.py --merge`
combines report dirs (later overrides earlier per config/case).

### How each S2/S3 step is checked (cheap form)
`results/av_smoke_ref_M0/` is the smoke-profile baseline from the tree at the M0 tag; after a step run
`scripts/av_report.py --config baseline --profile smoke --out <dir>` then `--compare results/av_smoke_ref_M0 <dir>`
(~6 min, 30 pairs, must be IDENTICAL unless the step is expected to move numbers). The full-profile lock is re-run
once at the end of S2.

### Notes
- 2026-09-30: **the compressible cases' default scheme is CompSPH, not Monaghan** (`C_q=2`, `Monaghan1992`); the
  plan's host is Monaghan (`C_q=0`, `Price2012_98`), so `av_report.py` must pass `scheme='Monaghan'` explicitly. Sod
  at 150 steps under C&D: Monaghan power all linear (0.031); CompSPH 0.036 = 0.028 lin + 0.008 quad; split residual 0 (additive).
- 2026-09-30: M0 baseline finding: **Monaghan default `C_q=0` => quadratic AV energy is exactly 0 in every baseline run**
  (the "non-zero on sod" marker holds only for CompSPH/CRKSPH hosts; verified CompSPH 0.028 lin + 0.008 quad).
- 2026-09-30 **M0 findings from the full baseline** (see `docs/av/av_baseline_2026-09-30.md`): (1) `C_q = 0` => Noh degenerate (zero viscosity in cold gas); (2) with `C_q = 2` Noh post-shock rho still -23..-31 %; (3) Monaghan gains energy on Sedov/Noh independent of C_q (OPEN_PROBLEMS §16) — confounds every strong-shock detector comparison on this host; (4) Gresho/Sod detector differences are clean and reproducible.
- 2026-09-30 **new cases in `av_report.py` — what they measure / caveats**
  - `sod2d/3d`: same window and metrics as `sod`, D&A IC (the sodND default `0.25/0.1795` is not used), `transverseSpacings=26` (D&A's light state needs a wider periodic slab than the default 20). 3D is slow (~4 s/step, 583 neighbours) and run at nx=20 (15 steps): **too coarse to be a quality number** (R4 rho err 22 %); a resolution/perf follow-up.
  - `sedov` (3D nx=40): peak rho/rho0 vs 4, peak-radius vs exact `r2`, E0 recovery. **Monaghan does not conserve energy here: see OPEN_PROBLEMS §16** (1D 1.0 -> 2.33; CompSPH exact). Every Sedov/Noh number from this host is confounded by it until §16 is understood.
  - `noh` (1D): post-shock rho over `|x| in [0.2, 0.8] v_s t` vs `rho0((g+1)/(g-1))` = 4. NoneSwitch on the Monaghan host gives ~2.0 (-50 %), far outside the plan's +-3 % gate: consistent with **no quadratic viscosity** (`C_q = 0`) failing on a strong shock -> a Phase 5A data point, not a bug report yet.
  - `yee`: stationary vortex, L1 of |v| for r<=3 vs analytic, peak-speed ratio, angular-momentum loss (t=4).
  - `linearWave`: run at **A = 1e-4, t = 1/4 crossing** (case default A=1e-6 is 7x noise in float32; t=1 crossing has zero analytic velocity). Compares the cos(kx) velocity coefficient to `-(A/rho0 c) sin(k c t)`.
  - `kelvinHelmholtz` (nx 128, t=1.5): McNally (2012) transverse-velocity mode amplitude; NoneSwitch gives 2.0e-2 at t=1.5 — the same scale as garciasenz2026 Table 3 KH1 (AV alone, 2.87e-2), against McNally 14.79e-2.
  - `rayleighTaylor` (nx 64, t=4): interface tips + max velocity, reported only.
- 2026-09-30: `results/` is gitignored, so "report committed" means copying the final `report.md` into `docs/` (e.g. `docs/av/av_baseline.md`).
- 2026-09-30: audit re-checked, no drift. Effort estimate for S2+S3 ≈ 3-4 days.
- 2026-09-30 (user decision, from the CRKSPH clean-up): **any CRKSPH numbers in the M0 baseline are taken at the current
  CRK constants** — limiter `(eta_crit, eta_fold) = (1/3, 0.2)` (known to be in the wrong units, CRKSPH_LIMITER_PLAN O2),
  `C_l = C_q = 1`, viscosity regulariser `eps^2 = 1e-2` (post OPEN_PROBLEMS §15 fix, commit d3081d9) — and the report
  should state that. The limiter and `C_l`/`C_q` are to be revisited together, later, when the higher-order / Riemann-MUSCL
  work exercises limiters too; if their constants change, the CRKSPH part of the baseline moves (Monaghan / CompSPH hosts
  do not use the CRK limiter). `C_l = 2` was tried (CRKSPH_LIMITER_PLAN note (b)) and not adopted.
- 2026-10-01 (user, CRKSPH_LIMITER_PLAN note (f)): the CRK limiter default is now `(eta_crit, eta_fold) = (1/n_h, 0.2/n_h)` = (0.25, 0.05) at n_h = 4 (derived from n_h per step). **The CRK rows of the M0 baseline (`results/av_M0d_*`, `docs/av/`) were taken at the old (1/3, 0.2) and no longer match the default**; `C_l` stays 1. **Re-taken 2026-10-01: the CRK reference is now `results/av_M0f_crk`** (`crkNone`, `crkCullenDehnen2010` x sod, sod2d, sedov, noh, gresho, yee, linearWave, kelvinHelmholtz, rayleighTaylor; 18 pairs, video on for all but 3D sedov / noh -- OPEN_PROBLEMS §18; not bit-lock-repeated: CRK runs are deterministic since the same day). Moves vs the old-constant S4 numbers: Gresho L1(v_phi) 0.040 -> 0.033, peak 1.055 -> 1.012, Yee L1 -16 %, Sod contactSpikeA halved, Noh post-shock rho error halved; Sedov and the rest of Sod unchanged (<1 %). Monaghan / CompSPH rows are unaffected (M0e). **Superseded 2026-10-07:** the `crkCullenDehnen2010` column of `results/av_M0f_crk` (and the S4 numbers) is a fixed-alpha = 1 run (CRK viscosity switch bug, Phase 1 clean-up items 2-3); the CRK reference is now `results/av_crk3_*`. `crkNone` is unchanged.

- 2026-10-06 **the M0e baseline is stale since 013a22f** (compressible-walls work: adaptive h clamped to
  `n_h (m/rho)^(1/d)` on fluid rows, OPEN_PROBLEMS §20). Smoke at HEAD vs `results/av_smoke_ref_M0e`: 30/30 pairs
  differ, `none` included. Phase 2 is therefore compared against a fresh none / C&D reference taken at HEAD
  (`results/av_M2_ref_{none,cd,noneQ,cdQ}`), not against M0e. Whether the clamp stays (and M0e is formally
  replaced by an M0g) is the §20 decision.
- 2026-10-06 (later) **M0e is valid again.** OPEN_PROBLEMS §20 resolved: the clamp degrades smooth pure-fluid flow (inviscid
  linear wave error 6x, Gresho angular-momentum loss +30-85 %) while helping shock contacts, so it is now wall-only
  (`SimulationConfig.supportVolumeClamp='walls'`, user rule: a boundary fix that degrades fluid behaviour is off for pure
  fluid). Pure-fluid runs at the new default are bit-identical to M0e (smoke 30/30; full 28/28 with the clamp off). The
  `results/av_M2_ref_*` columns are the clamp-on comparison, kept for the record; Phase 2's numbers at the default are
  `results/av_noclamp_rosswog2020{,Q}`. `av_report.py --supportVolumeClamp` selects the mode for any run.

- 2026-10-06 (evening) **the reference is now M0g** (`results/av_M0g_{baseline,baselineQ,s3,phase2}`, smoke
  `results/av_smoke_ref_M0g`; CRK `results/av_M0g_crk_all` once re-run): SUPPORT_SOLVER_PLAN fixed Owen's psi_H table
  (`owenTable='lattice'`, h was 2 % / 0.9 % too large on 2D / 3D lattices). vs M0e every headline metric moves <= 3.5 %, 1D
  at round-off; Phase 2 verdicts unchanged (KH 0.011, Gresho alphaMean 0.049).

## The host scheme

**Monaghan is the AV laboratory; CompSPH is the conservative cross-check.**

[`schemes/monaghan.py`](src/warpSPH/schemes/monaghan.py) is the plain pairwise
operator — a detector change there is not confounded by anything else, and it is
where Phase 6 already validated. **CRKSPH is deliberately excluded as the primary
host**: it bakes slope-limited reconstruction into
[`modules/crk/accel.py`](src/warpSPH/modules/crk/accel.py) /
[`dudt.py`](src/warpSPH/modules/crk/dudt.py), which is precisely the variable
Phase 4 is trying to isolate, and its compatible energy update is not an ODE
right-hand side (see PESPH_PLAN §4.2). CRKSPH appears in the bake-off as a
*reference point*, not as a detector host.

## Three rules this plan enforces

1. **"When do I dissipate?" ≠ "how do I dissipate?"** Detector, coefficient,
   pair velocity and operator are four independent axes. Phase 1 makes that
   structural, not merely conceptual.
2. **Never make the next formulation depend on the previous formulation's
   implementation.** No `SphenixAV(CullenDehnenAV)`. Every detector is a peer
   implementation of one interface.
3. **Every phase reports into [Part 0](#part-0--the-av-report)'s table before it
   is marked done.** The report is defined *first* precisely so that no phase
   gets to invent its own success criterion afterwards.

---

# Part 0 — The AV report

This is the contract. It is written before any phase because every later phase
reports into it, and because the roadmap's failure mode is accumulating
half-validated schemes that were each judged against a criterion chosen after the
fact.

## 0.1 Producer

New `scripts/av_report.py`, built on the existing benchmark plumbing rather than
a parallel one:

- [`benchmarks/common/report.py`](benchmarks/common/report.py) — `outDirFor`,
  `writeResults`, `mdTable`, `writeSummary`, `saveFig`. Already emits the rich
  `meta` block (timestamp, warp/torch versions, device, precision, and the
  `warpSPH@git` / `warpSPHCore@git` / `warpSPHIntegrators@git` SHAs) that makes a
  result reproducible.
- [`benchmarks/common/runner.py`](benchmarks/common/runner.py) — `StepTimer`,
  `_peakMemoryMB`, `tensorFootprintMB` for group F.
- Output: `results/av_<stamp>/{results.json, report.md, plots/}`.

CLI shape, mirroring [`scripts/validate_scheme.py`](scripts/validate_scheme.py):

```bash
scripts/av_report.py --config baseline            # named config from a registry
scripts/av_report.py --config rosswog2020 --profile smoke   # 20-step smoke
scripts/av_report.py --compare baseline rosswog2020         # side-by-side table
scripts/av_report.py --list
```

`--profile smoke` (coarse, ~20 steps, for CI-ish use) and `--profile full` (the
resolutions quoted below) follow `validate_scheme.py`'s convention exactly.

## 0.2 The metric table

Six groups. Every row names the case, the code that produces it today, and the
code to add.

### Group A — shock quality

| Metric | Case | Source | Gate |
|---|---|---|---|
| `L1(v_x)`, García-Senz Eq. (20), window `x ∈ [0.05, 0.868]` | `sod`, `sod2d`, `sod3d` | exact Riemann in [`sodSolution.py`](src/warpSPH/caseUtils/compressible/sod/sodSolution.py) `solve()` | per-phase, comparative |
| contact `P`-spike, `A`-spike | `sod` 1D | `_sod_contact_spike` in [`tests/test_shockCapturing.py`](tests/test_shockCapturing.py) | `abs(P_spike) < 0.5`, `abs(A_spike) < 0.5` — already asserted |
| region means vs exact (R3-R5) | `sod` 1D | phase6 log baseline: R3 ρ `+0.7%`, R4 ρ `−3.1%`, R4 P `+1%` | within ±5 pp of the Phase-0 number |
| shock width in units of `h` | `sod`, `noh` | new | reported, monotone under refinement |
| peak `ρ/ρ₀` vs `(γ+1)/(γ−1) = 4` | `sedov` 3D | new; `SedovSolution` already in [`sedovSolution.py`](src/warpSPH/caseUtils/compressible/sedov/sedovSolution.py) | **must approach 4 from below — overshoot is a failure** (garciasenz2026 Fig. 2) |
| post-shock `ρ_s = ρ₀((γ+1)/(γ−1))^dim` | `noh` | closed form already in [`cases/noh.py`](src/warpSPH/cases/noh.py) `shockState` | within ±3% |

**`L1` does not exist in this repo.** Only `relL2` at
[`metrics.py:20`](benchmarks/common/metrics.py#L20). Phase 0 adds it beside
`relL2`, with the same signature convention (reference second):

```
L1(a, b, mask=None) = mean(|a - b|)      over the masked sample set
```

This is García-Senz Eq. (20), `L1 = (1/N_S) Σ_{i∈S} |p_i^SPH − p_i^ref|` — a
*mean absolute*, not normalised. Do not quietly substitute a relative norm; every
number quoted from that paper is this one.

### Group B — dissipation accounting

| Metric | Source | Gate |
|---|---|---|
| integrated AV energy, **split linear (`C_l`) / quadratic (`C_q`)** | new; requires the operator to return the two contributions separately | reported |
| `mean α`, `peak α`, `fraction(α > 0.1)` | new in `compressibleDiagnostics` | per-phase |
| AV dissipation *power* `dE_AV/dt` | `switchState.dvdt_diss` is **declared and never populated** ([`switchState.py:28`](src/warpSPH/modules/shockCapturing/switchState.py#L28)) | reported |

The linear/quadratic split is the one genuinely new piece of plumbing here.
[`pi.py`](src/warpSPH/modules/dissipation/pi.py) computes a single `v_sig` per
formulation that already mixes `C_l·c` and `C_q·μ_ij` (e.g.
[`pi.py:125`](src/warpSPH/modules/dissipation/pi.py#L125)), so the split is
obtained by evaluating `Π` twice — once with `C_q = 0`, once with `C_l = 0` — in
the diagnostic path only, never in the hot path.

### Group C — smooth-flow preservation

| Metric | Case | Source | Gate |
|---|---|---|---|
| `L1(v_φ)` over `r ∈ [0, 0.5]` vs the analytic `v_φ` | `gresho` ([`greshoVortex.py`](src/warpSPH/cases/greshoVortex.py)) | garciasenz2026 Eq. (22) | comparative, §4.4 |
| peak-speed persistence ratio | `gresho` | promote `.tmp/probe_gresho_control.py` → `scripts/probe_greshoControl.py` | ≥ Phase-0 baseline |
| angular-momentum loss | `gresho`, `yee` | new; **port `_conservationMetrics`** from [`oscillatingDroplet.py:196`](src/warpSPH/cases/oscillatingDroplet.py#L196) — the only angular-momentum accounting in the repo, currently WC-only | reported |
| amplitude decay over N periods | `linearWave`, `yee` | new | ≤ Phase-0 baseline |

### Group D — instability growth

| Metric | Case | Reference | Gate |
|---|---|---|---|
| KH mode amplitude `A(t)` | `kelvinHelmholtz` | McNally et al. (2012): `A(t=1.5) = 14.79e−2` | the garciasenz2026 Table 3 ladder — see §4.4 |
| RT bubble / spike position | `rayleighTaylor` | — | reported |

### Group E — conservation

| Metric | Gate |
|---|---|
| total-energy drift | the existing per-scheme budget, quoted verbatim from [`tests/test_physics.py:284`](tests/test_physics.py#L284): `_ENERGY_DRIFT = {'CompSPH': 1e-5, 'CRKSPH': 1e-4, 'Monaghan': 5e-3}` |
| Sod slab quiet-region density spread | `spread < 0.02` ([`test_physics.py:154`](tests/test_physics.py#L154)) |
| Sedov `E0` recovery at `t=0` | `pytest.approx(spec.param('E0'), rel=1e-3)` ([`test_physics.py:658`](tests/test_physics.py#L658)) |
| angular-momentum drift | reported (new) |

### Group F — performance

| Metric | Source |
|---|---|
| ms/step, ms/RHS | `StepTimer` ([`benchmarks/common/runner.py`](benchmarks/common/runner.py)), CUDA-event based, with untimed warmup so warp kernel compilation never enters a number |
| neighbour interactions/sec | new; `adjacency.numNeighbors` **exists and is never recorded on a trajectory row** |
| extra gradient passes / neighbour loops per step | new; counted, not timed — this is the honest measure of "less machinery" |

## 0.3 The detector map

The original roadmap sketch (superseded by this document) buried this in Phase 6. It belongs here, because Phases 2, 4, 5B and
6 each produce one and they are only useful side by side.

Per-particle dump, via [`io/export.py`](src/warpSPH/io/export.py)'s `writeFrame`
with an extra field group (no new IO format):

```
particle id · position · α · ∇·v · d(∇·v)/dt · <detector-specific scalar>
```

Detector-specific scalar: `ε̇` (Rosswog), `Ξ`/`R` (C&D), `D`/`|T|`/`ξ` (Wadsley),
`S_i` (Sphenix), `φ_ab` aggregated per particle (SLR). Rendered by
`scripts/av_report.py --maps` as one panel per detector on the identical frame.

**This is more informative than another Sod profile plot** and is a required
deliverable of Phases 2, 4, 5B and 6.

## 0.4 Reproducibility lock

A configuration is *locked* when two runs of the identical `caseSpec.json` agree
to `< 1e-6` relative on every scalar in `results.json`. That is the M0 condition
and it is re-checked at every later milestone; a phase that breaks it has
introduced nondeterminism and is not done.

---

# Part 1 — Equation inventory

Transcribed from the local PDFs with their paper equation numbers. Notation is
the paper's, not the repo's; the mapping to repo symbols is called out where they
diverge.

## 1.1 Rosswog (2020) — entropy trigger · `rosswog2020entropy`

Entropy measure and its non-dimensionalised rate:

```
s_a  = P_a / ρ_a^Γ                                                      (15)
ε̇ⁿ_a = (|sⁿ_a − sⁿ⁻¹_a| / sⁿ⁻¹_a) · (τ_a / Δt),   τ_a = h_a / c_{s,a}   (16)
```

Decay, desired value and the switch-on function:

```
dα_a/dt   = −(α(t) − α₀) / (30 τ_a)                                     (17)
αⁿ_a,des  = α_max · S(lⁿ_a),          lⁿ_a ≡ log(ε̇ⁿ_a)                  (18)
S(x)      = 6x⁵ − 15x⁴ + 10x³                                           (19)
x         = min(max((lⁿ_a − l₀)/(l₁ − l₀), 0), 1)                       (20)
```

Constants the paper settles on after experiment: `l₀ = log(1e−4)`,
`l₁ = log(5e−2)`, `α_max = 1`, `α₀ = 0`. Below `ε̇ ≤ 1e−4` the scheme does not
switch on at all; at `ε̇ ≥ 5e−2` it is at `α_max`. The update is **instant up,
exponential down** — compare at each step with the desired value and raise
immediately if it exceeds the current α (their explicit borrowing from
cullen2010). Note the deliberately long `30 τ_a` decay.

The paper also gives the C&D comparison it benchmarks against, with the decay
constant it used for that comparison (`20 h/v_sig`, not the repo's `l = 0.05`
form):

```
αⁿ,CD_a,des = A_a / (0.25 (v_a,sig/h_a)² + A_a),  A = min(−d(∇·v)/dt, 0)  (21)
v_a,sig     = max_b [ c_s,ab − min(0, ṽ_ab·ê_ab) ]                        (22)
```

Separately, §2.1 gives the MAGMA2 **quadratic** midpoint reconstruction:

```
ṽⁱ_a = vⁱ_a + Φ_ab [ (∂_j vⁱ) δ_j + ½ (∂_l ∂_m vⁱ) δ_l δ_m ]_a ,  (δ_i)_a = ½(r_b − r_a)   (10)
Φ_ab = max(0, min(1, 4A_ab/(1+A_ab)²)) · { 1 if η_ab > η_crit ; exp(−((η_ab−η_crit)/0.2)²) else }  (11)
A_ab = (∂_δ v_a)ᵞ x^δ_ab x^ᵞ_ab / (∂_δ v_b)ᵞ x^δ_ab x^ᵞ_ab                (12)
η_ab = min(r_ab/h_a, r_ab/h_b),   η_crit = (32π/(3 N_nei))^(1/3)          (13)
```

with the artificial pressure `Q_a = α ρ_a (−c_s,a μ_a + 2μ_a²)` (7) and
`μ_a = min(0, v^δ_ab η^δ_a/(η_a² + ε²))` (8), `ε = 0.1`. Gradients via the
matrix-inversion correction (4)–(6) — that is García-Senz et al. (2012) IAD,
which this repo does **not** have (see §2.4).

> **Phase 3 note.** Rosswog's reconstruction is *quadratic* (second derivatives
> in Eq. 10). Frontiere/García-Senz is *linear*. Phase 3 implements linear first;
> quadratic is an optional extension, not a prerequisite for Phase 4.

## 1.2 García-Senz & Cabezón (2026) — switch-free SLR AV · `garciasenz2026`

Their operator (the Wadsley 2017 form, and what SPHYNX/SPH-EXA use):

> **Corrected 2026-10-07 (Phase 4, checked against the PDF):** the sign of the quadratic term was transcribed wrong
> here. The paper's Eq. (6) is `(−ᾱ c̄ ω + β ω²)/ρ̄` — Monaghan (1992) with the unscaled `ω = v·r̂` and pair means —
> which is the repo's `Price2012_98` term exactly (`v_sig = C_l c̄ − C_q ω`), so no new `ViscosityTerms` member.

```
Π_ab = { (−ᾱ_ab c̄_ab ω_ab + β ω_ab²)/ρ̄_ab   for ω_ab < 0 ; 0 otherwise }   (6)
ω_ab = v_ab · r̂_ab                                                          (2)
v_sig,ab = c_a + c_b − γ ω_ab ,   γ = 3                                     (3)
```

**`β = 2`, fixed, in every test in the paper.** Typical switch bounds
`α_min = 0.05`, `α_max = 1`. They also note `β = γα/2` follows from equating (1)
and (4), giving `β = 1.5α`, but that `β = 2α` is the more common choice because
it better suppresses post-shock oscillation.

Balsara limiter:

```
B_a = |∇·v|_a / (|∇·v|_a + |∇×v|_a + 1e−4 c_a/h_a)                          (8)
```

Slope-limited reconstruction — the heart of the paper:

```
v'_a,i = v_a,i − ½ φ_ab J_{v_a} x^T_ab                                      (11)
v'_b,i = v_b,i + ½ φ_ba J_{v_b} x^T_ab                                      (12)
```

with `x^T_ab = x_a − x_b` and `J_v` the velocity Jacobian, and the Frontiere
van-Leer-like limiter:

```
φ_ab  = max(0, min(1, 4F_ab/(1+F_ab)²)) · κ_ab                              (13)
κ_ab  = { exp(−((q_ab − q_crit)/q_fold)²)  if q_ab < q_crit ; 1 otherwise }  (14)
q_ab  = min(q_a, q_b),  q_a = |x_b − x_a|/h_a                               (15)
q_crit = (32π / (3 n_ba))^(1/3)                                             (16)
F_ab  = (x_ab J_{v_a} x^T_ab) / (x_ab J_{v_b} x^T_ab)                       (17)
```

`q_fold = 0.2` (Frontiere's recommended value). Balsara-modulated SLR:

```
v'_a,i = v_a,i − ½ (1 − B̄_ab^p) φ_ab J_{v_a} x^T_ab                         (18)
v'_b,i = v_b,i + ½ (1 − B̄_ab^p) φ_ba J_{v_b} x^T_ab                         (19)
```

with `B̄_ab = ½(B_a + B_b)` and `p ∈ {1, 2}`. Error metric:

```
L1 = (1/N_S) Σ_{i∈S} |p_i^SPH − p_i^ref|                                    (20)
```

**Table 1 — the Phase 4 config matrix**, verbatim:

| # | Method | SLR | B modulation | Switches | α |
|---|---|---|---|---|---|
| 1 | AV | No | – | No | `α = 1` |
| 2 | AVSW | No | – | Yes | `α ∈ [α_min, α_max]` |
| 3 | AVSLR | Yes | No | No | `α = 1` |
| 4 | AVSWSLR | Yes | No | Yes | `α ∈ [α_min, α_max]` |
| 5 | AVSLRB | Yes | `(1 − B)` | No | `α = 1` |
| 6 | AVSLRB2 | Yes | `(1 − B²)` | No | `α = 1` |
| 7 | AVSLRB2F | Yes | `(1 − B²)` | No | `α = α_max·max[F, B]`, `F = 0.5` |

> **Correction to carry forward.** The original sketch described this paper as
> "switch-free AV based on removing the local linear velocity component, with
> Balsara-type modulation". True, but incomplete in the way that matters: the
> paper's own §3/§5 ranking puts the *Balsara-modulated* variant **AVSLRB2**
> (row 6) above pure SLR (row 3), and pure SLR is one of the two variants that
> *overshoots* the Sedov compression limit. The variant to implement and
> recommend is AVSLRB2, not AVSLR.

## 1.3 Chen & Nixon (2025) — dynamic β · `chen2025`

No new operator. The claim, from the abstract and §5:

> …we suggest that the coefficient of the quadratic term should be
> time-dependent in a similar manner to the presently used 'switches' on the
> linear term. This can be simply achieved by setting `β_SPH` to be a constant
> multiple of `α_SPH` with `α_SPH` determined by an appropriate switch.

They take `α_max_SPH = 1` and `β_SPH/α_SPH = 2`. Their diagnostic for when the
quadratic term dominates (their Eq. 6):

```
α_num^quad / α_num^lin  =  135 β_SPH h  /  (62π α_SPH H)
```

with `h` the smoothing length and `H` the disc scale height. In a
non-disc geometry, `H` is replaced by the local gradient length scale; the
*ratio* is the reported quantity, not an absolute.

Their measured optimum for smooth discs is `α_SPH ≈ 0.1`, `β_SPH ≈ 0.2` — an
order of magnitude below PHANTOM's `α_max = 1`, `β = 2` default — and the whole
point is that this conflicts with the `β ≈ 2` that strong shocks need, which is
what motivates coupling.

> **See §2.2: this repo already couples β to α and has no fixed-β mode at all,
> so Phase 5A is inverted relative to the sketch.**

## 1.4 Borrow et al. (2022) — Sphenix · `borrow2022`

The operator:

```
dv_i/dt = −Σ_j m_j ζ_ij (f_ij ∇_i W_ij + f_ji ∇_j W_ji)                     (13)
du_i/dt = −½ Σ_j m_j ζ_ij v_ij·(f_ij ∇_i W_ij + f_ji ∇_j W_ji)              (14)
ζ_ij    = −α_V μ_ij v_sig,ij / (ρ̂_i + ρ̂_j)                                  (15)
μ_ij    = { v_ij·x̂_ij  if v_ij·x_ij < 0 ; 0 otherwise }                     (16)
v_sig,ij = c_i + c_j − β_V μ_ij ,   β_V = 3                                 (17)
```

Note (14) is **explicitly symmetrised**, which the plain SPH internal-energy
equation is not; the paper attributes the improvement to multiple-timestepping
errors in strongly viscous regions.

Pair coefficient and Balsara:

```
α_V,ij = ¼ (α_V,i + α_V,j)(B_i + B_j)                                       (19)
B_i    = |∇·v_i| / (|∇·v_i| + |∇×v_i| + 1e−4 c_i/h_i)                       (20)
```

Shock indicator and the update:

```
S_i = { h_i² max(−∇̇·v_i, 0)   for ∇·v_i ≤ 0 ; 0 for ∇·v_i > 0 }             (21)
∇̇·v_i(t+Δt) = (∇·v_i(t+Δt) − ∇·v_i(t)) / Δt                                 (22)
α_V,loc,i = α_V,max · S_i / (c_i² + S_i),   α_V,max = 2.0                    (23)
α_V,i ← { α_V,loc,i                              if α_V,i < α_V,loc,i
        ; (α_V,i + α_V,loc,i Δt/τ_V,i)/(1 + Δt/τ_V,i)  if α_V,i > α_V,loc,i } (24)
τ_V,i = γ_K ℓ_V h_i / c_i ,  ℓ_V = 0.05,  γ_K = kernel gamma
```

Two things to carry into the port (plus a transcription note, 2026-10-07: the PDF typesets Eq. 21 as
`−h² max(∇̇·v, 0)`, which would make `S ≤ 0`; the text and Eq. 23 need `S ≥ 0`, i.e. the minus inside the max — SWIFT's
form, used above. `γ_K` is Dehnen & Aly's cut-off/smoothing-length ratio, which *is* `sphKernelScale`; with `h` the
cut-off as stored here, `τ = γ_K ℓ h/c = ℓ H/c` and Eq. 21's `h² = (H/sphKernelScale)²`):

- **(24) is implicit**, which is why the paper can say the decay is *stable
  regardless of timestep* and `α_V` is strictly positive with a default
  `α_min = 0`. Do not re-derive it as an explicit relaxation.
- **B_i is applied in the pair coefficient (19), not inside the indicator
  S_i.** The paper flags this as deliberately different from most schemes: it
  lets `α_V,i` stay high in a shearing region while Balsara instantaneously
  releases it when the shock arrives.

**The efficiency claim, concretely: Sphenix needs no correction-matrix
inversion, no shear tensor and no second-order `V`.** Only `∇·v`, its finite
difference, `∇×v` and the sound speed. Against the repo's C&D path that removes
the `computeMWarp` neighbour loop
([`wp_computeM.py:208`](src/warpSPH/modules/shockCapturing/wp_computeM.py#L208))
and the `computeShearTensor` gradient pass
([`common.py:53`](src/warpSPH/modules/shockCapturing/common.py#L53)). That is
measurable, and §5B.4 gates on it.

## 1.5 Wadsley, Keller & Quinn (2017) — Gasoline2 detector · `wadsley2017`

Their motivating observation, which is the whole reason to implement this:

> A key problem with any viscosity approach based on divergence (including the
> CD approach) is that it creates viscosity in uniform compression… For a
> spherical flow, `∇·v = ∂v_r/∂r − 2v_r/r`. In spherical collapse, divergence is
> commonly dominated by the second term and is not a reliable shock indicator.

Their answer — the velocity gradient **in the direction of the pressure
gradient**:

```
∇P   = (γ−1) Σ_j m_j u_j ∇_i W(r_ij, h_i)                                   (21)
n̂    = ∇P / |∇P|                                                            (22)
dv/dn = Σ_{α,β} n_α V_αβ n_β                                                (23)
D    = (3/2) [ dv/dn + ⅓ max(−∇·v, 0) ]                                     (24)
```

`D` reaches the extremal value `∇·v` in a shock but is only negative when the
compression normal to the shock exceeds one third of the overall compression —
that is what makes it blind to uniform compression.

```
α_loc,i = α_max A_i / (A_i + v_sig,i²)                                      (25)
A_i     = 2 h_i² ξ_i max(−dD/dt, 0)                                         (26)
dα_i/dt = (α_loc,i − α_i)/τ_i ,   τ_i = h_i/(0.2 c_i)                       (27)
ξ_i     = ((1 − R_i)/2)⁴                                                    (28)
R_i     = Σ_j m_j (D_j/|T|_j) W_R,ij / Σ_j m_j W_R,ij ,  R ∈ [−1,1]         (29)
W_R,ij  = (1 − (r_ij/(2h_i))⁴)
```

`α_max = 2`. Instant raise when `α < α_loc`, otherwise (27).

Two structural points:

- **`T_αβ = ½(V_αβ + V_βα)` keeps the trace.** This is their explicit correction
  to C&D's *trace-free* `S_αβ`: removing the trace makes `|S|` non-zero in a
  strong shock (where the tensor is dominated by one eigenvalue) and blinds it
  to isotropic compression. In this repo, `T` is
  `Shear + (trace/dim)·I` from
  [`common.py:92-99`](src/warpSPH/modules/shockCapturing/common.py#L92) —
  one line, not a new kernel.
- **`W_R,ij` is deliberately not the SPH kernel**, because the kernel weights
  too few central particles and makes `R` noisy.

Stated behaviour of `ξ`, which gives the unit tests directly: `0` in expansion,
`≈ 1/16` in intermediate, noisy, *and uniformly compressing* regions, `1` in
shocks.

> **Resolved 2026-10-07 (Phase 6, §6.2):** C&D's `h` is the support itself (their footnote 2: `W = 0` for
> `r > h`), as stored here; Gasoline2's is half the support ("half the distance to the furthest neighbour"). So
> `H = h_CD = 2 h_G2` for every kernel, *not* via `sphKernelScale`, and in this repo's units Eq. (26) is
> `A = 0.5 H² ξ max(−Ḋ, 0)`, `τ = H/(0.4 c)`, `W_R = 1 − (r/H)⁴`. Eq. (24)'s `3/2`, `1/3` are 3D: blindness to
> `V = −kI` needs `1/d`, `D = ∇·v` in a planar shock needs `d/(d−1)`; the code uses `d/(d−1)[dv/dn + max(−∇·v,0)/d]`
> (the paper's form in 3D). See `modules/shockCapturing/Wadsley2017.py`.
>
> **The factor 2 in Eq. (26) does not transfer.** The paper says it was needed
> because of "the different `h` definition in CD". This repo stores `h` as the
> **cut-off radius**, not the smoothing scale — the exact `h`/`H` hazard
> documented at
> [`limiter.py:135-147`](src/warpSPH/modules/crk/limiter.py#L135). Re-derive the
> constant against this repo's convention; do not copy it. See §6.2.

---

# Part 2 — What the repo already has

Audited 2026-09-19. Several sketch phases are substantially smaller than they
look, and two are larger.

## 2.1 The existing surface

| Ingredient | Status |
|---|---|
| `ViscositySwitch` enum | ✅ 8 members, [`enumTypes.py:43`](src/warpSPH/enumTypes.py#L43) — but **4 of 8 are stubs that raise** (§2.3) |
| Dispatch | ✅ [`wrapper.py`](src/warpSPH/modules/shockCapturing/wrapper.py) — `if`/`elif` chains in *two* functions |
| C&D 2010 | ✅ [`CullenDehnen2010.py`](src/warpSPH/modules/shockCapturing/CullenDehnen2010.py), validated Phase 6, sign resolved |
| Cullen-Hopkins | ✅ [`CullenHopkins.py`](src/warpSPH/modules/shockCapturing/CullenHopkins.py) |
| R&H 2012 SPHS + entropy dissipation | ✅ [`ReadHayfield2012.py`](src/warpSPH/modules/shockCapturing/ReadHayfield2012.py) |
| Balsara limiter | ⚠️ implemented, but *inside* R&H ([`ReadHayfield2012.py:163`](src/warpSPH/modules/shockCapturing/ReadHayfield2012.py#L163)); `ViscositySwitch.Balsara1995` is a stub |
| 12 pairwise AV formulations | ✅ `ViscosityTerms` + [`pi.py`](src/warpSPH/modules/dissipation/pi.py) `computePi_actual` |
| Velocity-gradient tensor | ✅ `computeShearTensor` ([`common.py:53`](src/warpSPH/modules/shockCapturing/common.py#L53)) returns trace / trace-free Shear / Rotation |
| Correction matrix `M = Σ m_j x_ij ⊗ ∇W_ij` | ✅ [`wp_computeM.py:208`](src/warpSPH/modules/shockCapturing/wp_computeM.py#L208) |
| Slope limiter + midpoint reconstruction | ⚠️ exists, **CRK-internal** — see §2.2 |
| Entropy `A` as state + EOS both ways | ✅ `CompressibleState.entropies`, [`eos/idealGas.py`](src/warpSPH/modules/eos/idealGas.py) |
| Exact Riemann / Sedov / Noh solutions | ✅ all three, but used **only as plot overlays** |
| `L1` error norm | ❌ nothing; only `relL2` ([`metrics.py:20`](benchmarks/common/metrics.py#L20)) |
| Entropy(t), angular momentum(t), α statistics, AV dissipation | ❌ none on any compressible case |
| García-Senz 2012 IAD gradients | ❌ absent |
| Rosswog / Sphenix / Wadsley detectors | ❌ absent |

## 2.2 Finding: the reconstruction engine already exists, CRK-internal

This is the largest single scope reduction in the plan. Compare
[`modules/crk/limiter.py`](src/warpSPH/modules/crk/limiter.py) against
garciasenz2026 §2.3:

| Paper | Repo | Match |
|---|---|---|
| `φ_ab` van-Leer form (13) + `F_ab` (17) | `computeVanLeer` ([`limiter.py:41`](src/warpSPH/modules/crk/limiter.py#L41)), `limiterVL` ([`:24`](src/warpSPH/modules/crk/limiter.py#L24)) | exact |
| `κ_ab` (14), `q_ab` (15), `q_crit` (16), `q_fold = 0.2` | `crkLimiter` ([`limiter.py:119`](src/warpSPH/modules/crk/limiter.py#L119)), `eta_crit`/`eta_fold` in `CRKViscosity` ([`crkSPH.py:30`](src/warpSPH/configurations/crkSPH.py#L30), defaults `0.3333`/`0.2`) | exact |
| SLR (11)–(12) | `v_corr_i = phi_ij/2 * matmul(gradV_i, x_ij)` ([`dudt.py:134`](src/warpSPH/modules/crk/dudt.py#L134), [`accel.py:151`](src/warpSPH/modules/crk/accel.py#L151)) | exact |

Both cite Frontiere et al. (2017) as the source, so this is the same algorithm
arriving twice. **Phase 3 is an extraction, not a build.** What is genuinely
missing is only: (a) it is unreachable from any non-CRK scheme, (b) there is no
Balsara modulation of `φ` (Eqs. 18-19), and (c) there is no test that the
reconstruction actually annihilates a linear field.

One real discrepancy to resolve during the extraction:
[`accel.py:152`](src/warpSPH/modules/crk/accel.py#L152) uses
`matmul(gradV_j, x_ij)` while [`dudt.py:135`](src/warpSPH/modules/crk/dudt.py#L135)
uses `matmul(gradV_j, -x_ij)`. Eqs. (11)–(12) use the same `x^T_ab` on both sides
with opposite signs on the term, so the two call sites cannot both be right.

## 2.3 Finding: four enum members are stubs that raise

[`wrapper.py:44`](src/warpSPH/modules/shockCapturing/wrapper.py#L44) —
`Balsara1995`, `Colagrossi2004`, `MorrisMonaghan1997` and `Rosswog2000` are
`ViscositySwitch` members with no implementation; selecting one raises
`ValueError`. They are reachable from the CLI, because
[`io/parsers.py:19`](src/warpSPH/io/parsers.py#L19) `parseViscositySwitch`
matches on the member *name*.

Phase 2 replaces `Rosswog2000` — and **renames it `Rosswog2020`**, since the
target is the 2020 entropy-trigger paper, not Rosswog et al. (2000).

## 2.4 Finding: there is no constant-β mode

[`pi.py:70-73`](src/warpSPH/modules/dissipation/pi.py#L70):

```python
C_l = scalar_t(1.0)/scalar_t(2.0) * (alpha_i + alpha_j) * C_l_
C_q = scalar_t(1.0)/scalar_t(2.0) * (alpha_i + alpha_j) * C_q_
if viscosityParams.scaleBeta:
    C_q = C_q * C_l
```

So:

- **Default (`scaleBeta=False`): `β_eff = ᾱ · C_q`.** The quadratic coefficient
  is *already* proportional to the switched α. The repo ships Chen & Nixon's
  recommendation as its only behaviour and cannot express PHANTOM's constant β.
- **`scaleBeta=True`: `β_eff = ᾱ² · C_l · C_q`.** A second α factor. This matches
  no paper and the flag's own comment ("the quadratic viscosity term is scaled by
  the linear viscosity term") describes the *first* branch, not this one.

Consequences, both of which move work out of Phase 5A and into Phase 1:

1. **Every port in Phases 4, 5B and 6 is currently mis-specifiable.**
   García-Senz fixes `β = 2`, Sphenix fixes `β_V = 3`, Wadsley fixes `β = 2` —
   all constants, none α-scaled. Reproducing any of those papers requires a fixed-β
   policy that does not exist.
2. **Phase 5A is inverted.** The sketch framed it as "implement β(α)". The actual
   work is implementing *fixed* β as the baseline, then measuring the existing
   β(α) default against it. The implementation is nearly free; the measurement is
   the deliverable.

## 2.5 Naming and state hazards to clear before any new code

| Hazard | Where | Fix |
|---|---|---|
| `correctXi` / `sphKernel_xi` is a **kernel-normalisation** ξ, unrelated to C&D's Ξ limiter | [`pi.py:60`](src/warpSPH/modules/dissipation/pi.py#L60) vs [`CullenDehnen2010.py:142`](src/warpSPH/modules/shockCapturing/CullenDehnen2010.py#L142) | rename one; Wadsley Eq. (28) introduces a *third* ξ |
| `limitXi` declared **twice**; the first (`False`) is dead | [`viscositySwitchParameters.py`](src/warpSPH/configurations/moduleConfigurations/viscositySwitchParameters.py) — flagged in its own docstring | delete the dead one |
| `entropies` and `soundspeeds` carry the **same tag**; `pressures` is tagged `'damping'` | [`compressibleMonaghan.py:37-40`](src/warpSPH/systems/compressibleMonaghan.py#L37) | fix; **this closes PESPH_PLAN §2.1** |
| `crkSPH.py`'s `updateViscositySwitch` call is **commented out**, so `alpha0` never advances under CRKSPH | `schemes/crkSPH.py` ~line 339 | decide: restore or delete with a note |
| `switchState.dvdt_diss` declared, never populated | [`switchState.py:28`](src/warpSPH/modules/shockCapturing/switchState.py#L28) | populate (Group B) or remove |
| Compressible probes live in untracked `.tmp/` | `.tmp/probe_cullen_sod.py`, `probe_sod_nd.py`, `probe_gresho_control.py` | promote to `scripts/` (Phase 0) |

## 2.6 The AD constraint on every new kernel

The `gradcheck` skill covers `modules/dissipation` and `modules/shockCapturing`.
Two traps are already documented in the source and both are hit by the detectors
in this plan:

- **Loop-carried `wp.max` silently zeroes adjoints.** warp-lang 1.15.0's
  reverse-mode AD drops the adjoint of a max reassigned inside a neighbour loop.
  [`wp_vsig.py`](src/warpSPH/modules/shockCapturing/wp_vsig.py) solves it with an
  argmax pass plus a single re-evaluation — **read that docstring before writing
  any new max-over-neighbours term.** Rosswog Eq. (22) and Wadsley's `v_sig` both
  need it.
- **Un-guarded divisions must return early, not be NaN-patched afterwards.**
  Reverse AD differentiates the expression that was evaluated, not the value it
  was overwritten with. See the long note at
  [`limiter.py:88-104`](src/warpSPH/modules/crk/limiter.py#L88). Every limiter
  ratio in this plan (`F_ab`, `R_i`, `D/|T|`, `S/(c²+S)`) is one of these.

`tests/test_gradcheck_scripts.py` globs `scripts/gradcheck_*.py`, so a new script
is picked up automatically — no list to edit.

---

# Part 3 — The phases

Each phase: **Grounding · Build · Unit tests · Validation · Markers**. A phase is
done when every marker box is ticked and its Part 0 rows are filled in the report.

## Registering a new switch — the seven touch points

Referenced by Phases 2, 5B and 6 rather than repeated in each.

1. `ViscositySwitch` member — [`enumTypes.py:43`](src/warpSPH/enumTypes.py#L43).
   Give it a docstring comment naming the paper; `DensityDiffusionScheme`
   ([`enumTypes.py:221`](src/warpSPH/enumTypes.py#L221)) is the house style and
   `ViscositySwitch` currently has none of it.
2. `modules/shockCapturing/<Name>.py` implementing
   `compute<X>Terms(dt, particleState, simulationConfig, schemeConfig, supportScheme, adjacency) -> (alphas, ViscositySwitchState)`
   and `compute<X>Update(switchState, dt, dvdt, …) -> (alpha0s, ViscositySwitchState)`.
   A pass-through update is acceptable (C&D and R&H both do it).
3. Register in **both** dispatch functions in
   [`wrapper.py`](src/warpSPH/modules/shockCapturing/wrapper.py) — after Phase 1
   this is one registry entry instead of two `elif` branches.
4. New tunables → `ViscositySwitchConfig`, **and** both
   `viscositySwitchConfigToDict` and `dictToViscositySwitchConfig`, the latter
   using `.get(…, default)` so old HDF5/YAML still loads (that is what `ns` and
   `balsara_const` do).
5. New per-particle state → a field on `CompressibleState`
   ([`compressibleMonaghan.py:37`](src/warpSPH/systems/compressibleMonaghan.py#L37))
   **and** a copy in `CompressibleSystem.finalize` (`:93-101`). Omitting the copy
   silently resets it every substep. Same for `CompSPHState`.
6. New update-rate output → extend `ViscositySwitchState` and consume it in the
   scheme, following the `dudt_entropy` pattern at `monaghan.py:194-199`.
7. Verify it is reachable: `cases/compressible.py` `viscositySwitch` param,
   `io/parsers.py` name match, `runner/report.py:182` banner line.

---

## Phase 0 — Freeze the baseline · **M0**

### Goal
Know exactly what the code does today, in numbers, and be able to reproduce them.

### Build
- `L1` in [`benchmarks/common/metrics.py`](benchmarks/common/metrics.py), beside
  `relL2`. García-Senz Eq. (20), mean absolute, masked.
- Extend `compressibleDiagnostics`
  ([`cases/compressible.py:71`](src/warpSPH/cases/compressible.py#L71)) with:
  `entropy` (Σ m_i s_i, `s = P/ρ^γ`), `angularMomentum` (port
  `_conservationMetrics` from
  [`oscillatingDroplet.py:196`](src/warpSPH/cases/oscillatingDroplet.py#L196)),
  `alphaMean`, `alphaMax`, `alphaActiveFraction` (`α > 0.1`), `avPowerLinear`,
  `avPowerQuadratic`. All 14 compressible cases pick these up automatically.
- Populate `switchState.dvdt_diss`; add the twice-evaluated `Π` split (Group B).
- Record `adjacency.numNeighbors` mean/min/max on the trajectory row.
- Promote `.tmp/probe_cullen_sod.py`, `.tmp/probe_sod_nd.py`,
  `.tmp/probe_gresho_control.py` → `scripts/probe_sod1D.py`,
  `scripts/probe_sodND.py`, `scripts/probe_greshoControl.py`.
- Write `scripts/av_report.py` (§0.1).

### Reference configuration
Record and freeze, from
[`cases/compressible.py`](src/warpSPH/cases/compressible.py): kernel `B7`,
`integrationScheme='rungeKutta2'`, `supportMode='KernelMeanSymmetric'`,
`n_h=4.0`, `cflFactor=0.3`, `adaptiveDt=True`, `gamma=5/3`,
`adaptiveSupportScheme='Owen'`; diffusion
`buildDefaultDiffusionParamsCompressibleSPH` (`C_l=1`, `C_q=0`,
`viscosityTerm=Price2012_98`, `thermalConductivityTerm=Price2008`); host scheme
`Monaghan`. Three switch settings: `NoneSwitch`, `CullenDehnen2010`,
`ReadHayfield2012`.

> `C_q = 0` in the compressible default — the reference baseline currently has
> **no quadratic viscosity at all**, which is itself a finding for Phase 5A.

### Validation
- `scripts/run_tests.sh` — 89 pass.
- `python -m pytest tests/test_gradcheck_scripts.py` — all pass.
- `scripts/run_sweep.py` — every case, 5 steps, green.
- Full report run over `sod`, `sod2d`, `sod3d`, `sedov` (1/2/3D), `noh`,
  `gresho`, `yee`, `linearWave`, `kelvinHelmholtz`, `rayleighTaylor`.

### Thresholds
- **Reproducibility lock (§0.4):** two identical runs agree `< 1e-6` relative on
  every scalar in `results.json`.
- Every existing assertion in `tests/test_physics.py` and
  `tests/test_shockCapturing.py` still holds, unmodified.
- No new metric is *gated* in this phase — they are recorded as the reference.

### Markers
- [x] `L1` lands and is unit-tested against a hand-computed case (2026-09-30, `tests/test_bench_metrics.py`)
- [x] `compressibleDiagnostics` extended; 35-case smoke sweep + full test suite green after the change (2026-09-30)
- [~] linear/quadratic AV split populated (non-zero on `sod` only for `C_q>0` hosts; Monaghan default is exactly 0 — see notes)
- [x] three probes promoted out of `.tmp/`
- [x] `scripts/av_report.py --config baseline --profile full` produces
      `results/av_baseline_<stamp>/report.md`
- [x] reproducibility lock verified twice (10 cases x 3 configs + C_q=2 family)
- [x] report committed (`docs/av/av_baseline_2026-09-30.md` is in git)

**M0 — Reproducible.** Git milestone `milestone/av-baseline`.

---

## Phase 1 — Separate the dissipation architecture · **M1**

The most important software-engineering phase. Four axes, independently
replaceable:

```
ShockDetector ──▶ CoefficientPolicy ──▶ VelocityPairPolicy ──▶ DissipationOperator
   (when?)            (how much?)          (of what Δv?)            (how?)
```

### Build
- **Registry instead of dispatch.** Replace both `if`/`elif` chains in
  [`wrapper.py`](src/warpSPH/modules/shockCapturing/wrapper.py) with a
  `dict[ViscositySwitch, (termsFn, updateFn)]`. Adding a detector becomes one
  entry, not two branches that can diverge.
- **`CoefficientPolicy`** — new. `Fixed(α, β)` · `SwitchedAlphaFixedBeta(β)` ·
  `SwitchedAlphaCoupledBeta(C_β)`. Resolves §2.4: `SwitchedAlphaFixedBeta` is
  what every paper in Phases 4/5B/6 specifies and is **currently
  inexpressible**. Deprecate `scaleBeta` behind the new policy; keep the flag
  reading through to `SwitchedAlphaCoupledBeta` for config compatibility, with
  its double-α semantics documented as the historical behaviour.
- **`VelocityPairPolicy`** — an abstraction over `v_i, v_j` reaching
  `computePi_actual`. Phase 1 ships only `RawVelocity`, which must be a literal
  no-op. Phase 3 fills in the rest.
- **`DissipationOperator`** — the existing `ViscosityTerms` dispatch in
  [`pi.py`](src/warpSPH/modules/dissipation/pi.py), given the pair velocity as an
  argument rather than reading `v_i - v_j` internally
  ([`pi.py:78`](src/warpSPH/modules/dissipation/pi.py#L78)).
- **Clear every hazard in §2.5**, including the `entropies`/`pressures` state
  tags (closes PESPH_PLAN §2.1) and a decision on `crkSPH.py`'s commented-out
  update call.

### Unit tests
- Each policy in isolation: `Fixed` returns its constants; `SwitchedAlphaFixedBeta`
  leaves `C_q` untouched as α varies; `SwitchedAlphaCoupledBeta(2)` gives
  `β = 2ᾱ`.
- `RawVelocity` returns exactly `v_i, v_j` — identity, not approximately.
- Registry completeness: every `ViscositySwitch` member either resolves or raises
  a *named* `NotImplementedError` saying which phase implements it (replacing
  today's generic `ValueError`).

### Validation — this is a refactor, so the bar is regression, not improvement
- **`NoneSwitch`: bit-for-bit identical** to the Phase-0 baseline. Not "within
  tolerance" — identical, since `RawVelocity` and `Fixed` are no-ops.
- **`CullenDehnen2010` and `ReadHayfield2012`: within `1e-6` relative** on every
  Part 0 metric. Any drift beyond that is a bug in the extraction.
- 89 tests + all gradchecks + `run_sweep.py` green.
- Old `caseSpec.json` / HDF5 from Phase 0 still load and reproduce (that is what
  touch point 4's `.get(…, default)` protects).

### Markers
- [x] registry replaces both `elif` chains (S2 step 4, 2026-09-30)
- [x] `CoefficientPolicy` with all three variants; **fixed β now expressible** (S2 step 3, as `BetaMode` + switch; 2026-09-30)
- [x] `VelocityPairPolicy` with `RawVelocity` proven to be an identity (2026-10-07: `modules/reconstruction/pairVelocity.py` `rawPairVelocity`, `tests/test_velocityPairPolicy.py`, bitwise)
- [x] `computePi_actual` takes the pair velocity as an argument (2026-10-07: new `computePi_pair(..., u_ij, ...)`; `computePi_actual` is the thin `RawVelocity` wrapper, so no call site changed)
- [x] §2.5 hazards cleared; PESPH_PLAN §2.1 cross-referenced as closed (S2 steps 1-2)
- [x] `NoneSwitch` bit-for-bit vs M0 (S2 exit check; re-checked 2026-10-07 against M0g)
- [x] C&D and R&H within `1e-6` vs M0 on every report metric (bit-identical, tol 0: S2 exit check; 2026-10-07 smoke 30/30 vs `results/av_smoke_ref_M0g`)
- [x] gradchecks green (`gradcheck_dissipation` 2026-10-07; full `pytest tests` green)

**As built (2026-10-07).** The policy is not a config field yet: with only `Raw` it would be a setting nothing reads. Phase 3 adds the `velocityPairPolicy`
field on `DiffusionParameters` (dict round-trip with `.get`, like `betaMode`) together with the first consumer. Today the CRK call sites still pass already-reconstructed
`v_i`, `v_j` into `computePi_actual`; Phase 3 moves them onto `computePi_pair` with the policy supplying `u_ij`.
### Phase 1 clean-up pass (2026-10-07, user: "fix crksph, sweep the Pi equations")

**1. Pi formulations vs the papers** -- `tests/test_piFormulations.py` evaluates every `ViscosityTerms` member on random approaching pairs (unequal h, c, rho, alpha)
and compares the pair coefficient with an independent NumPy transcription of the paper's equation (Monaghan 2005 Eqs. 8.3-8.12, Price 2012 Eqs. 98/101/103, Marrone 2011 Eq. 5,
Wadsley 2017 Eqs. 17-18, Price 2008 conductivity). Result of the sweep:

| Formulation | Source | Before | After |
|---|---|---|---|
| `Monaghan1992` (CompSPH default) | Price 98 / Monaghan 8.10 | correct (one-sided c_i, c_j by design; averaged over the two `useJ` calls) | unchanged |
| `Price2012_98` (Monaghan-scheme default) | Price 101+103 | correct, but **mislabelled**: it is the signal-velocity form (alpha = C_l, beta/2 = C_q), not Eq. (98) (that is `Monaghan1992`) | unchanged, comment fixed |
| `Price2012`, `Monaghan1997a` | Price 103, Monaghan 8.11-8.12 (K = 1/2) | correct | unchanged |
| `Price2008` (conductivity) | Price 2008 `sqrt(|dP|/rho_bar)` | correct | unchanged |
| `MonaghanGingold1983` | Monaghan 8.3-8.4 | **wrong**: no 1/r (units off by a length) and **alpha (`C_l`, the switch) ignored** | fixed |
| `Cleary1998` | Monaghan 8.8-8.9 | **wrong**: extra 1/rho from `K/rho`; alpha applied twice (`alpha_i * C_l` with `C_l` already the pair mean) | fixed |
| `DeltaSPH` | Marrone 2011 Eq. (5) | **wrong**: scaling factor `h/xi` lacks the 1/r (same defect as MG1983) | fixed |
| `Default` | "Monaghan1992 with all bars" | same missing 1/r | fixed |
| `Wadsley2008` | Wadsley 2017 Eqs. 17-18 | **wrong**: `v_sig = C_l |w|`, no c_bar term (a quadratic term only) | now Eqs. (17)-(18) |
| `Monaghan1997b`, `Dukowicz` | Monaghan 1997 (4.7)/(4.8) | not on disk, **not verified** | unchanged |

None of the five wrong branches is selected by a shipped scheme (the schemes use `Price2012_98` and `Monaghan1992`), so no number in any baseline moved; this
is checked by the smoke bit-compare below. `C_q` means different things per family (beta for the mu-form `Monaghan1992` / `Default` / `Wadsley2008`; beta/2 for the
signal-velocity forms), as documented in `pi.py`. The coefficient policy now lives in one `switchedCoefficients` function shared by `computePi_pair` and CRKSPH.

**2. CRKSPH and the coefficient policy.** CRK's viscosity (momentum `accel.py` and energy `dudt.py`) is the Frontiere Eq. (69) one-sided `Q_i`, built from raw `C_l`, `C_q`:
it ignored the switched alpha and `BetaMode` completely (the `computePi_actual` results next to it were dead code), so **the viscosity switch had no effect on CRKSPH**
and `crkCullenDehnen2010` matched `crkNone`. Fixed 2026-10-07: `Q_i` / `Q_j` take their own particle's switched alpha and `BetaMode` through the shared
`switchedCoefficients`. `crkNone` is bit-identical before/after this fix alone (alpha = 1; smoke 4/4, `results/av_crk_{before,after}_*`). On its own it changed `crkCullenDehnen2010` only slightly,
because of item 3.

**3. CRK `alpha0s` is never advanced (found 2026-10-07, corrects the S2 note).** The S2 note said C&D / R&H under CRKSPH respond "instantly, with no decay memory". They do not: `alpha0s`
starts at 1 and `computeCullenTerms` returns it decayed by ONE step; with the `updateViscositySwitch` call deleted (S2) the stored value never changes, so
alpha is stuck at `1 - dt/tau (1 - alpha_loc)` ~ 0.97-0.98 everywhere (full-profile `results/av_crkswitch_*`: C&D alphaMean 0.97 (Gresho) / 0.98 (KH) / 0.98 (Sod), KH A(1.5) 0.0904 vs `crkNone` 0.0903).
Under CRKSPH, C&D / R&H were therefore effectively fixed alpha = 1; every `crkCullenDehnen2010` number in M0f (and the S4 numbers) is a fixed-alpha run.
**Fixed 2026-10-07 (reverses the S2 deletion): `schemes/crkSPH.py` calls `updateViscositySwitch` again**, as the Monaghan scheme does. Result (full profile, video on, no alarms; evidence and
frames in [`docs/av/crk_switch_2026-10-07/`](docs/av/crk_switch_2026-10-07/README.md)):

| full profile | crkNone (alpha = 1) | crkCullenDehnen2010 (live switch) |
|---|---|---|
| Sod L1(v_x); alphaMean / Max | 0.00669; 1 / 1 | 0.00840; 0.073 / 0.81 |
| Gresho L1(v_phi); peak speed | 0.0366; 1.016 | 0.0276; 1.067 |
| Gresho alphaMean / Max | 1 / 1 | 0.027 / 0.14 |
| KH A(1.5) (reference 0.148); alphaMean / Max | 0.0901; 1 / 1 | **0.1232**; 0.047 / 0.73 |

C&D now does what it is for: alpha ~ 0.03-0.07 over smooth flow (Gresho L1 -25 %), KH grows 37 % more than at fixed alpha = 1. Sod is slightly worse than fixed alpha (L1 +26 %, contact spike
P +5 %), the usual price of a lower floor. The Gresho peak speed above 1 is CRK's intrinsic spin-up (CRKSPH_LIMITER_PLAN), damped less now. **The CRK baseline columns (`results/av_M0f_crk`,
`crkCullenDehnen2010`) are superseded by `results/av_crk3_*`**; `crkNone` is unchanged (re-run agrees: Sod 8e-4, KH 2e-3 relative; Gresho 5 % in L1: that is the case's sensitivity, not formula drift -- in float64 a 1e-13 relative perturbation of the CFL factor moves the new code's own Gresho final state
by 2e-3 (v) / 4e-4 (x), the same size as the old-vs-new gap of 1.3e-3 / 3.6e-4, so CRK Gresho at t = 3 amplifies any perturbation by ~1e10; this bounds formula drift at that floor, it does not show 1e-10 agreement,
which is why the Eq. (69) reference test in `tests/test_piFormulations.py` exists). Regression test: `tests/test_crkSwitch.py`.

**4. `pi` is now a package** (`modules/dissipation/pi/`: `coefficients.py`, `pair.py`, `terms.py`, `dispatch.py`, `oneSided.py`). The three parallel `elif` chains of the old `pi.py`
(`v_sig`, `compute_mu_ij`, `compute_bars`) are one function per formulation over a `PairData` struct, with `pick` replacing `compute_bars` (the bars are formed once, inside `buildPair`).
New `ViscosityTerms.Frontiere2017` (CRK's one-sided `Q`, `h` own smoothing length, `eps^2 = 1e-2`, `min(0, .)`) with `computeFrontiereQ` as its CRK entry point;
the duplicated `Q` blocks in `crk/accel.py` and `crk/dudt.py` are gone. Checked: `tests/test_piFormulations.py` (paper references, now incl. Frontiere Eq. 69), Monaghan-host smoke 30/30 bit-identical to M0g,
CompSPH default (no switch) final states bit-identical to HEAD on Sod and Gresho, gradchecks. **One honest caveat:** with a *live* switch (alpha != 1) the compiler fuses `C_l c - C_q mu`
differently now that each formulation is its own function, so e.g. CompSPH + C&D differs from the pre-refactor code at float32 round-off (2e-6 relative at 60 steps, 5e-5 at 300 steps on Sod); no stored baseline
contains CompSPH with a switch.

**M1 closed 2026-10-07:** tag `milestone/dissipation-abstraction` created (4e662a1, local); CRKSPH's viscosity now honours the switched alpha and `betaMode` (clean-up item 2). Nothing outstanding for M1.

**M1 — Modular.** Git milestone `milestone/dissipation-abstraction`.

---

## Phase 2 — Rosswog (2020) entropy trigger · **M2**

Deliberately conservative: **existing Monaghan operator + new trigger, nothing
else changes.** First genuinely new physics in the plan.

### Grounding
§1.1, Eqs. (15)–(20). Constants: `l₀ = log(1e−4)`, `l₁ = log(5e−2)`,
`α_max = 1`, `α₀ = 0`, decay `30 τ_a`.

### Build
Seven touch points. Specifics:

- Rename `ViscositySwitch.Rosswog2000` → `Rosswog2020` (§2.3), keeping the enum
  value so stored configs still resolve; add a name alias in
  [`parsers.py`](src/warpSPH/io/parsers.py) for the old spelling.
- `modules/shockCapturing/Rosswog2020.py`.
- **`entropiesPrev` on `CompressibleState`, copied in
  `CompressibleSystem.finalize`.** Touch point 5, and the single most likely bug
  in this phase.
- **Eq. (16) is a difference between *time steps*, not RK stages.** The trigger
  must read the value at `t^{n−1}`, which means capturing it at step boundaries
  in the system's `finalize`, not inside the RHS. Getting this wrong makes the
  trigger read stage noise and is exactly the class of error already recorded
  against the WC density update. Write the test in §Unit before the kernel.
- `α_min`/`α_max` already exist on `ViscositySwitchConfig`; add `l0`, `l1`,
  `decayTau` (default 30.0), with dict round-trip.

**As built (2026-10-06).** Departures from the list above, with reasons:

- **New member `ViscositySwitch.Rosswog2020 = 8`, no rename.** S3 implemented `Rosswog2000` as the
  real Rosswog et al. (2000) divergence-source switch, so it is no longer a stub to replace.
- **Where the trigger runs.** The integrator clones `constant` fields into every stage, so no stage
  can see another's values, and the step-boundary entropy `s^n = s(u^n, rho(x^n))` exists only in the
  RHS state of stage 0 (later stages run on predictor states). The trigger therefore runs once per
  step in the systems' `finalize` (`wrapper.advanceViscositySwitchStep` -> `STEP_HOOKS` ->
  `advanceRosswog2020`), from `returnValues[0]`'s entropies, `h`, `c`. The stage-level terms function
  only passes the stored alpha through, so alpha is constant over a step's stages. **Consequence: a
  one-step lag** -- alpha^n (from s^n, s^{n-1}) is used from step n+1, where the paper uses it in
  step n. Wired into both `CompressibleSystem` (Monaghan) and `CompSPHSystem` (CompSPH, CRKSPH).
- State: `entropiesPrev`, `entropiesPrevTime` (Eq. 16's `Delta t = t^n - t^{n-1}` is taken from them,
  so it is right under adaptive dt) and `entropyRates` (epsdot of the last step, for the §0.3 map)
  on `CompressibleState` and `CompSPHState`. Set in `finalize`, not copied from a stage.
- Config: `entropy_eps0 = 1e-4`, `entropy_eps1 = 5e-2` (the thresholds themselves rather than their
  logs; Eq. 20's x does not depend on the log base), `entropy_decay = 30`; `alpha_0` is `alpha_min`.
- `h = support / 2`: the paper's kernels reach out to 2h (its footnote 2), and its thresholds were
  calibrated with that h. (The other switches use `support / sphKernel_xi`.)
- First step: records s^0 and sets alpha to `alpha_0` (case ICs start every switch at alpha = 1, which
  with a 30 tau decay would dissipate for a large part of a Sod run). The step-0 RHS still runs at the IC's alpha.
- Decay integrated exactly (`alpha_0 + (alpha - alpha_0) exp(-dt/(30 tau))`), then `max` with alpha_des.
- No warp kernel (all torch, outside the RHS), so no gradcheck extension; the trigger is not on the AD
  path at all (it runs in `finalize` on detached entropies).

### Unit tests — `tests/test_rosswogTrigger.py`
- `S(0) == 0`, `S(1) == 1`, `S'(0) == S'(1) == 0` (quintic smoothstep), monotone
  increasing on `[0,1]`.
- `α == 0` for `ε̇ ≤ 1e−4`; `α == α_max` for `ε̇ ≥ 5e−2`; strictly monotone
  between, on a manufactured `ε̇` sweep.
- **Step-boundary test:** a manufactured state where `s` is constant across a
  step but varies across RK stages must give `ε̇ == 0`. This is the guard for the
  stage-vs-step hazard above.
- Eq. (16) is dimensionless: scaling `Δt` and `τ_a` together leaves `ε̇`
  unchanged.
- Decay: with the source off, `α` halves in `30 τ ln2`, to `1%`.
- `scripts/gradcheck_shockCapturing.py` extended to cover the new module.

### Validation
Host scheme Monaghan. Compared against C&D and NoneSwitch at the M0 baseline.

| Case | Threshold | Anchor |
|---|---|---|
| `sod` 1D | `abs(P_spike) < 0.5`, `abs(A_spike) < 0.5`; `alpha.max() > 0.3`; `alpha.mean() < 0.3` | the existing `test_shockCapturing.py` acceptance, reused verbatim |
| `sod` 1D | `L1(v_x) ≤` the C&D value at M0 | garciasenz2026 Eq. (20) |
| `sod` 2D/3D | contact spike ≤ NoneSwitch on both `P` and `A` | the R&H acceptance pattern |
| `sedov` 3D | peak `ρ/ρ₀` approaches 4 **from below**, never overshoots | garciasenz2026 Fig. 2; rosswog2020entropy Fig. 3 shows the trigger matching exact |
| `noh` 1D | post-shock `ρ_s` within ±3% of `ρ₀((γ+1)/(γ−1))^dim` | [`cases/noh.py`](src/warpSPH/cases/noh.py) |
| `gresho` | `alphaMean < 0.05` **and** `L1(v_φ) ≤` C&D's | the paper's own smooth-flow number is `α ~ 0.01`; `0.05` is the gate |
| `kelvinHelmholtz` | `A(t)` ≥ the C&D value | "only a very moderate (and desired) switch-on in instability tests" |

**Two tests the sketch called for in prose and that the paper's construction
makes sharp:**

- **Resolution dependence.** Sweep `nx ∈ {100, 200, 400}` on `gresho`. The
  smooth-region `alphaMean` must **not grow** with resolution. A trigger that
  sharpens into noise as `Δx → 0` fails here.
- **Timestep dependence.** `cflFactor ∈ {0.15, 0.3}` on `sod` 1D at fixed `nx`.
  The α field must agree to `< 10%` in `L1`. Eq. (16) is non-dimensionalised by
  `τ_a/Δt` *precisely so this holds*, so it is a direct test of the transcription.

### Markers
- [x] `Rosswog2020` registered as a new member (2026-10-06; `Rosswog2000` is a real switch since S3, nothing to rename)
- [x] `entropiesPrev` on the state, set at the step boundary in `finalize` (2026-10-06)
- [x] step-boundary unit test passes before the physics is tuned (`test_step_boundary_reads_stage0_only`)
- [x] all `S(x)` / threshold / dimensionless / decay unit tests pass, plus Sod wiring on Monaghan and CompSPH (`tests/test_rosswogTrigger.py`, 2026-10-06)
- [x] gradcheck: n/a (no warp kernel; the trigger runs in `finalize`, off the AD path)
- [~] Sod 1D/2D/3D table above (2026-10-06): 1D passes; 2D/3D P spike marginally above none (+1.8 % / +0.3 %)
- [x] Sedov approaches 4 from below (2.53 / 2.09 at C_q 0 / 2, sharper than C&D's 2.15 / 1.83)
- [x] Gresho `alphaMean < 0.05`: 0.047 at nx 100 (L1 0.077 vs C&D 0.073)
- [x] resolution sweep: `alphaMean` non-increasing (Gresho nx 50/100/200: 0.074/0.047/0.029)
- [x] timestep sweep: α within 10% (Sod cfl 0.3 vs 0.15: 6.0 %)
- [x] detector map (§0.3) rendered vs C&D on Sedov (`scripts/probe_rosswogSedovMap.py`, `docs/av/av_phase2_sedov_alpha_map.png`): alpha ~ 1 everywhere, cold-gas saturation ahead of the shock
- [~] **Kelvin-Helmholtz: A(1.5) 0.012 vs C&D 0.098 at nx 128, sharp IC -- fails.** Explained 2026-10-07: the trigger fires on the sharp-IC density *contact* (no-shear control identical), and residual noise is the resolution floor. Smooth IC (`smoothDensity`) 0.037 at nx 128; **nx 256 smooth IC: 0.131 vs C&D 0.137** (gate `>= C&D` missed by 4.6 %), `docs/av/kh256_2026-10-07/`
- [x] report row added: `docs/av/av_phase2_rosswog2020_2026-10-06.md`, configs `rosswog2020`, `rosswog2020Q` (group `phase2`)

### Results (2026-10-06)
Full write-up and tables: [`docs/av/av_phase2_rosswog2020_2026-10-06.md`](docs/av/av_phase2_rosswog2020_2026-10-06.md),
at the default `supportVolumeClamp='walls'` against M0e (valid again, see Notes). Passes Sod 1D, Sedov (2.53 vs C&D 2.15),
Noh (C_q 2), Gresho alphaMean (0.047), both sweeps (dt 6.0 %; Gresho 0.074/0.047/0.029), linear wave / Yee (better than
C&D). **Fails Kelvin-Helmholtz** (A(1.5) 0.012 vs C&D 0.098, below even fixed alpha = 1): the trigger fires on the
density-contrast shear layer, most likely standard SPH's contact entropy noise, which MAGMA2's reconstruction suppresses.
Marginal: Sod 2D/3D P spike vs none (+1.8 % / +0.3 %), Gresho L1 6 % above C&D. Cold gas (Sedov's u = 0 background)
saturates the trigger at alpha = 1 (tau = h/c infinite).
**KH mechanism checked (2026-10-07, `scripts/probe_rosswogKHMap.py`, maps `docs/av/av_phase2_kh_map.png`, control `docs/av/av_phase2_kh_map_noshear.png`):**
the trigger fires on the **density contact, not the shear**. At t = 0.05, before any billow, epsdot ~ 0.1-1 and alpha ~ 1 along both rho 1:2
interfaces (92 % of particles within |y - 0.25|, |y - 0.75| < 0.05 have epsdot > eps_0, vs 29 % elsewhere; layer-mean alpha 0.56 vs 0.03 outside).
The **no-shear control** (v1 = v2 = 0, no seed) gives the same numbers (100 % / 0.56 at t = 0.05; layer alpha 0.19 vs 0.18 at t = 0.5), so shear plays no part.
The layer relaxes (alpha 0.56 -> 0.38 -> 0.18 at t = 0.05 / 0.2 / 0.5) but stays hot. Likely cause (not yet proved): the sharp IC's summation density is
smoothed over the interface while u is set from the sharp profile, so s = P/rho^gamma is wrong there at t = 0 and drifts as the contact relaxes.
Outside the layers 15-48 % of particles also sit above eps_0 (lattice-line stripes in the map; the low-resolution floor of finding 2). The t = 1.5 snapshot was missed (last step ended just short).
**Smooth-density IC test (2026-10-07, user's suggestion; Frontiere 2017 Eq. 100 / Robertson 2010: density ramps over `delta = 0.025` like `v_x`).**
New opt-in case param `smoothDensity` (`cases/kelvinHelmholtz.py`, `sampleKHH(..., smoothDensity=False)`; default 0 = the sharp step, so M0 and every stored KH config are unchanged -- the
smoothed assignments were dead commented-out code before). Maps: `docs/av/av_phase2_kh_map_smooth.png`; reports `docs/av/av_phase2_kh_smooth_{baseline,rosswog}_report.md`.
KH A(1.5), nx 128, Monaghan (McNally reference 0.148):

| IC | none (alpha = 1) | Cullen-Dehnen | Read-Hayfield | Rosswog 2020 (C_q 0) | Rosswog 2020 (C_q 2) |
|---|---|---|---|---|---|
| sharp (M0e) | 0.0204 | 0.0984 | n/a here | 0.0123 | n/a here |
| **smooth** | 0.0184 | 0.1064 | 0.0713 | **0.0372** | 0.0318 |

- The trigger map confirms the transient was a large part of the failure: at t = 0.05 layer-mean alpha 0.56 -> **0.23** (outside the layers 0.03 -> 0.001), and the layer is nearly
  quiet by t = 0.5 (alpha 0.045, 28 % of layer particles above eps_0, vs 72 % on the sharp IC). Rosswog's A(1.5) improves **3x** (0.012 -> 0.037) and now beats fixed alpha = 1; C&D barely moves (0.098 -> 0.106).
- **It does not close the gate**: A(1.5) is still 0.037 vs C&D 0.106 (the gate is `>= C&D`), and 25 % of the way to the reference. At t = 1.49 the roll-up has only just started, alpha
  in the layer is 0.3-0.8 and mottled, epsdot 0.01-0.1 across a still-laminar layer, alphaMax 0.82 vs C&D 0.55, alphaMean 0.084 vs 0.061. So a residual entropy noise on the interface remains -- the contact error of
  summation density (finding 1) rather than only the IC transient. Not yet understood past that; no threshold tuning (rule: derive, don't tune).
- **Resolution check (2026-10-07, smooth IC, `probe_rosswogKHMap.py --smooth --nx N`; maps `docs/av/av_phase2_kh_map_smooth{,_nx64,_nx256}.png`):** the residual interface noise is the resolution floor, not an intrinsic defect.

  | nx | layer-median epsdot (t ~ 0.5) | layer-mean alpha (t ~ 0.5) | A(0.49) | A(1.49) |
  |---|---|---|---|---|
  | 64 | 8.0e-4 | 0.393 | 0.0070 | 0.0136 |
  | 128 | 4.9e-5 | 0.047 | 0.0094 | 0.0373 |
  | 256 | 1.5e-5 (= the floor outside the layer) | 0.001 | 0.0120 | not run (t <= 0.5) |

  At nx 256 the trigger is off in the laminar layer (alpha 0.001) and the early amplitude is highest, so growth is no longer suppressed. **Not yet measured: A(1.5) at nx 256 against C&D at nx 256**
  (~40 min each on this host) -- that pair is the decisive Phase 2 KH number for the smooth IC.
- **nx 256, out to t = 3 (2026-10-07; evidence [`docs/av/kh256_2026-10-07/`](docs/av/kh256_2026-10-07/README.md), driver `scripts/probe_rosswogKHMap.py`, plot `scripts/plot_kh256_series.py`):**
  Rosswog A(1.5) **0.131** (nx 128: 0.037), C&D 0.137, fixed alpha = 1 only 0.037 (never develops the instability), McNally 0.148; Rosswog peaks at 0.209 (t = 2.11) like C&D (0.212). Rosswog's alpha falls to ~1e-4 in the
  laminar layer and rises on the roll-up filaments only. **The KH failure was resolution plus the sharp-IC contact transient, not the trigger**; strictly the gate is still missed by 4.6 % (smooth IC, nx 256; the
  sharp IC at nx 256 was not run). **Default CRKSPH (alpha = 1) is better still**: 0.131 sharp / 0.145 smooth IC at t = 1.5 (98 % of McNally), peak 0.23-0.24, 0.17-0.19 still at t = 3 versus ~0.105 for the Monaghan runs.
- The sharp-IC column stays the plan's gate (it is the M0 reference); whether the AV report's KH case should switch to the smooth IC (re-taking every KH column) is the user's decision.
**Open (user decides how to proceed):** (a) ~~confirm the KH mechanism with an epsdot map~~ done above: the contact, not the shear; smoothing the IC removes about half of it (3x amplitude), the rest is unexplained;
(b) re-test after Phase 3 (velocity reconstruction), which is the paper's setting -- the plan's order already puts it next;
(c) the paper's own remark that non-reconstructed SPH may need other `eps_0`/`eps_1` -- only with a derivation, not a
KH-tuned threshold. M2 stays open on the KH row.

**M2 — Entropy-aware.** Git milestone `milestone/rosswog-trigger`.

---

## Phase 3 — Velocity reconstruction infrastructure · **M3**

> **Strategically the most important milestone after Phase 1.** This is where the
> AV work stops being AV work and starts being the substrate for Phase 10's
> MUSCL/Riemann states.

### Grounding
§1.2 Eqs. (11)–(17) (linear SLR, Frontiere/García-Senz) and §1.1 Eqs. (10)–(13)
(quadratic, Rosswog). **Linear first.**

### Build — extraction, per §2.2
- New `modules/reconstruction/` with three `VelocityPairPolicy`
  implementations: `RawVelocity` (from Phase 1) · `LinearReconstruction` ·
  `LimitedReconstruction`.
- Move `limiterVL`, `computeVanLeer`, `crkLimiter` out of
  [`modules/crk/limiter.py`](src/warpSPH/modules/crk/limiter.py) into
  `modules/reconstruction/limiters.py`, preserving the AD guards and their
  explanatory comments verbatim (§2.6).
- Re-express [`modules/crk/accel.py`](src/warpSPH/modules/crk/accel.py) and
  [`dudt.py`](src/warpSPH/modules/crk/dudt.py) through the new module — and
  **resolve the `x_ij` / `-x_ij` sign discrepancy** between those two call sites
  (§2.2) against Eqs. (11)–(12).
- Velocity gradient: reuse `computeShearTensor`
  ([`common.py:53`](src/warpSPH/modules/shockCapturing/common.py#L53)). It takes
  the correction matrix as its first argument — pass the `M_inv` that
  `computeDivergence` already builds via `torch.linalg.pinv`
  ([`common.py:129`](src/warpSPH/modules/shockCapturing/common.py#L129)), or
  `None` for the uncorrected gradient.
- **Optional, deferred:** García-Senz 2012 IAD gradients (rosswog2020entropy
  Eqs. 4–6). The `Σ (m_j/ρ_j) x_ij ⊗ x_ij W` tensor IAD needs is close to what
  [`wp_computeM.py`](src/warpSPH/modules/shockCapturing/wp_computeM.py) already
  builds, so this is a new consumer of an existing kernel, not a new kernel.
  Not on M3's critical path.

### Unit tests — `tests/test_reconstruction.py`
The sketch's "important progression test", made numeric:

- **Linear-field annihilation.** Sample `v(r) = A r + b` on a lattice. The
  reconstructed pair difference must vanish:
  `max|Δv'_ab| / (‖A‖ h) < 1e-5` on a regular lattice with the `M`-corrected
  gradient; `< 1e-2` on a jittered lattice. This is the whole premise of SLR —
  if it fails, Phase 4 measures nothing.
- **Limiter behaviour.** `φ_ab → 1` in smooth flow; `φ_ab → 0` across a
  manufactured 1D velocity step. `κ_ab == 1` for `q_ab ≥ q_crit`, Gaussian below.
- **Conservation.** Reconstruction must not break the pairwise operator's
  antisymmetry: `‖Σ_i m_i dvdt_diss,i‖ / Σ_i m_i‖dvdt_diss,i‖ < 1e-12`. **No such
  test exists in the repo today** and it is the property most easily lost by a
  one-sided reconstruction.
- `scripts/gradcheck_reconstruction.py` — picked up automatically by
  `tests/test_gradcheck_scripts.py` (§2.6).

### Validation
- **CRKSPH regression.** The extraction must not move CRKSPH: energy drift
  within its existing `1e-4` budget, and `sedov`/`sod` report metrics within
  `1e-6` relative of M0. If the `x_ij` sign fix *does* move CRKSPH, that is a
  separate finding — record it, quantify it, do not silently absorb it.
- Monaghan + `LimitedReconstruction` runs `sod` and `gresho` without diverging.
  No quality claim yet; that is Phase 4.

### Markers
- [x] `modules/reconstruction/` exists with three policies (2026-10-07: Raw / Linear / Limited, + BalsaraLimited for Phase 4)
- [x] limiters moved with AD guards intact (`git mv` to `reconstruction/limiters.py`, `crk/limiter.py` re-exports)
- [x] `x_ij` sign discrepancy resolved against Eqs. (11)–(12), outcome recorded (settled at S4; one shared `linearPairVelocity` now; CRK passes `J^T`, harmless, see Phase 3 notes)
- [x] linear-field annihilation test passes at both tolerances (float64: 1e-5 / 1e-2; needed a volume-consistent `M`, OPEN_PROBLEMS §22)
- [x] limiter unit tests pass
- [x] **pairwise conservation test passes at `1e-12`** (float64: momentum <= 1.5e-15, energy <= 5e-14, all policies, Sod + Gresho)
- [x] gradcheck script added and green (`scripts/gradcheck_reconstruction.py`, incl. Balsara variants)
- [x] CRKSPH regression within budget (or the delta is quantified and explained) (sweep 2026-10-08: energy drift <= 3e-5 on all cases; Group A vs the post-switch-fix reference `av_crk3_*` within 2e-4 for crkNone and CRK + C&D on Sod. R8, 2026-10-08: the large CRK + C&D deltas the verdict first showed (0.64 Sod, 112x sod2d) were against M0g, which predates the 2026-10-07 CRK switch fix (alpha ~ 1 there); `av_sweep_verdict.py` now skips that stale reference. sod2d / sod3d / Sedov / Noh under CRK + C&D had no post-fix reference: the sweep's numbers are it)
- [x] Monaghan + reconstruction runs clean (sweep 2026-10-08: Linear and Limited on sod / gresho / sedov / noh / KH, 10/10)

### Build notes (2026-10-07; validation runs are in the overnight sweep)

- **Layout.** `modules/reconstruction/`: `pairVelocity.py` (`rawPairVelocity`, `linearPairVelocity` = Eqs. 11-12,
  `limitedPairPhi`, the dispatch `reconstructPairVelocity`), `limiters.py` (moved verbatim with `git mv`;
  `crk/limiter.py` re-exports), `gradient.py` (`computeVelocityJacobian`, `balsaraFromJacobian`,
  `reconstructionInputs` -- the one place the scheme and the diagnostics get params / J / B from),
  `diagnostics.py` (`computePairPhiMean`). CRK `accel` / `dudt` call `limitedPairPhi` + `linearPairVelocity`
  (the duplicated block is gone). `DiffusionParameters.velocityPairPolicy` (+ `reconstructionEtaCrit/Fold` <= 0 ->
  1/n_h, 0.2/n_h, `correctReconstructionGradient`), dict round trip with `.get`, banner line. CompSPH and CRKSPH
  raise if a policy is set (`requireRawPairVelocity`): only the Monaghan kernels implement it.
- **Energy.** With `u'` in the force (`Pi(u') (u'.x/r) grad W`) the heating must be `Pi(u') (u'.x/r)(v.x/r)` --
  one factor reconstructed, one raw (as CRK's dudt and Garcia-Senz do); `ux^2` would not conserve. Float64:
  momentum <= 1.5e-15, energy <= 5e-14 relative, every policy, Sod and Gresho (`tests/test_reconstruction.py`).
  Heating is no longer positive-definite pair by pair (Garcia-Senz §1 discusses the same property of SLR).
- **The Jacobian (two findings).** `warpOperation`'s vector gradient is already `J` (`Vs[c,g] = dv_c/dx_g`,
  checked on `v = (y,0)`); for `v = Ax`, `Vs = A M`, so the exact correction is `Vs M^-1` -- and `computeM` sums
  `m_j` where the gradient sums `m_j/rho_j`, so `M ~ rho I`. Both break the C&D opt-in corrected path
  (`divergenceScheme='cullen'` + `correctVelocityGradient`, off by default): **OPEN_PROBLEMS §22, fixed later
  2026-10-07** (`computeM` volume-weighted, `Vs M^-1`; the reconstruction's Jacobian bit-identical across the fix).
  Float64 `J` error 8e-16, annihilation residual 1.8e-15. CRKSPH passes `J^T` to its reconstruction
  (`schemes/crkSPH.py` `.mT`); only `x^T J x` and `u.x` enter there, both transpose-invariant, so CRK is unaffected
  (kept, to not move CRK).
- **Regression vs M0g (Monaghan smoke, Raw policy): not bit-identical, round-off.** 24/30 pairs identical; the
  rest: `avEnergyLinear` 2-9e-9 (none / C&D), R&H `sod2d` `alphaMax` 2.2e-7, `sod3d` `contactSpikeP` 2.9e-6.
  HEAD reproduces M0g exactly on those six (so it is this change), and in **float64 the R&H before/after
  difference is 7.7e-12**: FMA contraction moving with code across `wp.func` boundaries (gotcha (i)), no math
  change. CRK smoke before/after (float32): energy drift unchanged within one ulp (all <= 2.8e-5, budget 1e-4);
  sod-family derived errors (`entropyGain`, `R4_*`) 2-9e-6, Gresho/Yee angular-momentum metrics up to 3.6e-3
  (chaotic amplification, gotcha (ii)); **float64 CRK before/after: <= 3e-11** on every Sod / Noh / Sedov metric, the one
  outlier (crkCD Sedov `energyDrift`, 3.6 % relative) is 3.1e-15 vs 3.0e-15 absolute -- the extraction does not move
  CRKSPH beyond round-off (the plan's 1e-6 bar holds in float64; float32 derived error metrics move at 1e-6-1e-5).
- **Float64 C&D / MM97 crashed on the host** (`1 / sphKernel_xi(...)` is `int / wp.float64`, then `Tensor * wp.float64`;
  no overloads) -- fixed 2026-10-07 (`float(type(xi)(1.0) / xi)`; in float32 the host call already returns a Python
  float, so nothing changes there), found by the float64 regression check. Pre-existing: no float64 C&D run was possible.
- `tests/conftest.py` now honours `warpSPHCore_PRECISION` (default float32 unchanged) so a test can rerun itself in
  float64 (`test_reconstruction.py::test_float64`).

**M3 — Reconstruction-ready.** Git milestone `milestone/velocity-reconstruction`.

---

## Phase 4 — García-Senz & Cabezón (2026) reconstruction AV · **M4**

The question: **how much of the unwanted viscosity comes from feeding the AV
operator raw particle velocity differences?**

### Grounding
§1.2. Operator Eq. (6) with `v_sig` Eq. (3), `γ = 3`, **`β = 2` fixed** (needs
Phase 1's `SwitchedAlphaFixedBeta`, §2.4). Balsara Eq. (8). SLR Eqs. (11)–(17),
Balsara-modulated Eqs. (18)–(19) with `p = 2`.

### Build
- `ViscosityTerms` entry for Eq. (6) if none of the existing 12 matches it
  exactly — check `Wadsley2008` ([`pi.py:169`](src/warpSPH/modules/dissipation/pi.py#L169))
  first, which is close.
- Standalone Balsara `B_a` as a `CoefficientPolicy` modulation — it exists today
  only *inside* R&H ([`ReadHayfield2012.py:163`](src/warpSPH/modules/shockCapturing/ReadHayfield2012.py#L163));
  lift it out and retire the `Balsara1995` enum stub (§2.3).
- `BalsaraModulatedReconstruction` (Eqs. 18-19) as a fourth
  `VelocityPairPolicy`, `p` configurable.
- Config matrix = Table 1 rows 1-6. Row 7 (AVSLRB2F) is the turbulence variant;
  optional.

### Validation — thresholds are the paper's own ranking

This is the phase where the paper gives usable numbers, including one where
**reproducing a failure is the validation**.

| Case | Threshold | Paper anchor |
|---|---|---|
| `sod` | `L1(v_x)` for AVSLRB2 `≤` AVSW; post-shock velocity oscillation amplitude strictly lower under AVSLRB2 than under either switch run | §3.1 — "models ST2 and ST4 (i.e. those employing switches) show the worst performance… the best performance is delivered by ST1, followed by ST6" |
| `sedov` 3D | **AVSLR (row 3) and AVSWSLR (row 4) must overshoot `ρ/ρ₀ = 4`; AVSLRB2 (row 6) must not.** | §3.2 / Fig. 2 — "models S3 and S4 overshoot the strong-shock compression limit… while the remaining four stay below and approach from below" |
| `gresho` | `L1(v_φ)`: SLR family `<` AVSW `<` plain AV, strictly | §3.3 / Fig. 4 — "clearly shows the superior performance of AVSLR schemes over AVSW… the worst case is V1" |
| `kelvinHelmholtz` | ordering `AV < AVSW < SLR family` reproduced in `A(t=1.5)`, **and** the SLR family reaches `≥ 1.4×` the AVSW amplitude | Table 3: `KH1 2.87e−2`, `KH2 8.31e−2`, `KH3 12.58e−2`, `KH4 12.14e−2`, `KH5 12.01e−2`, `KH6 12.52e−2`; McNally reference `14.79e−2` |
| `rayleighTaylor` | mode growth not *below* AVSW | §3.5 |

**Headline measurement — the phase's actual deliverable:** integrated AV energy
(Group B) on `gresho`, raw Δv vs reconstructed Δv, same detector, same β. The
claim to be confirmed or refuted is a drop of **≥ 1 order of magnitude**. Report
the linear/quadratic split separately; it feeds Phase 5A.

**Setup note.** The paper's shock tube smooths the *pressure* but not the
particle distribution: `P₀ = −A tanh((x−x_c)/δ) + B`, `δ = 0.01`, `x_c = 0.5`,
`A = 0.45`, `B = 0.55` (their Eq. 21). The repo's `smoothIC` param
(`examples/sweeps/sod_highres.yaml`) may or may not match; check before comparing
L1 values against the paper's figures rather than against each other.

### Markers
- [x] Eq. (6) operator available with fixed `β = 2` (it is `Price2012_98` + `BetaMode.Fixed`, `C_q = 2`)
- [x] standalone Balsara `B_a`; `Balsara1995` stub retired (S3 did both; `balsaraFromJacobian` + `balsaraPairLimiter` for the pair)
- [x] `BalsaraModulatedReconstruction` with configurable `p` (`VelocityPairPolicy.BalsaraLimited`, `reconstructionBalsaraPower`)
- [x] Table 1 rows 1-6 all runnable from a config name (`av_report --config phase4`: gsAV ... gsAVSLRB2)
- [x] Sod: AVSLRB2 `L1 ≤` AVSW (sweep 2026-10-08: 0.00369 vs 0.00396)
- [✗] **Sedov: S3/S4 overshoot reproduced, S6 does not** (sweep 2026-10-08: not testable at 3D nx 40 -- every variant peaks at 2.1-2.3 of 4; S6 stays below trivially. Needs a much finer Sedov)
- [x] Gresho: strict ordering reproduced (sweep 2026-10-08: SLR halves plain AV, 0.21 -> 0.09, but does not beat the switch alone, 0.079; AVSWSLR best, 0.058; **bake-off 2026-10-09 row 6: 0.0720 vs C&D 0.0777, passes; alphaMean 0.0288 vs 0.0285; energy drift 1.6e-4 over the 1e-4 gate**)
- [~] KH: `A(t=1.5)` ladder reproduced; SLR `≥ 1.4×` AVSW (sweep 2026-10-08: ladder AV < AVSW < SLR reproduced, 0.019 / 0.082 / 0.092-0.131; only AVSWSLR reaches 1.4x)
- [x] raw-vs-reconstructed AV energy on Gresho, with the linear/quadratic split (sweep 2026-10-08: ratio 2.1x, not the paper's >= 10x -- headline claim not reproduced; split in the verdict)
- [x] detector map: `φ_ab` field vs α field (R5, 2026-10-08: φ row added to the sweep's maps, redrawn from the saved npz, `docs/av/sweep_2026-10-08/{sod2d,sedov2d}_detectors.png`. φ is ~0 (raw velocities, full AV) almost everywhere on Sod 2D -- hence AVSLRB2 = AV on Sod -- and in the whole shocked Sedov interior, 1 only in the quiet ambient gas, with lattice-aligned streaks ~0.3; unlike the switches' α it does not localise to the front, it marks disturbed vs quiet)
- [x] report rows added for all six variants (sweep 2026-10-08 verdict, Part 0 table)

**M4 — Reconstruction AV.** Git milestone `milestone/gsenz-reconstruction-av`.

---

## Phase 5A — Dynamic β (Chen & Nixon 2025) · **M5**

**Inverted relative to the sketch** — see §2.4. The repo already couples β to α
and has no fixed-β mode; Phase 1 added one. This phase is therefore a
*measurement*, and it is cheap.

> Case grounding decided with the repo owner: **no accretion-disc case is
> built.** Chen & Nixon's result is a steady-state disc surface density, which
> needs external point-mass gravity, an inner sink, an outer boundary and
> shell-averaged diagnostics — disproportionate scope for an orthogonal
> experiment. The mechanism (quadratic AV dominating in well-resolved smooth
> shear) is measured on existing shear cases instead, and the disc reproduction
> is left explicitly un-attempted.

### Build
- Nothing new beyond Phase 1's `CoefficientPolicy`. Document `scaleBeta`'s
  historical double-α semantics (§2.4) so the flag is not read as `β = C_β α`.
- Report Chen & Nixon's `135 β h / (62π α H)` ratio as a Group B row, with `H`
  taken as the local gradient length scale `|∇ρ|/ρ`⁻¹ outside a disc.
- One new cheap case: a **periodic differential-shear box** (2D, `v_x = v₀
  sin(2πy/L)`, uniform ρ and P) — the minimal setting in which the quadratic term
  acts with no shock present. Reuses `sample/regions2D.py` and
  `COMPRESSIBLE_DEFAULTS`; ~80 lines.

### Validation
Matrix: `β ∈ {fixed 2, fixed 0.2, C_β α with C_β = 2}` × detector `∈ {C&D,
Rosswog2020}`.

| Case | Threshold |
|---|---|
| `sod`, `noh`, `sedov` | every Group A metric within **5%** of the fixed-`β = 2` result — i.e. coupling β to α must not cost shock capture |
| `gresho` | angular-momentum loss **and** quadratic-AV energy both lower under `β = C_β α` than under fixed `β = 2` |
| `shearingNoh` 2D | quadratic-AV energy lower; shock front position unchanged within 3% |
| shear box (new) | quadratic-AV energy dominates at fixed β and is suppressed under coupling — the mechanism, isolated |

Also record what the M0 default actually is: `C_q = 0` in
`buildDefaultDiffusionParamsCompressibleSPH`, so the compressible baseline has
**no quadratic term at all**. Whether that is intentional is a finding this phase
must resolve one way or the other.

### Markers
- [x] `scaleBeta` semantics documented; no behaviour change (field comment + `BetaMode` docstring, Phase 1)
- [x] `135βh/(62παH)` ratio reported (`avPower.computeChenNixonRatio`, report column `chenNixonRatioMedian`)
- [x] periodic shear box case added and registered in `CASE_MODULES` (`cases/shearBox.py`, av_report case `shearBox`)
- [~] shock metrics within 5% under coupling (sweep 2026-10-08: Rosswog passes on Sod / Noh / Sedov / shearing Noh; C&D passes Noh / Sedov, fails Sod contact spike +15 %; shearing-Noh: R6, 2026-10-08, the metric is right -- the same config at shear vs = 0 reads 0.198 vs exact 0.200; at vs = 5 the AV turns the shear's kinetic energy (27 vs 1 in the converging flow) into heat, the slab expands and there is no Noh plateau, so 0.27-0.48 vs t/3 measures shear heating. The plan's check is coupled vs fixed 2: Rosswog unchanged (0.351 / 0.351), C&D moves 0.318 -> 0.275, i.e. *towards* the exact front -- a fail in the improving direction; `results/probe_r6_shearingNoh/`)
- [~] Gresho angular-momentum loss and quadratic AV energy both reduced (sweep 2026-10-08: quadratic energy ~10x lower for both; angular momentum lower for C&D, a tie for Rosswog, 0.00734 vs 0.00732)
- [x] shear-box mechanism isolated (sweep 2026-10-08: quadratic fraction fixed beta 0.41 (C&D) / 0.78 (Rosswog) -> ~0.05 coupled; the C&D 'FAIL' is only that 0.41 is below the 'dominates' bar)
- [x] **`C_q = 0` default resolved** — decided 2026-10-08 (user): `C_q = 2` coupled (beta = 2 alpha) for the Monaghan host
- [x] report rows added (sweep 2026-10-08 verdict)
- [x] documented as *not* reproducing chen2025's disc result, with the reason (2026-10-08, below)

**What 5A does and does not reproduce (2026-10-08).** Chen & Nixon's *result* is a steady accretion-disc surface
density (their optimum alpha ~ 0.1, beta ~ 0.2); it is **not reproduced and not attempted**: there is no disc case
(external point-mass gravity, inner sink, outer boundary, shell-averaged diagnostics -- out of scope by the
decision above), so neither the surface-density profile nor their measured alpha_num can be compared. What is
reproduced is the *mechanism* their argument rests on, on non-disc smooth flows (sweep 2026-10-08):
- at fixed beta = 2 the quadratic term carries a large share of the AV energy in well-resolved smooth shear
  (shear box quadratic fraction 0.41 C&D / 0.78 Rosswog; Gresho quadratic AV energy 0.0143 / 0.0106);
- coupling beta = 2 alpha removes it (shear box 0.05 / 0.05, Gresho 0.0014 / 0.0013, Chen & Nixon ratio median
  down 10-600x) without costing the shocks on Rosswog (all Group A within 0.6 %); C&D's Sod contact spike is the
  one exception (+15 %);
- fixed beta = 0.2 (their smooth-disc optimum) gets most of the same smooth-flow gain but costs shocks -- Rosswog
  Sedov radius -2.5 % vs -1.3 % coupled, shearing-Noh front 0.48 vs 0.35 -- which is exactly the conflict that
  motivates coupling. So the sign of every comparison matches their argument; the disc numbers stay unchecked.

**M5 — Dissipation-controlled.** Git milestone `milestone/dynamic-beta`.
Orthogonal: does not block Phases 5B, 6 or 7.

---

## Phase 5B — Sphenix (Borrow et al. 2022) · **M6**

The question is not whether Sphenix is scientifically novel. It is: **can most of
the C&D behaviour be had with considerably less machinery?** — which matters
specifically for a GPU code.

### Grounding
§1.4, Eqs. (15)–(24). `β_V = 3`, `α_V,max = 2`, `ℓ_V = 0.05`, `α_min = 0`.

### Build
Seven touch points. Specifics:

- `modules/shockCapturing/Sphenix2022.py`.
- Eq. (24) is **implicit** — implement it as written. An explicit relaxation
  loses the unconditional stability that §5B's threshold tests for.
- Eq. (19) puts Balsara in the **pair coefficient**, not the indicator. Reuse the
  standalone `B_a` lifted in Phase 4.
- `γ_K` (kernel gamma, the support/smoothing-length ratio) — the repo already has
  `sphKernelScale` and `sphKernel_xi`; pick the right one deliberately, given the
  `h`-as-cut-off convention.
- `∇̇·v` uses the same previous-step finite difference the C&D path already keeps
  in `particleState.divergence` (`monaghan.py:188`) — reuse it, do not add a
  second copy.

### Unit tests
- Eq. (24) stability: from any `α₀ ≥ 0` with `α_loc = 0`, iterate 1000 steps at
  `Δt/τ ∈ {0.1, 1, 10, 100}` — `α` stays in `[0, α₀]`, monotone decreasing, no
  overshoot below zero. This is the property the implicit form buys.
- `α_loc = α_max S/(c²+S)` → `0` as `S → 0`, → `α_max` as `S → ∞`.
- `B_i → 1` in pure compression, `→ 0` in pure rotation.
- Instant-raise branch takes precedence over the implicit branch.

### Validation

| Case | Threshold |
|---|---|
| `sod` 1D/2D/3D, `sedov`, `noh` | every Group A metric within **5%** of C&D at M0 |
| `gresho` | `L1(v_φ) ≤` C&D's; `alphaMean ≤` C&D's |
| `kelvinHelmholtz` | `A(t)` ≥ C&D's |
| **performance** | **ms/step ≥ 15% lower than C&D**, same case, same resolution, CUDA-event timed with warmup |
| **machinery** | neighbour-loop count **one lower** than C&D per step (`computeMWarp` dropped); recorded, not estimated |
| **robustness** | stable at `cflFactor × 4` where C&D is not — Eq. (24) is unconditionally stable by construction |

The 15% figure is a target derived from dropping one full neighbour pass
(`computeMWarp`) plus one gradient pass (`computeShearTensor`) out of the C&D
path. If the measured saving is materially below it, that is a finding about this
implementation's bottleneck, not a failure of the port — record which.

### Markers
- [x] `Sphenix2022.py`; enum, registry, config, dict round-trip (step-boundary hook like Rosswog2020)
- [x] Eq. (24) implemented implicitly; stability unit test at `Δt/τ = 100` (`tests/test_sphenix.py`)
- [x] Balsara in the pair coefficient, not the indicator (`balsaraPairLimiter`)
- [x] gradcheck green: n/a (no warp kernel; the switch runs at the step boundary, off the AD path, like Rosswog2020)
- [✗] shock metrics within 5% of C&D (sweep 2026-10-08, paper's ell_V 0.05: Sod L1 2.6x C&D. At the new default ell_V 5 Sod 0.0050 vs 0.0052 and Sedov -0.3 % vs -1.1 % pass; sod2d / sod3d / Noh at ell_V 5 come from the bake-off, R1; **bake-off 2026-10-09, row 6 vs C&D: Sod 1D +7.0 % (0.00456 vs 0.00426), Sod 2D 0.0159 vs 0.0186, Sod 3D equal, Sedov -0.41 % vs -0.38 %, Noh within 0.1 %, Noh 2D -12.0 % vs -11.7 % -- all but Sod 1D within 5 % or better; marker stays ✗ on the one 7 % miss**)
- [✗] Gresho `L1(v_φ) ≤` C&D (sweep 2026-10-08: 0.090 vs 0.073; ell_V 5: 0.077, still above)
- [x] **ms/step ≥ 15% lower** (or the shortfall diagnosed) (sweep 2026-10-08: 11 % lower, 9.18 vs 10.32 ms/step. **Diagnosed** (R4, 2026-10-08, Gresho nx 128, min of 2 reps): none 7.89, Rosswog2020 8.30, Sphenix 8.99, Sphenix without the Balsara pair limiter 8.36, C&D 10.00 ms/step -- Sphenix 10 % below C&D, 16 % without Balsara. The whole shortfall is the velocity-Jacobian loop the Balsara factor needs for curl v (0.63 ms); SWIFT gets div and curl from the density loop for free, this repo's density pass does not produce curl)
- [x] neighbour-loop count recorded (R3, 2026-10-08, per RHS evaluation: Sphenix 1 -- the velocity Jacobian for Balsara's curl; div v comes free from `-drho/dt / rho` -- vs C&D 3: velocity gradient, R (Eq. F.4), v_sig; Rosswog2020 0)
- [✗] `cflFactor × 4` stability demonstrated (sweep 2026-10-08: run; no advantage -- Sphenix and C&D both diverge on Sod and Gresho, both survive Sedov and Noh)
- [x] detector map vs C&D (sweep 2026-10-08: `docs/av/sweep_2026-10-08/{sod2d,sedov2d}_detectors.png`, at ell_V 0.05)
- [x] report rows added (sweep 2026-10-08 verdict)

**Build notes (2026-10-07).** `modules/shockCapturing/Sphenix2022.py`: step-boundary hook (like `Rosswog2020`), `div v`
from the stage-0 `divergence`, `h = H / sphKernelScale`, `tau = ell_V H / c`; operator = `Price2012` term, `C_l = 1`,
`C_q = 3` coupled, `balsaraPairLimiter` (config `sphenix`, plain gradient for B). Smoke pre-flight caught a 0/0 in
Eq. (23) for cold gas (`c = 0`: Sedov's ambient medium, Noh) -- guarded, unit test added. In cold gas the formula
itself sends alpha to alpha_max for any `S > 0` and never decays it (`tau ~ 1/c`); the same holds for Wadsley
(`v_sig -> 0`). Expect that in the Sedov / Noh alpha maps; it is the papers' formulas, not a port error.

**M6 — Performance-aware.** Git milestone `milestone/sphenix-av`.

---

## Phase 6 — Wadsley/Gasoline2 gradient shock detector · **M7**

The last detector before the AV work is declared complete. The question:
**can a velocity-gradient-based detector distinguish compression from an actual
shock, where divergence-based detectors cannot?**

### Grounding
§1.5, Eqs. (21)–(29). `α_max = 2`, `τ = h/(0.2c)`.

### 6.1 Build
- `modules/shockCapturing/Wadsley2017.py`.
- `∇P` via Eq. (21) — a one-sided `warpOperation` gradient of `(γ−1) m_j u_j`;
  the same gradient primitive `computeShearTensor` uses.
- `dv/dn = Σ n_α V_αβ n_β` from the existing `Vs` tensor.
- **`T_αβ` keeps the trace**: `T = Shear + (trace/dim)·I` from
  [`common.py:92-99`](src/warpSPH/modules/shockCapturing/common.py#L92). One
  line. Do **not** reuse C&D's trace-free `S` — §1.5.
- `W_R,ij = (1 − (r_ij/2h_i)⁴)` is a plain polynomial weight, not an SPH kernel;
  the paper is explicit that using the kernel makes `R` noisy.
- `R_i` clamped to `[−1, 1]` (Eq. 29's own caveat: `D` is a modification of
  `dv/dn` and noise pushes it out of range).
- `dD/dt` as a previous-step finite difference, following the existing
  `divergence` pattern; needs a `D_prev` state field + `finalize` copy
  (touch point 5).

### 6.2 The `h`-convention re-derivation — do this first

Eq. (26)'s factor 2 exists because Gasoline2's `h` differs from C&D's. This repo
stores `h` as the **cut-off radius**, a third convention (see the long comment at
[`limiter.py:135-147`](src/warpSPH/modules/crk/limiter.py#L135)). Before any
tuning:

- [x] write down `h_repo / h_CD` and `h_repo / h_Gasoline2` for kernel `B7` and
      Wendland2, in terms of `sphKernelScale` / `sphKernel_xi`;
- [x] derive the correct prefactor for `A_i` in this repo's units;
- [x] state it in the module docstring with the derivation, the way
      `CullenDehnen2010.py` now documents the B11 sign.

Copying `2` unexamined is the predictable failure mode of this phase.

### 6.3 Unit tests — these are the point of the paper

Manufactured velocity fields on a lattice, checking the detector directly:

| Field | Expected | Why it matters |
|---|---|---|
| uniform compression `v = −k r` | `α` stays at floor; `ξ ≤ 1/16 + tol` | **The discriminating test.** C&D gives `α ~ 0.5` *everywhere* here (their fig. 15); Wadsley's `D` is built to be blind to it. If this does not separate the two detectors, the port is wrong. |
| pure rotation `v = ω × r` | `α` at floor; `ξ = 0` in expansion | `T` keeps the trace, so unlike Balsara's curl comparator it is not fooled by differential rotation |
| pure shear | `α` at floor | |
| travelling sound wave | `α` at floor | linear, no shock |
| step discontinuity | `ξ → 1`, `α → α_max` | |

Plus: `D` takes the extremal value `∇·v` in a 1D shock (Eq. 24's stated
property); `R_i` stays inside `[−1,1]` on a noisy lattice.

### 6.4 Validation

| Case | Threshold |
|---|---|
| `sod`, `sedov`, `noh` | Group A metrics within **5%** of the best of {C&D, Rosswog, Sphenix} at their milestones |
| `gresho` | `L1(v_φ) ≤` C&D's |
| **spherical collapse / uniform compression** | `alphaActiveFraction` materially below C&D's — the paper's central claim, and the reason `sedov`'s early phase is the right place to look |
| `kelvinHelmholtz`, `rayleighTaylor` | growth ≥ C&D's |

### 6.5 Deliverable: the detector comparison

Per §0.3, the full map for all five detectors (C&D, R&H, Rosswog, Sphenix,
Wadsley) on the **identical frame** of `sedov` 2D and `sod` 2D:

```
particle id · position · α · ∇·v · d(∇·v)/dt · detector scalar
```

> This is likely to be more informative than another Sod profile plot, and it is
> the artefact that makes Phase 7's default selection defensible rather than
> aesthetic.

### Markers
- [x] **`h`-convention re-derivation done and documented (§6.2) — before tuning** (`H = h_CD = 2 h_G2`, prefactor 0.5; module docstring)
- [x] `Wadsley2017.py`; enum, registry, config, `D_prev` state (set by the step-boundary hook, which persists it like `entropiesPrev`)
- [x] `T` keeps the trace; `W_R,ij` is the polynomial weight, not the kernel
- [x] gradcheck green: n/a (no warp kernel; step-boundary hook, off the AD path)
- [x] **uniform-compression test separates Wadsley from C&D** (`test_uniform_compression_is_invisible`; sweep 2026-10-08 cylindrical Noh pre-shock alpha 4e-8 vs C&D 0.72, post-shock rho -1.3 % vs -6.5 %; alphaActiveFraction 0.39 vs 0.78 misses the '< 1/2' bar narrowly)
- [x] rotation / shear / sound-wave / step tests pass (R2, 2026-10-08: `test_pure_shear_is_invisible` (D < 1e-3 k with n along a real pressure gradient) and `test_sound_wave_keeps_alpha_at_floor` (linear wave, 300 steps, alpha < 1e-3) added; 9/9 green. The shear box's alphaMean 0.066 is a separate, uniform-pressure effect: see the note under the Phase 6 markers)
- [✗] shock metrics within 5% (sweep 2026-10-08: Noh passes; Sod L1 0.0092 vs 0.0052, Sedov radius -2.3 % vs -1.1 % -- the derived prefactor 0.5; prefactor 2 fixes Sod but costs Gresho / Noh, kept 0.5 by decision 2026-10-08)
- [x] Gresho `L1(v_φ) ≤` C&D (sweep 2026-10-08: 0.067 vs 0.073)
- [x] five-detector comparison maps rendered on identical frames (`docs/av/sweep_2026-10-08/{sod2d,sedov2d}_detectors.png`: α, ∇·v, and φ for the reconstructing configs)
- [x] report rows added (sweep 2026-10-08 verdict)

**Finding: Wadsley fires in smooth shear at uniform pressure (2026-10-08, R2; accepted as a property of the paper's detector, user 2026-10-08 -- no remedy).** The shear box
(pure shear, uniform P) runs Wadsley at alphaMean 0.066 against C&D's floor 0.02, while the manufactured pure-shear
unit test (n along an imposed pressure gradient) gives D = 0. Probe `results/probe_r2_shearBoxWadsley/` (video):
the relative pressure gradient stays tiny, |grad P| H / P ~ 2e-3, yet |D| / |T| has median ~1 all run long -- the
value D takes when n sits on the shear's compressive principal axis (|D| = 0.75 k vs |T| = k / sqrt 2). So
n = grad P / |grad P| (Eq. 22) is set by the small pressure perturbations the shear itself makes, D reads the shear
strain as compression, and the step-to-step change of D drives alpha (0.05-0.24 through the run). This is the paper's
detector as written (no floor on |grad P|), not a porting bug. A remedy would trust n only where the pressure
gradient is resolved above its own noise (a physics-derived floor, not a tuned threshold); that is a deviation from
the paper; the user decided (2026-10-08) to leave the detector paper-faithful. It also plausibly explains Wadsley's low KH / RT growth (0.081 vs
0.100; 0.317 vs 0.383): both are shear-dominated at near-uniform pressure.

**M7 — Detector-complete.** Git milestone `milestone/wadsley-detector`.

---

## Phase 7 — The AV bake-off · 🏁 **M8**

**The first stop-and-consolidate point.** Do not start Phase 8 here. Establish
what exists first.

### The matrix

Host scheme Monaghan throughout; CompSPH cross-check on the shortlist only.

| # | Detector | Pair velocity | β | Status |
|---|---|---|---|---|
| 1 | NoneSwitch (`α = 1`) | raw | fixed 2 | baseline |
| 2 | Cullen-Dehnen 2010 | raw | fixed 2 | baseline |
| 3 | Cullen-Hopkins | raw | fixed 2 | baseline |
| 4 | Read-Hayfield 2012 | raw | fixed 2 | baseline |
| 5 | Rosswog 2020 | raw | fixed 2 | M2 |
| 6 | Sphenix 2022 | raw | fixed 3 | M6 |
| 7 | Wadsley 2017 | raw | fixed 2 | M7 |
| 8 | none (`α = 1`) | limited | fixed 2 | M4 · AVSLR |
| 9 | none (`α = 1`) | limited + Balsara `p=2` | fixed 2 | M4 · **AVSLRB2** |
| 10 | Rosswog 2020 | limited | fixed 2 | M2 × M3 |
| 11 | Wadsley 2017 | limited | fixed 2 | M7 × M3 |
| 12 | Rosswog 2020 | limited | `2α` | + M5 |
| 13 | Sphenix 2022 | Sphenix (raw) | `2α` | + M5 |

Not every cell is required. Rows 1-4 come free from M0/M1; rows 5-9 are each
phase's own headline config; 10-13 are the cross terms worth spending time on.

### Deliverable — `results/av_foundation/report.md`

Part 0's table, filled for every row, with the M0 baseline in the first column.
Plus a short prose section per row: what it is good at, what it costs, and the
one sentence that would justify making it the default.

### Selection and the gate

- Pick the recommended default configuration.
- Write down, in `AV_PLAN.md` and in the repo README: *"This is the reference SPH
  dissipation configuration against which future hydrodynamic operators are
  compared."*
- Tag `vX.Y-av-foundation`.

**M8 is a hard gate.** Past it the project splits:

```
                     ┌───────────────┐
                     │ AV Foundation │
                     └───────┬───────┘
                         M8 PASS
                ┌────────────┴────────────┐
        production SPH             research branches
                │                         │
          PE-SPH → Hopkins          experimental AV
                │
             Godunov
```

The gate exists to stop the repo accumulating half-validated schemes. A new AV
idea after M8 goes on a research branch and reports into the same table; it does
not enter the production path without doing so.

### Markers
- [x] every matrix row runs from a named config (`av_report` `BAKEOFF_ROWS`; `--bakeoff` smoke pre-flight clean, 2026-10-08)
- [ ] Part 0 table filled for all rows
- [ ] per-row prose written
- [x] default selected, with the reason recorded (2026-10-08, user: Rosswog 2020 + limited reconstruction + coupled beta; KH best, shocks within the C&D-Q band)
- [x] README states the reference configuration
- [ ] tagged `vX.Y-av-foundation`

**🏁 M8 — SPH foundation complete.**

---

## Phase 7b — Riemann dissipation as an alternative AV · **M8b** (opened 2026-10-08, user)

**Why here.** Phases 3-6 left a reconstruction layer (gradient → limiter → left/right states) that is the first
half of a Godunov pair flux. Phase 7b adds the second half, a Riemann solver, and uses it the cheapest possible way:
as one more `ViscosityTerms` formulation, so it enters the bake-off table as rows and not as a new scheme. It is
also the unit-tested solver MFM/MFV (PESPH_PLAN §7) would need, so it is built to return the star state *and* the
face flux. Not a gate: M8 closes on the bake-off as it stands; 7b reports into the same table as an **addendum run**
(`--bakeoff` machinery, same matrix, extra rows), per the "new AV ideas report into the same table" rule above.

**Isolation (history).** The bake-off launches every case as a subprocess against this checkout's `src/`, so the work was first
built on branch `av-7b` in a worktree; on the user's call (2026-10-08, the tree must be whole before the run) it was moved onto
`dev` after the checks below. Defaults must stay **bit-identical** (smoke vs M0g) at every step.

### 7b.1 — `modules/riemann` (new, like `modules/reconstruction`)
- `solvers.py`: branch-free `wp.func`s for the ideal-gas 1D Riemann problem on the pair normal, `(rho, un, p)_L,R` plus
  `gamma` → `(p*, u*)`, and an HLLC `(rho, u, p)` face flux. Solvers behind a `RiemannSolver` enum: `Acoustic`
  (linearised, Godunov-SPH / Monaghan 1997), `PVRS` and `TRRS`/`TSRS` (Toro ch. 9 primitive-variable / two-rarefaction
  / two-shock), `HLLC` (Toro ch. 10, Davis or Einfeldt wave speeds). No iterations in kernels; the exact iterative
  solver stays the Python reference (`caseUtils/compressible/sod/sodSolution.py`).
- Tests (`tests/test_riemann.py`): the five Toro tests against the exact solver (p*, u* within the approximate
  solver's known error), symmetry `R(L,R) = -R(R,L)` (antisymmetric flux), vacuum / near-vacuum guards, float64 exact
  consistency (`L = R` gives `p* = p`, `u* = u`). `scripts/gradcheck_riemann.py` in the 25-module gradcheck set (adjoints
  must not divide before the guard, AV_PLAN §2.6).

### 7b.2 — Limiter family in `modules/reconstruction/limiters.py`
- `LimiterType` enum (Frontiere/`VanLeerSymmetric` = today's `4r/(1+r)^2`, the default, bit-identical; `Minmod`,
  `VanLeer` (textbook `2r/(1+r)`), `VanAlbada`, `MC`, `Superbee`, `Ospre`). `r = min(ri, rj)` is always <= 1 (`rj = 1/ri`),
  so each limiter is one function `psi(r)` on [0, 1], capped at 1 (phi multiplies a single midpoint extrapolation).
- Plumbing: `limiterType` int field on `CRKViscosity` and `DiffusionParameters` (dict round trip, old keys read back as
  the default; `enableVanLeerLimiter` / `forceVanLeer*` keep meaning "limiter on/off"); `limitedPairPhi` gets the type;
  call sites: `crk/accel.py`, `crk/dudt.py`, `reconstruction/pairVelocity.py`, `reconstruction/diagnostics.py`.
- Tests: each `psi` against its closed form, `psi(1) = 1`, `psi(0) = 0`, monotone, `psi <= 1`, `psi(r) / r` symmetry;
  gradcheck at and next to the kinks (minmod / MC / superbee); default-type run bitwise equal to M0g smoke.

### 7b.3 — Riemann dissipation term
- First stage (no scalar gradients): new `ViscosityTerms.Riemann*` entry in `dissipation/pi`: states are the particle
  values `(rho, P)` with the (optionally reconstructed, Phase 3) pair velocity along `x_ij`; `Pi_ij` is the effective
  pairwise term `2 p*_ij`-equivalent so momentum and heating stay antisymmetric / conservative like the other Pi
  formulations. No alpha switch (intrinsically limited); Balsara pair limiter is the shear control.
- Second stage: limited gradients of rho and P (`reconstruction/gradient.py` has only the velocity Jacobian) and
  MUSCL states at the pair midpoint with the 7b.2 limiter family (Godunov-SPH proper: Inutsuka 2002, Cha & Whitworth
  2003); the limiter and `C_l`/`C_q` are tuned together here (the open note in Phase 3's limiter paragraph).
- Validation: Sod 1D / 2D / 3D vs the exact solution (L1, Group A), Sedov, Noh, Gresho, KH (Group B-D), energy and
  momentum conservation (Group E), cost per step (Group F), all against the same thresholds as the other rows.

### 7b.4 — Bake-off addendum
Rows: Riemann (first order) raw; Riemann + limited velocity; Riemann + Balsara `p=2`; second stage if it lands.
Run with the existing `--bakeoff` job list plus the new rows, merged table in `results/av_foundation_7b/`.

### Build notes (2026-10-08, branch `av-7b`, uncommitted)

- **7b.1 done.** `modules/riemann/solvers.py`: `riemannStarState` (Acoustic, PVRS, TRRS, TSRS, Adaptive, HLLC) and `hllcFlux`;
  `RiemannSolver` lives beside the other dissipation enums in `diffusionParameters.py`. `tests/test_riemann.py` (37, also in
  float64) against a Newton exact solver on Toro's five problems; `scripts/gradcheck_riemann.py` 7/7. Findings that shaped the
  tests: the weak-wave solvers are second order in the wave strength (Sod, a pressure ratio of 10: acoustic -37 %, PVRS +81 % on
  p*; TRRS +1.2 %, TSRS +4 %, HLLC -8 %); on two equal colliding streams at |u| = a the acoustic family is 25 % low and TSRS 8.5 % low
  (the quadratic term the acoustic family lacks); HLLC with Toro's PVRS wave speeds puts Sod's momentum flux 22 % off the exact
  Godunov flux (mass 2 %, energy 3 %).
- **7b.2 done.** `LimiterType` (VanLeerFrontiere = default, Minmod, VanLeer, VanAlbada, MC, Superbee, Ospre) in
  `limiters.py::limiterPsi`; `limiterType` field on `DiffusionParameters` and `CRKViscosity` (dict round trip; old configs read
  back the default); plumbed through `computeVanLeer`, `limitedPairPhi`, `reconstructPairVelocity` and the five call sites.
  Default bit-identical: smoke `--compare` tol 0 on `phase3` and `crk` (20/20 pairs each), 68 CRK / reconstruction / dissipation
  tests and the three gradcheck scripts green. Observation: Frontiere's limiter `4r/(1+r)^2` has slope 4 at the origin, above
  the `psi <= 2r` bound of Sweby's TVD region that minmod / van Leer / MC / superbee respect; whether that matters for a pair
  extrapolation (`r` is a ratio of the two particles' gradients, not of successive differences) is not established.
  `r = min(r_i, r_j) <= 1` always, so only [0, 1] is ever evaluated.
- **7b.3 first stage done.** `ViscosityTerms.RiemannDissipation` (13): `Pi_ij = (p*(w) - p*(0)) (1/rho_i^2 + 1/rho_j^2)` with
  left state j, right state i, `w` the (reconstructed) pair velocity along `x_ij/r`; pair-symmetric (conservative, the usual
  heating applies), zero at relative rest, positive for compression, scaled by the pair-mean switched alpha. Acoustic with
  equal states is exactly Monaghan 1997a with alpha = 1 (`tests/test_piFormulations.py`); gamma is `rho c^2 / P` of the pair.
  `gradcheck_dissipation.py` covers every solver (viscous force and heating). `av_report` configs `riemann*` (group `phase7b`).
  **Sod 1D (nx default, full profile, video, `results/av7b_sod`):** L1(v_x) 0.00369 (gsAV 0.00369), P contact spike 0.069 (gsAV
  0.076), A spike 0.029 (0.037), energy drift 4e-6; the four solvers differ by < 4 % on every Sod metric (the Sod shock is weak
  enough that the quadratic term barely matters); raw vs limited velocity as for the other terms (limited: spike 0.063, but
  R4 pressure error -2.4e-3). Caveat: `avEnergyLinear/Quadratic` books all of a Riemann term as linear (it splits by `C_l`/`C_q`).
- **Moved onto `dev` 2026-10-08 (user: "the bake-off won't run on a half-working tree").** Applied in place (the `av-7b` worktree
  and branch are gone), uncommitted. Checks on `dev`: the full test suite (no failures), and the bake-off's own smoke pre-flight
  re-run (`--bakeoff --smoke`, all 18 jobs rc 0) is **bit-identical (tol 0) to the pre-7b pre-flight on every row, compSPH, kh256 and
  the maps** (`results/av_bakeoff_smoke_after7b`). Note the pre-flight rewrites `results/av_foundation/report.md`; the real run does too.
- **Validation (full profile, video, `docs/av/riemann_7b_2026-10-08/`):** Sod 2D, Noh, Gresho, Sedov for `riemann`, `riemannLimited`,
  `riemannLimitedB2`. On par with the Phase 4 rows everywhere; best on Sod 2D L1 (0.0155 vs gsAV 0.0201), Gresho is fixed by the limited
  velocity (0.210 -> 0.091) as for every term, Sedov shock radius slightly worse (-0.020 / -0.023 vs -0.015). A step-0 NaN on cold gas
  (Noh, Sedov) was a 0/0 in the term, fixed. No fliers or velocity alarms in the frames.
- **Stage 2 re-scoped (not built).** In this formulation Pi is `p*(w) - p*(0)`, so the reconstructed `rho`, `P` only enter through the
  impedance and the quadratic term: reconstructing them is a second-order effect and the velocity reconstruction (already there) carries
  the dissipation. The version that needs reconstructed pressures is Godunov-SPH proper (`p*` replaces the whole symmetric pressure sum,
  Inutsuka 2002 / Cha & Whitworth 2003): a different scheme, not an AV row. **Decided (user, 2026-10-08): build the reconstruction
  (the old stage 2) as the lead-up to Godunov SPH; tracked in [`GODUNOV_SPH_PLAN.md`](GODUNOV_SPH_PLAN.md) (layer 1).**

### Markers
- [x] `modules/riemann` solvers + `tests/test_riemann.py` + gradcheck
- [x] `LimiterType` enum plumbed (CRK and Monaghan paths), default bit-identical, tests + gradcheck
- [x] first-stage Riemann Pi term registered, Sod 1D vs exact (L1 on par with the best Phase 4 row)
- [~] scalar gradients + MUSCL states (second stage): moved to `GODUNOV_SPH_PLAN.md` layer 1 (user, 2026-10-08)
- [ ] addendum bake-off rows (`riemann*` configs exist; run after the main bake-off) + prose
- [x] on `dev` (applied in place, uncommitted; worktree retired)

---

# Part 4 — Phase 8 and beyond

## Phase 8 — Pressure-Entropy SPH → [`PESPH_PLAN.md`](PESPH_PLAN.md)

**This plan does not restate the PESPH work.** It has its own 525-line design
document covering both variants (PESPH-E, pressure-energy, Frontiere App. G /
Hopkins 2015 App. F2; and PESPH-A, pressure-entropy, Hopkins 2013) against the
papers, with a six-phase implementation sequence, an audit of what the repo
already has, and a `f_ij` naming hazard that will bite anyone who starts from the
equations alone.

What this plan owes it, and what it owes back:

**The interface contract (this plan → PESPH):**
- PESPH consumes the Phase-1 `DissipationOperator` **unchanged**. That is what
  makes "don't port every new AV at once" an enforceable property of the
  architecture rather than a piece of advice.
- The M8 recommended configuration is the AV that PESPH runs with, so the
  comparison is `old SPH + known AV` vs `new PE-SPH + same AV` and the
  hydrodynamic change is isolated.
- PESPH_PLAN §2 assumes `ViscositySwitch.CullenDehnen2010` is available and
  validated. It is, as of the Phase 6 work that is this plan's input.
- **PESPH_PLAN §2.1's `entropies`/`pressures` state-tag bug is fixed in Phase 1**
  of this plan (§2.5). Mark it closed there when Phase 1 lands.

**What PESPH needs that this plan does not build:** the entropic-function EOS
path is already written but dead (`modules/eos/gas.py`,
`EOSSource.specificEntropy`); `entropies` is `constant()` and must become
`integrated()` for PESPH-A; `SupportScheme.PartialSymmetric` exists in
`warpSPHCore` specifically for it. All three are PESPH_PLAN's scope, not this
plan's.

**Sequencing:** blocked on M8, per the gate above.

## Phases 9-10 — Hopkins general formulation, and Godunov

Kept as outlook only. The one architectural point worth carrying:

```
particle i → primitive variables → gradient → slope limiter
           → left/right reconstructed states → Riemann solver
           → pair flux → SPH particle update
```

**Phase 3's reconstruction layer is the third and fourth box in that chain.**
That is why M3 is the strategically load-bearing milestone of this plan: the
velocity-reconstruction work is not an AV experiment that happens to be reusable,
it is the first component of the eventual Godunov infrastructure, built early and
validated independently.

Hopkins (2013) `hopkins2013` and (2015) `hopkins2015`, Inutsuka (2002)
`inutsuka2002`, Cha & Whitworth (2003) `cha2003` are in `literature/` and stay
beside this plan. `frontiere2017` bridges both directions: it is the source of
the SLR in Phase 3 *and* of the CRKSPH/PESPH comparison set.

---

# Part 5 — Literature

All present in `literature/`. Track reading state here, not in a separate file.

| Paper | File | read | eqs extracted | implemented | benchmarked |
|---|---|---|---|---|---|
| Rosswog 2020, entropy trigger | `rosswog2020entropy_entropy-based-dissipation-trigger.pdf` | ☐ | ☑ §1.1 | ☐ P2 | ☐ |
| García-Senz & Cabezón 2026 | `garciasenz2026_heuristic-switches-sph-dissipation.pdf` | ☐ | ☑ §1.2 | ☐ P4 | ☐ |
| Chen & Nixon 2025 | `chen2025_minimizing-numerical-viscosity-discs.pdf` | ☐ | ☑ §1.3 | n/a | ☐ P5A |
| Borrow et al. 2022, Sphenix | `borrow2022_sphenix.pdf` | ☐ | ☑ §1.4 | ☐ P5B | ☐ |
| Wadsley et al. 2017, Gasoline2 | `wadsley2017_gasoline2-modern-sph-code.pdf` | ☐ | ☑ §1.5 | ☐ P6 | ☐ |
| Rosswog et al. 2000 (App. A) | `rosswog2000_merging-neutron-stars-asymmetric.pdf` | ☑ App. A | ☑ | ☑ `Rosswog2000` switch | ☑ S3 (sod, gresho) |
| Cullen & Dehnen 2010 | `cullen2010_inviscid-sph.pdf` | ☑ | ☑ | ☑ | ☑ phase6 |
| Read & Hayfield 2012, SPHS | `read2012_sphs-higher-order-dissipation-switch.pdf` | ☑ | ☑ | ☑ | ☑ phase6 |
| Frontiere et al. 2017, CRKSPH | `frontiere2017_crksph.pdf` | ☐ | ☐ | ☑ (CRK-internal, §2.2) | ☐ |
| Morris & Monaghan 1997 | `morris1997switch_a-switch-to-reduce-sph-viscosity.pdf` | ☐ | ☐ | ☐ stub (§2.3) | ☐ |
| Balsara 1995 | `balsara1995_von-neumann-stability-sph.pdf` | ☐ | ☐ | ⚠ inside R&H | ☐ |
| Price 2012, SPH & MHD | `price2012_sph-and-magnetohydrodynamics.pdf` | ☐ | ☐ | ☑ (`Price2012_98` default) | ☐ |
| García-Senz et al. 2012, IAD | `garciasenz2012_integral-approach-gradients.pdf` | ☐ | ☐ | ☐ optional P3 | ☐ |
| Dehnen & Aly 2012 | `dehnen2012_convergence-without-pairing-instability.pdf` | ☐ | ☐ | — | — |

**Next generation** — beside the plan, not in it: `hopkins2013_general-class-lagrangian-sph.pdf`,
`hopkins2015_new-class-meshfree-hydrodynamic-methods.pdf`,
`inutsuka2002_sph-riemann-solver-reformulation.pdf`,
`cha2003_godunov-particle-hydrodynamics.pdf`. PESPH-side background:
`saitoh2013_density-independent-sph.pdf`, `springel2002_sph-entropy-equation.pdf`.

## If you read only five

1. **Rosswog 2020** — the entropy trigger *and* the slope-limited reconstruction
   are in the same paper. the original sketch listed them as two literature threads;
   they are one, and the paper is unusually well aligned with the Phase 2 → Phase
   3 progression.
2. **Cullen & Dehnen 2010** — the modern time-dependent baseline everything else
   is measured against. Already implemented and validated here.
3. **García-Senz & Cabezón 2026** — the current switch-free direction, and the
   only one of the five with a published variant ranking usable as thresholds.
4. **Chen & Nixon 2025** — short, and the only paper that treats α and β as
   independent knobs.
5. **Hopkins 2013** — the destination, via PESPH_PLAN.

`read2012` + `wadsley2017` + `borrow2022` are comparative: read them when
implementing their detector, not before.

## Bibliography hygiene

If any entry here is added or corrected in `literature/ABSTRACTS.md`, run
`python scripts/check_literature.py` — it verifies each abstract verbatim against
its PDF. Use the `paper-lookup` skill for bibliographic fields; do not fill
volume/page/year from memory.

## Input from OPEN_PROBLEMS §15 (2026-09-30)

- CRK viscosity switch regulariser fixed to Frontiere Eq. (69)'s `eps^2 = 1e-2` in eta^2 units (was `1e-7 h^2` added to a
  dimensionless eta.eta, i.e. no regularisation; near-coincident approaching pairs blew up, see OPEN §15). M0's baseline
  lock should start from this state, not from before it.
- Open for the AV work: `C_l, C_q` are kernel-dependent in the paper (Table D.1: B7 2.0/1.0, Wendland 0.5/0.25, because
  mu scales with h); `buildDefaultDiffusionParamsCRKSPH` uses 1/1 for every kernel (check `smooth = H/xi` vs the paper's h
  first). The compressible dt has no approach/viscous term.

- `C_l` = 2 (paper, B7) measured 2026-09-30 (CRKSPH_LIMITER_PLAN note (b)): Sod ringing -70 % but Gresho spin-up +8 % -> +11.5 % (nx 64), +12.6 % -> +21.9 % (nx 96); user: keep 1, revisit together with the limiter.
