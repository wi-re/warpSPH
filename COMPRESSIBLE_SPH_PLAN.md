# warpSPH — compressible SPH plan (dissipation · CRKSPH · Godunov · PESPH · MFM/MFV)

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

Opened 2026-10-11 (user). One plan for every compressible scheme in the repo and the one that is about to arrive.
It replaces `AV_PLAN.md`, `CRKSPH_LIMITER_PLAN.md` and `GODUNOV_SPH_PLAN.md` (now in `docs/historic_plans/`,
with their evidence and long build logs intact) and takes over the MFM / MFV work that exists only in warpSPHCore
so far. [`PESPH_PLAN.md`](PESPH_PLAN.md) stays its own file and is Track D here.

The old plans were organised by *who proposed what*. This one is organised by **scheme**, with one shared deliverable:
a comparison matrix every scheme reports into (§2).

## 0. What went where

| Old plan | Now | Carried over | Dropped (and why) |
|---|---|---|---|
| `AV_PLAN.md` (2007 lines) | `docs/historic_plans/AV_PLAN.md` | Track A: the M8 tag, the unrun 7b rows, OPEN §23, the dt approach term, the `C_l`/`C_q` question, the RT drift question, gotchas (§7) | Phase 0-6 build/validation logs (done, evidence in `docs/av/`), the Part 5 reading-state table (`literature/ABSTRACTS.md` tracks it), shock-width metric, StepTimer, the Phase 5A disc result, the Sedov-overshoot gate (untestable at nx 40) |
| `CRKSPH_LIMITER_PLAN.md` (473) | `docs/historic_plans/CRKSPH_LIMITER_PLAN.md` | Track B: the closed findings in one paragraph, the parked variants, the open question the MFM Gresho result can answer | Notes O1-O4 and (a)-(h) in full (the numbers stay in the archived file); the P4 acceptance list (decided 2026-10-01) |
| `GODUNOV_SPH_PLAN.md` (133) | `docs/historic_plans/GODUNOV_SPH_PLAN.md` | Track C: blob, Wengen KH, solver ablation, the density-jump vs pairing trade-off, matrix rows | Layers L1-L3 build logs (done) |
| MFM in `~/dev/warpSPHCore` (`higher_order.md` Phase 8, `src/warpSPHCore/mfm/`) | Track E | the whole migration plan, the core's open items | — |
| `PESPH_PLAN.md` | **unchanged file**, Track D | §7 (MFM outlook) moved into Track E; header and the AV-gate paragraph retargeted | — |

## 1. Where the compressible family stands (2026-10-11)

| Scheme (`CompressibleSPHScheme`) | State | Reference / evidence |
|---|---|---|
| `Monaghan` | **default host.** Rosswog 2020 trigger + limited velocity reconstruction + coupled β (`C_q = 2`, β = 2α); README states it as the reference dissipation configuration | bake-off 2026-10-09, [`docs/av/bakeoff_2026-10-09/`](docs/av/bakeoff_2026-10-09/README.md) (13 rows × 12 cases, no row diverged) |
| `CompSPH` | works; its Rosswog + Sedov combination blows up at t = 0.056 | OPEN_PROBLEMS §23 |
| `CRKSPH` | limiter default (1/n_h, 0.2/n_h); deterministic; Gresho spin-up is intrinsic (a slowly converging tangential pressure pump the viscosity cancels); same-lattice triple point piles up (parked); curved-wall stagnation fails (OPEN §19) | archived `CRKSPH_LIMITER_PLAN.md`; OPEN §15, §19 |
| `GSPH` (Cha & Whitworth form) | baseline, no AV; equals standard SPH at constant `P` (measured) | `docs/av/godunov_l2_2026-10-08/`, `godunov_densityJump/` |
| `InutsukaGSPH` | Gaussian-kernel convolution form; stable for Gaussian width η ≤ 1; first-order shock switch on by default | `docs/av/godunov_l3_2026-10-08/` |
| `PESPH` | enum member + `PESPHVariant` only; `AdaptiveSupportScheme.NumberDensity` done; `buildScheme` raises | [PESPH_PLAN.md](PESPH_PLAN.md) |
| **MFM / MFV** | **not in the frontend.** Complete backend in warpSPHCore `main` (torch reference + warp kernels, GIZMO-parity Riemann chain, closure solve, 77 tests); unreleased (after the 0.6.0 bump) | warpSPHCore `higher_order.md` Phase 8 |

## 2. The organising deliverable: one comparison matrix

AV_PLAN's Phase 7 rule was "a new idea reports into the same table or it does not enter the production path". That rule
now covers every scheme, and the table is the plan's single acceptance artefact.

- **Rows** (scheme + configuration, each a named `AVConfig` in `scripts/av_report.py`): the 13 bake-off rows (Monaghan),
  CompSPH ×4, CRKSPH (`av_crk3_*`), the three Phase 7b `riemann*` rows (built, never run as an addendum), the `godunov` /
  `godunovL3` rows (already runnable: `av_sweep_overnight.py --godunov`), PESPH-A / PESPH-E when they exist, MFM and MFV.
- **Cases:** the 12 bake-off cases + KH nx 256 + Gresho nx 64 / 96 + triple point; **new:** blob, Wengen KH, the Cha 2010
  density-jump force test (exists as a probe), and a disordered-IC variant of the battery (X4).
- **Metrics and gates:** Part 0 of the archived AV_PLAN ([`docs/av/av_baseline_2026-09-30.md`](docs/av/av_baseline_2026-09-30.md) holds the table).
- **Output:** `docs/compressible/comparison_<date>/` with the verdict README, per-row notes, report.md, frames.
- **Rules carried over** (user decisions): build first, sweep once; refactors must be bit-identical against the smoke
  `--compare` (the reference is `results/av_smoke_ref_M0g`); never edit `src/` while a sweep runs (use a worktree and
  `PYTHONPATH=<wt>/src`); video on, one run at a time (CLAUDE.md).

## 3. Tracks

Worth ratings: **H** blocks something or answers a standing question, **M** improves a result, **L** nice to have.

### Track A — dissipation foundation (ex-AV_PLAN) — done, closing

**Done.** Dissipation is a set of swappable parts (detector · coefficient policy · pair-velocity policy · pair operator),
the `pi` package is the single source of truth (CRKSPH's `Q` goes through it), five detectors implemented and swept
(Rosswog 2020, García-Senz & Cabezón 2026 SLR, Chen & Nixon β, Sphenix, Wadsley/Gasoline2), Riemann dissipation as an AV
term (7b), the bake-off run, the default chosen (user, 2026-10-08), Sphenix `ell_V = 5`, Wadsley prefactor 0.5. Tag
`milestone/dissipation-abstraction` (4e662a1) marks Phase 1.

| # | Open item | Worth | Note |
|---|---|---|---|
| A1 | **M8 tag `vX.Y-av-foundation`** | H (user) | The unticked markers "Part 0 table filled" and "per-row prose" are satisfied by `docs/av/bakeoff_2026-10-09/{report,README}.md`; only the tag is left. Not mine to set |
| A2 | Phase 7b addendum rows (`riemann`, `riemannLimited`, `riemannLimitedB2`) | M | Run as part of the matrix sweep (X0), not as a separate addendum |
| A3 | OPEN §23: CompSPH + Rosswog, Sedov, \|v\| 3.6e5 × c₀ at step 122 | M | Does not touch the default (Monaghan rows complete). Suspect the cold-gas branch of the trigger (compare the Sphenix 0/0 guard), CompSPH's own viscosity form, or the point-source IC; first A/B: swap the switch only |
| A4 | The compressible dt has no approach / viscous term (OPEN §15 follow-up 2) | M | Cheap, applies to every shock row, and the lattice pile-up (B4) is invisible to an acoustic-only dt. Do it together with MFM's own signal-velocity dt (E1) as one change to `modules/timestep` |
| A5 | `C_l` / `C_q` are kernel-dependent in Frontiere Table D.1; ours is 1 / 1 for every kernel | M | User, 2026-09-30: revisit *with the limiter*, in the Riemann / MUSCL work. Folded into C5 |
| A6 | RT energy drift 4.3e-4 … 2.1e-3 in all 13 rows (gate 1e-4), "not investigated" | L | One A/B at half CFL tells whether it is the same RK2 time error as Sedov (OPEN §18, closed as expected). The Rosswog family is highest; check before calling it scheme-independent |
| A7 | `Monaghan1992`'s `1e-14 h` regulariser leaves an `r → 0` singularity (CompSPH's default) | L | CRK's `Frontiere2017` has the paper's `eps² = 1e-2` |
| A8 | Noh 2D post-shock density: only Wadsley passes ±3 % (-1.3 % / -3.0 %), and it is also the row whose pre-shock α stays ~0 | L | A link is "suggested, not tested". One run with Wadsley's detector forced to zero pre-shock α on another host would test it |

### Track B — CRKSPH (ex-CRKSPH_LIMITER_PLAN) — done, parked

**Done (2026-10-01).** Run-to-run nondeterminism fixed (an atomic scatter in the compatible-energy sum; `segment_sum`).
The limiter constants were in units of H where the paper's and Spheral's are in particle spacings; the default is now
derived per step as (1/n_h, 0.2/n_h). The Gresho vortex's kinetic-energy spin-up is a pressure pump of ~1 KE₀ over t = 3
from the non-central CRK pair pressure force, cancelled by the viscosity to within a few percent, converging only as
~nx^-0.4 so the net grows with resolution (-9 % / +2 % / +8 % at nx 48 / 64 / 96). Ruled out: the cusps, the pair
weights (`((V_i+V_j)/2)²` makes no difference), the support inconsistency, `C_l`. It is intrinsic to CRKSPH as formulated.
Guards: `tests/test_crkRegression.py`, `test_crkReproducible.py`, `test_crkLimiterDefault.py`.

| # | Open item | Worth | Note |
|---|---|---|---|
| B1 | **Does a Riemann flux through closed faces have no such pump?** | H | The core's Gresho runs say MFM / MFV show no secular spin-up (KE ends 0.80-0.92 of initial, L1 0.035-0.076) but are not like-for-like (Wendland C4, different neighbour count and stepper). The frontend Gresho at the same kernel / n_h as CRK and GSPH (E2) settles whether the pump is a property of CRK's pair force. Until then **do not spend more on the CRK limiter** |
| B2 | Local-spacing and softer-threshold limiter variants | L | Only matters with adaptive supports; parked 2026-10-01 |
| B3 | A symmetrised pair force that keeps linear reproduction (no tangential pressure work) | L | Research. The projected-central diagnostic removed the spin-up but collapsed the vortex |
| B4 | Same-lattice triple point: one column pair per interface merges (OPEN §15 residual) | L | Parked by user 2026-09-29; replication script in `scripts/`. Add to the MFM battery (E2) as a free comparison |
| B5 | Curved-wall stagnation (OPEN §19) | M | Owned by `COMPRESSIBLE_WALLS_PLAN.md`; not part of this plan (walls, see E.4) |

### Track C — Godunov SPH (ex-GODUNOV_SPH_PLAN) — open

**Done (2026-10-08).** L1 reconstruction of `rho`, `P` to the pair midpoint; L1b the papers' state limiters
(`VanLeerHarmonic` is the `GSPHConfig` default) and the first-order shock switch; L2 simplified GSPH (no AV; contact pressure
blip 6× smaller than the AV rows, best 1D velocity error, second-order states essential, weaker on Gresho and the Sedov
radius; time-centred energy was a negative result, 30× the drift); L3 Inutsuka 2002 (Gaussian density, cubic `V_ij²`,
`s*`; Gresho 0.076-0.091 and Sod velocity 0.0030 with no AV, but the Fig. 1 density-jump force is *not* removed at the
stable width η ≤ 1, and strong-shock energy needs the first-order switch).

| # | Open item | Worth | Note |
|---|---|---|---|
| C1 | Blob test and Wengen KH on `InutsukaGSPH` (new cases) | M | Murante 2011's discriminators between the simplified and the convolution form |
| C2 | Solver ablation (Murante Table 1) incl. Puri 2014's Roe-averaged HLLC speeds | M | Our HLLC uses Toro's PVRS speeds, which put Sod's contact speed at 0.61 against the exact 0.93; the core found the same with PVRS and a different (still inexact) value with Roe/Gaburov. Part of the Riemann consolidation X1 |
| C3 | Density-jump consistency vs pairing | L | Inutsuka's remedy (Eq. 81 plus a smaller width) kills the gain. Open science question; until it is settled `InutsukaGSPH` is a baseline, not a reference |
| C4 | Rows in the matrix | M | = X0 |
| C5 | Limiter family and `C_l`/`C_q` studied together (A5) | L | Observation from 7b.2: Frontiere's limiter `4r/(1+r)²` has slope 4 at the origin, above the Sweby TVD bound `ψ ≤ 2r`; whether that matters for a pair extrapolation is "not established" |
| C6 | Pluggable high-order interface states (warpSPHCore `Reconstructor.interfaceStates`, TENO / WENO) in the L1 slot | L | The core's Phase 6-7 waits on exactly this frontend coupling (base scheme decided 2026-10-09: the existing Godunov SPH). Park until C1-C2 are done; needs the user's priority call |
| C7 | Papers not on disk: van Leer 1997, Sirotkin & Yoh 2013, Dukowicz 1985, Rider 2000 | L | `literature/TO_ACQUIRE.md`; only needed for C2 |

### Track D — PESPH — in progress ([`PESPH_PLAN.md`](PESPH_PLAN.md), kept as is)

Status: Phase 0 done (enum, variant, EOS wiring found unnecessary, state tags fixed); Phase 2 part 1 done
(`AdaptiveSupportScheme.NumberDensity`, Newton, quadratic convergence, `tests/test_numberDensitySupport.py`). Next:
Phase 1 (`P̄` by summation), then `∂n/∂h`, `∂P̄/∂h`, `fH_ij` and the Kidder gate, then the momentum / energy kernels,
assembly, validation. The "gated on AV_PLAN M8" line is moot (user, 2026-10-09: the gate is minor); the AV configuration
PESPH should run with is the Track A default.

**Shared with Track E** (do these once): `AdaptiveSupportScheme.NumberDensity` is exactly the `n h^d = const` constraint
MFM uses for its supports (`V = 1/Σ_j W_ij`, `h = (N_ngb V / C)^{1/d}`); `totalEnergies` promoted from `constant()` to
`integrated('dEdt')` serves PESPH-E and MFM / MFV alike (`dEdt` already exists on the update); both need the "new
per-particle state needs a copy in `finalize`" rule (§7).

### Track E — MFM / MFV: migrating from warpSPHCore into the frontend

#### E.0 Why, and what kind of bet it is

MFM / MFV are the other branch of the SPH-vs-meshless fork (Hopkins 2015; GIZMO, SWIFT, OpenGadget3; FIRE-2/3 run MFM
because a particle stays a fixed parcel of mass). Three reasons for this repo: (1) the CRKSPH Gresho pump (B1) may not exist
in a Riemann-flux scheme; (2) MFM is CRKSPH's conservative differencing with the averaged flux replaced by a Godunov flux
(Frontiere Eq. 35 vs the MFM face vector; a structural correspondence, not an equivalence: different `ψ`, plus slope-limited
reconstruction), so the face machinery overlaps `modules/crk`; (3) GIZMO-MFM parity is the stated long-term goal (user,
2026-10-09). **The caveat that stays:** MFM is permanently outside the geometric-integration programme
(`warpSPHIntegrators/SPLITTING_PLAN.md`): its force depends on `v` through the Riemann states, so there is no potential,
no separability, no symplecticity. PESPH-A is the scheme that feeds that programme; MFM is the astro-fidelity bet. They do
not compose, and doing PESPH first keeps both open.

#### E.1 Inventory and where each piece should live

The convention in these repos: warpSPHCore holds **operators** (structured ABI, adjacency traversal, AD / JVP where
meaningful); the frontend holds **schemes** (state, config, step function, cases, pair loops that belong to one scheme, e.g.
`modules/godunov/wp_gsph.py`). The core's own Phase 8 note says "the solver, its warp pair-loop kernels and boundary
handling are the frontend's" — the warp kernels were then built in the core to prove the algebra at scale and for the GIZMO
comparison. The split below follows the convention and the seam that already exists in `wp_mfm.py`.

| Core piece | Lines | Role | Proposed home |
|---|---|---|---|
| `kernels.py`, `precision.py` | 114 / 48 | torch kernel weights; float32-aware thresholds (`faceCondMax` 1e6 f64, 2.1e4 f32; `pinvRtol`; floors 1e-30) | **stay** (precision policy is generic; the frontend imports it) |
| `geometry.py` (`MeshlessGeometry`) + `limiters.py` + `riemann.py` + `scheme.py` (`mfmRates`) | 289 / 93 / 188 / 171 | the **differentiable torch reference**, 55 tests | **stay in core as the oracle** (reverse mode and forward-mode AD work through it) |
| `wp_mfm.py` moments, gradients, limiter (App. B), closure residual / weights / matvec, coarse stencil | ~700 of 1130 | per-particle geometry: `omega`, `Ê⁻¹`, `N_cond`, matrix-LSQ gradients, the closure potential | **stay** (scheme-agnostic meshless geometry; the future home of the Godunov-SPH matrix-gradient option, X2) |
| `warpBackend.py` (`MeshlessWarp.build`, closure CG), `multigrid.py` | 210 / 231 | closure solve (graph-Laplacian Poisson, cell-aggregation multigrid preconditioner) | **stay** (generic numerics; 13-28 CG iterations from 16k to 1M particles) |
| `wp_mfm.py` limiter and flux stage: `mfmLimiter_Kernel` (App. B slope limiter), `alphaScalar`, `pairLimit`, `reconstructSide`, `sideTimeDerivative`, `faceFluxPair`, `hllcStar` + speeds (Gaburov / Roe / PVRS), `mfmFlux_Kernel` incl. the MFV mass-flux limiter and the first-order retry | ~500 | **the scheme**: limiting, reconstruction to the face, Hancock prediction, Riemann solve, mass-flux limit, conservative assembly | **moves to the frontend** (user, 2026-10-11: the mass-flux code is a scheme decision, not a core principle, and the frontend already holds the limiters): `modules/meshless/`, built on the consolidated `modules/riemann` (X1) and `modules/reconstruction` |
| `mfmTimestepWarp`, `signalTimestep` | ~50 | `2 C h / v_sig` per particle | **moves** into `modules/timestep` (together with A4) |
| `harness/mfm_driver.py` | 91 | reference stepper (single-stage Hancock, trapezoid drift, lagged supports) | **becomes** `schemes/meshless.py`; the driver stays in the core harness as the oracle stepper |
| `harness/gizmo_*.py`, `run_mfm_*.py`, `bench_mfm_backends.py` | — | evidence and GIZMO comparison | **stay** in core; the frontend reproduces their numbers (gate G1) |

The moved kernels call geometry `@wp.func`s that stay in the core (`pairFace`, `guardedFace`, `faceVector`, and the per-particle
`omega` / `Ê⁻¹` / gradient outputs), so **E0 promotes those to a documented public API** of `warpSPHCore.mfm` and the frontend's
kernels import them rather than copying them.

#### E.2 Stages and gates

Stage E4 (walls) was removed 2026-10-11 (E.4); later stages were renumbered. Sizes are guesses (S ≤ 1 day, M 2-4 days, L ~ a week); none of this has been measured.

| Stage | What | Gate | Size |
|---|---|---|---|
| **E0** core hand-over | Tag the core (`mfm-oracle-<date>`); export the `@wp.func`s above and `MeshlessWarp`'s stage functions as the public API; release the core as **0.6.1** (user, 2026-10-11: `mfm` is unreleased past 0.6.0, and 0.7 waits until everything works and is ready to ship); raise the frontend pin from `>=0.5.0` to `>=0.6.1`; the core branches `mfm-backend`, `mfm-guards`, `mfm-precision` are already contained in core `main` (`git branch --merged`), so they can be deleted whenever you like | Nothing changes numerically; core `tests/operations/test_mfm*.py` green | S |
| **E1** thin scheme, nothing moves | `CompressibleSPHScheme` member(s), `MeshlessConfig` (+ dict round trip with `.get` defaults), `MeshlessState` / system, `schemes/meshless.py` calling `MeshlessWarp.build / rates / timestep`, `SchemeBundle`, runner banner, scheme string for `av_report`. Settle the design points in E.3 | **G1 (passed 2026-10-11):** frontend reproduces the core harness: periodic Sod N = 200 / 400, all four mode × order combinations, L1(ρ) 1.4652e-2 / 7.2854e-3 (MFM order 2; the recorded 1.4188e-2 / 7.9946e-3 predate the oracle tag) and conservation ~1e-15 in float64 (all with `temporalScheme = 'hancock'`, the core driver's stage — the D4 default is now per-mode, MFM → SSPRK2 / MFV → Hancock, so the default MFM no longer reproduces the core harness; see the D4 status-log entry); Gresho N = 4096 L1(v) 0.0689 (MFM) / 0.0569 (MFV) (the recorded 0.066 / 0.051 predate the tag) within the chaotic spread (~10 %). Evidence: `scripts/g1_mfm_sod.py` / `scripts/g1_mfm_gresho.py` | M |
| **E2** frontend battery | MFM and MFV through the 12-case battery + KH 256 + triple point + blob, at the *frontend's* kernel and n_h; float32 and float64; jittered ICs (X4). **Answers B1** | **G2:** every case finishes without alarms; energy at round-off; Gresho KE has no secular spin-up at nx 48 / 64 / 96 at CRK's kernel / n_h. If it does spin up, B1's "the pump is CRK-specific" is refuted and that is recorded instead | M |
| **E3** move the limiter and flux stage | `modules/meshless/` (limiter, face reconstruction, `wp_flux.py`) on the consolidated Riemann library (X1) and the frontend's limiter module; the retry chain and MFV mass-flux limiter come with it; an EOS hook so the star state is not ideal-gas-only (the core needs `starFn` for that; the frontend's `modules/eos` is the natural source) | **G3:** limiter factors, per-pair flux and `dQ/dt` equal the core's `mfmLimiterWarp` / `mfmFluxWarp` to float32 precision (bitwise where the operation order is kept); the core's bitwise run-to-run reproducibility survives (its test, ported); G1 and G2 numbers unchanged | M-L |
| **E4** performance | The closure CG loop has host syncs (`bool(r.norm() <= target)`) and builds a multigrid per step, so the step cannot be CUDA-graph captured and will fall back silently (CLAUDE: keep the step sync-free). Options: fixed iteration count with warm start (`lam0`), on-device convergence flag | ms/step against GSPH / CRK on the Marrone-scale compressible cases; core numbers for reference: 54 ms rates and 0.21 s/step closure at 10⁶ particles | M |
| **E5** retire core duplicates | Decision D2 (open, "see first"): once the frontend is the only consumer of the warp limiter / flux stage and G3 has passed, tag the core, then remove `mfmLimiter_Kernel`, `mfmFlux_Kernel` and `rates()` there and keep the torch reference and the harness. Same pattern as the boundaries-repo retirement (tag keeps the oracle) | G1 on the tag | S |
| **E6** extensions (core's own list, none scheduled) | `psi_1` / `psi_2` selectable per row (0.25 vs GIZMO's 0.375), hierarchical time steps, entropy switches, first-moment-consistent faces (the disorder floor), adjoint / JVP of the warp kernels, the unclosed core's residual 10× GIZMO's, MFV Gresho trend with jitter, error bars from perturbed ensembles | — | — |

#### E.3 Design points E1 must settle

1. **State vector.** The conservative scheme's variable is `Q = (m, m v, m e_tot)`. Easiest conservative-to-round-off
   choice: integrate `Q` (momentum and total energy; mass only for MFV) and derive `ρ = m/V`, `v`, `P`, `u` each step.
   That needs `totalEnergies` integrated (shared with PESPH-E, Track D) and a momentum / mass field; for MFM the equivalent
   is `dvdt = (dp/dt)/m` with fixed `m`, but `u = E/m - v²/2` then gets its rounding from a difference of large numbers.
2. **Temporal scheme.** *Resolved (D4, 2026-10-11): per mode.* Measured on the 1-D sound wave (cfl sweep) and Gresho
   N = 4096 against the core driver: **MFM → SSPRK2** (rates at `dt = 0`, two stages — the temporal error is 10-30× smaller
   than Hancock's on the wave, Gresho L1 −21 %) and **MFV → Hancock + Euler** (the core driver's single stage — best at
   every cfl for MFV, whose mass-flux limiter carries a mode-specific excess SSPRK does not remove). **KDK rejected**: the
   half-kick drift is not time-centred (6.5-50× the Hancock error on the wave, order ~1.25). The step now owns the whole
   time step (the exact-delta contract of `schemes/artificialCompressible.py`) and a `temporalScheme` config option
   (`'auto'` = the per-mode default; `'hancock'` / `'ssprk'` / `'kdk'` pin a scheme) keeps every path — `'hancock'` is the
   G1 port-fidelity setting. Evidence: `scripts/d4_mfm_wave.py` / `scripts/d4_mfm_gresho.py` / `scripts/d4_mfm_parity.py`.
   The MFV frame velocity must be time-centred or MFV drops to first order (core finding: 1.0-1.2 vs 1.9).
3. **Supports.** Use `NumberDensity` (Track D) at fixed N_ngb instead of the driver's lagged `h <-> V`; the core found a
   stale `h` makes the closure residual first-order sensitive and fakes a "paper's A stalls on a lattice" result.
4. **Initial masses.** Set `m_i = ρ_i V_i` from the effective volumes (what the GIZMO comparison did); the sampler's
   `ρ dx^d` makes `m/V` differ from the IC density at disorder and jumps.
5. **Time step.** Per-particle `2 C h / v_sig` (Eq. 24-25), global minimum; one change with A4.
6. **Closure warm start** (`lam0`) must live in the state *and* be copied in `finalize`, or it resets every substep.
7. **Precision.** The frontend defaults to float32, where the thresholds differ (`faceCondMax` 2.1e4, closure `cg_tol` 1e-4);
   validate in both. Closure tolerance is set by the signal: ≤ 1e-4 for Gresho, 1e-6 for a 1e-4-amplitude wave on disordered particles.
8. **Enum.** Separate `CompressibleSPHScheme.MFM` and `.MFV` members (user, 2026-10-11), sharing one module and a config base class; the mode is fixed by the member, not a switch.
9. **AD.** The warp kernels are forward only; MFM in the frontend is not gradcheckable until the core item in E6 lands. The
   torch reference remains the differentiable path. State this in the config docstring.
10. **Hancock prediction frame.** Lab-frame prediction + Gaburov speeds together (GIZMO's combination) is what fixed MFV
    (Gresho 128², 0.108 → 0.082; GIZMO 0.069); each alone made both modes worse. Do not "improve" one without the other.

#### E.4 Out of scope

**Walls and boundaries.** Compressible walls are not a settled scheme in this repo yet (`COMPRESSIBLE_WALLS_PLAN.md` is still
exploring them), and analytic boundaries with compressible SPH go well beyond this plan (user, 2026-10-11). MFM / MFV run
periodic and free (Sod, Gresho, KH, RT, triple point, Sedov, Noh are all wall-free). One untested idea is parked, not planned:
at a boundary cell the closure residual `-Σ_j A_ij` is the area vector of the missing wall face, so a wall could be that face
carrying a reflected-state Riemann flux, with no wall particles and no lattice gaps (the failure mode of OPEN §19).

### Cross-cutting infrastructure

| # | Item | Worth | Note |
|---|---|---|---|
| X0 | **The matrix harness** (§2): add the `riemann*`, GSPH, MFM / MFV (and later PESPH) rows and the new cases to `scripts/av_report.py`; one overnight sweep | H | Smoke pre-flight once per day after src changes (memory `bakeoff-smoke-once-per-day`) |
| X1 | **One Riemann library.** Frontend `modules/riemann` (Acoustic / PVRS / TRRS / TSRS / Adaptive / HLLC, HLLC on PVRS speeds, `tests/test_riemann.py` against Toro's five problems, `gradcheck_riemann.py`) and the core `mfm` (HLLC with the Gaburov → Roe → PVRS → Rusanov chain, star-pressure validity retry, MFV bound) overlap. Both saw HLLC's star state inexact for strong waves (Sod: P\* 0.198 vs 0.303 exact, S\* 0.68 vs 0.927 with Roe speeds; 0.61 with PVRS) | H | Add `Gaburov`, `Roe`, Puri's speeds and the retry chain to `modules/riemann`; keep AD-safe (adjoints must not divide before the guard). Used by 7b, GSPH, InutsukaGSPH and E3. Prerequisite for E3 and C2 |
| X2 | **Matrix-inversion gradients for the SPH schemes** (PESPH_PLAN §7.5, "MAGMA2-style") | M | Helps Monaghan / CompSPH / GSPH / PESPH and is the meshless gradient already; `computeVelocityJacobian(corrected=…)` and OPEN §22's fix are the start. Test whether the core's `mfmGradientsWarp` as the L1 gradient improves GSPH L2 / L3 |
| X3 | Number-density support + `totalEnergies` integrated + `finalize` copy rule | H | Shared by D and E (see Track D) |
| X4 | **Disordered ICs** in the battery: jittered lattice and a glass / relaxed set | H | Three separate findings point here: MFM's closure is worth 3-5 orders on disordered particles (a uniform state on jittered particles does not converge with N without it, `gizmo_wave.py --static`; GIZMO's own first step already moves such a state by ~1e-2 in velocity), the CRK limiter plan's open "glass ICs", and the triple-point same-lattice artefact (B4). Every row of the matrix should run once on a lattice and once disordered. Check what `warpSPHCore.sampling` / `sample/` already provide before writing a generator |
| X5 | Performance / CUDA graphs for the compressible schemes | L | OPEN_PROBLEMS §33; E4 is the MFM instance |

## 4. Sequencing

```
E0 ─ E1 ─ E2 ───────────────┐
       │     └─ answers B1  ├─ X0 first matrix sweep ─ C1/C2/A2 rows
X3 ────┘                    │
X1 ───────── E3 ─ E4 ─ E5            D (PESPH Phases 1-5) runs alongside, sharing X3
```

1. **E0 → E1 → E2.** Largest new capability for the least new code (the core work is paid for), and the only way to
   answer B1. Nothing else waits on it.
2. **X3** is a few lines both D and E need; do it with E1.
3. **X0**: one sweep with every row that exists (including the never-run 7b rows and the GSPH rows) plus MFM / MFV. Build
   first, sweep once.
4. **X1 → E3**: consolidate Riemann before moving the flux kernel, otherwise the move copies the duplication into the frontend.
5. **E4** (performance) and **E5** (retire) follow once G3 holds.
6. **D (PESPH)** continues in parallel; it is independent until Phase 4 (assembly).
7. Everything marked L waits. Walls are out of scope (E.4).

A1 (the tag) and A3-A4 are independent small items to slot in anywhere.

## 5. Decisions (user, 2026-10-11)

| # | Decision | Outcome |
|---|---|---|
| D1 | Core / frontend split | **Changed from my proposal:** the limiter and flux stage, including the MFV mass-flux code, move to the frontend (a scheme decision, and the frontend already holds the limiters). Geometry, matrix gradients and the closure solve stay in the core |
| D2 | Does the core keep its warp limiter / flux stage after E3 | **Open, decided at the G3 gate** ("see first") |
| D3 | Enum shape | **Separate `MFM` and `MFV` members** |
| D4 | Temporal scheme default (E.3 item 2) | **Per mode (user, 2026-10-11):** MFM → SSPRK2, MFV → Hancock + Euler; a `temporalScheme` option (`'auto'` / `'hancock'` / `'ssprk'` / `'kdk'`) keeps all paths, `'hancock'` remaining the G1 port-fidelity setting; KDK rejected on the measurement |
| D5 | M8 tag (A1) | Yours to set; acknowledged, not set |
| D6 | Archive moves in §0 | Kept |
| D7 | Merge `dev` into `main` | Acknowledged; stays on `PLANS.md`'s list |
| — | Core release | **0.6.1**, not 0.7.0: 0.7 only once everything works and is ready to ship |
| — | Walls | Out of scope (E.4) |

## 6. Where the evidence is

`docs/av/` — bake-off (`bakeoff_2026-10-09`), sweep (`sweep_2026-10-08`), KH 256, Riemann 7b, Godunov L1-L3 and density jump,
CRK switch, baselines. `docs/support_solver/`. Archived plans carry the long logs: `docs/historic_plans/{AV_PLAN,
CRKSPH_LIMITER_PLAN,GODUNOV_SPH_PLAN}.md`. MFM evidence is in warpSPHCore: `higher_order.md` Phase 8 and
`higherOrderSPH/harness/` (`run_mfm_*.py`, `gizmo_*.py`, `bench_mfm_backends.py`). The frontend's G1 evidence is in this
repo: `scripts/g1_mfm_sod.py` (periodic double Sod vs the exact Riemann mirror) and `scripts/g1_mfm_gresho.py` (Gresho-Chan
vortex vs the analytic profile); both mirror `mfm_driver.step` with `temporalScheme = 'hancock'` and gate against the
re-blessed oracle `mfm-oracle-2026-10-11` (the harness directory is overridable with `WARPSPHCORE_HARNESS`). The D4
temporal-scheme measurement is in this repo: `scripts/d4_mfm_wave.py` (1-D sound wave, cfl sweep), `scripts/d4_mfm_gresho.py`
(Gresho N = 4096) and `scripts/d4_mfm_parity.py` (the refactored step vs the probes, one step: exact for hancock, ~1e-24
for ssprk / kdk). The battery results land here when E2 produces them.

## 7. Carried-over gotchas (from the archived plans)

- **Float32 contraction:** with a live switch (α ≠ 1), moving arithmetic across `wp.func` boundaries changes FMA fusion at
  1 ulp; only α = 1 paths and the Monaghan-host smoke are bit-identical across refactors.
- **CRK Gresho at t = 3 amplifies a 1e-13 perturbation to ~1e-3 even in float64**; MFM Gresho at N ≥ 16384 is likewise
  noise-limited. Do not read a Gresho L1 change below ~5 % as signal; use perturbed ensembles for error bars.
- **Reverse AD traps:** a loop-carried `wp.max` silently zeroes the adjoint (see `wp_vsig.py`); un-guarded divisions must
  return early, not be NaN-patched afterwards (`modules/crk/limiter.py`).
- **Before/after tool:** `git worktree add <scratch> HEAD` plus `PYTHONPATH=<wt>/src`.
- **Registering a scheme** (pattern: GSPH, verified by grep): `enumTypes.py`, `configurations/<x>.py` + `configurations/__init__.py`
  (dict round trip with `.get(…, default)`), `schemes/<x>.py` + `schemes/__init__.py`, `schemes/builder.py` (`SchemeBundle`);
  new per-particle state needs a field on the state class *and* a copy in the system's `finalize`, or it resets each substep.
- **Run rules** (CLAUDE.md): video on, progress streamed, stall watchdog, one GPU run at a time.

## Status log

- 2026-10-11: plan opened by consolidating AV_PLAN, CRKSPH_LIMITER_PLAN, GODUNOV_SPH_PLAN and the core MFM work; PESPH_PLAN
  retained and retargeted. Read against: the four plans, `PLANS.md`, `OPEN_PROBLEMS.md` §15 / §19 / §23, the core's
  `src/warpSPHCore/mfm/` and `higher_order.md` Phase 8, `schemes/gsph.py`, `schemes/builder.py`. No code changed, no runs.
- 2026-10-11 (later): your answers to §5 applied. The limiter and the MFV mass-flux code move to the frontend with the flux stage (D1 changed); MFM and MFV are separate enum members (D3); D2 is decided at the G3 gate; the core release is 0.6.1; the wall stage was removed and walls are out of scope.
- 2026-10-11 (E0): core hand-over done locally (push left to you). The `warpSPHCore.mfm` public API now documents the face `@wp.func`s (`faceVector`, `guardedFace`, `pairFace`), the per-particle moments / gradients (`mfmMomentsWarp`, `mfmGradientsWarp`) and the stage entry points (`mfmLimiterWarp`, `mfmFluxWarp`, `mfmTimestepWarp`); the core is 0.6.1, tagged `mfm-oracle-2026-10-11` (local only); the frontend pin is `>=0.6.1`; the merged core branches `mfm-backend` / `mfm-guards` / `mfm-precision` were deleted locally (their `origin/*` refs remain). Gate: `tests/operations/test_mfm{,_warp}.py` 110 passed before and after; no numerical change. Next: E1 (thin MFM / MFV scheme, separate enum members) together with X3, then E2 (the battery; answers B1).
- 2026-10-11 (E1 + X3): the thin MFM / MFV scheme is in, and G1 passed. E1: `CompressibleSPHScheme.MFM` / `.MFV` (separate members, D3), `MeshlessConfig` (+ dict round trip with `.get` defaults), `MeshlessState` integrating the conserved `Q = (m, m v, m e_tot)` (your decision; `ρ = m/V`, `v`, `P`, `u` derived per step), and `meshlessStep` mirroring the core driver's single-stage step (Hancock-predicted rates over the full dt, Q Euler, midpoint position drift with the un-predicted velocity, h lagged one step from the pre-move volume); the closure warm start `lam` lives in the state and is copied in `finalize`; `timeCentredFrame = True` by default; supports from the driver's `supports_from_volume`. E.3 item 3 (the `NumberDensity` support scheme) is deferred to E2. X3 (`12c0bac`): `totalEnergies` integrated as `dEdt` in `CompressibleState`; the kidder / linearWave ICs mass-weighted; kidder smoke bit-identical. **G1 (port fidelity, re-blessed):** the plan's Sod / Gresho targets predate the oracle tag — both core backends (warp and torch, agreeing to ~1 ulp) give the frontend's numbers, so the targets were re-blessed from `mfm-oracle-2026-10-11`. Periodic double Sod (Wendland2, n_h 7, t 0.12, f64, N = 200 / 400, steps in parentheses): MFM o1 2.1891e-2 (112) / 1.2688e-2 (217), MFM o2 1.4652e-2 (133) / 7.2854e-3 (253), MFV o1 2.1051e-2 (90) / 1.3247e-2 (185), MFV o2 1.0202e-2 (113) / 5.1568e-3 (222); conservation ≤ 1.3e-15; every number and step count equals the core. Gresho (N = 4096, Wendland4, n_h 30, project, o2, t 3.0): MFM L1(v) 0.0689, KE 0.8412 (max 1.0077), |dQ| 8.7e-17, 1187 steps; MFV L1(v) 0.0569, |dQ| 1.1e-16, 1355 steps — identical to the core. One documented artifact: MFV Gresho KE 0.9328 / 1.0221 (frontend) vs 0.8762 / 1.0074 (core) — a ~1e-16 cross-process IC difference (the closure warm start returns below tolerance without iterating, so the first closure differs at round-off) amplified by the MFV limiter's round-off sensitivity over 1355 steps; MFM shows no such amplification, and the step function and step loop are proven bitwise-faithful (`check_step_function` exact; a 1400-step lockstep against the core driver shows |dQ| = |dpos| = |dvol| = 0 at every checkpoint). Evidence: `scripts/g1_mfm_sod.py` and `scripts/g1_mfm_gresho.py` (the harness directory is overridable with `WARPSPHCORE_HARNESS`). Still pending: the sound-wave / D4 temporal-scheme measurement (Hancock + Euler vs SSPRK vs KDK) and the runner banner / scheme string for `av_report`. Next: D4, then E2 (the battery; answers B1).
- 2026-10-11 (D4): the temporal-scheme measurement (E.3 item 2) is done and D4 is decided **per mode** (your choice): MFM defaults to SSPRK2, MFV keeps Hancock + Euler; a `temporalScheme` config option (`'auto'` per-mode default, `'hancock'` / `'ssprk'` / `'kdk'`) keeps every path, `'hancock'` remaining the G1 port-fidelity setting. The step now owns the whole time step (the exact-delta contract of `schemes/artificialCompressible.py`) — it performs the resolved scheme internally and hands the integrator the exact per-step deltas, so the integrator must be single-evaluation (`forwardEuler` / `explicitEuler`; `validateIntegrationScheme` refuses the rest, which would run the step once per stage and blend). The production path's drift is now the core driver's midpoint drift (the E1 thin step's forward drift was a latent mismatch). **Measurement (f64, the D4 probes):** 1-D sound wave, N = 200, relative L2(ρ) at T = L/c, [cfl 0.5 / 0.4 / 0.2 / 0.1 / 0.05]: MFM hancock 1.993e-3 / 1.962e-3 / 1.576e-3 / 1.422e-3 / 1.430e-3 (floor ~1.34e-3), MFM ssprk 1.601e-3 / 1.519e-3 / 1.544e-3 / 1.548e-3 / 1.538e-3 (flat at the floor — the temporal part is 10-30× smaller), MFM kdk 2.832e-2 / 2.125e-2 / 1.019e-2 / 5.216e-3 / 2.912e-3 (order ~1.25, 6.5× hancock at cfl 0.2); MFV hancock 1.933e-3 / 1.715e-3 / 2.610e-3 / 3.513e-3 / 3.958e-3 (best at every cfl), MFV ssprk 3.963e-3 / 4.123e-3 / 4.457e-3 / 4.440e-3 / 4.433e-3, MFV kdk worst. Gresho N = 4096, L1(v) [cfl 0.2 / 0.1]: hancock MFM 0.0689 / 0.0585, hancock MFV 0.0569 / 0.0599, ssprk MFM 0.0547 / 0.0554, ssprk MFV 0.0616 / 0.0579, kdk MFM 0.0594 / 0.0574, kdk MFV 0.0565 / 0.0673 (cfl 0.1 moves L1 by ≤ 11 % — the spatial floor at N = 4096 dominates). Findings: KDK is decisively worst (the half-kick drift is not time-centred); the MFV mode-specific excess over the MFM floor (~2.6e-3 on the wave, a weakly dt-growing p < 1 component) is a core-side property (the core torch harness gives the identical 2.610e-3 at cfl 0.2, n = 200), common to all frontend candidates; h refresh (h-iters 2) is immaterial at this amplitude. **Production path (the refactored step, forwardEuler, dt from `g.timestep`):** one-step parity vs the D4 probes is exact for hancock (|dQ| = 0 — dQdt is the raw rate) and within ~1e-24 for ssprk / kdk (the exact-delta divide-then-multiply round-trip; dx / h bitwise) — `scripts/d4_mfm_parity.py`. G1 re-confirmed with `temporalScheme = 'hancock'`: Sod MFM o2 1.4652e-2 / 7.2854e-3, Gresho MFM 0.0689 / MFV 0.0569, every `stepfn(dQ,dx) = True,True`. New per-mode default (MFM → ssprk, MFV → hancock): Gresho MFM 0.0547 (vs 0.0689 hancock, −21 %) / MFV 0.0569 (unchanged); Sod (o2, N = 200) MFM 1.3681e-2 (vs 1.4652e-2) / MFV 1.0202e-2 (unchanged). Evidence: `scripts/d4_mfm_wave.py` / `scripts/d4_mfm_gresho.py` / `scripts/d4_mfm_parity.py`. Still pending from E1: the runner banner / scheme string for `av_report`. Next: E2 (the battery; answers B1).
