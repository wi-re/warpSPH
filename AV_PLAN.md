# warpSPH — Artificial-Dissipation Roadmap (AV_PLAN)

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

**Nothing in this plan is started.** What exists is its *input*: the work logged
in [`phase6.md`](phase6.md) / [`phase6_shock_capturing_log.md`](phase6_shock_capturing_log.md),
complete 2026-09-19, which wired Cullen & Dehnen (2010) into
[`schemes/monaghan.py`](src/warpSPH/schemes/monaghan.py), resolved the App. B
sign question (`D(∇·v)/Dt = ∇·(dv/dt) − tr(V²)`, minus), added
[`ReadHayfield2012.py`](src/warpSPH/modules/shockCapturing/ReadHayfield2012.py)
with its entropy-dissipation term, and validated Sod 1D/2D/3D plus the Gresho
control. Commits `dcd0423`, `652ed28`, `3cf40f5`, `128d82b`, branch `acsph-plan`,
not pushed.

That is the **Phase 0 baseline**: C&D and R&H on Monaghan, validated
qualitatively and with the contact-spike metric, but with no report, no L1, no
dissipation accounting and no reproducibility lock.

| Phase | Milestone | State |
|---|---|---|
| 0 | M0 Baseline locked | ☐ not started |
| 1 | M1 Old physics, new architecture | ☐ not started |
| 2 | M2 Entropy trigger validated | ☐ not started |
| 3 | M3 Reconstruction engine works | ☐ not started |
| 4 | M4 Smooth-flow dissipation characterised | ☐ not started |
| 5A | M5 Quadratic dissipation understood | ☐ not started |
| 5B | M6 Cheap modern switch characterised | ☐ not started |
| 6 | M7 Detector-complete | ☐ not started |
| 7 | 🏁 **M8 SPH-AV-FOUNDATION** — hard gate | ☐ not started |
| 8+ | → [`PESPH_PLAN.md`](PESPH_PLAN.md) | blocked on M8 |

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

```
Π_ab = { −(ᾱ_ab c̄_ab ω_ab + β ω_ab²)/ρ̄_ab   for ω_ab < 0 ; 0 otherwise }   (6)
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
S_i = { −h_i² max(∇̇·v_i, 0)   for ∇·v_i ≤ 0 ; 0 for ∇·v_i > 0 }             (21)
∇̇·v_i(t+Δt) = (∇·v_i(t+Δt) − ∇·v_i(t)) / Δt                                 (22)
α_V,loc,i = α_V,max · S_i / (c_i² + S_i),   α_V,max = 2.0                    (23)
α_V,i ← { α_V,loc,i                              if α_V,i < α_V,loc,i
        ; (α_V,i + α_V,loc,i Δt/τ_V,i)/(1 + Δt/τ_V,i)  if α_V,i > α_V,loc,i } (24)
τ_V,i = γ_K ℓ_V h_i / c_i ,  ℓ_V = 0.05,  γ_K = kernel gamma
```

Two things to carry into the port:

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
- [ ] `L1` lands and is unit-tested against a hand-computed case
- [ ] `compressibleDiagnostics` extended; all 14 compressible cases still run
- [ ] linear/quadratic AV split populated and non-zero on `sod`
- [ ] three probes promoted out of `.tmp/`
- [ ] `scripts/av_report.py --config baseline --profile full` produces
      `results/av_baseline_<stamp>/report.md`
- [ ] reproducibility lock verified twice
- [ ] report committed

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
- [ ] registry replaces both `elif` chains
- [ ] `CoefficientPolicy` with all three variants; **fixed β now expressible**
- [ ] `VelocityPairPolicy` with `RawVelocity` proven to be an identity
- [ ] `computePi_actual` takes the pair velocity as an argument
- [ ] §2.5 hazards cleared; PESPH_PLAN §2.1 cross-referenced as closed
- [ ] `NoneSwitch` bit-for-bit vs M0
- [ ] C&D and R&H within `1e-6` vs M0 on every report metric
- [ ] gradchecks green

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
- [ ] `Rosswog2020` replaces the stub; old spelling still parses
- [ ] `entropiesPrev` on the state **and** copied in `finalize`
- [ ] step-boundary unit test passes before the physics is tuned
- [ ] all `S(x)` / threshold / dimensionless unit tests pass
- [ ] gradcheck green
- [ ] Sod 1D/2D/3D table above
- [ ] Sedov approaches 4 from below
- [ ] Gresho `alphaMean < 0.05`
- [ ] resolution sweep: `alphaMean` non-increasing
- [ ] timestep sweep: α within 10%
- [ ] detector map (§0.3) rendered vs C&D on Sedov — rosswog2020entropy Fig. 4 is
      exactly this comparison
- [ ] report row added

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
- [ ] `modules/reconstruction/` exists with three policies
- [ ] limiters moved with AD guards intact
- [ ] `x_ij` sign discrepancy resolved against Eqs. (11)–(12), outcome recorded
- [ ] linear-field annihilation test passes at both tolerances
- [ ] limiter unit tests pass
- [ ] **pairwise conservation test passes at `1e-12`**
- [ ] gradcheck script added and green
- [ ] CRKSPH regression within budget (or the delta is quantified and explained)
- [ ] Monaghan + reconstruction runs clean

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
- [ ] Eq. (6) operator available with fixed `β = 2`
- [ ] standalone Balsara `B_a`; `Balsara1995` stub retired
- [ ] `BalsaraModulatedReconstruction` with configurable `p`
- [ ] Table 1 rows 1-6 all runnable from a config name
- [ ] Sod: AVSLRB2 `L1 ≤` AVSW
- [ ] **Sedov: S3/S4 overshoot reproduced, S6 does not**
- [ ] Gresho: strict ordering reproduced
- [ ] KH: `A(t=1.5)` ladder reproduced; SLR `≥ 1.4×` AVSW
- [ ] raw-vs-reconstructed AV energy on Gresho, with the linear/quadratic split
- [ ] detector map: `φ_ab` field vs α field
- [ ] report rows added for all six variants

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
- [ ] `scaleBeta` semantics documented; no behaviour change
- [ ] `135βh/(62παH)` ratio reported
- [ ] periodic shear box case added and registered in `CASE_MODULES`
- [ ] shock metrics within 5% under coupling
- [ ] Gresho angular-momentum loss and quadratic AV energy both reduced
- [ ] shear-box mechanism isolated
- [ ] **`C_q = 0` default resolved** — deliberate or an oversight, decided and recorded
- [ ] report rows added
- [ ] documented as *not* reproducing chen2025's disc result, with the reason

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
- [ ] `Sphenix2022.py`; enum, registry, config, dict round-trip
- [ ] Eq. (24) implemented implicitly; stability unit test at `Δt/τ = 100`
- [ ] Balsara in the pair coefficient, not the indicator
- [ ] gradcheck green
- [ ] shock metrics within 5% of C&D
- [ ] Gresho `L1(v_φ) ≤` C&D
- [ ] **ms/step ≥ 15% lower** (or the shortfall diagnosed)
- [ ] neighbour-loop count recorded
- [ ] `cflFactor × 4` stability demonstrated
- [ ] detector map vs C&D
- [ ] report rows added

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

- [ ] write down `h_repo / h_CD` and `h_repo / h_Gasoline2` for kernel `B7` and
      Wendland2, in terms of `sphKernelScale` / `sphKernel_xi`;
- [ ] derive the correct prefactor for `A_i` in this repo's units;
- [ ] state it in the module docstring with the derivation, the way
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
- [ ] **`h`-convention re-derivation done and documented (§6.2) — before tuning**
- [ ] `Wadsley2017.py`; enum, registry, config, `D_prev` state + `finalize` copy
- [ ] `T` keeps the trace; `W_R,ij` is the polynomial weight, not the kernel
- [ ] gradcheck green
- [ ] **uniform-compression test separates Wadsley from C&D**
- [ ] rotation / shear / sound-wave / step tests pass
- [ ] shock metrics within 5%
- [ ] Gresho `L1(v_φ) ≤` C&D
- [ ] five-detector comparison maps rendered on identical frames
- [ ] report rows added

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
- [ ] every matrix row runs from a named config
- [ ] Part 0 table filled for all rows
- [ ] per-row prose written
- [ ] default selected, with the reason recorded
- [ ] README states the reference configuration
- [ ] tagged `vX.Y-av-foundation`

**🏁 M8 — SPH foundation complete.**

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
