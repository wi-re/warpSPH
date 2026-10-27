# ACSPH: our implementation vs the authors' original code (dynGPUSPH)

> Plan: [ACSPH_PLAN.md](../../ACSPH_PLAN.md) (row in [PLANS.md](../../PLANS.md)). Written 2026-10-12.
> Nothing here was run; this is a read-through of both code bases. Every claim about
> the original code cites `file:line` at a named commit, so it can be re-checked.

## 0. Which code is "the original"

`~/dev/dynGPUSPH` is checked out on `main`, **which contains no ACSPH**. The dual-time
solver lives on two other branches:

| Branch | Last commit | What it is |
|---|---|---|
| `origin/Artificial-Incompressibility` @ `c7017ec` | 2023-05-16 | The AC development snapshot (110 commits since the fork on 2022-07-14, "Basic implementation"). Ends on a WIP commit ("Acoustic damper term"). |
| **`origin/FSI_devel` @ `2dbc27a`** | 2023-12-07 | Contains the branch above **plus 67 more commits**, among them "All changes 4 paper" (2023-11-13), "Use RK4 4 chap 4", "JST updates", "Working for dambreak" (2023-11-29), "Maybe the vpt terms work now", "Correct free/no slip visc and removed rubbish" (2023-12-03). The paper (CMAME 420, 2024) postdates it. |

**Treat `FSI_devel@2dbc27a` as the reference.** The May snapshot differs from it in ways
that matter (it has the `ṽ` advection terms ON, the wall repulsion force ON, Monaghan-type
wall/surface damping ON; the tip has all three off). Where the two differ below, both are
given. `main` (2023-11-04) is the WCSPH line and has no pseudo-time loop.

To re-read: `git -C ~/dev/dynGPUSPH worktree add --detach /path origin/FSI_devel`.

A side-by-side of every candidate (paper / ours / tip / compute) is in [CANDIDATE_CHANGES_OVERVIEW.md](CANDIDATE_CHANGES_OVERVIEW.md).

Paper-era code, all paths under `src/`: loop in `gpusph3d.cu:1090-1480`; pair physics
in `sph.cuh` (`getContrib2`); per-particle residual in `HashFunctions_cuda.cuh`
(`residualHash2D` ≈ 2279-2650); wall closure `noSlipHash3D` ≈ 3735-3900; shifting
`shiftandInterpolateD` ≈ 1830-2000.

## 1. What already matches (and two plan questions the code answers)

Our `schemes/incompressible/artificialCompressible.py` is a faithful port of the structure:

- Dual-time loop, BDF2 variable-step source frozen at the stage-0 value
  (`Get_PT_derivs`, `particles.cu`), `I_c = diag{0,1,1}`, `ṽ = v − Dx/Dt` as the
  position-row residual, explicit Butcher sweep with lower-triangular `A` and `b`.
- `β = CFL_τ h/Δτ`, `k1 = β²`, `k2 = 0.1 h β` (`HashFunctions_cuda.cuh:2421`), constant
  density and volume, difference-form `div v`, `(p_i+p_j)` pressure gradient,
  Monaghan-Gingold Laplacian with `K = 8/10` (`kernel.cuh viscous_term3`).
- Frozen pressure diffusion across RK stages, re-evaluated each dual-time iteration
  (`RKsubit == 0` branch).
- Adami wall pressure with the hydrostatic correction (`noSlipCell3`: `p_f + ρ_f (g + a_f − a_w)·r`).
  At the tip `a_f` is the surface-tension acceleration only (`Resv_wall = st`), so with ST
  off it is exactly our Eq. (61).
- External shifting with `x^n` re-labelled by the shift (their `deltaU_BDF` ≡ our `bdfShiftCorrection`),
  `v`, `p` projected to the shifted position with a renormalised gradient (`newP = p + δu·∇p`).
- Convergence metric is `‖ṽ‖₂ / N` with the **1/N** (not 1/√N): `getVectMagMinusError`,
  `HashFunctions.cu:1056-1076`. Plan §1.6 reproduced it verbatim; it is right.

**Answers to the plan's open questions** (ACSPH_PLAN.md Part 5, "ask the authors"):

| Plan | Question | Code says |
|---|---|---|
| §5.1 | `ε4 = min(0, κ4−ε2)` or `max`? | **`max`** (`HashFunctions_cuda.cuh:2469`). Our default is right. κ4 is 1/32 in the May snapshot but **0.1** at the tip. |
| §5.2 | Eq. (40) low-storage or general Butcher? | General: `RK_ca` lower-triangular + `RK_cb` (`gpusph3d.cu:905-914`). Ours matches. |
| §5.3 | stray `h` in `k2 D^p`? | No extra `h`: `Lapscale = 0.1·sr·β` multiplies a bare Laplacian (`:2421`). Ours matches. |
| §5.5 | what is `U_char`? | **`0.1·c0_ref`**, with `c0_ref` the input-file sound speed that also fixes `ν`: `max_vel = clamp(max|v|, 1e-5, 0.1·ssound)` (`gpusph3d.cu:1416`). Ours defaults `uChar=None` (= `|v|max`, "not what the paper does"). |
| §5.6 | Eq. (46) first term is a length | `dt_c = cfl·h` literally (`cfl.cu`), i.e. a reference velocity of 1. Ours (`REFERENCE_VELOCITY = 1`) matches. |
| `ṽ` terms (§1.7) | on or off? | May snapshot: **on**. Tip: **off** (`Resv` has `+divudu - vel*divdu` commented out, `ResP = -β²ρ roc + …` only). Paper and ours: off. Consistent. |

## 2. Differences that matter, most relevant first

"Align?" is my recommendation, not a decision. Effort is for our code.

### 2.1 The ones aimed at the contact-line / free-surface failure (WALL_CONTACT_PLAN, OPEN §8/§9)

| # | Original (tip) | Ours | Align? |
|---|---|---|---|
| **D1** | **Wall-row pressure is attenuated when a wall has an edge fluid neighbour.** `noSlipCell3` raises `issurf` if any fluid neighbour has `is_surf == 1`; then `if (issurf && P_inter < 0) P_inter *= lambda_sum/kernSum` (`HashFunctions_cuda.cuh:3871`), with `lambda_sum = Σ_fluid max(0, 2(λ_j−½)) W V`. The May snapshot hard-zeroes it (`P_inter = 0`). Re-evaluated every RK stage. **Wall rows only, negative only, only next to an actual edge particle.** | Nothing like it. `cavitationProjection` options are broader sets: `contact` = *all* wall rows + wall-adjacent fluid, `vicinity` = the dilated set. `wallPressures(clampNonNeg=False)`. | **Yes, first.** It is the one wall/free-surface mechanism the authors actually ship, and it is narrower than anything tried in WALL_CONTACT_PLAN §3. Cheap: a mask + a multiply in `wallPressureExtrapolation`. |
| **D2** | Surface detection, `λ`, normals, `L`, `∇p^L`, `Plap` are **recomputed every dual-time iteration** (`ResidualHash3`, `RKsubit == 0`: `deltaHash4D` then `detectSurfaceHashD`). | Frozen once per real step (`artificialCompressible_step`: "paper says nothing about re-detecting"). | **Yes, try.** D1 and D3 read `is_surf`/`λ`; frozen flags go stale as the pseudo-iteration moves particles. Cost: a detection per iteration, must stay sync-free for the CUDA graph. |
| **D3** | Pressure diffusion is **tapered by λ**: `ResP = −β²ρ roc + Laplacian · C2_interp(λ, 0.1, 0.4)` (`:2610`), a Wendland-C2 ramp that is 1 for `λ ≥ 0.4` and 0 for `λ ≤ 0.1`. | `k2 · D^p` everywhere. | **Yes.** One line; turns the diffusion off exactly on the poorly supported rows (thin film, isolated fragment) that carry the stale suction pressure. |
| **D4** | JST switch is a different formula and has **no surface override**: `J = 0.1 h² Σ_j |Plap_j − Plap_i|/|p_i+p_j|` (unweighted, `sph.cuh:724`), `ε2 = 0.5 min(2, J)`, `ε4 = max(0, 0.1 − ε2)`, **frozen at the first dual-time iteration of the step** (`JSTs[]`, `:2454-2469`); `ε2 = 1` at the surface is commented out (D3 replaces it). Paper/May: `χ` = pressure-ratio average, κ4 = 1/32, `ε2 = 1` in 𝕍. | The paper's version (`wp_jstSwitch.py`, `_jst`): `χ`, κ4 = 1/32, `biLap` forced on the dilated set. | **Only if running `.jst`.** AC-2L is the paper's default and the authors' `pmethod 0`; the JST switch is a secondary knob. Do D1-D3 first. |
| **D5** | Pseudo-time particle velocity is **bounced back** at walls: a fluid row within `0.25 Δx` of a wall particle whose `v·n < 0` has its normal component reversed *in the position-row residual only* (`Get_PT_derivs2_bounce_back`, `particles.cu:632`; flag set at `HashFunctions_cuda.cuh:2627`). Note `vel[i] - veli;` in that kernel is a no-op (typo for `=`), so `v` itself is untouched. | None. `noPenetrationShift` (mDBC) is off and not the same thing. | **Maybe.** Cheap, and it is the authors' no-penetration device for the *position* row, which is where our walled cases leak (hydrostaticColumn corner; plan step 5b). |

### 2.2 Convergence and pseudo-time control

| # | Original (tip) | Ours | Align? |
|---|---|---|---|
| **D6** | Stop test is **two-sided**: `log10(ε_v) < minres && log10(ε_p) < minres+1`, with `ε_p = ‖R_p‖₂/N · Δt/(ρ0·0.01·c0²)` (`gpusph3d.cu:1443`). | Velocity metric only. | Cheap. Adds a pressure gate; needs a reference `c0` (we have `referenceSoundSpeedForViscosity`). |
| **D7** | **Stall exit with a truncated metric.** If the last 5 `log10 ε_v` span < 0.1 % of their mean (`:1469`), switch to a metric that sorts the squared residuals and **drops the worst 5 %** of particles (`getVectMagSumError_Truncated`, `HashFunctions.cu:1091`); if the pressure history also spans < 0.05 % *and* `ε_v < minres`, stop (`:1470`). | Run to `maxPseudoIterations` (200) unless `ε_v < target`. The float32 plateau at −5.8 (`epsilonV` note in `configurations/artificialCompressible.py`) burns the iteration budget every step. | **Yes, as an option.** Directly addresses the "~187 ms/step, 200 iterations" cost (OPEN §33). Excluding the worst 5 % hides exactly the rows we care about at a contact line, so keep it off for contact-line diagnosis. |
| **D8** | `Δτ = Δt/(2·R)` for the **first two** pseudo-iterations, `Δt/R` after (`gpusph3d.cu:1105`). | Constant `Δτ`. | Cheap, try with D6/D7. |
| **D9** | RK3 uses `CFL_τ = 0.75` (`gpusph3d.cu:914`; the May snapshot had 1.0, the plan's table says 1.0). | `cflTau` default 0.5 for RK2; the plan quotes 1.0 for RK3. | Config only. Use 0.75 when comparing RK3. |
| **D10** | `U_char = 0.1 c0_ref` (see §1). | `uChar=None`. | Config only. Set `uChar = 0.1·referenceSoundSpeedForViscosity`. |
| **D11** | `α_PI = 1/(1 + c_b[s] Δτ·(3/2)/Δt)` with C++ **integer division `(3/2) = 1`** (`gpusph3d.cu:1178` and `:1353`): the BDF coefficient is 1/Δt not 1.5/Δt, and `c_b` (not the stage node) is used. | `α_PI = 1 + α_s Δτ α_t` (`usePointImplicit=True`). | Not worth copying the quirk. Paper says `α_PI = 1` works; for a like-for-like run set `usePointImplicit=False` and note that the authors' effective value differs from ours. |
| **D12** | Real timestep: `min(cfl·h, cfl·h/|v|max, 0.125 h²/ν)`, clamped to ×[0.8, 1.2]. **The force constraint is computed but commented out** (`cfl.cu:173,185`). | Same three terms **plus** `dt_accelerationConstraint` (on by default). | Config only: turn it off for a paper-faithful `Δt`. |

### 2.3 Extra dissipation in the momentum equation (code-only, none in the paper)

At the tip, `Resv` (`HashFunctions_cuda.cuh:2515-2544`) carries two things the paper does not:

- **Velocity JST dissipation:** `ε2v·β h ∇²v + ε4v·(−β h³ ∇⁴v)` with `ε2v = 0.01 min(10, J)`,
  `ε4v = max(0, 0.005 − 0.1 ε2v)`, blended to the 2nd-order term near walls with
  `C2_interp(walld, 2h, h)`. Uses `velLap` from `noSlipHash3D`.
- **Divergence damper:** `+ β h Σ(div v_i + div v_j)∇W V`, `∇(∇·v)` with the pseudo-sound speed.
  (Paper ref. [19], Sun et al.; the paper's own `k3` is the pressure-equation analogue and
  is zeroed. The May snapshot used the physical `c0` here.)

These are real stabilisers the published results may rely on, and they are the reason the
authors' code is quiet at free surfaces where ours has `ν` alone. **Align?** Not yet.
They add dissipation that is not in the paper's equations; try only if D1-D5 do not hold
the contact line, and A/B pure-fluid cases first (`boundary-fix-off-for-pure-fluid`).

### 2.4 Walls and shifting (lower priority)

| # | Original (tip) | Ours |
|---|---|---|
| **D13** | Wall particles carry an *extrapolated* velocity (Shepard of fluid `v`) **and an extrapolated `ṽ`** (`velPT[wall]`), wall normal from the wall-particle kernel-gradient sum (flipped inward); the free-slip mirror is applied *per pair* (`FREE_SLIP` in `getContrib2`). Viscous pair term uses `DAMPING_SLIP` (prescribed normal + fluid tangential), i.e. half the mirror's normal drag. | `computeBoundaryVelocities` (mDBC ghost/mirror) applied per stage. Equivalent in spirit; the pair-level details differ. |
| **D14** | Wall-neighbour pressure-Laplacian flux is one-sided: `ψ = 2((p_i−p_j) − ∇p_i·r)` (`sph.cuh`, `getContrib2`); only fluid pairs use `½(∇p_i+∇p_j)`. Wall rows get `λ = 1`, `L = 0` (`deltaHash4D`). | The shared δ-SPH kernel uses `(∇p_i+∇p_j)` for every neighbour kind, and `computeGradRhoL` runs `AllToAll`, so a wall row's gradient is its own renormalised estimate over a truncated support. Check how noisy that is at a contact line; the one-sided form is a small change in the kernel's wall branch. |
| **D15** | Shifting is Michel-like but not the paper's: `δu ∝ −0.2·U·Δt` (paper 0.5), `λ` cutoff 0.55 in the May `ShiftHash` / 0.4 at the tip, plus a **hard-sphere collision shift** (`0.5Δx` fluid, `0.9Δx` wall, `sph.cuh:392,425`) that acts even where `λ` is low, plus a `∇λ` edge-cohesion term capped at `0.01h`. | Michel et al. 2022 exactly as the paper writes it. Keep ours; the extras are post-paper experiments ("trying to make volume shifting more effective", 2023-12-07). |
| **D16** | TIC (`PGTIC_2`): renormalised `PG2` for non-surface rows with `kernSum < 0.95`, memory in `pg_switch_d` ("Fixed TIC across iterations biasing toward surface check"). `PGTIC` / `PGTIC_2` are compile flags; we do not know which the paper runs used. | `pressureForceTerm = nonConservative` (plain `(p_i+p_j)`); Antuono switch available. Ask the authors which flag. |

### 2.5 Present in the May snapshot, gone at the tip: do **not** port

Wall repulsion force `brpf ∝ 0.1 c0² m` (tip: computed then overwritten with 0, `:2448`);
compression-only Monaghan AV at the free surface (`α = 0.01`) and wall proximity damping
(`α = 0.1`); `ṽ` advection in momentum and pressure. Their removal in the paper-era code
matches the paper and our scheme.

## 3. Suggested order

**Status 2026-10-12:** D1 is implemented as `acParams.wallPressureEdgeRelax = 'off' | 'zero' | 'lambda'`
(`modules/incompressible/wallPressure._edgeRelaxFactor`, wired in `schemes/incompressible/artificialCompressible.py`; case param
`acWallEdgeRelax`, probe flag `--edgeRelax`; tests `test_edgeRelax*`). It uses the raw (undilated) edge set
and the `detectFreeSurface` λ, both frozen for the real step until D2 lands. Results (Marrone 3.1, nx 70, tLimit 2.45, one realisation each; logs
`scripts/out_contactLine/d1_m31_nx70_{lambda,zero}.log`; both runs were killed by hand once t had frozen with |v| still
growing, so neither wrote a completion record; wall time and ms/step were not recorded):

| Variant | First failure | Note |
|---|---|---|
| `off` (stored baseline `fl_m31_nx70_off.log`, earlier code) | t = 0.685, t\* ~ 2.8 | |
| `lambda` | alarm at step 1575, t = 0.698 (t\* 2.82) | no gain; |v| 496 -> 7e3 in one step |
| `zero` | first |v| > 100 at step 2067, t = 1.0545 (t\* ~ 4.26) | delayed by ~0.37 s; spiked to 3.3e3, decayed to 741, then re-grew to 8e3 with t pinned at 1.0553 |

**Verdict: D1 alone does not fix the nx 70 contact line.** `zero` buys about 1.5 t\* and then fails the same way;
`lambda` buys nothing. Neither is a candidate default. Caveats: n = 1 realisation each (WALL_CONTACT_PLAN §6 asks for
n >= 4 before calling a difference real), the baseline is a stored run rather than a rerun on today's code, no video was
inspected, and D1 reads the raw edge set and λ frozen for the step (D2 would refresh them). The next candidates are D3 and
D2, then D5.

Each step is a config flag defaulting to today's behaviour, with the Marrone 3.1 nx=70
`contact`-line case as the discriminating run (WALL_CONTACT_PLAN §3), video on, and the
pure-fluid checks (hydrostatic column, rotating patch, droplet) A/B'd before any default
changes.

1. **D1** (edge-conditioned wall-pressure attenuation, both the hard-zero and the λ-weighted form).
2. **D3** (λ taper on `k2 D^p`).
3. **D2** (re-detect per iteration), once D1/D3 exist to consume fresh flags.
4. **D5** (position-row bounce-back).
5. **D6-D8** (stall exit, pressure gate, Δτ warm-up) for cost, off for contact-line diagnosis.
6. Config-only alignments D9, D10, D12 for any like-for-like comparison with the paper's tables.
7. Hold D4 and §2.3 in reserve.

### D3 result (2026-10-12)

`acParams.diffusionLambdaTaper` (`schemes/artificialCompressible.diffusionTaper`, case param `acDiffusionTaper`, probe
`--taper`) built and tested. Marrone 3.1 nx 70, tLimit 2.45, one realisation, logs `scripts/out_contactLine/d3_m31_nx70_*.log`:

| Run | Outcome |
|---|---|
| `off`, fresh on today's code | diverged, step 1430, t = 0.6868 (t\* 2.777); matches the stored baseline (0.685) |
| `taper` | velocity alarm at step 1470, t = 0.6907 (t\* 2.79), \|v\| 629 at the alarm; stopped by hand |

**Pass criterion (user, 2026-10-12): any velocity alarm is a failure.** Marrone 3.1's velocity scale is a few m/s; \|v\| of
hundreds means a particle has been launched and the state is contaminated, whether or not \|v\| later decays (the D1 `zero`
run fell 3.3e3 -> 741 and then re-grew to 8e3). D3 alone gives ~0.004 s over the baseline: no effect. The planned
`taper` + edge `zero` run was not made: at best it repeats D1 `zero` (alarm at t\* ~4.3). Tried so far, none sufficient:
D1 (`zero`, `lambda`), D3.

### All-aligned run (2026-10-12): D1 `lambda` + D2 + D3 + D5 + D9/D10/D12 + Δt/Δτ 2, CFL_t 0.4

D2 `acParams.redetectSurfacePerIteration`, D5 `acParams.wallBounceBack` built (`wallBounceSetup`, `wallBounce`;
tests `test_wallBounce*`, `test_redetectAndBounce*`); `--authors` (case param `acAuthors`) switches everything on.
Marrone 3.1, nx 70, tLimit 2.45, video on, one realisation. Log `scripts/out_contactLine/all_m31_nx70_authors.log`,
run dir `marrone31_artificialCompressible_nx70_cavoff_authors_all`.

| | `contact` + `isolatedZeroPressure` (best before) | all-aligned |
|---|---|---|
| reached | t\* 9.908, 5493 steps | **t\* 9.910, 2446 steps** (mean dt 1.0e-3 vs 4.5e-4) |
| velocity alarms | (one 13.2 spike) | **0**; max \|v\| 8.4, p99 7.0 |
| flier uids / births | 14 / 58 | **0 / 0** |
| nPenetrating | 2 | 0 |
| ceiling riders (max / final) | 12 / 2 | 46 / 17 |
| sparse wall rows (max) | 4 | 16 (v to 6.2) |
| wall pressure min | 0 (clamped) | -57 (unclamped) |
| cost | not recorded | 1412 s wall, 577 ms/step averaged over the run (video + per-step forensics included; the progress row shows ~218 ms for a plain step) |

Frames (t = 0.68, 1.09, 1.62, 2.45 s): the far-wall run-up, which killed every earlier `off` run at t = 0.685, is clean;
the overturning wave closes a cavity at t = 1.6 without spray. Remaining defects: particles hanging from the ceiling
(about 7 visible at t = 1.6 and 2.45), a thin column clinging to the left wall at t = 2.45, wall pressure to -57.
These are the known sticking problem, not the blow-up.

**What this does and does not show.** It is the first configuration that clears the nx 70 failure *without* clamping
pressure. It does **not** say which piece matters: each of D1 / D3 alone failed (D1 `zero` delayed to t\* 4.3), D2 and D5
were never run alone, and the config-only changes (RK3, Δt/Δτ 2, CFL_t 0.4, no accel constraint) changed the timestep by
2x, which alone could matter. One realisation (WALL_CONTACT_PLAN §6 wants n >= 4). Next, as you suggested, switch things
off from this working point (leave-one-out) rather than build up.

### Attribution (2026-10-12): D5 is the whole effect, on this case

Leave-one-out from the all-aligned configuration (Marrone 3.1, nx 70, tLimit 2.45, one realisation each; driver
`loo.py`, logs `scripts/out_contactLine/loo_m31_nx70_no-*.log`, stop rule |v| > 50):

| Piece switched off | Result |
|---|---|
| Δt/Δτ 5 + CFL_t 0.2 (`dt`) | survives, t\* 9.908 (4704 steps) |
| RK2, CFL_τ 0.5 (`rk`) | survives, t\* 9.909 |
| acceleration constraint on (`accel`) | survives, t\* 9.910 |
| D2 per-iteration re-detection | survives, t\* 9.907 |
| D3 λ taper | survives, t\* 9.909 |
| D1 `lambda` | survives, t\* 9.910 |
| **D5 wall bounce-back** | **fails: \|v\| 77 at t = 0.975 (t\* 3.93)** |
| `uchar` | not run (config-only) |
| **everything except D5** (D5 alone on the default config) | **survives, t\* 9.91 (5017 steps)** |

D5 alone vs all-aligned vs the best earlier configuration (`contact` + `isolatedZeroPressure`):

| | best earlier | all-aligned | **D5 alone** |
|---|---|---|---|
| flier uids / births | 14 / 58 | 0 / 0 | **0 / 0** |
| max \|v\| | 13.2 | 8.4 | 10.1 |
| nPenetrating max / maxPenetrationDx | 2 / 1.99 | 0 / 0.27 | **0 / 0.27** |
| ceiling riders (max / final) | 12 / 2 | 46 / 17 | 44 / 18 |
| sparse wall rows (max) | 4 | 16 | 18 |
| wall pressure min | 0 (clamped) | −57 | −87 |

So on this case D5 is **necessary and sufficient**, and the timestep group, RK order, acceleration constraint, D1, D2 and D3
all do nothing for survival. Frames (t = 0.71, 1.28, 2.42) match the all-aligned run: clean far-wall run-up, no spray,
a few ceiling riders and a few particles near the left wall. The earlier partial gains (D1 `zero` t\* 4.3) were noise
next to D5. Open: n = 1; ceiling riders (44) and wall pressure (−87, unclamped) are worse than the clamped best-earlier
run; the position-only vs reflected-velocity question (`wallBounceTarget`) is queued.

### D5 variants: what the bounce reflects (2026-10-12/13)

`acParams.wallBounceTarget`, with everything except D5 off, Marrone 3.1 nx 70, tLimit 2.45, one realisation each:

| Target | What is reflected | Result |
|---|---|---|
| `position` (authors' actual behaviour; their `vel[i] - veli;` is a no-op) | only the velocity in the position row | **survives, t\* 9.91**, 0 fliers, nPenetrating 0 |
| `stage` (the literal fix of the typo: `vel` there is `RK_vel[iRK]`) | + the stage velocity the residual reads | **fails: \|v\| 58 at t = 1.57 (t\* 6.34)** |
| `both` | + the stored `v` after each iteration | **fails: \|v\| 6.7e3 at t = 2.27 (t\* 9.15)** |

So the authors' typo is **load-bearing** here: the bounce only works if it acts on the position update and leaves the
velocity the pressure solve sees alone. Reading: when the residual (or the state) carries the reflected velocity, a
particle running into the wall no longer appears as compression, so `div v` does not build the pressure that should stop
it, and the contact becomes elastic (restitution 1, no pressure work), which is the energy-pump mechanism already seen
with the noPen `finalize` replacement. n = 1 each, and `stage` only just crossed the 50 stop rule, but both fail where
`position` does not, and `both` is a full blow-up. A principled replacement should be a no-penetration constraint on the
position update itself, not a velocity bounce.

### Defaults flipped, and the ceiling riders (2026-10-13)

**Defaults (user, 2026-10-13):** the cheap alignments are now on by default. `acParams.diffusionLambdaTaper = True` (D3) and
`acParams.wallBounceBack = True` with `wallBounceTarget = 'position'` (D5, the stable variant: `stage` and `both` fail) are
`ArtificialCompressibleParams` defaults, so every ACSPH case gets them. The `dambreak` case additionally sets, behind
`acLowCostAlign` (default True; `acAuthorsOff='d3,d5,dt,uchar,accel'` or probe `--noAlign` restores the old behaviour):
Δt/Δτ 2 and CFL_t 0.4 (paper Sec. 4.5), D10 `U_char = 0.1 c0_ref`, D12 no acceleration constraint. Left **off** because they
cost 5-26 % and the leave-one-out showed no effect: D1 (+5-10 %), D2 (+18-26 %), D9 RK3 (+10 %); `--authors` switches those on.
`sloshingTank` gets D3/D5 only (config defaults); its timestep settings are untouched. `tests/test_artificialCompressible.py` passes
(63 tests, incl. `test_authorsCheapAlignmentsAreTheDefault`).

**Marrone 3.1 nx 70, tLimit 2.45, new defaults (n = 1):** `diverged=False`, 2490 steps, t\* 9.908, 0 velocity alarms, 0 flier births,
nPenetrating 0 (maxPenetrationDx 0.28), max |v| 9.7; ceiling-row count (`nCeiling`) 50 max, 10 at the end; wall-sparse rows max 17
(v to 6.4), 7 at the end; wall-pressure min -382 at its worst step (one transient, -9.3 at the end; the earlier D5-only run had
-87). 1475 s wall, 2490 steps, step time mean 512 ms (min 79, max 1731 ms), with per-step forensics and video on. GPU utilisation
is ~25 % with only this run on the GPU (user's report, and `nvidia-smi` 23 % in two later runs; an LM Studio `llama-server`
was seen at 95 % once before the first run, so the earlier wall times in this document may carry contention, but the 25 % does
not): launch-bound, filed in OPEN_PROBLEMS §33.
Frames at t = 1.26 and 2.32: clean far-wall run-up and overturning wave, no spray; thin one-particle-thick sheets slide along
the ceiling at t = 1.26 and a few 2-3-particle clusters hang from it at t = 2.32. Run `scripts/out_contactLine/marrone31_artificialCompressible_nx70_cavoff_newdefaults/`.

**What the "ceiling riders" are** (`scripts/probe_ceilingRiders.py`, stored states every 200 steps). The `nCeiling` counter
(any fluid row with a wall row within 1.5 dx above it) mixes three populations, and only the last two are the sticking:

| frame (t) | riders | bulk (nF > 4: the wave legitimately wetting the ceiling) | thin (1-4 fluid neighbours) | lone (nF = 0) |
|---|---|---|---|---|
| 800 (0.85) | 13 | 8 | 3 | 2 |
| 1000 (1.04) | 42 | 36 | 3 | 3 |
| 1400 (1.40) | 44 | 22 | 18 | 4 |
| 1800 (1.79) | 18 | 4 | 11 | 3 |
| 2489 (2.45) | 10 | 2 | 7 | 1 |

So the headline "17-46 riders" overstated the defect: at the first ceiling impact most are the bulk wetting the ceiling. The
persistent ones are a few thin sheets and lone particles (37 distinct uids over the run), several riding **> 1 s** along the
ceiling at constant speed (uid 80 / 1342: 0.4 -> 0.12 m/s over 1.05 s; uid 167: 4.67 m/s for 0.55 s, vy = 0.000, p = -0.10).

**Force balance, recomputed from the stored state** (Eq. 61 wall pressure, Wendland C2, `-V sum_j (p_i+p_j) gradW_ij`): every
rider with vy ~ 0 is in balance with gravity to within 0.1-0.8 of g = 9.81, and the **wall alone carries it** (lone particles: wall
+9.81 / +9.45 / +10.32 upward, fluid 0; thin sheets: wall +9 to +21, fluid -0.9 to -11). Their own pressure is p_i = -0.08 to -0.17,
about -rho g * 0.7 dx, i.e. the weight of one layer: the inverted-glass film equilibrium at one-particle thickness, held by the
wall's tension. Riders that are leaving (vy = -2.6, uid 249 / 416 / 1173 at t = 2.35) have only +2.4 to +2.9 from the wall: the
net is -7, they fall. So the wall is a **constraint force** that supplies whatever the rider needs (bilateral, as WALL_CONTACT_PLAN
§2.1 settled): nothing in the closure says "a wall cannot pull". This is the new-defaults, D5-on confirmation of the old mechanism,
not a new one; D1 (edge-attenuated wall pressure) never attacked it for a lone rider because the rider's own pressure supplies the tension.

**Does the old best clamp still help on top of D5? No, it hurts** (`--cav contact --isoZero` + the new defaults, Marrone 3.1 nx 70,
n = 1, log `scripts/out_contactLine/rider_m31_nx70_newdefaults_contactIso.log`): free fliers appear from t = 0.94 (> 5 by step 891),
**389 free fliers** by t = 1.68, velocity alarm at step 1836 (t = 1.679, t\* 6.79, |v| 2.1e5), stopped by the stall watchdog at
t\* 6.80 with |v| 1e7. Without the clamp (D5 alone) the same case has 0 fliers and reaches t\* 9.9. The `contact` projection's elastic
fragments (WALL_CONTACT_PLAN §2.6) are the spray source; D5 made the unclamped baseline good enough that the clamp is now the
worse option. The clamp stays off.

**Where this leaves the riders:** nothing tried so far (D1, D3, the clamps) releases a rider that the wall's bilateral constraint holds; the
open route is the one WALL_CONTACT_PLAN lists as W1 (per-contact normal relative velocity / an active-set wall pressure), and it
should be judged on the *thin + lone* count of `probe_ceilingRiders.py`, not on `nCeiling`.

### Launch overhead and D5's convergence cost (2026-10-13)

Dug into the ~25 % GPU utilisation (user report; OPEN_PROBLEMS §33 has the full entry). Marrone 3.1 nx 70, tLimit 2.45, video + per-step
forensics on, n = 1 each, logs `scripts/out_contactLine/{rider_m31_nx70_newdefaults,perf_full_graph,ramp_0.5,ramp_1.0}.log`:

| Configuration | steps | t\* | loop | wall | ms/step mean | max penetration / riders (nCeiling max, final) |
|---|---|---|---|---|---|---|
| eager (old probe default) | 2490 | 9.908 | 21 m 15 s | 1475 s | 512 | 0.28 dx / 50, 10 |
| `--cudaGraph` (bitwise identical diagnostics) | 2490 | 9.908 | 4 m 18 s | 333 s | 103.5 | same |
| + `--bounceRamp 0.5` | 2445 | 9.908 | 1 m 47 s | 175 s | 43.7 | 0.41 dx / 48, 19 |
| + `--bounceRamp 1.0` | 2472 | 9.910 | 1 m 52 s | 182 s | 45.5 | 0.46 dx / 49, 21 |

Two separate things. **(1) Graphs were never on:** `CaseSpec.cudaGraph` defaults to False and no probe passed it, so each
pseudo-iteration ran ~260 eager launches (GPU 23 %); on, it replays from a graph (GPU ~1 ms per replay, 47 % busy early on).
**(2) The dual-time loop does not converge through the ceiling-impact phase** (`iterations` per real step, 200 = the cap):

| t | 0.2-0.6 | 0.8-1.0 | 1.0-1.2 | 1.2-1.4 | 1.4-1.6 | 1.6-1.8 |
|---|---|---|---|---|---|---|
| mean iterations / step | 13-14 | 66 | 175 | 192 | 122 | 18 |

with eps_v around -4.7 against the -5 target, so a third of the run was spent at the cap (and ṽ was not converged during it). From the
checkpoint at t = 1.218 (eager, 10 steps): the rows carrying the whole Σ|ṽ|² are five-ten ceiling riders (uids 416, 585, 837, 921, 80;
nF 4-7, 0.26-0.5 dx from the wall, |ṽ| 0.1-0.3 at |v| 0.2-1.4), i.e. rows resting at D5's 0.25 dx threshold flipping between
bounced and not bounced from one pseudo-iteration to the next (a limit cycle). Same state, same steps: **D5 hard switch 200 iterations,
D5 off 12-16, D5 with the smooth ramp 13-16.** `acParams.wallBounceRamp` (0 = the authors' hard switch, default; in (0, 1] the reversal is
C2-continuous in the wall distance, full below `(1 - ramp) * distance`) removes the chatter and still clears the far-wall run-up; both ramps reach
t\* 9.9 with 0 fliers and 0 penetrating particles, but the trajectories differ (chaotic; max penetration 0.41-0.46 dx rather than 0.28, rider
counts within the spread), so n = 1 each is not enough to rank them; **decided (user, 2026-10-13): ramp 0.5 and `cudaGraph` are the defaults of the `dambreak` case** (and of `probe_contactLine.py`); the config-level
`wallBounceRamp` stays 0 for the other ACSPH cases. Also tried and reverted: capturing the graph at iteration 0 instead of running that iteration eagerly
(~7 ms per step) gives a wrong result from step 2 (max |v| 0.126 vs 0.022), so something in the eager first iteration is load-bearing for the
capture; not pursued.

### Sloshing tank, periodic box, AC-4 re-check (2026-10-13)

**Sloshing tank (SPHERIC TC10, ACSPH nx 100, tLimit 7, graph on, D3 taper + D5 hard-switch bounce = the config defaults at the time;
log `scripts/out_contactLine/slosh_nx100_r0.log`, n = 1).** Previously blew up at t = 4.917 (uid 958, WALL_CONTACT_PLAN W2).
Now **runs to t = 7.0001, `diverged=False`, 12 554 steps, 0 fliers, max \|v\| 5.3, nPenetrating not recorded, nCeiling max 15, wall-sparse rows max 11,
pwMin −366 (transient)**. Loop 25 m 14 s (mean 120.6 ms/step), total 3328 s, partly on a shared GPU (below). Sensor 1 (left wall, still-water line):
impact timing matches the measured record (2.388 / 4.038 / 5.637 s vs measured 2.375 / 4.067 / 5.672 s). Smoothed (10 ms) peaks 3.7 / 7.1 / 6.2 kPa against
measured 3.7 / 3.9 / 2.7 kPa; the measured first-four-peak repeatability band is 2.2-13.1 kPa, so the later two sit inside the experimental
scatter but on the high side; the unsmoothed ACSPH signal has single-step spikes (10.8 / 70.6 / 31.6 kPa). Between impacts the run-down plateau is
~300 Pa above the measured one. Plot sent to the user; frames show clean waves and a few particles under the ceiling.
`sloshingTank` now has the smooth bounce (`acBounceRamp` 0.5) and `cudaGraph=True` as case defaults (user, 2026-10-13); the ramp run at nx 100
and nx 200 is in flight (`slosh_ramp_nx{100,200}.log`).

**Periodic box (W3, `scripts/probe_periodicLevel.py`, nx 32, uMag 0.01 TGV lattice, graph on, tLimit 0.5, n = 1).** A uniform level
of +10 does **not** grow in a wall-free ACSPH box: pMean 10 -> 5.78 over t = 0.5 (slow decay, ~ -1.1 / s), pStd 0.21 -> 0.13, \|v\|max 0.010 -> 0.066
(the symmetric `(p_i + p_j)` force on a disordered lattice turns the level into a small velocity); p0 = 0 stays at -0.02. The sealed box
doubled every 0.05 s. So the exponential growth is not a property of the dual-time solve; it needs the walls (the Eq. 61 closure against the
`(p_i + p_j)` force). The sealed box is being re-run with the current defaults (`sealed_p10_*.log`) to confirm that D3/D5 do not change it.

**AC-4 standalone (hydrostaticColumn nx 24, 100 steps, current scheme, `probe_acsphPressureSmoothingComparison.py`).** Still diverges:
\|v\|max 9.9e17 by t = 0.005, the same with graphs off and with D5 off (1.7e17), i.e. none of today's changes (graph, D3, D5) touch it. AC-2L and
AC-JST are bounded (\|v\| 0.06 at t = 0.586). (`pSlopeRatio` prints NaN for the bounded runs in this probe at nx 24 with both graph on and off: a diagnostic
issue, not looked into.) Hypothesis in test (`scripts/probe_ac4Scale.py`, queued): `_biharmonic` scales Eq. (35) with `supports^2`, the kernel support
RADIUS, while `acParameters` uses the smoothing length `supports / xi` for beta and k2 -- if the paper's `h` is the smoothing length, our AC-4 is
`xi^2` (= 4 for Wendland C2) too strong, which would also put the explicit pseudo-time step of the biharmonic term beyond its stability limit.

**GPU contention (user report: "the simulation is the only thing running at 25 % / 50 % utilisation").** While the ramp run was at ~300 ms/step
(twice the earlier run at the same stage) `nvidia-smi pmon` showed an LM Studio `llama-server` (pid 3878008) at ~41 % SM next to the simulation
at 54 %. Timings in this document taken while it is busy are inflated; check `nvidia-smi pmon -c 1` before reading a wall time.

## 4. Things I could not determine

- The authors' `input_values.dat` is not in the repo: `pmethod`, `dTodTau`, `minres`,
  `cfln`, `srfac`, `α`, `c0_ref` are all runtime inputs. The per-case values in
  ACSPH_PLAN Part 2 are the best available.
- Which `PGTIC`/`NOSLIP`/`SHIFT_INSIDE` build flags the published runs used (the Makefiles
  carry none of them).
- Whether `FSI_devel` is the exact code behind the paper's figures or a few commits past it.
  The Dec 2023 commits still touch shifting and the viscous BC, so a final check against the
  authors would settle it.
