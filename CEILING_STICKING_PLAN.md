# Ceiling sticking in δ⁺-SPH — plan

> **Progress marker:** this plan's row in [PLANS.md](PLANS.md). Update its *Last worked* date and *Where it stands* whenever you work on this plan.

Scope (user, 2026-09-29): the ceiling-sticking particles and the particles
that leave the ceiling at unnatural speed, on **Marrone 3.1** only, with
**δ⁺-SPH (`sun2017DeltaSPH`, PST on) + `fourtakas2019` DDT + `symplecticEuler`
+ `english2025` mDBC** — the default combination. Not ACSPH. No hacky fixes:
whatever changes must follow from the equations actually being integrated.
Method: export restartable checkpoints, resume in front of the events, and
watch the sticking and the ejected particles directly, term by term.

Background (read first, not repeated here): `OPEN_PROBLEMS.md` §1 (fluid-fluid
truncation / Antuono switch) and §8 (wall closure); `MDBC_CONTACT_LINE_PLAN.md`
§7.5 (δ-SPH clamps diverge/ratchet), §8.3 (freeSlip reflects the normal
velocity, so separation is booked as expansion), §8.5 (toys), §12-13
(`loneDensityReset`, `mdbcOneSidedHydrostatic`).

## 1. Reproduction

Worst overnight seed (`OPEN_PROBLEMS.md` §5 follow-up): δ⁺ nx67 fourtakas2019,
`--jitter 1e-3 --seed 3`, max|v| 19.7, ρ extremes [0.81, 1.20], kicks at
t\* ≈ 5.2-5.7. Frame at t = 1.264 (t\* 5.1): single particles and 2-3-particle
clumps hang under the ceiling; one at ~9 m/s with the ceiling-wall rows next
to it at ρ ≈ 1.06.

Re-run with checkpoints (`scratchpad/ceiling_run.py`: `storeMode='states'`
every 500 steps, 55 MB each, plus a per-step trace of every fluid row within
4 dx of the ceiling or faster than 6 m/s) → `scripts/out_ceiling/seed3/`.
Same 19551 steps as the overnight run and an identical frame at step 13000:
the reproduction is exact. Wall surface at y = 0.4925 (first ceiling row at
0.5, its ghost at 0.485); a ceiling row k has lever arm (2k+1) dx.

## 2. What the trace shows (`scratchpad/ceiling_analyze.py`)

Two populations, both from the right-wall run-up that reaches the ceiling at
t\* ≈ 3.3-3.7:

- **Riders** (uids 404, 323, 242, 485, ...): 4-particle rafts (3 fluid
  neighbours each), 0.3-0.8 dx below the ceiling for ~10 500 steps
  (t\* 3.3 → 7.5, ~1 s), sliding at v_x ≈ -3.16 m/s, mean v_y = 0.000,
  ρ within 3e-3 of ρ0. Quiet.
- **Kicked cluster** (uids 889, 402, 808, 321, 726): arrive at t\* 3.6-3.7,
  ride quietly (ρ = 1.000, p ≈ 0) until t\* ≈ 4.5, then their pressure
  oscillates with **growing** amplitude (+7 → +106 → -210 → +531 → +1060 →
  +1358) while they stay at the ceiling, v_y swinging ±5-10 m/s per few steps
  (≈ 10⁴ g), and are released along the ceiling at 11-13 m/s at t\* ≈ 5.3-5.4.
  That is the run's max|v| 19.7. Nothing hits them from outside: the growth is
  self-excited in the pinned state.

## 3. Term-by-term, resumed from `state_11000` (`scratchpad/ceiling_forensics.py`)

Eager resume with hooks on the scheme's term functions; per watched row and
RHS call: the Antuono pressure force split by linearity in the pressure array
into the wall's own pressure `aWallP = -(1/ρ_i) Σ_wall V_j P_j ∇W`, the row's
own pressure times the wall truncation `aSelfW = -(P_i/ρ_i) Σ_wall V_j ∇W`,
and the fluid-fluid rest; continuity split into wall pairs / fluid pairs; DDT;
viscosity; english2025's ghost quantities. Resume fidelity: the resumed
cluster matches the original to 3-4 digits through t\* 4.9, then diverges
(eager vs graphed summation order, amplified by the instability below).

Code facts that matter (the memory saying the DDT runs fluid↔wall is stale):
the DDT is `FluidToFluid`; continuity is `AllToAll`,
`dρ_i/dt = -ρ_i Σ_j V_j (v_j - v_i)·∇_iW_ij`, wall rows carrying the freeSlip
mirror of the Shepard fluid velocity at their ghost (normal part reflected);
the pressure force is `-(1/ρ_i) Σ_j V_j (P_i + P_j) ∇_iW_ij` for surface rows,
wall rows included; noPen ('finalize') only acts on approaching rows and has
the right sign (boundary rows store x_b - x_g).

### 3.1 The kick: explicit-midpoint growth of the wall-contact acoustic mode

uid 889 at the onset: 46 wall neighbours, 4-9 fluid ones, 0.3 dx off the
ceiling. `aWallP ≈ aSelfW` (english2025's ghost value is the cluster's own
density, so the ceiling carries the particle's pressure): force ≈
`-(2P_i/ρ) G`, G ≈ 23.6 m⁻¹; continuity ≈ wall pairs only, dρ/dt ≈ 56 v_y
(= 2ρG, the 2 from the mirrored normal velocity); DDT ~0; gravity irrelevant.
A ρ-v spring against the wall, ω² = c²(2ρG)(2G/ρ).

`symplecticEuler` (DualSPHysics' predictor-corrector) advances x with the
time-centred velocity, x^{n+1} = x^n + dt (v^n + v^{n+1})/2, but
ρ^{n+1} = ρ^n + dt D(x_h, v_h) with the predictor velocity
v_h = v^n + dt/2 a^n, and v^{n+1} = v^n + dt a^{n+1/2}. For the (ρ, v) pair
that is the explicit midpoint rule: on ρ' = A v, v' = -B ρ (ω² = AB) one step
is [[1 - h²/2, dt A], [-dt B, 1 - h²/2]], h = ω dt, determinant
**1 + h⁴/4**: growth ≈ h⁴/8 per step at **every** dt, bounded only by
dissipation. In the bulk the DDT supplies it; at a sparse wall contact the DDT
is empty and only the artificial viscosity is left.

Measured on the cluster (uids 889/402/808, in phase): period 11.8 steps,
ω = 5480 s⁻¹, dt = 9.72e-5, **h = 0.53 → h⁴/8 = 0.0100/step undamped;
measured envelope growth 0.0067/step** (the difference is the viscosity).

Controls from the same checkpoint (cluster |P|max per t\* window 4.32 → 5.4):

| resume from step 11000 | cluster \|P\|max | cluster \|v\|max | run max\|v\| |
|---|---|---|---|
| symplecticEuler, CFL 0.3 (default) | 11 → 160 → 767 → 1969 | 19.6 | 18.97 |
| symplecticEuler, CFL 0.15 (h/2: growth /16, below damping) | 0.8 → 0.2 | 1.61 | 9.90 |
| RK4, CFL 0.3 (stable on the imaginary axis) | 0.8 → 0.2 | 1.61 | 9.57 |

**Verdict: the fast ejections are the time integrator.** The pinned state
supplies a stiff, undamped ρ-v mode; explicit midpoint amplifies it. The
particles still ride the ceiling in every control: the sticking itself (§3.2)
is a separate mechanism.

### 3.2 The sticking: the bilateral wall, in two equal halves

Riders (uids 404/323/242/485, t\* 4.32-4.6 averages): pressure force
+9.76 … +9.95 m/s², cancelling gravity; `aWallP` ≈ +4.4 … +7.3 constant and
`aSelfW` ≈ +2.6 … +5.6 (P_i ≈ -0.1 … -0.3, ρ = 0.99997), fluid-fluid ~0.
- `aWallP`: with 4 fluid neighbours at the ghost, english2025's trust ramp sets
  α = ρ0 (P_g = 0), so the ceiling carries only the hydrostatic extension
  `ρ0 g·(x_b - x_g) < 0`: pure tension.
- `aSelfW`: the row's own small tension, booked by continuity against the
  mirrored wall velocity (a separating row expands at twice its speed), acts
  through the (P_i + P_j) symmetric form. A soft spring (tiny h), so it rings
  slowly and stays bounded: a bound state, not an instability.

### 3.3 Fix for §3.1: time-centred continuity (candidate)

Density is a configuration variable (summation form ρ = ρ(x); continuity is
(∂ρ/∂x)·v), so it should be advanced with the same velocity as x:
ρ^{n+1} = ρ^n + dt D(x_h, (v^n + v^{n+1})/2). Then ρ_h = ρ + dt/2 A v,
v' = v - dt B ρ_h, ρ' = ρ_h + dt/2 A v' — Störmer-Verlet, determinant 1,
|λ| = 1 for h < 2 — and ρ stays consistent with the positions to second
order. D is linear in v (and the static-wall freeSlip mirror in the fluid
velocity), so `WeaklyCompressibleSystem._timeCentredContinuity` adds
dt [D(x_h, v̄) - D(x_h, v_h)] in `finalize` (one boundary-velocity gather and
two divergences, eager). Option `timeCentredContinuity` (default off,
symplecticEuler only; probe flag `--timeCentredContinuity`).

### 3.4 Validation of the fix

**Checkpoint A/B** (resume `state_11000`, `scratchpad/ceiling_compare.py`):
with the option the cluster's |P| decays 0.8 → 0.2 (as RK4 and CFL 0.15 do),
|v| ≤ 1.61, run max|v| 18.97 → 9.57. The trajectory agrees with the RK4 one
to ~4 digits (e.g. uid 889 at t = 1.1957: ρ 0.99999 / 0.99999, v_y -0.0166 /
-0.0168; the unfixed run there: ρ 1.045, P +426, v_y +3.7). The riders stay
on the ceiling: the sticking (§3.2) is untouched, as expected.

**Toy with an exact answer** (`scratchpad/toy_acoustic_integrator.py`,
`tests/test_timeCentredContinuity.py`): periodic box, no walls/gravity/PST,
α = δ = 0, (P_i + P_j) force (exact adjoint of the continuity divergence, so
the semi-discrete system conserves E = ½Σm|v|² + Σm c²(ρ-ρ0)²/2); standing
wave at rest, nx 64, c0 10, 400 steps.

| Courant | λ | ω dt | dlnE/step, current | h⁴/4 | dlnE/step, time-centred | E_end/E0 current → time-centred |
|---|---|---|---|---|---|---|
| 0.3 | 4 dx | 0.159 | 1.73e-4 | 1.59e-4 | -8e-6 | 1.069 → 1.000 |
| 0.3 | 6 dx | 0.196 | 3.65e-4 | 3.72e-4 | 4e-6 | 1.157 → 1.001 |
| 0.6 | 4 dx | 0.321 | 2.51e-3 | 2.67e-3 | -6e-6 | 2.72 → 1.02 |
| 0.6 | 6 dx | 0.400 | 5.82e-3 | 6.38e-3 | 4e-6 | 10.2 → 1.00 |

The current scheme pumps every acoustic mode at the explicit-midpoint rate,
walls or not; the DDT usually hides it in the bulk.

**Marrone 3.1, full runs, seeds 1-3** (`scratchpad/ceiling_tcc_batch.sh`,
`scripts/out_ceiling/m31_tcc/`, scored with the probe's own checks,
`scratchpad/ceiling_score.py`), against the overnight 2026-09-28 runs:

| seed | max\|v\| (t\*) before → after | ρ extremes before → after | P1 plateau / P2 peak before → after | checks |
|---|---|---|---|---|
| 1 | 16.1 (5.65) → 10.1 (5.50) | [0.88, 1.12] → [0.97, 1.03] | 0.46 / 0.25 → 0.46 / 0.25 | 9/9 → 9/9 |
| 2 | 11.4 (4.61) → 9.9 (5.49) | [0.88, 1.14] → [0.96, 1.03] | 0.46 / 0.25 → 0.46 / 0.25 | 9/9 → 8/9 |
| 3 | 19.7 (5.30) → 9.9 (5.49) | [0.81, 1.20] → [0.96, 1.03] | 0.46 / 0.29 → 0.46 / 0.25 | 9/9 → 9/9 |

The remaining max|v| ≈ 10 sits at t\* ≈ 5.49 in all three, as in the
`deltaSPH`-DDT runs (10.1-10.3): the plunging jet, physical. Seed 2's failed
check is "P2 back to quiescent": the probe is still wet at t\* 6.42 (just past
the 6.4 cutoff) at 0.12 while the baseline was already dry: the run-up tail
drains marginally later, the same check the `deltaSPH`-DDT runs fail 2/3.
Frames (seed 3, steps 12500-17000, baseline vs fixed): same flow, no new
spray, fewer loose droplets late; ceiling riders present in both. Cost
~145 s vs ~140 s per run.

So the OPEN §5 "counter-signal against fourtakas2019" (nx67 δ⁺ kicks in 2/3
seeds) was this integrator instability, not the DDT.

## 5. Relation to DualSPHysics' scheme and the Padé term (2026-09-29)

Sources: `dominguez2022` §2.5.2 (the DualSPHysics paper), its refs. 36-37;
`parshikov2002` App. B; `leimkuhler2016`; DualSPHysics
`JSphCpu::ComputeSymplecticPre/Corr` (`~/dev/DualSPHysics/src/source/JSphCpu.cpp`
1687/1850). Synced into `literature/` the same day.

- **What DualSPHysics writes down.** Eq. (32): symplectic *position* Verlet
  (drift ½, kick 1, drift ½; ref. 36 = Leimkuhler & Matthews' 2015 book).
  Eq. (33): "in the presence of viscous forces and density evolution … a
  velocity Verlet half step is used", i.e. v^{n+1/2} = v^n + Δt/2 F^n is
  introduced to have a mid-step velocity, and the position is put back on the
  trapezoid x^{n+1} = x^n + Δt (v^n + v^{n+1})/2. Eq. (34): the density
  "follows the half time steps of the symplectic position Verlet scheme"
  (ref. 37, Parshikov et al. 2000): ρ^{n+1/2} = ρ^n + Δt/2 R^n,
  ρ^{n+1} = ρ^n (2 - ε)/(2 + ε), ε = -(R^{n+1/2}/ρ^{n+1/2}) Δt, with
  R^{n+1/2} evaluated at the velocity-Verlet v^{n+1/2}. The code matches
  (ε from the corrector-stage `Ar` over the predicted ρ). warpSPH's
  `symplecticEuler` is the same scheme with the linear map instead of the Padé.
- **The inconsistency.** Treating ρ as a configuration variable, Eq. (32)
  itself gives ρ^{n+1} = ρ^{n+1/2} + Δt/2 R(v^{n+1}) — exactly
  `timeCentredContinuity`. DualSPHysics restores the trapezoid for x but not
  for ρ. In `leimkuhler2016`'s splitting language: position Verlet is A-B-A,
  continuity belongs in the A (drift) flows; driving it with a velocity from a
  separate half-kick is not a composition of sub-flows, so the (ρ, v) pair is
  not symplectic: explicit midpoint.
- **The Padé term does not change this.** `parshikov2002` (B.12) is a
  trapezoidal/Cayley treatment of the ρ *factor* in dρ/dt = -ρ ε̇ for a given
  strain rate (positivity, O(ε³) vs the exact exp(-ε)); it says nothing about
  which velocity makes ε̇ (Parshikov advances momentum by explicit Euler,
  B.11). Linearised, ρ^n(2-ε)/(2+ε) = ρ^n(1 - ε) + O(ε²): the plain update.
  `DELTASPH_VALIDATION_PLAN.md` §5.3 (historic) listed "Padé(1,1)/exp: |A| = 1,
  neutrally stable by construction (Cayley transform, modulus 1 for imaginary
  ε)"; that row was a hard-coded print in `scratchpad/acoustic_amplification.py`,
  never computed, and the argument confuses the real per-particle strain
  increment ε with the complex eigenvalue iωΔt of the pair. Measured
  (`scratchpad/toy_acoustic_integrator.py --pade`, Courant 0.6, λ 6 dx,
  ωΔt 0.40, amplitude 1e-3, h⁴/4 = 6.4e-3):

  | density update | d ln E / step | E_end / E0 (400 steps) |
  |---|---|---|
  | linear, predictor velocity (current) | 5.824e-3 | 10.21 |
  | Padé, predictor velocity (DualSPHysics) | 5.824e-3 | 10.21 |
  | linear, time-centred | -1.4e-6 | 0.998 |
  | Padé, time-centred | 1.2e-6 | 0.998 |

  (At amplitude 1e-4/1e-5 the time-centred rows show 1e-5-6e-4 residuals
  that grow as the amplitude *shrinks*: float32 roundoff of ρ = 1 ± ε, not
  dynamics.) Same §5.3 also called symplecticEuler's explicit-midpoint growth
  "benign"; it is, in the bulk where the DDT damps it, and it is not at a
  sparse wall contact (§3.1).
- **So** the time-centred continuity is not a new scheme: it is DualSPHysics'
  own Eq. (32) applied to the density as the paper's text says it should be.
  The Padé map can be kept or dropped independently (it only matters at
  O(ε²): positivity for large compressions).

## 6. Other integrators, and where the fix belongs (2026-09-29)

**Every Verlet-family integrator in `warpSPHIntegrators` mistreats density
the same way:** `verlet.py` / `ruth.py` / `util.py` advance quantity fields
with `applyQuantityUpdate(..., k, explicit_step(c·dt))` next to each *kick*,
using the stage rate. That treats ρ like a momentum. But dρ/dt = D(x, v) is
kinematic (linear in v), i.e. ρ is a configuration variable and belongs in the
*drifts*. Measured on the acoustic toy (`toy_acoustic_integrator.py
--integrators`, h = ωΔt ≈ 0.195, no dissipation, 400 steps):

| integrator | predicted d ln E/step | measured | E_end/E0 |
|---|---|---|---|
| symplecticEuler, rungeKutta2 (explicit midpoint on the pair) | h⁴/4 = 3.6e-4 | 3.65e-4 | 1.16 |
| velocityVerlet (ρ drifted right, closing kick sees ρ^n) | h²/2 = 1.9e-2 | 1.90e-2 | 2 056 |
| leapFrog, semiImplicitEuler (forward Euler on the pair) | h² = 3.8e-2 | 3.53e-2 | 5.4e5 |
| pefrl, vefrl | — | 3.7e-2 | ~1e6 |
| rungeKutta3 | -h⁴/12 | -1.2e-4 | 0.95 |
| rungeKutta4 | ≈ -h⁶/72 | -1.7e-6 | 0.999 |

RK methods treat every field alike, so there is no inconsistency to fix: the
pair just inherits the tableau's imaginary-axis behaviour (RK2 = the same
h⁴ growth; RK3/RK4 slightly damped, stable to √3 / 2√2). Nothing changes for
them.

Linear pair (ρ' = v, v' = -ρ), placement check:

| scheme | det, h = 0.2 / 0.5 / 1.9 |
|---|---|
| DKD, ρ in the drifts with the drift velocity (= `timeCentredContinuity`) | 1 / 1 / 1 |
| KDK, ρ in the drift, closing kick at the drifted ρ | 1 / 1 / 1 |
| DKD, ρ in the kick at v̄ (compatible-energy placement) | 1.02 / 1.125 / 2.8 |

So the density is **not** a kick-attached compatible quantity like the
thermal energy of `SPLITTING_PLAN.md` §2.7.1 (that one is correct *because*
work happens in the kick); it is a drift field. Same idea as Owen's compatible
energy (update with the mid-step velocity), different place in the composition.

**Where it belongs:** in the library as a field role, not in
`WeaklyCompressibleSystem.finalize` gated on the integrator name.
- `integrated('drhodt', tags=('density',), role='drift')` (fields.py already
  carries a `role` key);
- one optional system method, `drift_rates(state) -> update` (for WCSPH: the
  freeSlip wall velocity + `computeMomentum`, the kinematic part only; DDT
  stays with the stage rate, it is not velocity-driven);
- every drift in `verlet.py` / `ruth.py` / `util.py` (the
  `applyPositionUpdate(..., semi_implicit_position_step / verlet_position_step)`
  calls) also advances the drift fields with that drift's velocity, calling
  `drift_rates` on the drifted-from state when no stage already has the rate
  at that velocity (DKD's first half-drift reuses k0's; the second needs one
  kinematic pass; KDK needs one per drift). Systems without the method keep
  today's behaviour.
- Fixes symplecticEuler, velocityVerlet, leapFrog, semiImplicitEuler, PEFRL,
  VEFRL at once (each then gets its own symplectic stability region on the
  acoustic pair). Summation-density schemes (DFSPH, compSPH) are unaffected:
  their ρ is computed from x and is automatically a configuration quantity,
  which is why they never showed this.
- Cost: one kinematic pass (wall-velocity gather + one divergence) per drift
  that follows a kick: +1 per step for symplecticEuler, +3-4 for PEFRL/VEFRL.

**Implemented 2026-09-29 (user: go ahead, make it testable in the library).**
`warpSPHIntegrators` branch `drift-fields` (uncommitted): `drift.py`
(helpers), `integrated(..., drift_key=)`, the protocol methods
`drift_rates(velocities, aux=None, **kwargs)` / `drift_fields_enabled(**kwargs)`,
per-scheme drift paths in `verlet.py` (symplecticEuler: 1 evaluation at the
half state; velocityVerlet: predictor + trapezoid, 1; leapFrog: 2), `euler.py`
(semiImplicitEuler: 1) and `ruth.py` (PEFRL 6, VEFRL 4; trapezoid per drift).
Reference system + problems in `testing.py` (`DriftSystem`,
`acoustic_problem`, `drift_oscillator_problem`); `tests/test_drift_fields.py`
(26 tests: bounded acoustic energy with the drift path for all six, growth
without it, symplecticEuler's rate = h⁴/4 within 15 %, order 2 on a
position-dependent drift rate — the kick placement is first order for
velocityVerlet / leapFrog / PEFRL / VEFRL — and RK4 / midpoint / Heun
bitwise unaffected). CHANGELOG entry. Full library suite: 2846 passed,
373 skipped. Known limit (pre-existing): a field's non-velocity rate still rides
the kicks, which PEFRL/VEFRL handle at first order.

warpSPH: `WeaklyCompressibleState.densities` declares
`drift_key='drhodt_kin'`; the δ-SPH RHS reports the continuity part as
`update.drhodt_kin`; `WeaklyCompressibleSystem.drift_rates` (freeSlip wall
velocity + `computeMomentum` on a shallow copy) and `drift_fields_enabled`
(= `schemeConfig.timeCentredContinuity`, still default off). The `finalize`
hook is gone. `tests/test_timeCentredContinuity.py` now covers
symplecticEuler, velocityVerlet, leapFrog, PEFRL, VEFRL on the SPH acoustic
toy; with `test_cudaGraph.py`, `test_physics.py`, `test_loneDensityReset.py`:
92 passed. SPH toy (Courant 0.3, λ 6 dx, 400 steps), E_end/E0 without → with:
symplecticEuler 1.157 → 1.000, velocityVerlet 2056 → 0.9996, leapFrog 5.4e5 →
0.9998, semiImplicitEuler 5.5e5 → 1.083, pefrl 1.1e6 → 0.9999, vefrl 7.9e5 →
1.000; rungeKutta4 identical.

Marrone checks through the library path: the step-11000 resume reproduces the
`finalize`-hook run to float32 roundoff (ρ within 5e-7 over the first 50
steps); full runs (`scripts/out_ceiling/m31_drift/`, graphs on, no fallback
warnings): seeds 1 / 3 as before (max|v| 10.1 / 9.9, ρ [0.96-0.97, 1.03],
9/9); seed 2 9/9 but max|v| 10.4 at t\* 4.615 and ρ [0.90, 1.12] — §7.

## 7. A second kick mechanism: a fast lone rider resonating with the wall lattice (filed 2026-09-29, not chased)

Seed 2 with the drift path (`scripts/out_ceiling/seed2_drift/`, traced,
checkpoints): the run's max|v| 10.4 at t\* 4.615 is uid 80, a **lone**
particle that has ridden the ceiling since t\* 3.3 at s = 0.25 dx,
v_x = -6.87 m/s constant, ρ = 1.000 (the §3.2 bound state). From t\* ≈ 4.44 its
pressure rings up ±10 → ±1000 in ~420 steps (~0.011/step), period ~22 steps,
until it is thrown off at 8 m/s. The unfixed run has the same event at the same
step (11.4 m/s, t\* 4.612), so it is neither caused nor covered by §6.

- The ringing frequency locks to the wall-lattice ("washboard") frequency
  2π|v_x|/dx: 2874 vs 2884 s⁻¹ (steps 11500-11650), 2874 vs 2801 (11650-11745).
  A particle sliding over wall particles dx apart is modulated at that rate.
- Candidate energy source (hypothesis, not tested): for a lone particle
  english2025's neighbour ramp puts the ghost value at ρ0, so the ceiling rows
  carry only the hydrostatic term, while freeSlip still mirrors the particle's
  velocity in continuity. The wall's force (P_i · G, factor 1) and its
  continuity term (2ρG v_n, factor 2) are then not adjoint, and the pair is not
  energy-conserving (dE/dt ∝ P G v_n / ρ); with G modulated by the lattice it can
  pump. For the §3.1 cluster (nNb ≥ 5, wall pressure ≈ P_i) the factors match.
- Next, if picked up: resume `scripts/out_ceiling/seed2_drift/.../state_11000.h5`
  and test the adjoint hypothesis directly (e.g. measure the per-cycle work of
  the wall pair terms), before any change.

### 7.1 Energy budget: the pump is the noPen velocity replacement (2026-09-29)

Resume `seed2_drift/state_11000` with the drift fix
(`RUNDIR=scripts/out_ceiling/seed2_drift bash scratchpad/ceiling_ab.sh <tag> 11000 900
--seed 2 --uids 80 --patch <patch>`; `ceiling_forensics.py` now also records the
wall-pair continuity with the wall at rest and hooks `finalize`: state after the
integrator, PST displacement, noPen correction, state after finalize). uid 80
reproduces: lone (nF = 0, 46 wall neighbours), quiet to t\* 4.39, |P| → 1170 by
t\* 4.62, 10.4 m/s.

Per-step *discrete* budget (`scratchpad/ceiling_energy_step.py`), per mass,
E = ½|v|² + c²(ρ-ρ0)²/(2ρ0²) + g·y, each term with its exact discrete form
(dt·v̄·a_term for the kick, c²(ρ̄-ρ0)Δρ for the density). The integrator part
closes to 1e-5 against terms of 0.1-10 J/kg. Per ringing cycle (~23 steps):

| t\* | \|P\|max | wall pair (self force + wall continuity) | viscosity | **noPen** | PST |
|---|---|---|---|---|---|
| 4.437 | 30 | +0.001 | -0.004 | **+0.017** | 0 |
| 4.467 | 220 | +0.041 | -0.071 | **+0.44** | 0 |
| 4.504 | 354 | +0.064 | -0.19 | **+0.21** | 0 |
| 4.558 | 607 | +0.088 | -0.61 | **+2.18** | 0 |
| 4.602 | 1073 | -3.4 | -1.83 | **+8.2** | 0 |

- **The adjoint hypothesis holds but is minor.** The force's self term is
  exactly adjoint to a wall-pair continuity with the wall *at rest* (checked:
  identity error 0.02 against 1e5), and the mirrored-velocity continuity books
  more than that (the factor 2 on v_n, mostly normal), so the wall pair does
  create energy: +0.001 … +0.09 J/kg per cycle during the ring-up, 5-10x less
  than noPen, and negative once the particle is thrown about.
- **The pump is `mdbcNoPenShiftMode='finalize'`.** It fires on ~10 of the ~23
  steps of each cycle, whenever the particle moves into the ceiling, and
  *replaces* the velocity with `v^n + shift`, discarding the step's
  acceleration on that component. On the approach the wall compresses the
  particle and the pressure force brakes it by ~1 m/s per step; the replacement
  throws the braking away (step 578: v_y 1.40 → 0.31 after the integrator →
  1.31 after noPen) while ρ keeps the compression. Each such step creates the
  discarded pressure work (~0.8 J/kg); when the particle stops approaching,
  noPen goes quiet and the stored compression throws it back harder. Whether
  noPen fires depends on `r < 1.25 dx` to individual wall rows, which toggles
  as the particle passes each row: that is the lock to 2π|v_x|/dx.
- Same design in DualSPHysics (`JSphCpu.cpp` 1555-1566 Verlet, 1920-1927
  symplectic corrector: `rvelrhonew = velrhoprec + nopenshift`, density
  advanced with the full `Ar`). Its NoPen is gated on SlipMode ≥ NoSlip, so it
  never runs under free slip (Marrone's walls).

Controls from the same checkpoint (uid 80 |P|max / |v|max per t\* window, run max|v| over the 900 steps):

| noPen mode | [4.4, 4.5) | [4.55, 4.6) | [4.6, 4.65) | run max\|v\| | noPen steps |
|---|---|---|---|---|---|
| `finalize` (default) | 343 / 7.3 | 995 / 9.7 | 1170 / 10.4 | 10.39 | 437 |
| `off` | 0.8 / 6.87 | 0.4 / 6.85 | 0.4 / 6.83 | 6.87 | — |
| `derivative` | 0.8 / 6.87 | 0.4 / 6.85 | 0.4 / 6.83 | 6.87 | — |
| `impulse` (new) | 0.8 / 6.87 | 0.4 / 6.85 | 0.4 / 6.83 | 6.87 | 94 |

In all three quiet runs the rider slowly separates (s 0.25 → 0.50 dx by
t\* 4.7): the replacement is part of the *sticking* too (§3.2 / §4 item 2).

**Fix derived (`mdbcNoPenShiftMode='impulse'`, opt-in, default unchanged).**
A no-penetration correction on top of dx/dt = v, dv/dt = a, dρ/dt = D(x, v)
is a contact impulse: it changes v at an instant, and the configuration (x, and
ρ, a configuration variable, §6) is continuous across it. So
v^{n+1} = v_int + shift(v_int) on the *integrated* velocity, with x and ρ left
as integrated. With the kernel's shift = -factor (v_int·n) n (committed wall
rows are at rest at `finalize`, so v_rel = v), the normal velocity becomes
(1 - factor) v_n: a restitution e = 1 - factor, ΔKE = ½[(1-factor)² - 1] v_n²
≤ 0 for factor ∈ [0, 2], which is every row on the fluid side of a wall row.
(Separate edge, not changed: a row past a wall row's centre can get
factor < 0 from the ratio clamp [0.25, 1] → factor ∈ [-1, 2], which would
push it further in.) `WeaklyCompressibleSystem.finalize`; probe flag
`--noPenShift impulse`.

Toy (`scratchpad/toy_ceilingRider.py`, sweep `scratchpad/toy_rider_sweep.sh`):
the Marrone nx67 scheme unchanged, x-periodic channel (`semiPeriodic`), one
particle s0 = 0.25 dx under the ceiling at v_x, gravity down. Exact answer:
no energy source in the equations, so E never rises above E(0) at any v_x.
Seed v_y0 = 0.02 m/s, 1030 steps (0.1 s), v_x = 2, 4, 5, 6, 6.5, 7, 7.5, 8, 9,
10, 12 m/s (`scripts/out_ceiling/toy_rider/`):

| noPen | max(E - E0) [J/kg] | \|v_y\|max | \|P\|max 1st / 2nd half |
|---|---|---|---|
| `finalize` | **+1.1 … +2.1 at every v_x** | 1.06-1.41 (50-70x the seed) | 147-197 / 0.3-0.6; 25 at v_x = 10 |
| `impulse` | ≤ +1e-3 (float32 on E0 = 7-77); E_end < E0 at every v_x | 0.02 (the seed) | 2.7 / 1.05 |

Under `finalize` the toy shows the signature from the first contact: while the
particle moves into the ceiling v_y stays frozen (0.0152 m/s for 9 steps, then
0.042 for 12) as P rises, then it rebounds harder (E +0.0017 → +0.017 →
+1.7 J/kg by step 140). So the pump does not need the lattice resonance; the
lock to 2π|v_x|/dx is what sustains it on the Marrone rider.

**Marrone 3.1, seeds 1-3, full runs** (`scratchpad/ceiling_impulse_batch.sh` →
`scripts/out_ceiling/m31_impulse/`; drift fix + `--noPenShift impulse`, graphs
on, no fallback warnings, ~145 s each), against the drift-only runs
(`m31_drift`), scored with `ceiling_score.py`:

| seed | checks | P1 plateau | P2 peak | max\|v\| (t\*) | ρ extremes |
|---|---|---|---|---|---|
| 1 | 9/9 → 9/9 | 0.46 → 0.46 | 0.24 → 0.27 | 10.1 (5.50) → 9.6 (5.50) | [0.97, 1.03] → [0.99, 1.03] |
| 2 | 9/9 → 9/9 | 0.46 → 0.46 | 0.24 → 0.23 | **10.4 (4.62)** → 9.3 (5.50) | **[0.90, 1.12]** → [0.99, 1.02] |
| 3 | 9/9 → 9/9 | 0.46 → 0.46 | 0.29 → 0.25 | 9.9 (5.48) → 9.5 (5.50) | [0.96, 1.03] → [0.98, 1.02] |

Seed 2's lone-rider kick is gone; the remaining max|v| is the plunging jet
(t\* 5.50) in all three. Frames (seed 2, t\* 4.62 / 5.5 / 6.4 / 7.5, drift vs
impulse): same flow at every time; at 4.62 the baseline's colour ranges are set
by the ringing rider (10.1 m/s, ρ ±3.4 %), the impulse run's by the bulk
(6.8 m/s, ρ ±7e-4); at 7.5 the breaker crest sheds three small clumps at crest
speed (one in the baseline): jet break-up, not wall-launched. Ceiling riders
remain in both: the sticking (§3.2) is untouched. Tests
(`test_timeCentredContinuity`, `test_cudaGraph`, `test_physics`,
`test_loneDensityReset`): 92 passed.

Status: §7 explained and fixed as an opt-in; the default stays `finalize`
(user's call, gates in §4).

### 7.2 The default-combo set, full length, both fixes on (2026-09-29)

User: run the default runs over the entire run. The 2026-09-28 overnight set
(`scratchpad/run_overnight_batch_2026-09-28.sh`, fourtakas2019 half), re-run
with `--timeCentredContinuity --noPenShift impulse`
(`scratchpad/run_defaults_2026-09-29.sh` → `scripts/out_ceiling/defaults_2026-09-29/`,
`SUMMARY.md` by the same summariser; δ⁺ nx67 seeds from §7.1). Video on,
sequential, no velocity alarms, no divergence, graphs on. Flags added for it:
`probe_deltaSPHMarrone34.py --timeCentredContinuity` / `--noPenShift impulse`,
`run_sloshingTank.py --timeCentredContinuity` / `--noPenShift impulse`
(`sloshingTank` param `timeCentredContinuity`).

| run | checks before → after | max\|v\| before → after | ρ extremes before → after | notes |
|---|---|---|---|---|
| M3.1 δ⁺ nx67 ×3 | 9/9 ×3 → 9/9 ×3 | 16.1 / 11.4 / 19.7 → 9.6 / 9.3 / 9.5 | [0.81-0.88, 1.12-1.20] → [0.98-0.99, 1.02-1.03] | P1 0.46, P2 0.23-0.27 |
| M3.1 δ-SPH nx67 PST off | 8/9 → **7/9** | 10.2 → 7.4 (0 steps > 8 m/s) | [0.95, 1.05] → [0.98, 1.03] | P1 plateau 0.88 → 0.77 (fails both, known probe issue); **new: P2 back to quiescent** |
| M3.1 δ⁺ nx134 | 8/9 → 8/9 | **23.0 → 11.2** | **[0.81, 1.27] → [0.97, 1.04]** | bulk-density check now passes; **new: P2 back to quiescent** |
| M3.4 δ-SPH nx256 | 6/6 → 6/6 | 40.9 → 27.8 | ρ max 1.79 → 1.22 | KE_end 2.86 → 3.59 |
| M3.4 δ⁺ nx256 | 6/6 → 6/6 | 33.5 → 34.7 | ρ max 1.58 → 1.37 | KE_end 4.32 → 2.91 |
| sloshingTank t = 7 | runs to end → runs to end | per-0.5 s window max ≤ 6.0 → ≤ 7.7 | comparable (both 0.82-1.33) | smoothed Sensor 1 impacts 2.9/3.8/7.3 → 4.9/5.2/5.3 kPa (band 2.2-13.1) |

Frames: M3.1 nx134 at t\* 7.0 is visibly cleaner with the fixes (baseline:
15+ loose droplets, corrugated free surface on the left; fixed: a handful
off the breaker, smooth surface). M3.4 δ⁺ t\* 8 / 14 same flow; baseline has a
floor/fillet density hot spot at 14, the fixed run a few more slow droplets
off the obstacle jet. Global kinetic energy agrees to 1-2 % in every M3.1 pair
and in sloshing; M3.4's KE_end moves both ways (+25 % δ-SPH, -33 % δ⁺), i.e.
the late-flow spread, not a systematic gain.

**One systematic change: thin run-up jets along vertical walls climb higher
and faster.** Both new "P2 back to quiescent" failures: under `finalize` the
second run-up reaches P2 in several runs too (nx134: 17-18 wet neighbours,
t\* 6.9-7.3) but reads P2\* ≈ 0.003; with `impulse` the same wetting carries
pressure (median 0.12-0.22, peak 0.66 at t\* 7.0 for nx134; 0.12-0.18 in the
nx67 runs). sloshingTank: the fixed run's run-up tips reach 6.6 m/s (right
wall, t 3.19) and 7.7 m/s (up the left wall into the ceiling corner, t 4.13, ρ
0.88 there), baseline ≤ 5.2 m/s at those times. Plausible reason: `finalize`
discards the normal acceleration of every row it corrects, which holds film
tips against the wall and damps them. Not settled which is closer to the
experiment (Buchner: P2 dry after t\* 6.6, i.e. both runs over-shoot the late
run-up). Open item §4.4.

**Run-up forensics (2026-09-29, user: go for it).** Resume
`seed2_drift` (drift fix on) with `finalize` vs `impulse` from the same state,
watching every fluid row within 3 dx of the right wall
(`ceiling_forensics.py --region rightWall`, film budget
`scratchpad/ceiling_film_budget.py`):
- from step 16000 (t\* 6.29, P2 just drying): P2 traces identical to 3 decimals
  through t\* 6.9; `impulse` removes a little film energy at contacts
  (−1e-5 per 100 steps, 10-30 % of the integrator's change early; `finalize`
  ~1 %): dissipative, as designed, not a source;
- from step 14000 (t\* 5.5, before the plunging breaker) to t\* 6.9: KE equal to
  3-4 digits, max|v| identical, P2 within ±0.07 (noise), neither re-wets P2
  with pressure after 6.7.

So `impulse` does not make the run-up climb higher through anything it does
in the run-up itself; the full-run difference is flow history (the late flow is
chaotic). Remaining question, statistical: late P2 pressure in 3/5 `impulse`
runs vs 0/8 `finalize` runs (mixed nx / schemes) → seeds 4-6, both modes,
drift fix on (`scratchpad/ceiling_seeds456.sh` → `scripts/out_ceiling/m31_seeds456/`).

**Seeds 4-6 (both modes, drift fix on): not systematic.** Probe checks
`finalize` 9/9, 9/9, 8/9 vs `impulse` 9/9, 9/9, 8/9 — seed 6 fails "P2 back to
quiescent" in *both* (late P2\* 0.42 / 0.77). Late P2 pressure (t\* > 6.4, wet)
over the six drift-fix seeds: `finalize` seeds 5, 6 (0.14, 0.42), `impulse`
seeds 2, 5, 6 (0.18, 0.065, 0.77; 0.016 in seed 1). P1 0.46 everywhere, P2 peak
0.24-0.28; max|v| 9.9 (finalize) vs 9.4-9.5 (impulse), all the jet at t\* 5.5;
ρ [0.96, 1.03] vs [0.98-0.99, 1.03-1.04]. So the late run-up at P2 is the
chaotic late flow in both modes, and the check's t\* 6.4 cutoff sits inside
that spread; §4 item 4 closed as not-an-impulse-effect (the sloshing corner
jet is the same kind of late-flow difference, not re-tested separately).

Plotting: `probe_deltaSPHMarrone.py --report` / `probe_deltaSPHMarrone34.py
--traceFigure` now tag each legend entry with the noPen mode, drift fix and
seed from the file name (baseline and new runs had identical labels), and the
3.1 report caps its y-axis at the spike gates, naming off-scale runs. Before /
after figures: `scripts/out_ceiling/compare_2026-09-29/` (the 2026-09-28
`finalize` seed 2 reads P2\* 11.5 at t\* 6.3 there: a kicked particle on the
probe, gone in every new run).


## 4. Open

1. **Default-on decision (user) — DECIDED 2026-09-29: both on.** User:
   "change the defaults considering they generally seem to be improvements and
   avoid real instabilities with a principled fix": `timeCentredContinuity=True`,
   `mdbcNoPenShiftMode='impulse'` (config defaults; probe flags now on/off,
   `--no-timeCentredContinuity` / `--noPenShift finalize` for the old
   behaviour; probes record the effective values in their meta).
   warpSPHIntegrators `drift-fields` merged into its `main` (cb86424, full
   suite 2846 passed / 373 skipped). The gates below were run as §7.2
   (except the hydrostatic column and the gallery): Gates before flipping the default: Marrone
   3.4 nx256 (δ-SPH + δ⁺), sloshingTank t = 7 (sensors), a hydrostatic column
   at rest (negative control), the gallery's symplecticEuler cases; the
   nx134 Marrone δ⁺ runs (max|v| 23-30 there: same kick?). The same gates now
   apply to `mdbcNoPenShiftMode='impulse'` (§7.1), which touches every wall
   contact; sloshingTank matters most there, since `finalize` was adopted to
   fix its t = 4.57 s divergence (DELTASPH_VALIDATION_PLAN 5.13(c)). Also
   worth deciding: DualSPHysics does not run NoPen under free slip at all.
2. **The sticking itself (§3.2)** is the bilateral wall and is untouched. With
   the integrator no longer amplifying contact modes, the earlier δ-SPH
   complementarity attempts (MDBC_CONTACT_LINE_PLAN §8-9, rejected for spraying
   fliers) are worth re-reading: some of those fliers may have been this
   instability. A principled version needs both halves the forensics show:
   the english2025 hydrostatic extension (contact pressure ≥ 0) and the
   continuity booking of separation against the mirrored wall velocity.
   New input from §7.1: the noPen replacement is part of the hold too. With
   noPen `off`/`derivative`/`impulse` the seed-2 rider slowly separates
   (s 0.25 → 0.50 dx over t\* 4.4-4.7); the riders still stay on in the full
   runs. The remaining wall-pair energy mismatch (mirrored velocity books 2 v_n
   against a force that carries P_i once, §7.1) is small but real.
3. **Edge in the noPen kernel (not changed):** a row past a wall row's centre
   can get factor < 0 (ratio clamp [0.25, 1] → factor ∈ [-1, 2]), which would
   push it further in; under `impulse` that is |e| > 1.
4. **(Closed 2026-09-29, §7.2: realisation spread, both modes)** **Wall run-up tips under `impulse` (§7.2):** higher/faster thin jets along
   vertical walls (M3.1 P2 late re-wetting with pressure, sloshing corner jet
   7.7 m/s). Next: forensics on one run-up tip (as §7.1: per-step budget of the
   film-tip rows under finalize vs impulse) before any default decision.

## 8. The sticking: a unilateral wall contact, switched as a whole (2026-09-29)

> **Code parked (2026-09-30): §8-§9 are not in `dev`.** `unilateralWallContact` (+ refinements),
> `densityReinit`, `modules/mdbc/wallContact.py`, `tests/test_unilateralWallContact.py`, the `--fliers` /
> `--unilateralWallContact` / `--schemeOpt` probe options and the forensics scripts live on branch
> **`parked/ceiling-uniwall`** (user: the δ-SPH and other models are to stay free of these fragments).
> The text below describes that branch.

User: take on the ceiling sticking (§4 item 2) after the defaults flip.

**Design, from the equations and §7.1 / MDBC_CONTACT_LINE_PLAN §10.** The wall
enters a fluid row through three pair sums: the pressure force
`-(1/ρ_i) Σ_wall V_j (P_i + P_j) ∇W`, the continuity `-ρ_i Σ_wall V_j (v_j - v_i)·∇W`
(mirrored v_j) and the artificial viscosity. Batty 2007, Chentanez & Müller
2011 and Inglis 2017 all switch a contact **as a whole**: bilateral (in
momentum *and* continuity) or air (in neither), by the **sign of the contact
pressure**, never by the velocity sign (the Foster-Fedkiw rule of §9.4). The
failed δ-SPH attempts each switched half: §7.5 clamped only the pressure (the
continuity kept booking separation as expansion: stored deficits, ρ 0.21);
§8.3/§9 dropped only the receding continuity (a receding row with P > 0 keeps
being pushed but never unloads — by §7.1's analysis an energy source, which
fits "fliers launched off every wall").

Here, per fluid row i (no kernel change; the wall-pair parts by linearity, as
the §3/§7 forensics):
- `F_w,i = -(1/ρ_i) Σ_wall V_j (P_i + P_j) ∇W`, `G_i = Σ_wall V_j ∇W` (points
  into the wall);
- **open** iff i is air-reachable (surface or isolated row) **and**
  `F_w,i · G_i > 0` (the wall's net traction on i is tensile);
- open ⇒ subtract the wall-pair pressure force, continuity and artificial
  viscosity of row i (the wall is air for it); the mask is stored on the
  stage state so the drift path (§6) removes the same continuity pair.

What it cannot do by construction: open a floor at rest (traction
compressive), touch a sealed box (no surface rows), book a separation as
expansion (the continuity pair goes with the force). noPen (impulse) stays:
it is already unilateral. Option `schemeConfig.unilateralWallContact`, default
off.

Checks (exact answers): the `drop` toy (a block under a ceiling with air at
both contact lines: free fall, baseline fallRatio 0.67), the rider toy (a lone
particle under a ceiling: free fall from rest normal velocity), `column` (at
rest: unchanged), `sealed` (unchanged, bitwise); then Marrone 3.1 seeds 1-3
(ceiling riders, flier count, frames) and sloshingTank.

### 8.1 Implementation and toys (2026-09-29)

`modules/mdbc/wallContact.py` (`wallContactOpen`, `wallPairContinuity`,
`wallPairViscosity`), applied in `_deltaSPH_rhs` after the pressure force;
the mask is a declared state field (`WeaklyCompressibleState.wallContactOpen`)
because a graphed right-hand side only hands dataclass fields back on replay
(first version as an ad-hoc attribute: the drift path never saw it after the
capture step, so the continuity pair stayed in the density — the rider fell
freely but expanded to ρ 0.45). `finalize` carries the mask to the next step.

Air-reachability had to be **topological**, not per row: with "surface rows
only" the `drop` block released only at its two contact lines (fallRatio
0.884, the wall still carrying 30-60 % g through the interior top rows). In a
single-phase model the gap a separating row leaves is p = 0 air whenever it
connects to the atmosphere, i.e. the whole tensile contact patch with a free
surface edge opens; the patch grows from surface rows (and the previous
step's open rows) by one kernel support per evaluation, faster than sound.

| toy (exact answer) | off | on |
|---|---|---|
| `drop` nx32 t 0.3 (free fall ≈ 0.97; §8.5 'both' 0.984) | 0.669 | **0.981**, falls as one block, p ≈ 0 |
| rider toy (free fall s = 3.538 dx at 0.1 s, E conserved) | sticks | s = 3.538 / 3.539 (vx 0 / 7), P ≡ 0, \|ΔE\| ≤ 7e-5 |
| `column` (unchanged) | fall 0.0018, slope −12.46 | 0.0017, −12.66 |
| `sealed` (unchanged) | — | identical |

### 8.2 Marrone 3.1 (seed 1 first): the riders go, but the run-up peels off the vertical wall

`scratchpad/uniwall_m31.sh` (δ⁺ nx67, current defaults, `--fliers`: free
fliers / wall-sparse / ceiling rows per step). Seed 1, off → on: ceiling rows
(t\\* > 3.5) mean **22.6 → 0.07**, wall-sparse 2.32 → 0, free fliers 0 → 0,
max|v| 9.6 → 7.9; **checks 9/9 → 7/9**: P2 reads **1.8-2.7 ρgH at t\\* 3.3-3.9**
(Buchner: quiescent before 3.6), P2 late wet. Frames (t\\* 3.0-4.2): the
run-up sheet no longer climbs the right wall to the ceiling — it peels off the
vertical wall, curls back and falls onto itself (that impact is the P2 reading).
A thin sheet rising along a straight vertical wall has no physical reason to
separate. Hypothesis (being measured): the thin sheet carries the spurious
free-surface truncation tension of OPEN_PROBLEMS §1 in its own P_i; the
bilateral wall balanced it, the traction criterion reads it as the wall
pulling and opens the contact, and the sheet's fluid-fluid tension then pulls it
off the wall. If so the traction *sign* cannot tell real from spurious tension.

All three seeds (`scripts/out_ceiling/m31_uniwall/`), off → on:

| seed | checks | P1 / P2 peak | max\|v\| (t\\*) | ρ extremes | ceiling rows t\\*>3.5 mean | wall-sparse | free fliers (births, vmax) |
|---|---|---|---|---|---|---|---|
| 1 | 9/9 → 7/9 | 0.46/0.27 → 0.44/0.20 | 9.6 (5.50) → 7.9 (4.21) | [0.99,1.03] → [0.93,1.05] | 22.6 → 0.07 | 2.32 → 0 | 0 → 0 |
| 2 | 9/9 → 8/9 | 0.46/0.23 → 0.47/0.27 | 9.3 (5.50) → 7.5 (2.45) | [0.99,1.02] → [0.97,1.07] | 25.4 → 0.19 | 1.85 → 0 | 0 → 5 births, ≤ 5.8 m/s |
| 3 | 9/9 → 6/9 | 0.46/0.25 → 0.45/0.19 | 9.5 (5.50) → 8.4 (4.66) | [0.98,1.02] → [0.97,1.03] | 21.8 → 0.04 | 3.32 → 0 | 0 → 0 |

Failed checks with the option: P2 quiescent before impact (seeds 1, 3), P2
back to quiescent (all three), P2 raw spike (seed 3). The sticking is gone in
every seed; the run-up / P2 regression is systematic. Not shippable as is.

### 8.3 Forensics: what the traction criterion actually reads (2026-09-30)

User: the uniwall run looks hacky (sheets separate from vertical walls, particles
bounce off the ceiling before close contact). Measured with
`scratchpad/uniwall_diag.py` (Marrone 3.1 δ⁺ nx67 seed 1, uniwall on, eager; every
10th RHS call, log `scripts/out_ceiling/uniwall_diag/run.log`, buckets = 4000
calls, not time-aligned): every open row's traction `T = F_w·G` split into the
row's **own-pressure** part (`P_i · ΣV∇W`) and the **wall-ghost** part
(`Σ V P_j ∇W`), by wall orientation from `G`.

- **~85-95 % of open rows have `P_i < 0`** (vertical: 78-88 %, floor: 94-100 %,
  ceiling: 83-89 %); the own-pressure part exceeds the ghost part in 60-90 %;
  "tension from the wall alone" (`P_i ≥ 0`, ghost pulls) is only 5-20 %.
  The criterion is reading the row's own free-surface tension (§1's truncation
  artefact), not the wall pulling.
- **Floor contacts open** — thousands of row-samples over the run-up / spread
  phase, 55-100 % of them surface rows with `<P_i>` ≈ −0.4…−3. A film on a
  floor has gravity pressing it *into* the wall; it must never open. Vertical
  walls open as well (up to ~900 samples in the early run-up).
- Opening removes the whole wall-pair (`P_i + P_j`) sum, and with it the wall
  half of the row's `P_i ΣV∇W` consistency cancellation: the row becomes a
  truncated free-surface row that keeps `P_i · ΣV∇W_fluid`, the §1 spurious
  force. That is the mechanism behind the peel-off.

Consequence: the traction *sign* is not a contact criterion in WCSPH, since the
EOS lets every surface row carry tension the bilateral wall used to balance.
Batty / Chentanez / Inglis assume `p ≥ 0` at contacts (incompressible), where
this cannot arise. Candidate refinements (not tried): (a) test only the tension
the *wall* imposes (own pressure clamped at its air value 0: `P_i → max(P_i, 0)`
in `F_w`, `G` unchanged); (b) gate by the body force: a contact can only be in
genuine tension where the effective body acceleration `g − a_wall` has a
component pulling the fluid away from the wall (`n_in · g_eff > 0`), which is
the ceiling and nothing else; (c) park the option (opt-in, default off).

### 8.4 The two refinements, and what the fliers really are (2026-09-30)

Implemented (opt-in, default off; `wallContact.py`, tests in `tests/test_unilateralWallContact.py`, 5 pass):
`unilateralWallAirOwnPressure` (a: traction with `max(P_i, 0)`) and `unilateralWallBodyForceGate` (b: a row may
open only where the body force pulls it off the wall by more than along it, `g·n_in > |g_t|`, i.e. wall normal
within 45° of g; the first version, sign of `g·n_in` only, opened rows at random on a wall parallel to g because
`G·g` was rounding noise, caught by the unit test). Toys (drop nx32, column, sealed; a / b / both): drop free fall
0.981 in all, column and sealed identical to plain uniwall.

Marrone 3.1 δ⁺ nx67, seeds 1-2 (seed 3 not run: batch stopped by the user), baseline `off` = 9/9, riders 22.6 / 25.4,
`m31_uniwall2/`:

| | seed 1 checks | riders | fliers (births, vmax) | ρ range | seed 2 checks | riders | fliers |
|---|---|---|---|---|---|---|---|
| uniwall | 7/9 | 0.07 | 0 | [0.93, 1.05] | 8/9 | 0.19 | 3 (5, 5.8) |
| + (a) | 7/9 | 0.11 | 0 | [0.97, 1.06] | 8/9 | 0.20 | 0 |
| + (b) | **9/9** | 3.7 | **6 (30, 7.7)** | [0.96, 1.08] | 8/9 | 2.0 | 3 (8, 6.8) |
| + (a) + (b) | **9/9** | 2.0 | 6 (17, 6.8) | [0.96, 1.08] | **9/9** | 2.2 | 3 (12, 7.0) |

(b) fixes what §8.2 found (the run-up climbs the wall again, checks back) and releases most riders, **but the released
rows become lone particles and the run has fliers, a high density band and a particle stuck in the top-right
corner.** Videos: frames show small clumps leaving the ceiling at the run-up jet speed (6.7-7.7 m/s vs 6.8) and
splashing back into the free surface.

**Forensics of the fliers** (`scratchpad/flier_forensics.py`: every step, every flier's x, v, ρ, P, nF, nW;
`scripts/out_ceiling/flier_forensics/`; `scratchpad/flier_restitution.py`). Seed 1, (b):
- **Lone-particle impacts are nearly elastic and violent.** 16 free→contact→free events, 6 fliers: KE after/before
  mean **0.68**, range 0.09-1.13, **7 of 16 in 0.9-1.13** (one gains 13 %); a contact lasts 10-15 ms (~100 steps) and the
  single particle's pressure peaks at **150-760** (25-130 ρgH; ρ c v_n water-hammer scale with c = 97 m/s). Example
  (uid 80, t 1.159): v (−5.4, −2.9) → (−4.0, +3.9), ρ 1.036, P 337: the normal velocity is *reversed with restitution
  1.36*. Cause: a lone row is a compressible ball, stiffness c², with no cohesion and (inviscid, AV α = 0.01, approach
  pairs only) no dissipation; contact is half a period of its own compression mode. A real droplet merges with the pool.
- **Fliers' pair collisions** are also elastic (uid 323, t 1.695: 3 ms, P 148, vy ratio 1.01). Not isolated more
  than that; the "violent interaction" is the same compressible contact, at the same c²-scale pressures.
- **The corner particle is a clump of three** (uids 644, 725, 1131): they drift together from 1.3 dx to **0.01 dx** apart
  over 0.8 s (each P ≈ −0.8, nF = 2, nW = 35, v → 0). Negative pressures make the symmetric pair force attractive
  (tensile clumping) and the pair force vanishes at r → 0, so nothing separates them. Held there because the gate stays
  closed in a concave corner: `G` sums both walls and points diagonally, `g·n_in ≈ |g_t|` (likely, not measured).
- Collision dissipation test: `inviscidAlpha` 0.01 → 0.1 (gate, seed 1): **zero fliers, zero riders, ρ [0.995, 1.013]**,
  but the whole flow is damped (max|v| 7.7 → 4.6) and P2 run-up peak fails (8/9); α = 0.5: 6/9, max|v| 2.9. A blunt
  instrument, not usable on the reference case.

Reading: releasing a stuck row is the easy half. The released row is a sub-support fragment, which this model
represents as an elastic compressible ball, and the same population is in OPEN_PROBLEMS §1 / MDBC_CONTACT_LINE §12
(`loneDensityReset`: fliers up 4/4). The sticking was masking it. Untried: an AV coefficient that rises on
compression (Morris & Monaghan 1997 switch, already in the repo, `viscositySwitchParams`), and a per-wall-neighbour
gate for corners.

## 9. EXPERIMENT: density re-initialisation (2026-09-30)

User, on the alternatives to a Batty-style switch (MDBC_CONTACT_LINE_PLAN §8.1: the
CG methods never stick because the constraint is unilateral *and* their density is
re-derived from positions): try it, **only as an experiment** — the point of the
WCSPH density is that it is never a function of the current configuration, a
purely integrated quantity (memory, temporal-average incompressibility; a solver
not given ρ is truly ill-conditioned, which is worth investigating). A
summation-density floor breaks that, so it is opt-in and never a default
(PLANS.md decisions log).

Implemented: `schemeConfig.densityReinit` (default off), `systems/weaklyCompressible.py`
`finalize`: Rezavand et al. 2022 Eq. 2.8 (as transcribed in MDBC_CONTACT_LINE_PLAN §8.2;
the paper itself is not in `literature/`, not re-checked),
`ρ ← ρ_sum + max(0, ρ − ρ_sum) ρ₀/ρ` on fluid rows, `ρ_sum` = `computeDensities` on
the last stage's configuration and adjacency (as `loneDensityReset`). To first order a
deviation `ε = ρ − ρ₀` of a truncated-support row is damped by `ρ_sum/ρ` (a lone
row → ρ₀, i.e. §12's reset), a row below its geometric density is lifted to it.
Probe: `probe_deltaSPHMarrone.py --schemeOpt densityReinit [--schemeOpt mdbcOneSidedHydrostatic]`.

Marrone 3.1 δ⁺ nx67, current defaults, seeds 1-3, t\* to 7.68, video on
(`scripts/out_ceiling/m31_reinit/`, `scratchpad/reinit_m31.sh`, `scratchpad/reinit_summary.py`).
Baseline = the `off` runs of §8 (`m31_uniwall/`), same settings. Ceiling = fluid rows
within the ceiling band at t\* > 3.5 (mean / max):

| seed | | checks | ρ range | max\|v\| (late, t\*>3.5) | ceiling rows | wall-sparse | free fliers (births) |
|---|---|---|---|---|---|---|
| 1 | off | 9/9 | [0.989, 1.027] | 9.6 (9.6, t\* 5.5) | 22.6 / 37 | 2.32 | 0 |
| | reinit | 9/9 | [0.988, 1.012] | 7.0 (5.0) | **29.4** / 44 | 1.36 | 0 |
| | + one-sided hydro | 9/9 | [0.988, 1.012] | 7.0 (5.8) | 16.7 / 44 | 1.36 | 0 |
| 2 | off | 9/9 | [0.988, 1.024] | 9.3 | 25.4 / 39 | 1.85 | 0 |
| | reinit | 8/9 (P2 back to quiescent) | [0.988, 1.011] | 7.0 | **29.4** / 42 | 1.69 | 0 |
| | + one-sided hydro | 8/9 (same) | [0.988, 1.011] | 7.0 | 20.9 / 39 | 2.00 | 2 (2, ≤ 4.4 m/s) |
| 3 | off | 9/9 | [0.981, 1.024] | 9.5 | 21.8 / 33 | 3.32 | 0 |
| | reinit | 9/9 | [0.988, 1.011] | 7.0 | **35.0** / 50 | 5.55 | 1 (1, 3.0 m/s) |
| | + one-sided hydro | 9/9 | [0.988, 1.011] | 7.0 | 22.8 / 47 | 2.37 | 0 |

(7.0 is the same early event, t\* 2.46, in every run; seed 1's late maximum is the
number in brackets; seeds 2-3 late maxima are below 7.0.)

Reading:
- **It does not release the ceiling riders** — it slightly *increases* them
  (22-25 → 29-35). The sticking is therefore not mainly the continuity-booked
  density deficit (the plan's "half each" split overstated that half, or the
  other half is not just the hydrostatic term): with ρ floored at the geometric
  density the riders stay held by the wall pressure + fluid neighbours.
  One-sided hydrostatic on top: 29→17, 29→21, 35→23, i.e. back to about the
  baseline (≈ −25 % … 0 vs off), not a fix.
- **What it does do**: the ρ band tightens on both ends (upper 1.024-1.027 → 1.011-1.012)
  and the late fast ejection disappears in all three seeds (late max\|v\| 9.3-9.6 →
  ≤ 5-6). Checks: seeds 1 and 3 unchanged (9/9); **seed 2 drops to 8/9** ('P2 back to
  quiescent', which the baseline passes on that seed; it also fails with uniwall on it, and
  seeds 4-6 fail it 1/3 in *both* noPen modes, §7.2 — probably late-flow spread, but not
  shown here). So it acts on the §1/§7 kick population, which the integrator fixes had
  already reduced, and not
  on the sticking.
- **Not looked at**: the videos (recorded next to the npz files) were not inspected
  for this table; one realisation per variant; sloshingTank not run.
Conclusion: as an experiment it is informative (rules out the continuity deficit as
the main sticking mechanism, but is a cheap density-band / late-kick suppressor),
not a fix. The remaining candidates for the sticking are §8.3 (a) / (b)
(traction criterion without the fluid's own tension; body-force gating).

### 8.5 Decision: uniwall parked (user, 2026-09-30)

Every step of §8 needed a further criterion to repair the previous one: traction sign → (a) own-pressure clamp →
(b) gravity / wall-orientation gate (itself fixed once for sign noise, then a 45° split) → a per-neighbour corner
gate → collision damping for what it releases. That is a chain of case-shaped switches, not a physical fix, and it
will not generalise: the gate assumes a static wall and a fixed gravity (a wall moving downwards changes
`g_eff = g − a_wall` and with it which contacts may release; a gate written on `g` alone is wrong for it), and every
new geometry (corner, curved wall) needs another patch. The released rows are also the real defect: sub-support
fragments behave as elastic compressible balls (§8.4), a model property that a contact switch cannot fix.

State: `unilateralWallContact`, `unilateralWallAirOwnPressure`, `unilateralWallBodyForceGate` stay in the tree
(uncommitted at this point), opt-in, default off, tests pass; not to be extended. The sticking itself is unchanged and
open (cosmetic: 22-25 ceiling rows, no dynamics); a fix has to come from the model (cohesion / a treatment of
sub-support fragments, or the incompressible route), not from another gate.

### 8.6 Literature pointers on the elastic-fragment observation (2026-09-30)

No paper found that describes or quantifies the restitution of a lone SPH particle. Nearest (local `literature/`, excerpts
read, not full papers): Schechter & Bridson 2012 (basic SPH: "unnatural clusters instead of spray"; isolated particles
treated as ballistic; disallowing negative pressure at under-resolved boundaries, citing Batty 2007 / Chentanez & Müller
2011), Akinci 2013 (cohesion + curvature forces; SPH error creates isolated particles), Michel 2022 (isolated particles'
mass / volume deviate in ALE schemes), Dehnen & Aly 2012 (Wendland kernels avoid the pairing instability although the
kernel derivative vanishes at the origin; so the corner clump of §8.4 is tension-driven, not a zero-gradient effect);
outside `literature/`, abstract-level only: Ihmsen et al. 2012 (spray as a non-interacting diffuse class),
Losasso et al. 2008. User's scoping (2026-09-30): surface tension is out of scope (research work); Schechter's ballistic
isolated particles ≈ DualSPHysics' exclusion of out-of-range fluid; Michel's ALE handling of isolated particles goes with
the δ-ALE-SPH derivation already on the list.
