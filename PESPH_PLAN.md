# warpSPH — pressure-based SPH (PESPH) plan

## Why this exists

`schemes/` currently has three compressible schemes: `monaghan.py`, `compSPH.py` and
`crkSPH.py`. The fourth scheme in the reference comparison —
Frontiere, Raskin & Owen (2017), *CRKSPH*, JCP 332:160-209 — is **PESPH**, the
pressure-based formulation, and it is missing. Adding it closes the comparison set,
gives the compressible branch two strong astro-side baselines, and (in one of its two
variants) produces the first scheme in this repo that is a **clean
`f(state) -> update`** — no step-level energy reconstruction, no pairwise arrays
outliving a substep.

That last property is why this plan is sequenced **before** the integrator work in
`../warpSPHIntegrators/SPLITTING_PLAN.md` rather than after it. See §6.

Two variants exist and both are worth having:

- **PESPH-E (pressure-energy)** — pressure by summation over neighbours' *thermal
  energy*, evolving *total* energy. This is what Frontiere Appendix G and Hopkins (2015)
  Appendix F2 actually implement, and what GIZMO/FIRE run in production.
- **PESPH-A (pressure-entropy)** — pressure by summation over neighbours' *entropic
  function* `A`, evolving `A`. Hopkins (2013), MNRAS 428:2840. This is the variant that
  makes the scheme separable and variational.

They share roughly 90% of their code and should be **one scheme with a switch**, not two
schemes (§3.1).

Papers are in `../warpSPHIntegrators/literature/`:
`1-s2.0-S0021999116306453-main (1).pdf` (Frontiere et al.) and `stv195.pdf` (Hopkins
2015).

---

## 1. The equations

### 1.1 PESPH-E (pressure-energy), Frontiere Appendix G / Hopkins F2

```
n_i  = Σ_j W_ij(h_i)                                                        (G.3)
ρ_i  = Σ_j m_j W_ij(h_i)                                                    (G.2)
P̄_i  = Σ_j (γ−1) m_j u_j W_ij(h_i)                                          (G.1)

Dv_i/Dt = −Σ_j m_j (γ−1)² u_i u_j [ (f_ij/P̄_i) ∂_α W_ij(h_i)
                                  + (f_ji/P̄_j) ∂_α W_ij(h_j) ] + q_acc_ij   (G.4)

DE_i/Dt = m_i v_i·Dv_i/Dt
          + Σ_j m_i m_j (γ−1)² u_i u_j (f_ij/P̄_i) v_ij·∂_α W_ij(h_i)
          + v_ij·q_acc_ij                                                    (G.5)

q_acc_ij = ½ (ρ_i Π_i + ρ_j Π_j) (∂_α W_i + ∂_α W_j)/(ρ_i + ρ_j)             (G.6)

f_ij = [1 − (h_i / (ν(γ−1) n̄_i m_j u_j)) ∂P̄_i/∂h_i] [1 + (h_i/(ν n_i)) ∂n_i/∂h_i]⁻¹  (G.7)
∂n_i/∂h_i = −Σ_j h_i⁻¹ [ν W_ij(h_i) + η_ij ∂W/∂η|η_ij]                       (G.8)
∂P̄_i/∂h_i = −Σ_j (γ−1) m_j u_j h_i⁻¹ [ν W_ij(h_i) + η_ij ∂W/∂η|η_ij]        (G.9)
```

with `η_ij ≡ |x_j − x_i|/h_i` and `ν` the dimension. `Π` is Monaghan-Gingold (Eq. E.7)
under the Cullen-Dehnen switch (Appendix F). `u_i = E_i/m_i − v_i²/2`.

### 1.2 PESPH-A (pressure-entropy), Hopkins (2013)

Identical except that the pressure summation is over the entropic function and the
integrated thermodynamic variable is `A`:

```
P̄_i = [ Σ_j m_j A_j^(1/γ) W_ij(h_i) ]^γ
A_i = P_i / ρ_i^γ ,   u_i = A_i ρ_i^(γ−1)/(γ−1) ,   c_{s,i} = √(γ ρ_i^(γ−1) A_i)

DA_i/Dt = 0                              for adiabatic flow
DA_i/Dt = (γ−1) ρ_i^(1−γ) (du_i/dt)|_diss    with viscosity/conduction
```

and `∂P̄_i/∂h_i` follows from G.9 with the weight `m_j A_j^(1/γ)` and the outer `γ` power.

### 1.3 Naming hazard — read this before writing code

**Frontiere uses `f_ij` for two completely unrelated quantities:**

- Eq. (67) — the compSPH/CRKSPH **energy-partition** factor, constrained by
  `f_ij + f_ji = 1`.
- Eq. (G.7) — the PESPH **grad-h correction**, the analogue of compSPH's `Ω_i`.

`CompressibleState.f_ij` in this repo already means the first
(`modules/compSPH/balance.py`). Pick a different name for the PESPH term —
`gradH_ij`, or `fH_ij` — before any of this is typed, or the two will be conflated in
review.

---

## 2. Audit: what this repo already has

Checked 2026-09-15. Substantially more than expected:

| PESPH ingredient | Status |
|---|---|
| Cullen-Dehnen viscosity switch (App. G specifies it) | ✅ `ViscositySwitch.CullenDehnen2010` **and** `CullenHopkins` (`enumTypes.py:43`, `modules/shockCapturing/`) |
| Artificial conductivity (App. G requires it) | ✅ `modules/dissipation/wp_conductivity.py` |
| `∂W/∂h` primitive for grad-h terms | ✅ `sphKernelDkDh`, live in `modules/adaptiveSupport/wp_omega.py:81` (not just spike scripts) |
| Linearly-corrected velocity gradient (E.4/E.6) | ✅ used by compSPH; App. G says PESPH uses it too |
| Entropic-function EOS `A ↔ (u, P, c_s)` | ⚠️ **exists but is dead code** — `modules/eos/gas.py`, `EOSSource.specificEntropy`, with all four conversions written. Its own docstring: "Not wired into `eos/__init__.py` or any scheme/case in this repo." |
| `entropies` / `totalEnergies` on the state | ⚠️ present but `constant()`, not `integrated()` (`systems/compressibleMonaghan.py:37-38`) |
| `dEdt` on the update | ✅ `CompressibleSystemUpdate.dEdt`, and all three schemes already compute it |
| Number density `n_i = Σ_j W_ij` | ✅ effectively — CRK's `V_i⁻¹ = Σ_j W_i` (Frontiere Eq. 75) is exactly this kernel |
| Scheme registration | ✅ ~10-line `SchemeBundle` (`schemes/builder.py:97`) + one `CompressibleSPHScheme` enum entry |
| Validation cases | ✅ `caseUtils/compressible/`: sod, sedov, noh, kidder, gresho, yeeVortex, kelvinHelmholtz, rayleighTaylor, triplePoint, hydrostatic, blob — **Frontiere's own test battery** |

### 2.1 Two state-field bugs found during the audit

`systems/compressibleMonaghan.py:36-39`:

```python
totalEnergies : torch.Tensor = constant(tags=('energy',), default=None)
entropies     : torch.Tensor = constant(tags=('soundSpeed',), default=None)   # <-- wrong tag
pressures     : torch.Tensor = constant(tags=('damping',), default=None)      # <-- wrong tag
soundspeeds   : torch.Tensor = constant(tags=('soundSpeed',), default=None)
```

`entropies` and `soundspeeds` carry the **same** tag, and `pressures` is tagged
`'damping'`. Any tag-based lookup (`find_tagged_field` / `get_tagged_attr`) resolves
`soundSpeed` ambiguously and `pressure` not at all. Harmless today because nothing does
tag lookup on these — but PESPH-A promotes `entropies` to a first-class *integrated*
field, so fix the tags as part of Phase 0 rather than inheriting the ambiguity.

### 2.2 What is genuinely new

Short list:

1. **Pressure-by-summation kernel** (G.1 / §1.2) — structurally a density-summation
   variant, low risk.
2. **The grad-h `f_ij` kernel** (G.7-G.9) — needs `∂n_i/∂h_i` *and* `∂P̄_i/∂h_i`. This is
   the risk item, see §2.3.
3. **Momentum kernel** (G.4) — pattern-match `modules/compSPH/accel.py`.
4. **Energy kernel** (G.5 for PESPH-E, the entropy source for PESPH-A).
5. **State plumbing** — promote `totalEnergies` (PESPH-E) or `entropies` (PESPH-A) to
   `integrated()`, add `dAdt` to the update for the latter.

### 2.3 The risk item: number-density-driven `h`, and the grad-h derivatives

Hopkins is explicit that PESPH sets the smoothing length from **particle number
density, not mass density** — "reducing errors when there are particles of different
masses in the same kernel" (2015, App. F2).

This repo's `evaluateOptimalSupport` is mass-density driven. Both backends return
`(densities, supports, adjacency, …)` and neither exposes the `h`-derivatives PESPH
needs (`modules/adaptiveSupport/optimalSupportMonaghan.py:124`,
`optimalSupportOwen.py:159`; `computeH(rho, m, targetNeighbors, dim)` is the Monaghan
relation).

So Phase 2 needs a number-density h-solve variant plus two new derivative
accumulations in a `computeOmega`-shaped kernel. **This is where the estimate can
slip.** A wrong grad-h term produces roughly 1% energy-conservation drift that presents
as a mysterious bug rather than an obvious failure, and it can survive every test that
does not specifically probe it. Budget a dedicated check against the Kidder isentropic
case (`caseUtils/compressible/kidder/`), which is exactly the case sensitive to it, and
do not proceed to Phase 3 until it passes.

---

## 3. Design decisions

### 3.1 One scheme, one switch

PESPH-E and PESPH-A differ in exactly three places: the integrated thermodynamic
variable (`E` vs `A`), the pressure-summation weight, and the matching `∂P̄/∂h`.
Everything else — momentum kernel, viscosity, conductivity, grad-h machinery, h-solve,
orchestration, config, registration — is shared. Build one `schemes/pesph.py` with a
`PESPHVariant` switch on the config. Two scheme files would duplicate four warp kernels
for no gain.

### 3.2 PESPH-A is the better first target

Not merely because it is less work. Three properties coincide only in this variant:

- **It is a clean `f(state) -> update`.** `DA/Dt = 0` adiabatically plus a viscous
  source. No compatible-energy step-map, so no `ap_ij`/`av_ij`/`f_ij` pairwise arrays,
  no `compSPH_deltaU_multistep`, no `finalize` override (contrast
  `systems/compSPH.py:118`). It works with every registered integrator immediately.
- **It is separable** in the Hamiltonian sense — `A` is the Lagrangian constant that
  keeps the SPH potential a pure function of position. See
  `../warpSPHIntegrators/SPLITTING_PLAN.md` §2.3.1.
- **It is variational**, so it is genuinely symplectic under a suitable integrator —
  unlike CRKSPH (§4).

PESPH-E, by contrast, is the one scheme in the comparison set that **cannot** be made
separable: because it integrates total energy, `u_i = E_i/m_i − v_i²/2` is
velocity-dependent by definition, and `P̄_i` (G.1) is a kernel sum over the neighbours'
`u_j`. The conservative force therefore depends on velocity before any viscosity enters.
Build it as a baseline — it is what GIZMO and FIRE run — but do not expect it to feed
the integrator work.

### 3.3 Known weakness of PESPH-A, stated up front

Pressure-entropy handles **non-adiabatic** physics awkwardly: heating must be converted
back into entropy, and the `A ↔ u` round-trip degrades with strong shocks plus
conductivity. This is precisely why Hopkins' own GIZMO implementation uses
pressure-*energy*. Guard `A > 0` explicitly (the `A^(1/γ)` summation is undefined
otherwise) and expect PESPH-E to be the better choice on shock-dominated problems. The
two variants are complementary, which is a second argument for the switch design.

---

## 4. CRKSPH: observations and concerns from the same reading

Recorded here because they came out of the same pass over the paper and the code, and
because they change how `crkSPH.py` should be described — not because they require
action now.

### 4.1 CRKSPH is not variational, and the repo should not imply it is

Appendix B is explicit. Replacing `W` with the reproducing kernel `W^R` breaks
`∂_α W_ij = −∂_α W_ji`, so the usual Lagrangian derivation of a conservative momentum
equation does not work. Eq. (64) is instead obtained by **conservative differencing**
(MLSPH-style integration by parts), which buys exact pairwise antisymmetry — and hence
machine-precision linear momentum conservation — at the cost of a consistency error that
vanishes only for uniformly spaced particles.

The consequence worth recording: **there is no potential `V(q)` behind CRKSPH's
pressure force.** A kick substep `(q,v) ↦ (q, v + τF(q))` has Jacobian
`[[I,0],[τ ∂F/∂q, I]]`, which is always volume-preserving but is *symplectic* only when
`∂F/∂q` is symmetric, i.e. only when `F` is a gradient. So CRKSPH is order-preserving and
volume-preserving under a symplectic-style integrator, but has **no shadow Hamiltonian**
and no bounded-energy-error guarantee. compSPH, being Lagrangian-derived (its `Ω` terms
in Eq. E.5 *are* the grad-h corrections of the variational formulation), does have one.

### 4.2 The compatible energy update is not an ODE right-hand side

Frontiere Eqs. (65)-(68), implemented here as `modules/compSPH/balance.py` +
`modules/compSPH/multistep.py` + the `finalize` override at `systems/compSPH.py:118`:

```
u_i(t+Δt) = u_i(t) + Σ_j Δu_ij Δt
Δu_ij = (f_ij/2)[v_j(t) + v_j(t+Δt) − v_i(t) − v_i(t+Δt)]·(Dv_ij/Dt)
```

This is a **step-level algebraic map**, explicitly `Δt`-dependent, implicit in
`v(t+Δt)`, and tableau-aware (`compSPH_deltaU_multistep` consumes `butcherTerms` and the
per-stage pairwise arrays). It is outside the `f(state) -> update` contract that
`warpSPHIntegrators` is built on, which is why `compSPH`/`crkSPH` need a `finalize` hook
where PESPH-A will not.

Two concerns follow:

- **`f_ij` (Eq. 67) is piecewise constant.** It branches on `sign(Δu_ij)` and
  `sign(s_i − s_j)`, so it is non-differentiable. Anything that differentiates through
  the step — the gradcheck scripts, any future implicit solve — will be wrong at the
  branch points. `f_ij` should be frozen (evaluated once, held) rather than re-derived
  inside an iteration.
- **`Δt`-dependence blocks operator splitting.** Sub-flows run at different effective
  step sizes, so a `Δt`-dependent update has no consistent meaning across them.

`../warpSPHIntegrators/SPLITTING_PLAN.md` §2.7.1 gives a substep-local rewrite that
removes the pairwise storage entirely and restores exact conservation per substep. It is
not needed for PESPH-A and is not scoped here — noted so the connection is not lost.

### 4.3 Memory

`ap_ij` / `av_ij` / `f_ij` are `O(N · neighbours)` arrays carried across the whole step.
Frontiere flags the alternative — recomputing the pairwise accelerations — as possibly
preferable "on architectures such as GPU accelerated machines with limited memory and
significant FLOPs to burn," which is the Warp target. Worth revisiting independently of
this plan.

---

## 5. Implementation phases

### Phase 0 — groundwork (~1 day)

- Wire `modules/eos/gas.py` into `eos/__init__.py`; it already has every `A ↔ (u,P,c_s)`
  conversion PESPH-A needs and is currently unreachable.
- Fix the `entropies` / `pressures` tags (§2.1).
- Add `CompressibleSPHScheme.PESPH` and the `PESPHVariant` enum.

**Gate.** Existing schemes unchanged; full suite green. `idealGas(..., EOSSource.specificEntropy)`
round-trips `A → (u,P,c_s) → A` to roundoff.

### Phase 1 — pressure by summation (~2-3 days)

- `modules/pesph/pressure.py`: `P̄_i` for both variants (G.1 and the `A^(1/γ)` form).
- Number density `n_i` — reuse the CRK `V_i⁻¹` kernel shape.

**Gate.** On a uniform lattice with uniform `u` (resp. `A`), `P̄_i` matches the analytic
EOS pressure to kernel-normalization accuracy; the two variants agree on a state where
`A = P/ρ^γ` holds exactly.

### Phase 2 — grad-h and the number-density h-solve (~3-4 days, **the risk item**)

- Number-density-driven `evaluateOptimalSupport` variant.
- `∂n_i/∂h_i` (G.8) and `∂P̄_i/∂h_i` (G.9) in a `computeOmega`-shaped kernel.
- Assemble `f_ij` (G.7) — under its new non-colliding name (§1.3).

**Gate.** Finite-difference `∂n/∂h` and `∂P̄/∂h` against the analytic kernels to 1e-8 on
a scattered state. **Then the Kidder isentropic case** (§2.3): energy conservation to the
level Frontiere reports, not merely "no blow-up." Do not start Phase 3 until this passes
— a wrong `f_ij` is the failure mode that hides.

### Phase 3 — momentum and energy kernels (~3-4 days)

- G.4 momentum, `q_acc` viscosity (G.6) under the existing Cullen-Dehnen switch.
- PESPH-E: G.5 total-energy rate; promote `totalEnergies` to `integrated('dEdt')`.
- PESPH-A: entropy source; promote `entropies` to `integrated('dAdt')`, add `dAdt` to
  `CompressibleSystemUpdate`; guard `A > 0`.
- Artificial conductivity via the existing `wp_conductivity.py`.

**Gate.** Linear momentum conserved to machine precision (pairwise antisymmetry). Total
energy conserved to integrator tolerance on an adiabatic case. Acoustic wave reaches the
expected convergence order.

### Phase 4 — scheme assembly and registration (~1-2 days)

- `schemes/pesph.py` with the variant switch; `configurations/pesphConfig.py`;
  `SchemeBundle` + dict round-trip in `schemes/builder.py`.

**Gate.** Round-trips through `compressibleConfigToDict`/`dictToCompressibleConfig`.
Runs end to end on sod with a stock integrator.

### Phase 5 — validation against the paper (~1 week)

Run Frontiere's own battery, already present in `caseUtils/compressible/`: sod, sedov,
noh, kidder, gresho, yeeVortex, kelvinHelmholtz, rayleighTaylor, triplePoint,
hydrostatic, blob. Compare against the published figures, and against this repo's
`monaghan`/`compSPH`/`crkSPH` on the same cases.

**Gate.**
- Kidder: PESPH-A holds entropy to the isentropic solution better than `monaghan` —
  the property the variant exists for.
- Hydrostatic box: PESPH shows the contact-discontinuity behaviour of Frontiere Fig. B.2
  (PESPH degrades *less* than CRKSPH there — a discriminating check that the pressure
  formulation is actually doing its job).
- KH / blob: the mixing behaviour PESPH is known for, visibly better than `monaghan`.
- Sedov / Noh: shock capture no worse than `compSPH`. If PESPH-A is visibly worse here,
  that is §3.3 showing up, not a bug — record it and prefer PESPH-E for those cases.

> Run validation cases with `--video` (vispy) — see the note at the top of
> `BOUNDARY_DENSITY_PLAN.md`. A live window is the only way to see *where* a compressible
> scheme fails, not just *that* it did.

**Estimate: ~3-4 weeks for both variants**, with PESPH-A usable standalone at roughly
2.5 weeks if built first. Phase 2 carries most of the schedule risk.

---

## 6. Sequencing: why this comes before the integrator work

`../warpSPHIntegrators/SPLITTING_PLAN.md` proposes conservative/dissipative operator
splitting, and its Phase A names **an entropy-carrying, variational, velocity-independent
scheme** as the prototype target. PESPH-A is that scheme. But the dependency runs one
way, and this plan should land first, for four reasons:

1. **It has no integrator dependency.** PESPH-A is a plain `f(state) -> update`, so it
   runs with every scheme already registered in `warpSPHIntegrators` — RK4, Verlet,
   DIRK, anything. PESPH-E likewise. Nothing here waits on the split.
2. **Debugging isolation.** If PESPH has bugs, find them under RK4, where the integrator
   is not in question. Bringing up a new discretization and a new integrator
   simultaneously means every discrepancy has two candidate causes. This is the main
   practical argument.
3. **It produces the measured baseline the split is judged against.** Order, energy
   drift, and cost per case under ordinary integrators, before splitting is introduced —
   so step 2's benefit is *measured* rather than asserted. That is the standard the rest
   of `IMPLICIT_ROADMAP.md` holds itself to.
4. **It converts a hypothetical gate into a real one.** `SPLITTING_PLAN.md`'s
   separability certificate (`classify_separable`) currently has only toy problems to be
   right about. PESPH-A gives it a real SPH state — and, usefully, PESPH-E gives it a
   real *negative* case, since the two differ only in the thermodynamic variable yet land
   on opposite sides of the separability test.

So: **PESPH first (this plan), splitting second** (`SPLITTING_PLAN.md` Phases A-B wrap
around the finished scheme). The one thing worth doing early on the integrator side is
`classify_separable`, since PESPH-E vs PESPH-A is its sharpest test case and it costs
little.

---

## 7. Outlook: MFM, and the move from "an SPH code" to a meshless solver

This is an **orthogonal axis** to everything above and to `SPLITTING_PLAN.md`: those are
about the *timestepper* and the *thermodynamic discretization*; this is about the
**spatial discretization substrate**. Recorded here as a direction rather than a plan,
because PESPH is the natural last scheme to add before the question "should this be an
SPH code at all?" becomes worth asking.

### 7.1 Where the field actually is (2026)

Three families, and the 2015-era expectation that one would win has not materialized:

- **Moving mesh (AREPO)** — still the accuracy reference for cosmology (IllustrisTNG,
  Auriga, TNG-Cluster), extended to GR for neutron-star mergers around 2024.
- **Meshless finite mass / volume (MFM / MFV)** — the adoption winner since Hopkins
  (2015). Spread well beyond GIZMO: SWIFT implements both, OpenGadget3 added MFM in 2023.
- **Modern high-order SPH** — did not die, improved substantially. MAGMA2 (Rosswog 2020)
  with matrix-inversion gradients and slope-limited reconstruction; REMIX (2024) for
  mixing; SPHENIX in SWIFT. Cost is roughly 1.5-2.5x bare SPH, while MFM/MFV land
  comparable to or slightly faster than bare SPH.

Adoption specifics worth pinning down, because they are easy to misremember: **FIRE-1
used P-SPH; FIRE-2 and FIRE-3 both use MFM**, not MFV. The reason is not accuracy —
MFM forbids mass flux between cells, so a particle stays a fixed parcel of mass, which is
what keeps metal tracking, stellar-population bookkeeping and particle IDs tractable
under galaxy-formation subgrid physics. MFV's mass fluxes make it behave more like a
moving mesh and break that. **SPHERAL still develops CRKSPH** (releases through 2025.12.0)
and has added FSISPH for fluid-structure interaction — a trajectory pointing at
multi-material and solid coupling, not cosmology.

The intellectual shift worth noting: the 2010-2015 "SPH is broken" framing drove the
migration, but the specific failures (surface tension at contact discontinuities,
suppressed mixing, E0 errors) were largely diagnosed and fixed *within* SPH — pressure-based
formulations like PESPH being a substantial part of that — while MFM/MFV turned out to
have their own pathologies (particle disorder, slope-limiter sensitivity). The 2026
position is that discretization choice matters **less** than it appeared in 2015, and the
real differentiators are the subgrid physics you need, the conservation properties you
cannot give up, and cost.

### 7.2 The structural finding: CRKSPH and MFM share a substrate

This is the part that makes MFM a much shorter step for *this* codebase than a cold
estimate would suggest.

Frontiere et al. derive CRKSPH by assuming a generic interpolant `ψ` subject to

```
Σ_j ψ_j ≡ 1   ⇒   Σ_j ∇ψ_j = 0                                    (Eq. 24)
```

— that is, **a partition of unity**, satisfied in their case by `ψ_j = V_j W^R_ij`. That
is precisely the object MFM is built on. Carrying the derivation through gives

```
m_i DU_i/Dt = ½ Σ_j (F_i + F_j) (V_i ∂_α ψ_j − V_j ∂_α ψ_i)        (Eq. 35)
```

and MFM's update is

```
dU_i/dt = −Σ_j F̃_ij · A_ij ,    A_ij ~ V_i ∇ψ_j(x_i) − V_j ∇ψ_i(x_j)
```

**The effective face vector is the same object.** The difference is entirely in what sits
between: CRKSPH uses the *central average* `(F_i + F_j)/2`, MFM solves a **Riemann
problem** across the face for an upwind flux `F̃_ij`. Stated compactly:

> MFM is CRKSPH's conservative differencing with the flux upgraded from averaged to
> Godunov.

This is a structural correspondence, not an equivalence — the two use different `ψ`
(Hopkins follows Lanson & Vila; CRKSPH uses the RK-corrected kernel) and MFM adds
slope-limited reconstruction of the left/right states to the face. But it means the
face-construction machinery is not new work here.

### 7.3 What the repo already has toward MFM

| MFM ingredient | Status |
|---|---|
| Partition-of-unity interpolant with `Σψ = 1` | ✅ CRK's `ψ_j = V_j W^R_ij` (`modules/crk/`) |
| Effective face `V_i ∂ψ_j − V_j ∂ψ_i` | ✅ implemented as the symmetrized `gradw_ij · V_i V_j` in `modules/crk/accel.py` |
| Slope limiters | ✅ van Leer + eta limiter in `modules/crk/limiter.py` |
| Linearly-corrected gradient estimates | ✅ `M_i^{αβ}` (Frontiere E.6), used by compSPH |
| Neighbour/adjacency, adaptive support, periodicity | ✅ mature |
| **Riemann solver (HLLC in the face frame)** | ❌ **the genuinely missing piece** |
| Reconstruction of left/right states to the face | ⚠️ partial — CRK already reconstructs pair velocities (Eq. 71) with the same limiters |

So the missing core is a Riemann solver plus the boost into and out of the face frame,
not a new discretization substrate.

### 7.4 The architectural argument, and the hard caveat

**The argument for doing it:** MFM turns warpSPH from a code that implements SPH schemes
into a code that implements *meshless conservative discretizations*, of which SPH is one
family. The `ψ`/effective-face layer becomes a substrate that can host both, and the
module structure (`modules/{momentum,pressure,density,internalEnergy,dissipation}`),
currently kernel-sum-centric, would need to grow a flux-based sibling. That is a real
generalization of capability rather than another scheme in the registry.

**The caveat, and it is hard:** MFM is **permanently outside** the geometric-integration
work in `../warpSPHIntegrators/SPLITTING_PLAN.md`. The force comes from solving a Riemann
problem using left/right states that include velocity, so `F_cons` depends on `v` by
construction, there is no potential behind it, and there is no Hamiltonian structure at
all — not separable, not symplectic, not conformally symplectic. Every result in that
plan applies to the SPH family only.

These are therefore **two distinct bets**, and they do not compose:

| Bet | Scheme branch | What pays off |
|---|---|---|
| Geometric integration | PESPH-A / compSPH-entropy | Separable Hamiltonian, symplectic composition, viscous-CFL relief, conformal-symplectic diagnostics — all of `SPLITTING_PLAN.md` |
| Astro fidelity / generality | MFM | State of the art, broad adoption, Riemann-quality shock capture — and the integrator work becomes irrelevant to it |

Worth choosing deliberately rather than drifting into MFM and finding the splitting work
stranded. Both are defensible; doing PESPH first (§6) keeps the option open either way,
since PESPH is a strong baseline under both bets.

### 7.5 The shared prerequisite worth doing regardless

**MAGMA2-style gradient estimation** — integral-approximation / matrix-inversion gradients
rather than plain kernel gradients — improves `monaghan`, `compSPH`, `crkSPH` and PESPH
across the board without adding a scheme, and is *also* a prerequisite for MFM (which
needs accurate gradients for its reconstruction step). It is independent of both bets
above, cheaper than either, and the natural thing to do after PESPH lands regardless of
which direction is chosen next. `modules/crk/` already has the matrix machinery
(`M_i^{αβ}`, gradient renormalization `L_i`), so this is an extension rather than a build.

---

## Status

**Not started.** Plan written 2026-09-15 from Frontiere et al. (2017) Appendix G and
Hopkins (2015) Appendix F2, cross-checked against the existing `compSPH`/`crkSPH`
implementations. The §2 audit and the §2.1 tag bugs are verified against the code as of
this date; the §5 estimates are not yet tested against any implementation work.

§7 is an outlook, not a commitment. Its adoption claims (FIRE-2/3 on MFM, SPHERAL's
continued CRKSPH + FSISPH development, MFM in SWIFT and OpenGadget3) were verified
2026-09-15; the §7.1 synthesis about the SPH/MFM gap narrowing is a reading of the
literature rather than a citable consensus. The §7.2 correspondence between Frontiere
Eq. (35) and the MFM face vector is checked against the paper directly (Eqs. 24, 35) and
against `modules/crk/accel.py`.

### Reading

- Hopkins (2015), MNRAS 450:53 — MFM/MFV derivation, `../warpSPHIntegrators/literature/stv195.pdf`
- Rosswog (2020), MNRAS 498:4230 — MAGMA2, the modern high-order SPH reference
- Rosswog (2026), *SPH methods in the modelling of compact objects*, arXiv:2607.14828 —
  current review, the place to check §7.1's framing
- Groth et al. (2023), MNRAS 526:616 — MFM in OpenGadget3, a worked port into an
  existing SPH code, i.e. the closest thing to a template for §7
