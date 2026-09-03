# warpSPH — Weakly-Compressible Particle-Shifting Near Free Surfaces

Working document for the **δ⁺-SPH particle shift on the weakly-compressible
`deltaSPH` scheme** (`schemes/deltaSPH.py` /
`systems/weaklyCompressible.py::finalize`, gated by
`schemeConfig.shiftProperties.active`). This is **not** the DFSPH VD+PS shift —
that is `DFSPH_IMPROVEMENT_PLAN.md`'s. The two share `modules/shifting/` code
but are applied and graded separately.

Spun out of the sloshing-tank verification (`examples/sloshingTank/PLAN.md`),
where `deltaSPH` reproduced the sloshing kinematics but **diverged at the first
wave impact** (`t ≈ 3.5 s`, unchanged by 3× sound speed + 3× artificial
viscosity) — a free-surface particle-distribution / tensile instability that a
working surface shift would relieve.

---

## The problem

The δ⁺-SPH shift (`modules/shifting/delta.py`, Sun et al. 2017 scaling:
`δx = −CFL·Ma·2·h²·∇C`, `C` a number-density concentration) **is not
volume-preserving**. Applied every step near a free surface it produces a net
**outward drift of the surface particles**, so the fluid footprint grows
without bound while mass is conserved — an unphysical volume increase.

Why the drift: on the air side the kernel sum is truncated, so `∇C` at the
surface points *into the fluid* (toward the denser interior) and `−∇C` pushes
surface particles *outward* into the void. There is nothing at a free surface
to oppose it — no wall, and the deltaSPH pressure there is ≈ 0. Iterated, it
is a ratchet.

### What the code does about it today (`modules/shifting/wrapper.solveShifting`)

When `surfaceDetectionConfig.active`, the shift is suppressed near the surface,
by `shiftProperties.projectionScheme`:

- **`dot`** (default): surface particles (`fsm > 0.5`) keep only the
  *tangential* shift, scaled by `surfaceScaling` (**default 0.1**); any
  particle with `lMin < 0.4` (the min renormalisation-matrix eigenvalue, a
  kernel-deficiency proxy) has its shift **zeroed outright**.
- **`mat`**: `(I − n nᵀ)` projection scaled by `lMin²`, then surface particles
  zeroed anyway.
- fallback: surface + near-surface shift zeroed.

So the current answer to "shifting inflates the free surface" is **"don't shift
the free surface"**. That tradeoff was judged *somewhat worth it* — it keeps
the volume roughly bounded — but it leaves the surface region **under-regularised**:
particle clumping and voids form there with no shift to relieve them, which is
the mechanism behind the sloshing-tank impact divergence and the classic
`rotatingSquarePatch` corner erosion.

### Why the suppression is not enough

1. `dot`/`mat` remove the *instantaneous* normal component, not the
   *accumulated* outward displacement — drift still ratchets through
   normal-estimate error.
2. The surface normal is noisy exactly where it matters — **corners**.
3. `surfaceScaling = 0.1` on the tangential part still leaks outward motion via
   that normal-estimate error.
4. The hard `lMin < 0.4` cutoff is a sharp on/off boundary that itself seeds
   disorder one layer into the bulk.

**Goal: get the regularisation benefit at the free surface *without* the
volume growth, so the "just switch it off there" tradeoff no longer has to be
made.**

---

## Target case — `rotatingSquarePatch` (`squarePatch`)

`src/warpSPH/cases/rotatingSquarePatch.py`, run as `--scheme deltaSPH`. A 2×2
fluid square, `omega = 4`, free surface on all four sides **and corners**, no
wall, no gravity. The cleanest controlled probe:

- **Area drift is directly measurable** and, for the square, the physical
  answer is *area-conserving rigid rotation* (the surface deforms into four
  growing arms — real physics, [BK] §5 / the case docstring — but the enclosed
  area is fixed).
- **The corners are where the surface normal is worst**, so it stresses the
  exact failure mode.
- `--shape circle` is the **null experiment**: a circle in rigid rotation *is*
  an equilibrium, so any area drift there is pure shift artifact.
- `--shape triangleIsosceles` / `--shape star5` are sharper corner versions.

The case currently reports only `weaklyCompressibleDiagnostics` (KE, v_max, ρ
bounds) — **it has no area/volume metric**, so step 1 is to add one.

---

## Metrics to add to `rotatingSquarePatch.diagnostics`

| metric | definition | healthy behaviour |
|---|---|---|
| `sphVolume` | `Σ_i m_i / ρ_i` | flat (this is the SPH volume; drifts only with ρ error) |
| `hullArea` | convex-hull (or α-shape) area of the fluid point cloud | flat modulo the physical arms; **the "inflation" number** |
| `rmsRadius` | `sqrt(Σ_i m_i ‖x_i − x_cm‖² / Σ_i m_i)` | grows if the patch spreads |
| `surfaceFraction` | `surfaceIndicators.sum() / N_fluid` | flat; grows if the surface frays |
| `cornerRetention` | fluid extent along the initial diagonals ÷ initial | ~1 until the arms form; < 1 = corner erosion |

Plot each vs `t`. A working surface shift keeps `sphVolume`, `hullArea` and
`rmsRadius` flat (arms aside) with the shift **fully active at the surface**.

---

## Actionable list

### 1. Baseline & metrics — "how bad is it, does the switch-off help"

- Add the metrics above to `rotatingSquarePatch` (`diagnostics` + a plot
  panel). New probe `scripts/probe_squarePatchAreaConservation.py`.
- Matrix: `{shift off, shift on + surface-zeroed (today's default), shift on +
  surface NOT suppressed}` × `{box, circle}` × `nx ∈ {96, 192, 288}`, to
  `t = 1` (~0.6 rev at `omega = 4`).
- Deliverable: volume-growth rate (`d hullArea/dt`, `d sphVolume/dt`) for each
  cell; confirm the surface-zeroing actually bounds it and quantify the
  residual drift it still leaks (point 3 above). `circle` isolates the pure
  artifact.

### 2. A volume-preserving shift (the root fix)

The drift exists because `−D∇C` is not divergence-free. Options, cheapest
first:

- **2a. Background pressure** (≈ 2 lines). Add a small uniform `p_b` to the
  deltaSPH pressure (`p ← p + p_b`, or Antuono's form). A positive `p_b` at the
  free surface pushes particles *together*, directly opposing the shift's
  outward push, and needs no surface special-casing. Sweep `p_b` vs the area
  drift on `squarePatch`. Known cost: a spurious surface-tension-like force
  that rounds corners — the arms / `cornerRetention` are the tolerance test.
- **2b. Divergence-free projection of the shift field.** Before applying,
  project `δx` onto `∇·(δx) = 0` (a small least-squares / Poisson solve on the
  fluid). Volume-conserving by construction, no surface heuristic. Cost: one
  extra solve per shift iteration — reuse `modules/shifting/`'s Krylov drivers.
- **2c. Transport-velocity / consistent-δ⁺ formulation** (Adami et al. 2013,
  `literature/`; Sun et al. 2019 free-surface correction). Builds the
  regularisation into the momentum equation with a constant background
  pressure, inherently conservative. Larger change; do only if 2a/2b fall
  short.

### 3. Better surface treatment than hard-zero

- Replace the `lMin < 0.4` step + `surfaceScaling = 0.1` with a **smooth taper**
  in `lMin` (or in a normalised kernel-sum deficiency) — expose the threshold
  and the taper width via `ShiftProperties`.
- Apply the normal projection **cumulatively**: track each particle's
  accumulated normal displacement since it became a surface particle and damp
  against it, so drift cannot ratchet.

### 4. Re-enable the full surface shift + verify the payoff

Once 2 (or 2a) bounds the volume: set `surfaceScaling → 1.0`, drop the `lMin`
hard-zero, and check
- `squarePatch --scheme deltaSPH`: corners hold, `hullArea`/`sphVolume` flat;
- **`sloshingTank --scheme wcsph` survives past the first wave impact**
  (currently NaNs at `t ≈ 3.5 s`) — the downstream reason this plan exists.

---

## Success criteria

- `squarePatch --scheme deltaSPH`, shift fully active at the surface:
  `sphVolume` and `hullArea` conserved to **< 1 %** over `t = 1`; corners sharp
  until the physical arms form.
- `squarePatch --shape circle`: area drift **< 0.5 %** (the artifact is gone).
- Transfer: `sloshingTank --scheme wcsph` runs past the first impact without
  divergence.
- No regression on the periodic weakly-compressible cases (`tgv`,
  `taylorGreenVortex`, `kolmogorov`) — shifting there is already fine and must
  stay fine.

## Files

- `src/warpSPH/modules/shifting/wrapper.py` — `solveShifting`, the surface
  projection block.
- `src/warpSPH/modules/shifting/delta.py` — `computeDeltaShift`.
- `src/warpSPH/systems/weaklyCompressible.py` — `finalize`, where the shift is
  applied (and its `correctdrhodt` / `correctdvdt` consistency terms).
- `src/warpSPH/configurations/moduleConfigurations/shifting.py` —
  `ShiftProperties` (`surfaceScaling`, `threshold`, `projectionScheme`; add the
  `lMin` threshold + taper, and a `backgroundPressure`).
- `src/warpSPH/cases/rotatingSquarePatch.py` — the metrics.
- new `scripts/probe_squarePatchAreaConservation.py`.

## Notes / prior art

- `DFSPH_FINDINGS.md` ("The published CFL is calibrated against a metric a free
  surface silences") documents the same *expanding-particle* asymmetry for the
  DFSPH stopping metric — related framing, different scheme.
- `rotatingSquarePatch --scheme divergenceFree` is separately broken
  (`DFSPH_IMPROVEMENT_PLAN.md`: corner density → 0.5, [BK] §5 method
  limitation). Keep the deltaSPH and divergenceFree `squarePatch` results
  distinct — this plan is deltaSPH only.
- The δ⁺-SPH scaling and the Michel 2022 shifting-velocity alternative are
  described in `modules/shifting/delta.py`'s docstring.
