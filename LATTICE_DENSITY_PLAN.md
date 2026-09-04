# warpSPH — Carrying the lattice-density correction on the kernel, not on the mass

Working document. §3's kernel-side correction is landed and verified. §4 step 3
turned up something bigger than planned: `massRatio` (the other half of the
offset, thought to be an unavoidable sampler-fit fuzziness) was one line in
`sample/regular.py`, now fixed at the source — see its module docstring and
`DFSPH_FINDINGS.md` Part 54. `calibrateRestDensityMasses` is retired to
`calibrateRestDensity`, which no longer touches mass. §5's "Wendland2 min L"
sign-check and the closed-form evaluator (`warpSPHCore.util.latticeDensity`)
predate both and are unaffected.

---

## 1. What is actually wrong

`C_d` normalises the kernel's **integral**: `int W dV = 1`. An SPH density sum
needs the kernel's **lattice sum** to be 1, and it is not. For a defect-free
cubic lattice of spacing `s` with `h = n_h s`,

```
L(n_h) = C_d / n_h^d * sum_{n in Z^d} k(|n| / n_h)  !=  1
```

`L` is a pure function of `(kernel, dim, n_h)` — no particles, no resolution,
no case. It is the kernel's Fourier transform sampled on the reciprocal lattice
(Poisson summation), i.e. aliasing, and for the Wendland family every term of
that sum is positive, so **`L > 1` strictly at every `h`**: widening the support
buys the error down as `n_h^-(d+2k+1)` but never reaches zero, and a
Newton/bisection search on `h` for `L = 1` cannot converge.
`warpSPHCore.util.latticeDensityIsStrictlyAbove1` is the predicate;
`warpSPHCore/util/latticeDensity.py`'s module docstring carries the derivation.

Measured on `sloshingTank` (Wendland4, `n_h ~ 4.0`), with the new decomposition
printed by `calibrateRestDensityMasses --verbose`:

| nx  | `latticeFactor` L | `massRatio` | product | measured | model residual |
|-----|-------------------|-------------|---------|----------|----------------|
| 60  | 1.000411          | 1.002692    | 1.003104| 1.003103 | +2.4e-07 |
| 100 | 1.000392          | 1.014151    | 1.014548| 1.014547 | +1.6e-06 |
| 200 | 1.000415          | 1.000389    | 1.000804| 1.000806 | -1.6e-06 |

`L` is flat at ~1.0004 — it is the same physics at every resolution — while
`massRatio` is the nx lottery. **They are two different bugs** and the model
predicts their product to ~1e-6.

## 2. Why the current fix is a workaround

`calibrateRestDensityMasses` measures the density once and scales `m` by
`rho0 / rho_measured`, which absorbs **both** terms. For `massRatio` that is
correct and exact: the particle really was carrying the wrong mass for the cell
it occupies. For `L` it is not:

* `L` is a property of the kernel, not of the mass. Folding it into `m` means
  every operator that touches mass carries it too — momentum, the continuity
  equation, viscosity, every force — which is arithmetically identical to
  having rescaled `C_d`, except that it is invisible at the call site.
* The fluid's **total mass** changes by `1/L` (~0.04 % here, ~3.8 % at
  `n_h = 2`). For a closed domain that is a real, if small, change to the
  physical system, and it is not what the user asked for when they set `rho0`.
* It needs a sampled state to measure, so it only exists at initialisation. Any
  later change to `h` (adaptive support, `sizing`) silently invalidates it.
* It cannot distinguish "this sampling is bad" from "this kernel is like this",
  which is exactly the distinction §1's table shows matters.

## 3. Target design — DECIDED

Carry `L` as an explicit **kernel-side** correction: a flag plus a coefficient on
`kernelState`, resolved once on the host and cached.

```python
@wp.struct
class kernelState:
    kernelFunction: wp.int32
    supportMode: wp.uint32
    ...
    calibrateNormalization: wp.bool     # False (== warp's zero default) is off
    normalizationCoefficient: scalar_t  # 1/L; only read when the flag is set
```

### 3.1 Flag + coefficient, not a bare factor or a delta

Warp zero-initialises struct fields (verified: `@wp.struct class S: f: wp.float32`
-> `S().f == float32(0)`), and **five** places construct a `kernelState()` and set
only the fields they know about: `autograd/arg_extract.py:376`,
`util/stateBundle.py:113`, `crk/crk_density.py:56`,
`coreOperations/_jvpCommon.py:67`, and `warpSPH`'s
`modules/shifting/wp_implicitShifting.py:70`. A bare `normalizationCoefficient`
would therefore default to `0.0` and **silently zero the kernel** on any path
that was not converted.

Storing a *delta* (`1/L - 1`, zero-default == off) fixes that but trades a loud
failure for a silent one: "calibration requested, correction never computed"
becomes indistinguishable from "calibration off". The flag gets both, because
`wp.bool` also zero-defaults to `False`:

* unconverted site -> flag `False` -> correction off -> correct, not catastrophic;
* flag `True` with a zero/absent coefficient is an **invalid combination and
  raises**, on the host, where an exception is possible.

Two checks, at the two points where the information first exists:

* `OperationProperties.__post_init__` — frozen dataclasses still run it and can
  raise (they only cannot *assign*). `calibrateNormalization and not (n_h > 0)`
  is an error there, covering all seven backends at once.
* `extractStateInfo`, after the cached lookup — assert the resolved coefficient
  is finite and `> 0`, so a bad `(kernel, n_h, dim)` surfaces as an exception
  rather than a NaN field.

### 3.2 Apply at the public boundary, not at the 13 `eval_C_d` sites

There are 13 arithmetic `eval_C_d` sites across six files in `kernels/`, and the
`_`-suffixed leaves that hold them take a raw `kernel: wp.int32` rather than the
struct. Threading a scalar through all of them is unnecessary: **every one of
those functions is linear in `C_d`**, so scaling the *return value* of the public
entry point is identical arithmetic and touches one line each.

Verified preconditions for that being safe:

* every call to a `_`-suffixed leaf lives inside `kernels/` — nothing outside the
  directory can reach an uncorrected value;
* every external consumer (`coreOperations/`, `crk/`, `renorm.py`, `warpSPH`)
  calls only the public `kernelProperties`-taking functions;
* `sphKernelC_d` is exported but has zero consumers, so it is not an escape hatch.

The entry points to scale: `sphKernel`, `sphKernel_ij`, `sphKernelGradient`,
`sphKernelGradient_ij`, `sphKernelLaplacian`, `sphKernelHessian`,
`sphKernelDkDh`, `sphGradientDkDh`, `sphKernelDerivative`, and the six
`kernelJVP.py` entry points. A single
`resolveNormalization(kernelProperties) -> scalar_t` (returning `1.0` when the
flag is clear) keeps the branch in one place.

**Not** scaled: `sphKernelScale`, `sphKernel_xi`, `sphKernelN_H`,
`sphKernelC_d`. These are packing / support-radius properties, not kernel values.

### 3.3 The other three `kernelState` builders

* `coreOperations/_jvpCommon.py:67` `buildKernelState` — **must** carry both
  fields, and its signature has to grow to accept them. If the JVP's kernel state
  is uncorrected while the forward operator's is corrected, the tangent stops
  matching the primal and `scripts/gradcheck_*.py` fails. This function already
  carries a comment about exactly this bug class (gradientMode/laplacianMode
  defaulting to `0`, which matches neither enum) — second visit to the same trap.
* `crk/crk_density.py:56` — **stays off, deliberately, and the comment must say
  so.** CRK already enforces the reproducing conditions and reconstructs a
  constant field exactly; a lattice-normalisation correction on top of it would
  double-correct. The existing comment there says the hardcoded fields are
  "pre-existing behavior ... not touched by this migration", which would read as
  an oversight rather than a decision.
* `warpSPH`'s `wp_implicitShifting.py:70` — inherits `False`, no change needed.

### 3.4 Plumbing

`OperationProperties` is rebuilt *inside* each backend from flat arguments
(`wp_density.py:153` takes `(mode, kernel, operationMode, adjacency)` and
constructs the dataclass itself), so two new fields would otherwise mean seven
signature changes. Fix it permanently instead: **pass `operationProperties`
through to the backends** rather than the unpacked fields. Safe — the only
callers of the seven `_*_stateBackend` functions are `operations.py` and the
`coreOperations/__init__.py` re-exports; nothing external.

`extractStateInfo` needs no signature change; it already receives the whole
`OperationProperties`. Both `kernelState` build paths get the new `cfg` keys:
`autograd/arg_extract.py:376` and `util/stateBundle.py:141`.

### 3.5 `n_h` becomes a config member

`config.targetNeighbors` can be inverted with `nH_to_n_h`, but that recovers
`h/dx`, not the achieved `h/s` — measured on `sloshingTank` nx=100, nominal 4.0
against achieved 4.0282, i.e. `L` 1.000415 vs 1.000392, 5.5 % of the correction
and 2.3e-5 in absolute density. Two orders below the effect, so nominal is the
right call: it needs no particles and no initialisation ordering.

Store `n_h` directly instead of re-deriving it. It also fixes a live wart:
`simulationConfig.py:50` is
`targetNeighbors: int = field(default_factory=lambda: n_h_to_nH(4, 2))` — **dim
hardcoded to 2 in the dataclass default**, with only `buildSimulationConfig`'s
line 99 doing `n_h_to_nH(4, dim)`. An `n_h: float = 4.0` member with
`targetNeighbors` derived from it puts the 4 in one documented place.

### 3.6 Caching

`latticeDensityFactor` is already `functools.lru_cache`d on
`(kernel, n_h, dim, method)`, and the shell route costs `O(n_h^2)` scalar kernel
evaluations. Resolve it in `extractStateInfo` after the flag check; an `lru_cache`
hit is ~100 ns against everything else that function does per call.

### 3.7 The adaptive-support restriction

The correction is a function of `n_h = h/s`, which is **per particle** once
`adaptiveSupport` or `sizing` is active. Scalar, uniform-resolution only: one
factor from the configured `n_h`, and the flag stays clear when the support field
is non-uniform. A per-particle version would need a local spacing estimate, which
is exactly the quantity that is unreliable in a disordered flow — and the
correction is only *meaningful* on a lattice. (Making the neighbour sum exactly 1
per particle is Shepard/CRK territory, already in `crk/` and `renorm.py`: a
different correction with different conservation properties.)

### 3.8 What must not change

* Conservation. A uniform scalar multiplying `W`, `grad W` and `lap W` alike
  leaves antisymmetry of `grad W_ij` intact, so momentum conservation is
  untouched. Worth a test, not an assertion.
* Gradients. `scripts/gradcheck_*.py` must pass unchanged; the coefficient is a
  constant w.r.t. every differentiated input, so the adjoint is a plain scale.
* `deltaSPH` and the other density-evolution schemes never form a summation
  density and so never see `L` (step-0 density is 1.000000 at every `nx` — this
  is why the WCSPH SPHERIC validation was always clean). The correction is a
  no-op for them: nothing reads it.

### 3.9 How far the wiring actually reaches — READ THIS BEFORE §4

`OperationProperties` is constructed at **132 sites across `warpSPH`**. They all
keep working untouched, because both new fields default to off — but it means
"turn the correction on" is not one switch today. Only
`modules/density/density.py::computeDensities` reads
`config.n_h` / `config.calibrateNormalization`, which is deliberate and enough
for the initialisation problem this plan exists to solve: the summation density
is the operator the offset is *defined* on, and it is what
`calibrateRestDensityMasses` measures.

Every other operator (gradient, divergence, laplacian, curl, interpolate) still
runs uncorrected even with `config.calibrateNormalization=True`. That is a
**deliberate half-state, and it is not obviously the wrong one**: the density
sum is where `L` is a defect, whereas for a gradient the same rescale is just a
uniform change to the operator with no reference value to match. Deciding
whether the momentum side should carry it too is exactly what §4 step 3 is for
— do not extend the wiring before that comparison, or there will be nothing to
compare against.

If the answer turns out to be "all of them", do not edit 132 call sites: add a
`propsFromConfig(config, ...)` helper and migrate to it, which is worth doing on
its own merits.

### 3.10 Implementation order

1. `kernelState` fields + `resolveNormalization`; apply at the public boundary.
2. `OperationProperties` fields + `__post_init__` validation.
3. `operations.py` / the seven backends: pass `operationProperties` through.
4. `arg_extract` + `stateBundle`: resolve, validate, set both fields.
5. `_jvpCommon.buildKernelState`: carry them. `crk_density`: document staying off.
6. `warpSPH`: `n_h` on `SimulationConfig`, `targetNeighbors` derived,
   `calibrateNormalization` plumbed to the operator calls.
7. Tests: a lattice that reads `rho0` with the flag on; gradcheck; `test_physics`.

## 4. Migration

The two corrections are independent and should be separated, not swapped
wholesale:

1. ~~Land the kernel-side correction, off by default.~~ **Landed and
   verified.** With `calibrateNormalization=True` a defect-free lattice
   measures `1.0` to **4.4e-16** across Wendland2/4 and CubicSpline, dim 2 and
   3, and `n_h` of 3.0 / 4.0 / 4.7 — including the CubicSpline `n_h=4` case
   where `L < 1` and the correction goes the other way. With the flag off,
   every path is bit-unchanged (warpSPHCore `tests/operations`: 419 passed;
   `test_gradcheck_scripts.py`: 15 passed; `test_physics.py`: passed).
2. Add `calibrateRestDensityMasses(..., latticeTerm='mass' | 'kernel')`. Under
   `'kernel'` it corrects only `massRatio` in the mass — the part that really
   is a mass error — and hands `1/L` to the kernel. (Today both routes are
   live at once when the flag is set: the kernel removes `L` and the mass
   calibration then measures and removes whatever is left, which is correct
   but redundant — see the result of step 3 below for why it stays needed.)
3. Compare on `sloshingTank` at nx = 60 / 100 / 200 and on
   `hydrostaticColumn`: step-0 `|v|max`, `pairedFraction`, `voidFraction`, the
   Sensor-1 pressure trace. The expectation is *no visible change*, because the
   two routes are arithmetically equivalent for the density sum — the win is
   that the fluid mass is no longer perturbed and the correction survives an
   `h` change. **If something does move, that difference is the momentum-side
   term the mass route was silently applying**, and it needs explaining before
   the default flips.

   **Partially done, ahead of schedule — and it changes the framing.** Before
   building the `latticeTerm=` split, ran the more basic question: does
   `calibrateNormalization` alone (mass left untouched) already fix
   `sloshingTank`? All four combinations of
   `{calibrateRestDensity, calibrateNormalization}`, 20 steps, nx = 60/100/200:

   | config | nx=60 step0 maxV | nx=100 step0 maxV | nx=100 step20 paired |
   |---|---:|---:|---:|
   | raw (neither) | 0.198 | 1.703 | 0.243 |
   | kernel-only (no mass touch) | 0.150 | 1.655 | 0.240 |
   | mass-only (shipped today) | 0.0098 | 0.0098 | 0.000 |
   | both | 0.0098 | 0.0098 | 0.000 |

   `kernel-only` moves the measured density by exactly `1/L` (nx=100: 1.014546
   -> 1.014125, matching `L(4.0282) = 1.000392` to 5 digits — the correction is
   doing precisely what §3 designed it to do). But it leaves the startup
   impulse and the nx=100 delamination essentially untouched: `massRatio` (up
   to 1.4 % here) dwarfs `latticeFactor` (≤0.04 %) for this case, and the
   kernel-side fix only ever removes the latter. **So the mass route did not
   become optional by calibrating the kernel** — not for `sloshingTank`, whose
   problem was always dominated by the sampler's block-fit error rather than
   the lattice quadrature term (Part 53's own numbers already said this; the
   eval just confirms the kernel fix can't reach it).

   **Superseded within the hour: `massRatio` itself is now fixed, at the
   sampler, and step 2 above has landed.** What this section called "out of
   scope" — computing mass from the achieved spacing at sample time instead of
   measuring and rescaling after the fact — turned out to be one line
   (`sample/regular.py:62`, `area = dx ** domain.dim` using the pre-snap
   nominal spacing instead of the post-snap per-axis one particles are
   actually placed at). Root-caused to two stacking mechanisms — a structural
   non-integer domain/spacing ratio on every axis but one, and a pure
   floating-point `ceil()`-boundary flip on that one (measured: a `1e-6`-
   relative residual added an entire extra row of cells) — see
   `sample/regular.py`'s module docstring and `DFSPH_FINDINGS.md` Part 54 for
   the full derivation. Fixed, `massRatio` is `1 +/- ~1e-6` (float32 noise) at
   every `nx` tested, and `calibrateRestDensityMasses` is retired to
   `calibrateRestDensity` (`cases/weaklyCompressible.py`): it no longer scales
   mass at all in the expected path, only flips `calibrateNormalization` on and
   raises if `massRatio` deviates from 1 by more than a tolerance — which, with
   the sampler fixed, means a corrupted IC rather than expected noise. The
   `raw (neither)` row above is now **also** `0.0098` / `0.0098` / `0.000` at
   every nx — indistinguishable from the other three.
4. Flip the default only after (3). **Moot** — see the supersession above:
   there is no longer a mass-scaling default to flip. `calibrateRestDensity`
   sets `calibrateNormalization` on whenever it is called, gated by the
   existing "auto" logic on `sloshingTank`/`columnCollapse` (on iff the scheme
   is incompressible) — unchanged.

## 5. Gotchas found while implementing

* **`eval_k` from Python only accepts a raw float under float32.** Under a
  `configure(precision='float64')` run, `eval_k(0.25, 2, 0)` raises "no overload
  found" -- a `@wp.func` called from Python scope matches `scalar_t` exactly.
  `util/latticeDensity.py` casts through `scalar_t` (read from the module, not
  bound at import, so it follows a late `configure`). Anything else calling a
  `@wp.func` from the host has the same trap.
* **`L` is therefore evaluated in the working precision**, so it agrees with a
  double-precision reference only to ~5e-7 under float32. Fine for the
  coefficient (`L - 1 ~ 1e-3`, so ~1e-10 of absolute error in `1/L`), but a test
  asserting the shell sum is "exact" has to scale its tolerance to `scalar_t`.

## 6. Open questions

* ~~Achieved `h/s` or nominal `n_h`?~~ Settled in §3.5: nominal, stored on the
  config. The diagnostic in `latticeDensityDecomposition` keeps using achieved,
  since its job is to explain a measurement rather than to correct one.
* Boundary particles sampled at a different effective spacing (the five-layer
  Akinci band) have their own `n_h`, and a single scalar cannot serve both.
  Probably fine — the correction matters for the fluid interior — but measure it.
* ~~Kernel or volume (`correction.useVolume`)?~~ Settled: kernel. The volume
  route would miss every operator that bypasses the volume path, and §3.2 makes
  the kernel route a one-line change per entry point rather than the 13-site
  edit that made `useVolume` look attractive.
