# Frontend host-readback removal (warpSPH)

**Date:** 2026-08-17. Follow-on to warpSPHCore's
`docs/regression/real_workload_bottleneck_audit.md`, which fixed the two
core-side sync sites (1.08-1.20x) and identified 22.6 of the remaining 25.5
readbacks/step as frontend-only, all the same two patterns: `if someTensor:`
where a Python flag would do, and a masked `index_put_` in
`[deltaSPH - 17] - enforce updates`. This note fixes those.

**Tools (both in `../warpSPHCore/scripts/`):** `count_host_syncs.py` (per-site
census) and `bench_real_workload.py` (per-step wall clock). Workload: the
`dambreak` case configured as fully-periodic Kolmogorov flow, deltaSPH, RK2,
adaptive dt -- the same harness the core audit used.

## Sites fixed

| site | was | fix |
|---|---|---|
| `math/__init__.py:15` `getPeriodicPositions` | `if periodicity[i]:` per axis in a Python list comprehension (8/step) | one `torch.where(periodicity, wrapped, x)` over the whole axis vector, `periodicity` kept on device |
| `modules/mdbc/density2025.py:34` `computeMdbcDensity` | `if not torch.any(kinds == 1): return` (2/step) | `stateHasBoundaryParticles()` (new `modules/mdbc/_util.py`) |
| `modules/mdbc/velocity.py:177` `computeBoundaryVelocities` | same (2/step) | same |
| `modules/mdbc/wp_nopenshift.py:514` `computeMdbcNoPenShift` | same (2/step) | same |
| `modules/shifting/delta.py:57` `computeDeltaShift` | `if c_max < 1e-6: c_max = 0.1` (1/step) | `torch.where`, NaN-safe (NaN compares False either way) |
| `systems/weaklyCompressible.py:130` `finalize` | `max_velocity_magnitude = ...item()...` (1/step) | deleted -- fed only a commented-out `print`, dead |
| `systems/weaklyCompressible.py:230` `finalize` | `if torch.any(kinds != 0): densities[mask] = midRho[mask]` (1/step) | `torch.where`, no guard |
| `schemes/weaklyCompressible/deltaSPH.py`, `schemes/dfsph.py`, `"[deltaSPH - 17] - enforce updates"` | 3 unconditional masked `index_put_` zeroing `dxdt`/`dvdt`/`drhodt` for non-fluid particles (6/step) | `torch.where`, no `index_put_` |

**The `stateHasBoundaryParticles` fix, in detail.** `kinds` is a run-constant
field -- assigned once at particle generation
(`initializers/weaklyCompressible.py`) and never mutated afterward -- so
"does this run have any boundary particles" only needs a device readback
once per run, not once per step, exactly like `getPeriodicPositions`'s
domain periodicity. Unlike periodicity, though, there is no existing
Python-level source of truth for it: `currentState` itself is rebuilt fresh
every RK stage (`warpSPHIntegrators.fields._state_initialize` clones every
`constant`/`integrated` field into a new tensor object each call), so
caching on `currentState.kinds` doesn't survive past one stage. `config` is
the object that is actually stable for the whole run -- the runner mutates
it in place (`ctx.config.dt = ...`) rather than replacing it -- so the flag
is cached there instead, in `modules/mdbc/_util.py`:

```python
def stateHasBoundaryParticles(currentState, config) -> bool:
    cached = getattr(config, '_hasBoundaryParticles', None)
    if cached is None:
        cached = bool(torch.any(currentState.kinds == 1).item())
        config._hasBoundaryParticles = cached
    return cached
```

**A wrong first attempt, caught by testing rather than trusted.** The first
version of this fix checked `any(region.type == RegionType.Boundary for
region in config.regions)` -- a real Python-level, zero-sync signal -- on
the reasoning that `kind == 1` particles only ever come from
`RegionType.Boundary` regions. Two things broke it, both found by running
real cases rather than trusting the code-reading: the Kolmogorov preset
*does* carry a `RegionType.Boundary` region, just with zero particles in it
(region-presence != particle-presence), and the `tgv` case's config has no
`.regions` attribute at all (`AttributeError`, previously masked by the
`torch.any` guard short-circuiting before that line was ever reached). The
caching approach above doesn't have either problem and was verified against
both.

## Verification

- **Values and gradients, edge cases, before trusting each rewrite:**
  `getPeriodicPositions` across periodicity patterns of dim 2 and 3 and
  motion from 0.01x to 3x the box length (max diff `0.0`, incl. `torch.where`
  vs `if` gradient parity); the `c_max` branch including the NaN case (no
  finite velocities); the three `index_put_` -> `torch.where` rewrites
  across all-fluid / all-boundary / mixed / empty masks, values and
  gradients; the density masked-assign the same way.
- **`stateHasBoundaryParticles` against the cases that broke the first
  attempt:** ran `dambreak` (Kolmogorov preset, no boundary particles),
  `dambreak` (default preset, has walls), `tgv`, and `sod2d` for several
  steps each; the cached flag matched `torch.any(kinds == 1)` exactly every
  step in every case, with no crash (including the two that broke version
  one).
- **End-to-end A/B:** stashed every change, ran 20 steps of both the
  boundary-free (Kolmogorov) and boundary-containing (default `dambreak`)
  case with a fixed setup, diffed final `positions`/`velocities`/`densities`
  against the same run with the fix applied -- max abs diff `0.0` in both.
- **`pytest tests/`:** 104 passed, 1 skipped, exit 0 (matches the documented
  baseline).

## Measured effect

Census, this repo's sites only, 19,044 particles:

| | before | after |
|---|---:|---:|
| readbacks/step | 25.5 | **8.6** |

(-66%; the reduction lands almost entirely on the 8 sites above -- see
[Where the remaining readbacks are](#where-the-remaining-readbacks-are-nx128-filterwarpsph)
for what's left.)

Per-step wall clock, 60 measured steps, medians, forcing on (same harness,
same scenario the core audit used):

| particles | before (ms) | after (ms) | speedup |
|----------:|------------:|-----------:|--------:|
| 2,401     | 9.50        | 7.58       | **1.25x** |
| 19,044    | 10.75       | 8.70       | **1.24x** |
| 57,121    | 11.67       | 9.03       | **1.29x** |
| 155,236   | 17.11       | 13.65      | **1.25x** |

GPU share rose correspondingly (e.g. 40.7% -> 52.3% at 57k), confirming the
host was the bottleneck being removed, not something incidental. This beats
the two core-side fixes' own 1.08-1.20x, on eight one-line-to-a-dozen-line
changes.

## Where the remaining readbacks are (nx=128, `--filter warpSPH`)

25.5 -> 8.6/step. Nothing left is the `if someTensor:`/masked-`index_put_`
pattern this pass targeted; what's left is either core code (out of scope
here), genuine control flow, or one-time setup cost that amortizes toward
zero over a long run:

| per step | site | why it's still here |
|---------:|------|----------------------|
| 2.88 | `warpSPHCore/radiusSearch/verlet/build.py:74` | **core**, the Verlet rebuild decision -- already identified as the one genuinely irreducible sync in the core audit; not touched here |
| 1.00 | `warpSPH/runner/runner.py:306` `_run` | `if torch.any(torch.isnan(velocities)): break` -- **genuinely load-bearing**: a real Python `break` decision, not a maybe-skippable masked write. Left as-is per this pass's brief |
| 0.06 | `warpSPH/modules/timestep/weaklyCompressible.py:130` | `if uMax / c0 > 0.1: print(warning)` -- a diagnostic-only guard, out of scope (not in the original site list, negligible at 0.06/step) |
| 0.06 | `warpSPH/modules/mdbc/_util.py:22` `stateHasBoundaryParticles` | this pass's own fix -- one sync total per run, reported here because it still fires once; the per-step figure is that one hit diluted over the census's 17-step window |
| 0.35-0.47 each | `warpSPH/sample/regular.py`, `warpSPHCore/radiusSearch/compactHash/*`, `warpSPH/regions/contour.py`, `warpSPH/modules/noise/sampleDivergenceFree.py`, `warpSPH/rigidBody/ghostParticles.py` | one-time particle sampling / hashmap setup / contour extraction at initialization, not per-step solver work -- shows up as a fraction because the census divides a fixed setup cost by the number of measured steps; keeps shrinking as more steps run |

No skip was needed against the "genuinely load-bearing" bar except
`runner.py:306`'s NaN guard, which controls the run loop itself rather than
selecting a value -- that one needs restructuring (a device-side flag
checked less often, or accepting the sync), not substitution.
