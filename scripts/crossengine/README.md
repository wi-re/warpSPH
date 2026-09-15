# diffSPH cross-engine comparison harness

Investigation from `DELTASPH_VALIDATION_PLAN.md` Part 8 (2026-09-14): matches
warpSPH's `dambreak` case IC to diffSPH's `15_Dambreak.ipynb`, then bisects
where the two engines' full trajectories diverge by running each engine's own
raw step/shift function on the *other* engine's state.

All scripts assume the `diffSPHEnv` conda env (has both `warpSPH` and
`diffSPH` importable) and the matched dambreak CLI args:

```
--nx 64 --L 2.0 --n_h 4.0 --kernel Wendland4 --integrationScheme symplecticEuler
--supportMode SuperSymmetric --tLimit 4.0 --targetDt 0.0005 --band 7
--fillRatio 0.5 --fluidWidth 0.41666... --gravityMagnitude 10.0 --wallBC constant
```

- `dump_warpsph_ic.py` / `dump_diffsph_ic.py` -- build each engine's own IC
  under the matched params and print dx/c0/targetNeighbors/particle counts/
  density stats, for verifying the two ICs actually match.
- `compare_step_force.py` -- loads a mid-trajectory snapshot (from a
  completed warpSPH run's `trajectory.h5`), strips walls from both sides
  symmetrically, and calls each engine's own raw force function
  (`deltaSPH_step` / `deltaPlusSPHScheme`) on the identical fluid-only
  state, comparing `dvdt`/`drhodt` per particle.
- `compare_shift_vector.py` -- same idea for the shift/PST term
  (`computeDeltaShift` / `computeDeltaShifting`). **Sign convention**: the
  quantity that gets added directly to positions is warpSPH's own
  `shift`/`update`, and it is diffSPH's raw `computeDeltaShifting(...)`
  output *un-negated* (diffSPH's own caller negates then subtracts, so the
  two cancel) -- get this backwards and the cosine similarity comes out
  strongly *negative* instead of strongly positive.
- `run_diffsph_trajectory.py` -- diffSPH's own dambreak, faithfully
  reproducing the notebook (fixed dt computed once at setup, not
  recomputed per step), to completion.
- `run_diffsph_physics_on_warpsph_ic.py` -- diffSPH's full physics AND
  shift, run on warpSPH's own IC (fluid + warpSPH's own boundary/ghost
  mDBC complement, reusing warpSPH's `ghostIndices`/`ghostOffsets` arrays
  verbatim -- same per-particle convention in both codebases). Bands
  measured via warpSPH's own `dambreakCase.diagnostics`.
- `run_hybrid_physics_swap.py` / `run_hybrid_shift_swap.py` /
  `run_hybrid_shiftcore_only_swap.py` -- monkey-patch exactly one piece of
  warpSPH's own case/runner/integrator pipeline (the force step, the whole
  shift step, or just the shift step's inner raw kernel-gradient sum,
  respectively) to call into diffSPH instead, leaving everything else
  (including trajectory export/diagnostics) warpSPH's own code.
- `compare_step_terms.py` -- like `compare_step_force.py` but decomposed by
  individual force term (pressure/viscosity/DDT/continuity/gravity) instead
  of only the summed `dvdt`, at several snapshot times. This is what found
  pressure+continuity bit-exact and isolated the real disagreement to
  artificial viscosity and density-diffusion (DDT) specifically.
- `run_hybrid_term_swap.py --swap none|viscosity|ddt|both` -- the
  trajectory-level 2x2 ablation on those two terms specifically (same
  monkey-patch technique, applied to `computeVelocityDiffusion`/
  `computeDensityDiffusion` this time). Needs two extra state fields
  populated before calling diffSPH's DDT term that the snapshot comparison
  didn't need: `particles.numNeighbors` and the covariance/renormalization
  matrices from `computeCovarianceMatrices` (both crash with a clear
  `ValueError` if missing -- see the script). Result: DDT-swap-alone closes
  most of the density-band gap for free; viscosity-swap-alone is a net loss
  (density bands flat, retained kinetic energy roughly doubles).
- `run_wrongsign_ddt.py` -- **pure warpSPH, no diffSPH involved.** Selects
  the new `DensityDiffusionScheme.deltaSPH_wrongSign` (added in
  `enumTypes.py` + `modules/deltaSPH/wp_densityDelta.py`, per Part 4's own
  "add a distinct member, don't overload deltaSPH" instruction) -- the
  documented pre-`790a7c7` psi sign that degrades the Antuono bi-Laplacian
  to 2x the plain Molteni-Colagrossi Laplacian (2nd-order, not 4th).
  Reproduces almost all of the DDT-swap's density-band improvement with a
  one-line, same-codebase change -- see Part 8.11.

## Known gotchas found along the way

- A diffSPH state built by hand (bypassing `initializeSimulation`) needs
  `config['particle']['support']` set explicitly -- `computeTimestepWCSPH`
  reads it directly, not `state.supports.min()`; left unset it silently
  falls back to `support=1` and pins `dt` at the ceiling.
- Upstream diffSPH bug (not warpSPH's): `diffSPH/math.py::pinv2x2` masks the
  assignment with `|o1| > 1e-5` but reads the reciprocal with `|o1| > 1e-7` --
  shape-mismatches whenever a covariance eigenvalue lands between them.
  Triggers under warpSPH's denser mDBC ghost sampling; diffSPH's own sparser
  sampling apparently never hits it. `run_hybrid_physics_swap.py` and
  `run_hybrid_shiftcore_only_swap.py` both carry an in-process monkeypatch
  with a corrected single-threshold version -- diffSPH's own repo is
  untouched.
- Monkey-patching one of these functions must target the *importing*
  module's namespace, not the defining module's -- e.g.
  `warpSPH.schemes.builder.deltaSPH_step`, not
  `warpSPH.schemes.deltaSPH.deltaSPH_step`, because `builder.py` did
  `from .deltaSPH import deltaSPH_step` (binds the name once, at import
  time) rather than a lazy attribute lookup.
