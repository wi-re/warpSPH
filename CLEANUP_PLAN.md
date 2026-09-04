# warpSPH — Cleanup Plan

> ## ✅ COMPLETE — 09-04
>
> Nothing actively tracked remains open. Of the three items still open as of
> the last update, two were decided against (repo-weight rewrite: dropped, not
> a concern — see below; notebook simplification remainder: dropped, notebooks
> are fine as-is) and the third (dead commented-out code in 6 files) isn't, on
> its own, a strong enough reason to keep a standing plan document alive — per
> the user, 09-04; left below as an opportunistic note, not tracked work. The
> one item actually completed this pass: the `warp-lang` version pin — upgraded
> to 1.17.0, pinned, full `warpSPH` suite green, and cross-checked against
> `warpSPHCore` (419/420 tests green, the one failure pre-existing and
> unrelated; the specific ternary-adjoint bug this pin exists for was directly
> reproduced as fixed via `warpSPHCore/scripts/repro_ternary_adjoint_zeroing.py`
> — see "Deferred by decision" below). Kept as a historical record;
> `LESSONS_LEARNED.md` still holds the reusable lessons.

Status tracker for the cleanup sweep preceding forward-mode AD work. Core and
Integrators (separate repos) are already overhauled; this repo was the lagging
piece. For reusable lessons (bug classes, gotchas, process notes) rather than
current status, see `LESSONS_LEARNED.md` and, for AD/gradcheck specifically,
`.claude/skills/gradcheck/SKILL.md`.

## Status

| Phase | Status |
|---|---|
| 0 — GitHub/import renames, dead-code deletions, doc links | ✅ Done |
| 1 — Mechanical fixes (editable installs, config round-trip, `__init__.py`s) | ✅ Done |
| 2 — Examples → runnable scripts + first tests | ✅ Done |
| 2b — Every example as a runnable case (27/27) | ✅ Done |
| 3 — Structural (`SchemeBundle`, `compParams`→`schemeConfig`, `DomainDescription`, namespace) | ✅ Done, including `__all__` coverage (97%, remainder explicitly out of scope) |
| 3b — Post-reshuffle repair (`io/`, `math/`, `geometry/` packages) | ✅ Done — remaining backlog (below) resolved or dropped by decision, 09-04 |
| 4 — AD-readiness (gradcheck Tiers 0-2, `.detach()`/`.item()`/`.cpu()`/`.numpy()` audits) | ✅ **Done (2026-08-12)** — zero open findings; the Phase 4.2 `balanceTerm` segfault no longer reproduces either (2026-08-15, see below) |
| Repo weight (git-history rewrite) | 🚫 Dropped, not pursuing (09-04) |
| warp-lang version pin | ✅ **Done (09-04)** — 1.17.0 |

Naming is fully unified: local dir == GitHub repo == import == PyPI dist for all
four packages (`warpSPH`, `warpSPHCore`, `warpSPHIntegrators`, `warpSPHPlotting`) —
see README for the current layout, not reconstructed here.

## Remaining work

### Legibility (Phase 3 / 3b backlog — no correctness stakes)

- [x] **Module documentation — entire `src/warpSPH/` tree (done 2026-08-15).**
      Added as a tracked item 2026-08-15 — it had never been one, so the plans
      read as though the only remaining legibility work was `__all__` and dead
      comments. Originally scoped to `modules/`, `configurations/`, `regions/`,
      `schemes/`, `caseUtils/`; extended same-day, per the user, to cover
      everything else once those five landed clean. Measured 2026-08-15
      before/after:

      | scope | before | after |
      |---|---|---|
      | module docstrings | 53/277 (19%) | **277/277 (100%)** |
      | `__all__` coverage | 106/277 (38%) | **271/277 (97%)** |

      | directory | module docstrings | share |
      |---|---|---|
      | `cases/` | 30/30 | 100% |
      | `runner/` | 8/8 | 100% |
      | everything else in `src/warpSPH/` | 239/239 | **100%** |

      `cases/` and `runner/` were already at 100% from Phases 2/2b/3; this
      item covers every remaining file the cleanup sweep skipped because the
      physics/config/case-setup/math/geometry/io/rigid-body/state layers were
      never in scope for those phases. The 6 files still missing `__all__`
      (`caseUtils/waveEquation/__init__.py`, `configurations/waveEquationConfig.py`,
      `math/noiseFunctions/__init__.py`, `modules/sps/__init__.py`,
      `sample/waveSystem.py`, `systems/waveSystem.py`) were already documented
      before this pass and out of scope — not revisited.

      Done via thirteen parallel passes split by subpackage — `modules/` first
      (surfaceDetection+boundaryConditions+sps+internalEnergy;
      shockCapturing+adaptiveSupport+noise; dissipation+pressure+gravity+util;
      liu+incompressible+eos+deltaSPH+shifting;
      compSPH+crk+mdbc+momentum+timestep+density), then `configurations/`;
      `regions/`+`schemes/`; `caseUtils/` in two batches (the compressible
      case-family directories, then the rest + `waveEquation/`); then a second
      round covering the rest of the tree: `math/`+top-level package files;
      `geometry/`+`initializers/`+`io/`; `rigidBody/`+`systems/`+`utils/`; and
      `sample/`. Each file got a module docstring plus a grounded `__all__`
      (cross-checked against the owning `__init__.py`'s imports plus a
      repo-wide grep for cross-file usage, not guessed). Every touched file
      verified with `python -m py_compile`; full-repo verification after all
      thirteen passes landed: `python -m py_compile` over every file in `src/`,
      `python scripts/check_imports.py` (276 modules, 1613 first-party imports,
      all resolve), `scripts/run_tests.sh` (89/89 tests), and
      `scripts/run_sweep.py` (27/27 cases) all green, no import or logic
      breakage — run twice, once after the first five directories and again
      after the full-tree extension.

      Real (non-documentation) findings that surfaced while reading the code
      closely enough to write accurate docstrings:
      - `modules/eos/gas.py` (`idealGas`, `computeQuantitiesIdealGas`,
        `props.py`'s `fluidProperties`/`EOSSource`) is dead: not imported by
        `eos/__init__.py` or any scheme/case. The live ideal-gas path is
        `idealGas.py` → `idealGasEOS`. Its own `fluidProperties` dataclass is
        also a separate, smaller type from
        `configurations.moduleConfigurations.fluidProperties.fluidProperties`
        (the one actually used as `schemeConfig.fluid`) — a name collision
        that could confuse a future reader more than the dead code itself.
      - `modules/eos/weaklyCompressible.py:43-44` (resolved 2026-08-15): a
        `torch.clamp(..., min=0.8)` line was computed and then immediately
        overwritten by the next line, so it never took effect. Per the user,
        this is intentional and should stay that way — the clamp was "a very
        rough attempt to circumvent some instabilities"; non-physical
        densities are supposed to blow up rather than be silently masked, so
        they can actually be caught. Line commented out (not deleted) with a
        note that this is where a clamp would go if one were ever wanted,
        instead of leaving a dead-but-computed line that misleadingly implied
        an active guard.
      - **`schemes/crkSPH.py`'s `gradHState` `UnboundLocalError` (fixed
        2026-08-15).** `gradHState` was read (as an argument to
        `computeMomentumConsistent`) inside the `if currentState.divergence is
        None:` first-call fallback branch, but only ever assigned *after* that
        branch (`gradHState = None`, its adaptive-support branch also
        commented out). Any call where `divergence` starts unset — a fresh
        state's first CRK step — would raise `UnboundLocalError`. Likely why
        the Phase-4.2 `computeCompSPHBalanceTermWarp` segfault investigation
        (below) found CRK needed "a fully set-up state": that workaround
        incidentally avoids ever hitting this branch, but the crash was still
        live for any caller that doesn't pre-populate `divergence`. Fixed per
        the user: `gradHState = None` hoisted above the branch that reads it.
      - `schemes/deltaSPH.py`'s mDBC density/velocity calls look unconditional
        at the call site (one inline comment claiming a skip is stale), but
        both `computeMdbcDensity`/`computeBoundaryVelocities` no-op internally
        when there are no boundary-kind particles, so this is stale
        documentation, not a functional bug — corrected in the module
        docstring rather than left overstated.
      - `configurations/region.py`'s `parseInitialConditions` builds
        `outDict` but has no `return` statement, always yielding `None`; its
        siblings `parseParticleSet`/`unparseParticleSet` are defined but never
        called (`ParticleRegion.toDict`/`fromDict` hardcode `particles=None`
        with those calls commented out).
      - `configurations/moduleConfigurations/viscositySwitchParameters.py`:
        `ViscositySwitchConfig.limitXi` is declared twice in the dataclass
        (default `False`, then `True`) — only the second is live; confirmed
        via `modules/shockCapturing/CullenDehnen2010.py:196` that `True` is
        what's actually consumed.
      - `configurations/incompressible.py`'s `incompressibleConfigToDict`/
        `dictToIncompressibleSPHConfig` silently drop `regions`/`rigidBodies`
        on round-trip, unlike the otherwise-parallel `weaklyCompressible*`
        functions.
      - `schemes/crkSPH.py`'s `crkSPH_step` type-hints `schemeConfig:
        CompSPHConfig`, but `schemes/builder.py` actually passes a
        `CRKSPHConfig` (which has `.crkViscosityParams`, absent from
        `CompSPHConfig`) — harmless since Python doesn't enforce hints, but a
        stale/misleading annotation.
      - `caseUtils/compressible/triplePoint/equalMass.py`'s
        `sampleTriplePointEqualMass` takes `splitX`/`splitY` for the initial
        lattice layout, but its post-hoc region-density assignment re-derives
        the three region masks from hardcoded thresholds instead of the
        passed-in values — happens to match `cases/triplePoint.py`'s current
        defaults exactly, would silently disagree if a caller passed different
        splits.

      **More findings from the full-tree extension (2026-08-15), most
      relevant first:**
      - **`systems/weaklyCompressible.py:224-228` — the Padé pole-clamp fix
        (see [[project_delta_sph_instability_fixes]]-equivalent note above,
        "delta-SPH instability fixes") is currently dead.** `epsilon` is still
        computed and clamped to `[-1.5, 1.5]` ("Pade approximant has a pole at
        +-2, stay well clear of it"), but the Padé(1,1) density update that
        would consume it is commented out (`# self.state.densities =
        initialRho * (2 - epsilon) / (2+epsilon)`) and replaced by an
        unclamped exponential update, `initialRho * torch.exp(dt * drhodtMid /
        midRho)`, "to avoid negative densities" per its own inline comment.
        This looks like a deliberate, reasonable design change (exponential
        updates can't go negative; Padé(1,1) can, pole or not) rather than a
        bug, but it means the clamp this repo's history credits with fixing a
        past instability is not what's actually running now — worth
        confirming with the user whether the exponential form is the current
        intended behavior or a regression, and whether `epsilon`/its clamp
        should be deleted as dead code or restored to active use.
      - `modules/eos/gas.py`-class dead code, recurring: `initializers/
        weaklyCompressible.py:9-115`'s `initializeWeaklyCompressibleState` is
        never called from any case/runner/config path; the live path is
        `initializeState`/`initializeSimulation` (same file, line 131+).
      - `geometry/sdfFunctionality/implicitFunctions.py`: `sdEgg` mutated its
        input tensor in place (`p[:, 0] = torch.abs(p[:, 0])`) — the exact bug
        class already found and fixed in `sdMoon` in the same file (with an
        explanatory comment there), apparently missed when that fix was
        applied. **Fixed 2026-08-15**, mirroring `sdMoon`'s out-of-place
        pattern; verified a repeated call on the same tensor now returns
        identical results and leaves the input tensor unmodified.
      - `geometry/sdfFunctionality/implicitFunctions.py`: `sdRoundedCross` is
        a complete, correctly out-of-place implementation but is missing from
        `functionDict` and `__all__`, unlike every sibling shape function —
        unreachable through the public `getSDF` API. Looks like an omission,
        not intentional exclusion. Not fixed (adding it to the registry is a
        one-line change but changes public surface area, left for the user).
      - `math/noiseFunctions/generator.py:37` — `generateSimplex` called
        `_init(seed)` (defined in `util.py`) without importing it anywhere in
        the module: an unconditional `NameError` on every call, reachable via
        the public `generateOctaveNoise(..., kind='simplex')`/`generateNoise`
        (`'perlin'` is the default and the only kind used anywhere in this
        repo, which is why this was never caught). **Fixed 2026-08-15** by
        importing `_init` directly.
      - Fixing the above surfaced a second, deeper bug in the same dead code
        path: `math/noiseFunctions/util.py`'s `_init` uses `np.*` without
        importing `numpy` itself — it only "worked" by `np` leaking through
        `from .constants import *`, which had no `__all__` before this pass.
        **This pass's own `__all__` addition to `constants.py` (correctly
        scoped to real constants, excluding `np`) would have turned this into
        a live regression** if left as found. **Fixed 2026-08-15** by
        importing `numpy` directly in `util.py` instead of relying on
        wildcard-import leakage. Flagging the general risk: this repo has
        many `from .x import *` chains: `__all__` tightened elsewhere in this
        pass in files with *no* test/sweep coverage could have similar
        latent leakage dependents that neither `check_imports.py` nor
        `run_tests.sh`/`run_sweep.py` would catch, since none of them exercise
        every function in every module. `generateSimplex` was only caught
        because it was independently smoke-tested during this fix; verify
        this way, not assumed.
      - `math/noiseFunctions/generator.py`: after the fix above,
        `generateSimplex(..., dim=1)` returns a raw tuple while `dim=2`/`dim=3`
        return tensors — an inconsistent return type across dimensions, found
        while smoke-testing the import fix. Not investigated further (simplex
        noise is not used anywhere in this repo currently) or fixed.
      - `caseUtils/waveEquation/sample.py:73`: `addNoise(..., noiseType=
        'perlin')` (the default) calls `sampleVoronoi`, whose only import in
        that file is commented out — a guaranteed `NameError` on the default
        path. Consistent with the wave-equation subsystem's already-documented
        "not runnable as it stands" status (see "Won't fix / rejected" below);
        not fixed.
      - `sample/optimal.py:22-28`: `sampleOptimal` computes a jittered lattice
        then immediately overwrites `positions` with fully uniform random
        values, discarding the jitter entirely — the `jitter` parameter has no
        effect. Latent: `SamplingScheme.optimal`/`glass`/`jittered`/`random`
        are never selected by any case in this repo (default is `regular`).
      - `sample/optimal.py:68-88` + `sample/wp_deltaShift.py`: `sampleOptimal`
        adds `computeDeltaShiftWarp`'s raw return value straight onto particle
        positions, but that function's contract (per `modules/shifting/
        delta.py`, the live runtime shifting path) is to return an *unscaled*
        term the caller must scale by `-CFL * Ma * 2 * h^2` — which
        `modules/shifting/delta.py` does and `sampleOptimal` doesn't, even
        though it passes `CFL`/`computeMach`/`c_max` as if they mattered.
        Compounding it, those three args are computed into a local
        `shiftScaling` inside the warp kernel that is never applied to the
        output — dead on both sides of the call. Same latency caveat as above
        (unreachable via any current case).
      - `sample/shell.py:15-16` and `:42-45` — **unlike the two `sample/`
        findings above, this one is live**: `shell.py` is used by
        `caseUtils/compressible/sedov/initial.py` and
        `caseUtils/compressible/yeeVortex/sample.py`, both exercised in the
        27/27 sweep. Two computed-then-discarded values: `dr_init =
        torch.sqrt(dx**2 / np.pi)` is immediately overwritten by `dr_init =
        dx/2` on the next line (same pattern as the already-resolved
        `weaklyCompressible.py` EOS clamp), and `dTheta = 2*np.pi/n_hat` is
        used only as `theta += random.random() * dTheta * 0` — the trailing
        `* 0` silently makes the intended per-shell angular jitter a no-op.
        Neither crashes (the sweep passes either way), but both mean the
        shell sampler's actual initial-condition quality (radial spacing,
        angular jitter) is not what its own code implies it should be for
        `sedov`/`yeeVortex`'s spherical/radial ICs.
      - `systems/compSPH.py:63-71`: `apply_velocity_update` has an
        unreachable `return updated` after its real `return`, referencing an
        undefined name `updated` — harmless (dead code, never executes), but
        the commented-out logic above it suggests a "passive particles skip
        acceleration" path was started and abandoned mid-edit.
      - The `passive` field on every `*SystemUpdate` (`compressibleMonaghan.py`,
        `incompressible.py`, `weaklyCompressible.py`) is populated as an
        all-`False` mask by every scheme (`crkSPH.py`, `divergenceFree.py`, `compSPH.py`,
        `deltaSPH.py`) but never read anywhere except the dead code above —
        declared, filled in, never consumed.
      - `rigidBody/integrate.py`: `integrateRigidBody(rigidBody, dudt, dwdt,
        dt)` is always called with `dudt=dwdt=0` (from `systems/
        weaklyCompressible.py:225`/`systems/incompressible.py:284`). The
        function's signature implies force/torque-driven dynamics, but in
        practice a rigid body only ever moves at whatever
        `angularVelocity`/`linearVelocity` a case set once (e.g.
        `cases/movingObstacle.py`) — there is no live two-way fluid-drives-body
        coupling path despite the API shape suggesting one.
      - `rigidBody/ghostParticles.py:32-33`: `clampedDist = sdfValues` is
        computed then immediately overwritten by the next line — same
        computed-then-clobbered pattern as above.
      - `utils/timer.py`'s `TimedBlock.__exit__` never actually sets
        `cuda_ms` (the `elapsed_time`/`synchronize()` calls are commented
        out). Low-stakes: every current use site is itself commented-out
        profiling scaffolding in `schemes/deltaSPH.py`/`divergenceFree.py` that
        computes `cuda_ms` externally instead.
      - `systems/compSPH.py`/`compressibleMonaghan.py`: `entropies` is tagged
        `'soundSpeed'` (duplicating `soundspeeds`' own tag) and `pressures` is
        tagged `'damping'` — looks like copy-paste. Confirmed inert: the
        tag-lookup functions (`find_tagged_field`/`get_tagged_attr` from
        `warpSPHIntegrators`) are never called anywhere in this repo.
      - `configurations/region.py`, `moduleConfigurations/
        viscositySwitchParameters.py`, `configurations/incompressible.py`,
        `schemes/crkSPH.py`'s stale type hint, and
        `caseUtils/compressible/triplePoint/equalMass.py`'s ignored
        `splitX`/`splitY` were already listed above (found during the first
        five-directory pass, still open).

      None of the findings above beyond the ones marked "fixed"/"resolved"
      have been acted on — recorded for a future decision, not silently
      changed. This is deliberately a documentation pass, not a bug-fix pass;
      per the user (2026-08-15), keep recording what surfaces along the way
      rather than fixing it inline, except where a fix is small, isolated, and
      unambiguous enough to not need a design decision (the ones marked
      "fixed" above all clear that bar: an in-place-mutation matching an
      already-applied sibling fix, and two missing imports — one of them this
      pass's own regression, caught before it could land).

      This item and the `__all__` item below were done together, as planned —
      working out a module's real exports was most of the work of describing
      it, and splitting them would have meant reading each file twice.
- [x] **`__all__` coverage.** 106/277 files (38%) → **271/277 (97%)**, done
      together with the module-documentation item above (re-measured
      2026-08-15; was 94/274 on 2026-08-12 before that). The 6 remaining files
      without one were already documented before this pass and out of scope
      (listed in the module-documentation item above) — do opportunistically
      if those get touched, the way `schemes/` (converted to explicit imports)
      was the worked pattern before this pass.
- [~] **Dead commented-out code** in `schemes/crkSPH.py`,
      `modules/mdbc/wp_nopenshift.py`, `shockCapturing/CullenDehnen2010.py`,
      `schemes/divergenceFree.py`, `caseUtils/compressible/sod/sod.py`,
      `schemes/deltaSPH.py`. The "~421 lines" figure recorded here on 2026-08-12
      did not reproduce on 2026-08-15: a re-count over the same six files gave
      598 by one definition of "commented-out code" and 522 by a looser one, and
      every per-file number came out higher than listed. The counting rule was
      never written down, so **fix the definition first, then re-measure** —
      per LESSONS_LEARNED's "re-measure a stale plan's numbers" rule, which was
      itself derived from this kind of drift. **Dropped from active tracking,
      09-04** (per the user: not a strong enough reason to keep a whole plan
      document alive) — clean up opportunistically if one of these files gets
      touched for another reason, the way other dead-code removal has happened
      inline elsewhere this repo's history, rather than as scheduled work.
- [x] **Notebook simplification, pilot (2026-08-12): Sod.** Started with
      `01-sod-shock-tube-1d.py`/`01-Sod_Shock_Tube_1D(_resume).ipynb`, moved into
      `examples/compressible/01-sod/` (`sod_1d.py`/`.ipynb`,
      `sod_resume.py`/`.ipynb`) — the new per-case-directory convention for a case
      that needs more than one variant (numbered case dir, unnumbered contents).
      **Corrects the plan's original framing below**: a notebook that only
      re-derives its `.py` sibling should not become the same thin
      `caseMain()`-wrapped script — the user wants notebooks *fatter*, not
      thinner, because they're the place to prototype new physics/hooks. Sod's
      notebooks now build ICs via the real case code and call the same generic
      helpers the runner does (`buildContext`, `Case.setupPlot`/`updatePlot`,
      `encodeFrames`), but keep **the step loop itself unrolled and visible in the
      cell** instead of hidden behind `run()`, plus an explicit "parameters" cell
      surfacing every knob a CLI flag would (`nx`, `tLimit`, ...). Don't re-propose
      collapsing them into a thin wrapper.
      - Also switched Sod's examples (not `sodCase.defaults` — `run_sweep`/tests/
        `sod_highres.yaml` are untouched) to the datagen trajectory export scheme
        (`storeMode='trajectory'`, one growing `trajectory.h5`) instead of
        one-file-per-stored-step. This required real fixes, not a rename:
        `writeInitialData`/`writeFrame` (`warpSPH/io/export.py`) were hard-coded to
        the weakly-compressible/dam-break state shape and crashed on a compressible
        state (`schemeConfig.rigidBodies`, `state.ghostOffsets` don't exist on
        `CompSPHConfig`/`CompSPHState`) — now guarded with `getattr(..., default)`.
        Added an `extraFields` parameter (default `()`, every existing caller
        unaffected) so a case can declare additional per-frame fields; `Case` grew
        an `extraFields` attribute so `run()` threads it through automatically.
        Sod passes `('internalEnergies', 'supports')` — per the user, only
        `internalEnergies` is the core EOS state variable worth writing every frame
        (`pressures`/`soundspeeds`/`entropies` are recomputed from it via
        `idealGasEOS` on load, to keep file size down); `supports` also needs
        per-frame export because Sod's default adaptive support scheme drifts from
        the t=0 snapshot, unlike `masses`/`kinds`/`materials`/`UIDs` which are
        genuinely IC-only. Added `warpSPH.io.loadTrajectory`/`loadTrajectoryFrame`
        — a real generic reader (promoting the shape of `compressedLoader.ipynb`'s
        inline, plot-only logic into a library function that reconstructs a live,
        resumable `SimulationState`, which nothing did before).
      - **2D and 3D Sod: done (2026-08-12)**, `warpSPH/cases/sodND.py`
        (`sod2d`/`sod3d`, plus `examples/compressible/01-sod/sod_2d.py`/`_3d.py`
        and matching notebooks following `sod_1d.ipynb`'s pattern — parameters
        cell, real case code for the IC, unrolled step loop — with two cells
        that only make sense here: the sampler's integer choices inspected
        without building anything, and the density field drawn in the slab
        itself (2D) or a thin z-slice of it (3D)),
        sampled at equal particle *mass* rather than equal spacing --
        `caseUtils/compressible/sod/sodND.py`. The 3D plotting gap this entry
        used to cite turned out not to exist: the solution is 1D along the tube,
        so the existing six Sod profile panels work in any dimension as a
        scatter (`plotSod_` now indexes the x column explicitly, a no-op in 1D).
        Running actual 3D physics for the first time found two real bugs, both
        fixed and both regression-tested: warpSPHCore's B7 kernel had a **3D
        normalisation constant 16x too small** (1D and 2D correct), and Owen's
        adaptive-support psi LUT was cached in an **unkeyed module global**, so
        the first dimension a process touched won and everything after it
        relaxed supports against the wrong table.
      - **Backprop-over-trajectory demo: done (2026-08-12)**,
        `examples/compressible/01-sod/sod_backprop.ipynb`, both stages the user
        asked for — self-consistency recovery (perturbed `masses`, then `gamma`)
        and the fit to `sodSolution.solve`'s analytic Riemann solution. Not in
        CI (a 500-step BPTT is too expensive per commit); the logic is written
        as importable functions so a future
        `scripts/gradcheck_sod_trajectory.py` can lift it. Design record and
        measured results in that directory's `BACKPROP_PLAN.md`. Found and fixed
        a real severed-gradient bug on `balance.py`'s `gamma` kernel argument
        (the `dt` treatment of Phase 4.2, applied to the argument next to it),
        guarded by a new `run_balance_gamma_gradcheck` in
        `scripts/gradcheck_compSPH.py`.
- [x] **Notebook simplification, compressible family (2026-08-13).** All 13
      `examples/compressible/` slots (14 cases) converted; see
      `examples/compressible/MIGRATION_STATUS.md` for the full case list and
      the two notebook shapes (`profilePlot` 1D, `particlePlot` 2D field —
      the latter's window/event-loop-free core, `buildFieldPlotter`/
      `refreshFieldPlotter`, is new in `cases/plotting.py`, piloted on `08`).
      `14`/`15` merged into one `14-triplePoint/` directory the way `06`/`07`
      merged into `06-sedov/` earlier in this item.
- [~] **Notebook simplification, the rest — dropped, not pursuing further
      (per the user, 09-04).** Notebooks are fine as they are at this stage;
      the remaining slot below is left exactly as it was when this was last
      touched, not tracked as active work. `examples/weaklyCompressible/` is
      12 of 13 slots done (see its own `MIGRATION_PLAN.md` for the per-slot
      table, which is the authority here). Only slot **13**
      (`channelFlow.openFlowCase`) remains: slot 11 was finished 2026-08-14 with
      its case redesigned, and is no longer a `channelFlow` hook — this entry
      claimed "11 and 13 remain, both `channelFlow`" until 2026-08-15, which
      contradicted `MIGRATION_PLAN.md` on both the count and the case.
      `examples/incompressible/` is untouched — and two of its three notebooks are the weakly compressible
      cases under `--scheme divergenceFree`, so they follow 03 and 06 rather
      than needing cases of their own.
      **Read `PORTING_EXAMPLES.md` first** — the procedure,
      the notebook conventions that have a reason behind them, and (separately)
      how to take a case to 2D/3D, all written from doing Sod; the compressible
      pass above is a second worked example, particularly for any case that
      turns out to be 2D/field-plot shaped.
      - Boilerplate cell duplicated verbatim in the remaining notebooks, still
        on the pre-`warpSPHBootstrap` config path — cheapest win, do first.
      - Apply the Sod pilot's pattern (real case code + generic helpers, visible
        step loop, explicit parameters cell) rather than collapsing to thin
        wrappers; keep genuinely notebook-shaped analysis/plotting as its own cells.
      - Worst offender left: `13-open-flow.ipynb` — 57 cells, 1199 code LOC
        (re-measured 2026-08-15; this read "1704 LOC" before). The
        incompressible notebooks are the other half: 692 (`01-taylor-green-vortex`),
        697 (`periodic-random-flow`), 434 (`03-rotating-square-patch`), and
        608 in the `1d-test.ipynb` scratchpad that is deliberately not ported.
- [x] `datagen/weaklyCompressible/bak/` — gone; the directory no longer exists.

### Documentation and usability pass (2026-08-15)

A survey of the repo from a reader's/new-user's point of view, deliberately run
*without* reading these planning documents first, so the findings are what the
tree itself says rather than what the plans claim. Verified throughout with the
standing commands (89 tests, `check_imports.py`, 27/27 sweep).

- [x] **`scripts/check_imports.py` was silently under-checking.** It assumed a
      notebook cell's `source` is always a list of lines; nbformat also permits
      a single string, and iterating that walks it character by character. Two
      notebooks (`01-sod/sod_resume.ipynb`, `14-triplePoint/triplePoint_equalMass.ipynb`)
      were reported as bogus `SyntaxError`s and their imports never actually
      checked. Fixed; scanned first-party imports went 1596 → 1612. Worth noting
      given this file is listed above as a standing verification command.
- [x] **504 dead commands in `datagen/weaklyCompressible/cases/`.** Seven of the
      eight `.sh` batches still passed `--timeLimit`, which no longer exists —
      every one of them failed instantly with `unrecognized arguments`. Only
      `examples.sh` had been updated. All 28 flags used across the batches were
      diffed against `generator.py`'s parser; `--timeLimit` was the only stale
      one. Renamed to `--tLimit` throughout and smoke-tested. The typo'd
      near-duplicate `exmples.sh` (an older copy of `examples.sh`, missing the
      two kolmogorov lines) was deleted.
- [x] **Example naming brought in line with the documented convention.** Both
      `MIGRATION_STATUS.md` and `MIGRATION_PLAN.md` specify
      `<slot>-<name>.ipynb`, but the compressible family shipped
      `08-Hydrostatic.ipynb` next to `08-hydrostatic.py`. 12 files renamed (git
      records all as renames) and 108 references updated across 53 files;
      `render_examples.py --list` verified byte-identical afterwards, so no
      output artefact changed name. `siblingNotebook`'s case-insensitive/slot
      fallback stays — it is what makes a *new* example work before anyone has
      thought about naming.
- [x] **New documentation.** `CONTRIBUTING.md` (install order, the per-clone
      `nbstripout --install` step, what to run before committing — the repo has
      no hosted CI **by decision**, since the suite needs a GPU),
      `datagen/README.md`, and `examples/weaklyCompressible/EXAMPLES_SUMMARY.md`
      (the weakly-compressible gallery; the compressible one had no counterpart,
      so ~40 tracked GIF/MP4/PNG were reachable from nothing).
- [x] **`--help` made usable.** Every `CaseSpec` flag read `CaseSpec.<field>`;
      the enum-valued ones listed no values, and the `--no-x` twins were
      `argparse.SUPPRESS`ed, so `--no-periodic` was undiscoverable while the
      visible `--periodic` was a no-op. Added a per-field help table, enum
      choices read off the enums themselves, and `BooleanOptionalAction` so both
      polarities show on one line. `choices=` deliberately *not* used — the
      `parse*` helpers match case-insensitively and it would start rejecting
      `--kernel b7`.
- [x] **Packaging metadata.** `pyproject.toml` had no `authors` and no
      `[project.urls]`, and its `description` was copy-pasted from `warpSPHCore`
      ("neighbor search and operators") — i.e. PyPI advertised this package as
      the thing it depends on. Identity copied from `warpSPHIntegrators` for
      consistency across the four repos.
- [x] **The Phase 4.2 `computeCompSPHBalanceTermWarp` segfault is resolved.**
      `scripts/troubleshoot_balanceTerm_segfault.py` — which still described a
      live crash — is clean in all three modes and all six energy schemes; its
      "only `equalWork` survives" finding no longer holds. Cause, per the user,
      was **not** the out-of-bounds read the investigation was chasing:
      - The harness never built the reference states properly, so the minimal
        repro was an *invalid* input rather than a stripped-down valid one. Its
        "cold process / fresh allocations" lead was a dead end.
      - The real bug: `referenceVolumes` left as `None` instead of falling back
        to `queryVolumes`, so a reference state with no volume member passed the
        kernel a null array. Several entry points were affected, not just this
        one. Fixed upstream in `warpSPHCore` on 2026-08-06 by `120c4bf` ("make
        the state path the primary one") — the fallback now at
        `warpSPHCore/operations.py:52`. CRK specifically was fixed by passing a
        fully set-up state.

      Recorded in the script's docstring too. The elimination record there is
      kept, but its conclusions have to be read knowing the input was faulty.
      **Deleting the script is a call for the user, not cleanup.**
- [x] Smaller: `log.txt` (a committed pytest dump) removed and gitignored; an
      invalid escape sequence in `caseUtils/compressible/sod/sodSolution.py`
      made a raw docstring; `tests/test_gradcheck_scripts.py` now discovers the
      gradcheck scripts by glob instead of a hardcoded list that could silently
      stop covering one; six notebooks declaring nbformat 4.5 without cell `id`
      fields normalized; the wave-equation subsystem documented (below).

### Cross-repo (owned by warpSPHCore) — DONE (2026-08-12)

- [x] `warpSPHCore.util.support.volumeToSupport` now dispatches on
      `isinstance(volume, torch.Tensor)` internally. `warpSPH/utils/support.py`'s
      wrapper collapsed to a plain re-export; verified scalar and tensor paths agree
      across all three dimensions, full suite still green.
- [x] `datagen/weaklyCompressible/loader.ipynb` — removed. `compressedLoader.ipynb`
      already reads the current (compressed) export layout, making the hardcoded-path
      loader redundant rather than something to fix.
- [x] **`B7_C_d`'s 3D normalisation constant was 16x too small**
      (`warpSPHCore/kernels/kernelFunctions/B7.py`): `1024/(105*pi)` where the
      integral of `k` over the unit ball gives `16384/(105*pi)`. An SPH density
      sum over a uniform 3D lattice returned 1/16 of the mass it was built from;
      1D and 2D were already correct, and B7 is the compressible examples'
      default kernel. Found by `sod3d`, the repo's first 3D case. Every other
      kernel checked at the same time and correct in all three dimensions
      (`Poly6`/`Spiky` are off in 1D/2D, but those are the graphics kernels,
      normalised for 3D only — left alone).

### Won't fix / rejected (recorded so they aren't re-proposed)

- `math/` shadows the stdlib name — harmless under absolute imports, no replacement
  name was ever proposed, out of scope for the naming pass.
- Remaining basename collisions (`sod.py`, `noh.py`, `region.py`, `sdf.py`, the
  `caseUtils/compressible/*/sample.py` family) — each is case/package-namespaced on
  purpose and reads fine in context; not churned.
- `examples/weaklyCompressible/naca.ipynb`, `examples/incompressible/1d-test.ipynb`
  — deliberately have no case (standalone SDF viz / exploratory scratchpad, not
  published examples).
- **The wave-equation subsystem stays** (`schemes/waveEquation.py`,
  `systems/waveSystem.py`, `configurations/waveEquationConfig.py`,
  `sample/waveSystem.py`, `caseUtils/waveEquation/`). It has no registered case,
  no test and no example, so a survey reads it as dead — it is not. Per the user
  (2026-08-15) it is kept as a compact demo of the SPH operators over
  *unstructured, mesh-like data*: a moving-neighbourhood Laplacian on scattered
  points with a heterogeneous wave speed, which is the shape of problem
  graph/point-cloud ML models are posed on. Module docstrings and a README
  section now say so. Note it is **not runnable as it stands**: the path is
  `build_configs_from_casefile` → `sampleParticles` → `genInitial` →
  `finalizeWaveSystemSetup` → integrate with `f_wave_equation`, and nothing runs
  those five stages; no TOML casefile ships either, and `genInitial` allocates
  `nx**2` so it is 2D-only. Making it a usable demo is unscoped work, not cleanup.
- **No hosted CI.** Per the user (2026-08-15): the test suite and the case sweep
  both need a CUDA device, so GitHub runners would be slow and burn compute
  budget for little return. The checks are run locally before commits and bigger
  changes instead — see `CONTRIBUTING.md`. Don't propose a GitHub Actions
  workflow for tests, sweeps or `check_imports.py`.
- ~~Sedov's `initialization='hat'` — both notebooks crashed on this default; the case
  now defaults to `'singular'`. Reviving `'hat'` is a separate, unscoped job.~~ Done
  2026-08-13: `'hat'` now deposits E0 on the central particle and smooths it with one
  SPH interpolation pass over the finalized adaptive supports; it's the case default
  again. Sedov also moved into its own directory (`examples/compressible/06-sedov/`,
  1D/2D/3D) in the same pass, which surfaced a real bug: `B7_C_d(dim=3)` in the sibling
  `warpSPHCore` repo was 16x too small (a regression of the "B7's 3D kernel
  normalisation" bug noted in `PORTING_EXAMPLES.md` section 4.7 — apparently never
  fixed for the `sampleRegularParticles`/plain-`Density` code path Sod's own 3D builder
  doesn't use). Fixed there (uncommitted, separate repo); see
  `tests/test_physics.py::test_uniformLatticeDensityMatchesBuiltDensity` for the
  regression test.

## Deferred by decision

- **Repo weight / git-history rewrite — dropped, not a concern (per the user,
  09-04).** Not just deferred any further: the validation-data commits since
  (SPHERIC TestCase10's data/video files, ~intentionally~ added this session)
  mean the repo is deliberately taking on more weight at this stage, not
  trying to shed it. `nbstripout` stays installed and still applies going
  forward regardless. If a history rewrite is ever wanted it would need
  re-scoping from scratch against whatever the tree looks like then — the
  338 MB / 312-blob-version numbers below are stale and shouldn't be reused.
- **warp-lang version pin — DONE, 09-04.** Upgraded to **1.17.0** and pinned
  in `pyproject.toml` (`warp-lang>=1.17.0`, fixes the same-array-ternary/
  Interpolate adjoint bug at the source, so no more `access_optional`
  workaround requirement going forward). Don't downgrade to 1.15.0/1.16.0/
  1.12.0 — that's been explicitly ruled out.

## Notes

- Renames break the dill-encoded callables in the ~244 pre-existing local `.h5`
  datasets — accepted, since they're regenerable and no large dataset had been
  built yet.
- `scripts/check_imports.py`, `scripts/run_tests.sh` (89 tests, ~2 min), `scripts/
  run_sweep.py` (27/27 cases, ~4 min) are the standing verification commands; prefer
  them over hand-rolled loops. Counts re-measured 2026-08-15 (they read 58 and 25/25
  here until then, and the README said 42 — all three were stale). The `gradcheck` skill (`.claude/skills/gradcheck/
  SKILL.md`) covers the 15 gradcheck scripts specifically.
