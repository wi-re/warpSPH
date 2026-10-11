# Retiring the boundaries repo's solvers (E7) — 2026-10-11

Stage 3 item E7 of [ANALYTIC_BOUNDARIES_PLAN.md](../../../ANALYTIC_BOUNDARIES_PLAN.md), executed on the owner's go. The reference solvers `DeltaSPH2D` and `DFSPH2D` (the `sim/` sub-package of
`~/dev/curvatureBoundaries`, package `warpSPHBoundaries`) are gone from that repo; warpSPH hosts every scheme they carried and the validation cases that exercised them.

## What was done

**Boundaries repo** (`~/dev/curvatureBoundaries`, branch `retire-solvers`, commit `9a0c295`, **not pushed, not merged into its `main`**):

* Tag **`oracle-solvers-2026-10-11`** on its last pre-retirement commit (`71ddb8c`): the complete solver layer, its 39 test files, the studies, the example notebooks and the stored baselines stay runnable there
  (`git -C ~/dev/curvatureBoundaries worktree add /tmp/oracle oracle-solvers-2026-10-11`, put `/tmp/oracle/src` first on `PYTHONPATH`). The tag is local until pushed.
* Removed (−22 356 lines): `src/warpSPHBoundaries/sim/` (23 files, 4 409 lines), `tests/sim/` (39 files), `tests/warpsph/` (the reference-solver equalities), the solver runners and studies
  (`scripts/{deltasph_*,dfsph_*}.py`, 13 of the 20 studies: everything that imported `sim`), `notebooks/examples/` (nine flows on the two solvers), `results/{deltasph,dfsph,examples}`, `patches/warpsph/`
  (the merged upstreaming series), the solver docs (`dfsph-validation`, `deltasph-*`, `plan-upstreaming`, `audit-warpsph-boundary-hooks`, `plan-next-steps`) and the `docs/work/refs` probes that imported `sim`.
* Kept: `curvbound/`, `edge/`, `scene/` (the integral machinery and the provider), their tests (**882 passed** after the removal), the derivations, `PLAN.md`, `HANDOFF.md` (Part A bannered as historical), the
  kernel tables, the paper, the derivation-check and benchmark scripts. `domain_scene` (the box-domain scene helper that lived in `sim/dfsph2d.py`) moved to `scene/domains.py`; three scene tests that used solver
  fixtures (`test_scene`, `test_channel_pruning`, `test_fixed_adjacency`) build their scenes directly or lose the solver-state test. `pyproject.toml` loses the `solvers` extra and says what the package is now; the version
  (0.1.0) is untouched, a release is the owner's call.

**warpSPH:**

* The oracle's stored reference data are copied to `docs/analytic_boundaries/oracle_data/` (`slosh_B_nx200_series.npz` = `DeltaSPH2D`'s 7 s sloshing, `cmp_dam.npz` / `cmp_tank.npz` = the compiled omniSPH dam break and
  tank, `dambreak_B_nx67_series.npz`); `scripts/compare_deltaSloshing.py`, `probe_omniAnalyticDambreak.py`, `probe_incompressibleAnalyticDambreak.py`, `probe_omniOmniSPHLattice.py` read them from there. They no longer
  need the other repo, and the sibling repo's untracked `.tmp/` is not a dependency any more.
* The solver validation records are archived in `docs/analytic_boundaries/oracle_docs/` (`plan-next-steps.md` with the validation ladder and results, `deltasph-validation.md`, `dfsph-validation.md`,
  `deltasph-porting-notes.md`), verbatim from the tag.
* The nine oracle probes (`scripts/probe_omni{OracleTerms,MlsOracle,CompactOracle,FiltersOracle,FickianOracle,ViscosityOracle,OmniSPHLattice,MatchOriginal}.py`, `probe_taylorCouetteOracle.py`) keep working **against the
  tag**: `scripts/_oracle.py` exits with the checkout instructions when `warpSPHBoundaries.sim` is not importable. They are frozen: a probe is a term-by-term check of a scheme against its original, and the original
  no longer changes. `probe_omniMatchOriginal.py` additionally reads the sibling repo's untracked `.tmp/omni/runs` (about 1 GB, not copied).
* `src/` and `tests/` never imported `sim`; nothing in the library changed. `scripts/check_imports.py` is clean.

## The oracle's tests, accounted for

The boundaries repo's 39 solver test files were the solver's own regression suite; most test the oracle's internals and go with it. What each group's coverage is in warpSPH:

| group (files at the tag) | mirrored in warpSPH | disposition |
|---|---|---|
| wall-term equalities with `DeltaSPH2D` (`tests/warpsph/`: wall terms, loads, detector, shifting; 1e-13 float64) | the numbers are recorded in ANALYTIC_BOUNDARIES_PLAN.md ("Verified", 1e-13 float64 / 1.4e-4 float32 shifting); re-checking needs the oracle | **runnable at the tag only** (worktree + `PYTHONPATH`, both precisions) |
| periodic channel / periodic wall / pinned / Morris / complement / wall-moment tables / wedge moments / pressure-consistent / no-slip / wall loads | `test_periodicChannel`, `test_pinnedBand`, `test_wallViscosity`, `test_wallPressure`, `test_stokesArray`, `test_taylorCouette`, `test_analyticDivergenceFree` (booked loads), `test_analyticObstacle` (free body driven by the load) | mirrored, in the form each port's own gate took (momentum balance, tables vs a brute-force integral, the closure on a quadratic profile) |
| compact projection, TGV decay, closed preset | `test_compactProjection` | mirrored |
| hydrostatic tank at rest, dam break vs particles, obstacles, Michel / implicit shift | `test_analyticOmni`, `test_analyticDivergenceFree`, `test_analyticIisph`, `test_analyticDambreak`, `test_analyticMichel`, `test_analyticImplicitShift`, `test_analyticObstacle`, `test_analyticGraph` | mirrored |
| the oracle's own machinery: torch pair sums vs Warp, `fluidwarp`, `graphstep`, the fibre-bundle solver, the integrator adapter, `solvers.py`, wall-data reuse, float32 gate, posthoc loads | — | **gone with the code they test** |
| δ⁺ unit gates on the old solver: density diffusion zero on hydrostatic, wall continuity sign, **the unclamped wall pressure does not pull a ceiling particle**, no-penetration impulse law of the mDBC, the shift is normal and small at a flat wall | partly (`test_analyticDivergenceFree::analytic_no_penetration…`; the dam-break/Michel tests cover the shift end to end) | **accepted loss as unit tests.** The ceiling-suction gate is the one worth porting if δ⁺ wall-pressure work resumes (it needs a one-particle harness on `wallPressureAcceleration`) |
| `deltasph_validation.py` physics runs (tank rmseBulk / rmseNear, English wedge, Marrone 3.1 P1 arrival, SPHERIC sloshing) | tank, wedge, 7 s sloshing: `docs/analytic_boundaries/{delta_sloshing_gate,divfree_analytic,sloshing_ab,omni_*}_2026-10-09`; Marrone 3.1 is warpSPH's own `dambreak` on the particle walls | stored as evidence, not as pass/fail tests (each is a 5-90 min run); **accepted** |
| corner rig (oracle K/K_ref 0.968), plain Couette between plates | — | **never ported; dropped from the gate** (plan, left-to-close item 1) |

## Decisions taken on the way (owner may reverse any)

* **Oracle kept as a tag, not as stored term tables** (the checklist's preferred option: nothing lost, cheapest). The probes are frozen against it.
* **EXPERIMENTAL flags left as they are** (`runner.EXPERIMENTAL_ANALYTIC_WALL_SCHEMES`: omni, divergenceFree, dfsphReference, iisph). Removing the oracle does not change the open items the flag names (the Jacobi relaxation limit, the sloshing third-impact overshoot / early arrival, the `divergenceFree` disk oscillation); lifting it is a separate owner decision.
* **Not pushed:** the tag and the `retire-solvers` branch exist only locally in `~/dev/curvatureBoundaries`; `main` there still has the solvers.

## Leftovers to know about

* Empty directories with ignored `__pycache__` remain in the boundaries repo (`src/warpSPHBoundaries/sim/`, `tests/sim/`, `tests/warpsph/`, `results/deltasph/`); `sim/` as an empty directory still makes `import warpSPHBoundaries.sim` resolve as a namespace package (the guard in `scripts/_oracle.py` checks `sim.dfsph2d` for that reason). Delete them by hand.
* `dist/` (built wheels, 5.3 MB, ignored) in the boundaries repo contains the solver layer; stale, rebuild before any release.
