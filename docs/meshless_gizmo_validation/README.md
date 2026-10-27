# MFM / MFV vs GIZMO: evidence for the `closure='none'` and CFL defaults

Evidence behind two default changes (2026-10-12, `COMPRESSIBLE_SPH_PLAN.md` log, decisions in `PLANS.md`):

1. `MeshlessConfig.closure` default `'project'` -> `'none'`: GIZMO has no face-closure projection, and on identical ICs GIZMO
   behaves like our `none` (Yee vortex, Kelvin-Helmholtz 256), not like `project`.
2. `MeshlessConfig.cfl` 0.2 -> 0.1 with `cflLength='particle'`: GIZMO's step is `2 C L / v_sig`, `L = V^(1/d)`; ours was
   `2 C h / v_sig` with the kernel radius (~4 L at ~50 neighbours), i.e. a GIZMO CourantFac of ~0.8.

## Figures (`figures/`)

| file | shows |
|---|---|
| `kh256_t2.6_mfm_mfv_gizmo_vs_ours.png` | KH 256, identical IC, t = 2.6: GIZMO MFM / ours MFM / GIZMO MFV / ours MFV (new defaults), density and |v| |
| `kh256_t2.6_gizmo_vs_ours.png` | KH 256, t = 2.6: GIZMO MFM / ours `none` / ours `project` (the old default: density 0.04-44.5, void + clumps) |
| `yee_vs_gizmo_MFM_j0.25.png` | valid Yee vortex, jitter 0.25, scheme-consistent IC: per-particle error vs the exact solution, ours `project` / ours `none` / GIZMO MFM |
| `yee_direct_MFM_j0.25.png` | same without GIZMO: `project` vs `none` |
| `yee_summary.png`, `yee_frames_project_none_fg2.png` | the original (vacuum-disk) Yee IC: KE by closure variant; why that IC is *not* a valid closure test |

## Tables (`tables/`)

- `kh256_gizmo_{mfm,mfv}.txt`: GIZMO snapshot series (max |v|, fast-particle count, KE, density range, fraction outside [0.85, 2.35]).
- `yee_gizmo.txt`, `yee_valid_ours_grid.txt`, `ic_ab_original_vs_consistent_ic.txt`: valid Yee, GIZMO vs ours (project / none, original vs scheme-consistent IC).
- `closure_ab_battery_cfl0.2.txt`: project vs none across the compressible battery (cfl 0.2 on h -- the Hancock-noise-contaminated MFV numbers are explained in the plan log).
- `gizmo_kh_params_template.txt`: GIZMO's own `test/kh_mcnally_2d` parameter file used as the template.

## How GIZMO was run

Scratch copy of `~/dev/gizmo` (source untouched), built with `mpicc`, serial HDF5 (`-I/usr/include/hdf5/serial`), GSL headers extracted from the
`libgsl-dev` .deb (the system has only the runtime library); the `github-ubuntu` block of its Makefile was edited for those paths. `Config.sh`:

```
HYDRO_MESHLESS_FINITE_MASS   (or HYDRO_MESHLESS_FINITE_VOLUME for MFV)
BOX_PERIODIC
BOX_SPATIAL_DIMENSION=2
SELFGRAVITY_OFF
EOS_GAMMA=(5.0/3.0)          (KH; 1.4 for Yee)
KERNEL_FUNCTION=5            (quintic; ours is B7, ~50 neighbours vs GIZMO's DesNumNgb 40)
OUTPUT_IN_DOUBLEPRECISION
INPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
```

Parameters: the `kh_mcnally_2d` file (DesNumNgb 40, CourantFac 0.1, MaxSizeTimestep 0.02) with `InitCondFile`, `TimeMax`, `TimeBetSnapshot`
(and `BoxSize 10` for Yee) changed; `mpirun -np 8 --oversubscribe` (4 ranks for Yee). The ICs are exported from our own case samplers
by `tools/gizmo_ic_export.py` (same positions, masses, velocities, internal energies). GIZMO is always double precision (`MyFloat` / `MyDouble`
are `double`) and ran on CPU; ours default to fp32 on GPU. A float64 re-run of our side changes no conclusion (plan log).

## Tools (`tools/`)

Copies of the scratch drivers from `.tmp/` (they reference `.tmp/` and `results/` paths and are a record, not a maintained harness):
`gizmo_ic_export.py`, `gizmo_yee_run.sh`, `gizmo_yee_analyze.py`, `gizmo_kh_analyze.py`, `kh_mfm_mfv_gizmo_fig.py`, `kh_vs_gizmo_fig.py`,
`yee_valid.py` (valid periodic Yee IC + exact solution), `ic_audit.py` (IC consistency audit, `consistentIC`), `ic_ab.py`, `yee_direct.py`,
`yee_vs_gizmo.py`, `yee_closure_probe.py`, `closure_cap_probe.py`, `e2_kh256_confirm.py`, `e2_closure_battery.py`.

## Caveats

- Remaining gap to GIZMO: on the lattice Yee our `none` velocity error is ~30 % above GIZMO's (4.8e-3 vs 3.7e-3); not isolated (kernel / neighbour count suspected, not precision).
- MFV vs GIZMO MFV on KH is compared on KE history, max |v|, density range and frames; the billow-core fine structure was not compared quantitatively.
- Experimental, default-off, no effect under `none`: `faceGuard`, `closureCap`, `mfvUpdate` (research leftovers from this investigation).
