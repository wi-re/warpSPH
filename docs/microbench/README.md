# Module micro-benchmarks

`warpSPH/utils/microBench.py` counts torch ops, warp launches and host syncs of ONE callable and times it (host enqueue, wall, GPU busy);
`scripts/microbench_<module>.py` drive it over a family of implementations (skill `module-microbench`).

* `switches_sod2d_CompSPH_2026-10-14.txt` -- every viscosity switch (`modules/shockCapturing`): terms, update and step hook per call, then per
  step (2 RK stages), on `sod2d` CompSPH at N = 2.5k / 5k / 10k, RTX PRO 6000 Blackwell, a second agent was running tests on the same GPU.
  Counts are exact; the times are indicative. Findings: OPEN_PROBLEMS section 34.
* `wadsley_after_2026-10-14.txt`, `wadsley_detector_before_after_2026-10-14.txt` -- the Wadsley detector before (torch ops) and after (warp kernels,
  `Wadsley2017/kernels.py`): 6 -> 0 syncs and 110 -> 43 ops per detector call, 127 -> 51 ops and 8 -> 0 syncs per terms call
  (`scripts/microbench_wadsleyDetector.py`).
* `audit_monaghan_sod2d_{before,after}_2026-10-14.txt` -- the call audit of the Monaghan step (`audit_case.py sod2d --scheme Monaghan`) before and after its
  cleanup: integrator syncs 10.0 -> 5.2 per step (one Verlet validity check per stage instead of two, the Owen solve's host convergence read gone, the
  stage-time reads deferred), launches 33 -> 26, wall 8.76 -> 7.59 ms/step. Remaining syncs: 2 validity checks, ~2.2 list rebuild, 1 switch step hook (host time).
* `audit_crkSPH_sod2d_{before,after}_2026-10-14.txt` -- the CRKSPH step (`audit_case.py sod2d --scheme CRKSPH`) before and after its cleanup: integrator 904 -> 504 ops
  and 9.2 -> 3.2 syncs per step (the CRK correction terms were ~290 torch ops and 3 host reads per stage in `computeCRKTermsWarp`, now one warp kernel
  `modules/crk/terms.py`; the torch version stays for the forward-mode tangent path), wall 9.4 -> 7.6 ms/step. The remaining syncs are the Verlet validity check and rebuilds.
* `audit_{GSPH,InutsukaGSPH}_sod2d_{before,after}_2026-10-14.txt` -- the two Godunov steps before and after their cleanup (shared support solve, one validity check per
  stage, stage times deferred): GSPH integrator syncs 10.4 -> 4.2 per step, 7.05 -> 6.15 ms/step; Inutsuka 12.2 -> 8.2 syncs, wall unchanged (it still builds a second,
  larger-scale list per stage for the Gaussian reach and reads that reach on the host: 2 validity checks + 1 reach read per stage + rebuilds).
* `audit_{MFM,MFV}_sod2d_{before,after}_2026-10-14.txt` -- the meshless steps before and after the MFM / MFV split (a restructure: counts unchanged, sections now named
  `[mfm|mfv - NN]`). Per step: MFM (ssprk) 542 ops / 10 integrator syncs, MFV (hancock) 159 ops / 4 syncs, plus the timestep hook's geometry (320 ops, 9.3 syncs, in the
  core `MeshlessWarp.build`). 2 syncs per step are the geometry hand-over's host `torch.equal` (`common.py takeGeometry`); a system-owned hand-over keyed on tensor versions would remove them
  (not done: it trusts that the integrator's copy equals the stashed source).
* `audit_InutsukaGSPH_sod2d_deviceReach_2026-10-14.txt` -- Inutsuka after the Gaussian reach became a device scalar (the core's validity kernels accept a 0-d device `verletScale`): integrator syncs
  8.2 -> 6.2 per step, results bitwise equal to the host-read version, wall unchanged (~7.1 ms/step). Left: two validity checks per stage (the Owen solve's list at the default scale, then the larger-scale
  Gaussian list) + rebuilds.
