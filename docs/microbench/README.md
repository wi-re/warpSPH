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
