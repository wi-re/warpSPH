# GODUNOV_SPH_PLAN: the density-jump force test (Cha, Inutsuka & Nayakshin 2010, Fig. 1), 2026-10-08

`scripts/probe_densityJumpForce.py` (pytest: `tests/test_densityJump.py`). The 1D Sod lattice with `rho = 1 | 1/ratio` (two jumps: the middle and the periodic seam),
particle pressures set to exactly `P0 = 1`, velocities zero, adaptive support. The exact acceleration is zero everywhere. Reported: the largest |a| (units `P0 / (rho h)`)
within a few spacings of a jump and in the bulk. `densityJumpForce.png` (4:1, adaptive h) and `densityJumpForce_constantH.png` (constant h).

| scheme | 2:1 at the jumps | 4:1 at the jumps | bulk |
|---|---|---|---|
| standard SPH (symmetric) | 0.906 | 1.110 | 1e-4 |
| simplified GSPH (Cha & Whitworth, layer 2), 1st and 2nd order | 0.882 | 1.240 | 1e-4 |
| Inutsuka, Gaussian `h = spacing` (**the default**), cubic V | 0.905 | 1.105 | 2e-4 |
| Inutsuka, `h = spacing`, linear V | 1.100 | 1.214 | 2e-4 |
| Inutsuka, Gaussian = density width | 0.826 | 0.774 | 5e-4 |
| Inutsuka, `h = support / 3` (about 1.3 spacings), cubic V | **0.414** | 0.808 | 4e-4 |
| Inutsuka, `h = support / 3`, linear V | 0.471 | 0.923 | 2e-4 |

## Reading

- **The simplified GSPH is the standard SPH force at constant pressure** (identical at constant `h`; asserted in the tests): it keeps the O(1) spurious jump force Cha 2010 diagnoses.
  The Godunov SPH of layer 2 cannot cure the damped-KH / contact-gap problem that way; it is a baseline.
- **Inutsuka's convolution form does vanish in the continuum.** The `V_ij^2` algebra reproduces the exact integral (`tests/test_inutsukaVolumes.py`, 1D quadrature), and the exact
  identity gives `a = 0` for a pressure-uniform field (`scripts/probe_densityJumpQuadrature.py`).
- **At the stable default (Gaussian one spacing wide, Inutsuka's own Eq. 81) the lattice force is not reduced**: 0.905 vs 0.906 at 2:1, 1.105 vs 1.110 at 4:1. The cubic volume
  interpolant matters (linear: 1.10 at 2:1, worse than SPH).
- A wider Gaussian (1.3 spacings) halves the force at 2:1 (0.41) and gains only 27 % at 4:1 (the interpolation of `1/rho` along the pair axis cannot follow a jump a few
  spacings wide). That width **pairs particles on cold shocks** (Noh, Sedov diverge: `results/inutStab2.log`, `docs/av/godunov_l3_2026-10-08/`): the Gaussian is then wider than the
  spacing so the force `a(r)` is still rising at the nearest neighbours, the condition for the pairing instability.
- So on a lattice the consistency gain exists only at a width the scheme cannot run shocks with. What the default scheme buys instead is shown in the L3 table (Gresho, KH, Sod
  velocity), which is a different mechanism (symmetrised Gaussian density, Riemann states at the interface `s*`, `V_ij^2` pair volumes) from the Fig. 1 force removal.
