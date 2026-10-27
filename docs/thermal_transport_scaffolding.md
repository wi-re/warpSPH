# Thermal transport reference scaffolding

Plan index: [driven compressible turbulence and multiphysics](../PLANS.md).

2026-10-27: independent analytic references are ready for future small physical
transport controls. They do not implement a transport operator or demonstrate
that any SPH/meshless scheme passes physical heat diffusion or conducting walls.
Turbulence campaigns remain deferred.

## Caloric and transport convention

Use explicit specific heat at constant volume c_v and u=c_v T, with no arbitrary
energy offset. For a single ideal gas, c_p=gamma c_v, R=(gamma-1)c_v and
P=rho R T=(gamma-1)rho u. Gamma alone does not specify a dimensional temperature
scale: two gases with different gamma need declared heat capacities as well.
The reference helper is not a new production EOS/material configuration.

| Quantity | SI unit | Convention |
|---|---|---|
| u | J/kg | Specific internal energy |
| c_v, c_p, R | J/(kg K) | Constant caloric properties |
| kappa | W/(m K) | Physical Fourier conductivity, nonnegative |
| chi | m²/s | kappa/(rho c_v) in the stationary, fixed-density heat equation |
| alpha | m²/s | kappa/(rho c_p), the common convection thermal diffusivity |
| nu | m²/s | Physical kinematic viscosity, separate from artificial viscosity |
| q_x | W/m² | -kappa dT/dx, positive toward increasing x |

State which diffusivity a later configuration accepts. In particular,
alpha=chi/gamma for this ideal gas; using chi in a Rayleigh or Prandtl number
that assumes alpha changes the problem. Convection definitions and a gravity-
supported reference still need to be selected explicitly.

The current references solve rho c_v dT/dt=kappa d²T/dx² with fixed rho and
constant properties, without motion or gravity. They are conduction-operator
controls, not freely evolving compressible equilibria: holding rho fixed while
T varies creates a pressure gradient in an ideal gas. A future conduction-only
harness must freeze mechanical evolution explicitly. A coupled-flow test needs
its own mechanically consistent initial/reference state.

## Independent references

[scripts/_thermalReference.py](../scripts/_thermalReference.py) uses NumPy FP64
and no solver fields, fitted rates or artificial-conductivity coefficients.
Each profile returns T, u, signed q_x, exact integrated internal energy and
cumulative boundary heat. Energies are per unit transverse area, including for
the 1D reference. Domain coordinates satisfy 0<=x<=L; time is nonnegative.
The amplitude bound is deliberately sufficient, not a complete positivity test.

With chi=kappa/(rho c_v), the available solutions are:

- Periodic: T=T_0+A exp[-chi(2pi/L)²t] sin(2pi x/L). Total heat is constant;
  individual periodic faces can carry equal nonzero fluxes.
- Insulated: T=T_0+A exp[-chi(pi/L)²t] cos(pi x/L). Both face fluxes vanish and
  total heat is constant.
- Fixed-temperature plates: T=T_left+(T_right-T_left)x/L
  +A exp[-chi(pi/L)²t] sin(pi x/L). The exact transient heat entering is
  (2 rho c_v A L/pi)(exp[-chi(pi/L)²t]-1). A=0 gives steady nonzero through-flow
  for unequal plate temperatures, with zero net heat entering.

For an interval [0,L], positive inward flux is q_left=q_x(0) and
q_right=-q_x(L). The energy change equals the integral of q_left+q_right.
Report both faces separately: net heat or energy conservation alone cannot
measure steady through-flow or Nusselt number. A future Nusselt diagnostic must
name its conductive reference and averaging convention.

## Measurements to wire up with physical transport

Use the input conductivity and heat capacity to predict decay; never fit a
rate and then use it as the acceptance reference. Retain a no-transport control,
paired timesteps, spatial refinement and an independently measured energy/flux
balance. Capture initialization, actual accepted times, raw particle fields,
source hashes, conductivity/c_v conventions, precision and native integration
policy. Set acceptance tolerances for that operator before claiming a pass.

Keep cumulative physical conductive heat, prescribed-state projection exchange,
source heat and mechanical/source work in separate output/restart fields. The
existing boundaryProjectionBudget records accepted thermal/kinetic projection
changes; it is neither an actual wall heat flux nor a complete reaction budget.
Do not infer physical q by differentiating that counter. Closed conduction-only
controls can compare internal energy with conductive plus projection/source
heat. Moving compressible cases additionally require pressure/mechanical work;
use an appropriate full energy balance there.

Next implementation gates: physical transport plus its explicit stability bound,
conduction-only decay/insulation controls, then plate fluxes, and finally gravity
balance and convection. Geometry/boundary transport adapters and scheme coverage
must be measured separately. No wall model, production transport knobs, physical
viscosity, MHD infrastructure or GPU job is added by this scaffolding.

## Validation

[tests/test_thermalReference.py](../tests/test_thermalReference.py): 22 CPU checks
pass (0.13 s pytest execution; runtime startup excluded). Independent spatial
and temporal quadrature check energy and signed boundary heat; finite differences
check the PDE and Fourier flux. Other checks cover exact mode half-lives,
zero-conductivity controls and invalid material/domain inputs. These are oracle
checks without simulation steps, video or GPU use; no solver is qualified here.

Reproduce from the worktree root:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
WARP_CACHE_PATH=/tmp/warpsph-warp-cache \
/home/lu26029/miniconda3/envs/warp/bin/python -m pytest -o addopts= -q \
    tests/test_thermalReference.py
```
