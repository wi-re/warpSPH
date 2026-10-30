# MHD scaffolding and implementation contracts

Plan index: [driven compressible turbulence and multiphysics](../PLANS.md).

2026-10-27: independent CPU reference scaffolding is implemented in
[scripts/_mhdReference.py](../scripts/_mhdReference.py). No production magnetic
state, magnetic numerical flux, divergence cleaning or MHD case is implemented.
Thermal transport and longer turbulence campaigns retain their existing status.

## Existing core and integration constraints

The local audit found hydro MFM/MFV infrastructure for moving geometry,
reconstruction, pair fluxes, conserved staging, source work and restart. It did
not find an existing magnetic implementation in src/tests. These are useful
foundations, but do not automatically provide MHD support for any scheme.

In particular, [MeshlessSystem.conserved_velocity](../src/warpSPH/systems/meshless.py),
[temporal callbacks](../src/warpSPH/modules/meshless/temporal.py), sources and
primitive readers interpret Q[:,1:-1] as momentum and Q[:,-1] as total energy.
Appending magnetic columns would corrupt those interpretations. Before integration,
choose explicit field ownership/layout and replace such positional assumptions
with named accessors. The offline oracle arrays do not establish the production
checkpoint or conserved-vector ABI.

Separate spatial dimension from vector components: planar MHD controls should
retain all three velocity and magnetic components (2.5D), with no z derivative.
The current hydro vector types follow spatial dimension, so this needs a kernel,
reconstruction, source, boundary and output audit. Two-component magnetic fields
alone cannot represent the circularly polarized wave supplied here. A 2D test
also does not establish a 3D turbulent dynamo or saturation result.

Start the first magnetic implementation on MFM/MFV. Other SPH schemes need their
own consistent induction, magnetic stress, energy/entropy updates and stability
analysis; their working thermal/source adapters are not magnetic adapters.

## Reference units and quantities

The oracles take explicit magnetic permeability mu. In physical units use mu_0;
in a declared normalized system mu=1 is valid. Keep that choice in every input
manifest/checkpoint. E_B=|B|²/(2 mu), v_A=|B|/sqrt(mu rho), and
E_total=rho u+rho |v|²/2+E_B. Thermal pressure remains (gamma-1)rho u; recover u
by subtracting both kinetic and magnetic energy from total energy. A cooling
adapter must use that thermal u, not magnetic-plus-thermal specific energy.
These conventions agree with the [PHARE dimensional MHD equations](https://phare.readthedocs.io/en/latest/theory/mhd.html).

Magnetic diffusivity eta_m has units L²/time. Electrical resistivity has different
units and equals mu eta_m for constant scalar properties; Joule heat per volume
is eta_m |curl B|²/mu. For controlled dissipative cases declare nu and eta_m:
Rm=U L/eta_m and Pm=nu/eta_m, with explicit U,L conventions. Ideal MHD with
numerical dissipation does not prescribe physical Rm or Pm. Thermal diffusivity
conventions are in the [thermal scaffolding](thermal_transport_scaffolding.md).

## Implemented independent references

| Reference | What it measures now | Later solver control |
|---|---|---|
| Magnetic energy and exact finite field increment | E_B and delta E_B=(B+delta B/2) dot delta B/mu | Energy recovery and external magnetic injection at fixed volume |
| Directional slow/Alfvén/fast wave speeds | Hydro, parallel, perpendicular and rotational limits | Characteristic speeds and MHD timestep inputs |
| Circularly polarized Alfvén wave | Exact finite-amplitude ideal solution, including bulk drift and both directions | Phase/amplitude error, dissipation, boosted and oblique propagation |
| Periodic vector-potential mode | Divergence-free field/source shape, zero mean and prescribed advection | Induction-only advection and magnetic-driving source tests |
| Resistive force-free circular mode | Exponential decay and uniform Joule heating with constant total energy | Explicit resistivity, heating and diffusion timestep controls |
| Divergence summary | Dimensional RMS, declared-scale RMS/max and signed integral | Named discrete divergence diagnostics; signed cancellation cannot hide local errors |

The Alfvén reference is cross-checked against the setup in
[Athena++ cpaw.cpp](https://github.com/PrincetonUniversity/athena/blob/main/src/pgen/cpaw.cpp).
The resistive oracle is a Cartesian helical specialization of the force-free
logic documented by [PLUTO's linear force-free test](https://plutocode.ph.unito.it/Doxygen/Test_Problems/_m_h_d_2_resistive___m_h_d_2_linear___force___free_2init_8c.html):
B=(0,a cos(kx),a sin(kx)), curl B=-k B, a=A exp(-eta_m k²t).
Here |B| and Joule heating are spatially uniform, so uniform thermal pressure
rises without a Lorentz force or pressure gradient. This particular adiabatic
periodic solution avoids the spatially varying heating caveat of more general
force-free profiles. It is an analytic reference, not a PLUTO run comparison.

The vector-potential mode uses A_z=A sin(kx x) sin(ky y), B=curl A. Prescribed
advection solves the induction subproblem, but a generic such field is not a
mechanical equilibrium. Freeze mechanics for that operator control, or define
a suitable coupled test. Using its shape as a time-dependent source is a local
reference construction, not a selected magnetic turbulence driving prescription. Analytic solenoidality
does not guarantee zero divergence under a later particle/flux discretization.

## Contracts for the remaining components

| Component | Concrete implementation/qualification contract |
|---|---|
| State and geometry | Choose authoritative magnetic representation; derive B at every fresh geometry/stage. Test changing volume and MFV mass exchange, not just fixed-volume conversion. |
| Reconstruction and flux | Reconstruct all components; select a contact-resolving MHD Riemann method compatible with MFM face motion. Test uniform states, rotations, pair antisymmetry and hydro recovery. |
| Divergence control | Choose the cleaning/constraint method with its own evolved variables, dimensions and energy accounting. Qualify transport and damping of seeded divergence, with the actual force/flux divergence estimator identified. |
| Time integration | Stage magnetic and any cleaning fields with native fresh evaluations. Include fast waves, cleaning speed and explicit resistive stability bounds; preserve replay/rejection semantics. |
| Magnetic sources | Separate induction source, mechanical force work, imposed-field changes, cleaning exchange and resistive heating. Record supplied energy using the accepted temporal quadrature. |
| Restart and output | Save magnetic/cleaning authoritative fields, mu/normalization, representation/version, all source state and counters. Compare uninterrupted and checkpoint-resumed controls. Reject incompatible checkpoints explicitly. |
| Boundaries | Begin periodic. Then define normal/tangential B, tangential electric field, velocity and cleaning-state conditions consistently; record boundary Poynting flux. Thermal fixed-u bands do not supply these. |
| Diagnostics | Magnetic/kinetic/thermal energies, minimum thermal u/P, field flux/mean, named divergence measures, spectra and transfers. Log any energy/pressure corrections; positivity floors cannot masquerade as conservation. |
| Performance and HPC | Measure bytes/particle, neighbor storage, added stage/gradient/cleaning scratch and restart size; then eager/graph behavior. Existing hydro VRAM/timing does not establish MHD capacity on either local GPU or GH200. |

For the first MFM/MFV candidate, [Hopkins and Raives §§2.2–2.2.3](https://arxiv.org/html/1505.02783#S2.SS2)
evolve volume-integrated B and a cleaning scalar, use an MHD contact-resolving
solver, and replace acoustic signal speeds by fast magnetosonic speeds. Their
MFM discussion warns against imposing zero mass flux with a simpler approximate
solver that omits the contact. Their divergence terms are tied to the discrete
flux divergence; a generic gradient diagnostic is not interchangeable. This is
a formulation to evaluate, not an adopted local algorithm. The local Spheral
src search found no matching MHD/Alfvén/Dedner implementation for comparison.

## Magnetic driving and magnetoconvection

First extend periodic mechanically forced turbulence to an initial seed/mean B.
That establishes an MHD-driven flow without simultaneously inventing a magnetic
energy source. A later externally driven induction source should be solenoidal
by construction and have an explicit physical mechanism, correlation time,
normalization, mean-flux policy and power budget. The exact finite-increment
energy reference includes a quadratic term: B dot delta B alone misses energy
injected into initially zero field. It is a fixed-volume source identity; volume
change and coupled force/flux work require additional accounting. Do not label
an arbitrary acceleration as magnetic driving. MRI, shearing boxes and boundary
current injection each need separate equations and boundary controls.

Magnetoconvection joins two prerequisites: qualified thermal convection and
qualified MHD. Specify field orientation, gravity/stratification, nu, conductivity,
heat capacities, eta_m and magnetic/velocity/thermal boundary conditions. Choose
dimensionless definitions for that compressible setup before assigning control
numbers or onset targets. An imposed-field Chandrasekhar number needs declared
reference density, length and transport conventions; it is not automatically
an onset law for a stratified compressible layer.

Define perfectly conducting, imposed vertical-field and insulating/exterior
magnetic boundaries as distinct problems. A conducting wall constrains the
wall-frame tangential electric field; its normal field/flux depends on the
physical setup and initial flux, rather than a universal reflection rule.
An insulating exterior can require a matched potential field. Athena++ likewise
[warns that its reflecting magnetic boundary assumes zero normal face field](https://github.com/PrincetonUniversity/athena/wiki/Boundary-Conditions).
No universal magnetic wall adapter is added here.

Next coupled controls: zero-field recovery of thermal convection; uniform imposed
field on a stable conductive layer; field-orientation and magnetic suppression/
modification of convection; separate face heat and Poynting fluxes and the full
energy budget. Low magnetic Reynolds cases and fully evolving induction must be
named separately if both are supported. Chaotic magnetoconvection comes later.

## Validation and stopping point

[tests/test_mhdReference.py](../tests/test_mhdReference.py): 26 CPU checks pass.
The final combined MHD/thermal reference suite passes all 48 checks in 0.16 s
pytest execution (runtime startup excluded). Independent continuum flux
assembly and finite differences test the full ideal-MHD wave residual. Other
checks cover force-free balance, resistive diffusion, Joule heating, exact finite
magnetic energy changes, wave-speed limits/rotation, periodic advection, mean-flux/source-energy balance, local
versus signed divergence and invalid inputs. No GPU, simulation steps or video.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
WARP_CACHE_PATH=/tmp/warpsph-warp-cache \
/home/lu26029/miniconda3/envs/warp/bin/python -m pytest -o addopts= -q \
    tests/test_mhdReference.py
```

**Keep:** the independent references and measurement contracts. **Open:** every
production MHD path, cleaning/wall/source adapter and solver benchmark. No scheme
has been MHD-qualified by these oracle checks. The next magnetic work is a small
state/geometry/energy integration slice and zero-field recovery, followed by
waves, divergence controls and shocks, before driven magnetic flows.
