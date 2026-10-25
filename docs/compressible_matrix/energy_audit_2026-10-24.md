# CRKSPH energy and boundary audit, 24 October 2026

The matrix flags prompted a comparison with the discussion **Plan per-material
EOS for RM** (thread `01a15366-155b-7843-893c-0bfe89011e13`), its recorded
decisions in `RICHTMYER_MESHKOV_PLAN.md`, and the exact frozen matrix sources.
The evidence identifies a Kidder accepted-state boundary bug and a Noh
periodic-seam/reconstruction interaction. It does not establish a general
inability of CRKSPH to solve these problems.

## Which earlier lessons were carried over?

The discussion deliberately retained the production defaults. The experimental
continuous entropy-work derivative was qualified in research harnesses, with
and without compatible correction, and **was not promoted to the default**.
One-sided viscous work, final-work entropy sharing and a compatible midpoint
predictor were also kept as separately declared controls. The final matching
exercise changed the final entropy fraction while retaining raw interim du/dt.

The matrix resolved `compatibleEnergy=True`, `energyScheme=CRK`, fresh midpoint
RK2, `crkEnergyPredictor=raw`, `crkEnergyPartition=native`, averaged viscous
heating, summation density, and the original Owen support policy. These are
repository defaults, rather than the whole paper-matching configuration.
The matrix did not accidentally disable compatible energy or select the new
thermal derivative. Its frozen `compSPH.py`, CRK `dudt.py`, `pairWork.py` and
`energyControl.py` match the live sources inspected before this audit's fix.

The important distinctions from the discussion remain:

- Interim thermal prediction, a standalone thermal ODE, and final compatible
  work have different contracts. Conservation of global energy proves neither
  local positivity nor thermodynamic temporal order.
- A negative intermediate RK estimate can recover in the accepted compatible
  state. The matrix positivity flag tests accepted states; it does not inspect
  internal RK stages.
- Final compatible work must respect externally prescribed boundary states.
- Existing legacy uniform-gas EOS uses `sqrt(abs(u) * gamma * (gamma-1))`.
  The explicit-material ideal-gas EOS is stricter. A finite legacy trajectory
  therefore does not prove that every intermediate EOS input was admissible.
- The CompSPH gather thermal predictor versus mean-gradient force discrepancy
  remains documented; it was not silently corrected by the research studies.

## Kidder: compatible work overwrites the prescribed boundary energy

All negative final energies in the saved CRK run belong to boundary indices
0, 1, 198 and 199; the minimum over the interior indices 10 through 189 is
positive, about 635.55. The saved volume-weighted interior density L1 is
1.22075e-6. The much larger negative energy quoted for the whole state came
from the driven boundary bands.

The native boundary condition prescribes velocity, density and internal energy
inside every RHS. The integrator then assembles the accepted state and applies
final compatible work. The old case `postStep` restored only velocity, leaving
accepted boundary energy free to accumulate incompatible work. The next RHS
restored it again, which explains why the interior solution could remain good
while saved accepted boundary values became severely negative.

A 100-step replay on the original matrix timestep grid reproduces the first
negative accepted value exactly: u=-0.5735538238 at t=0.0002242763571, index 199.
Neither the final-work entropy control nor the experimental thermal derivative
removes this event. Reapplying the full analytic boundary state removes it.
Positions, velocity, density and energy in the interior remain **bitwise
identical** to the baseline after all 100 steps.

The production Kidder post-step hook now reapplies its existing full Dirichlet
prescription at the accepted time and refreshes the EOS. This introduces no
energy floor and changes no CRK/CompSPH energy formula. A separate 100-step
replay using the actual patched case has no nonpositive accepted or RHS energy.
Regression tests cover both CRKSPH and CompSPH configurations, the analytic
boundary values and EOS, and unchanged interior evolved fields. This is bounded
validation; the full Kidder matrix has not been rerun with the patch.

## Noh: the extra periodic outer seam dominates the global flag

The native case inherits `periodic=True`. Its velocity is `-sign(x)` on
[-1,1], producing central convergence and an additional separating velocity
jump across the periodic outer seam. The first negative accepted energies are
at indices 0, 1, 254 and 255, near x=+-0.996 and +-0.988, not at the central
implosion. At that first event, both preceding RHS energy inputs were positive.
The negative values arise in the accepted compatible update.

Original FP64 replay: first accepted minimum u=-3.180800841e-6 at t=0.000121.
Using the older exactly cold initialization also produces a negative minimum
at that same first step, about -3.19055e-6. This is not newly explained by
FP32 rounding or the added 1e-8 ambient energy. Final-work entropy sharing and
the experimental thermal derivative retain this first event. Turning off AV
velocity reconstruction removes nonpositive accepted energies over the
100-step control, isolating an interaction with the outer velocity jump.

The more useful boundary control retains **identical sampled particles and
the complete original timestep schedule**, but removes across-seam neighbor
connections after sampling. It completes all **1,928 steps to t=0.6**:

| Quantity | Open-edge control |
|---|---:|
| Minimum accepted internal energy over the trajectory | 9.84576e-9 |
| Final minimum internal energy | 9.99980e-9 |
| Absolute relative post-shock density error | 0.000252448 (0.0252448%) |
| Change in that density-error estimator from the periodic run | about 2.4e-15 |

Central final position/velocity differences from the periodic run are below
6.7e-16 / 9.1e-14; central density differs by at most 2.0e-10. The central
shock result was already accurate while the global flag detected the outer
setup artifact.

Crucially, the open-edge control still has intermediate RK excursions:
1,878 of 3,856 observed RHS states contain nonpositive energy, with worst
minimum -2.71257e-5. Every **accepted** state remains positive. This directly
reproduces the earlier discussion's lesson. The control uses the existing
legacy EOS continuation and does not qualify a strict-material EOS or a new
positivity-preserving predictor.

This is a diagnostic boundary control, not a silent change to the native Noh
default or to the recorded matrix. A future Noh column should explicitly
declare its nonperiodic/outer-inflow boundary protocol and rerun every method
under that same protocol, retaining separate stage and accepted-state checks.

## Evidence and validation

- Original matrix and source archive: `.tmp/compressible-matrix-20261024-v2/`.
- Bounded controls, scripts, complete per-step/stage summaries and arrays:
  `.tmp/crk-matrix-energy-audit/`.
- Full Noh control: `noh-open-full/noh-openEdges.json` and `.npz` under that
  audit directory. Its actual duration is recorded as 1,928 steps to t=0.6;
  the harness's generic scope string says 100-step and should not be used as
  the duration field.
- Patched production Kidder replay: `fixed/kidder-base.json` and `.npz`.
- FP64 focused validation: **49 passed**, covering the new boundary regression,
  compatible Butcher sums, actual midpoint state construction, experimental
  thermal identities, independent heating audits, entropy accounting, fused
  heating versus standalone operators, and pair-work padding/AD contracts.

Existing matrix attempts and the PDF have not been relabeled or overwritten.
Their flags accurately describe the globally checked accepted states of those
original runs, but alone do not identify the failing region or mechanism.
