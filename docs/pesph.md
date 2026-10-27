# Pressure-based SPH

Select `scheme='PESPH'` and `variant='PressureEnergy'` (E) or
`variant='PressureEntropy'` (A) in the solver configuration. Both variants share
pressure sums, grad-h and conservative force operators, and reuse the existing
support, EOS, dissipation, reconstruction, integrator and I/O components.

```python
from warpSPH.runner import RunConfiguration

energy = RunConfiguration('sod', solver={
    'scheme': 'PESPH', 'variant': 'PressureEnergy',
})
adiabatic = RunConfiguration('linearWave', solver={
    'scheme': 'PESPH', 'variant': 'PressureEntropy', 'dissipation': False,
})
# Call .prepare() to inspect a configuration, or .run() to simulate it.
```

E advances particle total energy and derives internal energy at every stage.
A advances entropy and keeps it fixed in adiabatic flow. Accepted states refresh
their thermodynamic closure; checkpoints retain the authoritative variables and
switch history. Mass density for output and thermodynamic density are distinct.

## Supported contract

Use one positive ideal gas with a shared gamma, eager execution, and periodic or
the existing open benchmark domains. Number-density adaptive supports enforce
`n h^d = constant`, with grad-h enabled and convergence checked. Fixed supports
require `adaptiveSupportScheme='NoScheme'` and grad-h disabled. Underconverged
support solves fail explicitly. Mixed materials, background pressure, solid or
analytic walls and CUDA graph execution are rejected.

Adiabatic A passes the tested discrete force/potential, Hamiltonian and stock
Verlet/PEFRL order checks. Automatic JVP classification currently encounters the
Torch/Warp storage boundary. Composition-aware lifecycle and conservative/
dissipative RHS declarations remain future timestepper work. Fixed-step evidence
does not qualify state-dependent adaptive stepping as symplectic.

## Numerical limits and experimental controls

- Dissipative A is guarded by default. Explicit `entropyHeating='SwiftLocal'`
  enables the approximate local conversion used in historical comparisons; it
  has measured conservation defects.
- Experimental `entropyHeating='NonlocalPower'` balances continuous global
  viscous power using the discrete thermal-energy sensitivity. It requires
  conductivity off, does not reproduce every local heat rate, and does not
  guarantee exact finite-step conservation or general heat-transport entropy
  production.
- Experimental A `entropyInitialization='ThermalEnergy'` inverts a prescribed
  thermal state for `1 < gamma < 2`. The default `PressureVolume` initialization
  can underinject thermal energy in a sharply localized blast.
- Cold E midpoint predictions can lose positivity despite satisfying the signal
  CFL. The tested source-off Sedov control required CFL 0.0375; 0.15 and 0.075
  failed. No automatic positivity controller or energy floor is provided.
- Native cold Noh/shearing-Noh failures, vortex/shear damping, coarse Sedov
  errors and cost relative to CRKSPH remain limitations. Opt-in
  `velocityPairPolicy='MonotoneLimited'` improves the tested Noh positivity but
  increases KH damping. It is not a default accuracy improvement.
- The shared conductivity strength multiplier is now applied correctly.
  Opt-in thermal-only Hopkins2015 conductivity has a separate signal and
  pressure limiter; it is not a momentum-viscosity selector.

Driven fluid bands have adapter coverage but are not fully physically qualified.
Conservation, endpoint completion and visual appearance alone do not establish
accuracy. The full numerical history and references are in
[PESPH_PLAN.md](../PESPH_PLAN.md).

## Evaluation and local evidence

`scripts/probe_pesph.py` provides acoustic refinement and Sod controls.
`scripts/evaluate_pesph.py` plans, runs and reports comparisons against CRKSPH;
use `--help` and choose an explicit output root for each experiment. Defaults
write under Git-ignored `.tmp/pesph/`. Full simulations record progress and video;
the small regression loops in `tests/test_pesph*.py` check operator and adapter
contracts without rendering.

Historical visual/quantitative reports, exact evaluated-source snapshots and
movies are local evidence rather than source files. The integration preserves
them under `.tmp/pesph/docs/validation/`, with an index at
`.tmp/pesph/index.json` and a SHA-256 manifest at
`.tmp/pesph/evidence-sha256.json`. Raw runs remain in the retained original
worktree, whose location is recorded in that index. These files do not accompany
a clone of the repository.
