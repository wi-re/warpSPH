# Integrator ownership audit (2026-10-24)

The compressible matrix schemes use the shared `warpSPHIntegrators` lifecycle.
There were unsafe alternative entry paths and stale meshless endpoint diagnostics;
the fixes below close those paths without replacing the measured temporal maps.

| Schemes | Time advancement | Solver responsibility |
| --- | --- | --- |
| Monaghan, GSPH, InutsukaGSPH | Selected ordinary integrator | Stage derivatives; summed density, support and EOS closure |
| CompSPH, CRKSPH | Selected ordinary integrator | Stage derivatives plus compatible pair-work finalization with the driver's weights; any required replay is driver-owned |
| PESPH-E, PESPH-A | Selected ordinary integrator | Stage derivatives and final thermodynamic closure; only the selected thermal variable is integrated |
| MFM | Package `ConservedLagrangian('ssprk')` by default | Geometry, reconstruction and conserved flux rates at each driver stage |
| MFV | Package `ConservedLagrangian('hancock')` by default | Geometry and Hancock-predicted spatial flux; driver owns the conserved update and drift |
| deltaSPH, sun2017DeltaSPH | Selected ordinary integrator | Stage derivatives; explicitly staged diffusion, continuity and post-step particle-shifting policy |
| divergenceFree (VD+PS) | Selected ordinary integrator | Pressure-projected derivative plus its documented post-step particle shift |
| IISPH | Package semi-implicit Euler | Pressure solve and acceleration; integrator advances position/velocity |
| Reference DFSPH, omni, Band | Package `ProjectedStep` | Explicit full pressure-projected map; not repeated or blended as an ODE RHS |
| Artificial compressibility | Single-evaluation Euler adapter | Explicit BDF dual-time solve with genuine pseudo-time iterations; multi-stage physical-time wrappers are forbidden |
| Wave equation / experimental acoustic core | Selected ordinary integrator | Spatial derivatives |

MFM/MFV are **not arbitrary-RK-compatible**. Their selected package driver appears
in the resolved integrator metadata. Hancock evaluates a predicted flux once;
SSPRK/KDK evaluate unpredicted fluxes at the driver's distinct stage states.
This deliberate temporal contract must not be relabelled as generic RK2/RK4.
The package reports no universal ODE order for these moving-quadrature maps.

## Fixes

1. Registered MFM/MFV RHS functions no longer fall back to an internally advanced
   `(new - old) / dt` result. Calls outside their driver lifecycle fail before
   touching the state. The historical map is explicitly named `legacyMeshlessStep`
   and isolated in `modules/meshless/compatibility.py` for the G1/D4 oracle scripts.
   It retains its single-evaluation guard and is not registered as a scheme RHS.
2. The experimental MFV primitive kick previously constructed a complete new
   conserved vector and converted it back into a stage derivative in postprocess.
   Registered schemes now reject `mfvUpdate='primitive'`, at configuration and RHS
   entry. The default conserved flux and package temporal maps are retained.
3. Godunov `timeCentredEnergy=True` previously inserted `v + a dt/2` into each
   energy RHS, on top of the chosen integrator's staging. Both registered Godunov
   schemes now reject it before sampling and at direct RHS entry. Standalone
   pair-operator research APIs remain separate. The default is still false.
4. Meshless finalization used to restore the **start-of-step** primitive fields
   onto the accepted `(x,Q)`. It now recomputes primitives from the accepted
   conserved vector and geometry. The historical support update and closure warm
   start are preserved. The next timestep reuses the accepted geometry through
   the existing state-owned, version-checked handover.
5. Artificial compressibility already validated single-evaluation integration at
   runner setup, but its direct step entry did not enforce the documented guard.
   It now does. Its physical BDF algorithm and pseudo-time solve are unchanged.
6. Evaluation snapshots now also retain `Q` and `totalEnergies` when present,
   preserving the integrated state independently of primitive diagnostics.

Boundary prescriptions and post-step regularization are separate numerical
operations, not extra physical-time RK stages. Compatible-energy finalization is
also an intentional work constraint, not permission to run a hidden time loop.

## Verification

`tests/test_compressibleIntegratorContracts.py` executes the actual compressible
RHSs through the production system lifecycle and checks:

- RHS calls leave integrated positions, velocities and the active thermal or
  conserved variable unchanged; input states and physical times remain owned by
  the driver.
- Expected driver stage counts, stage times and prediction intervals.
- Both Godunov RHSs give identical rates for the same state at different trial
  timesteps, with the finite-step experiment disabled.
- Both unsafe options and hidden whole-step fallback calls fail explicitly.
- All three package meshless maps conserve total `Q`, preserve MFM particle mass,
  obey their stage flux weights, and return diagnostics consistent with accepted
  `Q` and geometry.
- Multi-step positions, supports and conserved values agree with the retained
  historical map to floating-point roundoff. This also checks that refreshing
  endpoint diagnostics does not change the established physical time map.
- The timestep consumes the accepted geometry without an extra rebuild.

Existing flux, Godunov, runner, projection-lifecycle and dual-time tests supplement
these checks. The integrators package has its own independent oscillator,
autograd and continuation tests for `ConservedLagrangian`.
Those driver gradient checks do not add AD support to the currently forward-only
meshless spatial operators; their existing explicit guards remain in force.

Final validation:

- FP32 contracts plus matrix harness: 62 passed.
- FP64 contracts: 34 passed.
- Other scheme lifecycle / projection ownership checks: 15 passed.
- Integrator-package conserved-map oscillator, gradients and continuation: 10 passed.
- Existing meshless split/flux, GSPH, Inutsuka, artificial-compressibility and
  runner regression suites completed successfully, including the Godunov suites'
  FP64 subprocess checks.
- `git diff --check` passed. Machine-readable test results are retained under
  `.tmp/integrator-audit-20261024/`.

## Retained matrix results

No matrix simulation is rerun or rewritten by this audit. Prior MFM/MFV snapshots
contain lagged primitive diagnostics and the standard final NPZ did not save
`Q`; their accepted conserved state cannot be recovered exactly from those
lagged fields. Updating those scores requires a rerun. New snapshots retain
`Q` and future runs use accepted-state diagnostics. Previous timing measurements also predate this
diagnostic refresh and its geometry handover. These tests establish integration
ownership and state consistency, not positivity on strong-shock benchmarks or
universal convergence order for every configurable limiter and boundary.
