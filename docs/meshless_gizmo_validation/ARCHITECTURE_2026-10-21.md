# Compressible architecture consolidation, 21 October 2026

Scope: the MFM/MFV cleanup within the compressible-solver arc. ACSPH is deferred.
The guiding rule is to extend established mechanisms where possible, and expose
small reusable components where the numerical contract genuinely differs.

| Responsibility | Owner after this pass | Contract |
|---|---|---|
| Physical time staging | `warpSPHIntegrators.schemes.ConservedLagrangian` and `drivers/mechanics/conserved_lagrangian.py` | Existing lifecycle, fresh evaluations, no spatial-physics imports |
| Choosing the temporal policy | `configurations/meshless.py` | MFM auto SSPRK; MFV auto Hancock; generic `integrationMethod()` hook in runner |
| Geometry / closure | `warpSPHCore.mfm.MeshlessWarp` | Existing volume, inverse, conditioning, guards and closure policy |
| Preparing a reusable operator context | `warpSPHCore.autograd.PreparedLauncher` | Shared structured ABI, stream ownership and AD fallback; host traversal metadata |
| Primitive conversion | `modules/eos/conserved.py` | Fused ideal-gas forward conversion, Torch reverse/JVP fallback |
| Reconstruction / slope limiting | `modules/reconstruction/meshless.py` | Hopkins limiters and Hancock prediction; no implicit replacement by another limiter |
| Riemann policy | `modules/riemann/meshless.py` | Moving-face HLLC and its existing retry chain; shared contact algebra with `solvers.py` |
| Conserved face accumulation | `modules/meshless/flux.py` | One pair function; generated grid/list entry points |
| Compatible MFV primitive correction | `modules/meshless/stages.py`, `updates.py` | `postprocess`, like CRK's compatible-update hooks; experimental and off by default |
| Shared step state and diagnostics | `MeshlessSystem.initialize/preprocess/postprocess/finalize` | Explicit per-step holder; copied diagnostic fields; historical lag retained |
| Geometry handover | `modules/meshless/geometry.py` | State-owned ticket, explicit clone lineage, source/target/domain tensor versions |

The old direct exact-delta step API remains a thin compatibility adapter. Its
recurrences now call the integrators package; the normal runner no longer hides
a complete MFM/MFV time map inside forward Euler. The runner's selection hook is
generic rather than an MFM/MFV exception that pins Euler.

The specialized position predictor is retained deliberately. The historical
SSPRK map evaluates the second flux at `x + dt/2 (v + v*)`, then drifts with
`dt/2 (v + v_new)`. Ordinary Heun on `(x,Q)` evaluates at `x + dt v` and uses
a different final position combination. Replacing one with the other during
cleanup would change the method. The package driver makes that contract
explicit through `conserved_velocity(Q)` and `wrap_positions(x)` system hooks.
It does not advertise a universal ODE order for the Hancock/KDK consumer maps.

The two existing HLLC policies also remain explicit: the SPH solver uses Toro's
PVRS wave-speed estimate; the moving-face solver starts with Gaburov and retries
Roe, PVRS and Rusanov. Sharing contact algebra is safe; replacing that policy
chain is a numerical change. Likewise, the existing limiter sign function uses
`sign(0)=+1`, while the meshless pair limiter requires `sign(0)=0`.

## Grid and neighbour-list traversal

Traversal is host metadata. Kernel factories specialize this choice without a
device read. Broader mode/order/prediction variants were measured and rejected:
the small gain did not justify their compilation and maintenance cost. The grid and list
variants share reconstruction, Riemann and pair-accumulation functions. This is
a bounded flux pilot, not a claim that every core operator has been split.

Grid qualification exposed a momentum carry bug: `mfmFlux_Func_i` zeroed its
incoming momentum sum for each cell, whereas incoming mass and energy were
retained. Both the frontend and compatibility core path now carry momentum.
Neighbour-list evaluation has a single range and is unchanged by this fix.

## Other compressible consolidation candidates

- The core's old `MeshlessWarp.rates` / Torch meshless implementation remain
  compatibility/oracle paths. The normal frontend runs the shared frontend
  modules; deleting independent references is not part of this pass.
- Other operator families can adopt the now-generic prepared launcher and
  traversal specialization after profiling. No additional launch policy is
  promoted across all solvers on the basis of one flux benchmark.
- Primitive conversion is reusable but remains an ideal-gas operation. It does
  not silently enable material EOS choices in MFM/MFV.
- Existing shifting/BiCGStab wrappers already delegate to core Krylov solvers;
  the stationary pressure loops already use `runStationary`. These are examples
  of compatibility APIs backed by shared implementations, not new solver copies.

The runner records the active driver identifier/name separately in its resolved manifest, banner and plot provenance. `SimulationConfig.integrationScheme` retains its enum contract and round-trips through the shared configuration serializer; it is not overwritten with an unregistered driver-name string. State-owned geometry tickets never consume the legacy global handover slot.

The shared ideal-gas primitive kernel and meshless diagnostic refresh enter `autograd.stream.torchWarpStream` around native launches. Caller streams and external capture use the same established boundary as structured core operators, without inserting dependencies on Warp’s previously selected stream.

Native meshless entry kernels disable unused backward generation, matching the
public `MeshlessWarp` forward-only contract. Shared reconstruction and Riemann
`wp.func` implementations remain usable by differentiable kernels; the Torch
meshless reference and shared EOS reverse/JVP fallbacks are retained. Traversal
metadata is independent of prepared-launch eligibility, so a `no_grad` call
with a gradient-marked input still selects its actual neighbour topology.
