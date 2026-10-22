# Incompressible projection inventory — 21 October 2026

Scope: `divergenceFree` (local VD+PS variant), `dfsphReference`, `iisph`,
`omniIncompressible`, `band2018pb`, their common pressure operators, walls,
configuration and integration. AC is deferred by the user. This is an inventory
and architecture pass; numerical defaults below describe the current code and
are not endorsements or claims of paper fidelity.

## Sources inspected

Local author PDFs in `literature/` were read as text, including their equations
and algorithms; third-party documents are not added to Git. Primary links:

- [Cornelis et al. 2019, optimized sources](https://www.animation.rwth-aachen.de/publication/0559/), Eqs. 15–17 and implementation: VD pressure, a second density-derived position shift, velocity resampling.
- [Ihmsen et al. 2014, IISPH](https://freidok.uni-freiburg.de/data/112702), Eqs. 2, 4–6, 9–13 and Algorithm 1: symmetric pressure acceleration composed with continuity, fixed predicted-density RHS.
- [Bender & Koschier 2015, DFSPH](https://vci.rwth-aachen.de/publication/054/) and [2017 viscous extension](https://animation.rwth-aachen.de/media/papers/2016-TVCG-ViscousDFSPH.pdf): density and divergence constraints applied to velocity.
- [Band et al. 2018, pressure boundaries](https://cg.informatik.uni-freiburg.de/publications/2018_TOG_pressureBoundaries.pdf), Eqs. 8–21 and Algorithm 1: extended fluid/boundary pressure system.
- [Band et al. 2018, MLS pressure boundaries](https://doi.org/10.1016/j.cag.2018.08.001): fitted boundary pressure, a different method from solving boundary pressure unknowns.
- [Bender, Westhofen & Jeske 2023, consistent coupling](https://animation.rwth-aachen.de/media/papers/84/2023-VMV-SPH_ConsistentBoundaryHandling.pdf), static-boundary constraint/operator distinction, including Eq. 33.
- [SPlisHSPlasH DFSPH source](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/eccce86155776f6ac52d5080b1f720a52bf29450/SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp) and sibling `IISPH/TimeStepIISPH.cpp`, local commit `eccce86155776f6ac52d5080b1f720a52bf29450`.
- [omniSPH](https://github.com/wi-re/omniSPH), local commit `6f2207f8bd35c25141c689e21499098560e738a3`, `simulation/{SPH,fluidMechanics}.cpp`. Its literal implementation, not a generic DFSPH paper, is the oracle for the omni-derived route.

The reference source files were inspected locally (including the C++ loop
conditions). Web indexing could not retrieve several author PDFs or the omni
commit page; the local copies, hashes and commit identities are retained in the
evidence bundle. `evidence/source_inventory_2026-10-21.json` pins those inputs.
Spheral is a compressible solid/astro reference and has no corresponding
DFSPH/IISPH implementation here; SPlisHSPlasH is the applicable reference.

## Equation and policy inventory

Let `D(v)` be the code's scatter/difference divergence (positive for expansion),
`R(v) = -rho0 D(v)` its density-rate wrapper, and `a(p)` the symmetric pressure
acceleration with non-fluid acceleration rows zeroed. Signs refer to this code,
not a paper's opposite-sign divergence convention.

| Local route | Fixed RHS and operator | Correction and state | Boundary pressure and stopping |
|---|---|---|---|
| `divergenceFree`, shipped `DIVERGENCE_SOLVER='omni'` | Omni sources below; VD min 2/max 32/tol .01, CD min 3/max 256/tol .001 | VD into RHS acceleration. `INSTEP_CD='auto'` puts CD into velocity when gravity is active; otherwise finalize uses a density-driven position shift plus `grad(v) dx` resampling. `shifting != 'none'` disables that finalize shift. | Uses omni wall solve, plus separate VD+PS shift solver on particle walls; sources/BCs of those two are not interchangeable. No-penetration impulse precedes pressure. |
| `divergenceFree`, `DIVERGENCE_SOLVER='vdps'` | Legacy module divergence RHS `-R(v*)`, operator `dt * pressureShift(p)`; CD module `1 - (rho + dt R(v*))/rho0`, operator `dt**2 * pressureShift(p)` | Normal selected integrator advances the RHS; finalize owns PS. Historical density integration options are commented out and density is overwritten/summed. | Config-driven fixed/optimal Jacobi or Krylov; divergence mean gauge vs CD pressure clamp, source projection and optional gauge are separate. |
| `dfsphReference` | CD `s=1-rho/rho0-dt R(v*)/rho0`, `A=opDt R(a(p))/rho0`, `opDt=dt**2`. DF `s=-R(v*)/rho0`, `opDt=dt`. | Non-pressure kick, CD velocity kick, drift, rebuild/density, DF velocity kick. Raw CD pressure in `pressures`; raw DF carrier historically in `soundspeeds`, or dt-scaled carriers when damped warm start is selected. | Static Akinci particle coupling; local band scale automatically 2 if initially 1; wall pressure extrapolation off by default. Both pressure iterates nonnegative; DF deficiency thresholds 7/20 neighbors. Compression-only mean error `rho0 mean(-min(s-Ap,0))`; omega .3, budgets at least 100. |
| `iisph` | Same CD equation/diagonal as local reference, DF skipped | Same composed spatial stages and CD velocity kick/drift; post-drift density rebuilt even without DF | Same local reference wall/relaxation settings. It is not a literal port of the separate SPlisHSPlasH IISPH `d_ii/d_ij` coefficients. |
| `omniIncompressible` | DF `s=dt D(v*)`; CD `s=1-rho/rho0+dt D(v*)`; both `A=-dt**2 D(a(p))` | Same positions for both solves; accumulate non-pressure+DF+CD accelerations, optional XSPH, one velocity kick/drift. DF starts zero; CD starts .5 prior pressure. | Particle CD wall pressure Shepard by default, no per-iterate DF wall pressure. Analytic CD mirrored pressure plus hydrostatic offset/clamp or MLS; analytic DF wall disabled by default unless configured/compact. DF signed pressure, CD nonnegative except compatibility projection. Mean floor-clipped signed residual `mean(max(Ap-s,-.001))`; omega .3, DF 3 sweeps, CD 3–256. |
| `band2018pb` | `s=1-V0/V + dt D_V(v*)`, `A=-dt**2 D_V(a_V(p))` on fluid and selected boundary rows | Extended pressure vector `[p_f,p_b]`; actual volumes temporarily injected through `rho=m/V`; one kick/drift; prior warm start .5 | Boundary pressure is an unknown, not Shepard/MLS. Boundary acceleration zero. Only boundary samples with fluid contact enter; weak-diagonal rows dropped; .05 median-diagonal Tikhonov, auto constant-mode gauge, local base relaxation .1 scaled by V0. Same floor-clipped signed statistic, min 8/max 256. |

The time factors, pressure normalization, and residual units must be visible
at a caller boundary. In particular a DF tolerance cannot be copied from omni
into the reference route simply because both functions say “divergence”.
The local reference factor is `|sum_all V_j gradW|^2 + sum_all |V_j gradW|^2`
on fluid queries: the sum-of-squares includes static boundary neighbors.
The pinned SPlisHSPlasH factor restricts that back-reaction sum to fluid
neighbors. An old `_factor` wrapper docstring claimed the opposite of the
actual kernel; it is corrected here. Moreover, the volume-only factor is
invariant when masses and densities scale together, whereas this code's
physical-pressure acceleration/true diagonal scale as `1/rho0` in uniform
bulk. Reference names such as “kappa” must not imply its normalization is
identical to upstream's `pressure_rho2` field. That scaling mismatch needs
separate correction/normalization evidence before any physical-density
validation is claimed.

The old legacy constant-density module header also claimed a 0.9 predicted-
density floor that is absent from its current implementation. Its documentation
now reflects the actual unclipped `1-rhoStar/rho0` source.

Source assembly precedes per-solver row masks, deficiency exemptions,
compatibility projection and regularization. Arithmetic grouping is retained
for bitwise source/state comparison.

## External-force and wall inventory

| Route | External acceleration before pressure | Extra velocity/position treatment |
|---|---|---|
| VD+PS | gravity + per-particle body force + forcing/mass + velocity diffusion | boundary velocity extension; mDBC/analytic no-penetration impulse; XSPH and shifting in their original locations |
| Reference DFSPH/IISPH | gravity + body force + forcing/mass + velocity diffusion | boundary velocity extension; optional legacy XSPH acceleration; configured friction/XSPH before the solve |
| Omni | gravity + body force + forcing/mass; diffusion only with `omniViscosity` | analytic friction before solve; configured XSPH after solves; optional Fickian shift before drift |
| Band | gravity + forcing/mass | optional legacy XSPH after solve; no generic body-force or viscosity call in current route |

These differences are recorded, not silently harmonized. A new unified
“all non-pressure forces” call would otherwise add physics to Band and omni.
BC enforcement also differs in ordering relative to velocity extension and
force evaluation. Keep that order until separately measured.

`kind == 1` means physical boundary samples; `kind == 2` means mDBC ghost
queries and must not become pressure unknowns or reaction particles. Consistent
coupling temporarily gives boundary particles rho0 and optional Akinci mass,
then restores the original tensor objects even on exceptions. Band uses a
separate volume injection. They are distinct model contracts.

Analytic walls add integrals to density, the gradient sum/diagonal, divergence
and pressure force. Their geometric partition and discretization differ from
omni's thick triangle slab/MLS boundary. The known first-row diagonal discrepancy
and free-surface instability remain documented in `OPEN_PROBLEMS.md` §§31–32.
Changing an alpha alone does not establish a physically valid wall solve.
Static-wall refusal for reference DFSPH/IISPH and dynamic-body refusal for omni
remain. Load accounting is a reaction diagnostic, not evidence that those
schemes integrate bodies from the recorded load.

## Mismatches, caveats and disposition

1. **Source fidelity:** local `divergenceFree` is a hybrid of VD+PS and omni,
   with case-dependent CD placement. It must not be called unqualified DFSPH.
   Keep current behavior; document the selected route.
2. **DFSPH time ordering:** pinned SPlisHSPlasH does DF before non-pressure
   forces/CD/drift. Local reference does CD/drift/rebuild/DF. This can be viewed
   as moving DF across the time-step boundary, but startup, changing dt and
   carried fields differ. Keep local ordering and stop claiming literal order.
3. **Reference defaults:** local .3 relaxation, factor/band weighting, tolerance
   units and undamped/raw carry differ from SPlisHSPlasH's .5, percentage/dt
   thresholds and capped .5 damped dt-scaled carry. Preserve; no claim of exact
   reference matching. Upstream uses rest volumes `m/rho0` in these sums;
   the local composition uses apparent volumes `m/rho`. Existing long-time
   reference instability stays open.
4. **Omni loop bounds:** pinned C++ `do { ... } while(counter++ < 3 || ...)`
   executes at least four sweeps, whereas local fixed divergence/min-density
   is three. Local relaxation .3 also differs from C++ .5. Keep local budgets;
   literal C++ parity is not a result of this architecture pass.
5. **Band volume model:** Eqs. 12/15 paper wall-volume constants are replaced
   locally by full-band kernel sums; active contact/diagonal thresholds,
   Tikhonov and auto gauge are local variants. The actual error metric is
   floor-clipped signed mean, not the absolute mean claimed in the old header.
   Correct documentation, preserve the model.
6. **Affine/nonlinear walls:** hydrostatic offsets, mirror carried pressure,
   Shepard/MLS clamps and nonnegative analytic mirror pressure can make
   `a(0) != 0` or break superposition. The experimental omni Krylov branch's
   “same linear system” claim is conditional and unsafe for these closures:
   subtracting the affine base and explicitly disabling all relevant clamps
   would require a separately validated model change. Retain as experimental;
   no promotion or new default.
7. **Legacy XSPH acceleration:** reference `_xspH` negates `interp(c v)-v interp(c)`
   and divides by support, rather than the usual positive smoothing impulse.
   It is off by default; preserve and flag rather than fold it into ordinary
   XSPH under the assumption they are equivalent.
8. **Carrier/serialization debt:** `soundspeeds` is reused for divergence
   pressure, and several case-added wall/projection options are not fields on
   the config dataclass. Checkpoint compatibility is retained; a carrier/schema
   migration needs a versioned follow-up. The shared shifting codec, region/body codecs and pressure-check schedules
   now round-trip; case-added runtime options still need a schema migration.
9. **Force/load evaluation time:** local reference books both pressure loads
   after the drift even though CD acceleration was evaluated before it. Keep
   the current diagnostic convention visible; force-time correction needs a
   separate body/load validation.
10. **Full-map schemes under multistage wrappers (corrected):** reference/omni/Band mutate
    a full dt state and suppress ordinary updates. Repeating that map under
    RK can blend/multiply full steps. The package-owned `ProjectedStep` driver now evaluates these maps once;
    direct legacy multistage use fails clearly. Ordinary single-evaluation
    Euler maps are preserved bitwise in the comparison matrix.

No numerical mismatch above is repaired merely by rearranging code. A future
adoption requires one change, a declared matched initial state, hydrostatic
pressure/density and velocity criteria, periodic divergence/energy decay,
wall impulse/reaction where applicable, resolution/dt sensitivity and cost.

## Ownership after reorganization

- `schemes/incompressible`: ordered method compositions and compatibility
  switches; IISPH has its own entry module.
- `modules/incompressible/{sources,operators}`: explicit source conventions and
  shared pressure/divergence/geometry operations.
- `modules/incompressible/{reference,omni,band,policies}`: separate pressure
  formulations and iteration policies; all use existing Core `runStationary`.
- `modules/incompressible/{pressureBoundaries,consistent,wallPressure}`:
  boundary unknowns/volume injection, Akinci coupling and extrapolation remain
  separate reusable boundary models.
- `modules/incompressible/{loads,stages}`: common analytic load/viscosity
  bookkeeping, explicit force inclusion, compatibility update and state carry.
- `modules/xsph/interpolated`: the legacy interpolation filters, with velocity
  increment versus negated/support-scaled acceleration distinguished explicitly.
- Existing `analyticBoundary`, `mdbc`, `xsph`, shifting and force modules retain
  their physical responsibilities. `warpSPHIntegrators.schemes.ProjectedStep` owns one evaluation of each full
  map; kick/drift primitives live in its mechanics driver. `ProjectedSystem`
  finalization delegates field carry, body poses and particle-shift/resampling
  modules. `DFSPHReferenceSystem` remains a compatibility alias.

Qualification and exact numerical/performance limits are recorded separately
in `QUALIFICATION_2026-10-21.md` after implementation.
