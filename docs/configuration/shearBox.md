# shearBox configuration

Periodic differential-shear box (2D), compressible: steady shear, no shock (AV_PLAN Phase 5A).

Solver: `Monaghan`.

Generated from the runtime parameter contract. Values use the case's chosen units.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `schemaVersion` | constant | `1` |  |
| `case` | constant | `"shearBox"` |  |
| `variant` | string | `"default"` |  Constraints: {'enum': ['default']}. |
| `simulation.caseName` | string | `"shearBox"` | name used for the export directory and the banner |
| `simulation.nx` | integer | `64` | particles across the domain at the base resolution Constraints: {'exclusiveMinimum': 0}. |
| `simulation.dim` | integer | `2` | Dimension supported by this initial-condition sampler. Constraints: {'enum': [2]}. |
| `simulation.L` | number | `1.0` | domain edge length Constraints: {'exclusiveMinimum': 0}. |
| `simulation.n_h` | number | `4.0` | neighbours per smoothing length; converted to targetNeighbors Constraints: {'exclusiveMinimum': 0}. |
| `simulation.calibrateNormalization` | boolean | `false` | scale the kernel by 1/L so a perfect lattice reads rho0 (summation density only; LATTICE_DENSITY_PLAN.md); off by default |
| `simulation.densityCorrection` | boolean | `false` | subtract eps m W(0,h) from the raw density (Dehnen & Aly 2012 eq. 18/19, constants from warpSPHCore.util.densityCorrection); off by default |
| `simulation.periodic` | boolean | `true` | wrap the domain at its edges |
| `simulation.kernel` | string | `"B7"` | SPH kernel function Constraints: {'enum': ['Poly6', 'CubicSpline', 'QuarticSpline', 'QuinticSpline', 'B7', 'B8', 'HOCT4', 'Gaussian', 'Wendland2', 'Wendland4', 'Wendland6', 'Spiky', 'ViscosityKernel', 'CohesionKernel', 'AdhesionKernel']}. |
| `simulation.integrationScheme` | string | `"rungeKutta2"` | time integrator, from warpSPHIntegrators Constraints: {'enum': ['forwardEuler', 'rungeKutta2', 'heunsMethod', 'ralston2nd', 'rungeKutta3', 'heunsMethod3rd', 'ralston3rd', 'wray3rd', 'sspRK3', 'rungeKutta4', 'rungeKutta4alt', 'nystrom5th', 'leapFrog', 'symplecticEuler', 'velocityVerlet', 'pefrl', 'vefrl', 'epec', 'epecModified', 'tvdRK3', 'tvdRK2', 'semiImplicitEuler', 'explicitEuler', 'bogackiShampine', 'dormandPrince', 'cashKarp', 'backwardEuler', 'implicitMidpoint', 'trapezoidal', 'sdirk2', 'ab2', 'ab3', 'ab4', 'ab5', 'abm2', 'abm3', 'abm4', 'newmark', 'bdf1', 'bdf2', 'imexEuler', 'trbdf2', 'bdf3', 'esdirk324l2sa', 'esdirk436l2sa', 'ark324l2sa', 'ark436l2sa', 'bdf4', 'bdf5', 'am2', 'am3', 'am4', 'rkc1', 'rkc2', 'rkl2', 'ros3p', 'etd2rk', 'exprb32', 'gaussLegendre2', 'radauIia2', 'sbdf2', 'sbdf3', 'cnab2', 'ssprk104', 'ssprk53', 'ssprk63', 'ssprk73', 'ssprk83', 'ssprk54', 'kykssprk42', 'kyk2014dgssprk3s2', 'ssprk43', 'ssprk432', 'ssprk932', 'ssprkmsvs32', 'ssprkmsvs43', 'ssprk53_2n1', 'ssprk53_2n2', 'ssprk53_h', 'tsit5', 'owrenzen3', 'owrenzen4', 'owrenzen5', 'bs5', 'stefanov5', 'arushina2', 'arushina3', 'arushina6', 'msrk5', 'msrk6', 'sir54', 'raston4', 'rko65', 'rkm', 'psrk4p7q6', 'psrk3p6q5', 'psrk3p5q4', 'vern6', 'vern7', 'vern8', 'vern9', 'rkv76iia', 'tanyam7', 'tsitpap8', 'dp8', 'feagin10', 'feagin12', 'feagin14', 'kvaerno3', 'kvaerno4', 'kvaerno5', 'kencarp3', 'kencarp4', 'kencarp47', 'kencarp5', 'kencarp58', 'esdirk325l2sa', 'esdirk436l2sa2', 'esdirk437l2sa', 'esdirk547l2sa2', 'esdirk54i8l2sa', 'esdirk659l2sa', 'sspsdirk2', 'ars222', 'ars232', 'ars343', 'ars443', 'bhr553', 'cash4', 'cfnlirk3', 'hairer4', 'hairer42', 'sfstdirk4', 'sfstdirk5', 'sfstdirk6', 'sfstdirk7', 'sfstdirk8', 'gaussLegendre3', 'radauIia3', 'abm5', 'mebdf2', 'cnlf2']}. |
| `simulation.supportMode` | string | `"KernelMeanSymmetric"` | how a pair's two support radii combine into one Constraints: {'enum': ['Gather', 'Scatter', 'MeanSymmetric', 'KernelMeanSymmetric', 'SuperSymmetric', 'PartialSymmetric']}. |
| `simulation.gradientMode` | string | `"Difference"` | gradient discretisation Constraints: {'enum': ['Naive', 'Symmetric', 'Difference', 'Summation']}. |
| `simulation.laplacianMode` | string | `"Brookshaw"` | Laplacian discretisation Constraints: {'enum': ['Naive', 'Brookshaw', 'Dot', 'Default']}. |
| `simulation.samplingScheme` | string | `"regular"` | how the initial particles are placed Constraints: {'enum': ['regular', 'jittered', 'glass', 'optimal', 'random', 'densest']}. |
| `simulation.verletScale` | number / null | `null` | neighbour-list padding factor; unset uses the scheme's own Constraints: {'exclusiveMinimum': 0}. |
| `simulation.tLimit` | number | `2.0` | simulated time to stop at Constraints: {'exclusiveMinimum': 0}. |
| `simulation.dt` | number / null | `null` | fixed timestep; unset lets the case derive one Constraints: {'exclusiveMinimum': 0}. |
| `simulation.adaptiveDt` | boolean | `true` | recompute dt each step from the CFL condition |
| `simulation.cflFactor` | number | `0.3` | CFL safety factor, when adaptiveDt is on Constraints: {'exclusiveMinimum': 0}. |
| `simulation.minDt` | number / null | `1e-08` | floor on the adaptive timestep Constraints: {'exclusiveMinimum': 0}. |
| `simulation.maxDt` | number / null | `0.01` | ceiling on the adaptive timestep Constraints: {'exclusiveMinimum': 0}. |
| `simulation.nSteps` | integer / null | `null` | stop after this many steps instead of at tLimit Constraints: {'minimum': 0}. |
| `simulation.stallDtSteps` | integer / null | `null` | abort once dt has sat at minDt for this many consecutive steps (a stuck-dt watchdog); unset disables it |
| `simulation.stallProgress` | number / null | `null` | abort once the last 1000 steps advanced sim time by less than this fraction of tLimit (catches dt hovering just above minDt); unset disables it |
| `simulation.velocityAlarmFactor` | number / null | `100.0` | warn (the run continues) when max /v/ exceeds this multiple of the expected velocity; 0 disables it |
| `simulation.velocityScale` | number / null | `null` | expected velocity for the alarm; unset uses the case's own, else an estimate from the initial state |
| `simulation.velocityAlarmPlotInterval` | integer / null | `null` | while the velocity alarm is active, draw a frame every this many steps (needs --plot); unset leaves frames alone |
| `simulation.velocityAlarmStopRatio` | number / null | `null` | stop the run once max /v/ exceeds this multiple of the expected velocity; unset: the alarm never stops a run |
| `simulation.precision` | string | `"float32"` | scalar precision; resolved before import, so pass it to warpsph-run rather than to a case module Constraints: {'enum': ['float16', 'half', 'float32', 'single', 'float64', 'double']}. |
| `simulation.device` | string / null | `null` | torch device; unset picks cuda:0 when one is available |
| `simulation.execution` | string / null | `null` | eager, graph (CUDA-graph replay where the scheme supports it) or auto (graph below a per-scheme particle count); unset: eager Constraints: {'enum': ['eager', 'graph', 'auto', None]}. |
| `simulation.graphMaxParticles` | integer / null | `null` | execution=auto: graph below this many particles; unset uses the per-scheme table Constraints: {'exclusiveMinimum': 0}. |
| `simulation.iterationChecks` | string / null | `null` | relaxed-Jacobi loops read their convergence test back every iteration ('every') or at learned checkpoints ('adaptive', default) Constraints: {'enum': ['every', 'adaptive', None]}. |
| `simulation.krylovCheckEvery` | integer / null | `null` | Krylov solvers read their still-running flag every this many iterations; unset: 8 Constraints: {'exclusiveMinimum': 0}. |
| `simulation.cudaGraph` | boolean / null | `null` | legacy spelling of --execution graph / eager |
| `simulation.pipelineOutputs` | boolean / null | `null` | with a whole-step graph, compute step n's diagnostics while step n+1 replays (default on) |
| `simulation.plot` | boolean | `false` | draw frames while the run proceeds |
| `simulation.plotInterval` | integer | `1` | steps between drawn frames Constraints: {'exclusiveMinimum': 0}. |
| `simulation.show` | boolean | `true` | open a live window; ignored when no interactive backend exists |
| `simulation.holdPlot` | boolean | `false` | block on the final figure instead of closing it |
| `simulation.plotBackend` | string / null | `null` | matplotlib, vispy or pyVista; unset picks by dimension Constraints: {'enum': ['matplotlib', 'vispy', 'pyVista', None]}. |
| `simulation.plotBackendOptions` | object / null | `null` | plotBackendOptions |
| `simulation.asyncPlot` | boolean | `true` | asyncPlot |
| `simulation.store` | boolean | `false` | write simulation state to HDF5 |
| `simulation.storeMode` | string | `"states"` | states = one file per stored step; trajectory = one growing file Constraints: {'enum': ['states', 'trajectory']}. |
| `simulation.storeInterval` | integer | `500` | steps between stored states (storeMode=states) Constraints: {'exclusiveMinimum': 0}. |
| `simulation.exportInterval` | number | `0.002` | simulated time between frames (storeMode=trajectory) Constraints: {'exclusiveMinimum': 0}. |
| `simulation.exportRoot` | string / null | `null` | parent directory for run folders; unset uses $WARPSPH_EXPORT_ROOT, else 'export' |
| `simulation.resumeFrom` | string / null | `null` | path to a storeMode=states checkpoint .h5 to resume from (replaces the state/time case.initialConditions would set) |
| `simulation.resumeStepOffset` | integer | `0` | the checkpoint's own absolute step number, so this run's step numbering continues from there Constraints: {'minimum': 0}. |
| `simulation.video` | boolean | `false` | encode the exported frames with ffmpeg; skipped if it is missing |
| `simulation.progress` | boolean / null | `null` | show a progress bar; unset means "when a terminal is watching" |
| `simulation.verbose` | boolean | `false` | print extra detail during setup |
| `simulation.quiet` | boolean | `false` | suppress the banner, the progress bar (unless --progress) and the completion report |
| `params.gamma` | number | `1.6666666666666667` | Ideal-gas ratio of specific heats. Constraints: {'exclusiveMinimum': 1}. |
| `params.viscositySwitch` | string / null | `null` | Viscosity switch; null inherits the solver default. Used by CompSPH, CRKSPH, Monaghan. Constraints: {'enum': ['Balsara1995', 'Colagrossi2004', 'CullenDehnen2010', 'CullenHopkins', 'MorrisMonaghan1997', 'Rosswog2000', 'NoneSwitch', 'ReadHayfield2012', 'Rosswog2020', 'Sphenix2022', 'Wadsley2017', None]}. |
| `params.adaptiveSupportScheme` | string | `"Owen"` | Adaptive smoothing-support prescription. Used by CompSPH, CRKSPH, Monaghan, GSPH, InutsukaGSPH. Constraints: {'enum': ['NoScheme', 'Monaghan', 'Owen', 'NumberDensity']}. |
| `params.adaptiveSupportCorrections` | boolean | `false` | Include adaptive-support (grad-H) corrections. Used by CompSPH, CRKSPH, Monaghan, GSPH, InutsukaGSPH. |
| `params.rho0` | number | `1.0` | Reference density used in initialization/EOS setup. Units: density. Constraints: {'exclusiveMinimum': 0}. |
| `params.markerSize` | number | `4` | Legacy local plotting marker size. Constraints: {'exclusiveMinimum': 0}. |
| `params.v0` | number | `0.5` | Signed amplitude of the periodic shear velocity. Units: velocity. |
| `params.P0` | number | `1.0` | Uniform initial pressure. Units: pressure. Constraints: {'exclusiveMinimum': 0}. |
| `solver.scheme` | constant | `"Monaghan"` |  |
| `solver.diffusionParams.K` | number | `1.0` | Overall artificial-viscosity strength. Constraints: {'minimum': 0}. |
| `solver.diffusionParams.C_l` | number | `1.0` | Linear viscosity coefficient. Constraints: {'minimum': 0}. |
| `solver.diffusionParams.C_q` | number | `2.0` | Quadratic viscosity coefficient. Constraints: {'minimum': 0}. |
| `solver.diffusionParams.thermalConductivity` | number | `0.5` | Overall thermal-conductivity strength. Constraints: {'minimum': 0}. |
| `solver.diffusionParams.Cu_l` | number | `1.0` | Linear conductivity coefficient. Constraints: {'minimum': 0}. |
| `solver.diffusionParams.Cu_q` | number | `0.0` | Quadratic conductivity coefficient. Constraints: {'minimum': 0}. |
| `solver.diffusionParams.viscosityTerm` | string | `"Price2012_98"` | Artificial-viscosity formulation. Constraints: {'enum': ['Default', 'MonaghanGingold1983', 'Cleary1998', 'Monaghan1992', 'Monaghan1997a', 'Monaghan1997b', 'Dukowicz', 'Price2012_98', 'Price2012', 'Price2008', 'Wadsley2008', 'DeltaSPH', 'RiemannDissipation', 'Frontiere2017']}. |
| `solver.diffusionParams.betaMode` | string | `"Coupled"` | Quadratic coefficient coupling policy. Constraints: {'enum': ['Coupled', 'Fixed']}. |
| `solver.diffusionParams.velocityPairPolicy` | string | `"Limited"` | Pair velocity supplied to viscosity. Constraints: {'enum': ['Raw', 'Linear', 'Limited', 'BalsaraLimited']}. |
| `solver.viscositySwitchParams.scheme` | string | `"Rosswog2020"` | Artificial-viscosity switch. Constraints: {'enum': ['Balsara1995', 'Colagrossi2004', 'CullenDehnen2010', 'CullenHopkins', 'MorrisMonaghan1997', 'Rosswog2000', 'NoneSwitch', 'ReadHayfield2012', 'Rosswog2020', 'Sphenix2022', 'Wadsley2017']}. |
| `solver.viscositySwitchParams.alpha_min` | number | `0.0` | Lower switch coefficient. Constraints: {'minimum': 0}. |
| `solver.viscositySwitchParams.alpha_max` | number | `1.0` | Upper switch coefficient. Constraints: {'minimum': 0}. |
