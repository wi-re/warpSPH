# Literature

The papers this codebase is built against. **The documents themselves are not
in this repository and must not be added to it** — they are third-party
copyrighted material. What is tracked here is metadata: this manifest,
[`references.bib`](references.bib), and [`ABSTRACTS.md`](ABSTRACTS.md).

`.gitignore` enforces it two ways: everything in `literature/` is ignored
except `*.md` and `*.bib`, and `*.pdf` is ignored repo-wide. Neither is a
substitute for care, but both survive a careless `git add -A`.

Location: `literature/` (this directory). The PDFs sit inside the working tree,
so relative paths work and no per-file approval is needed to read one.

## The four files

| file | what it holds |
|---|---|
| `MANIFEST.md` (this) | what is here, and what each paper unblocks |
| [`references.bib`](references.bib) | BibTeX. Verified against the documents and their DOI records |
| [`ABSTRACTS.md`](ABSTRACTS.md) | every abstract, verbatim — the searchable index |
| [`ADDING.md`](ADDING.md) | how to add a paper, and what a sync has to do |

**To find a paper by subject, grep `ABSTRACTS.md`, not this file.** Titles are
a poor index of what a paper actually contains; abstracts are a good one:

```
grep -i -B12 'background pressure' literature/ABSTRACTS.md
grep -i -B12 'free surface'        literature/ABSTRACTS.md
```

Each hit is preceded by the block header carrying the bib key, the filename and
the venue, so `-B12` gets you from a phrase to the file to open.

## Naming

Files are `<bibkey>_<slug>.pdf`, and the bib key is `firstauthor + year`,
suffixed when that collides (`band2018` / `band2018pb`, `bender2019vmaps` /
`bender2019micropolar`). So the filename tells you the citation key and the
citation key tells you the filename — no lookup table, and
`scripts/check_literature.py` can check the correspondence mechanically.

Renaming to it on 2026-08-29 caught three things, because the incoming files
were named for how they had been downloaded rather than for what they are:

- `koschier18_viscosity.pdf` is **Weiler** et al. — Koschier is the second
  author. It is now `weiler2018_implicit-viscosity-solver.pdf`.
- `bender16_micropolar_sca.pdf` is SCA **2017**, and `bender17_micropolar.pdf`
  is TVCG **2019**. Both years were wrong, and in the same direction.
- `unpublished_analyticBoundaries.pdf` is not unpublished. It carries the
  2024 SPHERIC Workshop running header on every page; it is now
  `winchenbach2024integrals_analytic-boundary-integrals-2d.pdf`.

Two more arrived under publisher download names
(`1-s2.0-S002199911200229X-main.pdf`, `Boundary_Handling_and_Adaptive_Time-stepping_for_P.pdf`)
and had to be identified from their front matter alone.

## What is here

116 documents. The first 39 are the curated core — every row abstracted in
`ABSTRACTS.md` and annotated for what it unblocks. The remaining 77, listed
under **Extended set** below, arrived together on 2026-08-29 and were synced by
bibliographic record only: `references.bib` has an entry for each, but they are
not abstracted, not annotated for relevance, and three scanned arrivals with no
text layer were parked in `literature/scans/` rather than indexed here.

(`dehnen2012` was promoted from the extended set to the core on 2026-09-04,
supplied by the user for the kernel-choice question behind the
`columnCollapse` Wendland2-vs-Wendland4 experiment. `sun2019` and `sun2017`
were added to the core on 2026-09-03 for
`docs/historic_plans/WCSPH_SHIFTING_PLAN.md`.)

`venue` is the **published** venue, which for an author's-version or preprint
copy is not always what that copy's own front page says. Full bibliographic
detail is in `references.bib`; the abstract of every core row is in
`ABSTRACTS.md`.

**The incompressible scheme this codebase implements**

| plan | bib key | file | venue | what it is |
|---|---|---|---|---|
| `[C]` | `cornelis2019` | `cornelis2019_optimized-source-term.pdf` | The Visual Computer 35(4) 2019 | **The paper this scheme implements** (VD+PS). |
| `[BK]` | `bender2015` | `bender2015_divergence-free-sph.pdf` | SCA 2015 | DFSPH proper. The published CFL constant. |
| `[I]` | `ihmsen2014` | `ihmsen2014_implicit-incompressible-sph.pdf` | IEEE TVCG 20(3) 2014 | IISPH -- the solver the Jacobi loop discretises. |
| `[B]` | `band2018` | `band2018_mls-pressure-boundaries.pdf` | Computers & Graphics 76 2018 | MLS pressure boundaries. |
| `[BWJ23]` | `bender2023` | `bender2023_consistent-rigid-fluid-coupling.pdf` | VMV 2023 | The derivation behind `staticBoundary`. |

**Kernel choice & the pairing instability**

| plan | bib key | file | venue | what it is |
|---|---|---|---|---|
| — | `dehnen2012` | `dehnen2012_convergence-without-pairing-instability.pdf` | MNRAS 425(2) 2012 | The Wendland-kernel-for-SPH origin paper: linear stability analysis of why Wendland kernels avoid the pairing instability at any `N_H`, where truncated B-splines do not. Why this codebase runs Wendland2 at `n_h = 4`. |

**Boundary handling and fluid-rigid coupling**

| plan | bib key | file | venue | what it is |
|---|---|---|---|---|
| — | `adami2012` | `adami2012_generalized-wall-bc.pdf` | J. Comput. Phys. 231(21) 2012 | The wall BC band2018 Eq. 3 extrapolates from, including its hydrostatic term. |
| — | `akinci2012` | `akinci2012_versatile-rigid-fluid-coupling.pdf` | ACM TOG 31(4) 2012 | The boundary volume correction. |
| — | `schechter2012` | `schechter2012_ghost-sph.pdf` | ACM TOG 31(4) 2012 | Ghost particles for free-surface density loss. |
| — | `ihmsen2010` | `ihmsen2010_pcisph-boundary-timestep.pdf` | VRIPHYS 2010 | The adaptive timestep bender2015's CFL descends from. |
| — | `band2018pb` | `band2018pb_pressure-boundaries-iisph.pdf` | ACM TOG 37(2) 2018 | The full boundary PPE band2018 abbreviates: boundary samples enter the solve as unknowns. |
| — | `gissler2019` | `gissler2019_interlinked-pressure-solvers.pdf` | ACM TOG 38(1) 2019 | Two-way coupling by a second pressure solver on the rigid particles. |
| — | `koschier2017` | `koschier2017_density-maps.pdf` | SCA 2017 | Implicit (grid-sampled) boundary density instead of boundary particles. |
| — | `bender2019vmaps` | `bender2019vmaps_volume-maps.pdf` | MIG 2019 | Volume maps -- the successor to density maps; kernel not baked into the map. |
| — | `bender2020` | `bender2020_implicit-frictional-boundaries.pdf` | IEEE TVCG 26(10) 2020 | Journal extension of volume maps, adding implicit friction at the boundary. |

**Pressure solvers, non-pressure forces, multiphase**

| plan | bib key | file | venue | what it is |
|---|---|---|---|---|
| — | `bender2017` | `bender2017_divergence-free-sph-viscous.pdf` | IEEE TVCG 23(3) 2017 | The journal DFSPH: bender2015 plus a third, implicit viscosity solver. |
| — | `weiler2018` | `weiler2018_implicit-viscosity-solver.pdf` | CGF 37(2) 2018 | Implicit viscosity; the requirements list for a physically consistent viscosity. |
| — | `jeske2023` | `jeske2023_implicit-surface-tension.pdf` | ACM TOG 43(1) 2023 | Implicit cohesion-based surface tension, strongly coupled with implicit viscosity. |
| — | `bender2017micropolar` | `bender2017micropolar_micropolar-material-model.pdf` | SCA 2017 | Micropolar model recovering vorticity lost to numerical diffusion. |
| — | `bender2019micropolar` | `bender2019micropolar_turbulent-micropolar-foam.pdf` | IEEE TVCG 25(6) 2019 | Journal extension of bender2017micropolar, adding foam generation. |
| — | `boettcher2025` | `boettcher2025_implicit-porous-flow.pdf` | ACM TOG 44(6) 2025 | Porous flow with overlapping phases; a new density estimate that permits the overlap. |
| — | `bender2026` | `bender2026_primal-sph-solver.pdf` | CGF 2026 | A primal (not dual) pressure solver: stable to 1:1000 density ratios, strongly coupled to non-pressure forces. |
| — | `adami2013` | `adami2013_transport-velocity.pdf` | J. Comput. Phys. 241 2013 | Transport velocity. Closes plan 5 Q7 (background pressure). |
| — | `sun2017` | `sun2017_delta-plus-sph-model.pdf` | Comput. Methods Appl. Mech. Engrg. 315 2017 | The δ⁺-SPH origin paper: δ-SPH diffusion + PST together. Source of `delta.py`'s shift form, `wp_deltaShift`'s tensile term, and the free-surface `n`-nulling `surfaceNormal` extends. |
| — | `sun2019` | `sun2019_consistent-particle-shifting-delta-plus-sph.pdf` | Comput. Methods Appl. Mech. Engrg. 348 2019 | Consistent (quasi-Lagrangian) δ⁺-SPH: the δu divergence terms that make the WCSPH shift volume-conserving. Reference method for `docs/historic_plans/WCSPH_SHIFTING_PLAN.md` step 2. |

**Spatial adaptivity, data structures, analytic boundaries**

| plan | bib key | file | venue | what it is |
|---|---|---|---|---|
| — | `winchenbach2016` | `winchenbach2016_constrained-neighbor-lists.pdf` | SCA 2016 | Memory-bounded neighbor lists via locally adjusted support radii. |
| — | `winchenbach2017` | `winchenbach2017_continuous-adaptivity.pdf` | ACM TOG 36(4) 2017 | Continuous (not level-based) particle sizes, with mass redistribution. |
| — | `winchenbach2019` | `winchenbach2019_multi-level-memory.pdf` | VMV 2019 | Stacked hash-map data structures for highly adaptive SPH on GPUs. |
| — | `winchenbach2020mlm` | `winchenbach2020mlm_simulating-and-rendering.pdf` | CGF 39(6) 2020 | Journal extension of winchenbach2019, adding direct ray tracing off the same structure. |
| — | `winchenbach2020` | `winchenbach2020_semi-analytic-boundaries.pdf` | ACM TOG 39(6) 2020 | Analytic particle-plane interaction, extended to arbitrary geometry via SDFs. |
| — | `winchenbach2021` | `winchenbach2021_optimized-refinement.pdf` | ACM TOG 40(1) 2021 | A discretized objective function for refinement patterns; volume ratios to 1:1,000,000. |
| — | `winchenbach2024integrals` | `winchenbach2024integrals_analytic-boundary-integrals-2d.pdf` | SPHERIC 2024 | Analytic boundary integrals over triangle meshes, with barycentric boundary quantities. |
| — | `winchenbach2025analytic` | `winchenbach2025analytic_analytic-boundary-handling-2d.pdf` | J. Comput. Phys. 555 2026 | Closed-form boundary integrals for compact polynomials over triangles, via Chebyshev polynomials and 2F1. |
| — | `winchenbach2025diffsph` | `winchenbach2025diffsph_differentiable-sph.pdf` | J. Comput. Phys. 555 2026 | The differentiable PyTorch SPH framework this codebase's schemes are ported from. |

**Machine learning on SPH**

| plan | bib key | file | venue | what it is |
|---|---|---|---|---|
| — | `winchenbach2024sfbc` | `winchenbach2024sfbc_symmetric-basis-convolutions.pdf` | ICLR 2024 | Separable-basis continuous convolutions; even/odd symmetry as the stability lever. |
| — | `winchenbach2023spheric` | `winchenbach2023spheric_hybrid-sph-ml-framework.pdf` | SPHERIC 2023 | pytorchSPH: an open-source PyTorch SPH solver built to link directly to ML models. |
| — | `winchenbach2024pmac` | `winchenbach2024pmac_taylor-green-cross-validation.pdf` | PMAC 2024 | A Taylor-Green cross-validation benchmark for ML models, on a differentiable delta+-SPH solver. |
| — | `winchenbach2024spheric` | `winchenbach2024spheric_physically-motivated-ml.pdf` | SPHERIC 2024 | Symmetry-built ML model using Chebyshev/Fourier bases and SPH-informed kernel choices. |
| — | `winchenbach2025spheric` | `winchenbach2025spheric_morinet.pdf` | SPHERIC 2025 | Mori-Zwanzig view of WCSPH vs ISPH: density as a memory term, and its timescale dependence. |

## Extended set

Background and related work one step removed from the core, added 2026-08-29. **Synced by bibliographic record only:** each has a `references.bib` entry verified against its DOI record, but no relevance note — the `what it is` column here is just the paper's title. To judge what any of these is for, read the paper (grouped below by theme for browsing; `ABSTRACTS_EXTENDED.md` is the phrase-level index). All 78 have an abstract in [`ABSTRACTS_EXTENDED.md`](ABSTRACTS_EXTENDED.md) — from Crossref or (mostly) OpenAlex's inverted index where an API carries one, otherwise sliced from the PDF text layer; see that file's header for the fidelity caveats. Three scanned arrivals with no text layer (`gingold1977`, `monaghan1992`, `monaghan1994`) are held in `literature/scans/` and are not in `references.bib` or the counts above.

**SPH foundations, kernels & surveys**

| bib key | file | venue | title |
|---|---|---|---|
| `lucy1977` | `lucy1977_a-numerical-approach-fission-hypothesis.pdf` | The Astronomical Journal 82(12), 1977 | A Numerical Approach to the Testing of the Fission Hypothesis |
| `monaghan2005` | `monaghan2005_smoothed-particle-hydrodynamics-review.pdf` | Reports on Progress in Physics 68(8), 2005 | Smoothed Particle Hydrodynamics |
| `monaghan2002` | `monaghan2002_sph-compressible-turbulence.pdf` | Monthly Notices of the Royal Astronomical Society 335(3), 2002 | SPH Compressible Turbulence |
| `monaghan1989` | `monaghan1989_on-the-problem-of-penetration.pdf` | Journal of Computational Physics 82(1), 1989 | On the Problem of Penetration in Particle Methods |
| `price2010` | `price2010_spmhd-vector-potential.pdf` | Monthly Notices of the Royal Astronomical Society 401(3), 2010 | Smoothed Particle Magnetohydrodynamics – IV. Using the Vector Potential |
| `price2012` | `price2012_sph-and-magnetohydrodynamics.pdf` | Journal of Computational Physics 231(3), 2012 | Smoothed Particle Hydrodynamics and Magnetohydrodynamics |
| `koschier2022` | `koschier2022_survey-sph-methods-computer-graphics.pdf` | Computer Graphics Forum 41(2), 2022 | A Survey on SPH Methods in Computer Graphics |
| `ihmsen2014star` | `ihmsen2014star_sph-fluids-in-computer-graphics-star.pdf` | Eurographics 2014 - State of the Art Reports (STAR), 2014 | SPH Fluids in Computer Graphics |
| `muller2003` | `muller2003_particle-based-fluid-interactive-applications.pdf` | Proceedings of the 2003 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2003 | Particle-Based Fluid Simulation for Interactive Applications |
| `desbrun1996` | `desbrun1996_smoothed-particles-deformable-bodies.pdf` | Computer Animation and Simulation '96 (Eurographics Workshop), 1996 | Smoothed Particles: A New Paradigm for Animating Highly Deformable Bodies |
| `cleary1999` | `cleary1999_conduction-modelling-using-sph.pdf` | Journal of Computational Physics 148(1), 1999 | Conduction Modelling Using Smoothed Particle Hydrodynamics |
| `kruisbrink2018` | `kruisbrink2018_sph-particle-collisions.pdf` | Journal of Applied Mathematics and Physics 6(9), 2018 | SPH Particle Collisions for the Reduction of Particle Clustering, Interface Stabilisation and Wall Modelling |

**Weakly compressible SPH, free surfaces & δ-SPH**

| bib key | file | venue | title |
|---|---|---|---|
| `becker2007` | `becker2007_weakly-compressible-sph-free-surface.pdf` | Proceedings of the 2007 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2007 | Weakly Compressible SPH for Free Surface Flows |
| `colagrossi2003` | `colagrossi2003_interfacial-flows-by-sph.pdf` | Journal of Computational Physics 191(2), 2003 | Numerical Simulation of Interfacial Flows by Smoothed Particle Hydrodynamics |
| `marrone2011` | `marrone2011_delta-sph-violent-impact-flows.pdf` | Computer Methods in Applied Mechanics and Engineering 200(13-16), 2011 | δ-SPH Model for Simulating Violent Impact Flows |
| `he2014` | `he2014_robust-simulation-sparsely-sampled-thin-features.pdf` | ACM Transactions on Graphics 34(1), 2014 | Robust Simulation of Sparsely Sampled Thin Features in SPH-Based Free Surface Flows |

**Incompressible & projection-method SPH**

| bib key | file | venue | title |
|---|---|---|---|
| `cummins1999` | `cummins1999_an-sph-projection-method.pdf` | Journal of Computational Physics 152(2), 1999 | An SPH Projection Method |
| `shao2003` | `shao2003_incompressible-sph-free-surface.pdf` | Advances in Water Resources 26(7), 2003 | Incompressible SPH Method for Simulating Newtonian and Non-Newtonian Flows with a Free Surface |
| `hu2007` | `hu2007_an-incompressible-multi-phase-sph-method.pdf` | Journal of Computational Physics 227(1), 2007 | An Incompressible Multi-Phase SPH Method |
| `he2012` | `he2012_local-poisson-sph-viscous-incompressible.pdf` | Computer Graphics Forum 31(6), 2012 | Local Poisson SPH for Viscous Incompressible Fluids |
| `kang2014` | `kang2014_incompressible-sph-divergence-free-condition.pdf` | Computer Graphics Forum 33(7), 2014 | Incompressible SPH Using the Divergence-Free Condition |
| `cornelis2014` | `cornelis2014_iisph-flip-for-incompressible-fluids.pdf` | Computer Graphics Forum 33(2), 2014 | IISPH-FLIP for Incompressible Fluids |
| `solenthaler2009` | `solenthaler2009_predictive-corrective-incompressible-sph.pdf` | ACM Transactions on Graphics 28(3), 2009 | Predictive-Corrective Incompressible SPH |
| `macklin2013` | `macklin2013_position-based-fluids.pdf` | ACM Transactions on Graphics 32(4), 2013 | Position Based Fluids |
| `bodin2011` | `bodin2011_constraint-fluids.pdf` | IEEE Transactions on Visualization and Computer Graphics 18(3), 2012 | Constraint Fluids |
| `weiler2016` | `weiler2016_projective-fluids.pdf` | Proceedings of the 9th International Conference on Motion in Games (MIG), 2016 | Projective Fluids |
| `caltagirone2015` | `caltagirone2015_kinematics-scalar-projection.pdf` | Open Journal of Fluid Dynamics 5(2), 2015 | A Kinematics Scalar Projection Method (KSP) for Incompressible Flows with Variable Density |
| `takahashi2018` | `takahashi2018_efficient-hybrid-incompressible-sph-solver.pdf` | Computer Graphics Forum 37(1), 2018 | An Efficient Hybrid Incompressible SPH Solver with Interface Handling for Boundary Conditions |
| `raveendran2011` | `raveendran2011_hybrid-smoothed-particle-hydrodynamics.pdf` | Proceedings of the 2011 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2011 | Hybrid Smoothed Particle Hydrodynamics |
| `sin2009` | `sin2009_point-based-method-incompressible-flow.pdf` | Proceedings of the 2009 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2009 | A Point-Based Method for Animating Incompressible Flow |
| `degoes2015` | `degoes2015_power-particles.pdf` | ACM Transactions on Graphics 34(4), 2015 | Power Particles: An Incompressible Fluid Solver Based on Power Diagrams |

**Viscosity, viscoelasticity & thin films**

| bib key | file | venue | title |
|---|---|---|---|
| `morris1997` | `morris1997_low-reynolds-number-incompressible-sph.pdf` | Journal of Computational Physics 136(1), 1997 | Modeling Low Reynolds Number Incompressible Flows Using SPH |
| `morris1997switch` | `morris1997switch_a-switch-to-reduce-sph-viscosity.pdf` | Journal of Computational Physics 136(1), 1997 | A Switch to Reduce SPH Viscosity |
| `peer2015` | `peer2015_implicit-viscosity-formulation-for-sph.pdf` | ACM Transactions on Graphics 34(4), 2015 | An Implicit Viscosity Formulation for SPH Fluids |
| `peer2016` | `peer2016_prescribed-velocity-gradients-viscous-sph.pdf` | IEEE Transactions on Visualization and Computer Graphics 23(12), 2017 | Prescribed Velocity Gradients for Highly Viscous SPH Fluids with Vorticity Diffusion |
| `takahashi2015` | `takahashi2015_implicit-formulation-sph-viscous-fluids.pdf` | Computer Graphics Forum 34(2), 2015 | Implicit Formulation for SPH-Based Viscous Fluids |
| `clavet2005` | `clavet2005_particle-based-viscoelastic-fluid-simulation.pdf` | Proceedings of the 2005 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2005 | Particle-Based Viscoelastic Fluid Simulation |
| `batty2012` | `batty2012_discrete-viscous-sheets.pdf` | ACM Transactions on Graphics 31(4), 2012 | Discrete Viscous Sheets |

**Surface tension, drag & multiphase interfaces**

| bib key | file | venue | title |
|---|---|---|---|
| `akinci2013` | `akinci2013_versatile-surface-tension-and-adhesion.pdf` | ACM Transactions on Graphics 32(6), 2013 | Versatile Surface Tension and Adhesion for SPH Fluids |
| `zorilla2020` | `zorilla2020_surface-tension-particle-classification-monte-carlo.pdf` | Computers 9(2), 2020 | Accelerating Surface Tension Calculation in SPH via Particle Classification and Monte Carlo Integration |
| `gissler2017` | `gissler2017_generalized-drag-force.pdf` | Computers & Graphics 69, 2017 | Generalized Drag Force for Particle-Based Simulations |
| `hu2005` | `hu2005_multi-phase-sph-macroscopic-mesoscopic.pdf` | Journal of Computational Physics 213(2), 2006 | A Multi-Phase SPH Method for Macroscopic and Mesoscopic Flows |
| `solenthaler2008` | `solenthaler2008_density-contrast-sph-interfaces.pdf` | Proceedings of the 2008 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2008 | Density Contrast SPH Interfaces |
| `muller2005` | `muller2005_particle-based-fluid-fluid-interaction.pdf` | Proceedings of the 2005 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2005 | Particle-Based Fluid-Fluid Interaction |

**Boundary handling**

| bib key | file | venue | title |
|---|---|---|---|
| `kulasegaram2004` | `kulasegaram2004_variational-contact-algorithm-rigid-boundaries.pdf` | Computational Mechanics 33(4), 2004 | A Variational Formulation Based Contact Algorithm for Rigid Boundaries in Two-Dimensional SPH Applications |
| `monaghan2009` | `monaghan2009_sph-particle-boundary-forces.pdf` | Computer Physics Communications 180(10), 2009 | SPH Particle Boundary Forces for Arbitrary Boundaries |
| `ferrand2013` | `ferrand2013_unified-semi-analytical-wall-bc.pdf` | International Journal for Numerical Methods in Fluids 71(4), 2013 | Unified Semi-Analytical Wall Boundary Conditions for Inviscid, Laminar or Turbulent Flows in the Meshless SPH Method |
| `leroy2014` | `leroy2014_unified-semi-analytical-wall-bc-2d-isph.pdf` | Journal of Computational Physics 261, 2014 | Unified Semi-Analytical Wall Boundary Conditions Applied to 2-D Incompressible SPH |
| `chiron2019` | `chiron2019_sph-3d-complex-wall-boundaries.pdf` | Computer Physics Communications 234, 2019 | Fast and Accurate SPH Modelling of 3D Complex Wall Boundaries in Viscous and Non Viscous Flows |
| `harada2007` | `harada2007_sph-in-complex-shapes.pdf` | Proceedings of the 23rd Spring Conference on Computer Graphics (SCCG), 2007 | Smoothed Particle Hydrodynamics in Complex Shapes |
| `fujisawa2015` | `fujisawa2015_efficient-boundary-handling-modified-density.pdf` | Computer Graphics Forum 34(7), 2015 | An Efficient Boundary Handling with a Modified Density Calculation for SPH |
| `band2017` | `band2017_moving-least-squares-boundaries.pdf` | Workshop on Virtual Reality Interaction and Physical Simulation (VRIPHYS), 2017 | Moving Least Squares Boundaries for SPH Fluids |
| `huber2015` | `huber2015_boundary-handling-at-cloth-fluid-contact.pdf` | Computer Graphics Forum 34(1), 2015 | Boundary Handling at Cloth-Fluid Contact |
| `yildiz2009` | `yildiz2009_multiple-boundary-tangent-method.pdf` | International Journal for Numerical Methods in Engineering 77(10), 2009 | SPH with the Multiple Boundary Tangent Method |

**Fluid-solid coupling, elastic solids & granular materials**

| bib key | file | venue | title |
|---|---|---|---|
| `becker2009` | `becker2009_direct-forcing-lagrangian-rigid-fluid-coupling.pdf` | IEEE Transactions on Visualization and Computer Graphics 15(3), 2009 | Direct Forcing for Lagrangian Rigid-Fluid Coupling |
| `solenthaler2007` | `solenthaler2007_unified-particle-model-fluid-solid.pdf` | Computer Animation and Virtual Worlds 18(1), 2007 | A Unified Particle Model for Fluid-Solid Interactions |
| `muller2004` | `muller2004_interaction-of-fluids-with-deformable-solids.pdf` | Computer Animation and Virtual Worlds 15(3-4), 2004 | Interaction of Fluids with Deformable Solids |
| `akinci2013coupling` | `akinci2013coupling_coupling-elastic-solids-with-sph-fluids.pdf` | Computer Animation and Virtual Worlds 24(3-4), 2013 | Coupling Elastic Solids with Smoothed Particle Hydrodynamics Fluids |
| `deul2014` | `deul2014_position-based-rigid-body-dynamics.pdf` | Computer Animation and Virtual Worlds 27(2), 2016 | Position-Based Rigid-Body Dynamics |
| `losasso2008` | `losasso2008_two-way-coupled-sph-particle-level-set.pdf` | IEEE Transactions on Visualization and Computer Graphics 14(4), 2008 | Two-Way Coupled SPH and Particle Level Set Fluid Simulation |
| `yang2012` | `yang2012_realtime-two-way-coupling-meshless-fem.pdf` | Computer Graphics Forum 31(7), 2012 | Realtime Two-Way Coupling of Meshless Fluids and Nonlinear FEM |
| `batty2007` | `batty2007_fast-variational-framework-solid-fluid-coupling.pdf` | ACM Transactions on Graphics 26(3), 2007 | A Fast Variational Framework for Accurate Solid-Fluid Coupling |
| `peer2017` | `peer2017_implicit-sph-linearly-elastic-solids.pdf` | Computer Graphics Forum 37(6), 2018 | An Implicit SPH Formulation for Incompressible Linearly Elastic Solids |
| `kugelstadt2021` | `kugelstadt2021_fast-corotated-elastic-sph-solids.pdf` | Proceedings of the ACM on Computer Graphics and Interactive Techniques 4(3), 2021 | Fast Corotated Elastic SPH Solids with Implicit Zero-Energy Mode Control |
| `bell2005` | `bell2005_particle-based-simulation-granular-materials.pdf` | Proceedings of the 2005 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2005 | Particle-Based Simulation of Granular Materials |
| `zhu2005` | `zhu2005_animating-sand-as-a-fluid.pdf` | ACM Transactions on Graphics 24(3), 2005 | Animating Sand as a Fluid |

**Spatial adaptivity, multi-resolution & detail**

| bib key | file | venue | title |
|---|---|---|---|
| `adams2007` | `adams2007_adaptively-sampled-particle-fluids.pdf` | ACM Transactions on Graphics 26(3), 2007 | Adaptively Sampled Particle Fluids |
| `keiser2006` | `keiser2006_multiresolution-particle-based-fluids.pdf` | Department of Computer Science, ETH Zurich, No. 520, 2006 | Multiresolution Particle-Based Fluids |
| `solenthaler2011` | `solenthaler2011_two-scale-particle-simulation.pdf` | ACM Transactions on Graphics 30(4), 2011 | Two-Scale Particle Simulation |
| `orthmann2012` | `orthmann2012_temporal-blending-for-adaptive-sph.pdf` | Computer Graphics Forum 31(8), 2012 | Temporal Blending for Adaptive SPH |
| `ando2013` | `ando2013_highly-adaptive-liquid-simulations-tet-meshes.pdf` | ACM Transactions on Graphics 32(4), 2013 | Highly Adaptive Liquid Simulations on Tetrahedral Meshes |
| `liu2021` | `liu2021_turbulent-details-vorticity-refinement.pdf` | Computer Graphics Forum 40(1), 2021 | Turbulent Details Simulation for SPH Fluids via Vorticity Refinement |

**Reference solutions, grid-based methods & performance**

| bib key | file | venue | title |
|---|---|---|---|
| `ghia1982` | `ghia1982_high-re-lid-driven-cavity-multigrid.pdf` | Journal of Computational Physics 48(3), 1982 | High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method |
| `foster1996` | `foster1996_realistic-animation-of-liquids.pdf` | Graphical Models and Image Processing 58(5), 1996 | Realistic Animation of Liquids |
| `foster2001` | `foster2001_practical-animation-of-liquids.pdf` | Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 2001 | Practical Animation of Liquids |
| `stam1995` | `stam1995_depicting-fire-gaseous-phenomena-diffusion.pdf` | Proceedings of the 22nd Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 1995 | Depicting Fire and Other Gaseous Phenomena Using Diffusion Processes |
| `ihmsen2011` | `ihmsen2011_parallel-sph-implementation-multi-core-cpus.pdf` | Computer Graphics Forum 30(1), 2011 | A Parallel SPH Implementation on Multi-Core CPUs |

## What this closes

`DFSPH_IMPROVEMENT_PLAN.md` §5 previously listed four papers as **still
unavailable** and five as read-in-full from copies that were never in this
repo. All nine are now here, along with 27 others. Against the plan's open
items:

- **`cornelis2019` — the largest.** Its Fig. 3 (sinus amplitude) and Fig. 4
  (max density) are the shear-wave reference curves, and grading `shearWave`
  against them is the one remaining step of §4 item 8. **Also wanted from it:
  the setup** — domain size, resolution, `u0`, `nu`, wavenumber, time range,
  kernel and support radius — since a curve comparison is meaningless if the
  configuration differs. Note this copy is the author's version: the figures
  are the published ones, the front matter is not.
- **`adami2013`.** Closes §5 Q7 (background pressure), open since Part 7 — a
  background pressure that is *set* rather than allowed to drift.
- **`adami2012`.** The other half of Q7, and the wall BC that `[B]` Eq. 3
  extrapolates from. Its §7.1 freefall experiment is the argument for why a
  local force balance at the wall is not optional.
- **`akinci2012`.** Should settle whether `akinciBoundaryVolume`'s divergence
  under `minShift` is this codebase misapplying a one-layer correction to a
  five-layer band — the §2 hypothesis, so far inferred rather than read.
- **`schechter2012`.** The published remedy `[BK]` §5 cites for free-surface
  density underestimation, i.e. the `rotatingSquarePatch` corner loss.
- **`ihmsen2010`.** Background. Nothing is blocked on it, but it is where
  `[BK]`'s CFL condition comes from, and §1 has now rewritten that condition
  in `[BK]`'s units once already.
- **`bender2015` / `ihmsen2014` / `band2018` / `bender2023`.** Re-obtained.
  Having them back makes the *specific* claims re-checkable, which matters
  because the plan document has retracted several readings of them.

Newly here, not previously on the wanted list, and bearing on open items:

- **`band2018pb`** — the full boundary PPE that `[B]` abbreviates. `[B]`
  recomputes boundary pressure inside the solver iteration; this one puts the
  boundary samples into the PPE as unknowns. Relevant to §1.7 and to Q7.
- **`gissler2019`** — two-way coupling by a second pressure solver over the
  rigid particles. §4 records that `[BWJ23]` Eq. 35's `f_{k<-i}` is never
  applied; this is the other published way to do it.
- **`bender2017`** — the journal DFSPH. The two-solver formulation stated at
  length rather than at SCA page limits.
- **`winchenbach2025diffsph`** — the framework this codebase's schemes are
  ported from. A reference implementation more than a literature question.

## Adding a paper

Drop the PDF in and ask for a sync:

> Sync `literature/`: reconcile the PDFs against the manifest, verify the
> BibTeX and the abstracts against the documents, and rename anything that does
> not follow `<bibkey>_<slug>.pdf`.

then

```bash
python scripts/check_literature.py
```

[`ADDING.md`](ADDING.md) has the full procedure and the reason behind each step;
the `paper-lookup` skill (`.claude/skills/paper-lookup/SKILL.md`) has the lookup
mechanics — which API, which endpoint, how to get a clean abstract out of a
two-column PDF.

## Two limits worth knowing before relying on any of this

- **Values read off a plotted figure are good to about two significant
  figures.** This project's standard is exact reproduction, so a
  figure-derived number should be recorded as approximate and labelled as
  such — never presented like a measured row. If a paper states the same
  values in text or a table, those are worth far more than the plot.
- **A scanned PDF with no text layer cannot be read reliably**, and
  `check_literature.py` cannot see into it at all — an abstract quoted from one
  would pass unverified. Prefer publisher or arXiv copies over scans. All 36
  documents here have text layers.
