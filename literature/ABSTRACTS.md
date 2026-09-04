# Abstracts

Grep target for `literature/`. One block per PDF in this directory: its bib key,
its filename, the full bibliographic line, and the abstract as published. Every
abstract below is quoted verbatim from the source named in its `abstract from:`
line -- either the document's own front matter or its DOI record. None is a
paraphrase, and none is a summary of the body.

The PDFs are not in the repository (see [MANIFEST.md](MANIFEST.md)); this file
and [references.bib](references.bib) are the tracked metadata.

To find a paper by what it is about:

```
grep -i -B12 'transport velocity' literature/ABSTRACTS.md
```

Two things follow from "verbatim". Publisher typos are kept -- the SPHERIC 2023
abstract really does say "GPU accelation" -- and where an abstract quotes
mathematics, the text layer's rendering of it is kept as-is rather than
prettified. The `relevance` line is the exception: it is this repository's
editorial note on why the paper is here, not part of the published abstract.

`scripts/check_literature.py` re-matches every abstract below against its PDF,
word for word, so a paraphrase that crept in later would fail the build rather
than sit here looking plausible.


## The incompressible scheme this codebase implements

### `cornelis2019`

- **file:** `cornelis2019_optimized-source-term.pdf`
- **title:** An Optimized Source Term Formulation For Incompressible SPH
- **authors:** Jens Cornelis, Jan Bender, Christoph Gissler, Markus Ihmsen and Matthias Teschner
- **venue:** *The Visual Computer* 35(4):579-590, 2019
- **doi:** [10.1007/s00371-018-1488-8](https://doi.org/10.1007/s00371-018-1488-8)
- **copy here:** author's version
- **relevance:** **The paper this scheme implements** (VD+PS).
- **abstract from:** PDF p.1

> Incompressible SPH (ISPH) is a promising concept for the pressure computation
> in SPH. It works with large timesteps and the underlying pressure Poisson
> equation (PPE) can be solved very efficiently. Still, various aspects of
> current ISPH formulations can be optimized. This paper discusses issues of the
> two standard source terms that are typically employed in PPEs, i.e. density
> invariance (DI) and velocity divergence (VD). We show that the DI source term
> suffers from significant artificial viscosity, while the VD source term
> suffers from particle disorder and volume loss. As a conclusion of these
> findings, we propose a novel source term handling. A first PPE is solved with
> the VD source term to compute a divergence-free velocity field with minimized
> artificial viscosity. To address the resulting volume error and particle
> disorder, a second PPE is solved to improve the sampling quality. The result
> of the second PPE is used for a particle shift (PS) only. The divergence-free
> velocity field - computed from the first PPE - is not changed, but only
> resampled at the updated particle positions. Thus, the proposed source term
> handling incorporates velocity divergence and particle shift (VD+PS). The
> proposed VD+PS variant does not only improve the quality of the computed
> velocity field, but also accelerates the performance of the ISPH pressure
> computation. This is illustrated for IISPH - a recent ISPH implementation -
> where a performance gain factor of 1.6 could be achieved.

### `bender2015`

- **file:** `bender2015_divergence-free-sph.pdf`
- **title:** Divergence-Free Smoothed Particle Hydrodynamics
- **authors:** Jan Bender and Dan Koschier
- **venue:** *Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA)*, pp. 147-155, 2015
- **doi:** [10.1145/2786784.2786796](https://doi.org/10.1145/2786784.2786796)
- **relevance:** DFSPH proper. The published CFL constant.
- **abstract from:** PDF p.1

> In this paper we introduce an efficient and stable implicit SPH method for the
> physically-based simulation of incompressible fluids. In the area of computer
> graphics the most efficient SPH approaches focus solely on the correction of
> the density error to prevent volume compression. However, the continuity
> equation for incompressible flow also demands a divergence-free velocity field
> which is neglected by most methods. Although a few methods consider velocity
> divergence, they are either slow or have a perceivable density fluctuation.
> Our novel method uses an efficient combination of two pressure solvers which
> enforce low volume compression (below 0.01 %) and a divergence-free velocity
> field. This can be seen as enforcing incompressibility both on position level
> and velocity level. The first part is essential for realistic physical
> behavior while the divergence-free state increases the stability significantly
> and reduces the number of solver iterations. Moreover, it allows larger time
> steps which yields a considerable performance gain since particle
> neighborhoods have to be updated less frequently. Therefore, our
> divergence-free SPH (DFSPH) approach is significantly faster and more stable
> than current state-of-the-art SPH methods for incompressible fluids. We
> demonstrate this in simulations with millions of fast moving particles.

### `ihmsen2014`

- **file:** `ihmsen2014_implicit-incompressible-sph.pdf`
- **title:** Implicit Incompressible SPH
- **authors:** Markus Ihmsen, Jens Cornelis, Barbara Solenthaler, Christopher Horvath and Matthias Teschner
- **venue:** *IEEE Transactions on Visualization and Computer Graphics* 20(3):426-435, 2014
- **doi:** [10.1109/TVCG.2013.105](https://doi.org/10.1109/TVCG.2013.105)
- **relevance:** IISPH -- the solver the Jacobi loop discretises.
- **abstract from:** PDF p.1

> We propose a novel formulation of the projection method for Smoothed Particle
> Hydrodynamics (SPH). We combine a symmetric SPH pressure force and an SPH
> discretization of the continuity equation to obtain a discretized form of the
> pressure Poisson equation (PPE). In contrast to previous projection schemes,
> our system does consider the actual computation of the pressure force. This
> incorporation improves the convergence rate of the solver. Furthermore, we
> propose to compute the density deviation based on velocities instead of
> positions as this formulation improves the robustness of the time-integration
> scheme. We show that our novel formulation outperforms previous projection
> schemes and state-of-the-art SPH methods. Large time steps and small density
> deviations of down to 0.01 percent can be handled in typical scenarios. The
> practical relevance of the approach is illustrated by scenarios with up to 40
> million SPH particles.

### `band2018`

- **file:** `band2018_mls-pressure-boundaries.pdf`
- **title:** MLS pressure boundaries for divergence-free and viscous SPH fluids
- **authors:** Stefan Band, Christoph Gissler, Andreas Peer and Matthias Teschner
- **venue:** *Computers \& Graphics* 76:37-46, 2018
- **doi:** [10.1016/j.cag.2018.08.001](https://doi.org/10.1016/j.cag.2018.08.001)
- **relevance:** MLS pressure boundaries.
- **abstract from:** PDF p.1

> In this paper we present a novel method to predict pressure values at boundary
> particles in incompressible divergence-free SPH simulations (DFSPH). Our
> approach employs Moving Least Squares (MLS) to predict the pressure at
> boundary particles. Therefore, MLS computes hyperplanes that approximate the
> pressure field at the interface between fluid and boundary particles. We
> compare this approach with three previous techniques. One previous technique
> mirrors the pressure from fluid to boundary particles. Another one
> extrapolates the pressure from fluid to boundary particles, but uses a
> gradient that is computed with Smoothed Particle Hydrodynamics (SPH). The
> third one solves a pressure Poisson equation (PPE) for boundary particles. In
> our experiments, we indicate artifacts in the three previous approaches. We
> show that these artifacts are significantly reduced with our approach
> resulting in simulation steps that can be twice as large. We motivate that
> gradient-based extrapolation is more accurate than mirroring. We further
> motivate that, due to particle deficiency at the boundary, the SPH gradient is
> error prone. This is less the case for our proposed MLS gradient. Moreover,
> our approach is computationally less expensive as solving a PPE for the
> boundary particles. We present challenging and complex scenarios to illustrate
> the capabilities of our method. In addition, we demonstrate that the proposed
> boundary handling is applicable to highly viscous fluids.

### `bender2023`

- **file:** `bender2023_consistent-rigid-fluid-coupling.pdf`
- **title:** Consistent SPH Rigid-Fluid Coupling
- **authors:** Jan Bender, Lukas Westhofen and Stefan Rhys Jeske
- **venue:** *Vision, Modeling, and Visualization (VMV)*, pp. 209-217, 2023
- **doi:** [10.2312/vmv.20231244](https://doi.org/10.2312/vmv.20231244)
- **relevance:** The derivation behind `staticBoundary`.
- **abstract from:** DOI record (10.2312/vmv.20231244)

> A common way to handle boundaries in SPH fluid simulations is to sample the
> surface of the boundary geometry using particles. These boundary particles are
> assigned the same properties as the fluid particles and are considered in the
> pressure force computation to avoid a penetration of the boundary. However,
> the pressure solver requires a pressure value for each particle. These are
> typically not computed for the boundary particles due to the computational
> overhead. Therefore, several strategies have been investigated in previous
> works to obtain boundary pressure values. A popular, simple technique is
> pressure mirroring, which mirrors the values from the fluid particles. This
> method is efficient, but may cause visual artifacts. More complex approaches
> like pressure extrapolation aim to avoid these artifacts at the cost of
> computation time. We introduce a constraint-based derivation of
> Divergence-Free SPH (DFSPH) - a common state-of-the-art pressure solver. This
> derivation gives us new insights on how to integrate boundary particles in the
> pressure solve without the need of explicitly computing boundary pressure
> values. This yields a more elegant formulation of the pressure solver that
> avoids the aforementioned problems.


## Kernel choice and the pairing instability

### `dehnen2012`

- **file:** `dehnen2012_convergence-without-pairing-instability.pdf`
- **title:** Improving Convergence in Smoothed Particle Hydrodynamics Simulations Without Pairing Instability
- **authors:** Walter Dehnen and Hossam Aly
- **venue:** *Monthly Notices of the Royal Astronomical Society* 425(2):1068-1082, 2012
- **doi:** [10.1111/j.1365-2966.2012.21439.x](https://doi.org/10.1111/j.1365-2966.2012.21439.x)
- **copy here:** arXiv preprint (`arXiv:1204.2471v2`)
- **relevance:** The origin paper for using Wendland functions as SPH smoothing kernels, and the reason this codebase's default kernel is Wendland2 at `n_h = 4` rather than a cubic/quartic B-spline. Its linear stability analysis is the standing reference for diagnosing particle pairing/clumping when a case shows it (e.g. `columnCollapse`'s post-impact `pairedFraction` growth, DFSPH_IMPROVEMENT_PLAN.md "What's realistically open") and for reasoning about a kernel-order change (Wendland2 vs Wendland4) as a lever against it.
- **abstract from:** PDF p.1

> The numerical convergence of smoothed particle hydrodynamics (SPH) can be
> severely restricted by random force errors induced by particle disorder,
> especially in shear flows, which are ubiquitous in astrophysics. The
> increase in the number NH of neighbours when switching to more extended
> smoothing kernels at fixed resolution (using an appropriate definition for
> the SPH resolution scale) is insufficient to combat these errors.
> Consequently, trading resolution for better convergence is necessary, but
> for traditional smoothing kernels this option is limited by the pairing (or
> clumping) instability. Therefore, we investigate the suitability of the
> Wendland functions as smoothing kernels and compare them with the
> traditional B-splines. Linear stability analysis in three dimensions and
> test simulations demonstrate that the Wendland kernels avoid the pairing
> instability for all NH, despite having vanishing derivative at the origin
> (disproving traditional ideas about the origin of this instability;
> instead, we uncover a relation with the kernel Fourier transform and give
> an explanation in terms of the SPH density estimator). The Wendland
> kernels are computationally more convenient than the higher-order
> B-splines, allowing large NH and hence better numerical convergence (note
> that computational costs rise sub-linear with NH). Our analysis also shows
> that at low NH the quartic spline kernel with NH ≈ 60 obtains much better
> convergence then the standard cubic spline.


## Boundary handling and fluid-rigid coupling

### `akinci2012`

- **file:** `akinci2012_versatile-rigid-fluid-coupling.pdf`
- **title:** Versatile rigid-fluid coupling for incompressible SPH
- **authors:** Nadir Akinci, Markus Ihmsen, Gizem Akinci, Barbara Solenthaler and Matthias Teschner
- **venue:** *ACM Transactions on Graphics* 31(4), Article 62, 2012
- **doi:** [10.1145/2185520.2185558](https://doi.org/10.1145/2185520.2185558)
- **relevance:** The boundary volume correction.
- **abstract from:** DOI record (10.1145/2185520.2185558)

> We propose a momentum-conserving two-way coupling method of SPH fluids and
> arbitrary rigid objects based on hydrodynamic forces. Our approach samples the
> surface of rigid bodies with boundary particles that interact with the fluid,
> preventing deficiency issues and both spatial and temporal discontinuities.
> The problem of inhomogeneous boundary sampling is addressed by considering the
> relative contribution of a boundary particle to a physical quantity. This
> facilitates not only the initialization process but also allows the simulation
> of multiple dynamic objects. Thin structures consisting of only one layer or
> one line of boundary particles, and also non-manifold geometries can be
> handled without any additional treatment. We have integrated our approach into
> WCSPH and PCISPH, and demonstrate its stability and flexibility with several
> scenarios including multiphase flow.

### `schechter2012`

- **file:** `schechter2012_ghost-sph.pdf`
- **title:** Ghost SPH for animating water
- **authors:** Hagit Schechter and Robert Bridson
- **venue:** *ACM Transactions on Graphics* 31(4), Article 61, 2012
- **doi:** [10.1145/2185520.2185557](https://doi.org/10.1145/2185520.2185557)
- **relevance:** Ghost particles for free-surface density loss.
- **abstract from:** DOI record (10.1145/2185520.2185557)

> We propose a new ghost fluid approach for free surface and solid boundary
> conditions in Smoothed Particle Hydrodynamics (SPH) liquid simulations. Prior
> methods either suffer from a spurious numerical surface tension artifact or
> drift away from the mass conservation constraint, and do not capture realistic
> cohesion of liquid to solids. Our Ghost SPH scheme resolves this with a new
> particle sampling algorithm to create a narrow layer of ghost particles in the
> surrounding air and solid, with careful extrapolation and treatment of fluid
> variables to reflect the boundary conditions. We also provide a new, simpler
> form of artificial viscosity based on XSPH. Examples demonstrate how the new
> approach captures real liquid behaviour previously unattainable by SPH with
> very little extra cost.

### `band2018pb`

- **file:** `band2018pb_pressure-boundaries-iisph.pdf`
- **title:** Pressure Boundaries for Implicit Incompressible SPH
- **authors:** Stefan Band, Christoph Gissler, Markus Ihmsen, Jens Cornelis, Andreas Peer and Matthias Teschner
- **venue:** *ACM Transactions on Graphics* 37(2), Article 14, 2018
- **doi:** [10.1145/3180486](https://doi.org/10.1145/3180486)
- **relevance:** The full boundary PPE band2018 abbreviates: boundary samples enter the solve as unknowns.
- **abstract from:** DOI record (10.1145/3180486)

> Implicit incompressible SPH (IISPH) solves a pressure Poisson equation (PPE).
> While the solution of the PPE provides pressure at fluid samples, the embedded
> boundary handling does not compute pressure at boundary samples. Instead,
> IISPH uses various approximations to remedy this deficiency. In this article,
> we illustrate the issues of these IISPH approximations. We particularly derive
> Pressure Boundaries, a novel boundary handling that overcomes previous IISPH
> issues by the computation of physically meaningful pressure values at boundary
> samples. This is basically achieved with an extended PPE. We provide a
> detailed description of the approach that focuses on additional technical
> challenges due to the incorporation of boundary samples into the PPE. We
> therefore use volume-centric SPH discretizations instead of typically used
> density-centric ones. We further analyze the properties of the proposed
> boundary handling and compare it to the previous IISPH boundary handling. In
> addition to the fact that the proposed boundary handling provides physically
> meaningful pressure and pressure gradients at boundary samples, we show
> further benefits, such as reduced pressure oscillations, improved solver
> convergence, and larger possible time steps. The memory footprint of fluid
> samples is reduced and performance gain factors of up to five compared to
> IISPH are presented.

### `gissler2019`

- **file:** `gissler2019_interlinked-pressure-solvers.pdf`
- **title:** Interlinked SPH Pressure Solvers for Strong Fluid-Rigid Coupling
- **authors:** Christoph Gissler, Andreas Peer, Stefan Band, Jan Bender and Matthias Teschner
- **venue:** *ACM Transactions on Graphics* 38(1), Article 5, 2019
- **doi:** [10.1145/3284980](https://doi.org/10.1145/3284980)
- **relevance:** Two-way coupling by a second pressure solver on the rigid particles.
- **abstract from:** DOI record (10.1145/3284980)

> We present a strong fluid-rigid coupling for Smoothed Particle Hydrodynamics
> (SPH) fluids and rigid bodies with particle-sampled surfaces. The approach
> interlinks the iterative pressure update at fluid particles with a second SPH
> solver that computes artificial pressure at rigid-body particles. The
> introduced SPH rigid-body solver models rigid-rigid contacts as artificial
> density deviations at rigid-body particles. The corresponding pressure is
> iteratively computed by solving a global formulation that is particularly
> useful for large numbers of rigid-rigid contacts. Compared to previous SPH
> coupling methods, the proposed concept stabilizes the fluid-rigid interface
> handling. It significantly reduces the computation times of SPH fluid
> simulations by enabling larger time steps. Performance gain factors of up to
> 58 compared to previous methods are presented. We illustrate the flexibility
> of the presented fluid-rigid coupling by integrating it into DFSPH, IISPH, and
> a recent SPH solver for highly viscous fluids. We further show its
> applicability to a recent SPH solver for elastic objects. Large scenarios with
> up to 90 M particles of various interacting materials and complex contact
> geometries with up to 90 k rigid-rigid contacts are shown. We demonstrate the
> competitiveness of our proposed rigid-body solver by comparing it to Bullet.

### `koschier2017`

- **file:** `koschier2017_density-maps.pdf`
- **title:** Density maps for improved SPH boundary handling
- **authors:** Dan Koschier and Jan Bender
- **venue:** *Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA)*, pp. 1-10, 2017
- **doi:** [10.1145/3099564.3099565](https://doi.org/10.1145/3099564.3099565)
- **relevance:** Implicit (grid-sampled) boundary density instead of boundary particles.
- **abstract from:** PDF p.1

> In this paper, we present the novel concept of density maps for robust
> handling of static and rigid dynamic boundaries in fluid simulations based on
> Smoothed Particle Hydrodynamics (SPH). In contrast to the vast majority of
> existing approaches, we use an implicit discretization for a continuous
> extension of the density field throughout solid boundaries. Using the novel
> representation we enhance accuracy and efficiency of density and density
> gradient evaluations in boundary regions by computationally efficient lookups
> into our density maps. The map is generated in a preprocessing step and
> discretizes the density contribution in the boundary's near-field. In
> consequence of the high regularity of the continuous boundary density field,
> we use cubic Lagrange polynomials on a narrow-band structure of a regular grid
> for discretization. This strategy not only removes the necessity to sample
> boundary surfaces with particles but also decouples the particle size from the
> number of sample points required to represent the boundary. Moreover, it
> solves the ever-present problem of particle deficiencies near the boundary. In
> several comparisons we show that the representation is more accurate than
> particle samplings, especially for smooth curved boundaries. We further
> demonstrate that our approach robustly handles scenarios with highly complex
> boundaries and even outperforms one of the most recent sampling based
> techniques.

### `bender2019vmaps`

- **file:** `bender2019vmaps_volume-maps.pdf`
- **title:** Volume Maps: An Implicit Boundary Representation for SPH
- **authors:** Jan Bender, Tassilo Kugelstadt, Marcel Weiler and Dan Koschier
- **venue:** *Motion, Interaction and Games (MIG)*, pp. 1-10, 2019
- **doi:** [10.1145/3359566.3360077](https://doi.org/10.1145/3359566.3360077)
- **relevance:** Volume maps -- the successor to density maps; kernel not baked into the map.
- **abstract from:** PDF p.1

> In this paper, we present a novel method for the robust handling of static and
> dynamic rigid boundaries in Smoothed Particle Hydrodynamics (SPH) simulations.
> We build upon the ideas of the density maps approach which has been introduced
> recently by Koschier and Bender. They precompute the density contributions of
> solid boundaries and store them on a spatial grid which can be efficiently
> queried during runtime. This alleviates the problems of commonly used boundary
> particles, like bumpy surfaces and inaccurate pressure forces near boundaries.
> Our method is based on a similar concept but we precompute the volume
> contribution of the boundary geometry and store it on a grid. This maintains
> all benefits of density maps but offers a variety of advantages which are
> demonstrated in several experiments. Firstly, in contrast to the density maps
> method we can compute derivatives in the standard SPH manner by
> differentiating the kernel function. This results in smooth pressure forces,
> even for lower map resolutions, such that precomputation times and memory
> requirements are reduced by more than two orders of magnitude compared to
> density maps. Furthermore, this directly fits into the SPH concept so that
> volume maps can be seamlessly combined with existing SPH methods. Finally, the
> kernel function is not baked into the map such that the same volume map can be
> used with different kernels. This is especially useful when we want to
> incorporate common surface tension or viscosity methods that use different
> kernels than the fluid simulation.

### `bender2020`

- **file:** `bender2020_implicit-frictional-boundaries.pdf`
- **title:** Implicit Frictional Boundary Handling for SPH
- **authors:** Jan Bender, Tassilo Kugelstadt, Marcel Weiler and Dan Koschier
- **venue:** *IEEE Transactions on Visualization and Computer Graphics* 26(10):2982-2993, 2020
- **doi:** [10.1109/TVCG.2020.3004245](https://doi.org/10.1109/TVCG.2020.3004245)
- **relevance:** Journal extension of volume maps, adding implicit friction at the boundary.
- **abstract from:** PDF p.1 (verbatim; see note)

> In this article, we present a novel method for the robust handling of static
> and dynamic rigid boundaries in Smoothed Particle Hydrodynamics (SPH)
> simulations. We build upon the ideas of the density maps approach which has
> been introduced recently by Koschier and Bender. They precompute the density
> contributions of solid boundaries and store them on a spatial grid which can
> be efficiently queried during runtime. This alleviates the problems of
> commonly used boundary particles, like bumpy surfaces and inaccurate pressure
> forces near boundaries. Our method is based on a similar concept but we
> precompute the volume contribution of the boundary geometry. This maintains
> all benefits of density maps but offers a variety of advantages which are
> demonstrated in several experiments. First, in contrast to the density maps
> method we can compute derivatives in the standard SPH manner by
> differentiating the kernel function. This results in smooth pressure forces,
> even for lower map resolutions, such that precomputation times and memory
> requirements are reduced by more than two orders of magnitude compared to
> density maps. Furthermore, this directly fits into the SPH concept so that
> volume maps can be seamlessly combined with existing SPH methods. Finally, the
> kernel function is not baked into the map such that the same volume map can be
> used with different kernels. This is especially useful when we want to
> incorporate common surface tension or viscosity methods that use different
> kernels than the fluid simulation.

### `adami2012`

- **file:** `adami2012_generalized-wall-bc.pdf`
- **title:** A generalized wall boundary condition for smoothed particle hydrodynamics
- **authors:** Stefan Adami, Xiangyu Y. Hu and Nikolaus A. Adams
- **venue:** *Journal of Computational Physics* 231(21):7057-7075, 2012
- **doi:** [10.1016/j.jcp.2012.05.005](https://doi.org/10.1016/j.jcp.2012.05.005)
- **relevance:** The wall BC band2018 Eq. 3 extrapolates from, including its hydrostatic term.
- **abstract from:** PDF p.1

> In this paper we present a new formulation of the boundary condition at static
> and moving solid walls in SPH simulations. Our general approach is both
> applicable to two and three dimensions and is very simple compared to previous
> wall boundary formulations. Based on a local force balance between wall and
> fluid particles we apply a pressure boundary condition on the solid particles
> to prevent wall penetration. This method can handle sharp corners and complex
> geometries as is demonstrated with several examples. A validation shows that
> we recover hydrostatic equilibrium conditions in a static tank, and a
> comparison of the classical dam break simulation with state-of-the-art results
> in literature shows good agreement. We simulate various problems such as the
> flow around a cylinder and the backward facing step at Re = 100 to demonstrate
> the general applicability of this new method.

### `ihmsen2010`

- **file:** `ihmsen2010_pcisph-boundary-timestep.pdf`
- **title:** Boundary handling and adaptive time-stepping for PCISPH
- **authors:** Markus Ihmsen, Nadir Akinci, Marc Gissler and Matthias Teschner
- **venue:** *Workshop on Virtual Reality Interaction and Physical Simulation (VRIPHYS)*, pp. 79-88, 2010
- **doi:** [10.2312/PE/vriphys/vriphys10/079-088](https://doi.org/10.2312/PE/vriphys/vriphys10/079-088)
- **relevance:** The adaptive timestep bender2015's CFL descends from.
- **abstract from:** PDF p.1

> We present a novel boundary handling scheme for incompressible fluids based on
> Smoothed Particle Hydrodynamics (SPH). In combination with the
> predictive-corrective incompressible SPH (PCISPH) method, the boundary
> handling scheme allows for larger time steps compared to existing solutions.
> Furthermore, an adaptive time-stepping approach is proposed. The approach
> automatically estimates appropriate time steps independent of the scenario.
> Due to its adaptivity, the overall computation time of dynamic scenarios is
> significantly reduced compared to simulations with constant time steps.


## Pressure solvers, non-pressure forces, multiphase

### `bender2017`

- **file:** `bender2017_divergence-free-sph-viscous.pdf`
- **title:** Divergence-Free SPH for Incompressible and Viscous Fluids
- **authors:** Jan Bender and Dan Koschier
- **venue:** *IEEE Transactions on Visualization and Computer Graphics* 23(3):1193-1206, 2017
- **doi:** [10.1109/TVCG.2016.2578335](https://doi.org/10.1109/TVCG.2016.2578335)
- **relevance:** The journal DFSPH: bender2015 plus a third, implicit viscosity solver.
- **abstract from:** PDF p.1

> In this paper we present a novel Smoothed Particle Hydrodynamics (SPH) method
> for the efficient and stable simulation of incompressible fluids. The most
> efficient SPH-based approaches enforce incompressibility either on position or
> velocity level. However, the continuity equation for incompressible flow
> demands to maintain a constant density and a divergence-free velocity field.
> We propose a combination of two novel implicit pressure solvers enforcing both
> a low volume compression as well as a divergence-free velocity field. While a
> compression-free fluid is essential for realistic physical behavior, a
> divergence-free velocity field drastically reduces the number of required
> solver iterations and increases the stability of the simulation significantly.
> Thanks to the improved stability, our method can handle larger time steps than
> previous approaches. This results in a substantial performance gain since the
> computationally expensive neighborhood search has to be performed less
> frequently. Moreover, we introduce a third optional implicit solver to
> simulate highly viscous fluids which seamlessly integrates into our solver
> framework. Our implicit viscosity solver produces realistic results while
> introducing almost no numerical damping. We demonstrate the efficiency,
> robustness and scalability of our method in a variety of complex simulations
> including scenarios with millions of turbulent particles or highly viscous
> materials.

### `weiler2018`

- **file:** `weiler2018_implicit-viscosity-solver.pdf`
- **title:** A Physically Consistent Implicit Viscosity Solver for SPH Fluids
- **authors:** Marcel Weiler, Dan Koschier, Magnus Brand and Jan Bender
- **venue:** *Computer Graphics Forum* 37(2):145-155, 2018
- **doi:** [10.1111/cgf.13349](https://doi.org/10.1111/cgf.13349)
- **relevance:** Implicit viscosity; the requirements list for a physically consistent viscosity.
- **abstract from:** DOI record (10.1111/cgf.13349)

> In this paper, we present a novel physically consistent implicit solver for
> the simulation of highly viscous fluids using the Smoothed Particle
> Hydrodynamics (SPH) formalism. Our method is the result of a theoretical and
> practical in-depth analysis of the most recent implicit SPH solvers for
> viscous materials. Based on our findings, we developed a list of requirements
> that are vital to produce a realistic motion of a viscous fluid. These
> essential requirements include momentum conservation, a physically meaningful
> behavior under temporal and spatial refinement, the absence of ghost forces
> induced by spurious viscosities and the ability to reproduce complex physical
> effects that can be observed in nature. On the basis of several theoretical
> analyses, quantitative academic comparisons and complex visual experiments we
> show that none of the recent approaches is able to satisfy all requirements.
> In contrast, our proposed method meets all demands and therefore produces
> realistic animations in highly complex scenarios. We demonstrate that our
> solver outperforms former approaches in terms of physical accuracy and memory
> consumption while it is comparable in terms of computational performance. In
> addition to the implicit viscosity solver, we present a method to simulate
> melting objects. Therefore, we generalize the viscosity model to a spatially
> varying viscosity field and provide an SPH discretization of the heat
> equation.

### `jeske2023`

- **file:** `jeske2023_implicit-surface-tension.pdf`
- **title:** Implicit Surface Tension for SPH Fluid Simulation
- **authors:** Stefan Rhys Jeske, Lukas Westhofen, Fabian Löschner, José Antonio Fernández-Fernández and Jan Bender
- **venue:** *ACM Transactions on Graphics* 43(1), Article 13, 2023
- **doi:** [10.1145/3631936](https://doi.org/10.1145/3631936)
- **relevance:** Implicit cohesion-based surface tension, strongly coupled with implicit viscosity.
- **abstract from:** DOI record (10.1145/3631936)

> The numerical simulation of surface tension is an active area of research in
> many different fields of application and has been attempted using a wide range
> of methods. Our contribution is the derivation and implementation of an
> implicit cohesion force based approach for the simulation of surface tension
> effects using the Smoothed Particle Hydrodynamics (SPH) method. We define a
> continuous formulation inspired by the properties of surface tension at the
> molecular scale which is spatially discretized using SPH. An adapted variant
> of the linearized backward Euler method is used for time discretization, which
> we also strongly couple with an implicit viscosity model. Finally, we extend
> our formulation with adhesion forces for interfaces with rigid objects.
> Existing SPH approaches for surface tension in computer graphics are mostly
> based on explicit time integration, thereby lacking in stability for
> challenging settings. We compare our implicit surface tension method to these
> approaches and further evaluate our model on a wider variety of complex
> scenarios, showcasing its efficacy and versatility. Among others, these
> include but are not limited to simulations of a water crown, a dripping
> faucet, and a droplet toy.

### `bender2017micropolar`

- **file:** `bender2017micropolar_micropolar-material-model.pdf`
- **title:** A micropolar material model for turbulent SPH fluids
- **authors:** Jan Bender, Dan Koschier, Tassilo Kugelstadt and Marcel Weiler
- **venue:** *Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA)*, pp. 1-8, 2017
- **doi:** [10.1145/3099564.3099578](https://doi.org/10.1145/3099564.3099578)
- **relevance:** Micropolar model recovering vorticity lost to numerical diffusion.
- **abstract from:** PDF p.1

> In this paper we introduce a novel micropolar material model for the
> simulation of turbulent inviscid fluids. The governing equations are solved by
> using the concept of Smoothed Particle Hydrodynamics (SPH). As already
> investigated in previous works, SPH fluid simulations suffer from numerical
> diffusion which leads to a lower vorticity, a loss in turbulent details and
> finally in less realistic results. To solve this problem we propose a
> micropolar fluid model. The micropolar fluid model is a generalization of the
> classical NavierStokes equations, which are typically used in computer
> graphics to simulate fluids. In contrast to the classical Navier-Stokes model,
> micropolar fluids have a microstructure and therefore consider the rotational
> motion of fluid particles. In addition to the linear velocity field these
> fluids also have a field of microrotation which represents existing vortices
> and provides a source for new ones. However, classical micropolar materials
> are viscous and the translational and the rotational motion are coupled in a
> dissipative way. Since our goal is to simulate turbulent fluids, we introduce
> a novel modified micropolar material for inviscid fluids with a
> non-dissipative coupling. Our model can generate realistic turbulences, is
> linear and angular momentum conserving, can be easily integrated in existing
> SPH simulation methods and its computational overhead is negligible.

### `bender2019micropolar`

- **file:** `bender2019micropolar_turbulent-micropolar-foam.pdf`
- **title:** Turbulent Micropolar SPH Fluids with Foam
- **authors:** Jan Bender, Dan Koschier, Tassilo Kugelstadt and Marcel Weiler
- **venue:** *IEEE Transactions on Visualization and Computer Graphics* 25(6):2284-2295, 2019
- **doi:** [10.1109/TVCG.2018.2832080](https://doi.org/10.1109/TVCG.2018.2832080)
- **relevance:** Journal extension of bender2017micropolar, adding foam generation.
- **abstract from:** PDF p.1

> In this paper we introduce a novel micropolar material model for the
> simulation of turbulent inviscid fluids. The governing equations are solved by
> using the concept of Smoothed Particle Hydrodynamics (SPH). As already
> investigated in previous works, SPH fluid simulations suffer from numerical
> diffusion which leads to a lower vorticity, a loss in turbulent details and
> finally in less realistic results. To solve this problem we propose a
> micropolar fluid model. The micropolar fluid model is a generalization of the
> classical Navier-Stokes equations, which are typically used in computer
> graphics to simulate fluids. In contrast to the classical Navier-Stokes model,
> micropolar fluids have a microstructure and therefore consider the rotational
> motion of fluid particles. In addition to the linear velocity field these
> fluids also have a field of microrotation which represents existing vortices
> and provides a source for new ones. However, classical micropolar materials
> are viscous and the translational and the rotational motion are coupled in a
> dissipative way. Since our goal is to simulate turbulent fluids, we introduce
> a novel modified micropolar material for inviscid fluids with a
> non-dissipative coupling. Our model can generate realistic turbulences, is
> linear and angular momentum conserving, can be easily integrated in existing
> SPH simulation methods and its computational overhead is negligible. Another
> important visual feature of turbulent liquids is foam. Therefore, we present a
> post-processing method which considers microrotation in the foam particle
> generation. It works completely automatic and requires only one user-defined
> parameter to control the amount of foam.

### `boettcher2025`

- **file:** `boettcher2025_implicit-porous-flow.pdf`
- **title:** Implicit Incompressible Porous Flow using SPH
- **authors:** Timna Böttcher, Lukas Westhofen, Stefan Rhys Jeske and Jan Bender
- **venue:** *ACM Transactions on Graphics* 44(6), Article 268, 2025
- **doi:** [10.1145/3763325](https://doi.org/10.1145/3763325)
- **relevance:** Porous flow with overlapping phases; a new density estimate that permits the overlap.
- **abstract from:** DOI record (10.1145/3763325)

> We present a novel implicit porous flow solver using SPH, which maintains
> fluid incompressibility and is able to model a wide range of scenarios, driven
> by strongly coupled solid-fluid interaction forces. Many previous SPH porous
> flow methods reduce particle volumes as they transition across the solid-fluid
> interface, resulting in significant stability issues. We instead allow fluid
> and solid to overlap by deriving a new density estimation. This further allows
> us to extend SPH pressure solvers to take local porosity into account and
> results in strict enforcement of incompressibility. As a result, we can
> simulate porous flow using physically consistent pressure forces between fluid
> and solid. In contrast to previous SPH porous flow methods, which use explicit
> forces for internal fluid flow, we employ implicit non-pressure forces. These
> we solve as a linear system and strongly couple with fluid viscosity and solid
> elasticity. We capture the most common effects observed in porous flow, namely
> drag, buoyancy and capillary action due to adhesion. To achieve elastic
> behavior change based on local fluid saturation, such as bloating or
> softening, we propose an extension to the elasticity model. We demonstrate the
> efficacy of our model with various simulations that showcase the different
> aspects of porous flow behavior. To summarize, our system of strongly coupled
> non-pressure forces and enforced incompressibility across overlapping phases
> allows us to naturally model and stably simulate complex porous interactions.

### `bender2026`

- **file:** `bender2026_primal-sph-solver.pdf`
- **title:** Primal SPH Solver for Strongly Coupled Multiphase Simulations with High Density Ratios
- **authors:** Jan Bender, Stefan Rhys Jeske, Timna Böttcher and Fabian Löschner
- **venue:** *Computer Graphics Forum*, Article e70559, 2026
- **doi:** [10.1111/cgf.70559](https://doi.org/10.1111/cgf.70559)
- **relevance:** A primal (not dual) pressure solver: stable to 1:1000 density ratios, strongly coupled to non-pressure forces.
- **abstract from:** DOI record (10.1111/cgf.70559)

> In recent years, the Smoothed Particle Hydrodynamics (SPH) approach has been
> increasingly used for multiphase simulations involving interactions between
> diverse materials. A critical component of an SPH simulator is the pressure
> solver, which not only facilitates the simulation of compressible or
> incompressible fluids but also handles contact by preventing penetration
> between different materials. Currently, most SPH simulations in computer
> graphics employ implicit dual pressure solvers such as PBF, IISPH, or DFSPH.
> However, these solvers often exhibit instability when simulating high density
> ratios. Furthermore, they are difficult to strongly couple with many existing
> methods for non-pressure forces, which typically utilize primal formulations.
> Consequently, pressure and non-pressure solvers are often only weakly coupled,
> which can lead to stability issues. We present a novel implicit primal SPH
> pressure solver designed for multiphase simulations. Our method enables stable
> simulation of multiple interacting materials with large density ratios. We
> show that our solver robustly handles ratios of up to 1:1000 (e.g., air-water
> interactions) which was not possible with previous implicit SPH pressure
> solvers. Moreover, we demonstrate how our solver allows for strong coupling
> with existing implicit simulation methods for viscosity, elasticity, and
> surface tension. Overall, our strong coupling significantly improves stability
> in complex multiphase simulations involving fluids, highly viscous materials,
> and deformable solids.

### `adami2013`

- **file:** `adami2013_transport-velocity.pdf`
- **title:** A transport-velocity formulation for smoothed particle hydrodynamics
- **authors:** Stefan Adami, Xiangyu Y. Hu and Nikolaus A. Adams
- **venue:** *Journal of Computational Physics* 241:292-307, 2013
- **doi:** [10.1016/j.jcp.2013.01.043](https://doi.org/10.1016/j.jcp.2013.01.043)
- **relevance:** Transport velocity. Closes plan 5 Q7 (background pressure).
- **abstract from:** PDF p.1

> The standard weakly-compressible SPH method suffers from particle clumping and
> void regions for high Reynolds number flows and when negative pressures occur
> in the flow. As a remedy, a new algorithm is proposed that combines the
> homogenization of the particle configuration by a background pressure while at
> the same time reduces artificial numerical dissipation. The transport or
> advection velocity of particles is modified and an effective stress term
> occurs in the momentum balance that accounts for the difference between
> advection velocity times particle density and actual particle momentum. The
> present formulation can be applied for internal flows where the density
> summation is applicable. A wide range of test cases demonstrates unprecedented
> accuracy and stability of the proposed modification even at previously
> infeasible conditions.

### `sun2017`

- **file:** `sun2017_delta-plus-sph-model.pdf`
- **title:** The δplus-SPH model: Simple procedures for a further improvement of the SPH scheme
- **authors:** P. N. Sun, A. Colagrossi, S. Marrone and A. M. Zhang
- **venue:** *Computer Methods in Applied Mechanics and Engineering* 315:25-49, 2017
- **doi:** [10.1016/j.cma.2016.10.028](https://doi.org/10.1016/j.cma.2016.10.028)
- **relevance:** The δ⁺-SPH origin paper — δ-SPH density diffusion + a particle-shifting technique combined, plus a free-surface treatment for the shift. Source of the `δx = −CFL·Ma·2h²·∇C` displacement `modules/shifting/delta.py` implements, the `[1 + R (W_ij/W(Δx))ⁿ]` tensile-instability term (R=0.2, n=4) in `sample/wp_deltaShift`, and the free-surface normal-nulling that `sun2019` §2.4 — and this repo's `ShiftingProjectionScheme.surfaceNormal` — extends. Ref [5] in `sun2019`.
- **abstract from:** PDF p.1

> The present work is dedicated to the improvement of the δ-SPH scheme. This is an enhanced weakly-compressible SPH model widely used in recent years thanks to its benefits to the standard SPH scheme, to its low CPU costs and to its ease of implementation. Nonetheless, the δ-SPH still presents some drawbacks as other SPH models. For example, in some critical conditions it does not prevent the tensile instability and the consequent numerical fragmentation. Furthermore, even if the use of a diffusive term in the SPH continuity equation is able to reduce numerical high frequencies on the pressure field, the velocity gradients are generally noisy because of the irregularities of the particle spatial configurations, which, in specific flow conditions, can induce also extra numerical-dissipation. For these reasons a particle shifting technique is used to improve the model and a special treatment has been developed for particles that are close to the free-surface region. The introduction of the particle-shifting procedure is generalized in the context of multi-resolutions for which a novel algorithm is formulated to handle the particle re-positioning in the different resolution levels. The proposed algorithms can be straightforwardly implemented in an SPH model without requiring cumbersome code modifications. The δ + -SPH is validated on seven different benchmarks giving a wide panorama on the improvements of this new SPH model.

### `sun2019`

- **file:** `sun2019_consistent-particle-shifting-delta-plus-sph.pdf`
- **title:** A consistent approach to particle shifting in the δ-Plus-SPH model
- **authors:** P. N. Sun, A. Colagrossi, S. Marrone, M. Antuono and A.-M. Zhang
- **venue:** *Computer Methods in Applied Mechanics and Engineering* 348:912-934, 2019
- **doi:** [10.1016/j.cma.2019.01.045](https://doi.org/10.1016/j.cma.2019.01.045)
- **relevance:** The reference method for `docs/historic_plans/WCSPH_SHIFTING_PLAN.md` step 2. Recasts δ⁺-SPH in a quasi-Lagrangian frame (advection velocity `u + δu`), which adds `δu`-divergence terms to the continuity and momentum equations that make the particle shift volume-conserving without a free-surface heuristic. The codebase's `ShiftProperties.correctdrhodt` / `correctdvdt` (both default off, unvalidated) implement its Eq. (9)-(10) continuity/momentum terms; §2.4 is the `λ<0.55` + `(I−nnᵀ)` + `15°` curvature surface treatment that `modules/shifting/wrapper.py` is a partial port of.
- **abstract from:** PDF p.1

> In the present work a consistent inclusion of a particle shifting technique (PST) in the weakly compressible Smoothed Particle Hydrodynamic (SPH) models is discussed. Recently, it has been shown that the use of PST can largely improve both the accuracy and the robustness of SPH models. In particular, the δ + -SPH model is a weakly-compressible SPH model where a PST is adopted along with a diffusive term in the continuity equation that helps removing the high-frequency noise on the pressure field. This specific SPH model is able to overcome the main drawbacks that afflict the standard weakly-compressible SPH model. In this work we demonstrate that a consistent introduction of the PST inside the SPH model leads to a new set of equations where some additional terms containing the particle shifting velocity δu have to be taken into account. The effects of these δu-terms become crucial for problems in confined or periodic domains, as well as for long-time simulations of free-surface flows. The proposed scheme is tested against challenging benchmark cases, highlighting when the δu-terms play an important role or not. Further improvements of the PST algorithms for the numerical treatment of the scheme close to the free surface and along the solid boundaries are also discussed.


## Artificial compressibility (ACSPH)

The scheme of `ACSPH_PLAN.md` and its dependencies. Added 2026-09-05.

### `decourcy2024`

- **file:** `decourcy2024_incompressible-delta-sph-artificial-compressibility.pdf`
- **title:** Incompressible δ-SPH via artificial compressibility
- **authors:** Joe J. De Courcy, Thomas C. S. Rendall, Lucian Constantin, Brano Titurus and Jonathan E. Cooper
- **venue:** *Computer Methods in Applied Mechanics and Engineering* 420:116700, 2024
- **doi:** [10.1016/j.cma.2023.116700](https://doi.org/10.1016/j.cma.2023.116700)
- **copy here:** published version, open access (CC BY)
- **relevance:** **The paper `ACSPH_PLAN.md` implements.** Replaces the WCSPH equation of state with a pressure-evolution equation marched in pseudo-time to a divergence-free state at every real time step (BDF2 outer, Runge-Kutta inner). Its Eq. (33) pressure bi-Laplacian is `marrone2011`'s density operator with ρ→p, so this repo's `modules/deltaSPH/wp_densityDelta.py` already computes it once generalised off the density field. Three defects found on review and recorded in the plan's Part 5: Eq. (37)'s `ε₄ = min(0, κ₄−ε₂)` should be `max` (as printed the JST operator vanishes in smooth flow), Eq. (40) and Fig. 1 are mutually inconsistent for the 3- and 4-stage schemes, and Eq. (30) carries a stray `h` the other three statements of the same equation do not.
- **abstract from:** PDF p.1

> Smoothed particle hydrodynamics using artificial compressibility (ACSPH) is developed, with the inclusion of pressure smoothing terms. Theoretical links between pressure/velocity correction incompressible SPH and artificial compressibility are explored, illustrating that ACSPH may be considered an extension of, or closely related to, the 𝛿-SPH method. An implicit dual-time integration procedure is used to enforce an incompressible solution at every time-step, removing acoustic effects arising from the common assumption of weak compressibility. An established weakly-compressible quasi-Lagrangian 𝛿-SPH method is used for comparison against ACSPH, and a series of test cases show that ACSPH provides a similar solution cost to 𝛿-SPH. However, the residual acoustic effects in 𝛿-SPH are removed entirely in ACSPH, providing improved pressure prediction capabilities across all test cases, including intense fluid impacts. Improved modelling of fluid–structure-interaction cases and coupled energy dissipation are also recorded as a result of correctly capturing incompressible flow.

### `antuono2010`

- **file:** `antuono2010_free-surface-flows-numerical-diffusive-terms.pdf`
- **title:** Free-surface flows solved by means of SPH schemes with numerical diffusive terms
- **authors:** M. Antuono, A. Colagrossi, S. Marrone and D. Molteni
- **venue:** *Computer Physics Communications* 181(3):532-549, 2010
- **doi:** [10.1016/j.cpc.2009.11.002](https://doi.org/10.1016/j.cpc.2009.11.002)
- **relevance:** Origin of the renormalised-gradient correction to the density Laplacian — the "enhanced formulation for the second-order derivatives ... consistent and convergent all over the fluid domain" of the abstract, which is what lets the diffusive term reach the free surface. `decourcy2024` Eq. (33) is this operator recast in pressure (AC-2L, its default). Co-cited with `antuono2012` for the linear stability bound behind `k₂ ≤ 0.2hβ`.
- **abstract from:** PDF p.1

> A novel system of equations has been defined which contains diffusive terms in both the continuity and energy equations and, at the leading order, coincides with a standard weakly-compressible SPH scheme with artificial viscosity. A proper state equation is used to associate the internal energy variation to the pressure field and to increase the speed of sound when strong deformations/compressions of the fluid occur. The increase of the sound speed is associated to the shortening of the time integration step and, therefore, allows a larger accuracy during both breaking and impact events. Moreover, the diffusive terms allows reducing the high frequency numerical acoustic noise and smoothing the pressure field. Finally, an enhanced formulation for the second-order derivatives has been defined which is consistent and convergent all over the fluid domain and, therefore, permits to correctly model the diffusive terms up to the free surface. The model has been tested using different free surface flows clearly showing to be robust, efficient and accurate. An analysis of the CPU time cost and comparisons with the standard SPH scheme is provided.

### `antuono2012`

- **file:** `antuono2012_numerical-diffusive-terms-weakly-compressible.pdf`
- **title:** Numerical diffusive terms in weakly-compressible SPH schemes
- **authors:** M. Antuono, A. Colagrossi and S. Marrone
- **venue:** *Computer Physics Communications* 183(12):2570-2580, 2012
- **doi:** [10.1016/j.cpc.2012.07.006](https://doi.org/10.1016/j.cpc.2012.07.006)
- **relevance:** The "theoretical analysis of the diffusive term structure" the abstract promises is the highest-priority dependency of `ACSPH_PLAN.md`: why the plain density Laplacian (`decourcy2024`'s AC-2) cannot hold a hydrostatic gradient at a truncated free surface, why the corrected form is really a bi-Laplacian rather than a Laplacian, the frozen-diffusion technique, and the stability bound the `k₂ = 0.1hβ` choice sits under. Also the reference for what this codebase's `DensityDiffusionScheme` variants actually are.
- **abstract from:** PDF p.1

> A discussion on the use of numerical diffusive terms in SPH models is proposed. Such terms are, generally, added in the continuity equation, in order to reduce the spurious numerical noise that affects the density and pressure fields in weakly-compressible SPH schemes. Specific focus has been given to the theoretical analysis of the diffusive term structure, highlighting the main benefits and drawbacks of the most widespread formulations. Finally, specific test cases have been used to compare such formulations and to confirm the theoretical findings.

### `letouze2013`

- **file:** `letouze2013_critical-investigation-sph-free-surfaces.pdf`
- **title:** A critical investigation of smoothed particle hydrodynamics applied to problems with free-surfaces
- **authors:** D. Le Touzé, A. Colagrossi, G. Colicchio and M. Greco
- **venue:** *International Journal for Numerical Methods in Fluids* 73(7):660-691, 2013
- **doi:** [10.1002/fld.3819](https://doi.org/10.1002/fld.3819)
- **relevance:** Source of the rotating/stretching square-patch benchmark (this repo's `rotatingSquarePatch`). Supplies the initial pressure field — a Poisson solve, without which the case cannot be initialised at all — plus the analytic stretching solution and the BEM/LDFM reference data `decourcy2024` Figs. 11-12 and 21-22 plot against. Its discussion of acoustic frequencies in the pressure signal and of the sound-velocity choice is the same effect ACSPH exists to remove.
- **abstract from:** PDF p.1 (Wiley labels it SUMMARY)

> In this paper, an in-depth study of SPH method, in its original weakly compressible version, is achieved on dedicated 2D and 3D free-surface flow test cases. These rather critical prototype problems shall constitute suitable test cases to get through when building a free-surface SPH model. The present work aims at investigating various numerical aspects of this method, often little mentioned in literature. In particular, a great care is paid to the dynamic part of the solution, which is critical to the local hydrodynamic load prediction. The role of numerical errors in the development of acoustic frequencies in the pressure signals is discussed, as well as the influence of the choice of the sound velocity. On the shown test problems, it is also evidenced that some numerical tools are crucial to ensure the robustness and accuracy of the standard SPH method. The convergence of our model is heuristically proved on these nonlinear prototype tests, showing at the same time the very satisfactory level of accuracy reached. Through these tests, some other numerical specificities of the SPH method are discussed, such as the self-redistribution of the particles occurring during the Lagrangian evolution. A higher order model is also proposed, and its advantages and drawbacks are discussed.

### `michel2022`

- **file:** `michel2022_particle-shifting-techniques.pdf`
- **title:** On Particle Shifting Techniques (PSTs): Analysis of existing laws and proposition of a convergent and multi-invariant law
- **authors:** J. Michel, A. Vergnaud, G. Oger, C. Hermange and D. Le Touzé
- **venue:** *Journal of Computational Physics* 459:110999, 2022
- **doi:** [10.1016/j.jcp.2022.110999](https://doi.org/10.1016/j.jcp.2022.110999)
- **relevance:** The shifting law `decourcy2024` Eqs. (55)-(57) adopt, chosen there specifically because it carries no sound-speed or Mach dependence — which an artificial-compressibility scheme has no way to supply. This codebase implements the Mach-scaled `sun2017` law instead (`modules/shifting/delta.py`, whose docstring notes the Michel form sits in a comment, unused), so this is `ACSPH_PLAN.md` Part 4.2's gap. The "conditions that should be respected by a PST" the abstract sets out are also the cleanest available checklist for auditing the shifting this repo already has.
- **abstract from:** PDF p.1

> This paper addresses the Particle Shifting Technique (PST) in the SPH schemes. Improving the accuracy of SPH schemes leads to particle clustering along the flow streamlines which turns to be detrimental for the simulations. PSTs aim at avoiding this adverse effect by slightly disordering the particles, allowing to retrieve a regular particle distribution within the kernel interpolation support. The gain in accuracy is such that this technique is now commonly adopted by the SPH practitioners, however the conditions that should be respected by a PST are not clearly discussed in the literature. In this paper, such conditions are exposed and their fulfillment by the main existing PSTs of the literature is analyzed. None of these existing PSTs fully satisfying these conditions, a novel PST is introduced. The proposed PST is validated for three different SPH schemes on 2D and 3D test cases, in presence of free-surface and solid boundaries.

### `ramachandran2021`

- **file:** `ramachandran2021_dual-time-sph-incompressible.pdf`
- **title:** Dual-time smoothed particle hydrodynamics for incompressible fluid simulation
- **authors:** Prabhu Ramachandran, Abhinav Muta and M. Ramakrishna
- **venue:** *Computers & Fluids* 227:105031, 2021
- **doi:** [10.1016/j.compfluid.2021.105031](https://doi.org/10.1016/j.compfluid.2021.105031)
- **relevance:** The closest prior art to `decourcy2024`: EDAC plus dual-time stepping, where ACSPH is momentum-divergence-driven plus dual-time stepping. Source of the pseudo-time material-derivative correction (`decourcy2024` Eqs. 27-31, which that paper implements and then recommends leaving off) and of the `α_PI = 2Δt/(2Δt+3Δτ)` point-implicit weighting its Eq. (41) generalises. Note the "completely open source implementation and a reproducible manuscript" the abstract advertises — the only reference implementation of a dual-time SPH scheme available to check ours against.
- **abstract from:** PDF p.1

> In this paper we propose a dual-time stepping scheme for the Smoothed Particle Hydrodynamics (SPH) method. Dual-time stepping has been used in the context of other numerical methods for the simulation of incompressible fluid flows. Here we provide a scheme that combines the entropically damped artificial compressibility (EDAC) along with dual-time stepping. The method is accurate, robust, and demonstrates up to seven times better performance than the standard weakly-compressible formulation. We demonstrate several benchmarks showing the applicability of the scheme. In addition, we provide a completely open source implementation and a reproducible manuscript.

### `lobovsky2014`

- **file:** `lobovsky2014_experimental-dam-break-pressure-loads.pdf`
- **title:** Experimental investigation of dynamic pressure loads during dam break
- **authors:** L. Lobovský, E. Botia-Vera, F. Castellana, J. Mas-Soler and A. Souto-Iglesias
- **venue:** *Journal of Fluids and Structures* 48:407-434, 2014
- **doi:** [10.1016/j.jfluidstructs.2014.03.009](https://doi.org/10.1016/j.jfluidstructs.2014.03.009)
- **relevance:** The dam-break experiment behind the geometry and the four wall pressure probes of `decourcy2024` §4.5, and the source of the 2.5%/97.5% percentile bounds its Figs. 28/30 score against. The "substantial variability which has been statistically characterized" is why the comparison is against a band rather than a curve — worth knowing before reading any single-run agreement as meaningful. Pressure signals, wave heights and videos are published as Supplementary Materials.
- **abstract from:** PDF p.1

> The objective of this research work has been to conduct experimental measurements on a dam break flow over a horizontal dry bed in order to provide a detailed insight, with emphasis on the pressure loads, into the dynamics of the dam break wave impacting a vertical wall downstream the dam. The experimental setup is described in detail, comprising state of the art miniaturized pressure sensors, high sampling rate data acquisition systems and high frame-rate video camera. It is a 1:2 scale of the highly cited (Lee et al., 2002, Journal of Fluids Engineering, 124) article experimental apparatus. Kinematics has been analyzed focusing on the free surface and wave front evolution. Experimental observations regarding liquid height and wave front speed have found to be in agreement with existing literature. This agreement enables the authors, assuming a similar framework, to discuss the measured pressure loads as a consequence of the dam break wave front impacting on the downstream wall. These loads show a substantial variability which has been statistically characterized. The measured quantities have been compared with the scarce available data in the literature, whose consistency is discussed. Measurements have been conducted with two filling heights. Scaling effects for such heights are also analyzed. As a direct result of the present initiative, an extensive set of data for computational tools validation is provided as Supplementary Materials, including pressure signals, wave height measurements and experimental videos.

### `marrone2015`

- **file:** `marrone2015_energy-losses-in-water-impacts.pdf`
- **title:** Prediction of energy losses in water impacts using incompressible and weakly compressible models
- **authors:** S. Marrone, A. Colagrossi, A. Di Mascio and D. Le Touzé
- **venue:** *Journal of Fluids and Structures* 54:802-822, 2015
- **doi:** [10.1016/j.jfluidstructs.2015.01.014](https://doi.org/10.1016/j.jfluidstructs.2015.01.014)
- **relevance:** Supplies the analytic incompressible kinetic-energy drop the two-jet impact case is scored against (`decourcy2024` Fig. 25, this repo's `impact`). More broadly it is the reference for *why* a weakly-compressible model dissipates impact energy differently from an incompressible one — the discrepancy ACSPH exists to remove, quantified here against a Level-Set Finite Volume reference rather than against another SPH scheme.
- **abstract from:** PDF p.1

> In the present work the simulation of water impacts is discussed. The investigation is mainly focused on the energy dissipation involved in liquid impacts in both the frameworks of the weakly compressible and incompressible models. A detailed analysis is performed using a weakly compressible Smoothed Particle Hydrodynamics (SPH) solver and the results are compared with the solutions computed by an incompressible meshbased Level-Set Finite Volume Method (LS-FVM). Impacts are numerically studied using single-phase models through prototypical problems in 1D and 2D frameworks. These problems were selected for the conclusions to be of interest for, e.g., the numerical computation of the flow around plunging breaking waves. The conclusions drawn are useful not only to SPH or LS-FVM users but also for other numerical models, for which accurate results on benchmark test-cases are provided.


## Spatial adaptivity, data structures, analytic boundaries

### `winchenbach2016`

- **file:** `winchenbach2016_constrained-neighbor-lists.pdf`
- **title:** Constrained Neighbor Lists for SPH-based Fluid Simulations
- **authors:** Rene Winchenbach, Hendrik Hochstetter and Andreas Kolb
- **venue:** *Eurographics/ACM SIGGRAPH Symposium on Computer Animation (SCA)*, pp. 49-56, 2016
- **doi:** [10.2312/sca.20161222](https://doi.org/10.2312/sca.20161222)
- **relevance:** Memory-bounded neighbor lists via locally adjusted support radii.
- **abstract from:** DOI record (10.2312/sca.20161222)

> In this paper we present a new approach to create neighbor lists with strict
> memory bounds for incompressible Smoothed Particle Hydrodynamics (SPH)
> simulations. Our proposed approach is based on a novel efficient
> predictive-corrective algorithm that locally adjusts particle support radii in
> order to yield neighborhoods of a user-defined maximum size. Due to the
> improved estimation of the initial support radius, our algorithm is able to
> efficiently calculate neighborhoods in a single iteration in almost any
> situation. We compare our neighbor list algorithm to previous approaches and
> show that our proposed approach can handle larger particle numbers on a single
> GPU due to its strict guarantees and is able to simulate more particles in
> real time due to its benefits in regard to performance. Additionally we
> demonstrate the versatility and stability of our approach in several different
> scenarios, for example multi-scale simulations and with different kernel
> functions.

### `winchenbach2017`

- **file:** `winchenbach2017_continuous-adaptivity.pdf`
- **title:** Infinite continuous adaptivity for incompressible SPH
- **authors:** Rene Winchenbach, Hendrik Hochstetter and Andreas Kolb
- **venue:** *ACM Transactions on Graphics* 36(4), Article 102, 2017
- **doi:** [10.1145/3072959.3073713](https://doi.org/10.1145/3072959.3073713)
- **relevance:** Continuous (not level-based) particle sizes, with mass redistribution.
- **abstract from:** DOI record (10.1145/3072959.3073713)

> In this paper we introduce a novel method to adaptive incompressible SPH
> simulations. Instead of using a scheme with a number of fixed particle sizes
> or levels, our approach allows continuous particle sizes. This enables us to
> define optimal particle masses with respect to, e.g., the distance to the
> fluid's surface. A required change in mass due to the dynamics of the fluid is
> properly and stably handled by our scheme of mass redistribution. This
> includes temporally smooth changes in particle masses as well as sudden mass
> variations in regions of high flow dynamics. Our approach guarantees low
> spatial variations in particle size, which is a core property in order to
> achieve large adaptivity ratios for incompressible fluid simulations.
> Conceptually, our approach allows for infinite continuous adaptivity,
> practically we achieved adaptivity ratios up to 5 orders of magnitude, while
> still being mass preserving and numerically stable, yielding unprecedented
> vivid surface detail at comparably low computational cost and moderate
> particle counts.

### `winchenbach2019`

- **file:** `winchenbach2019_multi-level-memory.pdf`
- **title:** Multi-Level-Memory Structures for Adaptive SPH Simulations
- **authors:** Rene Winchenbach and Andreas Kolb
- **venue:** *Vision, Modeling, and Visualization (VMV)*, pp. 99-107, 2019
- **doi:** [10.2312/vmv.20191323](https://doi.org/10.2312/vmv.20191323)
- **relevance:** Stacked hash-map data structures for highly adaptive SPH on GPUs.
- **abstract from:** DOI record (10.2312/vmv.20191323)

> In this paper we introduce a novel hash map-based sparse data structure for
> highly adaptive Smoothed Particle Hydrodynamics (SPH) simulations on GPUs. Our
> multi-level-memory structure is based on stacking multiple independent data
> structures, which can be created efficiently from the same particle data by
> utilizing self-similar particle orderings. Furthermore, we propose three
> neighbor list algorithms that improve performance, or significantly reduce
> memory requirements, when compared to Verlet-lists for the overall simulation.
> Overall, our proposed method significantly improves the performance of
> spatially adaptive methods, allows for the simulation of unbounded domains and
> reduces memory requirements without interfering with the simulation.

### `winchenbach2020mlm`

- **file:** `winchenbach2020mlm_simulating-and-rendering.pdf`
- **title:** Multi-Level Memory Structures for Simulating and Rendering Smoothed Particle Hydrodynamics
- **authors:** Rene Winchenbach and Andreas Kolb
- **venue:** *Computer Graphics Forum* 39(6):527-541, 2020
- **doi:** [10.1111/cgf.14090](https://doi.org/10.1111/cgf.14090)
- **relevance:** Journal extension of winchenbach2019, adding direct ray tracing off the same structure.
- **abstract from:** DOI record (10.1111/cgf.14090)

> In this paper, we present a novel hash map-based sparse data structure for
> Smoothed Particle Hydrodynamics, which allows for efficient neighbourhood
> queries in spatially adaptive simulations as well as direct ray tracing of
> fluid surfaces. Neighbourhood queries for adaptive simulations are improved by
> using multiple independent data structures utilizing the same underlying
> self-similar particle ordering, to significantly reduce non-neighbourhood
> particle accesses. Direct ray tracing is performed using an auxiliary data
> structure, with constant memory consumption, which allows for efficient
> traversal of the hash map-based data structure as well as efficient
> intersection tests. Overall, our proposed method significantly improves the
> performance of spatially adaptive fluid simulations and allows for direct ray
> tracing of the fluid surface with little memory overhead.

### `winchenbach2020`

- **file:** `winchenbach2020_semi-analytic-boundaries.pdf`
- **title:** Semi-analytic boundary handling below particle resolution for smoothed particle hydrodynamics
- **authors:** Rene Winchenbach, Rustam Akhunov and Andreas Kolb
- **venue:** *ACM Transactions on Graphics* 39(6), Article 173, 2020
- **doi:** [10.1145/3414685.3417829](https://doi.org/10.1145/3414685.3417829)
- **relevance:** Analytic particle-plane interaction, extended to arbitrary geometry via SDFs.
- **abstract from:** DOI record (10.1145/3414685.3417829)

> In this paper, we present a novel semi-analytical boundary handling method for
> spatially adaptive and divergence-free smoothed particle hydrodynamics (SPH)
> simulations, including two-way coupling. Our method is consistent under
> varying particle resolutions and allows for the treatment of boundary features
> below the particle resolution. We achieve this by first introducing an
> analytic solution to the interaction of SPH particles with planar boundaries,
> in 2D and 3D, which we extend to arbitrary boundary geometries using signed
> distance fields (SDF) to construct locally planar boundaries. Using this
> boundary-integral-based approach, we can directly evaluate boundary
> contributions, for any quantity, allowing an easy integration into state of
> the art simulation methods. Overall, our method improves interactions with
> small boundary features, readily handles spatially adaptive fluids, preserves
> particle-boundary interactions across varying resolutions, can directly be
> implemented in existing SPH methods, and, for non-adaptive simulations,
> provides a reduction in memory consumption as well as an up to 2× speedup
> relative to current particle-based boundary handling approaches.

### `winchenbach2021`

- **file:** `winchenbach2021_optimized-refinement.pdf`
- **title:** Optimized Refinement for Spatially Adaptive SPH
- **authors:** Rene Winchenbach and Andreas Kolb
- **venue:** *ACM Transactions on Graphics* 40(1):1-15, 2021
- **doi:** [10.1145/3363555](https://doi.org/10.1145/3363555)
- **copy here:** author's version
- **relevance:** A discretized objective function for refinement patterns; volume ratios to 1:1,000,000.
- **abstract from:** DOI record (10.1145/3363555)

> In this article, we propose an improved refinement process for the simulation
> of incompressible low-viscosity turbulent flows using Smoothed Particle
> Hydrodynamics, under adaptive volume ratios of up to 1 : 1, 000, 000. We
> derive a discretized objective function, which allows us to generate ideal
> refinement patterns for any kernel function and any number of particles a
> priori without requiring intuitive initial user-input. We also demonstrate how
> this objective function can be optimized online to further improve the
> refinement process during simulations by utilizing a gradient descent and a
> modified evolutionary optimization. Our investigation reveals an inherent
> residual refinement error term, which we smooth out using improved and novel
> methods. Our improved adaptive method is able to simulate adaptive volume
> ratios of 1 : 1, 000, 000 and higher, even under highly turbulent flows, only
> being limited by memory consumption. In general, we achieve more than an order
> of magnitude greater adaptive volume ratios than prior work.

### `winchenbach2024integrals`

- **file:** `winchenbach2024integrals_analytic-boundary-integrals-2d.pdf`
- **title:** Fully Analytic Higher-Order Boundary Integrals for Two-Dimensional SPH
- **authors:** Rene Winchenbach, Andreas Kolb and Nils Thuerey
- **venue:** *2024 International SPHERIC Workshop, Berlin, June 18--20, 2024*
- **relevance:** Analytic boundary integrals over triangle meshes, with barycentric boundary quantities.
- **abstract from:** PDF p.1

> In this paper we present a fully analytic boundary handling approach for
> Smoothed Particle Hydrodynamics in 2D, which works by directly evaluating
> boundary contributions over triangle meshes using a novel integral
> factorization for both scalar and gradient terms. In contrast to prior methods
> that rely upon boundary surface particles or wall-renormalization approaches,
> our approach can be directly integrated into an SPH formulation without
> introducing additional terms, e.g., artificial volumes for boundary particles.
> Furthermore, our method enables assigning quantities, e.g., for pressure
> values, as constant or linearly varying per triangles using barycentric
> interpolations which enable more complex boundary interactions, whilst
> retaining a fully analytic formulation. Moreover, our proposed integral
> solution works for any triangle geometry, e.g., even degenerate triangle
> shapes, for polynomial compact kernel functions, e.g., B-spline or Wendland
> kernels, and regardless of particle and boundary element sizes. To validate
> our method, we compare the achieved results with a numerical boundary
> integral.

### `winchenbach2025analytic`

- **file:** `winchenbach2025analytic_analytic-boundary-handling-2d.pdf`
- **title:** Solving Boundary Handling Analytically in Two Dimensions for Smoothed Particle Hydrodynamics
- **authors:** Rene Winchenbach and Andreas Kolb
- **venue:** Journal of Computational Physics 555, 2026, article 114788
- **doi:** [10.1016/j.jcp.2026.114788](https://doi.org/10.1016/j.jcp.2026.114788) (preprint was arXiv:2507.21686)
- **relevance:** Closed-form boundary integrals for compact polynomials over triangles, via Chebyshev polynomials and 2F1.
- **abstract from:** arXiv API (Crossref record carries no abstract for this DOI)

> We present a fully analytic approach for evaluating boundary integrals in two
> dimensions for Smoothed Particle Hydrodynamics (SPH). Conventional methods
> often rely on boundary particles or wall re-normalization approaches derived
> from applying the divergence theorem, whereas our method directly evaluates
> the area integrals for SPH kernels and gradients over triangular boundaries.
> This direct integration strategy inherently accommodates higher-order boundary
> conditions, such as piecewise cubic fields defined via Finite Element
> stencils, enabling analytic and flexible coupling with mesh-based solvers. At
> the core of our approach is a general solution for compact polynomials of
> arbitrary degree over triangles by decomposing the boundary elements into
> elementary integrals that can be solved with closed-form solutions. We provide
> a complete, closed-form solution for these generalized integrals, derived by
> relating the angular components to Chebyshev polynomials and solving the
> resulting radial integral via a numerically stable evaluation of the Gaussian
> hypergeometric function 2F1. Our solution is robust and adaptable and works
> regardless of triangle geometries and kernel functions. We validate the
> accuracy against high-precision numerical quadrature rules, as well as in
> problems with known exact solutions. We provide an open-source implementation
> of our general solution using differentiable programming to facilitate the
> adoption of our approach to SPH and other contexts that require analytic
> integration over polygonal domains. Our analytic solution outperforms existing
> numerical quadrature rules for this problem by up to five orders of magnitude,
> for integrals and their gradients, while providing a flexible framework to
> couple arbitrary triangular meshes analytically to Lagrangian schemes,
> building a strong foundation for addressing several grand challenges in SPH
> and beyond.

### `winchenbach2025diffsph`

- **file:** `winchenbach2025diffsph_differentiable-sph.pdf`
- **title:** diffSPH: Differentiable Smoothed Particle Hydrodynamics for Hybrid Machine Learning Solutions in Fluid Mechanics
- **authors:** Rene Winchenbach and Nils Thuerey
- **venue:** Journal of Computational Physics 555, 2026, article 114769
- **doi:** [10.1016/j.jcp.2026.114769](https://doi.org/10.1016/j.jcp.2026.114769) (preprint arXiv:2507.21684 was titled "...for Adjoint Optimization and Machine Learning")
- **relevance:** The differentiable PyTorch SPH framework this codebase's schemes are ported from.
- **abstract from:** arXiv API (Crossref record carries no abstract for this DOI)

> We present diffSPH, a novel open-source differentiable Smoothed Particle
> Hydrodynamics (SPH) framework developed entirely in PyTorch with GPU
> acceleration. diffSPH is designed centrally around differentiation to
> facilitate optimization and machine learning (ML) applications in
> Computational Fluid Dynamics (CFD), including training neural networks and the
> development of hybrid models. Its differentiable SPH core, and schemes for
> compressible (with shock capturing and multi-phase flows), weakly compressible
> (with boundary handling and free-surface flows), and incompressible physics,
> enable a broad range of application areas. We demonstrate the framework's
> unique capabilities through several applications, including addressing
> particle shifting via a novel, target-oriented approach by minimizing physical
> and regularization loss terms, a task often intractable in traditional
> solvers. Further examples include optimizing initial conditions and physical
> parameters to match target trajectories, shape optimization, implementing a
> solver-in-the-loop setup to emulate higher-order integration, and
> demonstrating gradient propagation through hundreds of full simulation steps.
> Prioritizing readability, usability, and extensibility, this work offers a
> foundational platform for the CFD community to develop and deploy novel neural
> networks and adjoint optimization applications.


## Machine learning on SPH

### `winchenbach2024sfbc`

- **file:** `winchenbach2024sfbc_symmetric-basis-convolutions.pdf`
- **title:** Symmetric Basis Convolutions for Learning Lagrangian Fluid Mechanics
- **authors:** Rene Winchenbach and Nils Thuerey
- **venue:** *International Conference on Learning Representations (ICLR)*, 2024
- **arXiv:** [2403.16680](https://arxiv.org/abs/2403.16680)
- **relevance:** Separable-basis continuous convolutions; even/odd symmetry as the stability lever.
- **abstract from:** arXiv API

> Learning physical simulations has been an essential and central aspect of many
> recent research efforts in machine learning, particularly for
> Navier-Stokes-based fluid mechanics. Classic numerical solvers have
> traditionally been computationally expensive and challenging to use in inverse
> problems, whereas Neural solvers aim to address both concerns through machine
> learning. We propose a general formulation for continuous convolutions using
> separable basis functions as a superset of existing methods and evaluate a
> large set of basis functions in the context of (a) a compressible 1D SPH
> simulation, (b) a weakly compressible 2D SPH simulation, and (c) an
> incompressible 2D SPH Simulation. We demonstrate that even and odd symmetries
> included in the basis functions are key aspects of stability and accuracy. Our
> broad evaluation shows that Fourier-based continuous convolutions outperform
> all other architectures regarding accuracy and generalization. Finally, using
> these Fourier-based networks, we show that prior inductive biases, such as
> window functions, are no longer necessary. An implementation of our approach,
> as well as complete datasets and solver implementations, is available at
> https://github.com/tum-pbs/SFBC.

### `winchenbach2023spheric`

- **file:** `winchenbach2023spheric_hybrid-sph-ml-framework.pdf`
- **title:** A Hybrid Framework for Fluid Flow Simulations: Combining SPH with Machine Learning
- **authors:** Rene Winchenbach and Nils Thuerey
- **venue:** *2023 International SPHERIC Workshop, Rhodes, June 27--29, 2023*
- **relevance:** pytorchSPH: an open-source PyTorch SPH solver built to link directly to ML models.
- **abstract from:** PDF p.1

> Machine Learning and Data Science have been a quickly growing and highly
> impactful field of research; however, thus far they find virtually no adoption
> within Smoothed Particle Hydrodynamics, due to a variety of fundamental
> issues. Recent machine learning approaches have introduced a variety of
> approaches, e.g., graph convolutional networks [1], continuous convolutions
> [2] and graph neural networks [3]; however, these approaches are often not
> validated as is typical within the CFD/SPHERIC community. The present paper
> aims to provide a foundation that would enable a more tight connection between
> the two fields. We achieve this by providing an Open Source SPH simulation
> built upon the PyTorch machine learning framework, using a variety of fluid
> and boundary treatments. By utilizing PyTorch as the underlying foundation we
> can readily utilize a variety of acceleration techniques, e.g., GPU
> accelation, whilst retaining a high level of abstraction in the source code.
> By utilizing PyTorch, we can directly link our simulation framework with
> machine learning approaches, which are also implemented within our overall
> framework. Basing the entire codebase on python in this regard also enables
> the framework being used in online platforms, e.g., Google Colab, enabling
> researchers and students to work with our framework without requiring personal
> hardware. Utilizing the this tight coupling, it is readily possible to
> evaluate the ability of a machine learning approach to replace components of
> the SPH simulation, e.g., an SPH density summation, or even the entire
> simulation step. We include a variety of traditional benchmark scenarios,
> e.g., oscillating drops, breaking dam scenarios and flows past obstacles, for
> validation, as well as scripts to generate randomized data for training. Our
> codebase is available online under an MIT license at
> https://github.com/wi-re/pytorchSPH.

### `winchenbach2024pmac`

- **file:** `winchenbach2024pmac_taylor-green-cross-validation.pdf`
- **title:** Cross-Validation of SPH-based Machine Learning Models using the Taylor-Green Vortex Case
- **authors:** Rene Winchenbach and Nils Thuerey
- **venue:** *Particle Methods and Applications Conference (PMAC), Santa Fe, January 22--24, 2024*
- **relevance:** A Taylor-Green cross-validation benchmark for ML models, on a differentiable delta+-SPH solver.
- **abstract from:** PDF p.1

> Machine Learning (ML) research has been quickly growing and highly impactful
> in many areas of research in recent times, e.g., in Natural Language
> Processing; however, thus far ML finds very little adoption within
> Computational Fluid Dynamics (CFD) and especially little adoption within the
> Smoothed Particle Hydrodynamics (SPH) community, due to a variety of
> fundamental issues. One of the core issues is that a large variety of ML-based
> approaches to physical simulations, e.g., message passing networks [1] and
> continuous convolutions [2], are generally not validated against known
> validation test cases and instead are evaluated on limited, and oftentimes
> arbitrarily constructed, test cases. While these test cases serve as a useful
> comparison basis within ML research, they are not easily relatable to CFD
> applications and validation cases. We now aim to provide a step towards more
> validation by building on a widely known validation case to provide some
> understanding in the accuracy and generalization capabilities of state of the
> art ML models. Built upon a differentiable δ + -based SPH solver, we propose a
> novel cross-validation benchmark setup that comprises two primary components
> as (a) a validation setup built up on the the Taylor-Green Vortex validation
> case and (b) a training setup builtin upon randomized periodic and
> divergence-free flow fields. We then perform a cross-validation training where
> we train an ML-based simulation on the randomized initial conditions and
> evaluate on the Taylor-Green Vortex and vice-a-versa. For classical solvers,
> performance on a validation case is often times used as an indicator of
> performance on more arbitrary, yet similar, setups; however, for ML-based
> simulations such a relationship is not as clear-cut and is not well-studied
> thus far. Furthermore, we also perform a convergence study to relate the size
> of the ML model to the accuracy of the model, in relation to the included δ +
> -based solver for varying simulation parameters and scales. Using this
> process, and by including a full open-source implementation of all relevant
> components and by providing access to sample datasets for different scales and
> parameters, we aim to provide an important stepping stone leading towards more
> validation of ML-focused research and more acceptance of such methods within
> the CFD community and easy inclusion of validation cases into ML-focused
> research.

### `winchenbach2024spheric`

- **file:** `winchenbach2024spheric_physically-motivated-ml.pdf`
- **title:** Physically-Motivated Machine Learning Models for Lagrangian Fluid Mechanics
- **authors:** Rene Winchenbach and Nils Thuerey
- **venue:** *2024 International SPHERIC Workshop, Berlin, June 18--20, 2024*
- **relevance:** Symmetry-built ML model using Chebyshev/Fourier bases and SPH-informed kernel choices.
- **abstract from:** PDF p.1

> Machine learning models are often treated as black boxes with vast amounts of
> data, and they produce the desired predictions, given enough resources. While
> initial research in learning Lagrangian fluid simulations relied heavily on
> this approach[1], later results demonstrated that simply providing data is
> generally insufficient to learn physically correct behavior, e.g.,
> conservation of momentum[2]. Consequently, there has been a growing trend of
> machine learning approaches that directly enforce certain constraints, e.g.,
> symmetries, allowing them to significantly improve performance compared to
> prior methods by incorporating hard constraints in the model itself instead of
> soft data constraints. A key finding of these approaches is that including
> inductive biases based on domain knowledge, e.g., SPH theory, significantly
> improves these models. The present paper aims to demonstrate a physically
> motivated Machine Learning Model inherently built around symmetries and
> incorporates several insights from classical numerical approaches to improve
> existing methods. These insights include higherorder numerical interpolation,
> e.g., Chebyshev polynomials, SPHinformed kernel choices, e.g., Wendland kernel
> functions and coordinate system transformations. By building the Machine
> Learning Model directly upon higher-order basis functions with inherent
> symmetries, i.e., Fourier-series terms, the learned behavior of these models
> is also smoother and more physically plausible [3]. Several key aspects of our
> SPH simulation for data generation were also used to train our neural
> networks. We furthermore demonstrate that although these Machine Learning
> Models are trained solely on data-driven metrics, i.e., comparisons of
> particle position trajectories, they also exhibit much better performance
> concerning emergent properties, such as divergence-freedom, despite not being
> trained on any explicit physical constraints or loss terms. Finally, we
> demonstrate how these networks are also improving upon the performance of
> prior Machine Learning Models in the two-dimensional TaylorGreen Vortex
> validation case and how using this as a one-shot training setup yields neural
> networks that perform well in general simulation cases not seen in training.

### `winchenbach2025spheric`

- **file:** `winchenbach2025spheric_morinet.pdf`
- **title:** MoriNet: A Machine Learning-based Mori-Zwanzig Perspective on Weakly Compressible SPH
- **authors:** Rene Winchenbach and Nils Thuerey
- **venue:** *19th International SPHERIC Workshop, Barcelona, June 2025*
- **relevance:** Mori-Zwanzig view of WCSPH vs ISPH: density as a memory term, and its timescale dependence.
- **abstract from:** PDF p.1

> This work investigates the relationship between weakly compressible and
> incompressible Smoothed Particle Hydrodynamics (WCSPH and ISPH) using the
> Mori-Zwanzig formalism to further the development of efficient data-driven
> models. WCSPH approximates incompressible flows under certain conditions,
> e.g., low Mach number and density variations. We demonstrate that density
> fluctuations act as essential variables, primarily influencing the short-time
> evolution of the fluid, while the longtime behavior converges towards
> incompressible dynamics. Consequently, we posit that the importance of density
> as a feature for neural networks trained to emulate WCSPH simulations varies
> with timescale: dominating at short times and diminishing at longer times.
> Using the Mori-Zwanzig formalism, we treat particle positions and velocities
> as the relevant variables, implicitly modeling the density field through a
> memory term. Our results confirm this timescale dependence, demonstrating its
> implications for training neural networks and providing a theoretical
> justification for temporal coarse-graining approaches that simplify the
> learning task by focusing on longer timescales.

