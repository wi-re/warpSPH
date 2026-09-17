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

### `english2025`

- **file:** `english2025_river-flows-past-bridges.pdf`
- **title:** Smoothed particle hydrodynamics modelling of river flows past bridges
- **authors:** Aaron English, Renato Vacondio, Susanna Dazzi and José M. Domínguez
- **venue:** *Computers & Fluids* 303:106870, 2025
- **doi:** [10.1016/j.compfluid.2025.106870](https://doi.org/10.1016/j.compfluid.2025.106870)
- **relevance:** An improved mDBC boundary pressure/density procedure (Eqs. 8-11: a corrected-kernel-sum ghost density fit with a Shepard fallback, then pressure cloning via an *analytic* hydrostatic-gravity term -- not a fitted spatial gradient -- extrapolating ghost pressure to the boundary particle) plus a no-slip velocity extension to mDBC. `BOUNDARY_DENSITY_PLAN.md`'s candidate fix for the ghost-to-boundary extrapolation-distance amplification found in that investigation's Band-2018-based prototype.
- **abstract from:** PDF p.1

> In this work, Smoothed Particle Hydrodynamics (SPH) is assessed for the modelling of flow past bridges. An improved pressure extrapolation method and a no-slip extension for the widely used modified Dynamic Boundary Condition (mDBC) are presented. The no-slip condition is validated with benchmark test cases of Poiseuille flow and flow past a cylinder. The ability to simulate river flows past bridges is assessed by comparing with experimental measurements for two model bridges with multiple discharges. The results are also evaluated against numerical results from 2D Shallow Water Equation (SWE) simulations, which is the leading approach for this kind of flow. While both methods shows good agreement with the experimental data away from the bridge, the SWE assumptions fail in the immediate vicinity of the bridge. In this region, the SPH method demonstrates higher accuracy, captures additional flow features and offers deeper insight into local hydraulic behaviour. A new SPH restart procedure has been developed that enables high-resolution simulations to be initialized using results from lower-resolution simulations. This greatly reduces simulation run times for large and complex transient flow such as rivers. Advanced DualSPHysics boundary generation and pre-processing tools allow for easier creation of boundaries through STL files, and GPU acceleration on the latest hardware allow for faster simulation with larger domains. With all these features, the first full-scale SPH simulation of a real river flow past a bridge is presented, including the riverbed bathymetry and model of Ponte Vecchio on the Arno River (Italy).

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


### `english2022`

- **file:** `english2022_mdbc-general-purpose-sph.pdf`
- **title:** Modified dynamic boundary conditions (mDBC) for general-purpose smoothed particle hydrodynamics (SPH): application to tank sloshing, dam break and fish pass problems
- **authors:** Aaron English, José M. Domínguez, Renato Vacondio, Alejandro J. C. Crespo, Peter K. Stansby, Steven J. Lind, Luca Chiapponi and Moncho Gómez-Gesteira
- **venue:** *Computational Particle Mechanics* 9(5):911-925, 2022
- **doi:** [10.1007/s40571-021-00403-3](https://doi.org/10.1007/s40571-021-00403-3)
- **relevance:** The mDBC boundary density/pressure paper modules/mdbc/density2025.py implements Eq. 12 of. Distinct from english2025 (Computers & Fluids, river-bridges application paper) already in this manifest -- this is the original method paper. BOUNDARY_DENSITY_PLAN.md names it directly as the source of the Eq. 12 extrapolation this codebase implements.
- **abstract from:** PDF p.3

> Dynamic boundary conditions (DBC) for solid surfaces are standard in the weakly compressible smoothed particle hydrodynamics (SPH) code DualSPHysics. A stationary solid is simply represented by fixed particles with pressure from the
> equation of state. Boundaries are easy to set up and computations are relatively stable and efficient, providing robust numerical simulation for complex geometries. However, a small unphysical gap between the fluid and solid boundaries can form,
> decreasing the accuracy of pressures measured on the boundary. A method is presented where the density of solid particles
> is obtained from ghost positions within the fluid domain by linear extrapolation. With this approach, the gap between fluid
> and boundary is reduced and pressures in still water converge to hydrostatic, including the case of a bed with a sharp corner.
> The violent free-surface cases of a sloshing tank and dam break impact on an obstacle show pressures measured directly
> on solid surfaces in close agreement with experiments. The complex 3-D flow in a fish pass, with baffles to divert the flow,
> is simulated showing close agreement with measured water levels with weirs open and gates closed, but less close with
> gates open and weirs closed. This indicates the method is suitable for rapidly varying free-surface flows, but development
> for complex turbulent flows is necessary. The code with the modified dynamic boundary condition (mDBC) is available in
> DualSPHysics to run on CPUs or GPUs.

## Free-surface detection

### `barecasco2013`

- **file:** `barecasco2013_free-surface-detection-sph.pdf`
- **title:** Simple free-surface detection in two and three-dimensional SPH solver
- **authors:** Agra Barecasco, Hanifa Terissa and Christian Fredy Naa
- **venue:** arXiv:1309.4290 [physics.flu-dyn], 2013
- **arXiv:** [1309.4290](https://arxiv.org/abs/1309.4290)
- **relevance:** The Barecasco surface-detection scheme (configurations/incompressible.py's barecascoThreshold, and the detectFreeSurface mask discussed at length in DELTASPH_VALIDATION_PLAN.md §5.34-5.35): an angle-based coverage-vector test using overlapping spheres, distinct from the Marrone/Sun dilation approach already implemented alongside it. No DOI found on Crossref -- appears to be an arXiv-only preprint, never formally published.
- **abstract from:** arXiv API

> A simple free-surface particle detection method for two and three-dimensional SPH simulation has been implemented. The method uses sphere representation for the SPH particle. The fluid domain is covered by overlapping spheres. A sphere whose surface is not fully covered considered as boundary. To test particle boundary status, we used a sum of normalized relative position vectors from neighbouring particles to the test particle. By checking the existence of un- covered sphere surface by this vector sum, boundary status of the test particle can be determined. This boundary detection method can be easily embedded in the SPH solver algorithm.

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

### `sun2018`

- **file:** `sun2018_multi-resolution-delta-plus-sph-tensile-instability-control.pdf`
- **title:** Multi-resolution Delta-plus-SPH with tensile instability control: Towards high Reynolds number flows
- **authors:** P. N. Sun, A. Colagrossi, S. Marrone, M. Antuono and A. M. Zhang
- **venue:** *Computer Physics Communications* 224:63-80, 2018
- **doi:** [10.1016/j.cpc.2017.11.016](https://doi.org/10.1016/j.cpc.2017.11.016)
- **relevance:** The Tensile Instability Control (TIC) paper: §2.1 Eq. (9) is a per-pair switch on the pressure-gradient term, `F_ji = p_j+p_i` when `p_i >= 0` or `i` is in the free-surface region `SF`, else `F_ji = p_j-p_i` — conditioned only on the *query* particle `i`'s own state, never on the neighbour `j`'s pressure. This codebase's `PressureForceScheme.Antuono` (`modules/pressure/wp_surfaceAware.py`) computes `sw = (P_i >= 0 or P_j >= 0) or (mask_i == 1)`, which also fires whenever the *neighbour's* pressure is non-negative — a branch Eq. (9) does not have. Against a clamped mDBC wall (ghost density floored at `rho0`, so `P_j == 0`) this forces the symmetric form even when the query fluid particle is in genuine tension and not at the free surface, which is the mechanism `DELTASPH_VALIDATION_PLAN.md` §5.13's sloshingTank ceiling-hover particle traces to (a steady `dvdt_n` into the wall with `p_i ~ -0.4` against `p_b = 0`).
- **abstract from:** PDF p.1

> It is well known that the use of SPH models in simulating flow at high Reynolds numbers is limited because of the tensile instability inception in the fluid region characterized by high vorticity and negative pressure. In order to overcome this issue, the δ + -SPH scheme is modified by implementing a Tensile Instability Control (TIC). The latter consists of switching the momentum equation to a non-conservative formulation in the unstable flow regions. The loss of conservation properties is shown to induce small errors, provided that the particle distribution is regular. The latter condition can be ensured thanks to the implementation of a Particle Shifting Technique (PST). The novel variant of the δ + -SPH is proved to be effective in preventing the onset of tensile instability. Several challenging benchmark tests involving flows past bodies at large Reynolds numbers have been used. Within this a simulation characterized by a deforming foil that resembles a fish-like swimming body is used as a practical application of the δ + -SPH model in biological fluid mechanics.

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

### `molteni2009`

- **file:** `molteni2009_pressure-evaluation-density-diffusion.pdf`
- **title:** A simple procedure to improve the pressure evaluation in hydrodynamic context using the SPH
- **authors:** Diego Molteni and Andrea Colagrossi
- **venue:** *Computer Physics Communications* 180(6):861-872, 2009
- **doi:** [10.1016/j.cpc.2008.12.004](https://doi.org/10.1016/j.cpc.2008.12.004)
- **relevance:** The origin paper of the SPH density diffusion term (DDT), which `antuono2010`/`antuono2012` later correct with a renormalized-gradient term. This paper's own DDT has no such correction at all -- a plain density-difference-weighted SPH Laplacian in the continuity equation. `DELTASPH_VALIDATION_PLAN.md` §8.15 found DualSPHysics' simplest DDT option (`TDensity=1`) implements close to this formula verbatim, and that this codebase's own `deltaSPH_wrongSign` diagnostic (§8.11, the pre-fix psi sign) empirically degenerates toward the same un-renormalized Laplacian family -- this paper is the reference for what that family actually is and why it was introduced (damping spurious pressure noise, §1).
- **abstract from:** PDF p.1

> In literature, it is well know that the Smoothed Particle Hydrodynamics method can be affected by numerical noise on the pressure field when dealing with liquids. This can be highly dangerous when an SPH code is dynamically coupled with a structural solver. In this work a simple procedure is proposed to improve the computation of the pressure distribution in the dynamics of liquids. Such a procedure is based on the use of a density diffusion term in the equation for the mass conservation. This diffusion is a pure numerical effect, similar to the well known artificial viscosity originally proposed in SPH method to smooth out the shock discontinuities. As the artificial viscosity, the density diffusion used here goes to zero increasing the number of particles recovering consistency and convergence of the final numerical scheme adopted. Different artificial density diffusion formulas have been studied, paying attention to prevent unphysical changes of the flows. To show the improvements of the new scheme proposed here, a suitable set of examples, for which reference solutions or experimental data are available, has been tested.

### `fourtakas2019`

- **file:** `fourtakas2019_lust-boundary-condition-ddt-correction.pdf`
- **title:** Local uniform stencil (LUST) boundary condition for arbitrary 3-D boundaries in parallel smoothed particle hydrodynamics (SPH) models
- **authors:** Georgios Fourtakas, Jose M. Dominguez, Renato Vacondio and Benedict D. Rogers
- **venue:** *Computers & Fluids* 190:346-361, 2019
- **doi:** [10.1016/j.compfluid.2019.06.009](https://doi.org/10.1016/j.compfluid.2019.06.009)
- **copy here:** published version, open access (CC BY)
- **relevance:** Primarily the LUST fictitious-particle boundary condition, not directly relevant here -- but its §3.2 "Improved density diffusion term for gravity driven flows" is DualSPHysics' `DDT_DDT2`/`DDT_DDT2Full` (`TDensity=2/3`): cancels the *hydrostatic* background density difference analytically (from the gravity direction and the EOS, Eq. following its Eq. (13)) instead of via a renormalized SPH gradient estimate -- structurally distinct from `antuono2010`/`antuono2012`'s correction and from this codebase's `deltaSPH` DDT. `DELTASPH_VALIDATION_PLAN.md` §8.15/8.16: neither of DualSPHysics' two real DDT options uses the renormalized-gradient machinery this codebase implements; this is the modern, DualSPHysics-recommended one, specifically built to fix `DDT_DDT`'s (`molteni2009`'s) known near-wall/sloped-free-surface errors. Adapted to this codebase's isothermal EOS, its hydrostatic term collapses to the same formula `cases/dambreak.py`'s existing `hydrostaticInit` already computes.
- **abstract from:** PDF p.1

> This paper presents the development of a new boundary treatment for free-surface hydrodynamics using the smoothed particle hydrodynamics (SPH) method accelerated with a graphics processing unit (GPU). The new solid boundary formulation uses a local uniform stencil (LUST) of fictitious particles that surround and move with each fluid particle and are only activated when they are located inside a boundary. This addresses the issues currently affecting boundary conditions in SPH, namely the accuracy, robustness and applicability while being amenable to easy parallelization such as on a GPU. In 3-D, the methodology uses triangles to represent the geometry with a ray tracing procedure to identify when the LUST particles are activated. A new correction is proposed to the popular density diffusion term treatment to correct for pressure errors at the boundary. The methodology is applicable to complex arbitrary geometries without the need of special treatments for corners and curvature is presented. The paper presents the results from 2-D and 3-D Poiseuille flows showing convergence rates typical for weakly compressible SPH. Still water in a complex 3-D geometry with a pyramid demonstrates the robustness of the technique with excellent agreement for the pressure distributions. The method is finally applied to the SPHERIC benchmark of a dry-bed dam-break impacting an obstacle showing satisfactory agreement and convergence for a violent flow.

### `letouze2013`

- **file:** `letouze2013_critical-investigation-sph-free-surfaces.pdf`
- **title:** A critical investigation of smoothed particle hydrodynamics applied to problems with free-surfaces
- **authors:** D. Le Touzé, A. Colagrossi, G. Colicchio and M. Greco
- **venue:** *International Journal for Numerical Methods in Fluids* 73(7):660-691, 2013
- **doi:** [10.1002/fld.3819](https://doi.org/10.1002/fld.3819)
- **relevance:** Source of the rotating/stretching square-patch benchmark (this repo's `rotatingSquarePatch`). Supplies the initial pressure field — a Poisson solve, without which the case cannot be initialised at all — plus the analytic stretching solution and the BEM/LDFM reference data `decourcy2024` Figs. 11-12 and 21-22 plot against. Its discussion of acoustic frequencies in the pressure signal and of the sound-velocity choice is the same effect ACSPH exists to remove.
- **abstract from:** PDF p.1 (Wiley labels it SUMMARY)

> In this paper, an in-depth study of SPH method, in its original weakly compressible version, is achieved on dedicated 2D and 3D free-surface flow test cases. These rather critical prototype problems shall constitute suitable test cases to get through when building a free-surface SPH model. The present work aims at investigating various numerical aspects of this method, often little mentioned in literature. In particular, a great care is paid to the dynamic part of the solution, which is critical to the local hydrodynamic load prediction. The role of numerical errors in the development of acoustic frequencies in the pressure signals is discussed, as well as the influence of the choice of the sound velocity. On the shown test problems, it is also evidenced that some numerical tools are crucial to ensure the robustness and accuracy of the standard SPH method. The convergence of our model is heuristically proved on these nonlinear prototype tests, showing at the same time the very satisfactory level of accuracy reached. Through these tests, some other numerical specificities of the SPH method are discussed, such as the self-redistribution of the particles occurring during the Lagrangian evolution. A higher order model is also proposed, and its advantages and drawbacks are discussed.

### `vila1999`

- **file:** `vila1999_particle-weighted-methods-and-sph.pdf`
- **title:** On particle weighted methods and smooth particle hydrodynamics
- **authors:** J. P. Vila
- **venue:** *Mathematical Models and Methods in Applied Sciences* 9(2):161–209, 1999
- **doi:** [10.1142/s0218202599000117](https://doi.org/10.1142/s0218202599000117)
- **relevance:** The ALE-SPH formalism itself, and the reason the Riemann route needs a Riemann solver: conservation laws in **conservative** variables with Godunov-type finite-difference fluxes between particles. `michel2022` Sec. 4.2.1 is this scheme rewritten with a velocity decomposition; `PST_ALE_PLAN.md` stage D. Contrast `antuono2021`, which reaches an ALE in primitive variables and needs no such flux.
- **abstract from:** Crossref (publisher-deposited JATS abstract)
- **note:** The copy held here is an author's version paginated 1–48 rather than the published 161–209; the fields above are the DOI record's.

> This paper deal with designing of weighted particle approximation of conservation laws. New ideas concerning the use of variable smoothing length, renormalization and the use of Godunov type finite difference fluxes in particle methods are introduced and discussed in connection with standard implementation of the SPH method. A detailed analysis of boundary conditions approximation is also provided.

### `parshikov2002`

- **file:** `parshikov2002_sph-interparticle-contact-algorithms.pdf`
- **title:** Smoothed Particle Hydrodynamics Using Interparticle Contact Algorithms
- **authors:** Anatoly N. Parshikov and Stanislav A. Medin
- **venue:** *Journal of Computational Physics* 180(1):358–382, 2002
- **doi:** [10.1006/jcph.2002.7099](https://doi.org/10.1006/jcph.2002.7099)
- **relevance:** The scheme `michel2022` Sec. 4.2.2 validates on, recovered from `vila1999` by cancelling the mass fluxes: Riemann-solved velocity and stress at the contact point in place of pair averages, which is what removes the need for artificial viscosity. `PST_ALE_PLAN.md` stage C — the cheap consumer of the Riemann subsystem, and the check that it is right in an SPH setting before `vila1999` adds mass fluxes on top.
- **abstract from:** PDF p.1

> Smoothed particle hydrodynamics (SPH) is a modern effective technique of computer simulation in continuous media mechanics. SPH approximations are quite flexible and allow various constructions. In this paper, contact interaction between particles is introduced in SPH formulation. The concept is to insert in SPH approximations of a strength medium the velocity and stresses determined at the contact point by Riemann solution, instead of mean values between velocities and stresses of basic and surrounding particles. In this case, there is no need to use artificial viscosity. In a heat-conducting medium, the contact temperature is determined by the solution of a thermal discontinuity breakup and heat fluxes in particles are computed with the use of this temperature. The modified SPH approximations easily pass various standard tests and are easily realized in multidimensional codes. 

### `oger2016`

- **file:** `oger2016_quasi-lagrangian-transport-velocity-ale.pdf`
- **title:** SPH accuracy improvement through the combination of a quasi-Lagrangian shifting transport velocity and consistent ALE formalisms
- **authors:** G. Oger, S. Marrone, D. Le Touzé and M. de Leffe
- **venue:** *Journal of Computational Physics* 313:76–98, 2016
- **doi:** [10.1016/j.jcp.2016.02.039](https://doi.org/10.1016/j.jcp.2016.02.039)
- **relevance:** The weakly-compressible ALE study, and the closest prior art for the Riemann route (`PST_ALE_PLAN.md` stage B). It is also `michel2022` Table 1's third row: its transport velocity uses `U_char = Ma c₀`, the Mach scaling that breaks Galilean invariance and that an artificial-compressibility scheme has no way to supply — the same defect this codebase's `modules/shifting/delta.py` inherits from `sun2017`.
- **abstract from:** PDF p.1

> This paper addresses the accuracy of the weakly-compressible SPH method. Interpolation defects due to the presence of anisotropic particle structures inherent to the Lagrangian character of the Smoothed Particle Hydrodynamics (SPH) method are highlighted. To avoid the appearance of these structures which are detrimental to the quality of the simulations, a specific transport velocity is introduced and its inclusion within an Arbitrary Lagrangian Eulerian (ALE) formalism is described. Unlike most of existing particle disordering/shifting methods, this formalism avoids the formation of these anisotropic structures while a full consistency with the original Euler or Navier–Stokes equations is maintained. The gain in accuracy, convergence and numerical diffusion of this formalism is shown and discussed through its application to various challenging test cases.

### `lind2012`

- **file:** `lind2012_isph-free-surface-diffusion-shifting.pdf`
- **title:** Incompressible smoothed particle hydrodynamics for free-surface flows: A generalised diffusion-based algorithm for stability and validations for impulsive flows and propagating waves
- **authors:** S. J. Lind, R. Xu, P. K. Stansby and B. D. Rogers
- **venue:** *Journal of Computational Physics* 231(4):1499–1523, 2012
- **doi:** [10.1016/j.jcp.2011.10.027](https://doi.org/10.1016/j.jcp.2011.10.027)
- **relevance:** The Fick's-law PST every later shifting law descends from, this codebase's included — `michel2022` Table 1's first row and `PST_ALE_PLAN.md` Part 1.1's. Its `0.2h` displacement cap is the ancestor of `delta.py`'s own limiter, and its `U_char = h/Δt` is the one entry in that table whose characteristic velocity is not built from a sound speed.
- **abstract from:** PDF p.1

> The incompressible smoothed particle hydrodynamics (ISPH) method with projectionbased pressure correction has been shown to be highly accurate and stable for internal flows and, importantly for many problems, the pressure field is virtually noise-free in contrast to the weakly compressible SPH approach (Xu et al., 2009 [31]). However for almost inviscid fluids instabilities at the free surface occur due to errors associated with the truncated kernels. A new algorithm is presented which remedies this issue, giving stable and accurate solutions to both internal and free-surface flows. Generalising the particle shifting approach of Xu et al. (2009) [31], the algorithm is based upon Fick’s law of diffusion and shifts particles in a manner that prevents highly anisotropic distributions and the onset of numerical instability. The algorithm is validated against analytical solutions for an internal flow at higher Reynolds numbers than previously, the flow due to an impulsively started plate and highly accurate solutions for wet bed dam break problems at zero and small times. The method is then validated for progressive regular waves with paddle motion defined by linear theory. The accurate predictions demonstrate the effectiveness of the algorithm in stabilising solutions and minimising the surface instabilities generated by the inevitable errors associated with truncated kernels. The test cases are thought to provide a more thorough quantitative validation than previously undertaken.

### `quinlan2006`

- **file:** `quinlan2006_truncation-error-mesh-free-particle-methods.pdf`
- **title:** Truncation error in mesh-free particle methods
- **authors:** N. J. Quinlan, M. Basa and M. Lastiwka
- **venue:** *International Journal for Numerical Methods in Engineering* 66(13):2064–2085, 2006
- **doi:** [10.1002/nme.1617](https://doi.org/10.1002/nme.1617)
- **relevance:** Why `michel2022`'s consistency requirement is a real condition rather than a formality. SPH error depends on `h` and `Δx/h` *separately*, so refining at the fixed `Δx/h` every SPH code actually uses is its own convergence question — and that is the limit a PST has to vanish in. `β = (R/Δx)³` comes from counterbalancing the lowest-degree term of this expansion.
- **abstract from:** Crossref (publisher-deposited JATS abstract)
- **note:** Author order here is the DOI record's; `michel2022`'s reference list gives it as Quinlan, Lastiwka, Basa.

> A truncation error analysis has been developed for the approximation of spatial derivatives in smoothed particle hydrodynamics (SPH) and related first‐order consistent methods such as the first‐order form of the reproducing kernel particle method. Error is shown to depend on both the smoothing length h and the ratio of particle spacing to smoothing length, Δ x / h . For uniformly spaced particles in one dimension, analysis shows that as h is reduced while maintaining constant Δ x / h , error decays as h 2 until a limiting discretization error is reached, which is independent of h . If Δ x / h is reduced while maintaining constant h (i.e. if the number of neighbours per particle is increased), error decreases at a rate which depends on the kernel function's smoothness. When particles are distributed non‐uniformly, error can grow as h is reduced with constant Δ x / h . First‐order consistent methods are shown to remove this divergent behaviour. Numerical experiments confirm the theoretical analysis for one dimension, and indicate that the main results are also true in three dimensions. This investigation highlights the complexity of error behaviour in SPH, and shows that the roles of both h and Δ x / h must be considered when choosing particle distributions and smoothing lengths. Copyright © 2005 John Wiley & Sons, Ltd.

### `vanleer1979`

- **file:** `vanleer1979_towards-the-ultimate-conservative-difference-scheme-v.pdf`
- **title:** Towards the Ultimate Conservative Difference Scheme. V. A Second-Order Sequel to Godunov's Method
- **authors:** Bram van Leer
- **venue:** *Journal of Computational Physics* 32(1):101–136, 1979
- **doi:** [10.1016/0021-9991(79)90145-1](https://doi.org/10.1016/0021-9991(79)90145-1)
- **relevance:** MUSCL: the reconstruction that builds the left/right states a Riemann solver needs, which `michel2022` Secs. 4.2.1–4.2.2 use for both ALE schemes. `PST_ALE_PLAN.md` stage B. Its monotonicity algorithms are the family `modules/crk/limiter.py`'s van Leer limiter already belongs to, so half the machinery is here.
- **abstract from:** PDF p.1

> A method of second-order accuracy is described for integrating the equations of ideal compressible flow. The method is based on the integral conservation laws and is dissipative, so that it can be used across shocks. The heart of the method is a one-dimensional Lagrangean scheme that may be regarded as a second-order sequel to Godunov’s method. The second-order accuracy is achieved by taking the distributions of the state quantities inside a gas slab to be linear, rather than uniform as in Godunov’s method. The Lagrangean results are remapped with least-squares accuracy onto the desired Euler grid in a separate step. Several monotonicity algorithms are applied to ensure positivity, monotonicity and nonlinear stability. Higher dimensions are covered through time splitting. Numerical results for one-dimensional and two-dimensional flows are presented, demonstrating the efficiency of the method. The paper concludes with a summary of the results of the whole series “Towards the Ultimate Conservative Difference Scheme.”
### `antuono2021`

- **file:** `antuono2021_delta-ale-sph-model.pdf`
- **title:** The δ-ALE-SPH model: An arbitrary Lagrangian-Eulerian framework for the δ-SPH model with particle shifting technique
- **authors:** M. Antuono, P. N. Sun, S. Marrone and A. Colagrossi
- **venue:** *Computers & Fluids* 216:104806, 2021
- **doi:** [10.1016/j.compfluid.2020.104806](https://doi.org/10.1016/j.compfluid.2020.104806)
- **relevance:** The route to an ALE formalism that does **not** need a Riemann solver — primitive variables (ρ, u) and the standard weakly-compressible operators, where `vila1999` and `parshikov2002` need conservative variables and Riemann fluxes inside the spatial operators. Two things follow for this codebase. Its §5 constant-mass variant is *exactly* `ShiftProperties.correctdrhodt`/`correctdvdt` (`docs/historic_plans/WCSPH_SHIFTING_PLAN.md`), and the paper states plainly that such variants "cannot be regarded as ALE schemes on their own" — which settles what we do and do not already have. And its central finding is that the naive ALE-SPH is *unstable* (unbounded volume growth where the strain rate is high) until diffusion is added to **both** the density and the mass equations. `PST_ALE_PLAN.md` stage B′.
- **abstract from:** PDF p.1
- **text-layer:** `weaklycompressible` -> `weakly compressible`
- **text-layer:** `by modi given by a particle shifting fying the` -> `by modifying the`
- **text-layer:** `through a velocity u technique pst` -> `through a velocity u given by a particle shifting technique pst`

> The behaviour of a weakly-compressible SPH scheme obtained by rewriting the Navier-Stokes equations in an arbitrary Lagrangian-Eulerian (ALE) format is studied. Differently from previous works on ALE, which generally adopt conservative variables (i.e. mass and momentum) and rely on the use of Riemann solvers inside the spatial operators, the proposed model is expressed in terms of primitive variables (i.e. density and velocity) and is written by using the standard differential formulations of the weakly-compressible SPH schemes. Similarly to ALE-SPH models, the arbitrary velocity field is obtained by modifying the pure Lagrangian velocity of the material point through a velocity δu given by a Particle Shifting Technique (PST). We show that the above-mentioned ALE-SPH equations are, however, unstable when they are integrated in time. The instability appears in the form of large volume variations in those fluid regions characterised by high velocity strain rates. Nonetheless, the scheme can be stabilised if appropriate diffusion terms are included in both the equations of density and mass. This latter scheme, hereinafter called δ-ALE-SPH scheme, is validated against reference benchmark test-cases: the viscous flow around an inclined elliptical cylinder, the lid-driven cavity and a dam-break flow impacting a vertical wall.

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

## Geometric integration, operator splitting & conformal symplecticity

Background for `../../warpSPHIntegrators/SPLITTING_PLAN.md`'s conservative/dissipative
operator-splitting plan, added 2026-09-15.

### `yoshida1990`

- **file:** `yoshida1990_higher-order-symplectic-integrators.pdf`
- **title:** Construction of higher order symplectic integrators
- **authors:** Haruo Yoshida
- **venue:** *Physics Letters A* 150(5-7):262-268, 1990
- **doi:** [10.1016/0375-9601(90)90092-3](https://doi.org/10.1016/0375-9601(90)90092-3)
- **relevance:** The Yoshida triple-jump: composes a symmetric order-2 base method into order 4/6/8 with explicit real coefficients. SPLITTING_PLAN.md §2.5 uses its coefficients (gamma1 ≈ 1.351, gamma0 ≈ -1.702) as the reference point for why a dissipative half-step forces a backward sub-step under this composition.
- **abstract from:** PDF p.1
- **text-layer:** `reversiblesymplectic` -> `reversible symplectic`
- **text-layer:** `symplecticintegrator with` -> `symplectic integrator with`
- **text-layer:** `symplecticintegrators with fewersteps` -> `symplectic integrators with fewer steps`
- **text-layer:** `coefficientsare givenby solvinga set of simultaneousalgebraicequations` -> `coefficients are given by solving a set of simultaneous algebraic equations`

> For Hamiltonian systemsof the form H= T(p) + V(q) a method is shown to construct explicit and time reversible symplectic
> integrators of higher order. For any even order there exists at least one symplectic integrator with exact coefficients.The simplest
> one is the 4th order integrator which agrees with one found by Forest and by Ned. For 6th and 8th orders, symplectic integrators
> with fewer steps are obtained, for which the coefficients are given by solving a set of simultaneous algebraic equations numerically.

### `suzuki1990`

- **file:** `suzuki1990_fractal-decomposition-exponential-operators.pdf`
- **title:** Fractal decomposition of exponential operators with applications to many-body theories and Monte Carlo simulations
- **authors:** Masuo Suzuki
- **venue:** *Physics Letters A* 146(6):319-323, 1990
- **doi:** [10.1016/0375-9601(90)90962-N](https://doi.org/10.1016/0375-9601(90)90962-N)
- **relevance:** The 5-stage fourth-order composition SPLITTING_PLAN.md §2.5 recommends over Yoshida's for a dissipative half: its largest backward coefficient (w3 ≈ -0.658) is smaller in magnitude than Yoshida's (gamma0 ≈ -1.702), so it tolerates roughly 2.6x the viscous stiffness before the reversed sub-step amplifies comparably.
- **abstract from:** PDF p.1

> A new systematic scheme ofdecomposition of exponential operators is presented, namely exp [x(A + B)] = S,,, (x) + 0 (x'"+ I)
> for any positive integer m, where S,,,(x) =etIAet~e~e1~...etMA. A general scheme of construction of {t,} is given explicitly. The
> decomposition cap [x(A +B)] = [S,,,(x/n) 1 "+0 (x"'~In") yields a new efficient approach to quantum Monte Carlo simulations.

### `mclachlan2001`

- **file:** `mclachlan2001_conformal-hamiltonian-systems.pdf`
- **title:** Conformal Hamiltonian systems
- **authors:** Robert McLachlan and Matthew Perlmutter
- **venue:** *Journal of Geometry and Physics* 39(4):276-300, 2001
- **doi:** [10.1016/S0393-0440(01)00020-1](https://doi.org/10.1016/S0393-0440(01)00020-1)
- **relevance:** The definition SPLITTING_PLAN.md §2.4 cites for why constant-gamma damping (F_visc = -gamma*v) contracts the symplectic form by an exact factor exp(-gamma*t) -- the generalized det(D-Phi_h) check that section proposes.
- **abstract from:** PDF p.1

> Vector fields whose flow preserves a symplectic form up to a constant, such as simple mechanical
> systems with friction, are called “conformal”. We develop a reduction theory for symmetric conformal Hamiltonian systems, analogous to symplectic reduction theory. This entire theory extends
> naturally to Poisson systems: given a symmetric conformal Poisson vector field, we show that it
> induces two reduced conformal Poisson vector fields, again analogous to the dual pair construction
> for symplectic manifolds. Conformal Poisson systems form an interesting infinite-dimensional Lie
> algebra of foliate vector fields. Manifolds supporting such conformal vector fields include cotangent bundles, Lie–Poisson manifolds, and their natural quotients.

### `bhatt2016`

- **file:** `bhatt2016_conformal-symplectic-damped-hamiltonian.pdf`
- **title:** Second order conformal symplectic schemes for damped Hamiltonian systems
- **authors:** Ashish Bhatt, Dwayne Floyd and Brian E. Moore
- **venue:** *Journal of Scientific Computing* 66(3):1234-1259, 2016
- **doi:** [10.1007/s10915-015-0062-z](https://doi.org/10.1007/s10915-015-0062-z)
- **relevance:** Constructs Stormer-Verlet/implicit-midpoint schemes for exactly this codebase's split problem -- a separable H(q,p) = T(p) + V(q) plus linear damping -- and proves they preserve the exact dissipation-of-symplecticity rate. SPLITTING_PLAN.md §2.4's conformal check generalizes the direct 1-DOF phase-Jacobian test (tests/test_hamiltonian.py) using exactly this paper's target property.
- **abstract from:** PDF p.1

> Numerical methods for solving linearly damped Hamiltonian systems are constructed using the popular Störmer–Verlet and implicit midpoint methods. Each method is
> shown to preserve dissipation of symplecticity and dissipation of angular momentum of an
> N -body system with pairwise distance dependent interactions. Necessary and sufficient conditions for second order accuracy are derived. Analysis for linear equations gives explicit
> relationships between the damping parameter and the step size to reveal when the methods
> are most advantageous; essentially, the damping rate of the numerical solution is exactly preserved under these conditions. The methods are applied to several model problems, both ODEs
> and PDEs. Additional structure preservation is discovered for the discretized PDEs, in one
> case dissipation in total linear momentum and in another dissipation in mass are preserved
> by the methods. The numerical results, along with comparisons to standard Runge–Kutta
> methods and another structure-preserving method, demonstrate the usefulness and strengths
> of the methods.

### `goldman1996`

- **file:** `goldman1996_nth-order-operator-splitting.pdf`
- **title:** Nth-order operator splitting schemes and nonreversible systems
- **authors:** Daniel Goldman and Tasso J. Kaper
- **venue:** *SIAM Journal on Numerical Analysis* 33(1):349-367, 1996
- **doi:** [10.1137/0733018](https://doi.org/10.1137/0733018)
- **relevance:** The theorem SPLITTING_PLAN.md §2.5 rests on: no operator-splitting composition of order greater than 2 with real coefficients has all-positive coefficients, so any order-4+ composition run against a dissipative operator necessarily takes a backward sub-step somewhere.
- **abstract from:** DOI record (Crossref)

> This paper is concerned with partitioned Nth-order accurate split-operator schemes built using M distinct solution operators for semidiscrete equations of the form du_j/dt = f_j (u_k) + f_j (u_k), which arise, among others, from constant coefficient parabolic partial differential equations. We prove that, for every N > 3 and M > 2, each solution operator must be applied for at least one backward fractional time step during each complete time step. This result has important consequences for applications to the complex Ginzburg-Landau equation with periodic boundary conditions and other partial differential equations with both reversible and nonreversible components. Furthermore, for the special case of N = 3 and M = 2, we analytically determine all possible schemes.

### `blanes2008`

- **file:** `blanes2008_splitting-and-composition-methods.pdf`
- **title:** Splitting and composition methods in the numerical integration of differential equations
- **authors:** Sergio Blanes, Fernando Casas and Ander Murua
- **venue:** *Boletín de la Sociedad Española de Matemática Aplicada* 45:89-145, 2008
- **arXiv:** [0812.0377](https://arxiv.org/abs/0812.0377)
- **relevance:** The survey SPLITTING_PLAN.md's references lean on for the general composition/splitting landscape: covers Lie-Trotter and Strang as the order-1/2 base cases, the composition theorem's symmetry requirement, and the processing techniques (§2.5's "escape routes") that do not evade Goldman-Kaper for order >= 3.
- **abstract from:** arXiv API (0812.0377); the published Bol. Soc. Esp. Mat. Apl. record carries no abstract

> We provide a comprehensive survey of splitting and composition methods for the numerical integration of ordinary differential equations (ODEs). Splitting methods constitute an appropriate choice when the vector field associated with the ODE can be decomposed into several pieces and each of them is integrable or at least easier to integrate than the original problem.

### `hairer2006`

- **file:** `hairer2006_geometric-numerical-integration.pdf`
- **title:** Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations
- **authors:** Ernst Hairer, Christian Lubich and Gerhard Wanner
- **venue:** 2nd ed., Springer Series in Computational Mathematics 31 (Springer, 2006)
- **isbn:** 978-3-540-30663-4
- **relevance:** Carries essentially all of SPLITTING_PLAN.md's theory -- II.4 (composition methods and the order theorem), II.5 (splitting, Strang, the Baker-Campbell-Hausdorff expansion), III.4 (backward error analysis for splitting), V.4.1 (symmetric projection, the mechanism behind §2.7.1's substep-local compatible-energy rewrite).
- **abstract:** none (a book; no abstract to quote)

### `toro2009`

- **file:** `toro2009_riemann-solvers-and-numerical-methods.pdf`
- **title:** Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
- **authors:** Eleuterio F. Toro
- **venue:** 3rd ed. (Springer, 2009)
- **isbn:** 978-3-540-25202-3
- **relevance:** The HLLC reference for the MFM/MFV direction (PESPH_PLAN.md §7): the Riemann solver structural correspondence between CRKSPH's conservative differencing (Eq. 35) and the MFM effective-face flux points here for the piece this codebase does not yet have.
- **abstract:** none (a book; no abstract to quote)


## PESPH, CRKSPH, compSPH & meshless hydrodynamics (MFM)

Background for `PESPH_PLAN.md`, added 2026-09-15.

### `hopkins2013`

- **file:** `hopkins2013_general-class-lagrangian-sph.pdf`
- **title:** A general class of Lagrangian smoothed particle hydrodynamics methods and implications for fluid mixing problems
- **authors:** Philip F. Hopkins
- **venue:** *Monthly Notices of the Royal Astronomical Society* 428(4):2840-2856, 2013
- **doi:** [10.1093/mnras/sts210](https://doi.org/10.1093/mnras/sts210)
- **relevance:** The pressure-entropy PESPH paper. PESPH_PLAN.md §1.2/§3.2 builds PESPH-A on this: pressure by summation over the entropic function A rather than energy, which is what makes the scheme separable (SPLITTING_PLAN.md §2.3.1) and the recommended first target for the splitting integrator work.
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Various formulations of smoothed particle hydrodynamics (SPH) have been proposed, intended to resolve certain difficulties in the treatment of fluid mixing instabilities. Most have involved changes to the algorithm which either introduces artificial correction terms or violates what is arguably the greatest advantage of SPH over other methods: manifest conservation of energy, entropy, momentum and angular momentum. Here, we show how a class of alternative SPH equations of motion (EOM) can be derived self-consistently from a discrete particle Lagrangian – guaranteeing manifest conservation – in a manner which tremendously improves treatment of these instabilities and contact discontinuities. Saitoh & Makino recently noted that the volume element used to discretize the EOM does not need to explicitly invoke the mass density (as in the ‘standard’ approach); we show how this insight, and the resulting degree of freedom, can be incorporated into the rigorous Lagrangian formulation that retains ideal conservation properties and includes the ‘∇h’ terms that account for variable smoothing lengths. We derive a general EOM for any choice of volume element (particle ‘weights’) and method of determining smoothing lengths. We then specify this to a ‘pressure–entropy formulation’ which resolves problems in the traditional treatment of fluid interfaces. Implementing this in a new version of the gadget code, we show it leads to good performance in mixing experiments (e.g. Kelvin–Helmholtz and ‘blob’ tests). And conservation is maintained even in strong shock/blastwave tests, where formulations without manifest conservation produce large errors. This also improves the treatment of subsonic turbulence and lessens the need for large kernel particle numbers. The code changes are trivial and entail no additional numerical expense. This provides a general framework for self-consistent derivation of different ‘flavours’ of SPH.

### `hopkins2015`

- **file:** `hopkins2015_new-class-meshfree-hydrodynamic-methods.pdf`
- **title:** A new class of accurate, mesh-free hydrodynamic simulation methods
- **authors:** Philip F. Hopkins
- **venue:** *Monthly Notices of the Royal Astronomical Society* 450(1):53-110, 2015
- **doi:** [10.1093/mnras/stv195](https://doi.org/10.1093/mnras/stv195)
- **relevance:** MFM/MFV derivation (Appendix), and Appendix F2's pressure-energy PSPH -- the variant PESPH_PLAN.md builds PESPH-E on. Already cited from this repository's warpSPHIntegrators companion; the copy is duplicated here because both codebases cite it directly.
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present two new Lagrangian methods for hydrodynamics, in a systematic comparison with moving-mesh, smoothed particle hydrodynamics (SPH), and stationary (non-moving) grid methods. The new methods are designed to simultaneously capture advantages of both SPH and grid-based/adaptive mesh refinement (AMR) schemes. They are based on a kernel discretization of the volume coupled to a high-order matrix gradient estimator and a Riemann solver acting over the volume ‘overlap’. We implement and test a parallel, second-order version of the method with self-gravity and cosmological integration, in the code gizmo:1 this maintains exact mass, energy and momentum conservation; exhibits superior angular momentum conservation compared to all other methods we study; does not require ‘artificial diffusion’ terms; and allows the fluid elements to move with the flow, so resolution is automatically adaptive. We consider a large suite of test problems, and find that on all problems the new methods appear competitive with moving-mesh schemes, with some advantages (particularly in angular momentum conservation), at the cost of enhanced noise. The new methods have many advantages versus SPH: proper convergence, good capturing of fluid-mixing instabilities, dramatically reduced ‘particle noise’ and numerical viscosity, more accurate sub-sonic flow evolution, and sharp shock-capturing. Advantages versus non-moving meshes include: automatic adaptivity, dramatically reduced advection errors and numerical overmixing, velocity-independent errors, accurate coupling to gravity, good angular momentum conservation and elimination of ‘grid alignment’ effects. We can, for example, follow hundreds of orbits of gaseous discs, while AMR and SPH methods break down in a few orbits. However, fixed meshes minimize ‘grid noise’. These differences are important for a range of astrophysical problems.

### `saitoh2013`

- **file:** `saitoh2013_density-independent-sph.pdf`
- **title:** A density-independent formulation of smoothed particle hydrodynamics
- **authors:** Takayuki R. Saitoh and Junichiro Makino
- **venue:** *The Astrophysical Journal* 768(1):44, 2013
- **doi:** [10.1088/0004-637X/768/1/44](https://doi.org/10.1088/0004-637X/768/1/44)
- **relevance:** DISPH -- density-independent SPH -- which Hopkins (2013) builds the pressure-entropy formulation on and which Frontiere et al. cite as [60], the origin of the pressure-weighting idea PESPH generalizes.
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> The standard formulation of the smoothed particle hydrodynamics (SPH) assumes that the local density distribution is differentiable. This assumption is used to derive the spatial derivatives of other quantities. However, this assumption breaks down at the contact discontinuity. At the contact discontinuity, the density of the low-density side is overestimated while that of the high-density side is underestimated. As a result, the pressure of the low-density (high-density) side is overestimated (underestimated). Thus, unphysical repulsive force appears at the contact discontinuity, resulting in the effective surface tension. This tension suppresses fluid instabilities. In this paper, we present a new formulation of SPH, which does not require the differentiability of density. Instead of the mass density, we adopt the internal energy density (pressure) and its arbitrary function, which are smoothed quantities at the contact discontinuity, as the volume element used for the kernel integration. We call this new formulation density-independent SPH (DISPH). It handles the contact discontinuity without numerical problems. The results of standard tests such as the shock tube, Kelvin–Helmholtz and Rayleigh–Taylor instabilities, point-like explosion, and blob tests are all very favorable to DISPH. We conclude that DISPH solved most of the known difficulties of the standard SPH, without introducing additional numerical diffusion or breaking the exact force symmetry or energy conservation. Our new SPH includes the formulation proposed by Ritchie & Thomas as a special case. Our formulation can be extended to handle a non-ideal gas easily.

### `springel2002`

- **file:** `springel2002_sph-entropy-equation.pdf`
- **title:** Cosmological smoothed particle hydrodynamics simulations: the entropy equation
- **authors:** V. Springel and L. Hernquist
- **venue:** *Monthly Notices of the Royal Astronomical Society* 333(3):649-664, 2002
- **doi:** [10.1046/j.1365-8711.2002.05445.x](https://doi.org/10.1046/j.1365-8711.2002.05445.x)
- **relevance:** The entropy formulation and the canonical grad-h (Omega_i) derivation (SPLITTING_PLAN.md §2.3.1) -- the reference for why carrying entropy rather than internal energy keeps a compressible SPH scheme's conservative half a pure function of position. PESPH_PLAN.md §7.1 also cites it against the FIRE/GIZMO adoption history.
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We discuss differences in simulation results that arise between the use of either the thermal energy or the entropy as an independent variable in smoothed particle hydrodynamics (SPH). In this context, we derive a new version of SPH that, when appropriate, manifestly conserves both energy and entropy if smoothing lengths are allowed to adapt freely to the local mass resolution. To test various formulations of SPH, we consider point-like energy injection, as in certain models of supernova feedback, and find that powerful explosions are well represented by SPH even when the energy is deposited into a single particle, provided that the entropy equation is integrated. If the thermal energy is instead used as an independent variable, unphysical solutions can be obtained for this problem. We also examine the radiative cooling of gas spheres that collapse and virialize in isolation, and of haloes that form in cosmological simulations of structure formation. When applied to these problems, the thermal energy version of SPH leads to substantial overcooling in haloes that are resolved with up to a few thousand particles, while the entropy formulation is biased only moderately low for these haloes under the same circumstances. For objects resolved with much larger particle numbers, the two approaches yield consistent results. We trace the origin of the differences to systematic resolution effects in the outer parts of cooling flows. When the thermal energy equation is integrated and the resolution is low, the compressional heating of the gas in the inflow region is underestimated, violating entropy conservation and improperly accelerating cooling. The cumulative effect of this overcooling can be significant. In cosmological simulations of moderate size, we find that the fraction of baryons which cool and condense can be reduced by up to a factor ∼2 if the entropy equation is employed rather than the thermal energy equation, partly explaining discrepancies with semi-analytic treatments of galaxy formation. We also demonstrate that the entropy method leads to a greatly reduced scatter in the density–temperature relation of the low-density Lyα forest relative to the thermal energy approach, in accord with theoretical expectations.

### `frontiere2017`

- **file:** `frontiere2017_crksph.pdf`
- **title:** CRKSPH -- A Conservative Reproducing Kernel Smoothed Particle Hydrodynamics Scheme
- **authors:** Nicholas Frontiere, Cody D. Raskin and J. Michael Owen
- **venue:** *Journal of Computational Physics* 332:160-209, 2017
- **doi:** [10.1016/j.jcp.2016.12.004](https://doi.org/10.1016/j.jcp.2016.12.004)
- **relevance:** The CRKSPH/compSPH/PESPH source paper. §3 CRKSPH, §3.3 the compatible energy discretization (PESPH_PLAN.md §4.2's subject), Appendix B (why CRKSPH is not variational, §4.1), Appendix E compSPH, Appendix G PESPH (pressure-energy). Eq. (24)'s partition-of-unity condition is PESPH_PLAN.md §7.2's structural bridge to MFM. Matches the copy already in warpSPHIntegrators/literature/.
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a formulation of smoothed particle hydrodynamics (SPH) that utilizes a first-order consistent reproducing kernel, a smoothing function that exactly interpolates linear fields with particle tracers. Previous formulations using reproducing kernel (RK) interpolation have had difficulties maintaining conservation of momentum due to the fact the RK kernels are not, in general, spatially symmetric. Here, we utilize a reformulation of the fluid equations such that mass, linear momentum, and energy are all rigorously conserved without any assumption about kernel symmetries, while additionally maintaining approximate angular momentum conservation. Our approach starts from a rigorously consistent interpolation theory, where we derive the evolution equations to enforce the appropriate conservation properties, at the sacrifice of full consistency in the momentum equation. Additionally, by exploiting the increased accuracy of the RK method's gradient, we formulate a simple limiter for the artificial viscosity that reduces the excess diffusion normally incurred by the ordinary SPH artificial viscosity. Collectively, we call our suite of modifications to the traditional SPH scheme Conservative Reproducing Kernel SPH, or CRKSPH. CRKSPH retains many benefits of traditional SPH methods (such as preserving Galilean invariance and manifest conservation of mass, momentum, and energy) while improving on many of the shortcomings of SPH, particularly the overly aggressive artificial viscosity and zeroth-order inaccuracy. We compare CRKSPH to two different modern SPH formulations (pressure based SPH and compatibly differenced SPH), demonstrating the advantages of our new formulation when modeling fluid mixing, strong shock, and adiabatic phenomena.

### `owen2014`

- **file:** `owen2014_compatibly-differenced-total-energy-sph.pdf`
- **title:** A compatibly differenced total energy conserving form of SPH
- **authors:** J. Michael Owen
- **venue:** *International Journal for Numerical Methods in Fluids* 75(11):749-774, 2014
- **doi:** [10.1002/fld.3912](https://doi.org/10.1002/fld.3912)
- **relevance:** Frontiere et al.'s [48] -- the origin of the compatible energy discretization modules/compSPH/balance.py and multistep.py implement, and the subject of PESPH_PLAN.md §4.2's finding that the pairwise work-partition is a symmetric projection in disguise.
- **abstract from:** DOI record (Crossref)

> We describe a modified form of smoothed particle hydrodynamics (SPH) in which the specific thermal energy equation is based on a compatibly differenced formalism, guaranteeing exact conservation of the total energy. We compare the errors and convergence rates of the standard and compatible SPH formalisms on a variety of shock test problems with analytic answers. We find that the new compatible formalism reliably achieves the expected first‐order convergence for these analytic shock tests and, in all cases, improves the accuracy of the numerical solution over the standard formalism. We also examine the performance of our new formalism on a more complicated applied problem: the diversion of an asteroid by a kinetic impactor. We find the compatible discretization demonstrates measurable improvement in the convergence of properties such as the deflection velocity in this kind of applied problem as well. Copyright © 2014 John Wiley & Sons, Ltd.

### `cullen2010`

- **file:** `cullen2010_inviscid-sph.pdf`
- **title:** Inviscid smoothed particle hydrodynamics
- **authors:** Lee Cullen and Walter Dehnen
- **venue:** *Monthly Notices of the Royal Astronomical Society* 408(2):669-683, 2010
- **doi:** [10.1111/j.1365-2966.2010.17158.x](https://doi.org/10.1111/j.1365-2966.2010.17158.x)
- **relevance:** The Cullen-Dehnen shock-detection switch PESPH's Appendix G specifies (ViscositySwitch.CullenDehnen2010, already registered in enumTypes.py).
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In smoothed particle hydrodynamics (SPH), artificial viscosity is necessary for the correct treatment of shocks, but often generates unwanted dissipation away from shocks. We present a novel method of controlling the amount of artificial viscosity, which uses the total time derivative of the velocity divergence as shock indicator and aims at completely eliminating viscosity away from shocks. We subject the new scheme to numerous tests and find that the method works at least as well as any previous technique in the strong-shock regime, but becomes virtually inviscid away from shocks, while still maintaining particle order. In particular sound waves or oscillations of gas spheres are hardly damped over many periods.

### `balsara1995`

- **file:** `balsara1995_von-neumann-stability-sph.pdf`
- **title:** Von Neumann stability analysis of smoothed particle hydrodynamics—suggestions for optimal algorithms
- **authors:** Dinshaw S. Balsara
- **venue:** *Journal of Computational Physics* 121(2):357-372, 1995
- **doi:** [10.1016/S0021-9991(95)90221-X](https://doi.org/10.1016/S0021-9991(95)90221-X)
- **relevance:** The Balsara (1989/1995) shock-detection switch Hopkins' F2 (2015) specifies alongside Cullen-Dehnen; the older of the two, cited by both PESPH descriptions.
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a von Neumann stability analysis of the equations of smoothed particle hydrodynamics (SPH) along with a critical discussion of various parts of the algorithm. The stability analysis is done without any major restrictions and, hence, models the full Euler equations in one dimension. This then allows us to deduce optimal ranges for parameters that need to be used in SPH. Thus we show that for the commonly used M5 spline the ratio of smoothing length to interparticle distance should range between 1.0 to 1.4. We also show that the linear artificial viscosity coefficient and the coefficient of spatial filtering have to be bounded. The results of this von Neumann stability analysis provide us with several suggestions for future algorithm improvement. Because the SPH method is so unique we provide, wherever possible, comparisons with more familiar and well-used high resolution finite difference methods.

### `dilts1999`

- **file:** `dilts1999_mlsph-consistency-and-stability.pdf`
- **title:** Moving-least-squares-particle hydrodynamics—I. Consistency and stability
- **authors:** Gary A. Dilts
- **venue:** *International Journal for Numerical Methods in Engineering* 44(8):1115-1155, 1999
- **doi:** [10.1002/(SICI)1097-0207(19990320)44:8<1115::AID-NME547>3.0.CO;2-L](https://doi.org/10.1002/(SICI)1097-0207(19990320)44:8<1115::AID-NME547>3.0.CO;2-L)
- **relevance:** Frontiere et al.'s [14] -- the MLSPH formalism CRKSPH's conservative differencing is derived from, and, per PESPH_PLAN.md §7.2, the same weak-formulation route MFM's effective face is built on. One paper behind both the scheme this codebase has and the one it is considering.
- **abstract from:** PDF p.1

> The Smooth-Particle-Hydrodynamics (SPH) method is derived in a novel manner by means of a Galerkin
> approximation applied to the Lagrangian equations of continuum mechanics as in the finite-element
> method. This derivation is modified to replace the SPH interpolant with the Moving-Least-Squares (MLS)
> interpolant of Lancaster and Saulkaskas, and define a new particle volume which ensures thermodynamic
> compatibility. A variable-rank modification of the MLS interpolants which retains their desirable summation properties is introduced to remove the singularities that occur when divergent flow reduces the number
> of neighbours of a particle to less than the minimum required. A surprise benefit of the Galerkin SPH
> derivation is a theoretical justification of a common ad hoc technique for variable-h SPH. The new MLSPH
> method is conservative if an anti-symmetric quadrature rule for the stiffness matrix elements can be supplied.
> In this paper, a simple one-point collocation rule is used to retain similarity with SPH, leading to
> a non-conservative method. Several examples document how MLSPH renders dramatic improvements due
> to the linear consistency of its gradients on three canonical difficulties of the SPH method: spurious

### `garciasenz2012`

- **file:** `garciasenz2012_integral-approach-gradients.pdf`
- **title:** Improving smoothed particle hydrodynamics with an integral approach to calculating gradients
- **authors:** D. García-Senz, R. M. Cabezón and J. A. Escartín
- **venue:** *Astronomy & Astrophysics* 538:A9, 2012
- **doi:** [10.1051/0004-6361/201117939](https://doi.org/10.1051/0004-6361/201117939)
- **relevance:** The integral-approximation gradient estimator PESPH_PLAN.md §7.5 recommends as the shared upgrade across monaghan/compSPH/crkSPH/PESPH, independent of the MFM-vs-splitting choice and a prerequisite for MFM's own reconstruction step.
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Context. The smoothed particle hydrodynamics (SPH) technique is a well-known numerical method that has been applied to simulate the evolution of a wide variety of systems. Modern astrophysical applications of the method rely on the Lagrangian formulation of fluid Euler equations, which is fully conservative. A different scheme, based on a matrix approach to the SPH equations is currently being used in computational fluid dynamics. These matrix formulations achieve better interpolations of the physical magnitudes but they are, in general, not fully conservative. The matrix approach to the Euler equations has never been used in astrophysics.

### `rosswog2015`

- **file:** `rosswog2015_boosting-accuracy-sph.pdf`
- **title:** Boosting the accuracy of SPH techniques: Newtonian and special-relativistic tests
- **authors:** S. Rosswog
- **venue:** *Monthly Notices of the Royal Astronomical Society* 448(4):3628-3664, 2015
- **doi:** [10.1093/mnras/stv225](https://doi.org/10.1093/mnras/stv225)
- **relevance:** The matrix-inversion gradient form, the other half of PESPH_PLAN.md §7.5's gradient upgrade alongside garciasenz2012; the two combine in MAGMA2 (rosswog2020).
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We study the impact of different discretization choices on the accuracy of smoothed particle hydrodynamics (SPH) and we explore them in a large number of Newtonian and special-relativistic benchmark tests. As a first improvement, we explore a gradient prescription that requires the (analytical) inversion of a small matrix. For a regular particle distribution, this improves gradient accuracies by approximately 10 orders of magnitude and the SPH formulations with this gradient outperform the standard approach in all benchmark tests. Secondly, we demonstrate that a simple change of the kernel function can substantially increase the accuracy of an SPH scheme. While the ‘standard’ cubic spline kernel generally performs poorly, the best overall performance is found for a high-order Wendland kernel which allows for only very little velocity noise and enforces a very regular particle distribution, even in highly dynamical tests. Thirdly, we explore new SPH volume elements that enhance the treatment of fluid instabilities and, last, but not least, we design new dissipation triggers. They switch on near shocks and in regions where the flow – without dissipation – starts to become noisy. The resulting new SPH formulation yields excellent results even in challenging tests where standard techniques fail completely.

### `rosswog2020`

- **file:** `rosswog2020_magma2.pdf`
- **title:** The Lagrangian hydrodynamics code magma2
- **authors:** S. Rosswog
- **venue:** *Monthly Notices of the Royal Astronomical Society* 498(3):4230-4255, 2020
- **doi:** [10.1093/mnras/staa2591](https://doi.org/10.1093/mnras/staa2591)
- **relevance:** MAGMA2: integral-approximation + matrix-inversion gradients with slope-limited reconstruction, Wendland C6 with ~300 neighbours. The concrete code PESPH_PLAN.md §7.5 points at as the combined upgrade.
- **abstract from:** DOI record (Crossref)

> We present the methodology and performance of the new Lagrangian hydrodynamics code magma2, a smoothed particle hydrodynamics (SPH) code that benefits from a number of non-standard enhancements. By default it uses high-order smoothing kernels and wherever gradients are needed, they are calculated via accurate matrix inversion techniques, but a more conventional formulation with kernel gradients has also been implemented for comparison purposes. We also explore a matrix inversion formulation of SPH with a symmetrization in the particle indices that is not frequently used. We find interesting advantages of this formulation in some of the tests, for example, a substantial reduction of surface tension effects for non-ideal particle setups and more accurate peak densities in Sedov blast waves. magma2 uses artificial viscosity, but enhanced by techniques that are commonly used in finite-volume schemes such as reconstruction and slope limiting. While simple to implement, this approach efficiently suppresses particle noise, but at the same time drastically reduces dissipation in locations where it is not needed and actually unwanted. We demonstrate the performance of the new code in a number of challenging benchmark tests including, for example, multidimensional vorticity creating Schulz–Rinne-type Riemann problems and more astrophysical tests such as a collision between two stars to demonstrate its robustness and excellent conservation properties.

### `cabezon2017`

- **file:** `cabezon2017_sphynx.pdf`
- **title:** SPHYNX: an accurate density-based SPH method for astrophysical applications
- **authors:** R. M. Cabezón, D. García-Senz and J. Figueira
- **venue:** *Astronomy & Astrophysics* 606:A78, 2017
- **doi:** [10.1051/0004-6361/201630208](https://doi.org/10.1051/0004-6361/201630208)
- **relevance:** The production code built on garciasenz2012's integral-approach gradients -- a worked example of the §7.5 gradient upgrade at full-code scale, complementing MAGMA2.
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Aims. Hydrodynamical instabilities and shocks are ubiquitous in astrophysical scenarios. Therefore, an accurate numerical simulation of these phenomena is mandatory to correctly model and understand many astrophysical events, such as supernovas, stellar collisions, or planetary formation. In this work, we attempt to address many of the problems that a commonly used technique, smoothed particle hydrodynamics (SPH), has when dealing with subsonic hydrodynamical instabilities or shocks. To that aim we built a new SPH code named SPHYNX, that includes many of the recent advances in the SPH technique and some other new ones, which we present here.

### `gaburov2011`

- **file:** `gaburov2011_weighted-particle-mhd.pdf`
- **title:** Astrophysical weighted particle magnetohydrodynamics
- **authors:** Evghenii Gaburov and Keigo Nitadori
- **venue:** *Monthly Notices of the Royal Astronomical Society* 414(1):129-154, 2011
- **doi:** [10.1111/j.1365-2966.2011.18313.x](https://doi.org/10.1111/j.1365-2966.2011.18313.x)
- **relevance:** MFM-tier reference: a weighted-particle-method treatment adjacent to the Lanson-Vila / Hopkins MFM lineage, cited in PESPH_PLAN.md §7's MFM reading list.
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> This paper presents applications of a weighted meshless scheme for conservation laws to the Euler equations and the equations of ideal magnetohydrodynamics (MHD). The divergence constraint of the latter is maintained to the truncation error by a new meshless divergence cleaning procedure. The physics of the interaction between the particles is described by a one-dimensional Riemann problem in a moving frame. As a result, the necessary diffusion which is required to treat dissipative processes is added automatically. Our scheme therefore has no free parameters that control the physics of interparticle interaction, with the exception of the number of interacting neighbours which control the resolution and accuracy. The resulting equations have a form similar to smoothed particle hydrodynamics (SPH) equations, and therefore existing SPH codes can be used to implement the weighed particle scheme. The scheme is validated in several hydrodynamic and MHD test cases. In particular, we demonstrate for the first time the ability of a meshless MHD scheme to model magnetorotational instability in accretion discs.

### `lanson2008`

- **file:** `lanson2008_renormalized-meshfree-schemes.pdf`
- **title:** Renormalized meshfree schemes I: consistency, stability, and hybrid methods for conservation laws
- **authors:** Nathalie Lanson and Jean-Paul Vila
- **venue:** *SIAM Journal on Numerical Analysis* 46(4):1912-1934, 2008
- **doi:** [10.1137/S0036142903427718](https://doi.org/10.1137/S0036142903427718)
- **relevance:** The mathematical foundation Hopkins builds MFM on -- PESPH_PLAN.md §7.2's structural correspondence between CRKSPH's partition-of-unity interpolant and the MFM effective face traces back to this paper's renormalized meshfree derivatives.
- **abstract from:** DOI record (Crossref)

> This paper is devoted to the study of a new kind of meshfree scheme based on a new class of meshfree derivatives: the renormalized meshfree derivatives, which improve the consistency of the original weighted particle methods. The weak renormalized meshfree scheme, built from the weak formulation of general conservation laws, turns out to be L^2 stable under some geometrical conditions on the distribution of particles and some regularity conditions of the transport field. A time discretization is then performed by analogy with finite volume methods, and the L^1, L^infinity, and BV stabilities of the obtained time discretized scheme are studied. From the same analogy with finite volume methods, a hybrid particle scheme is built using the Godunov method and is numerically compared to the weak renormalized scheme.

### `groth2023`

- **file:** `groth2023_opengadget3-meshless-finite-mass.pdf`
- **title:** The cosmological simulation code OpenGadget3 -- implementation of meshless finite mass
- **authors:** Frederick Groth, Ulrich P. Steinwandel, Milena Valentini and Klaus Dolag
- **venue:** *Monthly Notices of the Royal Astronomical Society* 526(1):616-644, 2023
- **doi:** [10.1093/mnras/stad2717](https://doi.org/10.1093/mnras/stad2717)
- **relevance:** A worked port of MFM into an existing SPH code -- the closest thing to a template for PESPH_PLAN.md §7's MFM direction, and probably the highest practical value in that reading tier.
- **abstract from:** PDF p.1

> Subsonic turbulence plays a major role in determining properties of the intra cluster medium (ICM). We introduce a new
> Meshless Finite Mass (MFM) implementation in OpenGadget3 and apply it to this specific problem. To this end, we present
> a set of test cases to validate our implementation of the MFM framework in our code. These include but are not limited to: the
> soundwave and Kepler disk as smooth situations to probe the stability, a Rayleigh-Taylor and Kelvin-Helmholtz instability as
> popular mixing instabilities, a blob test as more complex example including both mixing and shocks, shock tubes with various
> Mach numbers, a Sedov blast wave, different tests including self-gravity such as gravitational freefall, a hydrostatic sphere,
> the Zeldovich-pancake, and a 1015 galaxy cluster as cosmological application. Advantages over SPH include increased
> mixing and a better convergence behavior. We demonstrate that the MFM-solver is robust, also in a cosmological context. We
> show evidence that the solver performs extraordinarily well when applied to decaying subsonic turbulence, a problem very
> difficult to handle for many methods. MFM captures the expected velocity power spectrum with high accuracy and shows a
> good convergence behavior. Using MFM or SPH within OpenGadget3 leads to a comparable decay in turbulent energy due
> to numerical dissipation. When studying the energy decay for different initial turbulent energy fractions, we find that MFM
> performs well down to Mach numbers M ~ 0.01. Finally, we show how important the slope limiter and the energy-entropy
> switch are to control the behavior and the evolution of the fluids.

### `monaghan2013`

- **file:** `monaghan2013_multi-fluid-high-density-ratios.pdf`
- **title:** A simple SPH algorithm for multi-fluid flow with high density ratios
- **authors:** J. J. Monaghan and Ashkan Rafiee
- **venue:** *International Journal for Numerical Methods in Fluids* 71(5):537-561, 2013
- **doi:** [10.1002/fld.3671](https://doi.org/10.1002/fld.3671)
- **relevance:** The Lagrangian derivation of the SPH momentum/continuity pair from a constrained variational principle -- background for PESPH_PLAN.md's claim that PESPH-A's separability follows the same variational route as compSPH.
- **abstract from:** DOI record (Crossref)

> In this paper, we describe an SPH algorithm for multi‐fluid flow, which is efficient, simple and robust. We derive the inviscid equations of motion from a Lagrangian together with the constraint provided by the continuity equation. The viscous flow equations then follow by adding a viscous term. Rigid boundaries are simulated using boundary force particles in a manner similar to the immersed boundary method. Each fluid is approximated as weakly compressible with a speed of sound sufficiently large to guarantee that the relative density variations are typically 1%. When the SPH force interaction is between two particles of different fluids, we increase the pressure terms. This simple procedure stabilizes the interface between the fluids. The equations of motion are integrated using a time stepping rule based on a second‐order symplectic integrator. When linear and angular momentum should be conserved exactly, they are conserved to within round‐off errors. We test the algorithm by simulating a variety of problems involving fluids with a density ratio in the range 1–1000. The first of these is a free surface problem with no rigid boundaries. It involves the flow of an elliptical distribution with one fluid inside the other. We show that the simulations converge as the particle spacing decreases, and the results are in good agreement with the exact inviscid, incompressible theory. The second test is similar to the first but involves the nonlinear oscillation of the fluids. As in the first test, the agreement with theory is very good, and the method converges. The third test is the simulation of waves at the interface between two fluids. The method is shown to converge, and the agreement with theory is satisfactory. The fourth test is the Rayleigh–Taylor instability for a configuration considered by other authors. Key parameters are shown to converge, and the agreement with other authors is good. The fifth and final test is how well the SPH method simulates gravity currents with density ratios in the range 2–30. The results of these simulations are in very good agreement with those of other authors and in satisfactory agreement with experimental results.Copyright © 2012 John Wiley & Sons, Ltd.

### `price2018`

- **file:** `price2018_phantom.pdf`
- **title:** Phantom: A Smoothed Particle Hydrodynamics and Magnetohydrodynamics Code for Astrophysics
- **authors:** Daniel J. Price, James Wurster, Terrence S. Tricco, Chris Nixon, Stéven Toupin, Alex Pettitt, Conrad Chan, Daniel Mentiplay, Guillaume Laibe, Simon Glover, Clare Dobbs, Rebecca Nealon, David Liptai, Hauke Worpel, Clément Bonnerot, Giovanni Dipierro, Giulia Ballabio, Enrico Ragusa, Christoph Federrath, Roberto Iaconi, Thomas Reichardt, Duncan Forgan, Mark Hutchison, Thomas Constantino, Ben Ayliffe, Kieran Hirsh and Giuseppe Lodato
- **venue:** *Publications of the Astronomical Society of Australia* 35:e031, 2018
- **doi:** [10.1017/pasa.2018.25](https://doi.org/10.1017/pasa.2018.25)
- **relevance:** A production comparison baseline (PESPH_PLAN.md §7.6): standard modern SPH, distinct from the MFM/CRK/PESPH lineage.
- **abstract from:** DOI record (Crossref)

> We present Phantom , a fast, parallel, modular, and low-memory smoothed particle hydrodynamics and magnetohydrodynamics code developed over the last decade for astrophysical applications in three dimensions. The code has been developed with a focus on stellar, galactic, planetary, and high energy astrophysics, and has already been used widely for studies of accretion discs and turbulence, from the birth of planets to how black holes accrete. Here we describe and test the core algorithms as well as modules for magnetohydrodynamics, self-gravity, sink particles, dust–gas mixtures, H 2 chemistry, physical viscosity, external forces including numerous galactic potentials, Lense–Thirring precession, Poynting–Robertson drag, and stochastic turbulent driving. Phantom is hereby made publicly available.

### `borrow2022`

- **file:** `borrow2022_sphenix.pdf`
- **title:** SPHENIX: smoothed particle hydrodynamics for the next generation of galaxy formation simulations
- **authors:** Josh Borrow, Matthieu Schaller, Richard G. Bower and Joop Schaye
- **venue:** *Monthly Notices of the Royal Astronomical Society* 511(2):2367-2389, 2022
- **doi:** [10.1093/mnras/stab3166](https://doi.org/10.1093/mnras/stab3166)
- **relevance:** The SWIFT project's own modern-SPH scheme -- a second comparison baseline (PESPH_PLAN.md §7.6), and the sibling of sandnes2025/REMIX, both built on the same SWIFT code (schaller2024).
- **abstract from:** DOI record (Crossref)

> Smoothed particle hydrodynamics (SPH) is a ubiquitous numerical method for solving the fluid equations, and is prized for its conservation properties, natural adaptivity, and simplicity. We introduce the Sphenix SPH scheme, which was designed with three key goals in mind: to work well with sub-grid physics modules that inject energy, be highly computationally efficient (both in terms of compute and memory), and to be Lagrangian. sphenix uses a Density-Energy equation of motion, along with a variable artificial viscosity and conduction, including limiters designed to work with common sub-grid models of galaxy formation. In particular, we present and test a novel limiter that prevents conduction across shocks, preventing spurious radiative losses in feedback events. Sphenix is shown to solve many difficult test problems for traditional SPH, including fluid mixing and vorticity conservation, and it is shown to produce convergent behaviour in all tests where this is appropriate. Crucially, we use the same parameters within sphenix for the various switches throughout, to demonstrate the performance of the scheme as it would be used in production simulations. sphenix is the new default scheme in the swift cosmological simulation code and is available open source.

### `sandnes2025`

- **file:** `sandnes2025_remix-sph.pdf`
- **title:** REMIX SPH -- improving mixing in smoothed particle hydrodynamics simulations using a generalised, material-independent approach
- **authors:** T. D. Sandnes, V. R. Eke, J. A. Kegerreis, R. J. Massey, S. Ruiz-Bonilla, M. Schaller and L. F. A. Teodoro
- **venue:** *Journal of Computational Physics* 532:113907, 2025
- **doi:** [10.1016/j.jcp.2025.113907](https://doi.org/10.1016/j.jcp.2025.113907)
- **relevance:** A modern mixing-focused SPH scheme, cited in PESPH_PLAN.md §7.1's synthesis of where the field stands post-2015; now published (was arXiv-only when the plan's outlook was written).
- **abstract from:** PDF p.1

> We present REMIX, a smoothed particle hydrodynamics (SPH) scheme designed to alleviate effects that typically
> suppress mixing and instability growth at density discontinuities in SPH simulations. We approach this problem
> by directly targeting sources of kernel smoothing error and discretisation error, resulting in a generalised, material-
> independent formulation that improves the treatment both of discontinuities within a single material, for example in
> an ideal gas, and of interfaces between dissimilar materials. This approach also leads to improvements in capturing
> wider hydrodynamic behaviour unrelated to mixing. We demonstrate marked improvements in three-dimensional test
> scenarios, focusing on cases with particles of equal mass across the simulation. This choice is particularly relevant for
> use cases in astrophysics and engineering – specifically those in which particles are free to evolve over a large range
> of density scales – where bespoke choices of unequal particle masses in the initial conditions cannot easily be used
> to address emergent and evolving density discontinuities. We achieve these improvements while maintaining sharp
> discontinuities; without introducing additional equation of state dependence in, for example, particle volume elements;
> and without contrived or targeted corrections. Our methods build upon a fully compressible and thermodynamically
> consistent core-SPH construction, retaining Galilean invariance as well as conservation of mass, momentum, and
> energy. REMIX is integrated in the open-source, state-of-the-art Swift code and is designed with computational
> efficiency also in mind, meaning that its improved hydrodynamic treatment can be used for high-resolution simulations
> without prohibitive cost to run-speed.

### `schaller2024`

- **file:** `schaller2024_swift.pdf`
- **title:** SWIFT: a modern highly parallel gravity and smoothed particle hydrodynamics solver for astrophysical and cosmological applications
- **authors:** Matthieu Schaller, Josh Borrow, Peter W. Draper, Mladen Ivkovic, Stuart McAlpine, Bert Vandenbroucke, Yannick Bahé, Evgenii Chaikin, Aidan B. G. Chalk, Tsang Keung Chan, Camila Correa, Marcel van Daalen, Willem Elbers, Pedro Gonnet, Loïc Hausammann, John Helly, Filip Huško, Jacob A. Kegerreis, Folkert S. J. Nobels, Sylvia Ploeckinger, Yves Revaz, William J. Roper, Sergio Ruiz-Bonilla, Thomas D. Sandnes, Yolan Uyttenhove, James S. Willis and Zhen Xiang
- **venue:** *Monthly Notices of the Royal Astronomical Society* 530(2):2378-2419, 2024
- **doi:** [10.1093/mnras/stae922](https://doi.org/10.1093/mnras/stae922)
- **relevance:** The code both borrow2022/SPHENIX and sandnes2025/REMIX are built on and validated in; background for PESPH_PLAN.md §7.1's adoption survey. First author is Schaller, not "Swift" -- the incoming filename named it for the project, not the author.
- **abstract from:** DOI record (Crossref)

> Numerical simulations have become one of the key tools used by theorists in all the fields of astrophysics and cosmology. The development of modern tools that target the largest existing computing systems and exploit state-of-the-art numerical methods and algorithms is thus crucial. In this paper, we introduce the fully open-source highly-parallel, versatile, and modular coupled hydrodynamics, gravity, cosmology, and galaxy-formation code Swift. The software package exploits hybrid shared- and distributed-memory task-based parallelism, asynchronous communications, and domain-decomposition algorithms based on balancing the workload, rather than the data, to efficiently exploit modern high-performance computing cluster architectures. Gravity is solved for using a fast-multipole-method, optionally coupled to a particle mesh solver in Fourier space to handle periodic volumes. For gas evolution, multiple modern flavours of Smoothed Particle Hydrodynamics are implemented. Swift also evolves neutrinos using a state-of-the-art particle-based method. Two complementary networks of sub-grid models for galaxy formation as well as extensions to simulate planetary physics are also released as part of the code. An extensive set of output options, including snapshots, light-cones, power spectra, and a coupling to structure finders are also included. We describe the overall code architecture, summarize the consistency and accuracy tests that were performed, and demonstrate the excellent weak-scaling performance of the code using a representative cosmological hydrodynamical problem with ≈300 billion particles. The code is released to the community alongside extensive documentation for both users and developers, a large selection of example test problems, and a suite of tools to aid in the analysis of large simulations run with Swift.

### `crain2023`

- **file:** `crain2023_hydrodynamical-simulations-galaxy-population.pdf`
- **title:** Hydrodynamical Simulations of the Galaxy Population: Enduring Successes and Outstanding Challenges
- **authors:** Robert A. Crain and Freeke van de Voort
- **venue:** *Annual Review of Astronomy and Astrophysics* 61(1):473-515, 2023
- **doi:** [10.1146/annurev-astro-041923-043618](https://doi.org/10.1146/annurev-astro-041923-043618)
- **relevance:** The current-generation review of galaxy-formation hydrodynamics methods (SPH, MFM/MFV, moving mesh) -- the place PESPH_PLAN.md §7.1's framing was checked against.
- **abstract from:** DOI record (Crossref)

> We review the progress in modeling the galaxy population in hydrodynamical simulations of the ΛCDM cosmogony. State-of-the-art simulations now broadly reproduce the observed spatial clustering of galaxies; the distributions of key characteristics, such as mass, size, and SFR; and scaling relations connecting diverse properties to mass. Such improvements engender confidence in the insight drawn from simulations. Many important outcomes, however, particularly the properties of circumgalactic gas, are sensitive to the details of the subgrid models used to approximate the macroscopic effects of unresolved physics, such as feedback processes. We compare the outcomes of leading simulation suites with observations, and with each other, to identify the enduring successes they have cultivated and the outstanding challenges to be tackled with the next generation of models. Our key conclusions include the following:▪Realistic galaxies can be reproduced by calibrating the ill-constrained parameters of subgrid feedback models. Feedback is dominated by stars and black holes in low-mass and high-mass galaxies, respectively.▪Adjusting or disabling the processes implemented in simulations can elucidate their impact on observables, but outcomes can be degenerate.▪Similar galaxy populations can emerge in simulations with dissimilar feedback implementations. However, these models generally predict markedly different gas flow rates into, and out of, galaxies and their halos. CGM observations are thus a promising means of breaking this degeneracy and guiding the development of new feedback models.

### `rosswog2015review`

- **file:** `rosswog2015review_sph-methods-compact-objects.pdf`
- **title:** SPH Methods in the Modelling of Compact Objects
- **authors:** Stephan Rosswog
- **venue:** *Living Reviews in Computational Astrophysics* 1:1, 2015
- **doi:** [10.1007/lrca-2015-1](https://doi.org/10.1007/lrca-2015-1)
- **relevance:** A field review distinct from rosswog2015 (the MNRAS numerics paper) and from rosswog2026, its 2026 major-revision update -- this is the original 2015 Living Reviews article.
- **abstract from:** PDF p.1

> We review the current status of compact object simulations that are based on the smooth
> particle hydrodynamics (SPH) method. The first main part of this review is dedicated to SPH
> as a numerical method. We begin by discussing relevant kernel approximation techniques and
> discuss the performance of different kernel functions. Subsequently, we review a number of
> different SPH formulations of Newtonian, special- and general relativistic ideal fluid dynamics. We particularly point out recent developments that increase the accuracy of SPH with
> respect to commonly used techniques. The second main part of the review is dedicated to the
> application of SPH in compact object simulations. We discuss encounters between two white
> dwarfs, between two neutron stars and between a neutron star and a stellar-mass black hole.
> For each type of system, the main focus is on the more common, gravitational wave-driven
> binary mergers, but we also discuss dynamical collisions as they occur in dense stellar systems
> such as cores of globular clusters.

### `rosswog2026`

- **file:** `rosswog2026_sph-compact-objects-review.pdf`
- **title:** SPH Methods in the Modelling of Compact Objects
- **authors:** Stephan Rosswog
- **venue:** arXiv:2607.14828 [astro-ph.HE], 2026
- **arXiv:** [2607.14828](https://arxiv.org/abs/2607.14828)
- **relevance:** "Major revision, updated and expanded" of `rosswog2015review` (its own change-summary, page 2) -- same title and author, same underlying Living Reviews article, but arXiv-only at this point and not yet carrying the Living Reviews DOI as a fast-track revision. This is the copy PESPH_PLAN.md §7.1's framing should be checked against, since it is the more current version.
- **abstract from:** arXiv API

> We review the current status of compact object simulations that are based on the Smoothed Particle Hydrodynamics (SPH) method. The first section of this review is dedicated to SPH as a numerical method for Newtonian, ideal gas dynamics and it should be fairly self-contained. It begins with the basics of the method, but also describes recent advances including various meshless derivatives or methods for treating shocks. A separate chapter summarizes general relativistic SPH, including its special relativistic limit, and it explains in some detail the recent development of full numerical relativity in SPH where matter is evolved together with a dynamical spacetime. The remainder of the review has an astrophysical focus, here we discuss the status of the simulations of white dwarf--white dwarf, neutron star--neutron star and neutron star--black hole systems. For each type of system the emphasis is on gravitational-wave-driven mergers, but we also briefly summarize dynamical collisions that can occur in locations with large stellar densities.

## Particle consistency and gradient renormalization

The correction-matrix lineage behind this codebase's `Li`/`useGradientRenormalization` machinery (`modules/crk`, `wp_surfaceAware.py`), synced 2026-09-17.

### `randles1996`

- **file:** `randles1996_recent-improvements-applications.pdf`
- **title:** Smoothed Particle Hydrodynamics: Some recent improvements and applications
- **authors:** P.W. Randles and L.D. Libersky
- **venue:** *Computer Methods in Applied Mechanics and Engineering* 139:375-408, 1996
- **doi:** [10.1016/s0045-7825(96)01090-0](https://doi.org/10.1016/s0045-7825(96)01090-0)
- **relevance:** Independently derives (citing Johnson-Beissel and Everhart) the corrective tensor `B` this codebase's `Li` matrix descends from (Eqs. 34-37), applying it to the momentum/energy divergence and outer-product kernel sums from the boundary-deficiency side, rather than `bonet1999`'s variational one.
- **abstract from:** PDF p.1

> The Smoothed Particle Hydrodynamics (SPH) computing technique has features which make it highly attractive for simulating dynamic response of materials involving fracture and fragmentation. However, full exploitation of the method's potential has been hampered by some unresolved problems including stability and the lack of generalized boundary conditions. We address these difficulties and propose solutions. Continuum damage modeling of fracture is discussed at length with scalar and tensor formulations proposed and tested within SPH. Several recent applications involving fracture with predicted fragment patterns and mass distributions are compared with experiment.

### `bonet1999`

- **file:** `bonet1999_variational-momentum-preservation.pdf`
- **title:** Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations
- **authors:** J. Bonet and T.-S.L. Lok
- **venue:** *Computer Methods in Applied Mechanics and Engineering* 180:97-115, 1999
- **doi:** [10.1016/s0045-7825(99)00051-1](https://doi.org/10.1016/s0045-7825(99)00051-1)
- **relevance:** The other foundational correction-matrix paper (with `randles1996`): derives the same gradient-renormalization matrix from a variational/momentum-conservation argument, and shows it is required for exact angular-momentum preservation under a linear velocity field -- the derivation this codebase's `Li` matrix is closer to in spirit.
- **abstract from:** PDF p.1
- **text-layer:** `linear velocity ®eld` -> `linear velocity field` (the PDF's text layer substitutes the registered-trademark glyph for the "fi" ligature; anchored to this phrase since `®eld`/`®elds` recur elsewhere on the page)
- **text-layer:** `free surface ¯ows` -> `free surface flows` (the PDF's text layer substitutes the macron glyph for the "fl" ligature)

> This paper presents a new variational framework for various existing Smooth Particle Hydrodynamic (SPH) techniques and presents a new corrected SPH formulation. The linear and angular momentum preserving properties of SPH formulations are also discussed. The paper will show that in general in order to preserve angular momentum, the SPH equations must correctly evaluate the gradient of a linear velocity field. A corrected algorithm that combines kernel correction with gradient correction is presented. The paper will illustrate the theory presented with several examples relating to simple free surface flows.

### `liu2006`

- **file:** `liu2006_restoring-particle-consistency.pdf`
- **title:** Restoring particle consistency in smoothed particle hydrodynamics
- **authors:** M.B. Liu and G.R. Liu
- **venue:** *Applied Numerical Mathematics* 56(1):19-36, 2006
- **doi:** [10.1016/j.apnum.2005.02.012](https://doi.org/10.1016/j.apnum.2005.02.012)
- **relevance:** A later paper in the same correction-matrix lineage as `randles1996`/`bonet1999`: restores particle consistency while keeping the smoothing kernel itself unmodified, rather than reconstructing the kernel.
- **abstract from:** PDF p.1

> Though the smoothed particle hydrodynamics (SPH) method has been widely applied to different areas, it is associated with some inherent numerical problems. One notable problem is the particle inconsistency that results from the particle approximation process and can lead to low approximation accuracy. In this paper, the particle inconsistency problem is investigated and some methods to improve the particle inconsistency are discussed. A new approach is proposed to restore the particle consistency. The new approach retains the conventional non-negative smoothing function instead of reconstructing a new smoothing function. A series of numerical studies have been carried out to verify the performance of the new approach. It is found the new approach can successfully restore the particle consistency and can therefore significantly improve the approximation accuracy.

### `liu1995`

- **file:** `liu1995_reproducing-kernel-particle-methods.pdf`
- **title:** Reproducing kernel particle methods
- **authors:** Wing Kam Liu, Sukky Jun and Yi Fei Zhang
- **venue:** *International Journal for Numerical Methods in Fluids* 20(8-9):1081-1106, 1995
- **doi:** [10.1002/fld.1650200824](https://doi.org/10.1002/fld.1650200824)
- **relevance:** The earliest paper in this set: RKPM's correction function predates and is the finite-element-adjacent parallel to `randles1996`/`bonet1999`'s SPH gradient-correction matrices, reached independently from a wavelet/window-function consistency argument.
- **abstract from:** DOI record (Crossref)

> A new continuous reproducing kernel interpolation function which explores the attractive features of the flexible time-frequency and space-wave number localization of a window function is developed. This method is motivated by the theory of wavelets and also has the desirable attributes of the recently proposed smooth particle hydrodynamics (SPH) methods, moving least squares methods (MLSM), diffuse element methods (DEM) and element-free Galerkin methods (EFGM). The proposed method maintains the advantages of the free Lagrange or SPH methods; however, because of the addition of a correction function, it gives much more accurate results. Therefore it is called the reproducing kernel particle method (RKPM). In computer implementation RKPM is shown to be more efficient than DEM and EFGM. Moreover, if the window function is C∞, the solution and its derivatives are also C∞ in the entire domain. Theoretical analysis and numerical experiments on the 1D diffusion equation reveal the stability conditions and the effect of the dilation parameter on the unusually high convergence rates of the proposed method. Two-dimensional examples of advection-diffusion equations and compressible Euler equations are also presented together with 2D multiple-scale decompositions.

### `chen1999`

- **file:** `chen1999_corrective-smoothed-particle-method-heat.pdf`
- **title:** A corrective smoothed particle method for boundary value problems in heat conduction
- **authors:** J.K. Chen, J.E. Beraun and T.C. Carney
- **venue:** *International Journal for Numerical Methods in Engineering* 46(2):231-252, 1999
- **doi:** [10.1002/(sici)1097-0207(19990920)46:2<231::aid-nme672>3.0.co;2-k](https://doi.org/10.1002/(sici)1097-0207(19990920)46:2<231::aid-nme672>3.0.co;2-k)
- **relevance:** CSPM: a Taylor-series-expansion correction layered on top of the kernel estimate, rather than a renormalization matrix -- a distinct correction strategy from `randles1996`/`bonet1999`/`liu2006`, aimed at the same boundary-particle-deficiency problem those three also address.
- **abstract from:** PDF p.1
- **text-layer:** `de"ciency` -> `deficiency` (the PDF's text layer substitutes a straight double-quote for the "fi" ligature)

> Combining the kernel estimate with the Taylor series expansion is proposed to develop a Corrective Smoothed Particle Method (CSPM). This algorithm resolves the general problem of particle deficiency at boundaries, which is a shortcoming in Standard Smoothed Particle Hydrodynamics (SSPH). In addition, the method's ability to model derivatives of any order could make it applicable for any time-dependent boundary value problems. An example of the applications studied in this paper is unsteady heat conduction, which is governed by second-order derivatives. Numerical results demonstrate that besides the capability of directly imposing boundary conditions, the present method enhances the solution accuracy not only near or on the boundary but also inside the domain. Published in 1999 by John Wiley & Sons, Ltd. This article is a U.S. government work and is in the public domain in the United States.

## High-order and WENO/MLS-reconstruction SPH

The modern successor direction `letouze2025`/`lind2020`/`meng2025` point to past this codebase's plain, first-order-consistent SPH gradients, synced 2026-09-17.

### `avesani2014`

- **file:** `avesani2014_moving-least-squares-weno-sph.pdf`
- **title:** A new class of Moving-Least-Squares WENO-SPH schemes
- **authors:** Diego Avesani, Michael Dumbser and Alberto Bellin
- **venue:** *Journal of Computational Physics* 270:278-299, 2014
- **doi:** [10.1016/j.jcp.2014.03.041](https://doi.org/10.1016/j.jcp.2014.03.041)
- **relevance:** Origin of the MLS-WENO-SPH line: per-particle MLS reconstructions blended by a nonlinear WENO weighting, fed into a Riemann-solved flux at the particle midpoint -- the modern high-order successor direction `letouze2025`/`lind2020`/`meng2025` point to, and the origin `vergnaud2023`/`gao2023` cite for the MLS-WENO-SPH line.
- **abstract from:** PDF p.1
- **text-layer:** `Oshertype` -> `Osher-type` (the PDF's text layer drops the hyphen across the line wrap)

> We present a new class of meshless Lagrangian particle methods based on the SPH formulation of Vila and Ben Moussa, combined with a new weighted essentially non-oscillatory (WENO) reconstruction technique on moving point clouds in multiple space dimensions. The key idea is to produce for each particle first a set of high order accurate Moving-Least-Squares (MLS) reconstructions on a set of different reconstruction stencils. Then, these reconstructions are combined with each other using a non-linear WENO technique in order to capture at the same time discontinuities and to maintain accuracy and low numerical dissipation in smooth regions. The numerical fluxes between interacting particles are subsequently evaluated using this MLS-WENO reconstruction at the midpoint between two particles, in combination with a Riemann solver that provides the necessary stabilization of the scheme based on the underlying physics of the governing equations. We propose the use of two different Riemann solvers: the Rusanov flux and an Osher-type flux. The use of monotone fluxes together with a WENO reconstruction ensures accuracy, stability, robustness and an essentially non-oscillatory solution without the artificial viscosity term usually employed in conventional SPH schemes. To our knowledge, this is the first time that the WENO method, which has originally been developed for mesh-based schemes in the Eulerian framework on fixed grids, is extended to meshfree Lagrangian particle methods like SPH in multiple space dimensions. We test the new algorithm on two dimensional blast wave problems and on the classical one-dimensional Sod shock tube problem for the Euler equations of compressible gas dynamics. We obtain a good agreement with the exact or numerical reference solution in all cases and an improved accuracy and robustness compared to existing standard SPH schemes.

### `king2020`

- **file:** `king2020_labfm-high-order-difference-schemes.pdf`
- **title:** High order difference schemes using the local anisotropic basis function method
- **authors:** J.R.C. King, S.J. Lind and A.M.A. Nasar
- **venue:** *Journal of Computational Physics* 415:109549, 2020
- **doi:** [10.1016/j.jcp.2020.109549](https://doi.org/10.1016/j.jcp.2020.109549)
- **relevance:** LABFM: constructs high-order (4th-8th) difference operators from anisotropic basis functions on disordered nodes; the paper's own framing is that SPH is LABFM's low-order limit, making this the explicit high-order generalization of the kernel-gradient machinery this codebase uses.
- **abstract from:** PDF p.1

> Mesh-free methods have significant potential for simulations in complex geometries, as the time consuming process of mesh-generation is avoided. Smoothed Particle Hydrodynamics (SPH) is the most widely used mesh-free method, but suffers from a lack of consistency. High order, consistent, and local (using compact computational stencils) mesh-free methods are particularly desirable. Here we present a novel framework for generating local high order difference operators for arbitrary node distributions, referred to as the Local Anisotropic Basis Function Method (LABFM). Weights are constructed from linear sums of anisotropic basis functions (ABFs), chosen to ensure exact reproduction of polynomial fields up to a given order. The ABFs are based on a fundamental Radial Basis Function (RBF), and the choice of fundamental RBF has small effect on accuracy, but influences stability. LABFM is able to generate high order difference operators with compact computational stencils (4th order with N ≈ 25 nodes, 8th order with N ≈ 60 nodes in two dimensions). At domain boundaries (with incomplete support) LABFM automatically provides one-sided differences of the same order as the internal scheme, up to 4th order. We use the method to solve elliptic, parabolic and mixed hyperbolic-parabolic partial differential equations (PDEs), showing up to 8th order convergence. The inclusion of hyperviscosity is straightforward, and can effectively provide stability when solving hyperbolic problems. LABFM is a promising new mesh-free method for the numerical solution of PDEs in complex geometries. The method is highly scalable, and for Eulerian schemes, the computational efficiency is competitive with RBF-FD for a given accuracy. A particularly attractive feature is that in the low order limit, LABFM collapses to Smoothed Particle Hydrodynamics (SPH), and there is potential for Arbitrary Lagrangian-Eulerian schemes with natural adaptivity of resolution and accuracy.

### `king2022`

- **file:** `king2022_labfm-isothermal-flows.pdf`
- **title:** High-order simulations of isothermal flows using the local anisotropic basis function method (LABFM)
- **authors:** J.R.C. King and S.J. Lind
- **venue:** *Journal of Computational Physics* 449:110760, 2022
- **doi:** [10.1016/j.jcp.2021.110760](https://doi.org/10.1016/j.jcp.2021.110760)
- **note:** The DOI's own path segment carries "2021", but the DOI record's `.issued` date and the journal's volume/issue dating (449, 2022) both place formal publication in 2022 -- "Available online 6 October 2021" is the preprint date. Year and volume here follow the DOI record per `ADDING.md`'s "DOI record wins" rule.
- **relevance:** Journal extension of `king2020` to full isothermal Navier-Stokes at up to 10th order, adding stabilisation and high-order boundary conditions.
- **abstract from:** PDF p.1
- **text-layer:** `Re p` -> `Re_p` (the subscript on the Reynolds number renders as a space in the text layer)

> Mesh-free methods have significant potential for simulations of flows in complex geometries, with the difficulties of domain discretisation greatly reduced. However, many mesh-free methods are limited to low order accuracy. In order to compete with conventional mesh-based methods, high order accuracy is essential. The Local Anisotropic Basis Function Method (LABFM) is a mesh-free method introduced in King et al. (2020) [20], which enables the construction of highly accurate difference operators on disordered node discretisations. Here, we introduce a number of developments to LABFM, in the areas of basis function construction, stencil optimisation, stabilisation, variable resolution, and high order boundary conditions. With these developments, direct numerical simulations of the Navier Stokes equations are possible at extremely high order (up to 10th order in characteristic node spacing internally). We numerically solve the isothermal compressible Navier Stokes equations for a range of geometries: periodic and channel flows, flows past a cylinder, and porous media. Excellent agreement is seen with analytical solutions, published numerical results (using a spectral element method), and experiments. The potential of the method for direct numerical simulations in complex geometries is demonstrated with simulations of subsonic and transonic flows through an inhomogeneous porous media at pore Reynolds numbers up to Re p = 968.

### `vergnaud2023`

- **file:** `vergnaud2023_high-order-sph-weno-reconstruction.pdf`
- **title:** Investigations on a high order SPH scheme using WENO reconstruction
- **authors:** A. Vergnaud, G. Oger and D. Le Touzé
- **venue:** *Journal of Computational Physics* 477:111889, 2023
- **doi:** [10.1016/j.jcp.2022.111889](https://doi.org/10.1016/j.jcp.2022.111889)
- **relevance:** 1D WENO reconstruction per interacting pair, completed by MLS, inside a Riemann-SPH formulation -- reaches 6th-order convergence and is shown to beat plain Riemann-SPH on accuracy per CPU time. Direct evidence for the field's practical route past the roughly 2nd-order ceiling this codebase's plain SPH gradients sit at.
- **abstract from:** PDF p.1

> Theoretically, a 2nd order convergence can be reached with the Smoothed Particle Hydrodynamics (SPH) method. However, depending on the spatial disorder of particles, the order of convergence observed in practice can be lower than one. In this paper, a methodology is proposed for the reconstruction of high order numerical fluxes in Riemann-SPH formulations for weakly-compressible flows, so as to increase the global order of convergence of the scheme. This methodology is based on the use of one-dimensional Weighted Essentially Non-Oscillatory (WENO) reconstructions applied at each pair of interacting particles, and each 1D WENO stencil is completed using Moving-Least-Squares (MLS) reconstructions. It is shown that a 6th order convergence can be reached with the proposed SPH-WENO scheme. The gain in accuracy and convergence properties of this scheme is shown and discussed through its application to various one-dimensional and two-dimensional test cases. In particular, the influence of the number of neighbor particles and of the spatial particle disorder is studied. Finally, it is shown that the proposed high-order SPH scheme provides a better accuracy to CPU time ratio than usual Riemann-SPH schemes. The treatment of boundary conditions on rigid walls with the proposed scheme is also discussed.

### `gao2023`

- **file:** `gao2023_mls-teno-sph-compressible-flows.pdf`
- **title:** A new smoothed particle hydrodynamics method based on high-order moving-least-square targeted essentially non-oscillatory scheme for compressible flows
- **authors:** Tianrun Gao, Tian Liang and Lin Fu
- **venue:** *Journal of Computational Physics* 489:112270, 2023
- **doi:** [10.1016/j.jcp.2023.112270](https://doi.org/10.1016/j.jcp.2023.112270)
- **relevance:** MLS-TENO-SPH: splits the domain into smooth/non-smooth regions via a TENO scale-separation test and only spends the expensive MLS high-order derivative where the flow is smooth, falling back to shock-capturing TENO elsewhere -- the compressible-flow counterpart to `vergnaud2023`'s weakly-compressible WENO-SPH.
- **abstract from:** PDF p.1

> In this study, we establish a hybrid high-order smoothed particle hydrodynamics (SPH) framework (MLS-TENO-SPH) for compressible flows with discontinuities, which is able to achieve genuine high-order convergence in smooth regions and also capture discontinuities well in non-smooth regions. The framework can be either fully Lagrangian, Eulerian or realizing arbitary-Lagrangian-Eulerian (ALE) feature enforcing the isotropic particle distribution in specific cases. In the proposed framework, the computational domain is divided into smooth regions and non-smooth regions, and these two regions are determined by a strong scale separation strategy in the targeted essentially non-oscillatory (TENO) scheme. In smooth regions, the moving-least-square (MLS) approximation is used for evaluating high-order derivative operator, which is able to realize genuine high-order construction; in non-smooth regions, the new TENO scheme based on Vila's framework with several new improvements will be deployed to capture discontinuities and high-wavenumber flow scales with low numerical dissipation. The present MLS-TENO-SPH method is validated with a set of challenging cases based on the Eulerian, Lagrangian or ALE framework. Numerical results demonstrate that the MLS-TENO-SPH method features lower numerical dissipation and higher efficiency than the conventional method, and can restore genuine high-order accuracy in smooth regions. Overall, the proposed framework serves as a new exploration in high-order SPH methods, which are potential for compressible flow simulations with shockwaves.

## SPH reviews and the DualSPHysics lineage

Broad-survey and DualSPHysics-project reference papers, synced 2026-09-17.

### `lind2020`

- **file:** `lind2020_review-converged-lagrangian-flow-modelling.pdf`
- **title:** Review of smoothed particle hydrodynamics: towards converged Lagrangian flow modelling
- **authors:** Steven J. Lind, Benedict D. Rogers and Peter K. Stansby
- **venue:** *Proceedings of the Royal Society A* 476(2241), 2020
- **doi:** [10.1098/rspa.2019.0801](https://doi.org/10.1098/rspa.2019.0801)
- **relevance:** General SPH review centred on convergence; read against `BOUNDARY_DENSITY_PLAN.md` §10's kernel-truncation-artifact investigation, since a symmetric-pressure-pair-sum truncation error is exactly a convergence failure mode of the kind this review surveys.
- **abstract from:** DOI record (Crossref)

> This paper presents a review of the progress of smoothed particle hydrodynamics (SPH) towards high-order converged simulations. As a mesh-free Lagrangian method suitable for complex flows with interfaces and multiple phases, SPH has developed considerably in the past decade. While original applications were in astrophysics, early engineering applications showed the versatility and robustness of the method without emphasis on accuracy and convergence. The early method was of weakly compressible form resulting in noisy pressures due to spurious pressure waves. This was effectively removed in the incompressible (divergence-free) form which followed; since then the weakly compressible form has been advanced, reducing pressure noise. Now numerical convergence studies are standard. While the method is computationally demanding on conventional processors, it is well suited to parallel processing on massively parallel computing and graphics processing units. Applications are diverse and encompass wave-structure interaction, geophysical flows due to landslides, nuclear sludge flows, welding, gearbox flows and many others. In the state of the art, convergence is typically between the first- and second-order theoretical limits. Recent advances are improving convergence to fourth order (and higher) and these will also be outlined. This can be necessary to resolve multi-scale aspects of turbulent flow.

### `meng2025`

- **file:** `meng2025_high-order-sph-review.pdf`
- **title:** High-order SPH: A Review of the Method and Applications
- **authors:** Zi-Fei Meng, Peng-Nan Sun, Ping-Ping Wang, Boo Cheong Khoo and A.-Man Zhang
- **venue:** *Archives of Computational Methods in Engineering* 33:1409-1444, 2026
- **doi:** [10.1007/s11831-025-10346-0](https://doi.org/10.1007/s11831-025-10346-0)
- **note:** The journal's own printed header dates the article "(2026) 33:1409-1444"; Crossref's `.issued` date is 2025-09-12 (first-online). Following this repository's `winchenbach2025analytic`/`winchenbach2025diffsph` precedent, the bib key keeps the earlier online year while the venue and year field here use the printed-volume year.
- **relevance:** Already read for `BOUNDARY_DENSITY_PLAN.md` §10 -- see that section rather than re-deriving its relevance here.
- **abstract from:** PDF p.1

> Smoothed Particle Hydrodynamics (SPH), a widely-used numerical method for simulating fluid flows with complex interfaces and boundaries, has undergone decades of development. This paper reviews the efforts towards advancing high-order SPH and its applications. The fundamentals of SPH are briefly introduced, followed by an analysis of errors arising in kernel and particle approximations. The relationship between consistency and accuracy is also discussed. To achieve high-order accuracy, this paper details three approaches: correcting SPH derivative operators, constructing kernel functions and implementing spatial reconstructions in the Riemann-SPH formulation. Finally, the applications of these high-accuracy SPH methods are described to show their potential for further development.

### `letouze2025`

- **file:** `letouze2025_free-surface-multiphase-review.pdf`
- **title:** Smoothed particle hydrodynamics for free-surface and multiphase flows: a review
- **authors:** David Le Touzé and Andrea Colagrossi
- **venue:** *Reports on Progress in Physics* 88(3):037001, 2025
- **doi:** [10.1088/1361-6633/ada80f](https://doi.org/10.1088/1361-6633/ada80f)
- **relevance:** Already read for `BOUNDARY_DENSITY_PLAN.md` §10 -- see that section rather than re-deriving its relevance here.
- **abstract from:** DOI record (Crossref)

> The smoothed particle hydrodynamics (SPH) method is expanding and is being applied to more and more fields, particularly in engineering. The majority of current SPH developments deal with free-surface and multiphase flows, especially for situations where geometrically complex interface configurations are involved. The present review article covers the last 25 years of development of the method to simulate such flows, discussing the related specific features of the method. A path is drawn to link the milestone articles on the topic, and the main related theoretical and numerical issues are investigated. In particular, several SPH schemes have been derived over the years, based on different assumptions. The main ones are presented and discussed in this review underlining the different contexts and the ways in which they were derived, resulting in similarities and differences. In addition, a summary is provided of the recent corrections proposed to increase the accuracy, stability and robustness of SPH schemes in the context of free-surface and multiphase flows. Future perspectives of development are identified, placing the method within the panorama of Computational Fluid Dynamics.

### `vacondio2021`

- **file:** `vacondio2021_grand-challenges-sph.pdf`
- **title:** Grand challenges for Smoothed Particle Hydrodynamics numerical schemes
- **authors:** Renato Vacondio, Corrado Altomare, Matthieu De Leffe, Xiangyu Hu, David Le Touzé, Steven Lind, Jean-Christophe Marongiu, Salvatore Marrone, Benedict D. Rogers and Antonio Souto-Iglesias
- **venue:** *Computational Particle Mechanics* 8(3):575-588, 2021
- **doi:** [10.1007/s40571-020-00354-1](https://doi.org/10.1007/s40571-020-00354-1)
- **relevance:** The SPHERIC Grand Challenges statement -- the field-level framing this codebase's own DualSPHysics cross-checks (`molteni2009`, `fourtakas2019`, `english2022`, `english2025` elsewhere in this manifest) sit inside.
- **abstract from:** PDF p.1

> This paper presents a brief review of grand challenges of Smoothed Particle Hydrodynamics (SPH) method. As a meshless method, SPH can simulate a large range of applications from astrophysics to free-surface flows, to complex mixing problems in industry and has had notable successes. As a young computational method, the SPH method still requires development to address important elements which prevent more widespread use. This effort has been led by members of the SPH rEsearch and engineeRing International Community (SPHERIC) who have identified SPH Grand Challenges. The SPHERIC SPH Grand Challenges (GCs) have been grouped into 5 categories: (GC1) convergence, consistency and stability, (GC2) boundary conditions, (GC3) adaptivity, (GC4) coupling to other models, and (GC5) applicability to industry. The SPH Grand Challenges have been formulated to focus the attention and activities of researchers, developers, and users around the world. The status of each SPH Grand Challenge is presented in this paper with a discussion on the areas for future development.

### `dominguez2022`

- **file:** `dominguez2022_dualsphysics-multiphysics.pdf`
- **title:** DualSPHysics: from fluid dynamics to multiphysics problems
- **authors:** J.M. Domínguez, G. Fourtakas, C. Altomare, R.B. Canelas, A. Tafuni, O. García-Feal, I. Martínez-Estévez, A. Mokos, R. Vacondio, A.J.C. Crespo, B.D. Rogers, P.K. Stansby and M. Gómez-Gesteira
- **venue:** *Computational Particle Mechanics* 9(5):867-895, 2022
- **doi:** [10.1007/s40571-021-00404-2](https://doi.org/10.1007/s40571-021-00404-2)
- **relevance:** The DualSPHysics reference/survey paper -- the code whose DDT and mDBC formulations this codebase's own `molteni2009`/`fourtakas2019`/`english2022`/`english2025` entries and its cross-engine harness (`DELTASPH_VALIDATION_PLAN.md` Part 8) are checked against.
- **abstract from:** PDF p.1

> DualSPHysics is a weakly compressible smoothed particle hydrodynamics (SPH) Navier-Stokes solver initially conceived to deal with coastal engineering problems, especially those related to wave impact with coastal structures. Since the first release back in 2011, DualSPHysics has shown to be robust and accurate for simulating extreme wave events along with a continuous improvement in efficiency thanks to the exploitation of hardware such as graphics processing units for scientific computing or the coupling with wave propagating models such as SWASH and OceanWave3D. Numerous additional functionalities have also been included in the DualSPHysics package over the last few years which allow the simulation of fluid-driven objects. The use of the discrete element method has allowed the solver to simulate the interaction among different bodies (sliding rocks, for example), which provides a unique tool to analyse debris flows. In addition, the recent coupling with other solvers like Project Chrono or MoorDyn has been a milestone in the development of the solver. Project Chrono allows the simulation of articulated structures with joints, hinges, sliders and springs and MoorDyn allows simulating moored structures. Both functionalities make DualSPHysics especially suited for the simulation of offshore energy harvesting devices. Lately, the present state of maturity of the solver goes beyond single-phase simulations, allowing multi-phase simulations with gas-liquid and a combination of Newtonian and non-Newtonian models expanding further the capabilities and range of applications for the DualSPHysics solver. These advances and functionalities make DualSPHysics an advanced meshless solver with emphasis on free-surface flow modelling.

