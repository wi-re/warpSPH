# Abstracts -- extended set

Companion to [`ABSTRACTS.md`](ABSTRACTS.md), for the 77 background papers added
to `references.bib` on 2026-08-29 (see the "Extended set" section of
[`MANIFEST.md`](MANIFEST.md)). Unlike `ABSTRACTS.md`, whose quotes are
transcribed from the page, the abstracts here come from one of three lower-effort
sources, named on each block's `abstract from:` line:

- **Crossref** -- the publisher's own deposited abstract text (effectively verbatim);
- **OpenAlex `abstract_inverted_index`** -- a reconstruction from per-word
  positions. Word order is right; original punctuation, capitalisation and line
  breaks are lost, so these are *not* a character-exact quote;
- **PDF text layer** -- for the papers no API carries (mostly Elsevier: JCP,
  CMAME, CPC), the abstract is sliced straight from the document's text layer.
  Ligatures are expanded (fi, fl, ...); a few soft hyphens the extractor swallowed
  survive as joins ("NavierStokes").

`scripts/check_literature.py` re-checks every block against its PDF -- the PDF
ones word-for-word, the API ones by word overlap -- so a block that has drifted
onto the wrong paper fails the build.

All 77 of the extended set have an abstract here.

To find one by subject: `grep -i -B10 'transport velocity' literature/ABSTRACTS_EXTENDED.md`

---

### `adams2007`

- **file:** `adams2007_adaptively-sampled-particle-fluids.pdf`
- **title:** Adaptively Sampled Particle Fluids
- **authors:** Bart Adams, Mark Pauly, Richard Keiser and Leonidas J. Guibas
- **venue:** ACM Transactions on Graphics 26(3), 2007
- **doi:** [10.1145/1239451.1239499](https://doi.org/10.1145/1239451.1239499)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> We present novel adaptive sampling algorithms for particle-based fluid simulation. We
> introduce a sampling condition based on geometric local feature size that allows focusing
> computational resources in geometrically complex regions, while reducing the number of
> particles deep inside the fluid or near thick flat surfaces. Further performance gains are
> achieved by varying the sampling density according to visual importance. In addition, we
> propose a novel fluid surface definition based on approximate particle-to-surface distances
> that are carried along with the particles and updated appropriately. The resulting surface
> reconstruction method has several advantages over existing methods, including stability under
> particle resampling and suitability for representing smooth flat surfaces. We demonstrate how
> our adaptive sampling and distance-based surface reconstruction algorithms lead to significant
> improvements in time and memory as compared to single resolution particle simulations, without
> significantly affecting the fluid flow behavior.

### `akinci2013`

- **file:** `akinci2013_versatile-surface-tension-and-adhesion.pdf`
- **title:** Versatile Surface Tension and Adhesion for SPH Fluids
- **authors:** Nadir Akinci, Gizem Akinci and Matthias Teschner
- **venue:** ACM Transactions on Graphics 32(6), 2013
- **doi:** [10.1145/2508363.2508395](https://doi.org/10.1145/2508363.2508395)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> Realistic handling of fluid-air and fluid-solid interfaces in SPH is a challenging problem.
> The main reason is that some important physical phenomena such as surface tension and adhesion
> emerge as a result of inter-molecular forces in a microscopic scale. This is different from
> scalar fields such as fluid pressure, which can be plausibly evaluated on a macroscopic scale
> using particles. Although there exist techniques to address this problem for some specific
> simulation scenarios, there does not yet exist a general approach to reproduce the variety of
> effects that emerge in reality from fluid-air and fluid-solid interactions. In order to
> address this problem, we present a new surface tension force and a new adhesion force.
> Different from the existing work, our surface tension force can handle large surface tensions
> in a realistic way. This property lets our approach handle challenging real scenarios, such as
> water crown formation, various types of fluid-solid interactions, and even droplet
> simulations. Furthermore, it prevents particle clustering at the free surface where inter-
> particle pressure forces are incorrect. Our adhesion force allows plausible two-way attraction
> of fluids and solids and can be used to model different wetting conditions. By using our
> forces, modeling surface tension and adhesion effects do not require involved techniques such
> as generating a ghost air phase or surface tracking. The forces are applied to the neighboring
> fluid-fluid and fluid-boundary particle pairs in a symmetric way, which satisfies momentum
> conservation. We demonstrate that combining both forces allows simulating a variety of
> interesting effects in a plausible way.

### `akinci2013coupling`

- **file:** `akinci2013coupling_coupling-elastic-solids-with-sph-fluids.pdf`
- **title:** Coupling Elastic Solids with Smoothed Particle Hydrodynamics Fluids
- **authors:** Nadir Akinci, Jens Cornelis, Gizem Akinci and Matthias Teschner
- **venue:** Computer Animation and Virtual Worlds 24(3-4), 2013
- **doi:** [10.1002/cav.1499](https://doi.org/10.1002/cav.1499)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We propose a method for handling elastic solids in smoothed particle hydrodynamics fluids. Our
> approach samples triangulated surfaces of solids using boundary particles. To prevent fluid
> particle tunneling in case of large expansions, additional boundary particles are adaptively
> generated to prevent gaps and undesired leakage. Furthermore, as an object compresses,
> particles are adaptively removed to avoid unnecessary computations. We demonstrate that our
> approach produces plausible interactions of smoothed particle hydrodynamics fluids with both
> slowly and rapidly deforming solids. Copyright © 2013 John Wiley & Sons, Ltd.

### `ando2013`

- **file:** `ando2013_highly-adaptive-liquid-simulations-tet-meshes.pdf`
- **title:** Highly Adaptive Liquid Simulations on Tetrahedral Meshes
- **authors:** Ryoichi Ando, Nils Thürey and Chris Wojtan
- **venue:** ACM Transactions on Graphics 32(4), 2013
- **doi:** [10.1145/2461912.2461982](https://doi.org/10.1145/2461912.2461982)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> We introduce a new method for efficiently simulating liquid with extreme amounts of spatial
> adaptivity. Our method combines several key components to drastically speed up the simulation
> of large-scale fluid phenomena: We leverage an alternative Eulerian tetrahedral mesh
> discretization to significantly reduce the complexity of the pressure solve while increasing
> the robustness with respect to element quality and removing the possibility of locking. Next,
> we enable subtle free-surface phenomena by deriving novel second-order boundary conditions
> consistent with our discretization. We couple this discretization with a spatially adaptive
> Fluid-Implicit Particle (FLIP) method, enabling efficient, robust, minimally-dissipative
> simulations that can undergo sharp changes in spatial resolution while minimizing artifacts.
> Along the way, we provide a new method for generating a smooth and detailed surface from a set
> of particles with variable sizes. Finally, we explore several new sizing functions for
> determining spatially adaptive simulation resolutions, and we show how to couple them to our
> simulator. We combine each of these elements to produce a simulation algorithm that is capable
> of creating animations at high maximum resolutions while avoiding common pitfalls like
> inaccurate boundary conditions and inefficient computation.

### `band2017`

- **file:** `band2017_moving-least-squares-boundaries.pdf`
- **title:** Moving Least Squares Boundaries for SPH Fluids
- **authors:** Stefan Band, Christoph Gissler and Matthias Teschner
- **venue:** Workshop on Virtual Reality Interaction and Physical Simulation (VRIPHYS), 2017
- **doi:** [10.2312/vriphys.20171080](https://doi.org/10.2312/vriphys.20171080)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> The paper shows that the SPH boundary handling of Akinci et al. [AIA 12] suffers from
> perceivable issues in planar regions due to deviations in the computed boundary normals and
> due to erroneous oscillations in the distance computation of fluid particles to the boundary.
> In order to resolve these issues, we propose a novel boundary handling that combines the SPH
> concept with Moving Least Squares. The proposed technique significantly improves the distance
> and normal computations in planar boundary regions, while its computational complexity is
> similar to Akinci's approach. We embed the proposed boundary handling into Implicit
> Incompressible SPH in a hybrid setting where it is applied at planar boundaries, while
> Akinci's technique is still being used for boundaries with complex shapes. Various benefits of
> the improved boundary handling are illustrated, in particular a reduced particle leakage and a
> reduced artificial boundary friction.

### `batty2007`

- **file:** `batty2007_fast-variational-framework-solid-fluid-coupling.pdf`
- **title:** A Fast Variational Framework for Accurate Solid-Fluid Coupling
- **authors:** Christopher Batty, Florence Bertails and Robert Bridson
- **venue:** ACM Transactions on Graphics 26(3), 2007
- **doi:** [10.1145/1239451.1239551](https://doi.org/10.1145/1239451.1239551)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Physical simulation has emerged as a compelling animation technique, yet current approaches to
> coupling simulations of fluids and solids with irregular boundary geometry are inefficient or
> cannot handle some relevant scenarios robustly. We propose a new variational approach which
> allows robust and accurate solution on relatively coarse Cartesian grids, allowing possibly
> orders of magnitude faster simulation. By rephrasing the classical pressure projection step as
> a kinetic energy minimization, broadly similar to modern approaches to rigid body contact, we
> permit a robust coupling between fluid and arbitrary solid simulations that always gives a
> well-posed symmetric positive semi-definite linear system. We provide several examples of
> efficient fluid-solid interaction and rigid body coupling with sub-grid cell flow. In
> addition, we extend the framework with a new boundary condition for free-surface flow,
> allowing fluid to separate naturally from solids.

### `batty2012`

- **file:** `batty2012_discrete-viscous-sheets.pdf`
- **title:** Discrete Viscous Sheets
- **authors:** Christopher Batty, Andres Uribe, Basile Audoly and Eitan Grinspun
- **venue:** ACM Transactions on Graphics 31(4), 2012
- **doi:** [10.1145/2185520.2185609](https://doi.org/10.1145/2185520.2185609)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> We present the first reduced-dimensional technique to simulate the dynamics of thin sheets of
> viscous incompressible liquid in three dimensions. Beginning from a discrete Lagrangian model
> for elastic thin shells, we apply the Stokes-Rayleigh analogy to derive a simple yet
> consistent model for viscous forces. We incorporate nonlinear surface tension forces with a
> formulation based on minimizing discrete surface area, and preserve the quality of triangular
> mesh elements through local remeshing operations. Simultaneously, we track and evolve the
> thickness of each triangle to exactly conserve liquid volume. This approach enables the
> simulation of extremely thin sheets of viscous liquids, which are difficult to animate with
> existing volumetric approaches. We demonstrate our method with examples of several
> characteristic viscous sheet behaviors, including stretching, buckling, sagging, and
> wrinkling.

### `becker2007`

- **file:** `becker2007_weakly-compressible-sph-free-surface.pdf`
- **title:** Weakly Compressible SPH for Free Surface Flows
- **authors:** Markus Becker and Matthias Teschner
- **venue:** Proceedings of the 2007 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2007
- **doi:** [10.2312/SCA/SCA07/209-218](https://doi.org/10.2312/SCA/SCA07/209-218)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a weakly compressible form of the Smoothed Particle Hydrodynamics method (SPH) for
> fluid flow based on the Tait equation. In contrast to commonly employed projection approaches
> that strictly enforce incompress- ibility, time-consuming solvers for the Poisson equation are
> avoided by allowing for small, user-defined density fluctuations. We also discuss an improved
> surface tension model that is particularly appropriate for single-phase free-surface flows.
> The proposed model is compared to existing models and experiments illustrate the accuracy of
> the approach for free surface flows. Combining the proposed methods, volume-preserving low-
> viscosity liquids can be efficiently simulated using SPH. The approach is appropriate for
> medium-scale and small-scale phenomena. Effects such as splashing and breaking waves are
> naturally handled.

### `becker2009`

- **file:** `becker2009_direct-forcing-lagrangian-rigid-fluid-coupling.pdf`
- **title:** Direct Forcing for Lagrangian Rigid-Fluid Coupling
- **authors:** Markus Becker, Hendrik Tessendorf and Matthias Teschner
- **venue:** IEEE Transactions on Visualization and Computer Graphics 15(3), 2009
- **doi:** [10.1109/TVCG.2008.107](https://doi.org/10.1109/TVCG.2008.107)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We propose a novel boundary handling algorithm for particle-based fluids. Based on a
> predictor-corrector scheme for both velocity and position, one- and two-way coupling with
> rigid bodies can be realized. The proposed algorithm offers significant improvements over
> existing penalty-based approaches. Different slip conditions can be realized and non-
> penetration is enforced. Direct forcing is employed to meet the desired boundary conditions
> and to ensure valid states after each simulation step. We have performed various experiments
> in 2D and 3D. They illustrate one- and two-way coupling of rigid bodies and fluids, the
> effects of hydrostatic and dynamic forces on a rigid body as well as different slip
> conditions. Numerical experiments and performance measurements are provided.

### `bell2005`

- **file:** `bell2005_particle-based-simulation-granular-materials.pdf`
- **title:** Particle-Based Simulation of Granular Materials
- **authors:** Nathan Bell, Yizhou Yu and Peter J. Mucha
- **venue:** Proceedings of the 2005 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2005
- **doi:** [10.1145/1073368.1073379](https://doi.org/10.1145/1073368.1073379)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Granular materials, such as sand and grains, are ubiquitous. Simulating the 3D dynamic motion
> of such materials represents a challenging problem in graphics because of their unique
> physical properties. In this paper we present a simple and effective method for granular
> material simulation. By incorporating techniques from physical models, our approach describes
> granular phenomena more faithfully than previous methods. Granular material is represented by
> a large collection of non-spherical particles which may be in persistent contact. The
> particles represent discrete elements of the simulated material. One major advantage of using
> discrete elements is that the topology of particle interaction can evolve freely. As a result,
> highly dynamic phenomena, such as splashing and avalanches, can be conveniently generated by
> this meshless approach without sacrificing physical accuracy. We generalize this discrete
> model to rigid bodies by distributing particles over their surfaces. In this way, two-way
> coupling between granular materials and rigid bodies is achieved.

### `bodin2011`

- **file:** `bodin2011_constraint-fluids.pdf`
- **title:** Constraint Fluids
- **authors:** Kenneth Bodin, Claude Lacoursière and Martin Servin
- **venue:** IEEE Transactions on Visualization and Computer Graphics 18(3), 2012
- **doi:** [10.1109/TVCG.2011.29](https://doi.org/10.1109/TVCG.2011.29)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a fluid simulation method based on Smoothed Particle Hydrodynamics (SPH) in which
> incompressibility and boundary conditions are enforced using holonomic kinematic constraints
> on the density. This formulation enables systematic multiphysics integration in which
> interactions are modeled via similar constraints between the fluid pseudoparticles and
> impenetrable surfaces of other bodies. These conditions embody Archimede's principle for
> solids and thus buoyancy results as a direct consequence. We use a variational time stepping
> scheme suitable for general constrained multibody systems we call SPOOK. Each step requires
> the solution of only one Mixed Linear Complementarity Problem (MLCP) with very few
> inequalities, corresponding to solid boundary conditions. We solve this MLCP with a fast
> iterative method. Overall stability is vastly improved in comparison to the unconstrained
> version of SPH, and this allows much larger time steps, and an increase in overall performance
> by two orders of magnitude. Proof of concept is given for computer graphics applications and
> interactive simulations.

### `caltagirone2015`

- **file:** `caltagirone2015_kinematics-scalar-projection.pdf`
- **title:** A Kinematics Scalar Projection Method (KSP) for Incompressible Flows with Variable Density
- **authors:** Jean-Paul Caltagirone and Stéphane Vincent
- **venue:** Open Journal of Fluid Dynamics 5(2), 2015
- **doi:** [10.4236/ojfd.2015.52019](https://doi.org/10.4236/ojfd.2015.52019)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> A new scalar projection method presented for simulating incompressible flows with variable
> density is proposed. It reverses conventional projection algorithm by computing first the
> irrotational component of the velocity and then the pressure. The first phase of the
> projection is purely kinematics. The predicted velocity field is subjected to a discrete
> Hodge-Helmholtz decomposition. The second phase of upgrade of pressure from the density uses
> Stokes’ theorem to explicitly compute the pressure. If all or part of the boundary conditions
> is then fixed on the divergence free physical field, the system required to be solved for the
> scalar potential of velocity becomes a Poisson equation with constant coefficients fitted with
> Dirichlet conditions.

### `chiron2019`

- **file:** `chiron2019_sph-3d-complex-wall-boundaries.pdf`
- **title:** Fast and Accurate SPH Modelling of 3D Complex Wall Boundaries in Viscous and Non Viscous Flows
- **authors:** L. Chiron, M. de Leffe, G. Oger and D. Le Touzé
- **venue:** Computer Physics Communications 234, 2019
- **doi:** [10.1016/j.cpc.2018.08.001](https://doi.org/10.1016/j.cpc.2018.08.001)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> The treatment of wall boundary conditions is a difficult issue in the SPH method and still
> represents a challenging topic in the scientific community. After a review of state of the art
> wall treatment methods, an SPH method for modelling viscous and non-viscous flows in the
> presence of 3D complex wall boundaries is presented. New developments embedded in the proposed
> method include the addition of a Laplacian operator adapted to the adopted formalism, as well
> as a cutface process for calculating the particle/wall interactions on any type of geometry.
> Validations are proposed on a 2D Poiseuille flow, a flow around a 2D cylinder, a 3D
> hydrostatic tank, and a 3D dambreak. Comparisons are performed with results from the
> literature. Finally, an industrial automotive application is presented as illustration of the
> method ability to deal with arbitrarily complex geometries.

### `clavet2005`

- **file:** `clavet2005_particle-based-viscoelastic-fluid-simulation.pdf`
- **title:** Particle-Based Viscoelastic Fluid Simulation
- **authors:** Simon Clavet, Philippe Beaudoin and Pierre Poulin
- **venue:** Proceedings of the 2005 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2005
- **doi:** [10.1145/1073368.1073400](https://doi.org/10.1145/1073368.1073400)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a new particle-based method for viscoelastic fluid simulation. We achieve realistic
> small-scale behavior of substances such as paint or mud as they splash on moving objects.
> Incompressibility and particle anti-clustering are enforced with a double density relaxation
> procedure which updates particle positions according to two opposing pressure terms. From this
> process surface tension effects emerge, enabling drop and filament formation. Elastic and non-
> linear plastic effects are obtained by adding springs with varying rest length between
> particles. We also extend the technique to handle interaction between fluid and dynamic
> objects. Various simulation scenarios are presented including rain drops, fountains, clay
> manipulation, and floating objects. The method is robust and stable, and can animate splashing
> behavior at interactive framerates.

### `cleary1999`

- **file:** `cleary1999_conduction-modelling-using-sph.pdf`
- **title:** Conduction Modelling Using Smoothed Particle Hydrodynamics
- **authors:** Paul W. Cleary and Joseph J. Monaghan
- **venue:** Journal of Computational Physics 148(1), 1999
- **doi:** [10.1006/jcph.1998.6118](https://doi.org/10.1006/jcph.1998.6118)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> Heat transfer is very important in many industrial and geophysical problems. Because these
> problems often have complicated fluid dynamics, there are advantages in solving them using
> Lagrangian methods like smoothed particle hydrodynamics (SPH). Since SPH particles become
> disordered, the second derivative terms may be estimated poorly, especially when materials
> with different properties are adjacent. In this paper we show how a simple alteration to the
> standard SPH formulation ensures continuity of heat flux across discontinuities in material
> properties. A set of rules is formulated for the construction of isothermal boundaries leading
> to accurate conduction solutions. A method for accurate prediction of heat fluxes through
> isothermal boundaries is also given. The accuracy of the SPH conduction solutions is
> demonstrated through a sequence of test problems of increasing complexity.

### `colagrossi2003`

- **file:** `colagrossi2003_interfacial-flows-by-sph.pdf`
- **title:** Numerical Simulation of Interfacial Flows by Smoothed Particle Hydrodynamics
- **authors:** Andrea Colagrossi and Maurizio Landrini
- **venue:** Journal of Computational Physics 191(2), 2003
- **doi:** [10.1016/S0021-9991(03)00324-3](https://doi.org/10.1016/S0021-9991(03)00324-3)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> An implementation of the smoothed particle hydrodynamics (SPH) method is presented to treat
> two-dimensional interfacial flows, that is, flow fields with different fluids separated by
> sharp interfaces. Test cases are presented to show that the present formulation remains stable
> for low density ratios. In particular, results are compared with those obtained by other
> solution techniques, showing a good agreement. The classical dam-break problem is studied by
> the present two-phase approach and the effects of density-ratio variations are discussed. The
> role of air entrapment on loads

### `cornelis2014`

- **file:** `cornelis2014_iisph-flip-for-incompressible-fluids.pdf`
- **title:** IISPH-FLIP for Incompressible Fluids
- **authors:** Jens Cornelis, Markus Ihmsen, Andreas Peer and Matthias Teschner
- **venue:** Computer Graphics Forum 33(2), 2014
- **doi:** [10.1111/cgf.12324](https://doi.org/10.1111/cgf.12324)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We propose to use Implicit Incompressible Smoothed Particle Hydrodynamics (IISPH) for pressure
> projection and boundary handling in Fluid‐Implicit‐Particle (FLIP) solvers for the simulation
> of incompressible fluids. This novel combination addresses two issues of existing SPH and FLIP
> solvers, namely mass preservation in FLIP and efficiency and memory consumption in SPH. First,
> the SPH component enables the simulation of incompressible fluids with perfect mass
> preservation. Second, the FLIP component efficiently enriches the SPH component with detail
> that is comparable to a standard SPH simulation with the same number of particles, while
> improving the performance by a factor of 7 and significantly reducing the memory consumption.
> We demonstrate that the proposed IISPH‐FLIP solver can simulate incompressible fluids with a
> quantifiable, imperceptible density deviation below 0.1%. We show large‐scale scenarios with
> up to 160 million particles that have been processed on a single desktop PC using only 15GB of
> memory. One‐ and two‐way coupled solids are illustrated.

### `cummins1999`

- **file:** `cummins1999_an-sph-projection-method.pdf`
- **title:** An SPH Projection Method
- **authors:** Sharen J. Cummins and Murray Rudman
- **venue:** Journal of Computational Physics 152(2), 1999
- **doi:** [10.1006/jcph.1999.6246](https://doi.org/10.1006/jcph.1999.6246)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> A new formulation is introduced for enforcing incompressibility in Smoothed Particle
> Hydrodynamics (SPH). The method uses a fractional step with the velocity field integrated
> forward in time without enforcing incompressibility. The resulting intermediate velocity field
> is then projected onto a divergence-free space by solving a pressure Poisson equation derived
> from an approximate pressure projection. Unlike earlier approaches used to simulate
> incompressible flows with SPH, the pressure is not a thermodynamic variable and the Courant
> condition is based only on fluid velocities and not on the speed of sound. Although larger
> time-steps can be used, the solution of the resulting elliptic pressure Poisson equation
> increases the total work per time-step. Efficiency comparisons show that the projection method
> has a significant potential to reduce the overall computational expense compared to weakly
> compressible SPH, particularly as the Reynolds number, Re, is increased. Simulations using
> this SPH projection technique show good agreement with finite-difference solutions for a
> vortex spin-down and Rayleigh–Taylor instability. The results, however, indicate that the use
> of an approximate projection to enforce incompressibility leads to error accumulation in the
> density field.

### `degoes2015`

- **file:** `degoes2015_power-particles.pdf`
- **title:** Power Particles: An Incompressible Fluid Solver Based on Power Diagrams
- **authors:** Fernando de Goes, Corentin Wallez, Jin Huang, Dmitry Pavlov and Mathieu Desbrun
- **venue:** ACM Transactions on Graphics 34(4), 2015
- **doi:** [10.1145/2766901](https://doi.org/10.1145/2766901)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> This paper introduces a new particle-based approach to incompressible fluid simulation. We
> depart from previous Lagrangian methods by considering fluid particles no longer purely as
> material points, but also as volumetric parcels that partition the fluid domain. The fluid
> motion is described as a time series of well-shaped power diagrams (hence the name power
> particles ), offering evenly spaced particles and accurate pressure computations. As a result,
> we circumvent the typical excess damping arising from kernel-based evaluations of internal
> forces or density without having recourse to auxiliary Eulerian grids. The versatility of our
> solver is demonstrated by the simulation of multiphase flows and free surfaces.

### `desbrun1996`

- **file:** `desbrun1996_smoothed-particles-deformable-bodies.pdf`
- **title:** Smoothed Particles: A New Paradigm for Animating Highly Deformable Bodies
- **authors:** Mathieu Desbrun and Marie-Paule Gascuel
- **venue:** Computer Animation and Simulation '96 (Eurographics Workshop), 1996
- **doi:** [10.1007/978-3-7091-7486-9_5](https://doi.org/10.1007/978-3-7091-7486-9_5)
- **abstract from:** PDF text layer, p.2 (ligatures expanded)

> This paper presents a new formalism for simulating highly deformable bodies with a particle
> system. Smoothed particles represent sample points that enable the approximation of the values
> and derivatives of local physical quantities inside a medium. They ensure valid and stable
> simulation of state equations that describe the physical behavior of the material. We extend
> the initial formalism, first introduced for simulating cosmological fluids, to the animation
> of inelastic bodies with a wide range of stiffness and viscosity. We show that the smoothed
> particles paradigm leads to a coherent definition of the object’s surface as an iso-surface of
> the mass density function. Implementation issues are discussed, including an efficient
> integration scheme using individually adapted time steps to integrate particle motion.
> Animation requires a linear complexity in the number of particles, offering reasonable time
> and memory use.

### `deul2014`

- **file:** `deul2014_position-based-rigid-body-dynamics.pdf`
- **title:** Position-Based Rigid-Body Dynamics
- **authors:** Crispin Deul, Patrick Charrier and Jan Bender
- **venue:** Computer Animation and Virtual Worlds 27(2), 2016
- **doi:** [10.1002/cav.1614](https://doi.org/10.1002/cav.1614)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We propose a position‐based approach for large‐scale simulations of rigid bodies at
> interactive frame rates. Our method solves positional constraints between rigid bodies and can
> therefore be seamlessly integrated into other position‐based methods. Interaction of particles
> and rigid bodies through common constraints enables two‐way coupling with deformables. The
> method exhibits exceptional performance and stability while being user controllable and easy
> to implement. Various results demonstrate the practicability of our method for the resolution
> of collisions, contacts, stacking and joint constraints. Copyright © 2014 John Wiley & Sons,
> Ltd.

### `ferrand2013`

- **file:** `ferrand2013_unified-semi-analytical-wall-bc.pdf`
- **title:** Unified Semi-Analytical Wall Boundary Conditions for Inviscid, Laminar or Turbulent Flows in the Meshless SPH Method
- **authors:** M. Ferrand, D. R. Laurence, B. D. Rogers, D. Violeau and C. Kassiotis
- **venue:** International Journal for Numerical Methods in Fluids 71(4), 2013
- **doi:** [10.1002/fld.3666](https://doi.org/10.1002/fld.3666)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> Wall boundary conditions in smoothed particle hydrodynamics (SPH) is a key issue to perform
> accurate simulations. We propose here a new approach based on a renormalising factor for
> writing all boundary terms. This factor depends on the local shape of a wall and on the
> position of a particle relative to the wall, which is described by segments (in two-
> dimensions), instead of the cumbersome fictitious or ghost particles used in most existing SPH
> models. By solving a dynamic equation for the renormalising factor, we significantly improve
> traditional wall treatment in SPH, for pressure forces, wall friction and turbulent
> conditions. The new model is demonstrated for cases including hydrostatic conditions for still
> water in a tank of complex geometry and a dam break over triangular bed profile with sharp
> angle where significant improved behaviour is obtained in comparison with the conventional
> boundary techniques. The latter case is also compared with a finite volume and volume-of-fluid
> scheme. The performance of the model for a two-dimensional laminar flow in a channel is
> demonstrated where the profiles of velocity are in agreement with the theoretical ones,
> demonstrating that the derived wall shear stress balances the pressure gradient. Finally, the
> performance of the model is demonstrated for flow in a schematic fish pass where both the
> velocity field and turbulent viscosity fields are satisfactorily reproduced compared with
> mesh-based codes.

### `foster1996`

- **file:** `foster1996_realistic-animation-of-liquids.pdf`
- **title:** Realistic Animation of Liquids
- **authors:** Nick Foster and Dimitri Metaxas
- **venue:** Graphical Models and Image Processing 58(5), 1996
- **doi:** [10.1006/gmip.1996.0039](https://doi.org/10.1006/gmip.1996.0039)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a comprehensive methodology for realistically animating liquid phenomena. Our
> approach unifies existing computer graphics techniques for simulating fluids and extends them
> by incorporating more complex behavior. It is based on the Navier–Stokes equations which
> couple momentum and mass conservation to completely describe fluid motion. Our starting point
> is an environment containing an arbitrary distribution of fluid, and submerged or
> semisubmerged obstacles. Velocity and pressure are defined everywhere within this environment
> and updated using a set of finite difference expressions. The resulting vector and scalar
> fields are used to drive a height field equation representing the liquid surface. The nature
> of the coupling between obstacles in the environment and free variables allows for the
> simulation of a wide range of effects that were not possible with previous computer graphics
> fluid models. Wave effects such as reflection, refraction, and diffraction, as well as
> rotational effects such as eddies, vorticity, and splashing are a natural consequence of
> solving the system. In addition, the Lagrange equations of motion are used to place buoyant
> dynamic objects into a scene and track the position of spray and foam during the animation
> process. Typical disadvantages to dynamic simulations such as poor scalability and lack of
> control are addressed by assuming that stationary obstacles align with grid cells during the
> finite difference discretization, and by appending terms to the Navier–Stokes equations to
> include forcing functions. Free surfaces in our system are represented as either a collection
> of massless particles in 2D, or a height field which is suitable for many of the water
> rendering algorithms presented by researchers in recent years.

### `foster2001`

- **file:** `foster2001_practical-animation-of-liquids.pdf`
- **title:** Practical Animation of Liquids
- **authors:** Nick Foster and Ronald Fedkiw
- **venue:** Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 2001
- **doi:** [10.1145/383259.383261](https://doi.org/10.1145/383259.383261)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a general method for modeling and animating liquids. The system is specifically
> designed for computer animation and handles viscous liquids as they move in a 3D environment
> and interact with graphics primitives such as parametric curves and moving polygons. We
> combine an appropriately modified semi-Lagrangian method with a new approach to calculating
> fluid flow around objects. This allows us to efficiently solve the equations of motion for a
> liquid while retaining enough detail to obtain realistic looking behavior. The object
> interaction mechanism is extended to provide control over the liquid s 3D motion. A high
> quality surface is obtained from the resulting velocity field using a novel adaptive technique
> for evolving an implicit surface.

### `fujisawa2015`

- **file:** `fujisawa2015_efficient-boundary-handling-modified-density.pdf`
- **title:** An Efficient Boundary Handling with a Modified Density Calculation for SPH
- **authors:** Makoto Fujisawa and Kenjiro T. Miura
- **venue:** Computer Graphics Forum 34(7), 2015
- **doi:** [10.1111/cgf.12754](https://doi.org/10.1111/cgf.12754)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We propose a new boundary handling method for smoothed particle hydrodynamics (SPH). Previous
> approaches required the use of boundary particles to prevent particles from sticking to the
> boundary. We address this issue by correcting the fundamental equations of SPH with the
> integration of a kernel function. Our approach is able to directly handle triangle mesh
> boundaries without the need for boundary particles. We also show how our approach can be
> integrated into a position-based fluid framework.

### `ghia1982`

- **file:** `ghia1982_high-re-lid-driven-cavity-multigrid.pdf`
- **title:** High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method
- **authors:** U. Ghia, K. N. Ghia and C. T. Shin
- **venue:** Journal of Computational Physics 48(3), 1982
- **doi:** [10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> The vorticity-stream function formulation of the two-dimensional incompressible NavierStokes
> equations is used to study the effectiveness of the coupled strongly implicit multigrid (CSI-
> MG) method in the determination of high-Re fine-mesh flow solutions. The driven flow in a
> square cavity is used as the model problem. Solutions are obtained for configurations with
> Reynolds number as high as 10.000 and meshes consisting of as many as 257 x 257 points. For Re
> = 1000, the (129 x 129) grid solution required 1.5 minutes of CPU time on the AMDAHL 470 V/6
> computer. Because of the appearance of one or more secondary vortices in the flow field,
> uniform mesh refinement was preferred to the use of one-dimensional gridclustering coordinate
> transformations.

### `gissler2017`

- **file:** `gissler2017_generalized-drag-force.pdf`
- **title:** Generalized Drag Force for Particle-Based Simulations
- **authors:** Christoph Gissler, Stefan Band, Andreas Peer, Markus Ihmsen and Matthias Teschner
- **venue:** Computers \& Graphics 69, 2017
- **doi:** [10.1016/j.cag.2017.09.002](https://doi.org/10.1016/j.cag.2017.09.002)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> Computing the forces acting from a surrounding air phase onto a particle-based fluid or rigid
> object is challenging. Simulating the air phase and modeling the interactions using a
> multiphase approach is computationally expensive. Furthermore, stability issues may arise in
> such multiphase simulations. In contrast, the effects from the air can be approximated
> efficiently by employing a drag equation. Here, for plausible effects, the parameterization is
> important but challenging. We present a drag force discretization based on the drag equation
> that acts on each particle separately. It is used to compute the effects of air onto particle-
> based fluids and rigid objects. Our presented approach calculates the exposed surface area and
> drag coefficient of each particle. For fluid particles, we approximate their deformation to
> improve the drag coefficient estimation. The resulting effects are validated by comparing them
> to the results of multiphase SPH simulations. We further show the practicality of our approach
> by combining it with different types of SPH fluid solvers and by simulating multiple, complex
> scenes.

### `harada2007`

- **file:** `harada2007_sph-in-complex-shapes.pdf`
- **title:** Smoothed Particle Hydrodynamics in Complex Shapes
- **authors:** Takahiro Harada, Seiichi Koshizuka and Yoichiro Kawaguchi
- **venue:** Proceedings of the 23rd Spring Conference on Computer Graphics (SCCG), 2007
- **doi:** [10.1145/2614348.2614375](https://doi.org/10.1145/2614348.2614375)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this paper, we propose an improved computation model of wall boundary in Smoothed Particle
> Hydrodynamics, a particle method for fluid simulation. Generally, particle methods calculate a
> wall boundary by converting it to wall particles. The proposed method uses a distance function
> calculated from a polygon model as a wall boundary. As a result, fluid motion in complex
> shapes can be simulated easily. Since the method does not use wall particles, it is able to
> represent a wall boundary without increasing the particle resolution. When a boundary is
> represented by wall particles, we have to generate a large number of wall particles. The
> proportion of the number of wall particles in total number of particles is high. However the
> proposed method does not need wall particles, it can reduce the total number of particles.
> After the simulation, surface mesh is usually constructed to visualize a simulation result
> from particles. However, it is difficult to generate smooth surface from them. We also propose
> a visualization method which can construct smooth fluid surfaces contacting with a wall
> boundary.

### `he2012`

- **file:** `he2012_local-poisson-sph-viscous-incompressible.pdf`
- **title:** Local Poisson SPH for Viscous Incompressible Fluids
- **authors:** Xiaowei He, Ning Liu, Sheng Li, Hongan Wang and Guoping Wang
- **venue:** Computer Graphics Forum 31(6), 2012
- **doi:** [10.1111/j.1467-8659.2012.03074.x](https://doi.org/10.1111/j.1467-8659.2012.03074.x)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Enforcing fluid incompressibility is one of the time‐consuming aspects in SPH. In this paper,
> we present a local Poisson SPH (LPSPH) method to solve incompressibility for particle based
> fluid simulation. Considering the pressure Poisson equation, we first convert it into an
> integral form, and then apply a discretization to convert the continuous integral equation to
> a discretized summation over all the particles in the local pressure integration domain
> determined by the local geometry. To control the approximation error, we further integrate our
> local pressure solver into the predictive‐corrective framework to avoid the computational cost
> of solving a pressure Poisson equation globally. Our method can effectively eliminate the
> large density deviations mainly caused by the solid boundary treatment and free surface
> topological change, and show advantage of a higher convergence rate over the
> predictive‐corrective incompressible SPH (PCISPH).

### `he2014`

- **file:** `he2014_robust-simulation-sparsely-sampled-thin-features.pdf`
- **title:** Robust Simulation of Sparsely Sampled Thin Features in SPH-Based Free Surface Flows
- **authors:** Xiaowei He, Huamin Wang, Fengjun Zhang, Hongan Wang, Guoping Wang and Kun Zhou
- **venue:** ACM Transactions on Graphics 34(1), 2014
- **doi:** [10.1145/2682630](https://doi.org/10.1145/2682630)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> Smoothed particle hydrodynamics (SPH) is efficient, mass preserving, and flexible in handling
> topological changes. However, sparsely sampled thin features are difficult to simulate in SPH-
> based free surface flows, due to a number of robustness and stability issues. In this article,
> we address this problem from two perspectives: the robustness of surface forces and the
> numerical instability of thin features. We present a new surface tension force scheme based on
> a free surface energy functional, under the diffuse interface model. We develop an efficient
> way to calculate the air pressure force for free surface flows, without using air particles.
> Compared with previous surface force formulae, our formulae are more robust against particle
> sparsity in thin feature cases. To avoid numerical instability on thin features, we propose to
> adjust the internal pressure force by estimating the internal pressure at two scales and
> filtering the force using a geometry-aware anisotropic kernel. Our result demonstrates the
> effectiveness of our algorithms in handling a variety of sparsely sampled thin liquid
> features, including thin sheets, thin jets, and water splashes.

### `hu2005`

- **file:** `hu2005_multi-phase-sph-macroscopic-mesoscopic.pdf`
- **title:** A Multi-Phase SPH Method for Macroscopic and Mesoscopic Flows
- **authors:** X. Y. Hu and N. A. Adams
- **venue:** Journal of Computational Physics 213(2), 2006
- **doi:** [10.1016/j.jcp.2005.09.001](https://doi.org/10.1016/j.jcp.2005.09.001)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> A multi-phase smoothed particle hydrodynamics (SPH) method for both macroscopic and mesoscopic
> flows is proposed. Since the particle-averaged spatial derivative approximations are derived
> from a particle smoothing function in which the neighboring particles only contribute to the
> specific volume, while maintaining mass conservation, the new method handles density
> discontinuities across phase interfaces naturally. Accordingly, several aspects of multi-phase
> interactions are addressed. First, the newly formulated viscous terms allow for a
> discontinuous viscosity and ensure continuity of velocity and shear stress across the phase
> interface. Based on this formulation thermal fluctuations are introduced in a straightforward
> way. Second, a new simple algorithm capable for three or more immiscible phases is developed.
> Mesocopic interface slippage is included based on the apparent slip assumption which ensures
> continuity at the phase interface. To show the validity of the present method numerical
> examples on capillary waves, three-phase interactions, drop deformation in a shear flow, and
> mesoscopic channel

### `hu2007`

- **file:** `hu2007_an-incompressible-multi-phase-sph-method.pdf`
- **title:** An Incompressible Multi-Phase SPH Method
- **authors:** X. Y. Hu and N. A. Adams
- **venue:** Journal of Computational Physics 227(1), 2007
- **doi:** [10.1016/j.jcp.2007.07.013](https://doi.org/10.1016/j.jcp.2007.07.013)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> An incompressible multi-phase SPH method is proposed. In this method, a fractional time-step
> method is introduced to enforce both the zero-density-variation condition and the velocity-
> divergence-free condition at each full time-step. To obtain sharp density and viscosity
> discontinuities in an incompressible multi-phase flow a new multi-phase projection
> formulation, in which the discretized gradient and divergence operators do not require a
> differentiable density or viscosity field is proposed. Numerical examples for Taylor–Green
> flow, capillary waves, drop deformation in shear flows and for Rayleigh–Taylor instability are
> presented and compared to theoretical solutions or references from literature. The results
> suggest good accuracy and convergence properties of the proposed method.

### `huber2015`

- **file:** `huber2015_boundary-handling-at-cloth-fluid-contact.pdf`
- **title:** Boundary Handling at Cloth-Fluid Contact
- **authors:** Markus Huber, Bernhard Eberhardt and Daniel Weiskopf
- **venue:** Computer Graphics Forum 34(1), 2015
- **doi:** [10.1111/cgf.12455](https://doi.org/10.1111/cgf.12455)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a robust and efficient method for the two‐way coupling between particle‐based fluid
> simulations and infinitesimally thin solids represented by triangular meshes. Our approach is
> based on a hybrid method that combines a repulsion force approach with a continuous
> intersection handling to guarantee that no penetration occurs. Moreover, boundary conditions
> for the tangential component of the fluid's velocity are implemented to model the different
> slip conditions. The proposed method is particularly useful for dynamic surfaces, like cloth
> and thin shells. In addition, we demonstrate how standard fluid surface reconstruction
> algorithms can be modified to prevent the calculated surface from intersecting close objects.
> For both the two‐way coupling and the surface reconstruction, we take into account that the
> fluid can wet the cloth. We have implemented our approach for the bidirectional interaction
> between liquid simulations based on Smoothed Particle Hydrodynamics (SPH) and standard
> mesh‐based cloth simulation systems.

### `ihmsen2011`

- **file:** `ihmsen2011_parallel-sph-implementation-multi-core-cpus.pdf`
- **title:** A Parallel SPH Implementation on Multi-Core CPUs
- **authors:** Markus Ihmsen, Nadir Akinci, Markus Becker and Matthias Teschner
- **venue:** Computer Graphics Forum 30(1), 2011
- **doi:** [10.1111/j.1467-8659.2010.01832.x](https://doi.org/10.1111/j.1467-8659.2010.01832.x)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> This paper presents a parallel framework for simulating fluids with the Smoothed Particle
> Hydrodynamics (SPH) method. For low computational costs per simulation step, efficient
> parallel neighbourhood queries are proposed and compared. To further minimize the computing
> time for entire simulation sequences, strategies for maximizing the time step and the
> respective consequences for parallel implementations are investigated. The presented
> experiments illustrate that the parallel framework can efficiently compute large numbers of
> time steps for large scenarios. In the context of neighbourhood queries, the paper presents
> optimizations for two efficient instances of uniform grids, that is, spatial hashing and index
> sort. For implementations on parallel architectures with shared memory, the paper discusses
> techniques with improved cache‐hit rate and reduced memory transfer. The performance of the
> parallel implementations of both optimized data structures is compared. The proposed solutions
> focus on systems with multiple CPUs. Benefits and challenges of potential GPU implementations
> are only briefly discussed.

### `ihmsen2014star`

- **file:** `ihmsen2014star_sph-fluids-in-computer-graphics-star.pdf`
- **title:** SPH Fluids in Computer Graphics
- **authors:** Markus Ihmsen, Jens Orthmann, Barbara Solenthaler, Andreas Kolb and Matthias Teschner
- **venue:** Eurographics 2014 -- State of the Art Reports (STAR), 2014
- **doi:** [10.2312/egst.20141034](https://doi.org/10.2312/egst.20141034)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Smoothed Particle Hydrodynamics (SPH) has been established as one of the major concepts for
> fluid animation in computer graphics. While SPH initially gained popularity for interactive
> free-surface scenarios, it has emerged to be a fully fledged technique for state-of-the-art
> fluid animation with versatile effects. Nowadays, complex scenes with millions of sampling
> points, one- and two-way coupled rigid and elastic solids, multiple phases and additional
> features such as foam or air bubbles can be computed at reasonable expense. This state-of-the-
> art report summarizes SPH research within the graphics community.

### `kang2014`

- **file:** `kang2014_incompressible-sph-divergence-free-condition.pdf`
- **title:** Incompressible SPH Using the Divergence-Free Condition
- **authors:** Nahyup Kang and Donghoon Sagong
- **venue:** Computer Graphics Forum 33(7), 2014
- **doi:** [10.1111/cgf.12490](https://doi.org/10.1111/cgf.12490)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this paper, we present a novel SPH framework to simulate incompressible fluid that
> satisfies both the divergence‐ free condition and the density‐invariant condition. In our
> framework, the two conditions are applied separately. First, the divergence‐free condition is
> enforced when solving the momentum equation. Later, the density‐invariant condition is applied
> after the time integration of the particle positions. Our framework is a purely Lagrangian
> approach so that no auxiliary grid is required. Compared to the previous density‐invariant
> based SPH methods, the proposed method is more accurate due to the explicit satisfaction of
> the divergence‐free condition. We also propose a modified boundary particle method for
> handling the free‐slip condition. In addition, two simple but effective methods are proposed
> to reduce the particle clumping artifact induced by the density‐invariant condition.

### `keiser2006`

- **file:** `keiser2006_multiresolution-particle-based-fluids.pdf`
- **title:** Multiresolution Particle-Based Fluids
- **authors:** Richard Keiser, Bart Adams, Philip Dutré, Leonidas J. Guibas and Mark Pauly
- **venue:** Department of Computer Science, ETH Zurich, No. 520, 2006
- **doi:** [10.3929/ethz-a-006780981](https://doi.org/10.3929/ethz-a-006780981)
- **abstract from:** PDF text layer, p.2 (ligatures expanded)

> We present a new multiresolution particle method for fluid simulation. The discretization of
> the fluid dynamically adapts to the characteristics of the flow to resolve fine-scale visual
> detail, while reducing the overall complexity of the computations. We introduce the concept of
> virtual particles to implement efficient refinement and coarsification operators, and to
> achieve a consistent coupling between particles at different resolution levels, leading to
> speedups of up to a factor of six as compared to single resolution simulations. Our system
> supports multiphase effects such as bubbles and foam, as well as rigid body interactions,
> based on a unified particle interaction metaphor. The waterair interface is tracked with a
> Lagrangian level set approach using a novel Delaunay-based surface contouring method that
> accurately resolves fine-scale surface detail while guaranteeing preservation of fluid volume.

### `koschier2022`

- **file:** `koschier2022_survey-sph-methods-computer-graphics.pdf`
- **title:** A Survey on SPH Methods in Computer Graphics
- **authors:** Dan Koschier, Jan Bender, Barbara Solenthaler and Matthias Teschner
- **venue:** Computer Graphics Forum 41(2), 2022
- **doi:** [10.1111/cgf.14508](https://doi.org/10.1111/cgf.14508)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Throughout the past decades, the graphics community has spent major resources on the research
> and development of physics simulators on the mission to computer‐generate behaviors achieving
> outstanding visual effects or to make the virtual world indistinguishable from reality. The
> variety and impact of recent research based on Smoothed Particle Hydrodynamics (SPH)
> demonstrates the concept's importance as one of the most versatile tools for the simulation of
> fluids and solids. With this survey, we offer an overview of the developments and still‐active
> research on physics simulation methodologies based on SPH that has not been addressed in
> previous SPH surveys. Following an introduction about typical SPH discretization techniques,
> we provide an overview over the most used incompressibility solvers and present novel insights
> regarding their relation and conditional equivalence. The survey further covers recent
> advances in implicit and particle‐based boundary handling and sampling techniques. While SPH
> is best known in the context of fluid simulation we discuss modern concepts to augment the
> range of simulatable physical characteristics including turbulence, highly viscous matter,
> deformable solids, as well as rigid body contact handling. Besides the purely numerical
> approaches, simulation techniques aided by machine learning are on the rise. Thus, the survey
> discusses recent data‐driven approaches and the impact of differentiable solvers on artist
> control. Finally, we provide context for discussion by outlining existing problems and
> opportunities to open up new research directions.

### `kruisbrink2018`

- **file:** `kruisbrink2018_sph-particle-collisions.pdf`
- **title:** SPH Particle Collisions for the Reduction of Particle Clustering, Interface Stabilisation and Wall Modelling
- **authors:** Arno C. H. Kruisbrink, Stan P. Korzilius, Frazer R. Pearce and Hervé P. Morvan
- **venue:** Journal of Applied Mathematics and Physics 6(9), 2018
- **doi:** [10.4236/jamp.2018.69158](https://doi.org/10.4236/jamp.2018.69158)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> The pair-wise forces in the SPH momentum equation guarantee the conservation of momentum, but
> they cannot prevent particle clustering and wall penetration. Particle clustering may occur
> for several reasons. A fundamental issue is the tensile instability, which is caused by
> negative numerical pressures. Clustering may also occur due to certain properties of the
> kernel gradient. Discontinuities in the pressure and its gradient, due to surface tension and
> gravity, may cause particle instabilities near the interface between two fluids. Wall
> penetration is also a form of particle clustering. In this paper the particle collision
> concept is introduced to suppress particle clustering. Here, the use of kinematic conditions
> (motion) rather than dynamic conditions (forces) is explored. These kinematic conditions are
> obtained from kinetic collision theory. Conservation of momentum is maintained, and under
> elastic conditions conservation of energy as well. The particle collision model only becomes
> active when needed. It may be seen as a particle shifting method, in the sense that the
> velocities are changed, and as a consequence of that the particle positions change. It is
> demonstrated in several case studies that the particle collision model allows for realistic
> (low) viscosities. It was also found to stabilise the interface between two fluids up to high,
> realistic density ratios (1000:1) in typical liquid-gas applications. As such it can be used
> as a multi-fluid model. The concept allows for real wave speed ratios (and far beyond), which,
> as well as real viscosities, are essential in the modelling of heat transfer applications. The
> collisions with walls allow for no-slip conditions at real viscosities while wall penetration
> is suppressed. In summary, the particle collision model makes SPH more robust for engineering.

### `kugelstadt2021`

- **file:** `kugelstadt2021_fast-corotated-elastic-sph-solids.pdf`
- **title:** Fast Corotated Elastic SPH Solids with Implicit Zero-Energy Mode Control
- **authors:** Tassilo Kugelstadt, Jan Bender, José Antonio Fernández-Fernández, Stefan Rhys Jeske, Fabian Löschner and Andreas Longva
- **venue:** Proceedings of the ACM on Computer Graphics and Interactive Techniques 4(3), 2021
- **doi:** [10.1145/3480142](https://doi.org/10.1145/3480142)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> We develop a new operator splitting formulation for the simulation of corotated linearly
> elastic solids with Smoothed Particle Hydrodynamics (SPH). Based on the technique of
> Kugelstadt et al. [2018] originally developed for the Finite Element Method (FEM), we split
> the elastic energy into two separate terms corresponding to stretching and volume
> conservation, and based on this principle, we design a splitting scheme compatible with SPH.
> The operator splitting scheme enables us to treat the two terms separately, and because the
> stretching forces lead to a stiffness matrix that is constant in time, we are able to
> prefactor the system matrix for the implicit integration step. Solid-solid contact and fluid-
> solid interaction is achieved through a unified pressure solve. We demonstrate more than an
> order of magnitude improvement in computation time compared to a state-of-the-art SPH
> simulator for elastic solids. We further improve the stability and reliability of the
> simulation through several additional contributions. We introduce a new implicit penalty
> mechanism that suppresses zero-energy modes inherent in the SPH formulation for elastic
> solids, and present a new, physics-inspired sampling algorithm for generating high-quality
> particle distributions for the rest shape of an elastic solid. We finally also devise an
> efficient method for interpolating vertex positions of a high-resolution surface mesh based on
> the SPH particle positions for use in high-fidelity visualization.

### `kulasegaram2004`

- **file:** `kulasegaram2004_variational-contact-algorithm-rigid-boundaries.pdf`
- **title:** A Variational Formulation Based Contact Algorithm for Rigid Boundaries in Two-Dimensional SPH Applications
- **authors:** S. Kulasegaram, J. Bonet, R. W. Lewis and M. Profit
- **venue:** Computational Mechanics 33(4), 2004
- **doi:** [10.1007/s00466-003-0534-0](https://doi.org/10.1007/s00466-003-0534-0)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> Smooth particle Hydrodynamics (SPH) is one of the most effective meshless techniques used in
> computational mechanics. SPH approximations are simple and allow greater flexibility in
> various engineering applications. However, modelling of particle-boundary interactions in SPH
> computations has always been considered an aspect that requires further research. A number of
> techniques have been developed to model particle-boundary interactions in SPH and allied
> methods. In this paper, an innovative approach is introduced to handle the contact between
> Lagrangian SPH particles and rigid solid boundaries. The formulation of boundary contact
> forces are derived based on a variational formulation, thus directly ensuring the
> conservativeness of the governing equations. In addition, the new elegant boundary contact
> force terms maintain the simplicity of the SPH governing equations.

### `leroy2014`

- **file:** `leroy2014_unified-semi-analytical-wall-bc-2d-isph.pdf`
- **title:** Unified Semi-Analytical Wall Boundary Conditions Applied to 2-D Incompressible SPH
- **authors:** A. Leroy, D. Violeau, M. Ferrand and C. Kassiotis
- **venue:** Journal of Computational Physics 261, 2014
- **doi:** [10.1016/j.jcp.2013.12.035](https://doi.org/10.1016/j.jcp.2013.12.035)
- **abstract from:** PDF text layer, p.2 (ligatures expanded)

> This work aims at improving the 2-D incompressible SPH model (ISPH) by adapting it to the
> unified semi-analytical wall boundary conditions proposed by Ferrand et al. [10]. The ISPH
> algorithm considered is as proposed by Lind et al. [25], based on the projection method with a
> divergence-free velocity field and using a stabilising procedure based on particle shifting.
> However, we consider an extension of this model to Reynolds-Averaged Navier-Stokes equations
> based on the k − ǫ turbulent closure model, as done in [10]. The discrete SPH operators are
> modified by the new description of the wall boundary conditions. In particular, a boundary
> term appears in the Laplacian operator, which makes it possible to accurately impose a von
> Neumann pressure wall boundary condition that corresponds to impermeability. The shifting and
> free-surface detection algorithms have also been adapted to the new boundary conditions.
> Moreover, a new way to compute the wall renormalisation factor in the frame of the unified
> semi-analytical boundary conditions is proposed in order to decrease the computational time.
> We present several verifications to the present approach, including a lid-driven cavity, a
> water column collapsing on a wedge and a periodic schematic fish-pass. Our results are
> compared to Finite Volumes methods, using Volume of Fluids in the case of free-surface flows.
> We briefly investigate the convergence of the method and prove its ability to model complex
> free-surface and turbulent flows. The results are generally improved when compared to a weakly
> compressible SPH model with the same boundary conditions, especially ∗ Corresponding author.
> tel: +33 (0)6 67 88 92 13 Email addresses: agnes.leroy@edf.fr (A. Leroy),
> damien.violeau@edf.fr (D. Violeau), martin.ferrand@edf.fr (M. Ferrand),
> christophe.kassiotis@enpc.fr (C. Kassiotis) Preprint submitted to Elsevier January 21, 2014 in
> terms of pressure prediction.

### `liu2021`

- **file:** `liu2021_turbulent-details-vorticity-refinement.pdf`
- **title:** Turbulent Details Simulation for SPH Fluids via Vorticity Refinement
- **authors:** Sinuo Liu, Xiaokun Wang, Xiaojuan Ban, Yanrui Xu, Jing Zhou, Jiří Kosinka and Alexandru C. Telea
- **venue:** Computer Graphics Forum 40(1), 2021
- **doi:** [10.1111/cgf.14095](https://doi.org/10.1111/cgf.14095)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> A major issue in smoothed particle hydrodynamics (SPH) approaches is the numerical dissipation
> during the projection process, especially under coarse discretizations. High‐frequency
> details, such as turbulence and vortices, are smoothed out, leading to unrealistic results. To
> address this issue, we introduce a vorticity refinement (VR) solver for SPH fluids with
> negligible computational overhead. In this method, the numerical dissipation of the vorticity
> field is recovered by the difference between the theoretical and the actual vorticity, so as
> to enhance turbulence details. Instead of solving the Biot‐Savart integrals, a stream
> function, which is easier and more efficient to solve, is used to relate the vorticity field
> to the velocity field. We obtain turbulence effects of different intensity levels by changing
> an adjustable parameter. Since the vorticity field is enhanced according to the curl field,
> our method can not only amplify existing vortices, but also capture additional turbulence. Our
> VR solver is straightforward to implement and can be easily integrated into existing SPH
> methods.

### `losasso2008`

- **file:** `losasso2008_two-way-coupled-sph-particle-level-set.pdf`
- **title:** Two-Way Coupled SPH and Particle Level Set Fluid Simulation
- **authors:** Frank Losasso, Jerry O. Talton, Nipun Kwatra and Ronald Fedkiw
- **venue:** IEEE Transactions on Visualization and Computer Graphics 14(4), 2008
- **doi:** [10.1109/TVCG.2008.37](https://doi.org/10.1109/TVCG.2008.37)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Grid-based methods have difficulty resolving features on or below the scale of the underlying
> grid. Although adaptive methods (e.g. RLE, octrees) can alleviate this to some degree,
> separate techniques are still required for simulating small-scale phenomena such as spray and
> foam, especially since these more diffuse materials typically behave quite differently than
> their denser counterparts. In this paper, we propose a two-way coupled simulation framework
> that uses the particle level set method to efficiently model dense liquid volumes and a
> smoothed particle hydrodynamics (SPH) method to simulate diffuse regions such as sprays. Our
> novel SPH method allows us to simulate both dense and diffuse water volumes, fully
> incorporates the particles that are automatically generated by the particle level set method
> in under-resolved regions, and allows for two way mixing between dense SPH volumes and grid-
> based liquid representations.

### `lucy1977`

- **file:** `lucy1977_a-numerical-approach-fission-hypothesis.pdf`
- **title:** A Numerical Approach to the Testing of the Fission Hypothesis
- **authors:** L. B. Lucy
- **venue:** The Astronomical Journal 82(12), 1977
- **doi:** [10.1086/112164](https://doi.org/10.1086/112164)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> A finite-size particle scheme for the numerical solution of twoand three-dimensional
> gasdynamic problems of astronomical interest is described and tested. The scheme is then
> applied to the fission problem for optically thick protostars. Results are given, showing the
> evolution of one such protostar from an initial state as a single rotating star to a final
> state as a triple system whose components contain 60% of the original mass. The decisiveness
> of this numerical test of the fission hypothesis and its relevance to observed binaries are
> briefly discussed.

### `macklin2013`

- **file:** `macklin2013_position-based-fluids.pdf`
- **title:** Position Based Fluids
- **authors:** Miles Macklin and Matthias Müller
- **venue:** ACM Transactions on Graphics 32(4), 2013
- **doi:** [10.1145/2461912.2461984](https://doi.org/10.1145/2461912.2461984)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> In fluid simulation, enforcing incompressibility is crucial for realism; it is also
> computationally expensive. Recent work has improved efficiency, but still requires time-steps
> that are impractical for real-time applications. In this work we present an iterative density
> solver integrated into the Position Based Dynamics framework (PBD). By formulating and solving
> a set of positional constraints that enforce constant density, our method allows similar
> incompressibility and convergence to modern smoothed particle hydro-dynamic (SPH) solvers, but
> inherits the stability of the geometric, position based dynamics method, allowing large time
> steps suitable for real-time applications. We incorporate an artificial pressure term that
> improves particle distribution, creates surface tension, and lowers the neighborhood
> requirements of traditional SPH. Finally, we address the issue of energy loss by applying
> vorticity confinement as a velocity post process.

### `marrone2011`

- **file:** `marrone2011_delta-sph-violent-impact-flows.pdf`
- **title:** $\delta$-SPH Model for Simulating Violent Impact Flows
- **authors:** S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. Le Touzé and G. Graziani
- **venue:** Computer Methods in Applied Mechanics and Engineering 200(13--16), 2011
- **doi:** [10.1016/j.cma.2010.12.016](https://doi.org/10.1016/j.cma.2010.12.016)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> A smoothed particle hydrodynamics model with numerical diffusive terms, hereinafter referred
> to as d-SPH [1] is used to analyze violent water flows. The boundary conditions on solid
> surfaces of arbitrary shape are enforced with a new technique based on fixed ghost particles.
> The violent impacts studied result from dam-break water flows striking obstacles of different
> shapes. The numerical results are validated against experimental data from the literature and
> solutions from a Navier–Stokes Level-Set solver. Predicted impact pressures are also compared
> with analytical solutions. The proposed scheme thus proves to be accurate and robust for the
> prediction of global and local loads of impact

### `monaghan1989`

- **file:** `monaghan1989_on-the-problem-of-penetration.pdf`
- **title:** On the Problem of Penetration in Particle Methods
- **authors:** Joseph J. Monaghan
- **venue:** Journal of Computational Physics 82(1), 1989
- **doi:** [10.1016/0021-9991(89)90032-6](https://doi.org/10.1016/0021-9991(89)90032-6)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> A method is described which prevents penetration when particle methods are used to simulate
> streams of fluid impinging on each other. The method does not produce dissipation but it does
> produce extra dispersion.

### `monaghan2002`

- **file:** `monaghan2002_sph-compressible-turbulence.pdf`
- **title:** SPH Compressible Turbulence
- **authors:** J. J. Monaghan
- **venue:** Monthly Notices of the Royal Astronomical Society 335(3), 2002
- **doi:** [10.1046/j.1365-8711.2002.05678.x](https://doi.org/10.1046/j.1365-8711.2002.05678.x)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this paper a smoothed particle hydrodynamics (SPH) version of the alpha turbulence model
> devised by Holm and his colleagues is formulated for compressible flow with a resolution that
> varies in space and time. The alpha model involves two velocity fields. One velocity field is
> obtained from the momentum equation, the other by averaging this velocity field as in the
> version of SPH called XSPH. The particles (fluid elements) are moved with the averaged
> velocity. In analogy to the continuum alpha model we obtain a particle Lagrangian from which
> the SPH alpha equations can be derived. The system satisfies a discrete Kelvin circulation
> theorem identical to that obtained with no velocity averaging. In addition, the energy, linear
> and angular momentum are conserved. We show that the continuum equivalent of the SPH equations
> are identical to the continuum alpha model, and we conjecture that they will have the same
> desirable features of the continuum model including the reduction of energy in the high-
> wavenumber modes even when the dissipation is zero. Regardless of issues concerning turbulence
> modelling, the SPH alpha model is a powerful extension of the XSPH algorithm, which reduces
> disorder at short length-scales and retains the constants of the motion. The SPH alpha model
> is simple to implement.

### `monaghan2005`

- **file:** `monaghan2005_smoothed-particle-hydrodynamics-review.pdf`
- **title:** Smoothed Particle Hydrodynamics
- **authors:** J. J. Monaghan
- **venue:** Reports on Progress in Physics 68(8), 2005
- **doi:** [10.1088/0034-4885/68/8/R01](https://doi.org/10.1088/0034-4885/68/8/R01)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this review the theory and application of Smoothed particle hydrodynamics (SPH) since its
> inception in 1977 are discussed. Emphasis is placed on the strengths and weaknesses, the
> analogy with particle dynamics and the numerous areas where SPH has been successfully applied.

### `monaghan2009`

- **file:** `monaghan2009_sph-particle-boundary-forces.pdf`
- **title:** SPH Particle Boundary Forces for Arbitrary Boundaries
- **authors:** J. J. Monaghan and J. B. Kajtar
- **venue:** Computer Physics Communications 180(10), 2009
- **doi:** [10.1016/j.cpc.2009.05.008](https://doi.org/10.1016/j.cpc.2009.05.008)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> This paper is concerned with approximating arbitrarily shaped boundaries in SPH simulations.
> We model the boundaries by means of boundary particles which exert forces on a fluid. We show
> that, when these forces are chosen correctly, and the boundary particle spacing is a factor of
> 2 (or more) less than the fluid particle spacing, the total boundary force on a fluid SPH
> particle is perpendicular to boundaries with negligible error. Furthermore, the variation in
> the force as a fluid particle moves, while keeping a fixed distance from the boundary, is also
> negligible. The method works equally well for convex or concave boundaries. The new boundary
> forces simplify SPH algorithms and are superior to other methods for simulating complicated
> boundaries. We apply the new method to (a) the rise of a cylinder contained in a curved basin,
> (b) the spin down of a fluid in a cylinder, and (c) the oscillation of a cylinder inside a
> larger fixed cylinder. The results of the simulations are in good agreement with those
> obtained using other methods, but with the advantage that they are very simple to implement.

### `morris1997`

- **file:** `morris1997_low-reynolds-number-incompressible-sph.pdf`
- **title:** Modeling Low Reynolds Number Incompressible Flows Using SPH
- **authors:** Joseph P. Morris, Patrick J. Fox and Yi Zhu
- **venue:** Journal of Computational Physics 136(1), 1997
- **doi:** [10.1006/jcph.1997.5776](https://doi.org/10.1006/jcph.1997.5776)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> The method of smoothed particle hydrodynamics (SPH) is extended to model incompressible flows
> of low Reynolds number. For such flows, modification of the standard SPH formalism is required
> to minimize errors associated with the use of a quasi-incompressible equation of state.
> Treatment of viscosity, state equation, kernel interpolation, and boundary conditions are
> described. Simulations using the method show close agreement with series solutions for Couette
> and Poiseuille flows. Furthermore, comparison with finite element solutions for flow past a
> regular lattice of cylinders shows close agreement for the velocity and pressure fields. The
> SPH results exhibit small pressure fluctuations near curved boundaries. Further improvements
> to the boundary conditions may be possible which will reduce these errors. A similar method to
> that used here may permit the simulation of other flows at low Reynolds numbers using SPH.
> Further development will be needed for cases involving free surfaces or substantially
> different equations of

### `morris1997switch`

- **file:** `morris1997switch_a-switch-to-reduce-sph-viscosity.pdf`
- **title:** A Switch to Reduce SPH Viscosity
- **authors:** Joseph P. Morris and Joseph J. Monaghan
- **venue:** Journal of Computational Physics 136(1), 1997
- **doi:** [10.1006/jcph.1997.5690](https://doi.org/10.1006/jcph.1997.5690)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> Smoothed particle hydrodynamics is a Lagrangian particle method for fluid dynamics which
> simulates shocks by using an artificial viscosity. Unlike Eulerian methods it is not
> convenient to reduce the effects of viscosity by means of switches based on spatial gradients.
> In this paper we introduce the idea of time-varying coefficients which fits more naturally
> with a particle formulation. Each particle has a viscosity parameter which evolves according
> to a simple source and decay equation. The source causes the parameter to grow when the
> particle enters a shock and the decay term causes it to decay to a small value beyond the
> shock. Tests on one-dimensional shocks and a two-dimensional shock–bubble interaction confirm
> that the method gives good results.

### `muller2003`

- **file:** `muller2003_particle-based-fluid-interactive-applications.pdf`
- **title:** Particle-Based Fluid Simulation for Interactive Applications
- **authors:** Matthias Müller, David Charypar and Markus Gross
- **venue:** Proceedings of the 2003 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2003
- **doi:** [10.2312/SCA03/154-159](https://doi.org/10.2312/SCA03/154-159)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Realistically animated fluids can add substantial realism to interactive applications such as
> virtual surgery simulators or computer games. In this paper we propose an interactive method
> based on Smoothed Particle Hydrodynamics (SPH) to simulate fluids with free surfaces. The
> method is an extension of the SPH-based technique by Desbrun to animate highly deformable
> bodies. We gear the method towards fluid simulation by deriving the force density fields
> directly from the Navier-Stokes equation and by adding a term to model surface tension
> effects. In contrast to Eulerian grid-based approaches, the particle-based approach makes mass
> conservation equations and convection terms dispensable which reduces the complexity of the
> simulation. In addition, the particles can directly be used to render the surface of the
> fluid. We propose methods to track and visualize the free surface using point splatting and
> marching cubes-based surface reconstruction. Our animation method is fast enough to be used in
> interactive systems and to allow for user interaction with models consisting of up to 5000
> particles.

### `muller2004`

- **file:** `muller2004_interaction-of-fluids-with-deformable-solids.pdf`
- **title:** Interaction of Fluids with Deformable Solids
- **authors:** Matthias Müller, Simon Schirm, Matthias Teschner, Bruno Heidelberger and Markus Gross
- **venue:** Computer Animation and Virtual Worlds 15(3-4), 2004
- **doi:** [10.1002/cav.18](https://doi.org/10.1002/cav.18)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this paper, we present a method for simulating the interaction of fluids with deformable
> solids. The method is designed for the use in interactive systems such as virtual surgery
> simulators where the real‐time interplay of liquids and surrounding tissue is important. In
> computer graphics, a variety of techniques have been proposed to model liquids and deformable
> objects at interactive rates. As important as the plausible animation of these substances is
> the fast and stable modeling of their interaction. The method we describe in this paper models
> the exchange of momentum between Lagrangian particle‐based fluid models and solids represented
> by polygonal meshes. To model the solid‐fluid interaction we use virtual boundary particles.
> They are placed on the surface of the solid objects according to Gaussian quadrature rules
> allowing the computation of smooth interaction potentials that yield stable simulations. We
> demonstrate our approach in an interactive simulation environment for fluids and deformable
> solids. Copyright © 2004 John Wiley & Sons, Ltd.

### `muller2005`

- **file:** `muller2005_particle-based-fluid-fluid-interaction.pdf`
- **title:** Particle-Based Fluid-Fluid Interaction
- **authors:** Matthias Müller, Barbara Solenthaler, Richard Keiser and Markus Gross
- **venue:** Proceedings of the 2005 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2005
- **doi:** [10.1145/1073368.1073402](https://doi.org/10.1145/1073368.1073402)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> The interesting and complex behavior of fluids emerges mainly from interaction processes.
> While interactions of fluids with static or dynamic solids has caught some attention in
> computer graphics lately, the mutual interaction of different types of fluids such as air and
> water or water and wax has received much less attention although these types of interaction
> are the basis for a variety of important phenomena.In this paper we propose a new technique to
> model fluid-fluid interaction based on the Smoothed Particle Hydrodynamics (SPH) method. For
> the simulation of air-water interaction, air particles are generated on the fly only where
> needed. We also model dynamic phase changes and interface forces. Our technique makes possible
> the simulation of phenomena such as boiling water, trapped air and the dynamics of a lava
> lamp.

### `orthmann2012`

- **file:** `orthmann2012_temporal-blending-for-adaptive-sph.pdf`
- **title:** Temporal Blending for Adaptive SPH
- **authors:** Jens Orthmann and Andreas Kolb
- **venue:** Computer Graphics Forum 31(8), 2012
- **doi:** [10.1111/j.1467-8659.2012.03186.x](https://doi.org/10.1111/j.1467-8659.2012.03186.x)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this paper, we introduce a fast and consistent smoothed particle hydrodynamics (SPH)
> technique which is suitable for convection–diffusion simulations of incompressible fluids. We
> apply our temporal blending technique to reduce the number of particles in the simulation
> while smoothly changing quantity fields. Our approach greatly reduces the error introduced in
> the pressure term when changing particle configurations. Compared to other methods, this
> enables larger integration time‐steps in the transition phase. Our implementation is fully
> GPU‐based to take advantage of the parallel nature of particle simulations.

### `peer2015`

- **file:** `peer2015_implicit-viscosity-formulation-for-sph.pdf`
- **title:** An Implicit Viscosity Formulation for SPH Fluids
- **authors:** Andreas Peer, Markus Ihmsen, Jens Cornelis and Matthias Teschner
- **venue:** ACM Transactions on Graphics 34(4), 2015
- **doi:** [10.1145/2766925](https://doi.org/10.1145/2766925)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> We present a novel implicit formulation for highly viscous fluids simulated with Smoothed
> Particle Hydrodynamics SPH. Compared to explicit methods, our formulation is significantly
> more efficient and handles a larger range of viscosities. Differing from existing implicit
> formulations, our approach reconstructs the velocity field from a target velocity gradient.
> This gradient encodes a desired shear-rate damping and preserves the velocity divergence that
> is introduced by the SPH pressure solver to counteract density deviations. The target gradient
> ensures that pressure and viscosity computation do not interfere. Therefore, only one pressure
> projection step is required, which is in contrast to state-of-the-art implicit Eulerian
> formulations. While our model differs from true viscosity in that vorticity diffusion is not
> encoded in the target gradient, it nevertheless captures many of the qualitative behaviors of
> viscous liquids. Our formulation can easily be incorporated into complex scenarios with one-
> and two-way coupled solids and multiple fluid phases with different densities and viscosities.

### `peer2016`

- **file:** `peer2016_prescribed-velocity-gradients-viscous-sph.pdf`
- **title:** Prescribed Velocity Gradients for Highly Viscous SPH Fluids with Vorticity Diffusion
- **authors:** Andreas Peer and Matthias Teschner
- **venue:** IEEE Transactions on Visualization and Computer Graphics 23(12), 2017
- **doi:** [10.1109/TVCG.2016.2636144](https://doi.org/10.1109/TVCG.2016.2636144)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Working with prescribed velocity gradients is a promising approach to efficiently and robustly
> simulate highly viscous SPH fluids. Such approaches allow to explicitly and independently
> process shear rate, spin, and expansion rate. This can be used to, e.g., avoid interferences
> between pressure and viscosity solvers. Another interesting aspect is the possibility to
> explicitly process the vorticity, e.g., to preserve the vorticity. In this context, this paper
> proposes a novel variant of the prescribed-gradient idea that handles vorticity in a
> physically motivated way. In contrast to a less appropriate vorticity preservation that has
> been used in a previous approach, vorticity is diffused. The paper illustrates the utility of
> the vorticity diffusion. Therefore, comparisons of the proposed vorticity diffusion with
> vorticity preservation and additionally with vorticity damping are presented. The paper
> further discusses the relation between prescribed velocity gradients and prescribed velocity
> Laplacians which improves the intuition behind the prescribed-gradient method for highly
> viscous SPH fluids. Finally, the paper discusses the relation of the proposed method to a
> physically correct implicit viscosity formulation.

### `peer2017`

- **file:** `peer2017_implicit-sph-linearly-elastic-solids.pdf`
- **title:** An Implicit SPH Formulation for Incompressible Linearly Elastic Solids
- **authors:** Andreas Peer, Christoph Gissler, Stefan Band and Matthias Teschner
- **venue:** Computer Graphics Forum 37(6), 2018
- **doi:** [10.1111/cgf.13317](https://doi.org/10.1111/cgf.13317)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We propose a novel smoothed particle hydrodynamics (SPH) formulation for deformable solids.
> Key aspects of our method are implicit elastic forces and an adapted SPH formulation for the
> deformation gradient that—in contrast to previous work—allows a rotation extraction directly
> from the SPH deformation gradient. The proposed implicit concept is entirely based on linear
> formulations. As a linear strain tensor is used, a rotation‐aware computation of the
> deformation gradient is required. In contrast to existing work, the respective rotation
> estimation is entirely realized within the SPH concept using a novel formulation with
> incorporated kernel gradient correction for first‐order consistency. The proposed implicit
> formulation and the adapted rotation estimation allow for significantly larger time steps and
> higher stiffness compared to explicit forms. Performance gain factors of up to one hundred are
> presented. Incompressibility of deformable solids is accounted for with an ISPH pressure
> solver. This further allows for a pressure‐based boundary handling and a unified processing of
> deformables interacting with SPH fluids and rigids. Self‐collisions are implicitly handled by
> the pressure solver.

### `price2010`

- **file:** `price2010_spmhd-vector-potential.pdf`
- **title:** Smoothed Particle Magnetohydrodynamics -- IV. Using the Vector Potential
- **authors:** Daniel J. Price
- **venue:** Monthly Notices of the Royal Astronomical Society 401(3), 2010
- **doi:** [10.1111/j.1365-2966.2009.15763.x](https://doi.org/10.1111/j.1365-2966.2009.15763.x)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this paper, we investigate the use of the vector potential as a means of maintaining the
> divergence constraint in the numerical solution to the equations of magnetohydrodynamics (MHD)
> using the Smoothed Particle Hydrodynamics (SPH) method. We derive a self-consistent
> formulation of the equations of motion using a variational principle that is constrained by
> the numerical formulation of both the induction equation and the curl operator used to obtain
> the magnetic field, which guarantees exact and simultaneous conservation of momentum, energy
> and entropy in the numerical scheme. This leads to a novel formulation of the MHD force term,
> unique to the vector potential, which differs from previous formulations. We also demonstrate
> how dissipative terms can be correctly formulated for the vector potential such that the
> contribution to the entropy is positive definite and the total energy is conserved. On a
> standard suite of numerical tests in one, two and three dimensions, we first find that the
> consistent formulation of the vector potential equations is unstable to the well-known SPH
> tensile instability, even more so than in the standard Smoothed Particle Magnetohydrodynamics
> (SPMHD) formulation where the magnetic field is evolved directly. Furthermore, we find that
> whilst a hybrid approach based on the vector potential evolution equation coupled with a
> standard force term gives good results for one- and two-dimensional problems (where dAz/dt=
> 0), such an approach suffers from numerical instability in three dimensions related to the
> unconstrained evolution of vector potential components. We conclude that use of the vector
> potential is not a viable approach for SPMHD.

### `price2012`

- **file:** `price2012_sph-and-magnetohydrodynamics.pdf`
- **title:** Smoothed Particle Hydrodynamics and Magnetohydrodynamics
- **authors:** Daniel J. Price
- **venue:** Journal of Computational Physics 231(3), 2012
- **doi:** [10.1016/j.jcp.2010.12.011](https://doi.org/10.1016/j.jcp.2010.12.011)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> This paper presents an overview and introduction to Smoothed Particle Hydrodynamics and
> Magnetohydrodynamics in theory and in practice. Firstly, we give a basic grounding in the
> fundamentals of SPH, showing how the equations of motion and energy can be self-consistently
> derived from the density estimate. We then show how to interpret these equations using the
> basic SPH interpolation formulae and highlight the subtle difference in approach between SPH
> and other particle methods. In doing so, we also critique several ‘urban myths’ regarding SPH,
> in particular the idea that one can simply increase the ‘neighbour number’ more slowly than
> the total number of particles in order to obtain convergence. We also discuss the origin of
> numerical instabilities such as the pairing and tensile instabilities. Finally, we give
> practical advice on how to resolve three of the main issues with SPMHD: removing the tensile
> instability, formulating dissipative terms for MHD shocks and enforcing the divergence
> constraint on the particles, and we give the current status of developments in this area.
> Accompanying the paper is the first public release of the ndspmhd SPH code, a 1, 2 and 3
> dimensional code designed as a testbed for SPH/SPMHD algorithms that can be used to test many
> of the ideas and used to run all of the numerical examples contained in the paper.

### `raveendran2011`

- **file:** `raveendran2011_hybrid-smoothed-particle-hydrodynamics.pdf`
- **title:** Hybrid Smoothed Particle Hydrodynamics
- **authors:** Karthik Raveendran, Chris Wojtan and Greg Turk
- **venue:** Proceedings of the 2011 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2011
- **doi:** [10.1145/2019406.2019411](https://doi.org/10.1145/2019406.2019411)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a new algorithm for enforcing incompressibility for Smoothed Particle Hydrodynamics
> (SPH) by preserving uniform density across the domain. We propose a hybrid method that uses a
> Poisson solve on a coarse grid to enforce a divergence free velocity field, followed by a
> local density correction of the particles. This avoids typical grid artifacts and maintains
> the Lagrangian nature of SPH by directly transferring pressures onto particles. Our method can
> be easily integrated with existing SPH techniques such as the incompressible PCISPH method as
> well as weakly compressible SPH by adding an additional force term. We show that this hybrid
> method accelerates convergence towards uniform density and permits a significantly larger time
> step compared to earlier approaches while producing similar results. We demonstrate our
> approach in a variety of scenarios with significant pressure gradients such as splashing
> liquids.

### `shao2003`

- **file:** `shao2003_incompressible-sph-free-surface.pdf`
- **title:** Incompressible SPH Method for Simulating Newtonian and Non-Newtonian Flows with a Free Surface
- **authors:** Songdong Shao and Edmond Y. M. Lo
- **venue:** Advances in Water Resources 26(7), 2003
- **doi:** [10.1016/S0309-1708(03)00030-7](https://doi.org/10.1016/S0309-1708(03)00030-7)
- **abstract from:** PDF text layer, p.1 (ligatures expanded)

> An incompressible smoothed particle hydrodynamics (SPH) method is presented to simulate
> Newtonian and non-Newtonian flows with free surfaces. The basic equations solved are the
> incompressible mass conservation and Navier–Stokes equations. The method uses
> prediction–correction fractional steps with the temporal velocity field integrated forward in
> time without enforcing incompressibility in the prediction step. The resulting deviation of
> particle density is then implicitly projected onto a divergence-free space to satisfy
> incompressibility through a pressure Poisson equation derived from an approximate pressure
> projection. Various SPH formulations are employed in the discretization of the relevant
> gradient, divergence and Laplacian terms. Free surfaces are identified by the particles whose
> density is below a set point. Wall boundaries are represented by particles whose positions are
> fixed. The SPH formulation is also extended to non-Newtonian flows and demonstrated using the
> Cross rheological model. The incompressible SPH method is tested by typical 2-D dam-break
> problems in which both water and fluid mud are considered. The computations are in good
> agreement with available experimental data. The different flow features between Newtonian and
> nonNewtonian flows after the dam-break are discussed.

### `sin2009`

- **file:** `sin2009_point-based-method-incompressible-flow.pdf`
- **title:** A Point-Based Method for Animating Incompressible Flow
- **authors:** Funshing Sin, Adam W. Bargteil and Jessica K. Hodgins
- **venue:** Proceedings of the 2009 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2009
- **doi:** [10.1145/1599470.1599502](https://doi.org/10.1145/1599470.1599502)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this paper, we present a point-based method for animating incompressible flow. The
> advection term is handled by moving the sample points through the flow in a Lagrangian
> fashion. However, unlike most previous approaches, the pressure term is handled by performing
> a projection onto a divergence-free field. To perform the pressure projection, we compute a
> Voronoi diagram with the sample points as input. Borrowing from Finite Volume Methods, we then
> invoke the divergence theorem and ensure that each Voronoi cell is divergence free. To handle
> complex boundary conditions, Voronoi cells are clipped against obstacle boundaries and free
> surfaces. The method is stable, flexible and combines many of the desirable features of point-
> based and grid-based methods. We demonstrate our approach on several examples of splashing and
> streaming liquid and swirling smoke.

### `solenthaler2007`

- **file:** `solenthaler2007_unified-particle-model-fluid-solid.pdf`
- **title:** A Unified Particle Model for Fluid-Solid Interactions
- **authors:** Barbara Solenthaler, Jürg Schläfli and Renato Pajarola
- **venue:** Computer Animation and Virtual Worlds 18(1), 2007
- **doi:** [10.1002/cav.162](https://doi.org/10.1002/cav.162)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a new method for the simulation of melting and solidification in a unified particle
> model. Our technique uses the Smoothed Particle Hydrodynamics (SPH) method for the simulation
> of liquids, deformable as well as rigid objects, which eliminates the need to define an
> interface for coupling different models. Using this approach, it is possible to simulate
> fluids and solids by only changing the attribute values of the underlying particles. We
> significantly changed a prior elastic particle model to achieve a flexible model for melting
> and solidification. By using an SPH approach and considering a new definition of a local
> reference shape, the simulation of merging and splitting of different objects, as may be
> caused by phase change processes, is made possible. In order to keep the system stable even in
> regions represented by a sparse set of particles we use a special kernel function for
> solidification processes. Additionally, we propose a surface reconstruction technique based on
> considering the movement of the center of mass to reduce rendering errors in concave regions.
> The results demonstrate new interaction effects concerning the melting and solidification of
> material, even while being surrounded by liquids. Copyright © 2007 John Wiley & Sons, Ltd.

### `solenthaler2008`

- **file:** `solenthaler2008_density-contrast-sph-interfaces.pdf`
- **title:** Density Contrast SPH Interfaces
- **authors:** Barbara Solenthaler and Renato Pajarola
- **venue:** Proceedings of the 2008 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA), 2008
- **doi:** [10.2312/SCA/SCA08/211-218](https://doi.org/10.2312/SCA/SCA08/211-218)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> To simulate multiple fluids realistically many important interaction effects have to be
> captured accurately. Smoothed Particle Hydrodynamics (SPH) has shown to be a simple, yet
> flexible method to cope with many fluid simulation problems in a robust way. Unfortunately,
> the results obtained when using SPH to simulate miscible fluids are severely affected,
> especially if density ratios become large. The undesirable effects reach from unphysical
> density and pressure variations to spurious and unnatural interface tensions, as well as
> severe numerical instabilities. In this work, we present a formulation based on SPH which can
> handle density discontinuities at interfaces between multiple fluids correctly without
> increasing the computational costs compared to standard SPH. The basic idea is to replace the
> density computation in SPH by a measure of particle densities and consequently derive new
> formulations for pressure and viscous forces. The new method enables the user to select the
> desired amount of interface tension according to the simulation problem at hand. We succeed to
> stably simulate multiple fluids with high density contrasts without the above described
> artifacts apparent in standard SPH simulations.

### `solenthaler2009`

- **file:** `solenthaler2009_predictive-corrective-incompressible-sph.pdf`
- **title:** Predictive-Corrective Incompressible SPH
- **authors:** Barbara Solenthaler and Renato Pajarola
- **venue:** ACM Transactions on Graphics 28(3), 2009
- **doi:** [10.1145/1531326.1531346](https://doi.org/10.1145/1531326.1531346)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> We present a novel, incompressible fluid simulation method based on the Lagrangian Smoothed
> Particle Hydrodynamics (SPH) model. In our method, incompressibility is enforced by using a
> prediction-correction scheme to determine the particle pressures. For this, the information
> about density fluctuations is actively propagated through the fluid and pressure values are
> updated until the targeted density is satisfied. With this approach, we avoid the
> computational expenses of solving a pressure Poisson equation, while still being able to use
> large time steps in the simulation. The achieved results show that our predictive-corrective
> incompressible SPH (PCISPH) method clearly outperforms the commonly used weakly compressible
> SPH (WCSPH) model by more than an order of magnitude while the computations are in good
> agreement with the WCSPH results.

### `solenthaler2011`

- **file:** `solenthaler2011_two-scale-particle-simulation.pdf`
- **title:** Two-Scale Particle Simulation
- **authors:** Barbara Solenthaler and Markus Gross
- **venue:** ACM Transactions on Graphics 30(4), 2011
- **doi:** [10.1145/1964921.1964976](https://doi.org/10.1145/1964921.1964976)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We propose a two-scale method for particle-based fluids that allocates computing resources to
> regions of the fluid where complex flow behavior emerges. Our method uses a low- and a high-
> resolution simulation that run at the same time. While in the coarse simulation the whole
> fluid is represented by large particles, the fine level simulates only a subset of the fluid
> with small particles. The subset can be arbitrarily defined and also dynamically change over
> time to capture complex flows and small-scale surface details. The low- and high-resolution
> simulations are coupled by including feedback forces and defining appropriate boundary
> conditions. Our method offers the benefit that particles are of the same size within each
> simulation level. This avoids particle splitting and merging processes, and allows the
> simulation of very large resolution differences without any stability problems. The model is
> easy to implement, and we show how it can be integrated into a standard SPH simulation as well
> as into the incompressible PCISPH solver. Compared to the single-resolution simulation, our
> method produces similar surface details while improving the efficiency linearly to the
> achieved reduction rate of the particle number.

### `stam1995`

- **file:** `stam1995_depicting-fire-gaseous-phenomena-diffusion.pdf`
- **title:** Depicting Fire and Other Gaseous Phenomena Using Diffusion Processes
- **authors:** Jos Stam and Eugene Fiume
- **venue:** Proceedings of the 22nd Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 1995
- **doi:** [10.1145/218380.218430](https://doi.org/10.1145/218380.218430)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> Developing a visually convincing model of fire, smoke, and other gaseousphenomena is among the
> most difficult and attractive problems in computer graphics. We have created new methods of
> animating a wide range of gaseous phenomena, including the particularly subtle problem of
> modelling "wispy" smoke and steam, using far fewer primitives than before. One significant
> innovation is the reformulation and solution of the advection-diffusion equation for densities
> composed of "warped blobs". These blobs more accurately model the distortions that gases
> undergo when advected by wind fields. We also introduce a simple model for the flame of a fire
> and its spread. Lastly, we present an efficient formulation and implementation of global
> illumination in the presence of gases and fire. Our models are specifically designed to permit
> a significant degree of user control over the evolution of gaseous phenomena. Keywords: fire,
> smoke, gaseous phenomena, diffusion, advection, warped blobbies, light tra...

### `takahashi2015`

- **file:** `takahashi2015_implicit-formulation-sph-viscous-fluids.pdf`
- **title:** Implicit Formulation for SPH-Based Viscous Fluids
- **authors:** Tetsuya Takahashi, Yoshinori Dobashi, Issei Fujishiro, Tomoyuki Nishita and Ming C. Lin
- **venue:** Computer Graphics Forum 34(2), 2015
- **doi:** [10.1111/cgf.12578](https://doi.org/10.1111/cgf.12578)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We propose a stable and efficient particle‐based method for simulating highly viscous fluids
> that can generate coiling and buckling phenomena and handle variable viscosity. In contrast to
> previous methods that use explicit integration, our method uses an implicit formulation to
> improve the robustness of viscosity integration, therefore enabling use of larger time steps
> and higher viscosities. We use Smoothed Particle Hydrodynamics to solve the full form of
> viscosity, constructing a sparse linear system with a symmetric positive definite matrix,
> while exploiting the variational principle that automatically enforces the boundary condition
> on free surfaces. We also propose a new method for extracting coefficients of the matrix
> contributed by second‐ring neighbor particles to efficiently solve the linear system using a
> conjugate gradient solver. Several examples demonstrate the robustness and efficiency of our
> implicit formulation over previous methods and illustrate the versatility of our method.

### `takahashi2018`

- **file:** `takahashi2018_efficient-hybrid-incompressible-sph-solver.pdf`
- **title:** An Efficient Hybrid Incompressible SPH Solver with Interface Handling for Boundary Conditions
- **authors:** Tetsuya Takahashi, Yoshinori Dobashi, Tomoyuki Nishita and Ming C. Lin
- **venue:** Computer Graphics Forum 37(1), 2018
- **doi:** [10.1111/cgf.13292](https://doi.org/10.1111/cgf.13292)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We propose a hybrid smoothed particle hydrodynamics solver for efficientlysimulating
> incompressible fluids using an interface handling method for boundary conditions in the
> pressure Poisson equation. We blend particle density computed with one smooth and one spiky
> kernel to improve the robustness against both fluid–fluid and fluid–solid collisions. To
> further improve the robustness and efficiency, we present a new interface handling method
> consisting of two components: free surface handling for Dirichlet boundary conditions and
> solid boundary handling for Neumann boundary conditions. Our free surface handling
> appropriately determines particles for Dirichlet boundary conditions using Jacobi‐based
> pressure prediction while our solid boundary handling introduces a new term to ensure the
> solvability of the linear system. We demonstrate that our method outperforms the
> state‐of‐the‐art particle‐based fluid solvers.

### `weiler2016`

- **file:** `weiler2016_projective-fluids.pdf`
- **title:** Projective Fluids
- **authors:** Marcel Weiler, Dan Koschier and Jan Bender
- **venue:** Proceedings of the 9th International Conference on Motion in Games (MIG), 2016
- **doi:** [10.1145/2994258.2994282](https://doi.org/10.1145/2994258.2994282)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> We present a new method for particle based fluid simulation, using a combination of Projective
> Dynamics and Smoothed Particle Hydrodynamics (SPH). The Projective Dynamics framework allows
> the fast simulation of a wide range of constraints. It offers great stability through its
> implicit time integration scheme and is parallelizable in large parts, so that it can make use
> of modern multi core CPUs. Yet existing work only uses Projective Dynamics to simulate various
> kinds of soft bodies and cloth. We are the first ones to incorporate fluid simulation into the
> Projective Dynamics framework. Our proposed fluid constraints are derived from SPH and
> seamlessly integrate into the existing method. Furthermore, we adapt the solver to handle the
> constantly changing constraints that appear in fluid simulation. We employ a highly parallel
> matrix-free conjugate gradient solver, and thus do not require expensive matrix
> factorizations.

### `yang2012`

- **file:** `yang2012_realtime-two-way-coupling-meshless-fem.pdf`
- **title:** Realtime Two-Way Coupling of Meshless Fluids and Nonlinear FEM
- **authors:** Lipeng Yang, Shuai Li, Aimin Hao and Hong Qin
- **venue:** Computer Graphics Forum 31(7), 2012
- **doi:** [10.1111/j.1467-8659.2012.03196.x](https://doi.org/10.1111/j.1467-8659.2012.03196.x)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this paper, we present a novel method to couple Smoothed Particle Hydrodynamics (SPH) and
> nonlinear FEM to animate the interaction of fluids and deformable solids in real time. To
> accurately model the coupling, we generate proxy particles over the boundary of deformable
> solids to facilitate the interaction with fluid particles, and develop an efficient method to
> distribute the coupling forces of proxy particles to FEM nodal points. Specifically, we employ
> the Total Lagrangian Explicit Dynamics (TLED) finite element algorithm for nonlinear FEM
> because of many of its attractive properties such as supporting massive parallelism, avoiding
> dynamic update of stiffness matrix computation, and efficient solver. Based on a
> predictor‐corrector scheme for both velocity and position, different normal and tangential
> conditions can be realized even for shell‐like thin solids. Our coupling method is entirely
> implemented on modern GPUs using CUDA. We demonstrate the advantage of our two‐way coupling
> method in computer animation via various virtual scenarios.

### `yildiz2009`

- **file:** `yildiz2009_multiple-boundary-tangent-method.pdf`
- **title:** SPH with the Multiple Boundary Tangent Method
- **authors:** M. Yildiz, R. A. Rook and A. Suleman
- **venue:** International Journal for Numerical Methods in Engineering 77(10), 2009
- **doi:** [10.1002/nme.2458](https://doi.org/10.1002/nme.2458)
- **abstract from:** OpenAlex `abstract_inverted_index` (word order preserved; original punctuation not recoverable)

> In this article, we present an improved solid boundary treatment formulation for the smoothed
> particle hydrodynamics (SPH) method. Benchmark simulations using previously reported boundary
> treatments can suffer from particle penetration and may produce results that numerically blow
> up near solid boundaries. As well, current SPH boundary approaches do not properly treat
> curved boundaries in complicated flow domains. These drawbacks have been remedied in a new
> boundary treatment method presented in this article, called the multiple boundary tangent
> (MBT) approach. In this article we present two important benchmark problems to validate the
> developed algorithm and show that the multiple boundary tangent treatment produces results
> that agree with known numerical and experimental solutions. The two benchmark problems chosen
> are the lid‐driven cavity problem, and flow over a cylinder. The SPH solutions using the MBT
> approach and the results from literature are in very good agreement. These solutions involved
> solid boundaries, but the approach presented herein should be extendable to time‐evolving,
> free‐surface boundaries. Copyright © 2008 John Wiley & Sons, Ltd.

### `zhu2005`

- **file:** `zhu2005_animating-sand-as-a-fluid.pdf`
- **title:** Animating Sand as a Fluid
- **authors:** Yongning Zhu and Robert Bridson
- **venue:** ACM Transactions on Graphics 24(3), 2005
- **doi:** [10.1145/1073204.1073298](https://doi.org/10.1145/1073204.1073298)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> We present a physics-based simulation method for animating sand. To allow for efficiently
> scaling up to large volumes of sand, we abstract away the individual grains and think of the
> sand as a continuum. In particular we show that an existing water simulator can be turned into
> a sand simulator with only a few small additions to account for inter-grain and boundary
> friction.We also propose an alternative method for simulating fluids. Our core representation
> is a cloud of particles, which allows for accurate and flexible surface tracking and
> advection, but we use an auxiliary grid to efficiently enforce boundary conditions and
> incompressibility. We further address the issue of reconstructing a surface from particle data
> to render each frame.

### `zorilla2020`

- **file:** `zorilla2020_surface-tension-particle-classification-monte-carlo.pdf`
- **title:** Accelerating Surface Tension Calculation in SPH via Particle Classification and Monte Carlo Integration
- **authors:** Fernando Zorilla, Marcel Ritter, Johannes Sappl, Wolfgang Rauch and Matthias Harders
- **venue:** Computers 9(2), 2020
- **doi:** [10.3390/computers9020023](https://doi.org/10.3390/computers9020023)
- **abstract from:** Crossref (publisher-deposited JATS abstract)

> Surface tension has a strong influence on the shape of fluid interfaces. We propose a method
> to calculate the corresponding forces efficiently. In contrast to several previous approaches,
> we discriminate to this end between surface and non-surface SPH particles. Our method
> effectively smooths the fluid interface, minimizing its curvature. We make use of an approach
> inspired by Monte Carlo integration to estimate local normals as well as curvatures, based on
> which the force can be calculated. We compare different sampling schemes for the Monte Carlo
> approach, for which a Halton sequence performed best. Our overall technique is applicable, but
> not limited to 2D and 3D simulations, and can be coupled with any common SPH formulation. It
> outperforms prior approaches with regard to total computation time per time step in dynamic
> scenes. Additionally, it is adjustable for higher quality in small scale scenes with dominant
> surface tension effects.
