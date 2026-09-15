# Papers to acquire

Fetch list assembled 2026-09-15 to support `PESPH_PLAN.md` and
`../warpSPHIntegrators/SPLITTING_PLAN.md`. Everything already in `literature/` has been
excluded — notably Monaghan 2002/2005, Morris & Monaghan 1997, Price 2012,
Dehnen & Aly 2012, and the Weiler/Peer/Takahashi implicit-viscosity set are all present.

Suggested filenames follow this folder's `<bibkey>_<slug>.pdf` convention; drop the PDFs
in and run the usual sync (`ADDING.md`).

**On the DOIs:** full citations (author / year / journal / volume / page) are given for
everything, because those resolve through any institutional search even if a DOI has a
typo. DOIs are from memory except where noted as verified, so treat the citation as
authoritative and the DOI as a convenience.

---

## Tier 1 — blocking for `PESPH_PLAN.md`

Cannot implement the scheme correctly without these.

| Paper | Citation | DOI | Filename |
|---|---|---|---|
| **Hopkins 2013** — *A general class of Lagrangian SPH methods and implications for fluid mixing problems* | MNRAS 428, 2840 (2013) | `10.1093/mnras/sts210` | `hopkins2013_general-class-of-lagrangian-sph.pdf` |
| **Saitoh & Makino 2013** — *A density-independent formulation of SPH* | ApJ 768, 44 (2013) | `10.1088/0004-637X/768/1/44` | `saitoh2013_density-independent-sph.pdf` |
| **Springel & Hernquist 2002** — *Cosmological SPH simulations: the entropy equation* | MNRAS 333, 649 (2002) | `10.1046/j.1365-8711.2002.05445.x` | `springel2002_sph-entropy-equation.pdf` |

**Hopkins 2013 is the single most important one.** It is the actual pressure-entropy
paper — the variant `PESPH_PLAN.md` §1.2 and §3.2 build on. Both papers currently in
`../warpSPHIntegrators/literature/` (Frontiere Appendix G, Hopkins 2015 Appendix F2)
describe only the pressure-*energy* form, so the entropy formulation is currently
undocumented anywhere in the repo.

Saitoh & Makino is DISPH, which Hopkins builds on and Frontiere cites as [60].
Springel & Hernquist is both the entropy formulation *and* the canonical grad-h
derivation — directly relevant to Phase 2, the risk item.

## Tier 2 — compSPH / CRKSPH provenance

Needed to review or modify the existing schemes, and for `PESPH_PLAN.md` §4.

| Paper | Citation | DOI | Filename |
|---|---|---|---|
| **Owen 2014** — *A compatibly differenced total energy conserving form of SPH* | Int. J. Numer. Meth. Fluids 75, 749–774 (2014) | `10.1002/fld.3912` ✔ verified | `owen2014_compatibly-differenced-total-energy-sph.pdf` |
| **Cullen & Dehnen 2010** — *Inviscid smoothed particle hydrodynamics* | MNRAS 408, 669 (2010) | `10.1111/j.1365-2966.2010.17158.x` | `cullen2010_inviscid-sph.pdf` |
| **Balsara 1995** — *von Neumann stability analysis of SPH — suggestions for optimal algorithms* | J. Comput. Phys. 121, 357 (1995) | `10.1006/jcph.1995.1140` | `balsara1995_von-neumann-stability-sph.pdf` |
| **Dilts 1999** — *Moving-least-squares-particle hydrodynamics I: consistency and stability* | Int. J. Numer. Meth. Engng 44, 1115–1155 (1999) | — | `dilts1999_mlsph-consistency-and-stability.pdf` |

Owen 2014 is Frontiere's [48] — the origin of the compatible energy discretization that
`modules/compSPH/balance.py` and `multistep.py` implement, and the subject of
`PESPH_PLAN.md` §4.2. Also on arXiv as 0906.3029 if the Wiley route is awkward.

Dilts is Frontiere's [14], the MLSPH formalism CRKSPH's conservative differencing is
taken from — and, per `PESPH_PLAN.md` §7.2, the same derivation route MFM uses. Worth
having for both reasons.

## Tier 3 — modern SPH gradients (`PESPH_PLAN.md` §7.5)

The "do this regardless of which bet you take" upgrade.

| Paper | Citation | DOI | Filename |
|---|---|---|---|
| **García-Senz, Cabezón & Escartín 2012** — *Improving SPH with an integral approach to calculating gradients* | A&A 538, A9 (2012) | `10.1051/0004-6361/201117939` | `garciasenz2012_integral-approach-gradients.pdf` |
| **Rosswog 2015** — *Boosting the accuracy of SPH techniques: Newtonian and special-relativistic tests* | MNRAS 448, 3628 (2015) | `10.1093/mnras/stv225` | `rosswog2015_boosting-accuracy-of-sph.pdf` |
| **Rosswog 2020** — *The Lagrangian hydrodynamics code MAGMA2* | MNRAS 498, 4230 (2020) | `10.1093/mnras/staa2591` ✔ verified | `rosswog2020_magma2.pdf` |
| **Cabezón, García-Senz & Figueira 2017** — *SPHYNX: an accurate density-based SPH method* | A&A 606, A78 (2017) | `10.1051/0004-6361/201630208` | `cabezon2017_sphynx.pdf` |
| **Rosswog 2026** — *SPH methods in the modelling of compact objects* (updated review) | arXiv:2607.14828 | — | `rosswog2026_sph-compact-objects-review.pdf` |

García-Senz 2012 is the integral-approximation gradient; Rosswog 2015 is the
matrix-inversion form; MAGMA2 is the code that combines them with slope-limited
reconstruction. These three are the concrete content of §7.5. The 2026 review is the
place to sanity-check §7.1's framing of where the field stands — it is the most recent
thing on the list and the one most worth reading first.

## Tier 4 — MFM (`PESPH_PLAN.md` §7)

Only needed if the MFM direction is taken, but Lanson & Vila and Groth are the two that
would actually shorten the work.

| Paper | Citation | DOI | Filename |
|---|---|---|---|
| **Lanson & Vila 2008** — *Renormalized meshfree schemes I: consistency, stability and hybrid methods for conservation laws* | SIAM J. Numer. Anal. 46, 1912–1934 (2008) | `10.1137/S0036142903427718` | `lanson2008_renormalized-meshfree-schemes.pdf` |
| **Groth et al. 2023** — *The cosmological simulation code OpenGadget3 — implementation of meshless finite mass* | MNRAS 526, 616 (2023) | `10.1093/mnras/stad2717` ✔ vol/page verified | `groth2023_opengadget3-meshless-finite-mass.pdf` |
| **Gaburov & Nitadori 2011** — *Astrophysical weighted particle magnetohydrodynamics* | MNRAS 414, 129 (2011) | `10.1111/j.1365-2966.2011.18313.x` | `gaburov2011_weighted-particle-mhd.pdf` |
| **Toro** — *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. | Springer (2009), ISBN 978-3-540-25202-3 | — | (book — HLLC reference) |

Lanson & Vila is the mathematical foundation Hopkins builds MFM on, and it is the hard
one to find outside an institutional subscription. Groth et al. is a worked port of MFM
*into an existing SPH code*, i.e. the closest thing to a template for §7 — probably the
highest practical value in this tier.

## Tier 5 — geometric integration (`../warpSPHIntegrators/SPLITTING_PLAN.md`)

Different library, same project. The book is worth more than the rest combined.

| Work | Citation | DOI / ISBN | Filename |
|---|---|---|---|
| **Hairer, Lubich & Wanner** — *Geometric Numerical Integration*, 2nd ed. | Springer Series in Comput. Math. 31 (2006) | ISBN 978-3-540-30663-4 | (book) |
| **Yoshida 1990** — *Construction of higher order symplectic integrators* | Phys. Lett. A 150, 262–268 (1990) | `10.1016/0375-9601(90)90092-3` | `yoshida1990_higher-order-symplectic-integrators.pdf` |
| **Suzuki 1990** — *Fractal decomposition of exponential operators…* | Phys. Lett. A 146, 319–323 (1990) | `10.1016/0375-9601(90)90962-N` | `suzuki1990_fractal-decomposition-exponential-operators.pdf` |
| **Goldman & Kaper 1996** — *Nth-order operator splitting schemes and nonreversible systems* | SIAM J. Numer. Anal. 33, 349–367 (1996) | `10.1137/0733018` | `goldman1996_nth-order-operator-splitting.pdf` |
| **McLachlan & Perlmutter 2001** — *Conformal Hamiltonian systems* | J. Geom. Phys. 39, 276–300 (2001) | `10.1016/S0393-0440(01)00020-1` | `mclachlan2001_conformal-hamiltonian-systems.pdf` |
| **Bhatt, Floyd & Moore 2016** — *Second order conformal symplectic schemes for damped Hamiltonian systems* | J. Sci. Comput. 66, 1234–1259 (2016) | `10.1007/s10915-015-0062-z` | `bhatt2016_conformal-symplectic-damped-hamiltonian.pdf` |
| **Blanes, Casas & Murua 2008** — *Splitting and composition methods in the numerical integration of differential equations* | Bol. Soc. Esp. Mat. Apl. 45, 89–145 (2008); arXiv:0812.0377 | — | `blanes2008_splitting-and-composition-methods.pdf` |

Chapters II.4 (composition and the order theorem), II.5 (splitting, Strang, BCH), III.4
(backward error analysis) and V.4.1 (symmetric projection) of Hairer-Lubich-Wanner are
cited throughout `SPLITTING_PLAN.md` and carry essentially all of its theory. If only one
item on this page gets acquired, make it that.

Goldman & Kaper is the no-positive-coefficients theorem for order > 2 — the hard
obstruction behind §2.5's regime gating. McLachlan & Perlmutter plus Bhatt et al. are the
conformal-symplectic property that §2.4 turns into a validation gate. Blanes et al. is a
readable survey covering most of the rest.

Optional, only if the complex-coefficient escape route in §2.5 is ever pursued:
Castella, Chartier, Descombes & Vilmart, *Splitting methods with complex times for
parabolic equations*, BIT 49, 487 (2009), `10.1007/s10543-009-0235-y`; and
Hansen & Ostermann, *High order splitting methods for analytic semigroups exist*,
BIT 49, 527 (2009), `10.1007/s10543-009-0236-x`. Flagged out of scope in the plan.

## Tier 6 — comparison baselines (background)

Not needed to build anything; useful when writing up results against the field.

| Paper | Citation | DOI | Filename |
|---|---|---|---|
| **Price et al. 2018** — *Phantom: an SPH and MHD code for astrophysics* | PASA 35, e031 (2018) | `10.1017/pasa.2018.25` | `price2018_phantom.pdf` |
| **Borrow et al. 2022** — *SPHENIX: an SPH scheme for galaxy formation* | MNRAS 511, 2367 (2022) | `10.1093/mnras/stab3166` | `borrow2022_sphenix.pdf` |
| **Sandnes et al. 2024** — *REMIX SPH: improving mixing in SPH* | arXiv:2407.18587 | — | `sandnes2024_remix-sph.pdf` |

---

## Priority if the list is too long

1. **Hopkins 2013** — blocks PESPH-A entirely.
2. **Hairer, Lubich & Wanner** (book) — carries all of `SPLITTING_PLAN.md`'s theory.
3. **Springel & Hernquist 2002** — entropy formulation *and* grad-h, which is Phase 2's
   risk item.
4. **Owen 2014** — needed to touch `compSPH`/`crkSPH` safely.
5. **Rosswog 2020 (MAGMA2)** — the §7.5 upgrade that pays off under either direction.
