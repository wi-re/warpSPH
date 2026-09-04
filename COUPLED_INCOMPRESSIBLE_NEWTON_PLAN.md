# warpSPH — Automatic Newton-Krylov vs. IISPH for Incompressibility: Scoping

> ## Sequenced, not dropped (per the user, 09-04)
>
> This is still wanted — there is already downstream motivation (an implicit
> WCSPH scheme, among other things pointed at "coupled pressure/velocity")
> that needs exactly this kind of solver. Answers "Open questions needing a
> decision" item 1 below: the fully-coupled Newton solve is wanted, not
> satisfied in spirit by the Krylov work. What's actually blocking it isn't
> architecture uncertainty, it's sequencing — every one of these downstream
> plans needs a **working, stable implicit incompressible scheme as the
> comparison baseline first**, and that's exactly what
> `DFSPH_IMPROVEMENT_PLAN.md` is (`divergenceFree`, stabilized as of that
> plan's Part 56). `DFSPH_IMPROVEMENT_PLAN.md` is the current, ongoing
> priority; this plan is next once that baseline is in a state worth building
> a comparison against, not before.

Scoping-only document for `warpSPHCore/docs/historic_plans/warpier_forward_mode_plan.md`'s
Phase 6/Goal 4 ("automatic vs. IISPH for incompressibility"), written per that phase's own
instruction to scope it as a separate plan once Phase 4's shifting-comparison findings were in.
**No implementation in this document** — it re-derives what Phase 6's one-paragraph sketch
actually requires, finds that sketch was based on an assumption that doesn't hold, and proposes
a much smaller, concretely bounded Phase 1 in its place. Read this before building anything
against the original Phase 6 framing.

## tl;dr

Phase 6's sketch ("an automatic Newton-Krylov pressure-Poisson solve built from composed
`warpOperationJVP` calls... IISPH is rung 1, this would be rung 3 of `warpSPHIntegrators`'s
solver ladder") turns out to be based on a premise that's false for the sub-problem IISPH
actually solves: **the pressure-Poisson matvec needs no JVP at all.** It's already
`Divergence(Gradient(p)/rho)` — two plain `warpOperation` calls composed directly, because `p`
*is* the unknown the operator acts on, not a tangent direction. That whole sub-problem —
"replace IISPH's Jacobi smoother with a proper Krylov solve of the identical linear system,
built from composable primitives" — is **already done**, landed in
`docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md` (CG/BiCG/BiCGStab/GMRES/MINRES in
`modules/incompressible/krylov.py`), with zero JVP anywhere in it.

What's left of Phase 6's actual ambition — a genuinely *coupled*, *nonlinear* Newton solve over
pressure **and** velocity together, with EOS and boundary/free-surface built in — is real, but
it is not a mechanical follow-on to the Krylov work above; it is a new capability with no
existing reference implementation in this codebase to build "automatic" against or validate
correctness with (unlike the shifting comparison, which had `exactHessian` as a hand-built
ground truth). Building it well needs product/architecture decisions (see "Open questions
needing a decision" below) before a phased implementation plan can be written responsibly.
**Recommendation: don't build the full coupled solve speculatively.** Section "A concretely
bounded Phase 1" below proposes a much smaller, self-contained piece of genuine nonlinearity
that already exists in production and is currently only approximated — a real target with a
clear success metric, no new architecture, and an honest place for `warpOperationJVP` reasoning
to actually earn its keep.

## Why Phase 6's premise doesn't hold for the pressure-Poisson sub-problem

`modules/pressure/iisph.py` (`computePressureAccelIISPH`) and
`modules/incompressible/drift.py` (`computePressureShiftIISPH`) — the two calls
`solveDivergenceFree`/`solveIncompressible`'s Jacobi loop and every one of
`docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md`'s five new Krylov solvers use as the matvec — are:

```python
def computePressureAccelIISPH(state, pressureValues, config, ...):
    return -warpOperation(state, OperationProperties(operation=WarpOperation.Gradient,
                          gradientMode=GradientScheme.Symmetric, ...),
                          queryValues=pressureValues, ...) / state.densities[:, None]

def computePressureShiftIISPH(state, config, pressureAccels, ...):
    return -warpOperation(state, OperationProperties(operation=WarpOperation.Divergence,
                          gradientMode=GradientScheme.Difference, ...),
                          queryValues=pressureAccels, ...)
```

Both are plain `warpOperation` calls — the same generic, already-general-purpose forward SPH
operator dispatch every explicit scheme uses — applied directly to the trial pressure field.
There is no hand-rolled per-pair kernel here the way `implicitShifting.py`'s Hessian was (a
`sphKernelHessian` call bypassing `OperatorSpec` entirely, needing a real from-scratch
second-derivative to replace generically). The IISPH matvec was *already* "automatic" in Phase
4's sense — built from composable core operators, no bespoke derivation — before this plan
existed. `warpOperationJVP` was never actually load-bearing for this piece: `p` is the unknown
the matvec acts on, not a tangent direction through some other computation, so there was never a
differentiation step to automate here.

This means Phase 6's own comparison axis ("automatic path avoids how much bespoke pressure-solver
machinery" vs. `solveIncompressible`) already resolves in the automatic path's favor with *zero*
new code: `docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md`'s Krylov solvers reuse the exact same composable matvec
IISPH's own Jacobi loop does, just replacing the fixed-point relaxation with a real Krylov method.
In `warpSPHIntegrators/NOTES.md` §3.4's ladder terms, that work is best read as **rung 2** (a
proper linear solve of the frozen-state stage system) done with matrix-free primitives that
happen to need no FD or JVP because the system is exactly linear in the unknown already — not
"rung 3" in the sense Phase 6 meant (JVP matvecs standing in for a Jacobian that would otherwise
need forming), because there is no Jacobian being avoided here.

## What's actually still missing (the genuine "materially bigger lift")

Phase 6's deferral reasons were three: *coupled pressure/velocity DOFs*, *boundary and
free-surface handling*, and *EOS coupling*. Re-examined against the current tree:

- **Coupled DOFs.** `divergenceFree_step` (`schemes/divergenceFree.py`) is a splitting/projection scheme, not an
  implicit coupled solve: `dvdt` (forces, gravity, diffusion) is computed explicitly *once*,
  `solveDivergenceFree` then solves a **linear** pressure-Poisson correction against that frozen
  `dvdt`, and the pressure correction is added back in afterward
  (`dvdt + dvdt_diss + dvdt_pressure`). `solveIncompressible`'s density-error variant is the same
  shape: `predictedVelocities = v + dt*dvdt` is computed once, outside the pressure loop, and
  never refed back into the momentum equation as pressure iterates. **No scheme in this codebase
  re-linearizes momentum and pressure together inside an outer Newton loop.** There is no
  existing "coupled pressure/velocity Newton solve, hand-built" to compare an automatic one
  against — unlike shifting, where `exactHessian` already existed as ground truth before Phase 4
  started.
- **Boundary/free-surface.** Handled today by masking (`kinds != 0` rows zeroed/excluded) in both
  the Jacobi and Krylov paths — the same pattern the shifting solve uses. Not a blocker on its
  own; would carry over to a coupled solve the same way.
- **EOS coupling.** Confirmed **not currently exercised on the incompressible path at all**:
  `weaklyCompressibleEOS` is called from `schemes/deltaSPH.py` (the *explicit* weakly-compressible
  scheme) but is dead-commented in `schemes/divergenceFree.py` (`# currentState.pressures =
  weaklyCompressibleEOS(...)`, line 92) — the incompressible/IISPH path computes pressure purely
  from the Poisson solve and never consults an equation of state. Weakly-compressible-explicit and
  incompressible-implicit are two structurally separate scheme families today, selected at the
  scheme level, not two modes of one nonlinear system. "EOS coupling" for a unified Newton solve
  would mean deciding whether/how to merge those families, which is a scheme-architecture
  decision, not a solver one.

Building the coupled solve Phase 6 sketched would mean inventing a new scheme from scratch (a real
mixed pressure-velocity state vector, a real nonlinear residual, a real decision about whether/how
EOS enters), with no existing implementation to validate against and no existing bug reports or
user request motivating it. That is exactly the shape of work this doc's instructions say needs a
decision from you before scoping further, not something to build on spec.

## A concretely bounded Phase 1 (proposed, not started)

Rather than the open-ended coupled solve, there is one genuine, already-known nonlinearity in the
existing production code that is currently only *approximated*, has a clear correctness bar, needs
no new architecture, and is small enough to actually finish:

**`solveIncompressible`'s `clamp(pressure, min=0.0)`** (`modules/incompressible/incompressible.py`
line 145, and the Krylov path's `gauge='nonnegative'` post-projection in `krylov.py`). This is a
box-constrained linear complementarity problem in disguise — solve `A p = b` subject to `p >= 0`
— and today it's handled by solving the unconstrained linear system to convergence and then
clamping the result, which `docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md`'s own Scope section already flags
honestly: *"a true constrained solve for the inactive `solveIncompressible` `clamp(p, min=0)`
inequality — here we approximate with a linear solve + post-projection clamp, documented as
such."*

A real fix is a **projected/active-set Newton step** (or a semismooth-Newton reformulation of the
KKT stationarity condition `min(p, -residual) = 0`), which is where `warpOperationJVP`-style
composition would have an honest, novel job to do: the active-set mask itself is state-dependent
(which rows are clamped changes iteration to iteration), so the effective operator being solved
changes shape mid-solve in a way a fixed matvec doesn't capture, and getting the linearization of
that right generically (rather than by hand, per this one variant) is a legitimate small-scale
version of the "automatic vs. hand-derived" comparison Phase 4 ran for shifting — bounded to one
already-identified, already-documented gap instead of a from-scratch architecture.

This is offered as a candidate, not a commitment — it wasn't asked for, and
`solveIncompressible`/the density-error variant is not on the live `divergenceFree_step` path today (see
`docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md`'s own "Current state" section), so the value of fixing it depends
on whether that path matters to you. Flagging it here so the option is visible next to the bigger
ask it was found while scoping.

## Open questions needing a decision (before any coupled-solve plan can be written)

1. **Answered, 09-04 (see the banner at the top): still wanted, not satisfied by the Krylov
   work.** There's real downstream motivation — an implicit WCSPH scheme among other work that
   needs coupled pressure/velocity — but all of it needs a working, stable implicit incompressible
   scheme as the comparison baseline first. That baseline is `DFSPH_IMPROVEMENT_PLAN.md`'s
   `divergenceFree` scheme, and getting it there is the current priority; this plan resumes once
   that baseline is solid enough to build a comparison against, not before.
2. If still wanted: should it unify with the explicit weakly-compressible/EOS scheme family, or
   stay a pressure-only extension (e.g. a Newton step that only linearizes the density-error
   residual's nonlinear pieces — the `clamp`, and `rhoStar`'s dependence on `dt*divergence` — while
   still treating velocity as a per-step frozen predictor, i.e. a *bigger* Phase 1 than the one
   proposed above but well short of full DOF coupling)?
3. Is the bounded Phase 1 above (projected Newton for `solveIncompressible`'s `p >= 0` clamp)
   worth doing on its own, independent of the coupled-solve question, given that path is not on
   `divergenceFree_step` today?

## Pointers

- `warpSPHCore/docs/historic_plans/warpier_forward_mode_plan.md` — Phase 6/Goal 4, the origin of
  this scoping request.
- `docs/historic_plans/INCOMPRESSIBLE_SOLVER_PLAN.md` (this repo) — the Krylov-solver work that already closes the
  linear pressure-Poisson sub-problem; read its "Scope" section for the `clamp(p, min=0)` caveat
  this document's Phase 1 proposal builds on.
- `warpSPHIntegrators/NOTES.md` §3.4 — the solver-ladder framing Phase 6 was reasoning from;
  still the right frame for a future coupled solve, just not for the pressure-only sub-problem.
- `schemes/divergenceFree.py`, `schemes/deltaSPH.py`, `modules/incompressible/{incompressible,
  divergenceFree}.py`, `modules/eos/weaklyCompressible.py` — the architecture facts this document
  is based on.
