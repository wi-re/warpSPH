"""Phase 4 step 3 of `warpSPHCore/warpier_forward_mode_plan.md`: "`Hess C . v`
is a JVP of that JVP" -- validated against `wp_implicitShifting.py`'s own `H`
output, the same way `test_implicitShiftingGradientJVP.py` validated step 2's
`grad C` against its `J` output.

`implicitShifting.py`'s Newton solve needs `Hess(C) @ x` (the BiCGStab
matvec, `implicitShifting._multiplyLaplacianBlock`): `out_i = diagBlock_i @
x_i - sum_{j in N(i)} H_ij @ x_j`, `diagBlock_i = sum_{j != i} omega_j H_ij`,
built from `wp_implicitShifting.computeShiftingPairTerms`'s hand-rolled
per-pair `sphKernelHessian` call plus a hand-written `torch.einsum`/
`scatter_sum` assembly -- no autograd involved, and (per the module's own
docstring, corrected during this same Phase 4 step 3) self-pairs are dropped
before assembly as an *exact* translation-invariance identity (the self
term's true contribution to `d^2 C_i/dx_i^2` is zero for any finite pairwise
kernel Hessian -- `sphKernelHessian` is itself well-defined and finite at
`r=0`, not unstable there; an earlier version of that docstring claimed
otherwise and was wrong), and the off-diagonal block is `-omega_k H_ik`, not
the naively-placed `omega_j H_ij` (the same identity, one sign away --
needed for the assembled operator to come out symmetric).

`warpSPHCore.warpOperationHVP` (Phase 4 step 3,
`coreOperations.wp_densityHVP.computeSPHDensityPositionHVP`) reduces to
exactly `sum_{j != i} m_j * H_ij @ (v_i - v_j)` -- the same closed form as
`_multiplyLaplacianBlock(diagBlock, Hw, v, ...)` once expanded algebraically
-- but was derived by differentiating `warpOperationJVP`'s own position
tangent a second time ("a JVP of that JVP"), then reusing
`kernels.hessian.sphKernelHessian` (the same building block
`computeShiftingPairTerms` calls) through `warpOperationHVP`'s own
adjacency/self-pair handling, not by transcribing `_multiplyLaplacianBlock`.
Agreement here is therefore evidence the composed-JVP route reaches the same
operator, not a tautology. (Whether the composed-JVP route needs the
self-pair drop *at all* turns out to depend on how it's called: for this
test's own `tangentQueryPositions == tangentReferencePositions` usage --
the same particle moving in both roles -- dropping is a bitwise no-op, since
the self term's `(v_i - v_i) = 0` factor already annihilates it; see
`warpSPHCore/tests/operations/test_forward_mode_tier2_density_hvp_self_pair.py`
for that check made explicit, and `wp_densityHVP.py`'s own docstring for why
it's still required for the *asymmetric* tangent construction
`implicitShiftingAutomatic.py` uses to isolate `diagBlock`.)
"""

from __future__ import annotations

import torch

from warpSPH.utils import buildDomainDescription
from warpSPH.configurations.simulationConfig import buildConfig
from warpSPH.sample.regular import sampleRegularParticles
from warpSPH.modules.shifting.wp_implicitShifting import computeShiftingPairTerms
from warpSPH.modules.shifting.implicitShifting import _multiplyLaplacianBlock
from warpSPH.math import scatter_sum
from warpSPHCore import ParticleState, SupportScheme, buildVerletList, warpOperationHVP, OperationProperties
from warpSPHCore.enumTypes import WarpOperation, OperationDirection


def _jitteredLatticeState(nx, dim, L, device, dtype, jitter, seed):
    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, _integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                      dtype=dtype, dx=L / nx, cflFactor=0.3)
    template = sampleRegularParticles(nx=nx, domain=domain, targetNeighbors=config.targetNeighbors)
    n = template.positions.shape[0]
    particleDx = template.masses.pow(1 / dim).mean().item()

    gen = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(template.positions.shape, generator=gen).to(device=device, dtype=dtype)
    positions = template.positions + jitter * particleDx * noise

    state = ParticleState(
        positions=positions, supports=template.supports, masses=template.masses,
        kinds=torch.zeros(n, device=device, dtype=torch.int32),
        densities=torch.full((n,), 1000.0, device=device, dtype=dtype),
    )
    return state, config, domain


def test_densityPositionHVP_matches_implicitShiftings_ownHessianMatvec():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dim, nx, L, rho0 = 2, 8, 1.0, 1000.0

    state, config, domain = _jitteredLatticeState(nx=nx, dim=dim, L=L, device=device, dtype=dtype,
                                                    jitter=0.1, seed=1234)
    n = state.positions.shape[0]
    adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)

    # --- Hand-built reference: implicitShifting.py's own H, assembled by ---
    # --- its own _multiplyLaplacianBlock matvec. ---------------------------
    _K, _J, H = computeShiftingPairTerms(state, domain, config.kernel, adjacency)
    pairMask = adjacency.livePairs()[0] != adjacency.livePairs()[1]  # implicitShifting.py drops self-pairs before assembly
    i, j, Hm = adjacency.livePairs()[0][pairMask], adjacency.livePairs()[1][pairMask], H[pairMask]
    omega = state.masses / rho0
    Hw = Hm * omega[j, None, None]
    diagBlock = scatter_sum(Hw, i, dim=0, dim_size=n)

    gen = torch.Generator(device='cpu').manual_seed(5678)
    v = torch.randn(n, dim, generator=gen).to(device=device, dtype=dtype)

    reference = _multiplyLaplacianBlock(diagBlock, Hw, v.flatten(), i, j, n, dim).view(n, dim)

    # --- Automatic: warpOperationHVP(Density) with omega as the mass channel, ---
    # --- v applied to both query and reference roles (the same particles moving --
    # --- together, matching implicitShifting's own usage). -----------------------
    omegaState = ParticleState(positions=state.positions, supports=state.supports, masses=omega,
                               kinds=state.kinds, densities=None)
    props = OperationProperties(kernel=config.kernel, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

    automatic = warpOperationHVP(omegaState, props, domain, adjacency=adjacency,
                                 tangentQueryPositions=v, tangentReferencePositions=v)

    torch.testing.assert_close(automatic, reference, rtol=1e-4, atol=1e-6)


def test_automaticHessianMatvec_isSymmetric():
    """Phase 4 step 5's other pitfall check ("is the resulting operator
    symmetric without anyone having to work out the `-omega_k H_ik`
    off-diagonal sign by hand?"), tested directly on the composed operator
    rather than inferred from its bit-close agreement with a known-symmetric
    hand-built reference above -- the plan's own standing preference (see
    Phase 3's own "verify empirically rather than assume") for checking a
    symmetry claim rather than assuming it transfers.

    `implicitShiftingAutomatic.py`'s own matvec closure calls
    `warpOperationHVP` with `tangentQueryPositions == tangentReferencePositions
    == x` (the same particles moving together) -- exactly Hess(C) applied to
    `x`, with no explicit diagonal/off-diagonal block construction anywhere
    in that call. `<matvec(a), b> == <a, matvec(b)>` for two *independent*
    random directions is therefore a check on the composed JVP-of-a-JVP
    itself, not on any hand-written assembly (there isn't one to get wrong
    here)."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32  # matches this file's other test; warpSPHCore's precision is set at import
    dim, nx, L, rho0 = 2, 8, 1.0, 1000.0

    state, config, domain = _jitteredLatticeState(nx=nx, dim=dim, L=L, device=device, dtype=dtype,
                                                    jitter=0.1, seed=1234)
    n = state.positions.shape[0]
    adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)
    omega = state.masses / rho0
    omegaState = ParticleState(positions=state.positions, supports=state.supports, masses=omega,
                               kinds=state.kinds, densities=None)
    props = OperationProperties(kernel=config.kernel, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

    def matvec(x):
        return warpOperationHVP(omegaState, props, domain, adjacency=adjacency,
                                tangentQueryPositions=x, tangentReferencePositions=x)

    gen = torch.Generator(device='cpu').manual_seed(2468)
    a = torch.randn(n, dim, generator=gen).to(device=device, dtype=dtype)
    b = torch.randn(n, dim, generator=gen).to(device=device, dtype=dtype)

    lhs = (matvec(a) * b).sum()
    rhs = (a * matvec(b)).sum()
    torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-6)
