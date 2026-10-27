"""Phase 4 step 2 of `warpSPHCore/warpier_forward_mode_plan.md`: "`grad C` is
exactly Tier 2.1's Density-operator position JVP... Validate this identity
first, standalone, against `wp_implicitShifting.py`'s `J` output."

`implicitShifting.py`'s Newton solve for implicit particle shifting needs
`grad_i C = d(C_i)/d(x_i)` -- each particle's own concentration gradient
w.r.t. its own position, treating every neighbor's position as fixed. It
gets this today by calling `sphKernelGradient` directly, per pair, and
scatter-summing (`wp_implicitShifting.computeShiftingPairTerms`'s `J`,
assembled into `Jw` by `implicitShifting._buildSystem`) -- hand-written,
pair-indexed physics with no autograd involved.

`C_i = sum_j omega_j * W_ij` is exactly `Density_i = sum_j m_j * W_ij` with
`omega` standing in for mass (`warpier_forward_mode_plan.md` Phase 4, item
2), so the *same* gradient should fall out of `warpSPHCore.warpOperationJVP`
applied to the Density operator, with `omega` passed as the mass channel and
the *query*-side position tangent set to each coordinate basis vector in
turn (the reference side left at its default zero tangent, so only the
"own position" partial derivative is taken, matching `Jw`'s own scope) --
this is genuinely a different code path (Tier-2 JVP assembly through
`kernels.kernelJVP.sphKernelJVP`, not a direct `sphKernelGradient` call), so
agreement here is not a tautology.
"""

from __future__ import annotations

import torch

from warpSPH.utils import buildDomainDescription
from warpSPH.configurations.simulationConfig import buildConfig
from warpSPH.sample.regular import sampleRegularParticles
from warpSPH.modules.shifting.wp_implicitShifting import computeShiftingPairTerms
from warpSPH.math import scatter_sum
from warpSPHCore import ParticleState, ParticleTangentState, SupportScheme, buildVerletList, warpOperationJVP, OperationProperties
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


def test_densityPositionJVP_matches_implicitShiftings_ownGradient():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    dim, nx, L, rho0 = 2, 8, 1.0, 1000.0

    state, config, domain = _jitteredLatticeState(nx=nx, dim=dim, L=L, device=device, dtype=dtype,
                                                    jitter=0.1, seed=1234)
    n = state.positions.shape[0]
    adjacency = buildVerletList(state, domain, config.verletScale, SupportScheme.SuperSymmetric, None)

    # --- Hand-built reference: implicitShifting.py's own J, assembled into Jw ---
    _K, J, _H = computeShiftingPairTerms(state, domain, config.kernel, adjacency)
    pairMask = adjacency.livePairs()[0] != adjacency.livePairs()[1]  # implicitShifting.py drops self-pairs before assembly
    i, j, Jm = adjacency.livePairs()[0][pairMask], adjacency.livePairs()[1][pairMask], J[pairMask]
    omega = state.masses / rho0
    JwReference = scatter_sum(Jm * omega[j, None], i, dim=0, dim_size=n)

    # --- Automatic: warpOperationJVP(Density) with omega as the mass channel, ---
    # --- one coordinate-basis query-position tangent per dimension. -------------
    omegaState = ParticleState(positions=state.positions, supports=state.supports, masses=omega,
                               kinds=state.kinds, densities=None)
    props = OperationProperties(kernel=config.kernel, operation=WarpOperation.Density,
                                supportMode=SupportScheme.Gather, operationMode=OperationDirection.AllToAll)

    zeroSupports = torch.zeros(n, device=device, dtype=dtype)
    columns = []
    for d in range(dim):
        basis = torch.zeros(n, dim, device=device, dtype=dtype)
        basis[:, d] = 1.0
        columns.append(warpOperationJVP(omegaState, props, domain, adjacency=adjacency,
                                        queryTangentState=ParticleTangentState(
                                            positions=basis, supports=zeroSupports, masses=None)))
    JwAutomatic = torch.stack(columns, dim=1)

    torch.testing.assert_close(JwAutomatic, JwReference, rtol=1e-5, atol=1e-6)
