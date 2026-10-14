"""`modules/crk/terms.py`: the CRK correction terms from one warp kernel against `warpSPHCore.crk.computeCRKTermsWarp` (the torch version the
forward-mode path differentiates through), in 1-3D, with the degenerate-particle fallback, the tape fallback, and no host read."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='warp kernels')

from warpSPH.modules.crk.terms import computeCRKTerms, crkFactors
from warpSPHCore.crk import computeCRKTermsWarp as computeCRKTermsTorch
from warpSPHCore import computeCRKFactors, get_torch_precision


def _moments(dim, n=3000, seed=0):
    dt = get_torch_precision()
    g = torch.Generator(device='cuda').manual_seed(seed)
    r = lambda *s: torch.randn(*s, device='cuda', dtype=dt, generator=g)
    a = r(n, dim, dim)
    m_2 = a @ a.mT + 2.0 * torch.eye(dim, device='cuda', dtype=dt)           # SPD, well conditioned
    m_1 = 0.3 * r(n, dim)
    m_0 = 4.0 + r(n).abs()
    nn = torch.full((n,), 12, device='cuda', dtype=torch.int32)
    return m_0, m_1, m_2, 0.2 * r(n, dim), 0.2 * r(n, dim, dim), 0.2 * r(n, dim, dim, dim), nn, torch.ones(n, device='cuda', dtype=dt)


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-30)).item()


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_kernelMatchesTheTorchTerms(dim):
    args = _moments(dim)
    tol = 2e3 * torch.finfo(args[0].dtype).eps
    for name, a, b in zip('A B gradA gradB'.split(), computeCRKTerms(*args), computeCRKTermsTorch(*args)):
        assert _rel(a, b) < tol, name


@pytest.mark.parametrize('dim', [2, 3])
def test_degenerateParticlesGetTheIdentityCorrection(dim):
    m_0, m_1, m_2, dm0, dm1, dm2, nn, h = _moments(dim)
    nn[:100] = 1                                    # fewer than two neighbours
    m_2[100:200] = 0.0                              # singular moment matrix
    A, B, gradA, gradB = computeCRKTerms(m_0, m_1, m_2, dm0, dm1, dm2, nn, h)
    sel = slice(0, 200)
    assert torch.equal(A[sel], torch.ones_like(A[sel]))
    for t in (B, gradA, gradB):
        assert torch.equal(t[sel], torch.zeros_like(t[sel]))
    assert torch.isfinite(A).all() and torch.isfinite(gradB).all()


def test_tapeInputsUseTheTorchVersion():
    args = list(_moments(2, n=64))
    args[0] = args[0].requires_grad_(True)
    A, *_ = computeCRKTerms(*args)
    assert A.requires_grad


def test_noHostSync():
    args = _moments(3)
    computeCRKTerms(*args)
    torch.cuda.set_sync_debug_mode('error')
    try:
        computeCRKTerms(*args)
    finally:
        torch.cuda.set_sync_debug_mode('default')


def test_factorsMatchTheCorePipelineOnARealState():
    from warpSPH.cases import importAll
    from warpSPH.runner import run
    from warpSPH.runner.case import getCase
    importAll()
    r = run(getCase('sod2d'), quiet=True, progress=False, nSteps=5, scheme='CRKSPH', plot=False, video=False)
    st, cfg, adj = r.state.state, r.ctx.config, r.state.adjacency
    ref = computeCRKFactors(st, cfg.domain, cfg.kernel, adjacency=adj)
    new = crkFactors(st, cfg.domain, cfg.kernel, adjacency=adj)
    tol = 2e3 * torch.finfo(st.positions.dtype).eps
    assert _rel(new[0], ref[0]) < tol and _rel(new[1], ref[1]) < tol
    for name in ('A', 'B', 'gradA', 'gradB'):
        assert _rel(getattr(new[2], name), getattr(ref[2], name)) < tol, name
