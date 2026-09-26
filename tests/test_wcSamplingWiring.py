"""Wiring of `config.samplingScheme` into the WCSPH build path.

`cases/tgvWeaklyCompressible.buildSystem` routes every *non-regular*
`samplingScheme` to `sample.bySamplingScheme.sampleParticles` (which skips
the regular scheme's jitter+shuffle decorrelation) while `regular` keeps the
historical lattice+shuffle start. These tests pin both halves of that
contract: the dispatcher's branches behave as documented, and the case's
build produces a relaxed (non-crystalline) start for `optimal` with the
WCSPH uniform-density stamp kept.
"""

import torch

from warpSPH.configurations import buildConfig
from warpSPH.geometry import SamplingScheme
from warpSPH.sample import sampleRegularParticles
from warpSPH.sample.bySamplingScheme import sampleParticles
from warpSPH.utils.domain import buildDomainDescription


def _config(samplingScheme, nx=8):
    # float32: the warp kernels in the relaxation path (radius search,
    # density) are compiled for the process-global precision, which is
    # float32 unless warpSPHCore_PRECISION says otherwise.
    domain = buildDomainDescription(2.0, 2, periodic=True,
                                    device=torch.device('cpu'),
                                    dtype=torch.float32)
    config, _integrator = buildConfig(
        domain=domain, dim=2, n_h=4.0,
        samplingScheme=samplingScheme,
        nx=nx, dx=2.0 / nx,
        device=torch.device('cpu'), dtype=torch.float32)
    return config


def test_dispatcher_regular_is_the_plain_lattice():
    # The regular branch must stay the untouched regular lattice: it is the
    # default, and the case's own shuffle path builds on exactly this.
    config = _config(SamplingScheme.regular, nx=8)
    ps = sampleParticles(8, config)
    reg = sampleRegularParticles(8, config.domain, config.targetNeighbors)
    assert torch.allclose(ps.positions, reg.positions)
    assert torch.allclose(ps.masses, reg.masses)


def test_dispatcher_optimal_relaxes_off_the_lattice():
    import warp as wp
    wp.init()
    config = _config(SamplingScheme.optimal, nx=8)
    ps = sampleParticles(8, config)
    assert ps.positions.shape == (64, 2)
    box = config.domain.max - config.domain.min
    rel = ps.positions - config.domain.min
    assert torch.all(rel >= -1e-6) and torch.all(rel < box + 1e-6)
    # the mass sum still closes on the box volume
    assert abs(float(ps.masses.sum()) - float(box.prod())) < 1e-6
    # relaxed, i.e. NOT the plain regular lattice the setup alone would give
    reg = sampleRegularParticles(8, config.domain, config.targetNeighbors)
    assert not torch.allclose(ps.positions, reg.positions, atol=1e-6)


def test_tgv_wc_optimal_sampling_builds_relaxed_state():
    # The case-level half of the wiring: `samplingScheme='optimal'` must
    # reach the built system (relaxed positions) with the WCSPH stamps kept.
    import warp as wp
    wp.init()
    from warpSPH.cases import importAll
    from warpSPH.runner import buildContext, getCase
    from warpSPH.runner.caseSpec import CaseSpec
    importAll()
    case = getCase('tgv-wc')
    spec = (CaseSpec(caseName=case.name, scheme=case.scheme,
                     params=dict(case.params)).merged(**case.defaults)
            .merged(nx=16, samplingScheme='optimal'))
    ctx = buildContext(case, spec)
    system = case.buildSystem(ctx)
    assert system.state.positions.shape == (256, 2)
    box = ctx.config.domain.max - ctx.config.domain.min
    rel = system.state.positions - ctx.config.domain.min
    assert torch.all(rel >= -1e-6) and torch.all(rel < box + 1e-6)
    reg = sampleRegularParticles(16, ctx.config.domain, ctx.config.targetNeighbors)
    assert not torch.allclose(system.state.positions, reg.positions, atol=1e-6)
    # only positions come from the glass; the WCSPH uniform-density / zero-
    # velocity stamps stay (the shuffle path returns the same stamps)
    assert torch.allclose(system.state.densities,
                          torch.ones_like(system.state.densities))
    assert torch.allclose(system.state.velocities,
                          torch.zeros_like(system.state.velocities))
