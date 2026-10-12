"""Builds an `AcousticCoreSystem` (`JFNK_PLAN.md` Phase B) on a periodic,
boundary-free domain seeded with a Taylor-Green-vortex-like velocity field,
reusing `cases/periodic/tgvWeaklyCompressible.py`'s sampling/domain approach
(`sampleParticles` on a periodic `DomainDescription`) per Phase B step 2,
rather than building sampling from scratch. Not a `Case` -- there is no
`Case`/CLI wrapping this, matching `sample/waveSystem.py`'s own
"last stage of the pipeline, not an entry point" framing -- callers (tests,
notebooks) call this directly.
"""

import math
from typing import Optional

import torch

from ..configurations import AcousticCoreConfig, SimulationConfig, buildConfig
from ..systems.acousticCore import AcousticCoreState, AcousticCoreSystem
from ..utils import buildDomainDescription
from .bySamplingScheme import sampleParticles
from warpSPHCore import SupportScheme, buildVerletList

__all__ = ['buildPeriodicVortexAcousticCoreSystem']


def buildPeriodicVortexAcousticCoreSystem(
    nx: int = 24,
    dim: int = 2,
    L: float = 2.0,
    uMag: float = 0.05,
    rho0: float = 1.0,
    soundSpeed: float = 10.0,
    cflFactor: float = 0.1,
    forcingAmplitude: float = 0.0,
    forcingWavenumber: float = 4.0,
    densityDiffusionCoefficient: float = 0.0,
    velocityDiffusionCoefficient: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """A rotational, divergence-free (in the continuum limit) periodic
    vortex field -- the same family `cases/periodic/tgvWeaklyCompressible.py` seeds,
    just without that case's shuffling/`Case` machinery -- on a uniform
    initial density `rho0`. Returns `(system, config, schemeConfig)`.

    `forcingAmplitude`/`forcingWavenumber` (`JFNK_PLAN.md` Phase E1) thread
    straight through to `AcousticCoreConfig` -- pass `uMag=0.0` alongside a
    nonzero `forcingAmplitude` for a quiescent-start Kolmogorov-flow-like
    system driven entirely by the forcing term, rather than the vortex IC.

    `densityDiffusionCoefficient`/`velocityDiffusionCoefficient` (Phase
    E1.5) thread straight through the same way -- both `0.0` (off) by
    default, recovering Phase B/E1 exactly.
    """
    device = device if device is not None else (
        torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    dtype = dtype if dtype is not None else torch.float32

    domain = buildDomainDescription(l=L, dim=dim, periodic=True, device=device, dtype=dtype)
    config, _integrator = buildConfig(dim=dim, nx=nx, domain=domain, device=device,
                                      dtype=dtype, dx=L / nx, cflFactor=cflFactor)

    particles = sampleParticles(nx, config)
    n = particles.positions.shape[0]
    k = 2 * math.pi / L

    velocities = torch.zeros_like(particles.positions)
    if dim >= 2:
        x, y = particles.positions[:, 0], particles.positions[:, 1]
        velocities[:, 0] = uMag * torch.cos(k * x) * torch.sin(k * y)
        velocities[:, 1] = -uMag * torch.sin(k * x) * torch.cos(k * y)

    state = AcousticCoreState(
        positions=particles.positions, supports=particles.supports,
        masses=particles.masses, densities=torch.full_like(particles.positions[:, 0], rho0),
        velocities=velocities,
        kinds=torch.zeros(n, device=device, dtype=torch.int32),
        materials=torch.zeros(n, device=device, dtype=torch.int32),
        UIDs=torch.arange(n, device=device, dtype=torch.int32),
        UIDcounter=n,
    )
    adjacency = buildVerletList(state, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
    system = AcousticCoreSystem(state=state, adjacency=adjacency, domain=config.domain, t=0.0)
    schemeConfig = AcousticCoreConfig(restDensity=rho0, soundSpeed=soundSpeed,
                                      forcingAmplitude=forcingAmplitude, forcingWavenumber=forcingWavenumber,
                                      densityDiffusionCoefficient=densityDiffusionCoefficient,
                                      velocityDiffusionCoefficient=velocityDiffusionCoefficient)
    return system, config, schemeConfig
