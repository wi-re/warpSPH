"""Builds a divergence-free 2D initial velocity field from a Perlin/Simplex noise potential (rotated gradient, `(dpsi/dy, -dpsi/dx)`), optionally SDF-ramped to zero at boundary regions.

The noise potential is generated once on a grid (``generateNoiseInterpolator``)
and interpolated onto particle positions by ``warpSPH.math.interpolation``'s
torch ``RegularGridInterpolator``, which keeps the field resident on the
querying device. It replaced scipy's host-only interpolator, whose
device -> host -> device round-trip per call dominated the per-step Kolmogorov
forcing in ``caseUtils/weaklyCompressible.py`` (~68% of host self-time at 155k
particles). Unlike scipy's, it is autograd-aware, so ``d(noise)/d(x)`` is a real
(piecewise-linear) gradient rather than an unconditional zero; the forcing call
site detaches its positions to keep its previous behaviour.
``generateRamp`` tapers the field to zero near boundary SDF regions
over a bandwidth set by ``schemeConfig.bandwith``, using a quintic smoothstep.
The resulting velocity is unit-normalized by its maximum magnitude and zeroed
on non-fluid (``kinds != 0``) particles.
"""

from warpSPH.configurations.compute import neighborSearchOptions

from ...math.interpolation import RegularGridInterpolator
from ..density.density import computeDensities
from warpSPHCore import *
import torch
import numpy as np
from ...math.noise import generateNoise

__all__ = ['sampleDivergenceFreeNoise']


def getSpacing(nx, domain: DomainDescription):
    periodicity = domain.periodic
    dxs = []
    for d in range(domain.dim):
        l = domain.max[d] - domain.min[d]
        dx = l/(nx if periodicity[d] else nx-1)
        dxs.append(dx.item())
    return np.min(dxs)

def generateNoiseInterpolator(fluidResolution, noiseResolution, domain: DomainDescription, dim = 2, octaves = 3, lacunarity=2, persistence = 0.5, baseFrequency = 1, tileable = False, kind = 'perlin', seed = 1248097, device = None):
    """A callable sampling a freshly generated noise field at arbitrary positions.

    The field is generated once on the host and then lives on whichever device
    first queries it (or `device`, if given), so a per-step caller pays no host
    round-trip: see `warpSPH.math.interpolation.RegularGridInterpolator`, which
    reproduces the `bounds_error = False, fill_value = None` scipy behaviour used
    here -- linear interpolation inside the grid, linear extrapolation outside it.

    The returned closure is differentiable w.r.t. its argument, unlike the scipy
    version it replaces; callers that do not want `d(noise)/d(x)` (the per-step
    Kolmogorov forcing) detach their positions before calling.
    """
    _, _, noiseField = generateNoise(noiseResolution, dim = dim, octaves = octaves, lacunarity=lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)
    grid = []
    dx = 1/2 * getSpacing(fluidResolution, domain)
    for d in range(domain.dim):
        grid.append(torch.linspace(domain.min[d].detach().cpu().item() + dx, domain.max[d].detach().cpu().item() - dx, noiseResolution, dtype = torch.float64))
    interpolator = RegularGridInterpolator(grid, noiseField, bounds_error = False, fill_value = None)
    if device is not None:
        interpolator.to(device = device, dtype = noiseField.dtype)
    return lambda x: interpolator(x).reshape(x.shape[0])


from ...geometry.sdf import operatorDict

def rampDivergenceFree(positions, noise, sdf_func, offset, d0 = 0.25):
    sdf = sdf_func(positions)
#     r = sdf / d0 /2  + 0.5
    r = (sdf - offset) / d0 / 0.5 - 1
#     ramped = r * r * (3 - 2 * r)
    ramped = 15/8 * r - 10/8 * r**3 + 3/8 * r**5
#     ramped = r
    ramped[r >= 1] = 1
    ramped[r <= -1] = -1
#     ramped[r <= 0] = 0
#     ramped[r <= -1] = -1
    
    # return sdf
    return (ramped /2 + 0.5) * (noise)

def generateRamp(perennialState, config, schemeConfig):
    regions = config.regions
    boundary_sdfs = [region.sdf for region in regions if region.type == RegionType.Boundary]
    # print(boundary_sdfs)
    combined_sdf = lambda x: boundary_sdfs[0](x)[0]
    for sdf in boundary_sdfs[1:]:
        combined_sdf = operatorDict['union'](combined_sdf, lambda x, sdf = sdf: sdf(x)[0])


    buffer = schemeConfig.bandwith
    dx = perennialState.masses.mean().pow(1/perennialState.positions.shape[1]).cpu().item() / schemeConfig.fluid.restDensity**(1/perennialState.positions.shape[1])

    ramp = rampDivergenceFree(perennialState.positions, torch.ones_like(perennialState.densities), combined_sdf, 
                              offset = dx/2, 
                              d0 = buffer * perennialState.supports)
    
    # print(f"Ramp min: {ramp.min().item()}, max: {ramp.max().item()}, mean: {ramp.mean().item()}")
    return ramp


from ...configurations.region import RegionType

def sampleDivergenceFreeNoise(particleState, domain, config, schemeConfig, nxGrid, octaves = 3, lacunarity = 2, persistence = 0.5, baseFrequency = 1, tileable = True, kind = 'perlin', seed = 45906734):
    # neighborhood, neighbors = evaluateNeighborhood(particleState, domain, config['kernel'], verletScale = config['neighborhood']['verletScale'], mode =  SupportScheme.SuperSymmetric, priorNeighborhood=None)
    
    adjacency = buildVerletList(
        particleState,
        config.domain,
        verletScale=config.verletScale,
        supportMode=SupportScheme.SuperSymmetric,
        priorNeighborhood=None,
        verbose=False,
        **neighborSearchOptions(config))

    noiseGen = generateNoiseInterpolator(nxGrid, nxGrid, domain, dim = domain.dim, octaves = octaves, lacunarity = lacunarity, persistence = persistence, baseFrequency = baseFrequency, tileable = tileable, kind = kind, seed = seed)

    dtype = particleState.positions.dtype
    ramp = generateRamp(particleState, config, schemeConfig) if len([r for r in config.regions if r.type == RegionType.Boundary]) > 0 else 1
    potential = noiseGen(particleState.positions).to(dtype) * ramp

    rho = computeDensities(particleState, config, schemeConfig, adjacency)
    priorDensity = particleState.densities.clone() 
    particleState.densities = rho

    gradTerm = warpOperation(
        particleState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Gradient,
            supportMode = SupportScheme.Gather, # cullen switch E.1 in the CRK paper uses gather for density estimation
        ),
        queryValues = potential,
        domain = config.domain,
        adjacency = adjacency,
    )

    # gradTerm = SPHOperation(particleState, potential, config['kernel'], neighbors.get('noghost')[0], neighbors.get('noghost')[1], Operation.Gradient, SupportScheme.Scatter, GradientMode.Difference)
    
    # sph_op(particleState, particleState, domain, config['kernel'], sparseNeighborhood, operation = 'gradient', supportScheme = 'symmetric', quantity = potential, gradientMode = 'difference')

    velocities = torch.stack([gradTerm[:,1], -gradTerm[:,0]], dim = 1)
    velocities = velocities / torch.linalg.norm(velocities, dim = 1, keepdim = True).max()

    velocities[particleState.kinds != 0, :] = 0

    particleState.densities = priorDensity
    return velocities