"""Opt-in CRK density schedules for fresh explicit midpoint.

``paper`` resets Eq76 density at the beginning of each full step, integrates
Eq79 within that step, and retains geometric Eq75 volume at every RHS. The
accepted endpoint carries the continuity density until the next reset; there
is no extra endpoint summation or energy/entropy adjustment at that reset.

The separate ``spheral`` controller resets V=m/rho and material density;
integrate density within stages and apply ContinuityVolumePolicy *after* rho
updates. RK moments are rebuilt from that stage's evolved volumes. This does
not replace warpSPH's support solver or conservative force/energy formulation.
"""
import math
import torch
from warpSPHCore import SupportScheme, buildVerletList
from warpSPHIntegrators.state.fields import update_component

from ...configurations.compute import neighborSearchOptions
from .terms import crkFactors


def kernelExtent(kernel):
    # Mathematical spline coordinate extents, not the D&A variance-based
    # kernelScale: supports here are compact cutoff radii, H_spheral=extent/R.
    values = {'CubicSpline': 2., 'QuarticSpline': 2.5, 'QuinticSpline': 3., 'B7': 3.5, 'B8': 4.}
    name = kernel.name
    if name not in values:
        raise ValueError('crkEvolution=spheral requires a supported B-spline kernel')
    return values[name]


def boundedVolume(volume, mass, rho, drhodt, multiplier, supports, dim, extent):
    sphere = (2., math.pi, 4.*math.pi/3.)[dim-1]
    minimum = .5*mass/rho
    maximum = sphere*(supports/extent).pow(dim)
    trial = volume - multiplier*mass/rho.square()*drhodt
    return torch.maximum(minimum, torch.minimum(maximum, trial))


def prepareStep(system, config, schemeConfig):
    s = system.state
    if s.crkMassDensities is None:
        s.crkMassDensities = s.densities.clone()
    paper=schemeConfig.crkEvolution=='paper'
    s.crkVolumes = None if paper else s.masses/s.crkMassDensities
    system.adjacency = buildVerletList(s, config.domain, verletScale=config.verletScale,
        supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=system.adjacency,
        **neighborSearchOptions(config))
    _, rho, _ = crkFactors(s, config.domain, config.kernel, system.adjacency,
        densityPolicy='massReplacement' if paper else 'sameMaterial', volumes=s.crkVolumes)
    s.crkMassDensities = rho
    s.densities = rho
    from ..eos import refreshEOS
    refreshEOS(s, schemeConfig)


def updateContinuityState(system, update, spec, config, schemeConfig=None):
    if spec.blend.reference_state is not None or spec.blend.self_scale is not None or isinstance(update, list):
        raise ValueError('Experimental CRK volume policy supports explicit midpoint updates only')
    update_component(system, update, spec, 'crk_density', 'density_derivative')
    s = system.state
    s.densities = s.crkMassDensities
    if getattr(schemeConfig,'crkEvolution','spheral')=='paper':
        # Geometric Eq75 is recomputed by crkFactors at each RHS; never evolve
        # or clamp a volume in this mode. Only density follows Eq79 mid-step.
        s.crkVolumes=None
        return
    s.crkVolumes = boundedVolume(s.crkVolumes, s.masses, s.densities, update.drhodt,
        spec.derivative_dt, s.supports, s.positions.shape[1], kernelExtent(config.kernel))
