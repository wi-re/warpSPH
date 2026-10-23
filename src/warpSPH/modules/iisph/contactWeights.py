"""Shepard interpolation weights at the closest wall contact of each query."""

from typing import NamedTuple
from types import SimpleNamespace
import torch
from warpSPHCore import buildVerletList, SupportScheme
from ...configurations.compute import neighborSearchOptions
from .operators import IISPHKernelData, computeIISPHKernelGeometry, launchIISPHKernel


class IISPHContactWeights(NamedTuple):
    pairs: IISPHKernelData
    denominator: torch.Tensor


def computeIISPHContactWeights(state, config, points, wallGradient, restVolumes):
    # Contact-centered support differs from fluid-centered support.
    query = SimpleNamespace(positions=points.contiguous(), supports=state.supports)
    adjacency = buildVerletList(query, config.domain, supportMode=SupportScheme.Gather,
        referenceParticles=state, **neighborSearchOptions(config))
    pairs = computeIISPHKernelGeometry(query, config, adjacency, reference=state)
    denominator = torch.empty_like(restVolumes)
    launchIISPHKernel('contact_denominator', state, [pairs.indices, pairs.offsets,
        pairs.counts, pairs.weights, restVolumes, state.kinds.to(torch.int32), wallGradient, denominator])
    missing = (state.kinds == 0) & (wallGradient.square().sum(-1) > 1.e-24) & (denominator <= 0)
    if bool(missing.any()):
        raise ValueError("IISPH contact pressure has no fluid support at a solid contact")
    return IISPHContactWeights(pairs, denominator)
