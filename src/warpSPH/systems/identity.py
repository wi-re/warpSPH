"""Independent particle provenance and rigid-body membership.

``materials`` is a physical material index, never an emitter or body index.
These keyword-only fields preserve existing state constructor signatures.
``emitters=None`` means no recorded provenance (one unlabeled source).
``bodyIDs=None`` is reserved for old states whose boundary membership lived
in ``materials``; new region initializers always write explicit body IDs.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from warpSPHIntegrators.state import BaseState, constant


@dataclass(kw_only=True)
class ParticleIdentityState(BaseState):
    emitters: Optional[torch.Tensor] = constant(default=None, tags=('particle_emitter',))
    bodyIDs: Optional[torch.Tensor] = constant(default=None, tags=('particle_body',))


def particleEmitters(state):
    """Passive source labels for diagnostics/export; never read by dynamics."""
    ids = getattr(state, 'emitters', None)
    return torch.zeros_like(state.materials, dtype=torch.int32) if ids is None else ids


def particleBodyIDs(state):
    """Body membership; -1 on fluid rows, legacy boundary-label fallback."""
    ids = getattr(state, 'bodyIDs', None)
    if ids is None:
        return torch.where(state.kinds == 0, -torch.ones_like(state.materials), state.materials)
    return ids


def initialMaterialIDs(region, positions):
    """A region may contain one material or a position-dependent material map."""
    value = (region.initialConditions or {}).get('materials', 0)
    if callable(value):
        value = value(positions)
    ids = torch.as_tensor(value, device=positions.device)
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('initialConditions.materials must contain integer material IDs')
    if ids.ndim == 0:
        ids = ids.expand(positions.shape[0])
    if ids.shape != positions.shape[:1]:
        raise ValueError('initialConditions.materials must be scalar or have one ID per particle')
    if bool(((ids < 0) | (ids > torch.iinfo(torch.int32).max)).any()):
        raise ValueError('initialConditions.materials must contain nonnegative int32 IDs')
    return ids.to(torch.int32).clone()
