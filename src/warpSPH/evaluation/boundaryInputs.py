"""Declared geometry and UID-indexed inputs for the native boundary dispatch.

These objects are pure: an evaluation cannot advance a forcing clock or mutate
another attempt's coefficients. Arbitrary closures need their own audited adapter.
"""
from dataclasses import dataclass

import torch

from .continuation import uid_order
from .layout import PARTICLE_FIELDS, UPDATE_FIELDS, ParticleLayout


@dataclass(frozen=True)
class AllParticles:
    def __call__(self, positions):
        return positions.new_full((len(positions),), -1), torch.zeros_like(positions)


@dataclass(frozen=True)
class PlaneRegion:
    """Negative side of dot(x, normal)-offset; normal is a unit vector."""
    normal: torch.Tensor
    offset: float

    def __post_init__(self):
        n=self.normal
        if n.ndim != 1 or not bool(torch.isfinite(n).all()) or not bool(torch.isclose(n.norm(),n.new_tensor(1.))):
            raise ValueError('plane normal must be a finite unit vector')
        if not bool(torch.isfinite(n.new_tensor(self.offset))):raise ValueError('plane offset must be finite')
        object.__setattr__(self,'normal',n.clone())

    def __call__(self,positions):
        n=self.normal.to(positions)
        return positions@n-self.offset,n.expand_as(positions)


@dataclass(frozen=True)
class UIDValues:
    """A native Dirichlet/update/force callback with explicit, owned UID rows.

    Force values keep the native force units (the RHS divides by particle mass).
    Values are constant in time; time-dependent/stateful sources are not inferred.
    """
    ids: torch.Tensor
    values: torch.Tensor

    def __post_init__(self):
        ParticleLayout.create(self.ids,torch.arange(len(self.ids),device=self.ids.device))
        if (not self.values.ndim or self.values.shape[0] != len(self.ids)
                or self.values.device != self.ids.device or not bool(torch.isfinite(self.values).all())):
            raise ValueError('UID boundary values need one finite row per declared particle')
        object.__setattr__(self,'ids',self.ids.clone())
        object.__setattr__(self,'values',self.values.clone())

    def __call__(self,state,config,scheme,positions,distance,normal,t,dt):
        return self.values[uid_order(self.ids,state.UIDs)]


def validate_boundaries(boundaries,state):
    from ..configurations.moduleConfigurations.boundaryConditions import BoundaryCondition
    for bc in boundaries:
        if type(bc) is not BoundaryCondition or type(bc.sdf) not in (AllParticles,PlaneRegion):
            raise ValueError('boundaries need an explicit pointwise geometry adapter')
        if type(bc.sdf) is PlaneRegion and len(bc.sdf.normal) != state.positions.shape[1]:
            raise ValueError('boundary plane dimension differs from the state')
        for callbacks,allowed in ((bc.dirichletFunctions,{'velocities','internalEnergies'}),
                                  (bc.updateFunctions,{'dxdt','dvdt','dudt','dEdt','drhodt','dVdt','sourceAcceleration','sourceHeat'})):
            for name,fn in (callbacks or {}).items():
                if name not in allowed:raise ValueError('boundary target has no admitted particle axis')
                _validate_value(fn,state)
                if callbacks is bc.dirichletFunctions and (getattr(state,name,None) is None or fn.values.shape != getattr(state,name).shape):
                    raise ValueError('Dirichlet values differ from the target field shape')
                if callbacks is bc.updateFunctions:
                    shape=state.positions.shape if name in ('dxdt','dvdt','sourceAcceleration') else state.masses.shape
                    if fn.values.shape != shape:raise ValueError('boundary update values differ from the target field shape')
        for fn in bc.forcingFunctions or ():
            _validate_value(fn,state)
            if fn.values.shape != state.positions.shape:raise ValueError('forcing values must be particle vectors')


def _validate_value(fn,state):
    if type(fn) is not UIDValues:raise ValueError('boundary callbacks require an explicit UID input adapter')
    uid_order(fn.ids,state.UIDs)
    if fn.values.is_floating_point() and fn.values.dtype != state.positions.dtype:
        raise ValueError('boundary input precision differs from the state')


def layout_config(config):
    """Private configuration whose row-index caches belong to the new layout."""
    from copy import copy
    result=copy(config)
    result._kindIndexCache={}
    result._hasBoundaryParticles=None
    return result
