"""Accepted-boundary row transformations for ghost states and particle bodies.

No integration attempt calls these utilities. They return owned state/config
views; callers commit both together after acceptance, without retaining a graph,
an old stage cache or an unadapted numerical history.
"""
from copy import copy,deepcopy
from dataclasses import fields,replace

import torch

from .layout import ParticleLayout
from .boundaryInputs import layout_config

BASE_ROWS=frozenset('emitters bodyIDs positions velocities supports masses densities kinds materials UIDs pressures soundspeeds surfaceIndicators surfaceNormals surfaceLambdas ghostIndices ghostOffsets'.split())


def _rows(state):
    from ..systems.weaklyCompressible import WeaklyCompressibleState
    from ..systems.incompressible import IncompressibleState,DFSPHState
    from ..systems.artificialCompressible import ArtificialCompressibleState
    schemas={WeaklyCompressibleState:BASE_ROWS|{'boundaryAccelerations'},
             IncompressibleState:BASE_ROWS|{'boundaryAccelerations'},
             DFSPHState:BASE_ROWS|{'boundaryAccelerations','divergencePressures','densityPressureImpulse','divergencePressureImpulse'},
             ArtificialCompressibleState:BASE_ROWS}
    if type(state) not in schemas:raise TypeError('ghost layout requires an audited boundary state schema')
    rows=schemas[type(state)]
    names={f.name for f in fields(state)}
    if names != rows|{'UIDcounter'} or set(vars(state)) != names:
        raise TypeError('boundary fields changed: declare all particle/global/index axes')
    for name in rows:
        value=getattr(state,name)
        if value is not None and (not isinstance(value,torch.Tensor) or not value.ndim or value.shape[0] != len(state.UIDs)):
            raise ValueError(f'boundary {name} needs one row per particle')
    return rows


def permute_boundary_state(state,order):
    """Gather every declared field and conjugate the ghost/boundary index map."""
    rows=_rows(state)
    layout=ParticleLayout.create(state.UIDs,order)
    indices=state.ghostIndices
    if indices is not None:
        n=len(indices)
        if indices.dtype not in (torch.int32,torch.int64) or indices.shape != (n,) or bool(((indices < -1)|(indices >= n)).any()):
            raise ValueError('invalid ghost/boundary row indices')
        valid=indices>=0
        owners=torch.arange(n,device=indices.device)[valid]
        if not torch.equal(indices[indices[valid]],owners.to(indices.dtype)):
            raise ValueError('ghost/boundary index relation must be reciprocal')
        if bool(((state.kinds[valid]+state.kinds[indices[valid]]) != 3).any()) or bool(((state.kinds[valid] != 1)&(state.kinds[valid] != 2)).any()):
            raise ValueError('ghost pairs must connect boundary and ghost rows')
        if bool(((state.kinds == 2)&~valid).any()):raise ValueError('every ghost needs a boundary owner')
    values={name:(None if getattr(state,name) is None else getattr(state,name)[order]) for name in rows}
    if indices is not None:
        gathered=indices[order]
        values['ghostIndices']=torch.where(gathered>=0,layout.q[gathered.clamp_min(0)],gathered).to(indices.dtype)
    values['UIDcounter']=state.UIDcounter
    return type(state)(**values)


def _body_rows(ids,target):
    if ids.ndim != 1 or ids.dtype != target.dtype or ids.device != target.device or ids.unique().numel() != ids.numel():
        raise ValueError('body IDs must be unique particle IDs on the state device')
    p=torch.argsort(target)
    loc=torch.searchsorted(target[p],ids)
    if bool((loc>=len(target)).any()) or not torch.equal(target[p[loc]],ids):
        raise ValueError('rigid body owns a foreign particle UID')
    return p[loc]


def boundary_bodies(bodies,state):
    """Keep compact body-local order and replace global masks with scatter rows."""
    from ..configurations.rigidBody import RigidBody
    result=[]
    for body in bodies or ():
        if type(body) is not RigidBody:raise TypeError('unadapted rigid body schema')
        rows=_body_rows(body.particleUIDs,state.UIDs)
        ghosts=_body_rows(body.ghostParticleUIDs,state.UIDs)
        if bool((state.kinds[rows] != 1).any()) or bool((state.kinds[ghosts] != 2).any()):
            raise ValueError('body particle UID kinds differ from the state')
        if len(rows) != len(ghosts) or (len(rows) and not torch.equal(state.ghostIndices[rows].long(),ghosts)):
            raise ValueError('body-local particle and ghost order differs from the relation map')
        owned=deepcopy(body)
        owned.particleIndices,owned.ghostParticleIndices=rows,ghosts
        result.append(owned)
    return result


def reorder_boundary_system(system,order,*,config,schemeConfig):
    """Atomic next-state/config view for the native one-step WCSPH route.

    Do not use with retained multistep memory or captured CUDA graphs. Other
    families' extra system histories/providers require their own acceptance adapter.
    """
    from ..systems.weaklyCompressible import WeaklyCompressibleSystem
    if type(system) is not WeaklyCompressibleSystem:
        raise TypeError('accepted boundary-system adaptation currently requires WCSPH')
    if set(vars(system)) != {f.name for f in fields(system)}:
        raise TypeError('undeclared boundary system memory needs an acceptance adapter')
    if getattr(schemeConfig,'boundaryProvider',None) is not None or any(getattr(body,'representation',None) is not None for body in getattr(schemeConfig,'rigidBodies',None) or ()):
        raise ValueError('analytic boundary providers need a separate acceptance adapter')
    from .boundaryInputs import validate_boundaries
    validate_boundaries(getattr(schemeConfig,'boundaryConditions',()),system.state)
    state=permute_boundary_state(system.state,order)
    cfg=layout_config(config)
    scheme=copy(schemeConfig)
    scheme.rigidBodies=boundary_bodies(getattr(schemeConfig,'rigidBodies',None),state)
    stages=None
    if system.deltaStages is not None:
        from ..modules.deltaSPH.stages import DeltaStages
        owner=system.deltaStages
        if type(owner) is not DeltaStages or set(vars(owner)) != {'diffusion','surfaceStart','previousSurfaceStart','surfaceStats'}:
            raise ValueError('unadapted boundary stage memory')
        def mask(value):
            if value is None:return None
            if not isinstance(value,torch.Tensor) or value.shape != (len(order),):raise ValueError('surface memory needs one row per particle')
            return value[order]
        stages=DeltaStages(surfaceStart=mask(owner.surfaceStart),previousSurfaceStart=mask(owner.previousSurfaceStart),surfaceStats=deepcopy(owner.surfaceStats))
    next_system=replace(system,state=state,adjacency=None,deltaStages=stages)
    return next_system,cfg,scheme
