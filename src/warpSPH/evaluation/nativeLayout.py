"""Accepted-boundary layouts for existing native fixed-count solver maps.

Preparation is an application operation between attempts. The native integrator
and every RHS keep their actual input order, including graph capture/replay.
"""
from copy import copy
from dataclasses import fields, replace

import torch

from .boundaryInputs import layout_config, validate_boundaries
from .boundaryLayout import _rows, _body_rows, permute_boundary_state
from .layout import morton_order, remap_adjacency

HISTORY_ROWS=('positionsPrev','velocitiesPrev','positionsPrev2','velocitiesPrev2','positionStep','positionLow')


def native_system(system):
    from ..systems.compSPH import CompSPHSystem
    from ..systems.weaklyCompressible import WeaklyCompressibleSystem
    from ..systems.incompressible import IncompressibleSystem,IISPHSystem,ProjectedSystem,DFSPHSystem
    from ..systems.artificialCompressible import ArtificialCompressibleSystem
    if type(system) not in (CompSPHSystem,WeaklyCompressibleSystem,IncompressibleSystem,IISPHSystem,ProjectedSystem,DFSPHSystem,ArtificialCompressibleSystem):
        raise TypeError('native layout requires an audited system family')
    if set(vars(system)) - {f.name for f in fields(system)} - {'stepDiagnostics'}:
        raise TypeError(f'undeclared native system memory needs an axis adapter: {set(vars(system)) - {f.name for f in fields(system)} - {"stepDiagnostics"}}')
    if type(system) is CompSPHSystem:
        from .layout import validate_state_schema
        validate_state_schema(system.state)
    else:_rows(system.state)


def system_view(system,order,*,adjacency=None):
    native_system(system)
    result=copy(system)
    from ..systems.compSPH import CompSPHSystem
    if type(system) is CompSPHSystem:
        from .layout import permute_state,ParticleLayout
        ParticleLayout.create(system.state.UIDs,order)
        result.state=permute_state(system.state,order)
        result.pairWork=result.supportGrid=result.listGrid=result.rerunRequest=result.spatialContext=None
    else:result.state=permute_boundary_state(system.state,order)
    result.adjacency=adjacency
    if hasattr(system,'stepDiagnostics'):
        from ..modules.diagnostics.weaklyCompressible import _LazyStepDiagnostics
        diag=system.stepDiagnostics
        if type(diag) is not _LazyStepDiagnostics:
            raise TypeError('unknown native diagnostics axes')
        accel,mask,shift=diag.raw
        result.stepDiagnostics=_LazyStepDiagnostics(accel[order],mask[order],
            None if shift is None else tuple(v[order] for v in shift))
    for name in HISTORY_ROWS:
        if not hasattr(system,name):continue
        value=getattr(system,name)
        if value is not None and value.shape != system.state.positions.shape:
            raise ValueError(f'{name} must have the declared particle-vector layout')
        setattr(result,name,None if value is None else value[order])
    if getattr(system,'deltaStages',None) is not None:
        from ..modules.deltaSPH.stages import DeltaStages
        old=system.deltaStages
        if type(old) is not DeltaStages or set(vars(old)) != {'diffusion','surfaceStart','previousSurfaceStart','surfaceStats'}:
            raise ValueError('unadapted native stage memory')
        result.deltaStages=DeltaStages(surfaceStart=None if old.surfaceStart is None else old.surfaceStart[order],
            previousSurfaceStart=None if old.previousSurfaceStart is None else old.previousSurfaceStart[order],
            surfaceStats=copy(old.surfaceStats))
    result.canonicalExport=True
    return result


def bodies_view(bodies,state):
    from ..configurations.rigidBody import RigidBody
    result=[]
    for body in bodies or ():
        if type(body) is not RigidBody:raise TypeError('unknown native body schema')
        owned=copy(body)
        for f in fields(body):
            value=getattr(body,f.name)
            if isinstance(value,torch.Tensor):setattr(owned,f.name,value.clone())
        owned.particleIndices=_body_rows(body.particleUIDs,state.UIDs)
        owned.ghostParticleIndices=_body_rows(body.ghostParticleUIDs,state.UIDs)
        if len(owned.particleIndices) and not torch.equal(state.ghostIndices[owned.particleIndices].long(),owned.ghostParticleIndices):
            raise ValueError('body-local ghost pairing differs from the particle relation')
        result.append(owned)
    return result


def _validate_provider(provider):
    if provider is None:return
    from warpSPHBoundaries.scene import AnalyticBoundary
    if type(provider) is not AnalyticBoundary:
        raise ValueError('native provider requires an audited pointwise geometry adapter')
    # AnalyticBoundary caches kernel layouts, not per-particle geometry. Its
    # fixed-adjacency arrays are freshly evaluated at each RHS position set.


def config_view(config,scheme,state):
    cfg=layout_config(config);sc=copy(scheme)
    if hasattr(sc,'_jacobiIterationHistory'):sc._jacobiIterationHistory=dict(sc._jacobiIterationHistory)
    if hasattr(scheme,'rigidBodies'):
        sc.rigidBodies=bodies_view(scheme.rigidBodies,state)
        cfg.rigidBodies=sc.rigidBodies
    provider=getattr(scheme,'boundaryProvider',None)
    _validate_provider(provider)
    if provider is not None:
        sc.boundaryProvider=copy(provider)
        sc.boundaryProvider.rigidBodies=[b for b in sc.rigidBodies if b.representation is not None]
    for name in ('_rhsGraph',):
        if hasattr(sc,name):delattr(sc,name)
    # Legacy unowned stage caches must not cross a layout generation.
    for name in ('_frozenDiffusionCache','_surfaceMaskStepStart','_surfaceMaskStepT'):
        if hasattr(sc,name):delattr(sc,name)
    if hasattr(sc,'_acsphGraphValidated'):sc._acsphGraphValidated=False
    return cfg,sc


class NativeMortonLayout:
    def __init__(self,system,integrator,scheme,*,retain_initial_order=False):
        native_system(system)
        if integrator.implicit or integrator.metadata.history_length:
            raise ValueError('native accepted layouts currently require a one-step explicit/map integrator')
        validate_boundaries(getattr(scheme,'boundaryConditions',()),system.state)
        for provider in (getattr(scheme,'boundaryProvider',None),getattr(system,'analyticBoundaryProvider',None),getattr(system,'contactBoundaryProvider',None)):
            _validate_provider(provider)
        self.ids=None
        self.generation=0
        self.sorts=0
        self.retain_initial_order=retain_initial_order
        self.pending=None
        self.uid_owner=None
        self.graph_observation=None

    @staticmethod
    def _uid_token(ids):
        return (ids.data_ptr(),ids._version,tuple(ids.shape),ids.dtype,ids.device)

    def trust_native_output(self,system):
        """Native maps preserve declared constant UID rows; hooks do not own them.

        Bind the returned constant buffer before calling hooks. Any subsequent
        write/replacement must pass the value check, including a reordered clone.
        Direct adapter callers without this native transition take that check too.
        """
        ids=system.state.UIDs
        if ids.shape!=self.ids.shape or ids.dtype!=self.ids.dtype or ids.device!=self.ids.device:
            raise ValueError('native particle roster changed')
        self.uid_owner=self._uid_token(ids)

    def _check_ids(self,system):
        ids=system.state.UIDs
        token=self._uid_token(ids)
        if token!=self.uid_owner:
            if (ids.shape!=self.ids.shape or ids.dtype!=self.ids.dtype or ids.device!=self.ids.device
                    or not torch.equal(ids,self.ids)):
                raise ValueError('native particle order changed outside the accepted layout adapter')
            self.uid_owner=token

    @staticmethod
    def _owner(system,config):
        def tensor(value):return (value.data_ptr(),value._version,tuple(value.shape),value.dtype,value.device)
        return (id(system),id(config),id(system.adjacency),config.verletScale,
            *(tensor(getattr(system.state,n)) for n in ('UIDs','positions','supports')),
            *(tensor(getattr(config.domain,n)) for n in ('min','max','periodic')))

    def flags(self,system,config):
        """Device flags, batched with the runner's existing scalar readback."""
        self._check_ids(system)
        observation=self.graph_observation
        self.graph_observation=None
        if observation is not None and observation[0]==self._owner(system,config):
            return observation[1]
        return self._stale(system,config)

    @staticmethod
    def _stale(system,config):
        from warpSPHCore import verletStaleFlag
        ids=system.state.UIDs
        return (ids.new_tensor(True,dtype=torch.bool) if system.adjacency is None else
            verletStaleFlag(system.state.positions,system.state.supports,system.adjacency,config.domain,config.verletScale).reshape(()))

    def capture_observation(self,system,config):
        """Read-only final-envelope check, outside the numerical step/RHS.

        Its host flag travels with the graph wrapper's existing decisions.
        Final-state staleness requests preparation of the *next* attempt; it
        does not invalidate the completed numerical step.
        """
        from ..utils.stepDecisions import recordingStepDecisions,StepDecision
        decisions=recordingStepDecisions()
        if decisions is None:raise RuntimeError('layout observation requires graph decision capture')
        def apply(host,accepted):
            self.graph_observation=(self._owner(accepted,config),bool(host[0]))
        values=self._stale(system,config).to(system.state.positions.dtype).reshape(1)
        decisions.append(StepDecision(values,lambda host:False,apply))

    def remember(self,system,config,stale):
        self.pending=(self._owner(system,config),bool(stale))

    def prepare(self,system,config,scheme):
        """Return the entire next continuation before entering an attempt."""
        ids=system.state.UIDs
        if self.ids is None and self.retain_initial_order:
            prepared=system_view(system,torch.arange(len(ids),device=ids.device),adjacency=system.adjacency)
            cfg,sc=config_view(config,scheme,prepared.state)
            for name in ('analyticBoundaryProvider','contactBoundaryProvider'):
                if hasattr(prepared,name) and getattr(prepared,name) is getattr(scheme,'boundaryProvider',None):
                    setattr(prepared,name,getattr(sc,'boundaryProvider',None))
            self.ids=ids.clone();self.trust_native_output(prepared);self.generation+=1
            return prepared,cfg,sc,True
        if self.ids is not None and (ids.shape!=self.ids.shape or ids.dtype!=self.ids.dtype or ids.device!=self.ids.device):
            raise ValueError('native particle roster changed')
        stale=system.adjacency is None
        if self.ids is not None:
            self._check_ids(system)
            if self.pending is not None and self.pending[0]==self._owner(system,config):
                stale=self.pending[1]
            else:
                stale=bool(self.flags(system,config))
            self.pending=None
        if self.ids is None or stale:
            order=morton_order(system.state.positions,config.domain)
            # Initial UID and relation validation happens even for identity.
            prepared=system_view(system,order)
            if hasattr(prepared.state,'wallVolumes') and prepared.state.wallVolumes is None and getattr(config,'_wallVolumes',None) is not None:
                from .continuation import uid_order
                owner=getattr(config,'_wallVolumeUIDs',None)
                if owner is None:raise ValueError('cached wall volumes need a UID owner')
                prepared.state.wallVolumes=config._wallVolumes[uid_order(owner,prepared.state.UIDs)]
            identity=torch.equal(order,torch.arange(len(order),device=order.device))
            if self.ids is None or not identity:
                cfg,sc=config_view(config,scheme,prepared.state)
                for name in ('analyticBoundaryProvider','contactBoundaryProvider'):
                    if hasattr(prepared,name) and getattr(prepared,name) is getattr(scheme,'boundaryProvider',None):
                        setattr(prepared,name,getattr(sc,'boundaryProvider',None))
                self.ids=prepared.state.UIDs.clone();self.trust_native_output(prepared);self.generation+=1;self.sorts+=not identity
                return prepared,cfg,sc,True
            self.ids=ids.clone()
        return system,config,scheme,False


def update_view(update,order):
    """Explicit native update axes, including exact-map diagnostic records."""
    from ..systems.weaklyCompressible import WeaklyCompressibleSystemUpdate
    from ..systems.incompressible import IncompressibleSystemUpdate
    from ..systems.artificialCompressible import ArtificialCompressibleSystemUpdate
    declared={WeaklyCompressibleSystemUpdate:{'dxdt','dvdt','drhodt','passive','drhodt_kin'},
        IncompressibleSystemUpdate:{'dxdt','dvdt','drhodt','passive'},
        ArtificialCompressibleSystemUpdate:{'dxdt','dvdt','dpdt','passive'}}
    if type(update) not in declared or {f.name for f in fields(update)} != declared[type(update)]:
        raise ValueError('native update needs declared axes')
    row_extras={'positionStep','positionLow','positionTarget','positionDxdt'}
    global_extras={'pseudoIterations','epsilonV','epsilonP','bdfOrder','exitReason','orderAudit'}
    records={'iisphResult','divergenceResult','densityResult','referenceFields'}
    if set(vars(update))-declared[type(update)]-row_extras-global_extras-records:
        raise ValueError('unknown native update metadata needs an axis adapter')
    result=copy(update)
    for name in declared[type(update)]|row_extras:
        if not hasattr(update,name):continue
        value=getattr(update,name)
        if value is not None and (not isinstance(value,torch.Tensor) or not value.ndim or value.shape[0]!=len(order)):
            raise ValueError(f'native update {name} needs one row per particle')
        setattr(result,name,None if value is None else value[order])
    from ..modules.iisph.solver import IISPHResult
    for name in records:
        value=getattr(update,name,None)
        if value is None:continue
        if name=='referenceFields':
            # The reference exact map owns these named row fields explicitly.
            allowed={'velocities','pressures','lambdas','normals','edges'}
            if set(value)!=allowed:raise ValueError('unknown reference-map field axes')
            setattr(result,name,{k:v[order] for k,v in value.items()})
        elif type(value) is IISPHResult:
            setattr(result,name,replace(value,pressure=value.pressure[order],acceleration=value.acceleration[order],source=value.source[order]))
        else:raise ValueError('unknown native projection record')
    return result


def canonical_native_view(system,stages=None,*,include_adjacency=True,include_stage_adjacency=True):
    p=torch.argsort(system.state.UIDs,stable=True)
    adjacency=system.adjacency
    view=system_view(system,p,adjacency=remap_adjacency(adjacency,p) if adjacency is not None and include_adjacency else None)
    view.canonicalExport=False
    if stages is None:return view,None
    out=[]
    for stage in stages:
        if not isinstance(stage.aux,(tuple,list)) or len(stage.aux) not in (2,3):raise ValueError('native stage export requires UID-bearing state aux')
        adj,state,*extra=stage.aux
        if extra:
            from numbers import Real
            def scalars(value):
                return isinstance(value,Real) or isinstance(value,(list,tuple)) and all(scalars(v) for v in value)
            if not scalars(extra[0]):raise ValueError('native stage diagnostic axes are unknown')
        p=torch.argsort(state.UIDs,stable=True)
        adj=remap_adjacency(adj,p) if adj is not None and include_stage_adjacency else None
        out.append(stage._replace(update=update_view(stage.update,p),aux=(adj,permute_boundary_state(state,p),*extra)))
    return view,out
