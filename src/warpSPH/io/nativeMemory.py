"""Cache-free, UID-labelled numerical memory for native fixed-count maps.

The sampled template reconstructs geometry and callbacks; this file contains
only numeric history and body state. Unknown geometry must declare an adapter.
"""
from dataclasses import fields
import hashlib
import json
import math

import torch

from ..evaluation.nativeLayout import HISTORY_ROWS,bodies_view,native_system
from .continuation import portable

POSE=('centerOfMass','orientation','angularVelocity','linearVelocity','load')
BODY_STATIC=('mass','inertia','particlePositions','particleVelocities','particleMasses',
    'particleUIDs','particleBoundaryDistances','particleBoundaryNormals','ghostParticlePositions',
    'ghostParticleUIDs','ghostParticleBoundaryDistances','ghostParticleBoundaryNormals',
    'bodyID','kind','dynamic')


def _geometry(provider,owned_bodies):
    if provider is None:return None
    from ..evaluation.nativeLayout import _validate_provider
    _validate_provider(provider)
    from warpSPHBoundaries.scene.scene import BoxRep,SurfaceRep,ImplicitRep,DiskArrayRep,SdfRep
    axes={BoxRep:('lo','hi','solid'),SurfaceRep:('vertices','edges','background'),
          ImplicitRep:('shape','policy'),DiskArrayRep:('centres','radii'),
          SdfRep:('d','origin','spacing','fallback','maxKappaH','gradTol','minCurvatureDenominator','maxCurvatureVariationH2','backend')}
    def describe(rep):
        if rep is None:return None
        if type(rep) not in axes:raise ValueError('checkpoint analytic geometry needs a portable adapter')
        return [type(rep).__name__,{k:describe(getattr(rep,k)) if k=='fallback' else portable(getattr(rep,k)) for k in axes[type(rep)]}]
    bodies=[]
    for body in provider.scene.bodies:
        reps=[]
        for rep in body.reps:
            reps.append(describe(rep))
        description=dict(reps=reps,bodyId=body.bodyId,bc=body.bc,bcValue=portable(body.bcValue),
            periodic=portable(body.periodic),periodicSupport=body.periodicSupport)
        if not any(b.representation is body for b in owned_bodies):
            description['fixedPose']={k:portable(getattr(body,k)) for k in ('center','angle','linearVelocity','angularVelocity','linearAcceleration','angularAcceleration')}
        bodies.append(description)
    return dict(bodies=bodies,volumeMode=provider.scene.volumeMode)


def signature(config,schemeConfig):
    # Sampling regions do not participate in stepping. Their reconstructed UID
    # roster, body-local geometry and live analytic representation are checked.
    excluded={'regions','rigidBodies','boundaryProvider','cudaGraph'}
    scheme={f.name:portable(getattr(schemeConfig,f.name)) for f in fields(schemeConfig) if f.name not in excluded}
    owned_bodies=getattr(schemeConfig,'rigidBodies',())
    def value(body,key):
        v=getattr(body,key)
        # An unbounded static analytic wall has legitimate infinite mass and
        # inertia. Admit these two declared constants without admitting NaNs
        # or infinities in physical particle/history/pose fields.
        if key in ('mass','inertia') and isinstance(v,torch.Tensor) and bool(torch.isposinf(v).all()):
            return ['positive_infinity',str(v.dtype),list(v.shape)]
        return portable(v)
    bodies=[{k:value(b,k) for k in BODY_STATIC} for b in owned_bodies]
    payload=[portable(config),scheme,bodies,_geometry(getattr(schemeConfig,'boundaryProvider',None),owned_bodies),
             config.compute.iterationChecks]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _write(group,name,value):
    if value is None:group.attrs[name]='None'
    elif isinstance(value,torch.Tensor):group.create_dataset(name,data=value.detach().cpu().numpy())
    else:group.attrs[name]=value


def _read(group,name,template,device,*,shape=None):
    if name in group:
        value=torch.as_tensor(group[name][()],device=device)
        if shape is not None and value.shape != shape:raise ValueError(f'checkpoint {name} shape differs from its axis')
        if isinstance(template,torch.Tensor) and (value.shape!=template.shape or value.dtype!=template.dtype):
            raise ValueError(f'checkpoint {name} precision/shape differs from the template')
        if not bool(torch.isfinite(value).all()):raise ValueError(f'checkpoint {name} must be finite')
        return value
    if name not in group.attrs:raise ValueError(f'checkpoint missing native {name}')
    value=group.attrs[name]
    if isinstance(value,str) and value=='None':return None
    if not math.isfinite(float(value)):raise ValueError(f'checkpoint {name} must be finite')
    if isinstance(template,torch.Tensor) or shape is not None:raise ValueError(f'checkpoint {name} requires a numeric tensor')
    return value


def writeNativeMemory(group,system,*,config,schemeConfig,integrator,next_dt):
    native_system(system)
    group.attrs.update(version=1,signature=signature(config,schemeConfig),integrator=integrator.identifier.name)
    _write(group,'next_dt',next_dt)
    schedule=getattr(schemeConfig,'_jacobiIterationHistory',{})
    if set(schedule)-{'acsphDualTime'} or any(type(v) is not int or v<1 for v in schedule.values()):
        raise ValueError('native stopping schedule needs a declared scalar adapter')
    group.attrs['iterationHistory']=json.dumps(schedule,sort_keys=True)
    order=torch.argsort(system.state.UIDs,stable=True)
    # The list's build envelope is algorithmic memory: native closures can
    # depend on its candidate topology. Rebuild its CSR from numeric anchors,
    # never from serialized offsets, addresses or executable search objects.
    spatial=group.create_group('spatial')
    _write(spatial,'UIDs',system.state.UIDs)
    adj=system.adjacency
    for name,value in (('positions',None if adj is None else adj.queryPositions),
                       ('supports',None if adj is None else adj.querySupports)):
        _write(spatial,name,value)
    history=group.create_group('history')
    for name in HISTORY_ROWS:
        if hasattr(system,name):
            value=getattr(system,name)
            _write(history,name,None if value is None else value[order])
    if hasattr(system,'dtPrev'):_write(history,'dtPrev',system.dtPrev)
    if hasattr(system,'deltaStages'):
        masks=group.create_group('deltaMasks')
        for name in ('surfaceStart','previousSurfaceStart'):
            value=getattr(system.deltaStages,name,None)
            _write(masks,name,None if value is None else value[order])
    bodies=group.create_group('bodies')
    for i,body in enumerate(getattr(schemeConfig,'rigidBodies',())):
        bg=bodies.create_group(str(i))
        for name in POSE:_write(bg,name,getattr(body,name))


def readNativeMemory(file,system,*,config,schemeConfig,integrator):
    native_system(system)
    if 'nativeMemory' not in file:
        if float(system.t)!=0:raise ValueError('native Morton restart requires persisted native numerical memory')
        return
    group=file['nativeMemory']
    if (group.attrs.get('version')!=1 or group.attrs.get('signature')!=signature(config,schemeConfig)
            or group.attrs.get('integrator')!=integrator.identifier.name):
        raise ValueError('native checkpoint configuration/integrator differs from the sampled template')
    device=system.state.positions.device
    history=group['history']
    expected={n for n in HISTORY_ROWS if hasattr(system,n)}|({'dtPrev'} if hasattr(system,'dtPrev') else set())
    if set(history)|set(history.attrs) != expected:raise ValueError('native checkpoint history schema differs')
    values={}
    for name in expected:
        value=_read(history,name,None,device,shape=system.state.positions.shape if name in HISTORY_ROWS else None)
        if isinstance(value,torch.Tensor) and name in HISTORY_ROWS and value.dtype!=system.state.positions.dtype:
            raise ValueError('native history precision differs from the particle state')
        values[name]=value
    if float(system.t)!=0 and hasattr(system,'dtPrev'):
        required=('positionsPrev','velocitiesPrev')
        if schemeConfig.acParams.incrementalPositions:required+=('positionStep','positionLow')
        if values['dtPrev'] is None or float(values['dtPrev'])<=0 or any(values[n] is None for n in required):
            raise ValueError('accepted ACSPH checkpoint has incomplete BDF memory')
    bodies=bodies_view(getattr(schemeConfig,'rigidBodies',()),system.state)
    if set(group['bodies']) != {str(i) for i in range(len(bodies))}:raise ValueError('checkpoint body roster differs')
    for i,body in enumerate(bodies):
        bg=group['bodies'][str(i)]
        if set(bg)|set(bg.attrs) != set(POSE):raise ValueError('checkpoint body schema differs')
        for name in POSE:setattr(body,name,_read(bg,name,getattr(body,name),device))
    dt=_read(group,'next_dt',None,device)
    if dt is None or float(dt)<=0:raise ValueError('checkpoint next dt must be positive')
    for name,value in values.items():setattr(system,name,value)
    if hasattr(system,'deltaStages'):
        from ..modules.deltaSPH.stages import DeltaStages
        masks=group['deltaMasks']
        if set(masks)|set(masks.attrs) != {'surfaceStart','previousSurfaceStart'}:
            raise ValueError('checkpoint delta mask schema differs')
        system.deltaStages=DeltaStages(**{name:_read(masks,name,None,device,shape=system.state.UIDs.shape)
            for name in ('surfaceStart','previousSurfaceStart')})
    schemeConfig.rigidBodies=config.rigidBodies=bodies
    if getattr(schemeConfig,'boundaryProvider',None) is not None:
        schemeConfig.boundaryProvider.rigidBodies=[b for b in bodies if b.representation is not None]
        from ..boundary.provider import bindBodies
        bindBodies(schemeConfig.boundaryProvider,bodies)
    config.dt=dt
    schedule=json.loads(group.attrs['iterationHistory'])
    if not isinstance(schedule,dict) or set(schedule)-{'acsphDualTime'} or any(type(v) is not int or v<1 for v in schedule.values()):
        raise ValueError('checkpoint stopping schedule has invalid scalar memory')
    schemeConfig._jacobiIterationHistory=schedule
    spatial=group['spatial']
    if set(spatial)|set(spatial.attrs) != {'UIDs','positions','supports'}:
        raise ValueError('native checkpoint spatial seed schema differs')
    ids=_read(spatial,'UIDs',system.state.UIDs,device)
    from ..evaluation.continuation import uid_order
    order=uid_order(system.state.UIDs,ids)
    anchors=_read(spatial,'positions',system.state.positions,device)
    supports=_read(spatial,'supports',system.state.supports,device)
    if (anchors is None)!=(supports is None):raise ValueError('checkpoint has an incomplete list envelope')
    from ..evaluation.nativeLayout import system_view
    adopted=system_view(system,order)
    if anchors is not None:
        if not bool((supports>0).all()):raise ValueError('checkpoint list supports must be positive')
        from copy import copy
        from warpSPHCore import buildVerletList,SupportScheme
        from ..configurations.compute import neighborSearchOptions
        seed=copy(adopted.state)
        seed.positions,seed.supports=anchors,supports/config.verletScale
        adopted.adjacency=buildVerletList(seed,config.domain,verletScale=config.verletScale,
            supportMode=SupportScheme.SuperSymmetric,**neighborSearchOptions(config))
    system.__dict__.update(vars(adopted))
    schemeConfig.rigidBodies=config.rigidBodies=bodies_view(bodies,system.state)
    if getattr(schemeConfig,'boundaryProvider',None) is not None:
        schemeConfig.boundaryProvider.rigidBodies=[b for b in schemeConfig.rigidBodies if b.representation is not None]
    return True
