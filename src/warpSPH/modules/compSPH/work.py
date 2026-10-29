"""Native CompSPH force/work recomputation without expanded pair arrays.

The shared pair function preserves mean-gradient force and gather heating.
Partition only after the completed stage acceleration/enforced updates exist.
Walls/materials/AD and alternative partitions use the expanded reference route.
"""
from typing import Any
import torch
import warp as wp
from warp.types import vector
from warpSPHCore import *
from ..dissipation import DiffusionParameters
from ..crk.partition import crkEntropyFraction
from ..crk.storage import derivativeInputs
from ...enumTypes import EnergyScheme
from .accel import computeCompSPHPair,_COMPSPH_ACCEL


def useRecomputedWork(system,state,scheme,wall,index):
    return (getattr(scheme,'compPairStorage','recompute')=='recompute'
        and scheme.compatibleEnergy and scheme.energyScheme==EnergyScheme.CRK
        and wall is None and getattr(scheme,'materials',None) is None
        and getattr(scheme,'stepMode','eager') in (None,'eager')
        # Execution='graph' can be inherited without setting stepMode.
        and not getattr(scheme,'supportSolveOnGrid',False)
        and not getattr(scheme,'inPlaceAdjacency',False)
        and system.pairWork is not None and system.pairWork.weights is not None
        and index is not None and not derivativeInputs(*vars(state).values()))


@wp.kernel
def _compWorkKernel(
    queryState:Any,referenceState:Any,domainState:domainData,
    useAdjacency:wp.bool,adjacencyState:adjacencyData,gridState:gridData,correctionData:Any,
    kernelProperties:kernelState,
    queryVelocities:wp.array(dtype=vector(length=Any,dtype=scalar_t)),referenceVelocities:wp.array(dtype=vector(length=Any,dtype=scalar_t)),
    queryEnergies:wp.array(dtype=scalar_t),referenceEnergies:wp.array(dtype=scalar_t),
    individualCs:wp.bool,queryCs:wp.array(dtype=scalar_t),referenceCs:wp.array(dtype=scalar_t),
    viscositySwitch:wp.bool,queryAlphas:wp.array(dtype=scalar_t),referenceAlphas:wp.array(dtype=scalar_t),
    explicitPressure:wp.bool,queryPressures:wp.array(dtype=scalar_t),referencePressures:wp.array(dtype=scalar_t),
    viscosityParams:DiffusionParameters,calculateDudt:wp.bool,retainPairs:wp.bool,
    half:wp.array(dtype=vector(length=Any,dtype=scalar_t)),finalHalf:wp.array(dtype=vector(length=Any,dtype=scalar_t)),
    dt:wp.array(dtype=scalar_t),gamma:scalar_t,weight:scalar_t,add:wp.bool,reduce:wp.bool,
    work:wp.array(dtype=vector(length=Any,dtype=scalar_t)),rate:wp.array(dtype=scalar_t),
):
    i=wp.tid()
    xi,hi,mi,rhoi,ki=getParticle(queryState,i)
    useL,Li=getL_i(correctionData,i)
    useH,omega=getGradH_i(correctionData,i)
    useVolume,Vi=getVolume_i(correctionData,i)
    useCRK,Ai,Bi,gradAi,gradBi=getCRK_i(correctionData,i)
    csi=access_optional(queryCs,i,individualCs,viscosityParams.c_s)
    alphai=access_optional(queryAlphas,i,viscositySwitch,scalar_t(1.0))
    Pi=queryPressures[i]
    energy=scalar_t(0.0)
    begin=adjacencyState.neighborOffsets[i]
    for k in range(adjacencyState.numNeighbors[i]):
        at=begin+offset_t(k)
        j=wp.int32(adjacencyState.neighborList[at])
        ap,av,heating=computeCompSPHPair(i,j,domainState.dim,xi,hi,mi,rhoi,referenceState,domainState,kernelProperties,
            useL,Li,useH,omega,correctionData.referenceOmegas,useVolume,Vi,correctionData.referenceVolumes,
            useCRK,Ai,Bi,gradAi,gradBi,queryVelocities[i],referenceVelocities,queryEnergies[i],referenceEnergies,
            individualCs,csi,referenceCs,viscositySwitch,alphai,referenceAlphas,
            explicitPressure,Pi,referencePressures,viscosityParams,False)
        xj,hj,mj,rhoj,kj=getParticle(referenceState,j)
        pair=ap+av
        estimated=mi*wp.dot(half[j]-half[i],pair)*dt[0]
        fraction=crkEntropyFraction(Pi,rhoi,gamma,referencePressures[j],rhoj,gamma,estimated)
        weighted=weight*(fraction*pair)
        if reduce:
            dot=scalar_t(0.0)
            for axis in range(domainState.dim):dot+=(finalHalf[j][axis]-finalHalf[i][axis])*weighted[axis]
            energy+=dot
        elif add:work[at]=work[at]+weighted
        else:work[at]=weighted
    rate[i]=energy


_SPEC=OperatorSpec(kernel=_compWorkKernel,outputs=(OutputSpec(dtype=scalar_t,shape=ShapeOf.QUERY),),extras=_COMPSPH_ACCEL.extras+(
    ExtraSpec('half',ExtraKind.TENSOR),ExtraSpec('finalHalf',ExtraKind.TENSOR),ExtraSpec('dt',ExtraKind.TENSOR),
    ExtraSpec('gamma',ExtraKind.SCALAR),ExtraSpec('weight',ExtraKind.SCALAR),ExtraSpec('add',ExtraKind.SCALAR),
    ExtraSpec('reduce',ExtraKind.SCALAR),ExtraSpec('work',ExtraKind.TENSOR)))


def accumulateCompWork(system,state,update,config,scheme,adjacency,gradH,index,dt):
    pw=system.pairWork;weight=pw.weights[index]
    if weight==0:return
    reducing=pw.halfVelocity is not None
    if not reducing:
        if pw.adjacency is not None and pw.adjacency is not adjacency:
            pw.consistent,pw.work=False,None
            return
        if not pw.consistent:return
    new=pw.work is None
    work=state.positions.new_empty((1 if reducing else adjacency.paddedI.numel(),config.dim)) if new or reducing else pw.work
    half=state.velocities+.5*dt*update.dvdt
    ctx=SPHContext(query=state,properties=OperationProperties(kernel=config.kernel,supportMode=SupportScheme.KernelMeanSymmetric),domain=config.domain,adjacency=adjacency,corrections=Corrections(gradH=gradH))
    alphas=state.alphas if state.alphas is not None else state.internalEnergies
    rate=launchOperator(_SPEC,ctx,queryVelocities=state.velocities,referenceVelocities=state.velocities,
        queryEnergies=state.internalEnergies,referenceEnergies=state.internalEnergies,
        individualCs=True,queryCs=state.soundspeeds,referenceCs=state.soundspeeds,
        viscositySwitch=state.alphas is not None,queryAlphas=alphas,referenceAlphas=alphas,
        explicitPressure=True,queryPressures=state.pressures,referencePressures=state.pressures,
        conductivityParams=scheme.diffusionParams,calculateDudt=False,retainPairs=False,
        half=half,finalHalf=pw.halfVelocity if reducing else half,
        dt=torch.as_tensor(dt,device=state.positions.device,dtype=state.positions.dtype).reshape(1),
        gamma=scalar_t(scheme.gamma),weight=scalar_t(weight),add=not new,reduce=reducing,work=work)
    if reducing:pw.deltaU=rate if pw.deltaU is None else pw.deltaU+rate
    else:pw.adjacency,pw.work=adjacency,work
