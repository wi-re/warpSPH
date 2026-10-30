"""Direct X6 work from shared CRK pair physics, after stage acceleration is known.

No ap/av/f arrays: force-only first sweep, then recompute each pair, partition
using the native stage-half-velocity estimator and write/accumulate W in place.
Read-only/tangent and specialized wall/control paths keep expanded operators.
"""
import torch
import warp as wp
from typing import Any
from warp.types import vector,matrix
from warpSPHCore import *
from ..compSPH.chunkedWork import workBuffer
from .accel import computeCrkSPHPair,_CRK_ACCEL
from .partition import crkEntropyFraction
from ..dissipation import DiffusionParameters
from ..eos.materials import MaterialTable,materialRuntime,uniformMaterialTable


@wp.kernel
def _crkWorkKernel(
    queryState:Any,referenceState:Any,domainState:domainData,
    useAdjacency:wp.bool,adjacencyState:adjacencyData,gridState:gridData,correctionData:Any,
    kernelProperties:kernelState,
    queryVelocities:wp.array(dtype=vector(length=Any,dtype=scalar_t)),referenceVelocities:wp.array(dtype=vector(length=Any,dtype=scalar_t)),
    queryEnergies:wp.array(dtype=scalar_t),referenceEnergies:wp.array(dtype=scalar_t),
    individualCs:wp.bool,queryCs:wp.array(dtype=scalar_t),referenceCs:wp.array(dtype=scalar_t),
    viscositySwitch:wp.bool,queryAlphas:wp.array(dtype=scalar_t),referenceAlphas:wp.array(dtype=scalar_t),
    explicitPressure:wp.bool,queryPressures:wp.array(dtype=scalar_t),referencePressures:wp.array(dtype=scalar_t),
    viscosityParams:DiffusionParameters,crkViscosityParams:Any,
    queryVelocityTensor:wp.array(dtype=matrix(shape=(Any,Any),dtype=scalar_t)),referenceVelocityTensor:wp.array(dtype=matrix(shape=(Any,Any),dtype=scalar_t)),
    calculateDudt:wp.bool,retainPairs:wp.bool,
    half:wp.array(dtype=vector(length=Any,dtype=scalar_t)),finalHalf:wp.array(dtype=vector(length=Any,dtype=scalar_t)),
    dt:wp.array(dtype=scalar_t),gamma:scalar_t,weight:scalar_t,add:wp.bool,reduce:wp.bool,
    work:wp.array(dtype=vector(length=Any,dtype=scalar_t)),materialTable:MaterialTable,
    queryMaterials:wp.array(dtype=wp.int32),referenceMaterials:wp.array(dtype=wp.int32),
    rowStart:wp.int32, rate:wp.array(dtype=scalar_t),
):
    i=wp.tid()
    xi,hi,mi,rhoi,ki=getParticle(queryState,i)
    useL,Li=getL_i(correctionData,i)
    useVolume,Vi=getVolume_i(correctionData,i)
    useCRK,Ai,Bi,gradAi,gradBi=getCRK_i(correctionData,i)
    vi=queryVelocities[i]
    csi=access_optional(queryCs,i,individualCs,viscosityParams.c_s)
    alphai=access_optional(queryAlphas,i,viscositySwitch,scalar_t(1.0))
    Pi=queryPressures[i]
    gi=gamma
    validi=True
    if materialTable.count>0:
        im=wp.int32(queryMaterials[i])
        validi=im>=0 and im<materialTable.count
        if validi:gi=materialTable.entries[im].gamma
    energy=scalar_t(0.0)
    begin=adjacencyState.neighborOffsets[i]
    for k in range(adjacencyState.numNeighbors[i]):
        at=begin+offset_t(k)
        j=wp.int32(adjacencyState.neighborList[at])
        ap,av,pairHeating=computeCrkSPHPair(i,j,domainState.dim,xi,hi,mi,rhoi,referenceState,correctionData,domainState,kernelProperties,
            useL,Li,Vi,Ai,Bi,gradAi,gradBi,vi,referenceVelocities,individualCs,csi,referenceCs,
            viscositySwitch,alphai,referenceAlphas,explicitPressure,Pi,referencePressures,
            viscosityParams,crkViscosityParams,queryVelocityTensor[i],referenceVelocityTensor,False)
        xj,hj,mj,rhoj,kj=getParticle(referenceState,j)
        gj=gamma
        validj=True
        if materialTable.count>0:
            jm=wp.int32(referenceMaterials[j])
            validj=jm>=0 and jm<materialTable.count
            if validj:gj=materialTable.entries[jm].gamma
        pair=ap+av
        estimated=mi*wp.dot(half[j]-half[i+rowStart],pair)*dt[0]
        fraction=crkEntropyFraction(Pi,rhoi,gi,referencePressures[j],rhoj,gj,estimated)
        if not validi or not validj:fraction=scalar_t(wp.nan)
        weighted=weight*(fraction*pair)
        if reduce:
            dot=scalar_t(0.0)
            for axis in range(domainState.dim):dot+=(finalHalf[j][axis]-finalHalf[i+rowStart][axis])*weighted[axis]
            energy+=dot
        elif add:work[at]=work[at]+weighted
        else:work[at]=weighted
    rate[i]=energy


_SPEC=OperatorSpec(rowChunked=True,kernel=_crkWorkKernel,outputs=(OutputSpec(dtype=scalar_t,shape=ShapeOf.QUERY),),extras=_CRK_ACCEL.extras+(
    ExtraSpec('half',ExtraKind.TENSOR,RowRole.REFERENCE),ExtraSpec('finalHalf',ExtraKind.TENSOR,RowRole.REFERENCE),ExtraSpec('dt',ExtraKind.TENSOR,RowRole.REPLICATED),
    ExtraSpec('gamma',ExtraKind.SCALAR),ExtraSpec('weight',ExtraKind.SCALAR),ExtraSpec('add',ExtraKind.SCALAR),ExtraSpec('reduce',ExtraKind.SCALAR),
    ExtraSpec('work',ExtraKind.CHUNK_TENSOR,lambda ctx,e: RowRole.REPLICATED if e['reduce'] else RowRole.CHUNK),ExtraSpec('materialTable',ExtraKind.SCALAR),
    ExtraSpec('queryMaterials',ExtraKind.TENSOR,RowRole.QUERY),ExtraSpec('referenceMaterials',ExtraKind.TENSOR,RowRole.REFERENCE),ExtraSpec('rowStart',ExtraKind.SCALAR,RowRole.ROW_START)))


def accumulateCRKWork(system,state,update,config,scheme,adjacency,stage,index,dt):
    pw=system.pairWork
    weight=pw.weights[index]
    if weight==0:return
    reducing=pw.halfVelocity is not None
    if not reducing:
        if pw.adjacency is not None and pw.adjacency is not adjacency:
            pw.consistent,pw.work=False,None
            return
        if not pw.consistent:return
    new=pw.work is None
    work=workBuffer(pw,state,adjacency,reducing)
    half=state.velocities+.5*dt*update.dvdt
    table=materialRuntime(scheme,state.densities)
    ctx=SPHContext(query=state,properties=OperationProperties(kernel=config.kernel,supportMode=SupportScheme.KernelMeanSymmetric),domain=config.domain,adjacency=adjacency,
        corrections=Corrections(volumes=(stage.volumes,None),crk=stage.crkState))
    rate=launchOperator(_SPEC,ctx,queryVelocities=state.velocities,referenceVelocities=state.velocities,
        queryEnergies=state.internalEnergies,referenceEnergies=state.internalEnergies,individualCs=True,
        queryCs=state.soundspeeds,referenceCs=state.soundspeeds,viscositySwitch=True,queryAlphas=state.alphas,referenceAlphas=state.alphas,
        explicitPressure=True,queryPressures=state.pressures,referencePressures=state.pressures,
        conductivityParams=scheme.diffusionParams,crkViscosityParams=stage.viscosityParams,
        queryVelocityTensor=stage.velocityGradient,referenceVelocityTensor=stage.velocityGradient,
        calculateDudt=False,retainPairs=False,half=half,finalHalf=pw.halfVelocity if reducing else half,
        dt=torch.as_tensor(dt,device=state.positions.device,dtype=state.positions.dtype).reshape(1),
        gamma=scalar_t(scheme.gamma),weight=scalar_t(weight),add=not new,reduce=reducing,work=work,rowStart=0,
        materialTable=table.table if table is not None else uniformMaterialTable(state.positions.device),
        queryMaterials=state.materials if table is not None else state.kinds,
        referenceMaterials=state.materials if table is not None else state.kinds)
    if reducing:pw.deltaU=rate if pw.deltaU is None else pw.deltaU+rate
    else:pw.adjacency,pw.work=adjacency,work
