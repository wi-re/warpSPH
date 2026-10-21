"""Material-volume weighting for the existing dimension-generic Owen moment.

Only the neighbor contribution changes: weight=1 within a material, and
(m_j/rho_j)/(m_i/rho_i) across materials (pinned Spheral SPHSmoothingScale).
Lookup, relaxation, bounds, support convention and stage timing are unchanged.
The standard unweighted operator remains the default and is not modified.
"""
from typing import Any
import torch
import warp as wp
from warpSPHCore import (scalar_t,offset_t,domainData,adjacencyData,gridData,kernelState,
    OperationDirection,sphKernel,sphKernelGradient,safe_sqrt,checkDirectionality_i,checkDirectionality_j)
from warpSPHCore.autograd import SPHContext,OperatorSpec,OutputSpec,ExtraSpec,ExtraKind,launchOperator
from warpSPHCore.util import getParticleData
from warpSPHCore.radiusSearch.grid_util import getIndexRangeLane,laneSum
from warpSPHCore.math import computeDistanceVec
from .owenHermite import quadratic_gradient


@wp.func
def _sums(i:wp.int32,lane:wp.int32,lanes:wp.int32,query:Any,reference:Any,
    domain:domainData,useAdjacency:wp.bool,adj:adjacencyData,grid:gridData,properties:kernelState,
    ids:wp.array(dtype=wp.int32),weighted:bool,tabulated:bool,gradientCoefficients:wp.array(dtype=scalar_t),gradientStep:scalar_t):
    pi=getParticleData(query,i);value=scalar_t(0.);gradient=scalar_t(0.)
    if properties.operationMode!=wp.static(OperationDirection.TrueAllToToAll.value):
        if not checkDirectionality_i(pi.kind,properties.operationMode):return value,gradient
    # Match the original psi operator's arithmetic and h rescaling.
    factor=scalar_t(1.)/wp.pow(pi.support,-scalar_t(domain.dim))
    gradFactor=scalar_t(1.)/wp.pow(pi.support,-scalar_t(domain.dim+1))
    for o in range(grid.numOffsets if not useAdjacency else 1):
        begin,count=getIndexRangeLane(i,o,lane,lanes,useAdjacency,adj,grid,query,domain)
        if begin>=0:
            for k in range(count):
                at=begin+offset_t(k)
                j=wp.int32(adj.neighborList[at] if useAdjacency else grid.sortIndex[at])
                pj=getParticleData(reference,j)
                if properties.operationMode!=wp.static(OperationDirection.TrueAllToToAll.value):
                    if not checkDirectionality_j(pj.kind,properties.operationMode):continue
                weight=scalar_t(1.)
                if weighted and ids[i]!=ids[j]:weight=pj.mass*pi.density/(pi.mass*pj.density)
                W=sphKernel(pi.position,pj.position,pi.support,pj.support,properties,domain)
                grad=sphKernelGradient(pi.position,pj.position,pi.support,pj.support,properties,domain)
                value+=weight*(factor*W)
                moment=gradFactor*safe_sqrt(wp.dot(grad,grad))
                if tabulated:
                    delta=computeDistanceVec(pi.position,pj.position,domain)
                    moment=scalar_t(0.)
                    if wp.dot(delta,delta)>scalar_t(0.):
                        moment=quadratic_gradient(safe_sqrt(wp.dot(delta,delta))/pi.support,gradientCoefficients,gradientStep)
                gradient+=weight*moment
    return value,gradient


@wp.kernel
def _kernel(query:Any,reference:Any,domain:domainData,useAdjacency:wp.bool,adj:adjacencyData,
    grid:gridData,corrections:Any,properties:kernelState,ids:wp.array(dtype=wp.int32),
    weighted:bool,tabulated:bool,gradientCoefficients:wp.array(dtype=scalar_t),gradientStep:scalar_t,
    values:wp.array(dtype=scalar_t),gradients:wp.array(dtype=scalar_t)):
    i=wp.tid()
    if i<query.positions.shape[0]:
        a,b=_sums(i,0,1,query,reference,domain,useAdjacency,adj,grid,properties,ids,weighted,tabulated,gradientCoefficients,gradientStep)
        values[i]=wp.pow(a,scalar_t(1.)/scalar_t(domain.dim))
        gradients[i]=wp.pow(b,scalar_t(1.)/scalar_t(domain.dim))


@wp.kernel
def _tiled(query:Any,reference:Any,domain:domainData,useAdjacency:wp.bool,adj:adjacencyData,
    grid:gridData,corrections:Any,properties:kernelState,ids:wp.array(dtype=wp.int32),
    weighted:bool,tabulated:bool,gradientCoefficients:wp.array(dtype=scalar_t),gradientStep:scalar_t,
    values:wp.array(dtype=scalar_t),gradients:wp.array(dtype=scalar_t)):
    i,lane=wp.tid()
    a,b=_sums(i,lane,wp.block_dim(),query,reference,domain,useAdjacency,adj,grid,properties,ids,weighted,tabulated,gradientCoefficients,gradientStep)
    a,b=laneSum(a),laneSum(b)
    if lane==0:
        values[i]=wp.pow(a,scalar_t(1.)/scalar_t(domain.dim))
        gradients[i]=wp.pow(b,scalar_t(1.)/scalar_t(domain.dim))


_SPEC=OperatorSpec(kernel=_kernel,tiledKernel=_tiled,
    outputs=(OutputSpec(dtype=scalar_t),OutputSpec(dtype=scalar_t)),extras=(ExtraSpec('ids',ExtraKind.TENSOR),
        ExtraSpec('weighted',ExtraKind.SCALAR),ExtraSpec('tabulated',ExtraKind.SCALAR),
        ExtraSpec('gradientCoefficients',ExtraKind.TENSOR),ExtraSpec('gradientStep',ExtraKind.SCALAR)))


def computeMaterialPsi0Warp(queryParticles,operationProperties,domain,*,adjacency=None,weighted=True,table=None):
    ids=queryParticles.materials if weighted else queryParticles.kinds
    if ids.dtype!=torch.int32 or ids.shape!=queryParticles.masses.shape or ids.device!=queryParticles.masses.device:
        raise ValueError('Material Owen moments require int32 material IDs matching particle masses')
    ctx=SPHContext(query=queryParticles,properties=operationProperties,domain=domain,adjacency=adjacency)
    gradient=queryParticles.supports[:0] if table is None else table.tensors(queryParticles.supports.device,queryParticles.supports.dtype)['gradData']
    step=1. if table is None else float(table.arrays['gradMeta'][2])
    return launchOperator(_SPEC,ctx,ids=ids,weighted=weighted,tabulated=table is not None,gradientCoefficients=gradient,gradientStep=scalar_t(step))
