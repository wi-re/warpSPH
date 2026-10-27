"""Opt-in explicit-midpoint energy controls; native Butcher path unchanged.

At each update Spheral weights conservative pair work by sign-filtered node
raw dudt. The midpoint predictor and the final update both start from the
step's initial state. Only the latest RHS is needed and is released after use.
An entropy alternative evaluates the published weights with the same final
pair-work sign as the dudt policy, separating formula and sign-estimator effects.
"""
from dataclasses import dataclass
import torch
from ..compSPH.pairWork import _pairVector
from ...math.scatter import segment_sum


def enabled(config):
    return (getattr(config,'crkEnergyPartition','native')!='native' or
            getattr(config,'crkEnergyPredictor','raw')!='raw')


@dataclass
class EnergyStage:
    index: int
    adjacency: object
    pairVector: torch.Tensor
    rawDudt: torch.Tensor
    entropy: object = None
    gamma: object = None


def retain(system,state,update,adjacency,index,config):
    if system.pairWork.weights is None or tuple(system.pairWork.weights)!=(0.,1.) or index not in (0,1):
        raise ValueError('CRK energy controls require the explicit midpoint tableau (0,1)')
    # Pair vector is f*a for native weights, otherwise the unweighted a.
    vector=_pairVector(state,config.energyScheme) if config.crkEnergyPartition=='native' else state.ap_ij+state.av_ij
    system.pairWork.crkEnergyStage=EnergyStage(index,adjacency,vector,update.dudt)
    if config.crkEnergyPartition=='entropy':
        gamma=torch.as_tensor([m.eos.gamma for m in config.materials],device=state.densities.device,dtype=state.densities.dtype)[state.materials.long()]
        system.pairWork.crkEnergyStage.gamma=gamma
        system.pairWork.crkEnergyStage.entropy=state.pressures/state.densities.pow(gamma)


def spheralRate(adjacency,pairAcceleration,rawDudt,baseVelocity,totalAcceleration,multiplier):
    i,j=adjacency.paddedI,adjacency.paddedJ
    half=baseVelocity+.5*multiplier*totalAcceleration
    work=torch.einsum('ij,ij->i',half[j]-half[i],pairAcceleration)
    # Pinned Spheral Scalar is double even when this control runs in float32.
    eps=torch.finfo(torch.float64).eps
    wi=(rawDudt[i]*work.sign()).clamp_min(eps)
    wj=(rawDudt[j]*work.sign()).clamp_min(eps)
    fraction=wi/(wi+wj)
    return segment_sum(fraction*work,adjacency.numNeighbors)


def entropyRate(adjacency,pairAcceleration,entropy,gamma,baseVelocity,totalAcceleration,multiplier):
    """Paper Eq.63 using the final compatible pair-work sign, not its RHS estimate."""
    i,j=adjacency.paddedI,adjacency.paddedJ
    half=baseVelocity+.5*multiplier*totalAcceleration
    work=torch.einsum('ij,ij->i',half[j]-half[i],pairAcceleration)
    si,sj=entropy[i],entropy[j];denom=si.abs()+sj.abs()+1e-14
    fraction=torch.where(work>=0,sj.abs()/denom,si.abs()/denom)
    fraction=torch.where((gamma[i]!=gamma[j])|(si==sj),.5,fraction)
    return segment_sum(fraction*work,adjacency.numNeighbors)


def energyBeforeUpdate(system,update,spec,config):
    if not enabled(config):return None
    stage=system.pairWork.crkEnergyStage
    if stage is None:raise RuntimeError('Missing CRK RHS energy data for midpoint state update')
    if spec.blend.reference_state is not None or spec.blend.self_scale is not None or isinstance(update,list):
        raise ValueError('CRK energy controls require unblended explicit midpoint updates')
    system.pairWork.crkEnergyStage=None
    if stage.index==0 and config.crkEnergyPredictor=='raw':return None
    s=system.state;h=spec.derivative_dt
    if config.crkEnergyPartition=='spheral':
        rate=spheralRate(stage.adjacency,stage.pairVector,stage.rawDudt,s.velocities,update.dvdt,h)
    elif config.crkEnergyPartition=='entropy':
        rate=entropyRate(stage.adjacency,stage.pairVector,stage.entropy,stage.gamma,s.velocities,update.dvdt,h)
    else:
        half=s.velocities+.5*h*update.dvdt;i,j=stage.adjacency.paddedI,stage.adjacency.paddedJ
        rate=segment_sum(torch.einsum('ij,ij->i',half[j]-half[i],stage.pairVector),stage.adjacency.numNeighbors)
    return s.internalEnergies+h*rate
