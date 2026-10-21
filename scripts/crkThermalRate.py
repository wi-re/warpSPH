"""Research-only continuous thermal ODE from the entropy work distribution.

This is the dt->0 limit of Frontiere 2017 Eqs.61/63, not its Eq.78, not an
unchanged Spheral option, and not the finite-step compatible algorithm. Same
gamma pure gas only; no threshold/floor is added to nonzero entropy sums.
"""
import torch
from warpSPH.math.scatter import segment_sum


def entropy_fraction(si,sj,power):
    total=si.abs()+sj.abs()
    denominator=torch.where(total>0,total,torch.ones_like(total))
    fraction=torch.where(power>=0,sj.abs()/denominator,si.abs()/denominator)
    return torch.where(total>0,fraction,torch.full_like(fraction,.5))


def thermal_entropy_rate(adjacency,pair_acceleration,entropy,velocity):
    i,j=adjacency.paddedI,adjacency.paddedJ
    power=torch.einsum('ij,ij->i',velocity[j]-velocity[i],pair_acceleration)
    fraction=entropy_fraction(entropy[i],entropy[j],power)
    return segment_sum(fraction*power,adjacency.numNeighbors)
