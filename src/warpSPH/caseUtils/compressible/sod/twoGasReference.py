"""Exact two-gamma Sod profile: left rarefaction, contact, right shock.

The pressure solve is independent of the simulation's pair forces. Supports
the zero-velocity initial states used by Spheral's SodSolutionGasGas.
Equal initial pressures reduce to a stationary two-gas contact.
"""
import math
from functools import lru_cache
import torch
from ..richtmyerMeshkov.reference import interfaceStarState


@lru_cache(maxsize=32)
def sodWaves(left, right):
    star = interfaceStarState(left, right)
    r, v, p, g = left
    stationary = v == 0 and right[1] == 0 and p == right[2]
    if v != 0 or right[1] != 0 or (not stationary and not right[2] < star['pressure'] < p):
        raise ValueError('twoGasSod reference requires stationary states producing a left fan and right shock')
    c = math.sqrt(g * p / r)
    cs = c * (star['pressure'] / p)**((g-1)/(2*g))
    return dict(star, fanHead=-c, fanTail=star['velocity']-cs)


def sodProfile(x, time, left, right):
    waves = sodWaves(tuple(left), tuple(right))
    r, _, p, g = left
    rr, _, pr, gr = right
    if time <= 0:
        return dict(density=torch.where(x < 0, r, rr), pressure=torch.where(x < 0, p, pr),
                    velocity=torch.zeros_like(x))
    xi = x / time
    c = math.sqrt(g * p / r)
    # Left fan Riemann invariant; clamp outside fan before fractional powers.
    cf = (2*c - (g-1)*xi) / (g+1)
    ratio = (cf/c).clamp(min=1e-15)
    rho = torch.where(xi <= waves['fanHead'], r,
        torch.where(xi < waves['fanTail'], r * ratio**(2/(g-1)), waves['densityLeft']))
    pressure = torch.where(xi <= waves['fanHead'], p,
        torch.where(xi < waves['fanTail'], p * ratio**(2*g/(g-1)), waves['pressure']))
    velocity = torch.where(xi <= waves['fanHead'], 0.,
        torch.where(xi < waves['fanTail'], 2/(g+1)*(c+xi), waves['velocity']))
    rightOfContact = xi > waves['velocity']
    unshocked = xi > waves['waveRight']
    rho = torch.where(rightOfContact, torch.where(unshocked, rr, waves['densityRight']), rho)
    pressure = torch.where(unshocked, pr, pressure)
    velocity = torch.where(unshocked, 0., velocity)
    return dict(density=rho, pressure=pressure, velocity=velocity)
