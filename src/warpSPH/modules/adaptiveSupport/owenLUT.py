"""Lookup-table builder and linear-interpolation queries for the Owen adaptive-support scheme.

``computeOwen`` builds a per-``(kernel, dim)`` table (via
``wp_psi.generatePSILut_warp``) mapping neighbor count ``n_h`` to the psi/psiH/N_H
statistics computed over a lattice, and exposes callable
``fromPsiH``/``fromPsi`` lookups (always evaluated in float64, then cast back
to the caller's dtype). ``interpolateLUT``/``linearInterpolateLUT`` do the
underlying searchsorted-based linear interpolation and can look up any one of
the four quantities (``n_h``, ``psi``, ``psiH``, ``n_H``) from any other.
"""

import torch
from warpSPHCore import *
from .wp_psi import generatePSILut_warp

__all__ = ['computeOwen', 'interpolateLUT']


def linearInterpolateLUT(LUT, x, xvalues):
    ileft = (torch.searchsorted(xvalues.contiguous(), x, right = True) - 1).clamp(min = 0, max = len(xvalues) - 2)
    iright = ileft + 1
    alpha = (x - xvalues[ileft]) / (xvalues[iright] - xvalues[ileft])
    alpha = alpha.clamp(min = 0.0, max = 1.0)
    
    return LUT[ileft] * (1 - alpha) + LUT[iright] * alpha

def get_n_h(Ln_h, Lpsi, LpsiH, LN_H, psi = None, psiH = None, n_H = None):
    if psi is not None:
        return linearInterpolateLUT(Ln_h, psi, Lpsi)
    elif psiH is not None:
        return linearInterpolateLUT(Ln_h, psiH, LpsiH)
    elif n_H is not None:
        return linearInterpolateLUT(Ln_h, n_H, LN_H)
    else:
        raise UserWarning("Nothing provided to interpolate")
    
def interpolateLUT(LUT, dim, which = 'n_h', n_h = None, psi = None, psiH = None, n_H = None):
    args = [n_h, psi, psiH, n_H]
    dtypes = [a.dtype if a is not None else None for a in args]
    devices = [a.device if a is not None else None for a in args]
    
    dtype = next((dtype for dtype in dtypes if dtype is not None), None)
    device = next((device for device in devices if device is not None), None)
    
    if len(LUT[1]) == 3:
        Ln_h = LUT[0].to(dtype = dtype, device = device)
        Lpsi = LUT[1][dim - 1].to(dtype = dtype, device = device)
        LpsiH = LUT[2][dim - 1].to(dtype = dtype, device = device)
        LN_H = LUT[3][dim - 1].to(dtype = dtype, device = device)
    else:
        Ln_h = LUT[0].to(dtype = dtype, device = device)
        Lpsi = LUT[1][0].to(dtype = dtype, device = device)
        LpsiH = LUT[2][0].to(dtype = dtype, device = device)
        LN_H = LUT[3][0].to(dtype = dtype, device = device)
    
    # print(Ln_h)
    
    n_h = n_h if n_h is not None else get_n_h(Ln_h, Lpsi, LpsiH, LN_H, psi, psiH, n_H)
    if which == 'n_h':
        return n_h
    elif which == 'psi':
        return linearInterpolateLUT(Lpsi, n_h, Ln_h)
    elif which == 'psiH':
        return linearInterpolateLUT(LpsiH, n_h, Ln_h)
    elif which == 'n_H':
        return linearInterpolateLUT(LN_H, n_h, Ln_h)

class computeOwen:
    def __init__(self, kernel: KernelFunctions, dim: int, nLUT =511, nMin = 1.0, nMax = 5.0, table: str = 'shell'):
        self.kernel = kernel
        n_h, psi, psiH, N_H = generatePSILut_warp(kernel, n_min = nMin, n_max = nMax, nLut = nLUT, table = table)
        LUTorch = [n_h, [psi[:,dim-1]], [psiH[:,dim-1]], [N_H[:,dim-1]]]
        self.LUT = LUTorch
        self.dim = dim
        
    def __call__(self, psiH_):
        return interpolateLUT(self.LUT, self.dim, which = 'n_h', psiH = psiH_.to(torch.float64)).to(dtype = psiH_.dtype, device = psiH_.device)
    
    def fromPsiH(self, psiH_):
        return interpolateLUT(self.LUT, self.dim, which = 'n_h', psiH = psiH_.to(torch.float64)).to(dtype = psiH_.dtype, device = psiH_.device)
    def fromPsi(self, psi_):
        return interpolateLUT(self.LUT, self.dim, which = 'n_h', psi = psi_.to(torch.float64)).to(dtype = psi_.dtype, device = psi_.device)

