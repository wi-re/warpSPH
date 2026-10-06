"""Builds the Owen (1998) psi/psiH/N_H lookup table by evaluating kernel sums on a lattice, for a range of target neighbor counts.

``computePsi`` evaluates, for each sampled ``n_h`` in ``[n_min, n_max]``
(``nLUT`` uniform samples), a kernel-weighted sum over an idealized regular
lattice with spacing set so it has that neighbor count (a static +/-20-cell
window in each active dimension), giving the ``psi_0``/``psi_0_H`` statistics
and the reference neighbor volume ratio ``N_H``. ``generatePSILut_warp``
drives this for dimensions 1-3 in one call and always computes in
``torch.float64`` (independent of the caller's configured precision) since
the table is built once and reused.
"""

import math
from warpSPHCore import *
import warp as wp
from warpSPHCore import safe_sqrt, scalar_t
from warpSPHCore.kernels.eval_kernel import eval_k, eval_dkdq, eval_C_d

__all__ = ['generatePSILut_warp']

@wp.kernel
def computePsi(
    kernel: wp.int32,
    dim: wp.int32,
    n_min: scalar_t,
    n_max: scalar_t,
    nLUT: wp.int32,
    n_out: wp.array(dtype=scalar_t, ndim=1), # type: ignore
    psi: wp.array(dtype=scalar_t, ndim=1), # type: ignore
    psiH: wp.array(dtype=scalar_t, ndim=1), # type: ignore
    N_H: wp.array(dtype=scalar_t, ndim=1), # type: ignore
    psiHLattice: wp.array(dtype=scalar_t, ndim=1) # type: ignore
):
    i = wp.tid()
    if i >= nLUT:
        return
    
    dn = (n_max - n_min) / (scalar_t(nLUT) - scalar_t(1.0))
    n_h = n_min + scalar_t(i) * dn

    n_out[i] = n_h

    h = scalar_t(2.0)
    spacing = h / n_h
    dxx = spacing
    v = iPow(dxx, dim)
    x_range = wp.int32(20)
    y_range = wp.int32(20) if dim >= 2 else wp.int32(0)
    z_range = wp.int32(20) if dim >= 3 else wp.int32(0)

    vH = scalar_t(0.0)
    if dim == 1:
        vH = scalar_t(2.0) * h
    elif dim == 2:
        vH = scalar_t(math.pi) * h * h
    elif dim == 3:
        vH = scalar_t(4.0/3.0) * scalar_t(math.pi) * h * h * h

    numNeighbors = wp.int32(0)
    kSum = scalar_t(0.0)
    gradSum = scalar_t(0.0)


    for x in range(-x_range, x_range + 1):
        for y in range(-y_range, y_range + 1):
            for z in range(-z_range, z_range + 1):
                x2 = scalar_t(x) * scalar_t(x) * dxx * dxx
                y2 = scalar_t(y) * scalar_t(y) * dxx * dxx
                z2 = scalar_t(z) * scalar_t(z) * dxx * dxx
                r2 = x2 + y2 + z2
                r = safe_sqrt(r2)
                
                q = abs(r) / h
                W = eval_k(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim)
                gradW = (eval_dkdq(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(h, dim + 1))

                if q <= scalar_t(1.0):
                    numNeighbors += 1

                    kSum += W
                    gradSum += gradW

    deta = scalar_t(1.0) / n_h
    etar = deta
    eta_max_ = scalar_t(1.0)

    result = scalar_t(0.0)
    while etar < eta_max_:
        correction = scalar_t(0.0)
        if dim == 1:
            correction = scalar_t(2.0)
        elif dim == 2:
            correction = scalar_t(2.0) * scalar_t(math.pi) * etar / deta
        elif dim == 3:
            correction = scalar_t(4.0) * scalar_t(math.pi) * (etar * etar) / (deta * deta)

        q = etar
        val = wp.abs(eval_dkdq(q, dim, kernel) * eval_C_d(dim, kernel) / iPow(eta_max_, dim + 1))
        result += val * correction
        etar += deta

    psiH_0 = result ** (scalar_t(1.0) / scalar_t(dim))

    hReferenceFactor_WH = scalar_t(1.0) / iPow(scalar_t(2.0), dim)
    hActualFactor_WH = scalar_t(1.0) / iPow(h, dim)
    hScaling_WH = hReferenceFactor_WH / hActualFactor_WH

    psi_0 = (hScaling_WH * kSum)**scalar_t(scalar_t(1.0)/scalar_t(dim))

    psi[i] = psi_0
    psiH[i] = psiH_0
    N_H[i] = vH / v
    # the same statistic the runtime measures (`wp_psi0.py`: sum_j h^(d+1) |grad W_ij|), summed over the lattice
    # itself rather than the shell approximation above, which is exact only in 1D (SUPPORT_SOLVER_PLAN step 1a)
    psiHLattice[i] = (wp.abs(gradSum) * iPow(h, dim + 1)) ** (scalar_t(1.0) / scalar_t(dim))


    
import torch
from warpSPHCore import KernelFunctions
from warpSPHCore import *
def generatePSILut_warp(
        
    kernel: KernelFunctions,
    n_min: float,
    n_max: float,
    nLut: int,
    table: str = 'shell',
):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # torch_t = get_torch_precision()

    torch_t = torch.float64

    psis = torch.zeros((nLut,3), dtype=torch_t, device=device)
    psiHs = torch.zeros((nLut,3), dtype=torch_t, device=device)
    N_Hs = torch.zeros((nLut,3), dtype=torch_t, device=device)
    psiHLats = torch.zeros((nLut,3), dtype=torch_t, device=device)

    for d in range(1, 4):
        # print(f'Generating LUT for dim {d}...')
        n_out_wp = wp.zeros(nLut, dtype=scalar_t)
        psi_wp = wp.zeros(nLut, dtype=scalar_t)
        psiH_wp = wp.zeros(nLut, dtype=scalar_t)
        N_H_wp = wp.zeros(nLut, dtype=scalar_t)
        psiHLat_wp = wp.zeros(nLut, dtype=scalar_t)

        wp.launch(
            computePsi,
            dim = nLut,
            inputs = [
                kernel.value,
                d,
                n_min,
                n_max,
                nLut,
                n_out_wp,
                psi_wp,
                psiH_wp,
                N_H_wp,
                psiHLat_wp
            ]
        )
        psis[:, d-1] = wp.to_torch(psi_wp).clone().to(device)
        psiHs[:, d-1] = wp.to_torch(psiH_wp).clone().to(device)
        N_Hs[:, d-1] = wp.to_torch(N_H_wp).clone().to(device)
        psiHLats[:, d-1] = wp.to_torch(psiHLat_wp).clone().to(device)
    return wp.to_torch(n_out_wp).clone().to(device), psis, (psiHLats if table == 'lattice' else psiHs), N_Hs