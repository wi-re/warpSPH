"""Moments of the Morris pair weight over the solid next to a flat or circular wall: the geometry tables of the no-slip wall viscosity closure (`wallViscosity.py`).

Port of `warpSPHBoundaries/sim/wallmoments.py` (`curved_moments`, `CurvedWallMoments`; the Morris weight and the Monaghan-Gingold "alpha" weight both, the closure uses the Morris one). The fluid pair sum of the
scheme is truncated at a wall; the missing part is the same integral over the solid region `S`, with a continuation of the velocity relative to the wall that is a polynomial of the signed wall distance
`s'` (`w(s') = a s' + L s'^2 / 2`, `w = 0` at the wall: no slip). For the Morris weight `K(r) = W'(r) r / (r^2 + eta^2)` (a scalar times the identity) the tensors are

    T_k^{c'c} = int_S K(r) (e_c' . e_c(x')) s'^k dA' ,    k = 0, 1, 2 ;   slot k = 3: M1_n = int_S K (y . n) dA'  (the first moment of the solid: the Morris weight does not annihilate a rigid ROTATION of the wall)

for a circular wall of signed curvature `kappa_w` (`> 0` a solid disk, fluid outside; `< 0` a cavity; 0 the plane) at the distance `d` of the particle, by a polar quadrature of the actual solid region about
the particle, tabulated on a grid (`d / H`, `kappa_w H`) once per kernel and support and read bilinearly. Wendland C2 (the kernel gradient is written out).
"""
import math

import torch

from ..incompressible.compactProjection import MORRIS_ETA2, _dwendland2

__all__ = ['curvedMoments', 'CurvedWallMoments']

F64 = torch.float64


def dW(r, H):
    return _dwendland2(r, H)


def _gl(n, device):
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.as_tensor(x, dtype=F64, device=device), torch.as_tensor(w, dtype=F64, device=device)




def curvedMoments(H, d, kw, nth=160, nr=48, weight="alpha"):
    """T_k^{c' c}(d, kappa_w) for a circular wall of signed curvature `kw` (> 0: solid disk of radius 1/kw, fluid outside; < 0: cavity of radius 1/|kw|, fluid inside; 0: the plane) at the distance `d` of the particle, by a
    polar quadrature of the actual solid region about the particle.  The continuation of the velocity relative to the wall is  w_ext(x') = sum_c (a_c s' + L_c s'^2 / 2) e_c(x')  with s' the signed distance and e_c(x') the (n, t) frame AT x'
    (the frame turns with the position, as in a flow along the wall); the tensor entries are  T_k^{c'c} = int_S L(r) (e_c' . y_hat)(y_hat . e_c(x')) s'^k dA'  with the fixed frame of the particle on the left;  k = 0 uses the fixed frame
    on both sides (the particle's own velocity).  Diagonal by the mirror symmetry t -> -t.  Returns [B, 3 (k), 2 (n, t)].  `d`, `kw` are [B] tensors, d in [0, H).
    `weight = "morris"`: the scalar Morris pair weight K(r) = W'(r) r / (r^2 + eta^2) times the identity instead of L(r) y_hat (x) y_hat: T_k^{c'c} = int_S K(r) (e_c' . e_c(x')) s'^k dA' (k = 0: int_S K dA' on both components).
    Slot k = 3 is the first moment of the solid M1_n = int_S K (y . n) dA' (Morris only; the tangential part vanishes by symmetry): the Morris weight does not annihilate a rigid ROTATION of the wall (the alpha weight does),
    the wall integral of v_wall(x_i + y) - v_wall(x_i) = Omega x y is Omega x M1.  Returns [B, 4, 2]."""
    dev = d.device
    B = d.shape[0]
    out = torch.zeros((B, 4, 2), dtype=F64, device=dev)
    xr, wr = _gl(nr, dev)
    xt, wt = _gl(nth, dev)
    flat = kw.abs() * H < 1e-6
    for sign in (1, -1, 0):
        sel = (flat if sign == 0 else (~flat & (kw * sign > 0))) & (d < H)
        if not bool(sel.any()):
            continue
        dd, kk = d[sel], kw[sel]
        nb = dd.shape[0]
        if sign == 0:
            R = torch.full_like(dd, 1e9)
        else:
            R = 1.0 / kk.abs()
        if sign >= 0:                                                            # convex (or plane): a solid disk
            cm2 = dd * (2 * R + dd) / (R + dd) ** 2                               # cos^2 of the tangent angle
            thmax = torch.acos(torch.sqrt(cm2.clamp(max=1.0)))
            v = 0.5 * (xt + 1.0)                                                  # theta = thmax (1 - v^2), v in [0, 1]
            th = thmax[:, None] * (1.0 - v[None, :] ** 2)                         # [nb, nth]
            jac = 2.0 * thmax[:, None] * v[None, :] * (0.5 * wt)[None, :] * 2.0   # |dtheta| = 2 thmax v dv, dv = 0.5 dx, times 2 (theta < 0 symmetric)
            Rp = (R + dd)[:, None]
            disc = (Rp * torch.cos(th)) ** 2 - (dd * (2 * R + dd))[:, None]
            sq = torch.sqrt(disc.clamp(min=0.0))
            r_lo = Rp * torch.cos(th) - sq
            r_hi = torch.minimum(Rp * torch.cos(th) + sq, torch.full_like(th, H))
            ok = r_lo < H
        else:                                                                       # cavity: the solid is outside the circle of radius R, centre at distance R - d along n
            th = 0.5 * math.pi * (xt + 1.0)[None, :].expand(nb, -1)                  # theta in [0, pi], symmetric in the sign of sin(theta)
            jac = (0.5 * math.pi * wt)[None, :].expand(nb, -1) * 2.0
            Rd = (R - dd)[:, None]
            b = Rd * torch.cos(th)
            r_lo = b + torch.sqrt(b ** 2 + (dd * (2 * R - dd))[:, None])
            r_hi = torch.full_like(th, H)
            ok = r_lo < H
        r = r_lo[:, :, None] + 0.5 * (r_hi - r_lo).clamp(min=0.0)[:, :, None] * (xr[None, None, :] + 1.0)      # [nb, nth, nr]
        wrr = 0.5 * (r_hi - r_lo).clamp(min=0.0)[:, :, None] * wr[None, None, :]
        if weight == "morris":
            L = dW(r.reshape(-1), H).reshape(r.shape) * r / (r * r + MORRIS_ETA2 * H * H)
        else:
            L = dW(r.reshape(-1), H).reshape(r.shape) / r
        ct, st = torch.cos(th)[:, :, None], torch.sin(th)[:, :, None]
        if sign >= 0:                                                               # y = r (-cos, -sin) (towards the solid); c = -(R + d) n; frame at x': n' = (y - c) / |y - c|
            yx, yy = -r * ct, -r * st
            cx = -(R + dd)[:, None, None]
            ux, uy = yx - cx, yy
            rho = torch.sqrt(ux * ux + uy * uy)
            sp = rho - R[:, None, None]
            nn_x, nn_y = ux / rho, uy / rho
        else:
            yx, yy = r * ct, r * st                                                 # y = r (cos, sin); centre c = (R - d) n
            cx = (R - dd)[:, None, None]
            ux, uy = yx - cx, yy
            rho = torch.sqrt(ux * ux + uy * uy)
            sp = R[:, None, None] - rho
            nn_x, nn_y = -ux / rho, -uy / rho                                       # towards the centre = into the fluid
        yh_x, yh_y = yx / r, yy / r
        tt_x, tt_y = -nn_y, nn_x                                                    # tau' = J n'
        if weight == "morris":
            enn, ett = nn_x, tt_y                                                   # e_n . n', e_t . tau'
            f0n = f0t = torch.ones_like(yh_x)
        else:
            enn = yh_x * (yh_x * nn_x + yh_y * nn_y)                                # (e_n . y_hat)(y_hat . n')
            ett = yh_y * (yh_x * tt_x + yh_y * tt_y)                                # (e_t . y_hat)(y_hat . tau')
            f0n, f0t = yh_x * yh_x, yh_y * yh_y                                     # fixed frame (k = 0)
        w = (jac[:, :, None] * wrr * L * r) * ok[:, :, None]
        for k in range(3):
            sk = sp ** k
            if k == 0:
                out[sel, 0, 0] = (w * f0n).sum((1, 2))
                out[sel, 0, 1] = (w * f0t).sum((1, 2))
            else:
                out[sel, k, 0] = (w * enn * sk).sum((1, 2))
                out[sel, k, 1] = (w * ett * sk).sum((1, 2))
        if weight == "morris":
            out[sel, 3, 0] = (w * yx).sum((1, 2))                                   # y . n with n = e_x of the particle frame
    return out


class CurvedWallMoments:
    """`curvedMoments` on a grid (d / H, kappa_w H) built once, bilinear lookup; kappa_w H in [-kmax, kmax] (R / H >= 1 / kmax)."""

    def __init__(self, H, device, nd=65, nk=29, kmax=0.7, weight="alpha"):
        self.H, self.kmax, self.nd, self.nk, self.weight = float(H), float(kmax), nd, nk, weight
        dg = torch.linspace(0.0, 1.0, nd, dtype=F64, device=device) * H
        kg = torch.linspace(-kmax, kmax, nk, dtype=F64, device=device) / H
        D, K = torch.meshgrid(dg, kg, indexing="ij")
        tab = curvedMoments(H, D.reshape(-1), K.reshape(-1), weight=weight).reshape(nd, nk, 4, 2)
        self.table = tab.permute(3, 2, 0, 1).contiguous()                            # [2 (n, t), 4 (k = 0, 1, 2; 3 = M1), nd, nk]
        self.table[:, :, -1, :] = 0.0                                                # d = H: no solid within the support

    def eval(self, d, kw):
        """[2, 4, N] at distances `d` and wall curvatures `kw` (clamped to the grid)."""
        fd = (d / self.H).clamp(0.0, 1.0) * (self.nd - 1)
        fk = ((kw * self.H).clamp(-self.kmax, self.kmax) + self.kmax) / (2 * self.kmax) * (self.nk - 1)
        i0 = fd.floor().long().clamp(max=self.nd - 2)
        j0 = fk.floor().long().clamp(max=self.nk - 2)
        a, b = fd - i0, fk - j0
        t = self.table
        v = (t[:, :, i0, j0] * ((1 - a) * (1 - b)) + t[:, :, i0 + 1, j0] * (a * (1 - b)) + t[:, :, i0, j0 + 1] * ((1 - a) * b) + t[:, :, i0 + 1, j0 + 1] * (a * b))
        return v * (d < self.H).to(F64)


