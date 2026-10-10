"""The no-slip wall closure of the Morris viscous term for analytic walls (`schemeConfig.wallViscosityClosure = 'noslipMoment'`): the wall part of the viscous acceleration of a particle near a wall,
from the moments of the pair weight over the solid (`wallMoments.py`: curved tables) or, with `complementMoments` (the default with Morris), from the complement of the particle's own DISCRETE fluid moments.

Port of `warpSPHBoundaries/sim/wallclosure.py` (`NoSlipClosure.term`, `complement_moments`, `curvature`; the corner wedge tables are not ported: `cornerWedgeTables` is the oracle's experimental option).
The Morris operator `a_i = sum_j V_j nu (rho_i + rho_j) / rho_i K_ij (v_i - v_j)` is truncated at a wall; the missing part is the same integral over the solid with a continuation of the velocity relative
to the wall that is a polynomial of the wall distance, `w(s) = a s + (L / 2) s^2` per component in the frame (n, t), `w(d) = w_i`, and the viscous balance of the particle (the fluid sum `viscf` plus this
wall term `= nu_p (lap w)`, `nu_p = nu_used cal` the realised viscosity of the discrete operator; the normal and the tangential component with the curvature terms of the wall) gives the 2 x 2 system for
(`a`, `L`) and the wall term `A = -beta (G1 w + Hm L)`, `beta = 2 nu_used wm / rho`. Complement moments: the wall moments are the full-plane value minus the particle's own discrete fluid moments (fixed
particle frame), so fluid + wall is exact for linear and quadratic fields on the actual neighbourhood; where the support is not complete (a free surface next to the wall, `coverage = sum V W + mu lambda`
below 0.97) they are blended into the geometric tables. A rigid rotation of the wall adds the term `-beta Omega J M1`.
"""
import math
from typing import Any, Optional

import os

import torch

from ..incompressible.compactProjection import MORRIS_ETA2, _dwendland2, _anyPeriodic, minimumImage, morrisCalibration
from .wallMoments import CurvedWallMoments

__all__ = ['complementSums', 'closureAcceleration', 'wallNoSlipAcceleration']

F64 = torch.float64


def _complementSumsCore(x, masses, rho, fluid, i, j, n, d, H: float, length, flags):
    """`complementSums` on explicit tensors (no state / adjacency / domain objects: compilable): `length`, `flags` the periodic box lengths and flags (None: no periodic dimension)."""
    # no boolean-mask gathers (a host sync each, and not capturable in a CUDA graph): the fluid pairs inside the support are selected by a 0 / 1 weight over the whole pair list
    dlt = x[i] - x[j]
    if length is not None:
        dlt = dlt - flags * length * torch.round(dlt / length)
    y = -dlt
    r = y.norm(dim=1)
    sel = ((i != j) & fluid[i] & fluid[j] & (r < H)).to(F64)
    K = _dwendland2(r, H) * r / (r * r + MORRIS_ETA2 * H * H)
    V = masses / rho                                                     # the apparent volumes m / rho of the units rest density 1 (mass = volume)
    mu = V[j] * (rho[i] + rho[j]) / (2.0 * rho[i])
    st = (y * n[i]).sum(1) + d[i]
    w = sel * mu * K
    N = x.shape[0]
    z = torch.zeros(N, dtype=F64, device=x.device)
    S0 = z.index_add(0, i, w)
    S1 = z.index_add(0, i, w * st)
    S2 = z.index_add(0, i, w * st * st)
    SM = torch.zeros((N, 2), dtype=F64, device=x.device).index_add_(0, i, w[:, None] * y)
    return S0, S1, S2, SM


def _compiled(name, fn, schemeConfig):
    """`fn`, or its `torch.compile`d form with `WARPSPH_COMPILE_WALLS=1` / `schemeConfig.compileWalls` (see `_closure`)."""
    if not (os.environ.get('WARPSPH_COMPILE_WALLS') == '1' or getattr(schemeConfig, 'compileWalls', False)):
        return fn
    c = _COMPILED.get(name)
    if c is None:
        c = _COMPILED[name] = torch.compile(fn, dynamic=False)
    return c


def complementSums(state: Any, config: Any, adjacency: Any, fluid: torch.Tensor, n: torch.Tensor, d: torch.Tensor, rho: torch.Tensor, H: float, schemeConfig: Any = None):
    """The discrete fluid moments of the Morris weight in the frame (n_i, d_i) of each particle's wall (fixed particle frame, `y = x_j - x_i`, `s~ = y . n + d`, `mu_j = V_j (rho_i + rho_j) / (2 rho_i)`,
    `K = W'(r) r / (r^2 + eta^2 h^2)`): `S0 = sum mu K`, `S1 = sum mu K s~`, `S2 = sum mu K s~^2`, `SM = sum mu K y`, over the fluid pairs of the Verlet list inside the support."""
    domain = config.domain
    length = flags = None
    if getattr(domain, 'periodic', None) is not None and _anyPeriodic(domain):
        length = (domain.max - domain.min).to(F64)
        flags = torch.as_tensor(domain.periodic, device=length.device).to(F64)
    return _compiled('complementSums', _complementSumsCore, schemeConfig)(
        state.positions.to(F64), state.masses.to(F64), rho, fluid, adjacency.i.long(), adjacency.j.long(), n, d, float(H), length, flags)


def curvature(scene, bi, x, n, t, H, dx):
    """`kappa = div n` of the signed distance at the particles (tangential derivative of the wall normal, central difference over half a spacing): 1 / r for a convex circle, -1 / r concave, 0 on a plane."""
    eps = 0.5 * dx
    npl = scene.signed_distance(x + eps * t, body=bi, supportMax=H)[1]
    nmi = scene.signed_distance(x - eps * t, body=bi, supportMax=H)[1]
    return ((npl - nmi) * t).sum(1) / (2.0 * eps)


def _bodyGeometry(scene, x, H: float, dx: float):
    """Distance, normal, hit flag and curvature of every body at the particles, stacked: ([B, N], [B, N, 2], [B, N], [B, N]). One signed-distance evaluation per body plus the two of its curvature (the wall term used to
    evaluate the first twice); one function so it compiles (`WARPSPH_COMPILE_WALLS`)."""
    ds, ns, hs, ks = [], [], [], []
    for bi in range(len(scene.bodies)):
        dsd, nsd, hit = scene.signed_distance(x, body=bi, supportMax=H)
        t = torch.stack([-nsd[:, 1], nsd[:, 0]], 1)
        ds.append(dsd)
        ns.append(nsd)
        hs.append(hit)
        ks.append(curvature(scene, bi, x, nsd, t, H, dx))
    return torch.stack(ds), torch.stack(ns), torch.stack(hs), torch.stack(ks)


def closureAcceleration(tables, x, w, n, d, kap, viscf, rho, nuUsed, cal, wallMass, omega, sums=None, coverage=None, ownMask=None):
    """The wall part of the viscous acceleration [N, 2] of a no-slip wall for the particles at `x` (see the module docstring). `w`: velocity relative to the rigid wall motion at the particle, `n`, `d`, `kap`:
    wall normal into the fluid, distance and wall curvature `div n` there, `viscf`: the fluid pair viscous acceleration of the particle, `sums`: the complement moments of `complementSums` (None: the geometric
    curved tables), `coverage` the partition of unity, `ownMask`: the particles this body's wall closes (the nearest body within the support), `omega`: the wall's angular velocity."""
    t = torch.stack([-n[:, 1], n[:, 0]], 1)
    N = x.shape[0]
    Tc = tables.eval(d, kap / (1.0 - kap * d))                                              # [2 (n, t), 4, N]: moments over the actual circular solid of curvature kappa_w = 1 / R (signed)
    T = torch.zeros((N, 3, 2, 2), dtype=F64, device=x.device)
    T[:, :, 0, 0] = Tc[0, :3].T
    T[:, :, 1, 1] = Tc[1, :3].T
    M1 = torch.stack([Tc[0, 3], torch.zeros_like(Tc[0, 3])], 1)
    if sums is not None:
        Tt, M1t, kapt = T, M1, kap
        S0, S1, S2, SM = sums
        I2 = -2.0 * cal
        Tk = (-S0, -S1, 0.5 * I2 - S2)                                                     # + I_0, d I_0, d^2 I_0 cancel in the closure
        T = torch.zeros((N, 3, 2, 2), dtype=F64, device=x.device)
        for k, Tkk in enumerate(Tk):
            T[:, k, 0, 0] = Tkk
            T[:, k, 1, 1] = Tkk
        M1 = torch.stack([-(SM * n).sum(1), -(SM * t).sum(1)], 1)
        kap = torch.zeros_like(kap)                                                         # fixed-frame continuation: no curvature terms
        if coverage is not None:                                                            # incomplete support: towards the geometric tables
            c = ((coverage - 0.92) / 0.05).clamp(0.0, 1.0)
            T = c[:, None, None, None] * T + (1.0 - c)[:, None, None, None] * Tt
            M1 = c[:, None] * M1 + (1.0 - c)[:, None] * M1t
            kap = (1.0 - c) * kapt
        if ownMask is not None:
            T = T * ownMask[:, None, None, None].to(F64)
            M1 = M1 * ownMask[:, None].to(F64)
    beta = 2.0 * nuUsed * wallMass / rho                                                    # Morris pair weight 2 nu_used V_w K(r) (identity), rho_w = rho_i
    nup = nuUsed * cal                                                                      # the realised viscosity of the discrete operator
    cn = 1.0                                                                                # no grad div part: the normal component has no factor 3
    bn = beta / nup
    wv = torch.stack([(w * n).sum(1), (w * t).sum(1)], 1)
    fv = torch.stack([(viscf * n).sum(1), (viscf * t).sum(1)], 1)
    Om = beta * omega                                                                       # a rigid rotation Omega of the wall: the Morris weight does not annihilate it; wall part -beta Omega J M1
    rot = torch.stack([-Om * M1[:, 1], Om * M1[:, 0]], 1)
    fv = fv - rot                                                                           # the fluid sum carries the opposite (the full-plane integral vanishes): removed before the balance of w
    dd = d[:, None, None]
    G1 = T[:, 1] / dd - T[:, 0]
    Hm = 0.5 * (T[:, 2] - dd * T[:, 1])
    Dm = torch.zeros_like(Hm)
    Dm[:, 0, 0] = cn
    Dm[:, 1, 1] = 1.0 + 0.5 * kap * d
    rv = torch.stack([torch.zeros_like(d), kap * wv[:, 1] / d - kap * kap * wv[:, 1]], 1)
    rhs = fv / nup - bn[:, None] * (G1 @ wv[:, :, None])[:, :, 0] - rv
    Mm = Dm + bn[:, None, None] * Hm                                                        # 2 x 2 by Cramer's rule
    det = Mm[:, 0, 0] * Mm[:, 1, 1] - Mm[:, 0, 1] * Mm[:, 1, 0]
    Lv = torch.stack([Mm[:, 1, 1] * rhs[:, 0] - Mm[:, 0, 1] * rhs[:, 1], Mm[:, 0, 0] * rhs[:, 1] - Mm[:, 1, 0] * rhs[:, 0]], 1) / det[:, None]
    Av = -beta[:, None] * ((G1 @ wv[:, :, None])[:, :, 0] + (Hm @ Lv[:, :, None])[:, :, 0]) - rot
    return Av[:, 0:1] * n + Av[:, 1:2] * t


_COMPILED = {}


def _closure(schemeConfig):
    """`closureAcceleration`, or its `torch.compile`d form with `WARPSPH_COMPILE_WALLS=1` / `schemeConfig.compileWalls`: the closure is ~1700 small elementwise torch ops per step (a launch each: the step is CPU-bound without a CUDA graph and
    GPU-latency-bound with one), which the compiler fuses into a handful of kernels. Not bitwise equal to the eager form (a fused expression rounds differently): opt-in."""
    if not (os.environ.get('WARPSPH_COMPILE_WALLS') == '1' or getattr(schemeConfig, 'compileWalls', False)):
        return closureAcceleration
    fn = _COMPILED.get('closure')
    if fn is None:
        fn = _COMPILED['closure'] = torch.compile(closureAcceleration, dynamic=False, fullgraph=bool(os.environ.get('WARPSPH_COMPILE_FULLGRAPH')))
    return fn


def _tables(schemeConfig, H, device):
    cache = schemeConfig.__dict__.setdefault('_wallMomentTables', {})
    key = (float(H), 'morris')
    if key not in cache:
        cache[key] = CurvedWallMoments(float(H), device, weight='morris')
    return cache[key]


def wallNoSlipAcceleration(state: Any, config: Any, schemeConfig: Any, adjacency: Any, wall: Any, viscf: torch.Tensor, perBody: bool = False):
    """The no-slip wall term of the Morris viscous acceleration of every fluid particle next to an analytic wall: `viscf` the fluid-only Morris acceleration of the particles (the closure balances it),
    `schemeConfig.diffusionParams.viscidNu` the viscosity of the fluid operator, `schemeConfig.morrisCalibration` the realised / nominal viscosity of the discrete operator (None: the long-wave lattice
    sum of the actual spacing, `compactProjection.morrisCalibration`), `schemeConfig.complementMoments` (default True) the discrete complement moments instead of the geometric tables."""
    from .wallTerms import _wallVelocity
    provider = schemeConfig.boundaryProvider
    scene = provider.scene
    dev = state.positions.device
    x = state.positions.to(F64)
    fluid = state.kinds == 0
    rho = state.densities.to(F64)
    v = state.velocities.to(F64)
    H = float(wall.support)
    dx = float(config.dx)
    nu = float(schemeConfig.diffusionParams.viscidNu)
    cal = getattr(schemeConfig, 'morrisCalibration', None)
    if cal is None:
        key = (H, dx)                                                                          # the lattice sum of the actual spacing: a host constant, read from the device once (a sync per call otherwise)
        cache = schemeConfig.__dict__.setdefault('_morrisCalCache', {})
        if key not in cache:
            cache[key] = morrisCalibration(H, dx, float(state.masses.double()[fluid].mean()))
        cal = cache[key]
    complement = getattr(schemeConfig, 'complementMoments', True)
    if getattr(schemeConfig, 'cornerWedgeTables', False):
        raise NotImplementedError('cornerWedgeTables (the oracle marks it experimental) is not ported')
    tables = _tables(schemeConfig, H, dev)
    nb = len(scene.bodies)
    sums = coverage = own = None
    dall, nall, hall, kall = _compiled('geometry', _bodyGeometry, schemeConfig)(scene, x, H, dx)
    if complement:
        from ...modules.density import computeDensities
        coverage = computeDensities(state, config, schemeConfig, adjacency).to(F64)                     # sum V W + mu lambda: the partition of unity
        nearest = dall.argmin(0)
    out = torch.zeros_like(v)
    parts = []
    for bi in range(nb):
        dsd, nsd, hit, kap = dall[bi], nall[bi], hall[bi], kall[bi]
        dd = dsd.clamp(min=0.25 * dx)
        on = (hit & (dsd < H) & fluid)[:, None]
        if complement:
            sums = complementSums(state, config, adjacency, fluid, nsd, dd, rho, H, schemeConfig)
            own = (nearest == bi) & (dall[bi] < H)
        term = _closure(schemeConfig)(tables, x, v - _wallVelocity(wall, bi).to(F64), nsd, dd, kap, viscf.to(F64), rho, nu, cal, wall.wm, wall.omega[bi] if wall.omega is not None else 0.0,
                                      sums=sums, coverage=coverage, ownMask=own)
        term = torch.where(on, term, torch.zeros_like(term))
        out = out + term
        parts.append(term)
    res = torch.stack(parts) if perBody else out
    return res.to(state.velocities.dtype)
